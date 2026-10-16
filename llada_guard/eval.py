"""Masked-scope scoring helpers for LLaDA-Guard inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from .data import PROMPT_BLOCK_ID, RESPONSE_BLOCK_ID, build_eval_loader
from .model import MASK_ID

_MASK_SEED_EXAMPLE_STRIDE = 104729


def _build_cached_eval_loader(
    examples,
    tokenizer,
    max_seq_length,
    batch_size,
    gen_only=False,
    skip_cls=False,
    prompt_format: str = "aegis_taxonomy",
):
    """Compatibility wrapper for the deterministic public eval loader."""
    del skip_cls
    return build_eval_loader(
        examples,
        tokenizer,
        max_seq_length,
        batch_size,
        gen_only=gen_only,
        prompt_format=prompt_format,
    )


def _mask_seed(batch, batch_idx: int, base_seed: int, t_idx: int) -> int | torch.Tensor:
    """Stable mask seed keyed by example id when available."""
    example_index = batch.get("example_index")
    if torch.is_tensor(example_index) and bool((example_index >= 0).all().item()):
        return example_index.to(dtype=torch.long) * _MASK_SEED_EXAMPLE_STRIDE + int(base_seed) + int(t_idx)
    return int(base_seed) + int(batch_idx) * 97 + int(t_idx)


def _base_llada_lm(model):
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _prefer_native_rms_norm(model) -> None:
    """Patch LLaDA RMSLayerNorm modules to PyTorch's native RMSNorm fast path."""
    if not hasattr(F, "rms_norm"):
        return
    modules = getattr(_base_llada_lm(model), "modules", None)
    if not callable(modules):
        return
    for module in modules():
        if module.__class__.__name__ != "RMSLayerNorm" or getattr(module, "_native_rms_norm", False):
            continue
        normalized_shape = getattr(module, "normalized_shape", None)
        weight = getattr(module, "weight", None)
        eps = getattr(module, "eps", None)
        if normalized_shape is None or weight is None or eps is None:
            continue

        def native_forward(x: torch.Tensor, rms_module=module) -> torch.Tensor:
            y = F.rms_norm(x, rms_module.normalized_shape, rms_module.weight, rms_module.eps)
            bias = getattr(rms_module, "bias", None)
            if bias is not None:
                y = y + bias
            return y

        module.forward = native_forward
        module._native_rms_norm = True


def _prefer_inplace_silu(model) -> None:
    """Enable in-place SiLU on existing activation modules."""
    modules = getattr(_base_llada_lm(model), "modules", None)
    if not callable(modules):
        return
    for module in modules():
        if module.__class__.__name__ == "SiLU":
            module.inplace = True


def apply_eval_inference_fastpaths(model) -> None:
    """Apply idempotent inference-only module fast paths used by eval scoring."""
    _prefer_native_rms_norm(model)
    _prefer_inplace_silu(model)


def _mask_ratio_upper_bound(eps: float, max_t: float) -> float:
    return max(float(eps), min(1.0, float(max_t)))


def _scale_mask_ratio(t_values, scale: float, eps: float, max_t: float = 1.0):
    """Scale a masking ratio while keeping it inside a valid range."""
    upper = _mask_ratio_upper_bound(eps, max_t)
    if abs(float(scale) - 1.0) <= 1e-12 and upper >= 1.0:
        return t_values
    if torch.is_tensor(t_values):
        return torch.clamp(t_values.to(dtype=torch.float32) * float(scale), min=eps, max=upper)
    return max(eps, min(upper, float(t_values) * float(scale)))


def _sample_relative_mask_indices(
    block_ids: torch.Tensor,
    real_lengths: torch.Tensor,
    eps: float = 1e-3,
    target_block_ids: tuple[int, ...] = (1, 2),
    t_override: float | torch.Tensor | None = None,
    max_t: float = 1.0,
    seed: int | torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Sample exact-k relative mask positions for the selected target region."""
    device = block_ids.device
    batch_size = block_ids.shape[0]
    upper = _mask_ratio_upper_bound(eps, max_t)
    if t_override is None:
        base = torch.rand(batch_size, device=device)
        offset = torch.arange(batch_size, device=device).float() / max(batch_size, 1)
        t_values = (upper - eps) * ((base / max(batch_size, 1) + offset) % 1.0) + eps
    elif torch.is_tensor(t_override):
        t_values = torch.clamp(t_override.to(device=device, dtype=torch.float32), min=eps, max=upper)
    else:
        t_values = torch.full(
            (batch_size,),
            max(eps, min(upper, float(t_override))),
            device=device,
        )

    row_seeds = None
    generator = None
    if torch.is_tensor(seed):
        row_seeds = seed.detach().to(device="cpu", dtype=torch.long).tolist()
    elif isinstance(seed, (list, tuple)):
        row_seeds = [int(value) for value in seed]
    elif seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    if row_seeds is not None and len(row_seeds) != batch_size:
        raise ValueError(f"Expected {batch_size} row seeds, got {len(row_seeds)}")

    positions = torch.arange(block_ids.shape[1], device=device).unsqueeze(0)
    selections: list[torch.Tensor] = []
    for idx in range(batch_size):
        target_mask = torch.zeros(block_ids.shape[1], dtype=torch.bool, device=device)
        for block_id in target_block_ids:
            target_mask |= block_ids[idx] == block_id
        target_positions = (
            target_mask & (positions[0] < real_lengths[idx])
        ).nonzero(as_tuple=True)[0]
        target_len = target_positions.numel()
        if target_len == 0:
            selections.append(torch.empty(0, dtype=torch.long, device=device))
            continue
        n_mask = max(1, min(target_len, int(round(float(t_values[idx]) * target_len))))
        row_generator = generator
        if row_seeds is not None:
            row_generator = torch.Generator(device=device)
            row_generator.manual_seed(int(row_seeds[idx]))
        perm = torch.randperm(target_len, generator=row_generator, device=device)
        selections.append(perm[:n_mask])
    return selections, t_values


def _relative_to_absolute_mask(
    block_ids: torch.Tensor,
    real_lengths: torch.Tensor,
    relative_indices: list[torch.Tensor],
    target_block_ids: tuple[int, ...] = (1, 2),
) -> torch.Tensor:
    """Map relative target-region positions to absolute sequence positions."""
    device = block_ids.device
    batch_size, seq_len = block_ids.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for idx in range(batch_size):
        rel_idx = relative_indices[idx]
        if rel_idx.numel() == 0:
            continue
        target_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for block_id in target_block_ids:
            target_mask |= block_ids[idx] == block_id
        target_positions = (
            target_mask & (positions[0] < real_lengths[idx])
        ).nonzero(as_tuple=True)[0]
        mask[idx, target_positions[rel_idx]] = True

    return mask


def _full_target_mask(
    block_ids: torch.Tensor,
    real_lengths: torch.Tensor,
    target_block_ids: tuple[int, ...],
) -> torch.Tensor:
    """Build a mask covering every token in the selected target region."""
    device = block_ids.device
    positions = torch.arange(block_ids.shape[1], device=device).unsqueeze(0)
    target_mask = torch.zeros_like(block_ids, dtype=torch.bool)
    for block_id in target_block_ids:
        target_mask |= block_ids == block_id
    return target_mask & (positions < real_lengths.unsqueeze(1))


def _pad_to_match_length(
    tensor: torch.Tensor,
    target_len: int,
    pad_id: int,
) -> torch.Tensor:
    """Right-pad a batch tensor to a target length."""
    if tensor.shape[1] >= target_len:
        return tensor
    pad_len = target_len - tensor.shape[1]
    pad = torch.full((tensor.shape[0], pad_len), pad_id, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1)


def _pad_mask_to_match_length(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """Right-pad a boolean mask to a target length."""
    if mask.shape[1] >= target_len:
        return mask
    pad_len = target_len - mask.shape[1]
    pad = torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype, device=mask.device)
    return torch.cat([mask, pad], dim=1)


def _build_attention_mask_from_lengths(
    real_lengths: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    """Build a right-padded attention mask from per-example real lengths."""
    positions = torch.arange(target_len, device=real_lengths.device).unsqueeze(0)
    return (positions < real_lengths.unsqueeze(1)).to(dtype=torch.long)


def _compute_masked_token_losses_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked token CE, aggregated per example, from precomputed logits."""
    device = input_ids.device
    batch_size = input_ids.shape[0]

    if not mask.any():
        zeros = logits.float().sum(dim=(1, 2)) * 0.0
        return zeros, zeros

    raw_ce = F.cross_entropy(logits[mask], input_ids[mask], reduction="none").float()
    example_idx = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand_as(mask)[mask]
    )
    per_ex_sum = torch.zeros(batch_size, device=device, dtype=raw_ce.dtype)
    per_ex_sum.scatter_add_(0, example_idx, raw_ce)
    denom = mask.sum(dim=1).float()
    per_ex_loss = per_ex_sum / denom.clamp(min=1)
    return per_ex_loss, -per_ex_loss


def _compute_paired_masked_token_outputs(
    model,
    safe_input_ids: torch.Tensor,
    safe_mask: torch.Tensor,
    safe_real_lengths: torch.Tensor,
    unsafe_input_ids: torch.Tensor,
    unsafe_mask: torch.Tensor,
    unsafe_real_lengths: torch.Tensor,
    pad_id: int,
    mask_id: int = MASK_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one paired safe/unsafe forward and return padded ids + logits."""
    max_len = max(safe_input_ids.shape[1], unsafe_input_ids.shape[1])
    safe_ids = _pad_to_match_length(safe_input_ids, max_len, pad_id)
    unsafe_ids = _pad_to_match_length(unsafe_input_ids, max_len, pad_id)
    safe_attention_mask = _build_attention_mask_from_lengths(safe_real_lengths, max_len)
    unsafe_attention_mask = _build_attention_mask_from_lengths(unsafe_real_lengths, max_len)
    safe_mask_padded = _pad_mask_to_match_length(safe_mask, max_len)
    unsafe_mask_padded = _pad_mask_to_match_length(unsafe_mask, max_len)

    combined_ids = torch.cat([safe_ids, unsafe_ids], dim=0)
    combined_attention_mask = torch.cat([safe_attention_mask, unsafe_attention_mask], dim=0)
    combined_mask = torch.cat([safe_mask_padded, unsafe_mask_padded], dim=0)
    noisy = combined_ids.clone()
    noisy[combined_mask] = mask_id
    logits = model(input_ids=noisy, attention_mask=combined_attention_mask).logits

    half = safe_input_ids.shape[0]
    return safe_ids, logits[:half], unsafe_ids, logits[half:]


def _target_token_counts(
    block_ids: torch.Tensor,
    real_lengths: torch.Tensor,
    *,
    target_block_ids: tuple[int, ...],
) -> torch.Tensor:
    """Count valid tokens in the scored target region for each row."""
    positions = torch.arange(block_ids.shape[1], device=block_ids.device).unsqueeze(0)
    target_mask = torch.zeros_like(block_ids, dtype=torch.bool)
    for block_id in target_block_ids:
        target_mask |= block_ids == block_id
    valid_target = target_mask & (positions < real_lengths.unsqueeze(1))
    return valid_target.sum(dim=1).clamp(min=1).float()


def _adaptive_optimal_t_values(
    batch,
    *,
    target_block_ids: tuple[int, ...],
    repeats: int,
    sigma_over_c: float,
    min_t: float,
    max_t: float,
) -> tuple[torch.Tensor, ...]:
    """Bias-variance optimal t per row: (sigma^2 / (2 c^2 R n))^(1/3)."""
    repeats = max(1, int(repeats))
    counts = _target_token_counts(
        batch["gen_safe_block_ids"],
        batch["gen_safe_real_lengths"],
        target_block_ids=target_block_ids,
    )
    t = ((float(sigma_over_c) ** 2) / (2.0 * repeats * counts)).pow(1.0 / 3.0)
    t = torch.clamp(t, min=float(min_t), max=float(max_t))
    return tuple(t.clone() for _ in range(repeats))


def _slice_batch_rows(batch: dict[str, torch.Tensor], row_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Select a subset of rows from a batch dict."""
    row_mask = row_mask.to(dtype=torch.bool)
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == row_mask.shape[0]:
            sliced[key] = value[row_mask]
        else:
            sliced[key] = value
    return sliced


def _score_batch_masked_scope_sequential(
    model,
    batch,
    batch_idx,
    t_values,
    pad_id,
    response_mask_scale=1.0,
    adaptive_t_repeats: int | None = None,
    adaptive_t_sigma_over_c: float = 1.0,
    adaptive_t_min: float = 1e-3,
    adaptive_t_max: float = 0.5,
):
    """Score a batch with prompt/response scoped masks, one t at a time."""
    device = batch["gen_safe_input_ids"].device
    batch_size = batch["gen_safe_input_ids"].shape[0]
    has_response = batch.get("has_response")
    if has_response is None:
        has_response = (batch["gen_safe_block_ids"] == RESPONSE_BLOCK_ID).any(dim=1).float()
    has_response = has_response > 0.5
    has_prompt_only = bool((~has_response).any().item())
    has_with_response = bool(has_response.any().item())

    if has_prompt_only and has_with_response:
        scores = torch.zeros(batch_size, device=device)
        prompt_mask = ~has_response
        response_mask = has_response
        if prompt_mask.any():
            scores[prompt_mask] = _score_batch_masked_scope_sequential(
                model,
                _slice_batch_rows(batch, prompt_mask),
                batch_idx,
                t_values,
                pad_id,
                response_mask_scale=response_mask_scale,
                adaptive_t_repeats=adaptive_t_repeats,
                adaptive_t_sigma_over_c=adaptive_t_sigma_over_c,
                adaptive_t_min=adaptive_t_min,
                adaptive_t_max=adaptive_t_max,
            )
        if response_mask.any():
            scores[response_mask] = _score_batch_masked_scope_sequential(
                model,
                _slice_batch_rows(batch, response_mask),
                batch_idx,
                t_values,
                pad_id,
                response_mask_scale=response_mask_scale,
                adaptive_t_repeats=adaptive_t_repeats,
                adaptive_t_sigma_over_c=adaptive_t_sigma_over_c,
                adaptive_t_min=adaptive_t_min,
                adaptive_t_max=adaptive_t_max,
            )
        return scores.detach()

    active_t_values = t_values
    target_block_ids = (PROMPT_BLOCK_ID,) if has_prompt_only else (RESPONSE_BLOCK_ID,)
    if adaptive_t_repeats is not None:
        active_t_values = _adaptive_optimal_t_values(
            batch,
            target_block_ids=target_block_ids,
            repeats=adaptive_t_repeats,
            sigma_over_c=adaptive_t_sigma_over_c,
            min_t=adaptive_t_min,
            max_t=adaptive_t_max,
        )

    score_accum = torch.zeros(batch_size, device=device)
    for t_idx, t_value in enumerate(active_t_values):
        if has_prompt_only:
            prompt_relative_indices, _ = _sample_relative_mask_indices(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                t_override=t_value,
                seed=_mask_seed(batch, batch_idx, 1337, t_idx),
                target_block_ids=(PROMPT_BLOCK_ID,),
            )
            safe_mask = _relative_to_absolute_mask(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                prompt_relative_indices,
                target_block_ids=(PROMPT_BLOCK_ID,),
            )
            unsafe_mask = _relative_to_absolute_mask(
                batch["gen_unsafe_block_ids"],
                batch["gen_unsafe_real_lengths"],
                prompt_relative_indices,
                target_block_ids=(PROMPT_BLOCK_ID,),
            )
            scope_safe_mask = safe_mask | _full_target_mask(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                target_block_ids=(RESPONSE_BLOCK_ID,),
            )
            scope_unsafe_mask = unsafe_mask | _full_target_mask(
                batch["gen_unsafe_block_ids"],
                batch["gen_unsafe_real_lengths"],
                target_block_ids=(RESPONSE_BLOCK_ID,),
            )
        else:
            response_t_value = _scale_mask_ratio(t_value, response_mask_scale, 1e-3)
            response_relative_indices, _ = _sample_relative_mask_indices(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                t_override=response_t_value,
                seed=_mask_seed(batch, batch_idx, 7331, t_idx),
                target_block_ids=(RESPONSE_BLOCK_ID,),
            )
            safe_mask = _relative_to_absolute_mask(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                response_relative_indices,
                target_block_ids=(RESPONSE_BLOCK_ID,),
            )
            unsafe_mask = _relative_to_absolute_mask(
                batch["gen_unsafe_block_ids"],
                batch["gen_unsafe_real_lengths"],
                response_relative_indices,
                target_block_ids=(RESPONSE_BLOCK_ID,),
            )
            scope_safe_mask = safe_mask
            scope_unsafe_mask = unsafe_mask

        safe_ids, safe_logits, unsafe_ids, unsafe_logits = _compute_paired_masked_token_outputs(
            model,
            batch["gen_safe_input_ids"],
            scope_safe_mask,
            batch["gen_safe_real_lengths"],
            batch["gen_unsafe_input_ids"],
            scope_unsafe_mask,
            batch["gen_unsafe_real_lengths"],
            pad_id,
        )
        max_len = safe_ids.shape[1]
        safe_mask_padded = _pad_mask_to_match_length(safe_mask, max_len)
        unsafe_mask_padded = _pad_mask_to_match_length(unsafe_mask, max_len)
        _, safe_logp = _compute_masked_token_losses_from_logits(
            safe_logits,
            safe_ids,
            safe_mask_padded,
        )
        _, unsafe_logp = _compute_masked_token_losses_from_logits(
            unsafe_logits,
            unsafe_ids,
            unsafe_mask_padded,
        )
        score_accum += unsafe_logp - safe_logp

    return (score_accum / len(active_t_values)).detach()


def _score_batch_masked_scope_tiled(
    model,
    batch,
    batch_idx,
    t_values,
    pad_id,
    response_mask_scale=1.0,
    adaptive_t_repeats: int | None = None,
    adaptive_t_sigma_over_c: float = 1.0,
    adaptive_t_min: float = 1e-3,
    adaptive_t_max: float = 0.5,
):
    """Score a batch with prompt/response scoped masks, tiled over t values."""
    device = batch["gen_safe_input_ids"].device
    has_response = batch.get("has_response")
    if has_response is None:
        has_response = (batch["gen_safe_block_ids"] == RESPONSE_BLOCK_ID).any(dim=1).float()
    has_response = has_response > 0.5
    has_prompt_only = bool((~has_response).any().item())
    has_with_response = bool(has_response.any().item())
    batch_size = batch["gen_safe_input_ids"].shape[0]

    if has_prompt_only and has_with_response:
        scores = torch.zeros(batch_size, device=device)
        prompt_mask = ~has_response
        response_mask = has_response
        if prompt_mask.any():
            scores[prompt_mask] = _score_batch_masked_scope_tiled(
                model,
                _slice_batch_rows(batch, prompt_mask),
                batch_idx,
                t_values,
                pad_id,
                response_mask_scale=response_mask_scale,
                adaptive_t_repeats=adaptive_t_repeats,
                adaptive_t_sigma_over_c=adaptive_t_sigma_over_c,
                adaptive_t_min=adaptive_t_min,
                adaptive_t_max=adaptive_t_max,
            )
        if response_mask.any():
            scores[response_mask] = _score_batch_masked_scope_tiled(
                model,
                _slice_batch_rows(batch, response_mask),
                batch_idx,
                t_values,
                pad_id,
                response_mask_scale=response_mask_scale,
                adaptive_t_repeats=adaptive_t_repeats,
                adaptive_t_sigma_over_c=adaptive_t_sigma_over_c,
                adaptive_t_min=adaptive_t_min,
                adaptive_t_max=adaptive_t_max,
            )
        return scores.detach()

    target_block_ids = (PROMPT_BLOCK_ID,) if has_prompt_only else (RESPONSE_BLOCK_ID,)
    active_t_values = t_values
    if adaptive_t_repeats is not None:
        active_t_values = _adaptive_optimal_t_values(
            batch,
            target_block_ids=target_block_ids,
            repeats=adaptive_t_repeats,
            sigma_over_c=adaptive_t_sigma_over_c,
            min_t=adaptive_t_min,
            max_t=adaptive_t_max,
        )

    all_safe_masks = []
    all_unsafe_masks = []
    for t_idx, t_value in enumerate(active_t_values):
        if has_prompt_only:
            relative_indices, _ = _sample_relative_mask_indices(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                t_override=t_value,
                seed=_mask_seed(batch, batch_idx, 1337, t_idx),
                target_block_ids=(PROMPT_BLOCK_ID,),
            )
        else:
            response_t_value = _scale_mask_ratio(t_value, response_mask_scale, 1e-3)
            relative_indices, _ = _sample_relative_mask_indices(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                t_override=response_t_value,
                seed=_mask_seed(batch, batch_idx, 7331, t_idx),
                target_block_ids=(RESPONSE_BLOCK_ID,),
            )

        all_safe_masks.append(
            _relative_to_absolute_mask(
                batch["gen_safe_block_ids"],
                batch["gen_safe_real_lengths"],
                relative_indices,
                target_block_ids=target_block_ids,
            )
        )
        all_unsafe_masks.append(
            _relative_to_absolute_mask(
                batch["gen_unsafe_block_ids"],
                batch["gen_unsafe_real_lengths"],
                relative_indices,
                target_block_ids=target_block_ids,
            )
        )

    num_t_values = len(active_t_values)

    def _repeat_batch_tensor(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x] * num_t_values, dim=0)

    repeated_safe_ids = _repeat_batch_tensor(batch["gen_safe_input_ids"])
    repeated_unsafe_ids = _repeat_batch_tensor(batch["gen_unsafe_input_ids"])
    repeated_safe_lengths = _repeat_batch_tensor(batch["gen_safe_real_lengths"])
    repeated_unsafe_lengths = _repeat_batch_tensor(batch["gen_unsafe_real_lengths"])
    tiled_safe_mask = torch.cat(all_safe_masks, dim=0)
    tiled_unsafe_mask = torch.cat(all_unsafe_masks, dim=0)

    if has_prompt_only:
        full_response_safe_mask = _full_target_mask(
            batch["gen_safe_block_ids"],
            batch["gen_safe_real_lengths"],
            target_block_ids=(RESPONSE_BLOCK_ID,),
        )
        full_response_unsafe_mask = _full_target_mask(
            batch["gen_unsafe_block_ids"],
            batch["gen_unsafe_real_lengths"],
            target_block_ids=(RESPONSE_BLOCK_ID,),
        )
        scope_safe_mask = tiled_safe_mask | _repeat_batch_tensor(full_response_safe_mask)
        scope_unsafe_mask = tiled_unsafe_mask | _repeat_batch_tensor(full_response_unsafe_mask)
    else:
        scope_safe_mask = tiled_safe_mask
        scope_unsafe_mask = tiled_unsafe_mask

    safe_ids, safe_logits, unsafe_ids, unsafe_logits = _compute_paired_masked_token_outputs(
        model,
        repeated_safe_ids,
        scope_safe_mask,
        repeated_safe_lengths,
        repeated_unsafe_ids,
        scope_unsafe_mask,
        repeated_unsafe_lengths,
        pad_id,
    )
    max_len = safe_ids.shape[1]
    tiled_safe_mask = _pad_mask_to_match_length(tiled_safe_mask, max_len)
    tiled_unsafe_mask = _pad_mask_to_match_length(tiled_unsafe_mask, max_len)
    _, safe_logp = _compute_masked_token_losses_from_logits(
        safe_logits,
        safe_ids,
        tiled_safe_mask,
    )
    _, unsafe_logp = _compute_masked_token_losses_from_logits(
        unsafe_logits,
        unsafe_ids,
        tiled_unsafe_mask,
    )
    return (unsafe_logp - safe_logp).view(num_t_values, batch_size).mean(dim=0).detach()


def _select_eval_labels_and_valid_mask(batch, *, masked_scope: bool):
    """Choose labels/masks to match prompt-only or response-conditioned scoring."""
    if masked_scope:
        has_response = batch.get("has_response")
        prompt_labels = batch.get("prompt_is_unsafe")
        prompt_valid = batch.get("prompt_label_mask")
        response_labels = batch.get("response_is_unsafe")
        response_valid = batch.get("response_label_mask")
        if (
            has_response is not None
            and prompt_labels is not None
            and prompt_valid is not None
            and response_labels is not None
            and response_valid is not None
        ):
            has_response_mask = has_response > 0.5
            labels = torch.where(has_response_mask, response_labels, prompt_labels)
            valid = torch.where(has_response_mask, response_valid > 0.5, prompt_valid > 0.5)
            return labels, valid

    labels = batch["is_unsafe"]
    valid = batch.get("conv_label_mask", torch.ones_like(labels)) > 0.5
    return labels, valid


@torch.no_grad()
def collect_eval_scores(
    model,
    eval_loader,
    device,
    t_values=(0.15, 0.15, 0.15),
    pad_id: int = 0,
    tile_t_values: bool = True,
    masked_scope: bool = True,
    response_mask_scale: float = 1.0,
    return_indices: bool = False,
    adaptive_t_repeats: int | None = None,
    adaptive_t_sigma_over_c: float = 1.0,
    adaptive_t_min: float = 1e-3,
    adaptive_t_max: float = 0.5,
    progress: bool = True,
):
    """Collect masked-scope unsafe-minus-safe scores for an eval loader."""
    if not masked_scope:
        raise ValueError("This release supports masked_scope=True scoring only.")

    model.eval()
    apply_eval_inference_fastpaths(model)
    y_true = []
    scores = []
    example_indices = []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="LLaDA eval", disable=not progress)):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        if tile_t_values:
            try:
                batch_scores = _score_batch_masked_scope_tiled(
                    model,
                    batch,
                    batch_idx,
                    t_values,
                    pad_id,
                    response_mask_scale=response_mask_scale,
                    adaptive_t_repeats=adaptive_t_repeats,
                    adaptive_t_sigma_over_c=adaptive_t_sigma_over_c,
                    adaptive_t_min=adaptive_t_min,
                    adaptive_t_max=adaptive_t_max,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_scores = _score_batch_masked_scope_sequential(
                    model,
                    batch,
                    batch_idx,
                    t_values,
                    pad_id,
                    response_mask_scale=response_mask_scale,
                    adaptive_t_repeats=adaptive_t_repeats,
                    adaptive_t_sigma_over_c=adaptive_t_sigma_over_c,
                    adaptive_t_min=adaptive_t_min,
                    adaptive_t_max=adaptive_t_max,
                )
        else:
            batch_scores = _score_batch_masked_scope_sequential(
                model,
                batch,
                batch_idx,
                t_values,
                pad_id,
                response_mask_scale=response_mask_scale,
                adaptive_t_repeats=adaptive_t_repeats,
                adaptive_t_sigma_over_c=adaptive_t_sigma_over_c,
                adaptive_t_min=adaptive_t_min,
                adaptive_t_max=adaptive_t_max,
            )

        labels, valid = _select_eval_labels_and_valid_mask(batch, masked_scope=masked_scope)
        if not valid.any():
            continue
        scores.extend(batch_scores[valid].detach().cpu().tolist())
        y_true.extend(labels[valid].detach().cpu().tolist())
        if return_indices:
            example_indices.extend(batch["example_index"][valid].detach().cpu().tolist())

    if return_indices:
        return y_true, scores, example_indices
    return y_true, scores


def binary_metrics_from_scores(
    y_true,
    scores,
    threshold: float = 0.0,
    *,
    scores_are_probabilities: bool = False,
):
    """Compute binary metrics from raw paired generative scores."""
    if not y_true:
        return {
            "f1": 0.0,
            "n": 0,
            "threshold": float(threshold),
            "score_mean": 0.0,
            "score_std": 0.0,
            "acc_safe": 0.0,
            "acc_unsafe": 0.0,
            "score_mean_safe": 0.0,
            "score_mean_unsafe": 0.0,
            "score_gap": 0.0,
            "score_std_safe": 0.0,
            "score_std_unsafe": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "ece": 0.0,
            "aece": 0.0,
        }

    y_pred = [1 if score > threshold else 0 for score in scores]
    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n": len(y_true),
        "threshold": float(threshold),
        "score_mean": float(torch.tensor(scores).mean().item()) if scores else 0.0,
        "score_std": float(torch.tensor(scores).std(unbiased=False).item()) if scores else 0.0,
    }
    if 0 < sum(y_true) < len(y_true):
        metrics["auprc"] = average_precision_score(y_true, scores)

    y_true_t = torch.tensor(y_true)
    scores_t = torch.tensor(scores) if scores else torch.tensor([])
    y_pred_t = torch.tensor(y_pred)
    safe_mask = y_true_t == 0
    unsafe_mask = y_true_t == 1
    metrics["acc_safe"] = float((y_pred_t[safe_mask] == 0).float().mean().item()) if safe_mask.any() else 0.0
    metrics["acc_unsafe"] = float((y_pred_t[unsafe_mask] == 1).float().mean().item()) if unsafe_mask.any() else 0.0
    score_safe = scores_t[safe_mask]
    score_unsafe = scores_t[unsafe_mask]
    metrics["score_mean_safe"] = float(score_safe.mean().item()) if score_safe.numel() > 0 else 0.0
    metrics["score_mean_unsafe"] = float(score_unsafe.mean().item()) if score_unsafe.numel() > 0 else 0.0
    metrics["score_gap"] = metrics["score_mean_unsafe"] - metrics["score_mean_safe"]
    metrics["score_std_safe"] = float(score_safe.std(unbiased=False).item()) if score_safe.numel() > 0 else 0.0
    metrics["score_std_unsafe"] = float(score_unsafe.std(unbiased=False).item()) if score_unsafe.numel() > 0 else 0.0

    scores_np = np.array(scores, dtype=np.float64)
    if scores_are_probabilities:
        probs = np.clip(scores_np, 0.0, 1.0)
    else:
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores_np, -30, 30)))
    labels_np = np.array(y_true, dtype=np.float64)
    metrics["brier"] = float(np.mean((probs - labels_np) ** 2))
    probs_for_nll = np.clip(probs, 1e-12, 1.0 - 1e-12)
    metrics["nll"] = float(
        -np.mean(labels_np * np.log(probs_for_nll) + (1.0 - labels_np) * np.log(1.0 - probs_for_nll))
    )

    num_bins = 15
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, num_bins - 1)
    weighted_error = 0.0
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        count = mask.sum()
        if count > 0:
            weighted_error += count * abs(float(probs[mask].mean()) - float(labels_np[mask].mean()))
    metrics["ece"] = weighted_error / len(probs)

    adaptive_error = 0.0
    for bucket in np.array_split(np.argsort(probs), min(num_bins, len(probs))):
        if len(bucket) == 0:
            continue
        adaptive_error += len(bucket) * abs(float(probs[bucket].mean()) - float(labels_np[bucket].mean()))
    metrics["aece"] = adaptive_error / len(probs)
    return metrics
