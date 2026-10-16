"""Routed binary ranking and true-branch masked reconstruction objectives."""

import torch
import torch.nn.functional as F

from .eval import (
    _compute_masked_token_losses_from_logits,
    _compute_paired_masked_token_outputs,
    _full_target_mask,
    _pad_mask_to_match_length,
    _relative_to_absolute_mask,
    _sample_relative_mask_indices,
)
from .model import MASK_ID


def training_loss(model, batch, *, objective, pad_id, mask_id=MASK_ID):
    """Use prompt supervision for prompt-only rows, response supervision otherwise.

    Ranking is twice BCE on log p(text|unsafe) - log p(text|safe).
    Generative training minimizes mean masked-token CE on the true branch.
    Both use the historical stratified random corruption levels and shared
    candidate mask positions, without inverse-corruption weighting.
    """
    if objective not in {"ranking", "generative"}:
        raise ValueError(f"Unknown objective: {objective}")
    prompt_rows = batch["has_response"] <= 0.5
    labels = torch.where(
        prompt_rows, batch["prompt_is_unsafe"], batch["response_is_unsafe"]
    )
    valid = (
        torch.where(
            prompt_rows, batch["prompt_label_mask"], batch["response_label_mask"]
        )
        > 0.5
    )
    if not valid.all():
        raise ValueError("Every training row must have a label for its scoring scope")

    safe_blocks = batch["gen_safe_block_ids"]
    safe_lengths = batch["gen_safe_real_lengths"]
    prompt_indices, levels = _sample_relative_mask_indices(
        safe_blocks, safe_lengths, target_block_ids=(1,)
    )
    response_indices, _ = _sample_relative_mask_indices(
        safe_blocks, safe_lengths, target_block_ids=(2,), t_override=levels
    )
    masks, score_masks = {}, {}
    for branch in ("safe", "unsafe"):
        blocks = batch[f"gen_{branch}_block_ids"]
        lengths = batch[f"gen_{branch}_real_lengths"]
        prompt = _relative_to_absolute_mask(
            blocks, lengths, prompt_indices, target_block_ids=(1,)
        )
        response = _relative_to_absolute_mask(
            blocks, lengths, response_indices, target_block_ids=(2,)
        )
        full_response = _full_target_mask(blocks, lengths, target_block_ids=(2,))
        # Synthetic responses on prompt-only rows are hidden, but not scored.
        masks[branch] = torch.where(
            prompt_rows[:, None], prompt | full_response, response
        )
        score_masks[branch] = torch.where(prompt_rows[:, None], prompt, response)
        if not score_masks[branch].any(dim=1).all():
            raise ValueError("Training row has no scoreable tokens after truncation")

    safe_ids, safe_logits, unsafe_ids, unsafe_logits = (
        _compute_paired_masked_token_outputs(
            model,
            batch["gen_safe_input_ids"],
            masks["safe"],
            safe_lengths,
            batch["gen_unsafe_input_ids"],
            masks["unsafe"],
            batch["gen_unsafe_real_lengths"],
            pad_id,
            mask_id=mask_id,
        )
    )
    safe_ce, _ = _compute_masked_token_losses_from_logits(
        safe_logits,
        safe_ids,
        _pad_mask_to_match_length(score_masks["safe"], safe_ids.shape[1]),
    )
    unsafe_ce, _ = _compute_masked_token_losses_from_logits(
        unsafe_logits,
        unsafe_ids,
        _pad_mask_to_match_length(score_masks["unsafe"], unsafe_ids.shape[1]),
    )
    scores = safe_ce - unsafe_ce
    if objective == "ranking":
        loss = 2 * F.binary_cross_entropy_with_logits(scores, labels.float())
    else:
        loss = torch.where(labels > 0.5, unsafe_ce, safe_ce).mean()
    return loss, {
        "score_mean": scores.detach().mean().item(),
        "mask_ratio": levels.mean().item(),
    }
