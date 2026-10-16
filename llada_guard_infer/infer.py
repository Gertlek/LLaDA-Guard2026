from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from llada_guard.eval import _build_cached_eval_loader, collect_eval_scores
from llada_guard.model import load_checkpoint

DEFAULT_BASE_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_CHECKPOINT = "Gertlek/LLaDA-Guard"
OPTIMIZED_T_VALUES = (0.15, 0.15, 0.15)
DEFAULT_MAX_SEQ_LENGTH = 1536


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(_default_device())


def load_llada_guard(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    base_model: str = DEFAULT_BASE_MODEL,
    *,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
):
    """Load the LLaDA-Guard PEFT adapter and tokenizer."""
    return load_checkpoint(
        base_model,
        str(checkpoint),
        torch_dtype=torch_dtype,
        device=device or _default_device(),
    )


def make_eval_example(
    user_message: str,
    agent_response: str | None = None,
    *,
    is_safe: bool = True,
) -> dict:
    """Build the normalized example format expected by the masked-scope scorer."""
    label = "safe" if is_safe else "unsafe"
    example = {
        "user_message": user_message,
        "agent_response": agent_response,
        "is_safe": bool(is_safe),
        "violated_categories": [],
        "conv_label_mask": 1,
    }
    if agent_response is None:
        example.update({"prompt_label": label, "response_label": ""})
    else:
        example.update({"prompt_label": "safe", "response_label": label})
    return example


def _prepare_examples(examples: Iterable[dict]) -> list[dict]:
    prepared = []
    for example in examples:
        user_message = str(example.get("user_message") or example.get("prompt") or "").strip()
        if not user_message:
            continue
        raw_response = example.get("agent_response", example.get("response"))
        agent_response = None if raw_response is None else str(raw_response).strip()
        if agent_response == "":
            agent_response = None
        is_safe = bool(example.get("is_safe", True))

        normalized = make_eval_example(
            user_message,
            agent_response,
            is_safe=is_safe,
        )
        normalized.update({k: v for k, v in example.items() if k not in normalized})
        prepared.append(normalized)
    return prepared


@torch.inference_mode()
def score_examples(
    model,
    tokenizer,
    examples: Iterable[dict],
    *,
    batch_size: int = 1,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    t_values: tuple[float, ...] = OPTIMIZED_T_VALUES,
    response_mask_scale: float = 1.0,
    progress: bool = False,
) -> list[float]:
    """Score prompt-only or prompt-response examples with masked-scope routing."""
    prepared = _prepare_examples(examples)
    if not prepared:
        return []
    prepared = [
        {
            **example,
            "example_index": idx,
        }
        for idx, example in enumerate(prepared)
    ]

    device = _model_device(model)
    loader = _build_cached_eval_loader(
        prepared,
        tokenizer,
        max_seq_length,
        batch_size,
        gen_only=True,
    )
    _, scores, indices = collect_eval_scores(
        model,
        loader,
        device,
        t_values=t_values,
        pad_id=tokenizer.eos_token_id or 0,
        masked_scope=True,
        response_mask_scale=response_mask_scale,
        return_indices=True,
        progress=progress,
    )
    ordered_scores: list[float | None] = [None] * len(prepared)
    for index, score in zip(indices, scores):
        ordered_scores[int(index)] = float(score)

    missing = [idx for idx, score in enumerate(ordered_scores) if score is None]
    if missing:
        raise RuntimeError(f"Missing scores for example indices: {missing}")
    return [float(score) for score in ordered_scores]


def score_optimized_3x015(
    checkpoint: str | Path,
    user_message: str,
    agent_response: str | None = None,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> float:
    """Load the adapter and score a single prompt or prompt-response pair."""
    model, tokenizer = load_llada_guard(
        checkpoint=checkpoint,
        base_model=base_model,
        torch_dtype=torch_dtype,
        device=device,
    )
    model.eval()
    scores = score_examples(
        model,
        tokenizer,
        [make_eval_example(user_message, agent_response)],
        batch_size=1,
        max_seq_length=max_seq_length,
    )
    return scores[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a prompt or prompt-response pair with LLaDA-Guard.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--prompt", default="How do I bake a chocolate cake?")
    parser.add_argument("--response", default=None)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--json", action="store_true", help="Print a JSON result.")
    args = parser.parse_args()

    score = score_optimized_3x015(
        checkpoint=args.checkpoint,
        user_message=args.prompt,
        agent_response=args.response,
        base_model=args.base_model,
        torch_dtype=args.torch_dtype,
        device=args.device,
        max_seq_length=args.max_seq_length,
    )
    prediction = "unsafe" if score > 0.0 else "safe"
    mode = "response" if args.response is not None else "prompt"

    if args.json:
        print(json.dumps({"score": score, "prediction": prediction, "mode": mode}, indent=2))
        return

    print("Eval:       optimized masked-scope 3x t=0.15")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode:       {mode}")
    print(f"Score:      {score:.4f}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
