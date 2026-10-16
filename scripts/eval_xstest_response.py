"""Evaluate LLaDA-Guard on the XSTest-Response benchmark.

The dataset loader uses ``allenai/xstest-response`` with the
``response_harmfulness`` split. If the dataset is gated in your environment,
accept the dataset terms on Hugging Face and authenticate before running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llada_guard.data import eval_binary_labels, load_xstest_response
from llada_guard.eval import binary_metrics_from_scores
from llada_guard_infer import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CHECKPOINT,
    DEFAULT_MAX_SEQ_LENGTH,
    OPTIMIZED_T_VALUES,
    load_llada_guard,
    score_examples,
)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_xstest_response(n: int | None) -> list[dict]:
    try:
        examples = load_xstest_response()
        return examples if n is None else examples[:n]
    except Exception as exc:
        raise SystemExit(
            "Could not load allenai/xstest-response. Accept the dataset terms "
            "on Hugging Face and authenticate with a token that can access it."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the optimized LLaDA-Guard scorer on XSTest-Response.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--t-values", type=float, nargs="+", default=list(OPTIMIZED_T_VALUES))
    parser.add_argument("--response-mask-scale", type=float, default=1.0)
    parser.add_argument("-n", type=int, default=0, help="Examples to evaluate; 0 means all.")
    parser.add_argument("--output", type=Path, default=Path("results/xstest_response_eval.json"))
    args = parser.parse_args()

    n = args.n if args.n > 0 else None
    examples = _load_xstest_response(n)
    y_true = eval_binary_labels(examples)

    model, tokenizer = load_llada_guard(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        torch_dtype=args.torch_dtype,
        device=args.device,
    )
    model.eval()
    scores = score_examples(
        model,
        tokenizer,
        examples,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        t_values=tuple(args.t_values),
        response_mask_scale=args.response_mask_scale,
        progress=True,
    )
    metrics = dict(binary_metrics_from_scores(y_true, scores, threshold=0.0))

    payload = {
        "dataset": "xstest_response",
        "split": "response_harmfulness",
        "num_examples": len(examples),
        "base_model": args.base_model,
        "checkpoint": str(args.checkpoint),
        "t_values": list(args.t_values),
        "response_mask_scale": args.response_mask_scale,
        "threshold": 0.0,
        "metrics": metrics,
        "scores": scores,
        "labels": y_true,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"Dataset:    XSTest-Response ({len(examples)} examples)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"AUPRC:      {metrics.get('auprc', float('nan')):.4f}")
    print(f"F1:         {metrics.get('f1', float('nan')):.4f}")
    print(f"Acc safe:   {metrics.get('acc_safe', float('nan')):.4f}")
    print(f"Acc unsafe: {metrics.get('acc_unsafe', float('nan')):.4f}")
    print(f"Saved:      {args.output}")


if __name__ == "__main__":
    main()
