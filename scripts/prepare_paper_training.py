#!/usr/bin/env python3
"""Prepare binary training data balanced by route and safety label."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from llada_guard.training_data import file_sha256
from llada_guard.training_sources import (
    SOURCE_REVISIONS,
    load_aegis,
    load_wildguardtrain,
)

FAMILIES = ("prompt_safe", "prompt_unsafe", "response_safe", "response_unsafe")


def materialize_rows(examples: list[dict]) -> list[dict]:
    rows = []
    for example in examples:
        prompt_label = example["prompt_label"]
        response_label = example["response_label"]
        if prompt_label in {"safe", "unsafe"}:
            rows.append(
                {
                    "user_message": example["user_message"],
                    "agent_response": None,
                    "is_safe": prompt_label == "safe",
                    "prompt_label": prompt_label,
                    "response_label": "",
                    "source": example["source"],
                }
            )
        if example["agent_response"] is not None and response_label in {
            "safe",
            "unsafe",
        }:
            rows.append(
                {
                    **example,
                    "is_safe": response_label == "safe",
                }
            )
    return rows


def route_family(row: dict) -> str:
    route = "prompt" if row["agent_response"] is None else "response"
    label = "safe" if row["is_safe"] else "unsafe"
    return f"{route}_{label}"


def sample_balanced(rows: list[dict], total: int, seed: int) -> list[dict]:
    if total <= 0 or total % len(FAMILIES):
        raise ValueError("target-total must be positive and divisible by four")
    rng = random.Random(seed)
    target = total // len(FAMILIES)
    selected = []
    for family in FAMILIES:
        pool = [row for row in rows if route_family(row) == family]
        if len(pool) < target:
            raise ValueError(f"{family}: need {target} examples, found {len(pool)}")
        selected.extend(rng.sample(pool, target))
    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--target-total", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target_total <= 0 or args.target_total % 4:
        parser.error("--target-total must be positive and divisible by four")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite existing training data or manifest"
        )

    rows = materialize_rows(load_aegis() + load_wildguardtrain())
    selected = sample_balanced(rows, args.target_total, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "sha256": file_sha256(args.output),
        "seed": args.seed,
        "target_total": args.target_total,
        "source_revisions": SOURCE_REVISIONS,
        "sampling": "uniform within each route/label family",
        "test_overlap_filtered": False,
        "families": dict(Counter(route_family(row) for row in selected)),
        "sources": dict(Counter(row["source"] for row in selected)),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(selected)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
