"""Prepare hash-verified benchmark cohorts and output an evaluation table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time


MANIFEST = Path(__file__).with_name("benchmark_manifest.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize(payload):
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def normalize(rows):
    examples, labels = [], []
    for row in rows:
        examples.append({"user_message": row["user_message"],
                         "agent_response": row.get("agent_response"),
                         "is_safe": bool(row["is_safe"])})
        labels.append(0 if row["is_safe"] else 1)
    return {"examples": examples, "labels": labels}


def verify(payload, entry):
    examples, labels = payload["examples"], payload["labels"]
    if len(examples) != entry["n"] or len(labels) != entry["n"]:
        raise ValueError("Cohort size differs from manifest")
    if any(y not in (0, 1) for y in labels) or sum(labels) != entry["unsafe"]:
        raise ValueError("Cohort labels differ from manifest")
    actual = hashlib.sha256(serialize(payload)).hexdigest()
    if actual != entry["sha256"]:
        raise ValueError(f"Cohort content/order differs: expected {entry['sha256']}, got {actual}")


def from_source(name, source):
    """Retain source order, except the explicitly seeded SafeRLHF sampling."""
    rows, safe_pool, unsafe_pool = [], [], []
    for row in source:
        prompt = (row.get("user_input" if name == "toxicchat" else "prompt") or "").strip()
        if not prompt:
            continue
        response = None
        if name == "toxicchat":
            if not bool(row.get("human_annotation")):
                continue
            safe = row.get("toxicity", 0) == 0
        elif name == "openai_mod":
            safe = not any(row.get(c, 0) == 1 for c in ("S", "H", "V", "HR", "SH", "S3", "H2", "V2"))
        elif name == "xstest":
            label = (row.get("label") or "").strip().lower()
            if label not in ("safe", "unsafe"):
                continue
            safe = label == "safe"
        elif name == "xstest_response":
            label = (row.get("label") or "").strip().lower()
            if label not in ("harmful", "unharmful"):
                continue
            safe = label == "unharmful"
            response = row["response"].strip() if row.get("response") else None
        elif name == "beavertails":
            safe = bool(row.get("is_safe", True))
            response = (row.get("response") or "").strip() or None
        elif name == "saferlhf":
            for index in (0, 1):
                safe = bool(row.get(f"is_response_{index}_safe", False))
                example = {"user_message": prompt,
                           "agent_response": str(row.get(f"response_{index}") or "").strip() or None,
                           "is_safe": safe}
                (safe_pool if safe else unsafe_pool).append(example)
            continue
        else:
            raise ValueError(f"Unknown dataset: {name}")
        rows.append({"user_message": prompt, "agent_response": response, "is_safe": safe})
    if name == "saferlhf":
        rng = random.Random(0)
        safe_pool = rng.sample(safe_pool, 1000) if len(safe_pool) > 1000 else safe_pool
        unsafe_pool = rng.sample(unsafe_pool, 1000) if len(unsafe_pool) > 1000 else unsafe_pool
        rows = safe_pool + unsafe_pool
        rng.shuffle(rows)
    return normalize(rows)


def prepare(args, manifest):
    from datasets import Dataset, load_dataset

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for name, entry in manifest["datasets"].items():
        if name == "phtest":
            source = [json.loads(line) for line in args.phtest_jsonl.read_text().splitlines() if line.strip()]
            if any(not r["is_safe"] or r.get("agent_response") is not None for r in source):
                raise ValueError("PHTest input must be the saved harmless prompt-only cohort")
            payload = normalize(source)
            provenance[name] = {"input_sha256": sha256(args.phtest_jsonl)}
        else:
            if name == "saferlhf" and args.saferlhf_arrow:
                source = Dataset.from_file(str(args.saferlhf_arrow))
                provenance[name] = {"input_sha256": sha256(args.saferlhf_arrow)}
            else:
                source = load_dataset(entry["repo"], name=entry.get("config"), split=entry["split"])
                provenance[name] = {"dataset_fingerprint": source._fingerprint,
                                    "upstream_revision": None}
            payload = from_source(name, source)
        verify(payload, entry)
        (output / f"cohort-{name}.json").write_bytes(serialize(payload))
        print(f"{name}: {entry['n']} rows, SHA-256 verified", flush=True)
    (output / "preparation.json").write_bytes(serialize({
        "manifest_sha256": sha256(MANIFEST), "sources": provenance,
        "note": "Upstream revisions are not established; exact prepared content and order are enforced by manifest hashes.",
    }))


def format_evaluation_table(results):
    lines = ["| Dataset | N | AUPRC (%) | F1 (%) | Safe accuracy (%) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    mixed = []
    for name, result in results.items():
        metrics = result["metrics"]
        if "auprc" in metrics:
            mixed.append(metrics)
            ap, f1 = f"{100 * metrics['auprc']:.2f}", f"{100 * metrics['f1']:.2f}"
        else:
            ap, f1 = "N/A", "N/A"
        lines.append(f"| {name} | {result['n']} | {ap} | {f1} | {100 * metrics['acc_safe']:.2f} |")
    if mixed:
        ap = 100 * sum(m["auprc"] for m in mixed) / len(mixed)
        f1 = 100 * sum(m["f1"] for m in mixed) / len(mixed)
        lines.append(f"| Macro average (mixed-label datasets) | - | {ap:.2f} | {f1:.2f} | - |")
    return "\n".join(lines) + "\n"


def evaluation_table(args, manifest):
    import torch
    from huggingface_hub import snapshot_download
    from sklearn.metrics import average_precision_score, f1_score
    from llada_guard_infer import load_llada_guard, score_examples

    cohorts = {}
    for name, entry in manifest["datasets"].items():
        payload = json.loads((args.cohorts / f"cohort-{name}.json").read_text())
        verify(payload, entry)
        cohorts[name] = payload
    checkpoint = Path(snapshot_download(manifest["release"]["repo"],
                                       revision=manifest["release"]["revision"],
                                       allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))
    if sha256(checkpoint / "adapter_model.safetensors") != manifest["release"]["adapter_sha256"]:
        raise ValueError("Released adapter hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).parent
    import llada_guard
    code_paths = list(package.glob("*.py")) + list(Path(llada_guard.__file__).parent.glob("*.py"))
    (args.output / "protocol.json").write_bytes(serialize({
        "manifest": manifest, "manifest_sha256": sha256(MANIFEST),
        "batch_size": args.batch_size, "torch_version": torch.__version__,
        "code_sha256": {f"{p.parent.name}/{p.name}": sha256(p) for p in code_paths},
    }))
    torch.manual_seed(42)
    model, tokenizer = load_llada_guard(str(checkpoint), device=args.device, torch_dtype="bfloat16")
    model.eval()
    results = {}
    for name, payload in cohorts.items():
        torch.manual_seed(42)
        start = time.perf_counter()
        scores = score_examples(model, tokenizer, payload["examples"], batch_size=args.batch_size,
                                max_seq_length=1536, t_values=(.15, .15, .15), progress=True)
        labels = payload["labels"]
        if len(scores) != len(labels):
            raise ValueError("Inference changed cohort size")
        predicted = [int(s > 0) for s in scores]
        metrics = {"f1": float(f1_score(labels, predicted, zero_division=0)),
                   "acc_safe": sum(y == 0 and p == 0 for y, p in zip(labels, predicted)) / labels.count(0)}
        if 0 < sum(labels) < len(labels):
            metrics["auprc"] = float(average_precision_score(labels, scores))
        results[name] = {"n": len(labels), "metrics": metrics,
                         "seconds": time.perf_counter() - start}
        (args.output / f"predictions-{name}.json").write_bytes(serialize({"labels": labels, "scores": scores}))
        (args.output / "evaluation_table.json").write_bytes(serialize(results))
        print(name, results[name], flush=True)
    mixed = [r for name, r in results.items() if name != "phtest"]
    (args.output / "complete.json").write_bytes(serialize({
        "status": "complete",
        "macro_percent": {k: 100 * sum(r["metrics"][k] for r in mixed) / len(mixed) for k in ("auprc", "f1")},
    }))
    table = format_evaluation_table(results)
    (args.output / "evaluation_table.md").write_text(table)
    print(table, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--phtest-jsonl", type=Path, required=True)
    prep.add_argument("--saferlhf-arrow", type=Path)
    evaluate_parser = commands.add_parser("evaluation_table")
    evaluate_parser.add_argument("--cohorts", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=4)
    evaluate_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    (prepare if args.command == "prepare" else evaluation_table)(args, manifest)


if __name__ == "__main__":
    main()
