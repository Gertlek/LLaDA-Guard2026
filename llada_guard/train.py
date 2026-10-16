"""Train a LLaDA-Guard adapter from hash-verified local data snapshots."""

import argparse
from contextlib import nullcontext
from functools import partial
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from .data import SafetyDataset, collate_fn
from .eval import apply_eval_inference_fastpaths
from .objectives import training_loss
from .training_data import file_sha256, load_training_data
from .training_runtime import RouteMixedBatchSampler, build_scheduler, shard_examples


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def enable_checkpointing(model, count):
    base = model.get_base_model()
    blocks = base.model.transformer.blocks
    if not 0 <= count <= len(blocks):
        raise ValueError("Invalid number of checkpointed transformer blocks")
    for block in blocks[:count]:
        original = block.forward

        def forward(*args, _original=original, **kwargs):
            return checkpoint(_original, *args, use_reentrant=False, **kwargs)

        block.forward = forward


def validate_config(cfg):
    required = {"model", "revision", "objective", "lora", "training"}
    if set(cfg) != required:
        raise ValueError(f"Config keys must be {sorted(required)}")
    if cfg["model"] != "GSAI-ML/LLaDA-8B-Instruct":
        raise ValueError("This reproduction path requires LLaDA-8B-Instruct")
    if len(cfg["revision"]) != 40 or any(
        c not in "0123456789abcdef" for c in cfg["revision"]
    ):
        raise ValueError("An immutable model revision is required")
    if cfg["objective"] not in {"ranking", "generative"}:
        raise ValueError("objective must be ranking or generative")
    expected_training = {
        "epochs",
        "batch_size",
        "accumulation",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_length",
        "max_grad_norm",
        "seed",
        "checkpoint_blocks",
        "workers",
        "world_size",
    }
    if set(cfg["training"]) != expected_training:
        raise ValueError(f"training keys must be {sorted(expected_training)}")
    if set(cfg["lora"]) != {"rank", "alpha", "dropout"}:
        raise ValueError("lora requires rank, alpha and dropout")
    for key in ("epochs", "batch_size", "accumulation", "max_length", "world_size"):
        if type(cfg["training"][key]) is not int or cfg["training"][key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("warmup_steps", "checkpoint_blocks", "workers", "seed"):
        if type(cfg["training"][key]) is not int or cfg["training"][key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    for key in ("learning_rate", "max_grad_norm", "weight_decay"):
        value = cfg["training"][key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
        if key != "weight_decay" and value == 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("rank", "alpha"):
        if type(cfg["lora"][key]) is not int or cfg["lora"][key] < 1:
            raise ValueError(f"lora.{key} must be a positive integer")
    if not 0 <= cfg["lora"]["dropout"] < 1:
        raise ValueError("lora.dropout must be in [0, 1)")


def run(args):
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    examples, manifest = load_training_data(args.data)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world != cfg["training"]["world_size"]:
        raise ValueError(
            f"Config requires {cfg['training']['world_size']} ranks; got {world}"
        )
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("max_updates must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": len(examples),
                    "world_size": world,
                    "global_batch": world
                    * cfg["training"]["batch_size"]
                    * cfg["training"]["accumulation"],
                    "objective": cfg["objective"],
                },
                indent=2,
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires a CUDA GPU; use --dry-run for data verification"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    try:
        _train(args, cfg, examples, manifest, output, rank, world, device)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _train(args, cfg, examples, manifest, output, rank, world, device):
    training = cfg["training"]
    seed = training["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", cfg)
        write_json(output / "data.json", manifest)
        write_json(
            output / "run.json",
            {
                "world_size": world,
                "max_updates": args.max_updates,
                "model_revision": cfg["revision"],
                "data_sha256": file_sha256(args.data),
                "source": {
                    p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")
                },
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "peft",
                        "accelerate",
                        "datasets",
                    )
                },
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
            },
        )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["revision"], trust_remote_code=True
    )
    base = AutoModel.from_pretrained(
        cfg["model"],
        revision=cfg["revision"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
    )
    lora = cfg["lora"]
    model = get_peft_model(
        base,
        LoraConfig(
            revision=cfg["revision"],
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules="all-linear",
            bias="none",
        ),
    )
    # PEFT promotes adapters to FP32; the recovered trainer used the base dtype.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != base.dtype:
            parameter.data = parameter.data.to(base.dtype)
    apply_eval_inference_fastpaths(model)
    enable_checkpointing(model, training["checkpoint_blocks"])
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world > 1
        else model
    )
    examples = shard_examples(examples, seed, rank, world)
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    pad_id = tokenizer.eos_token_id or 0
    dataset = SafetyDataset(examples, tokenizer, max_seq_length=training["max_length"])
    loader = DataLoader(
        dataset,
        batch_sampler=RouteMixedBatchSampler(
            [bool(ex.get("agent_response")) for ex in examples], training["batch_size"]
        ),
        collate_fn=partial(collate_fn, pad_id=pad_id),
        num_workers=training["workers"],
        persistent_workers=training["workers"] > 0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        fused=device.type == "cuda",
    )
    accumulation = training["accumulation"]
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    scheduler = build_scheduler(
        optimizer, training["warmup_steps"], steps_per_epoch * training["epochs"]
    )
    update = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(training["epochs"]):
        wrapped.train()
        window_loss = 0.0
        started = time.monotonic()
        for index, batch in enumerate(loader):
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            end_window = (index + 1) % accumulation == 0 or index + 1 == len(loader)
            window_size = min(
                accumulation, len(loader) - (index // accumulation) * accumulation
            )
            sync = wrapped.no_sync() if world > 1 and not end_window else nullcontext()
            with sync:
                loss, _ = training_loss(
                    wrapped, batch, objective=cfg["objective"], pad_id=pad_id
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / window_size).backward()
            window_loss += loss.detach().item() / window_size
            if not end_window:
                continue
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["max_grad_norm"], error_if_nonfinite=True
            )
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            reduced = torch.tensor(window_loss, device=device)
            if world > 1:
                torch.distributed.all_reduce(reduced)
                reduced /= world
            if rank == 0:
                record = {
                    "update": update,
                    "epoch": epoch,
                    "loss": reduced.item(),
                    "grad_norm": norm.item(),
                    "lr": lr,
                    "seconds": time.monotonic() - started,
                }
                with (output / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
            stop = args.max_updates is not None and update >= args.max_updates
            window_loss = 0.0
            started = time.monotonic()
            if stop:
                break
        if args.max_updates is not None and update >= args.max_updates:
            break
    if rank == 0:
        model.save_pretrained(output / "last")
        tokenizer.save_pretrained(output / "last")
        write_json(output / "complete.json", {"updates": update})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--data", required=True, help="Training JSONL with its preparation manifest"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Smoke run limit; does not shorten the LR schedule",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
