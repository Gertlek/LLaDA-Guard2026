"""Opt-in CPU integration test using the pinned LLaDA code, with tiny random weights."""

import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from llada_guard import train
from llada_guard.data import collate_fn
from llada_guard.eval import binary_metrics_from_scores, collect_eval_scores
from llada_guard_infer import load_llada_guard


@pytest.mark.skipif(
    not os.environ.get("LLADA_TEST_CACHE"),
    reason="Set LLADA_TEST_CACHE to cached pinned LLaDA code",
)
@pytest.mark.parametrize("objective", ["ranking", "generative"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_llada_train_save_reload(tmp_path, monkeypatch, objective, dtype):
    exercise_training(tmp_path, monkeypatch, objective, dtype=dtype)


def exercise_training(
    tmp_path, monkeypatch, objective, rank=0, world=1, dtype=torch.float32
):
    torch.set_num_threads(2)
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/paper_ranking.json").read_text()
    )
    cfg["objective"] = objective
    cfg["lora"].update(rank=2, alpha=4, dropout=0.0)
    cfg["training"].update(
        world_size=world,
        workers=0,
        batch_size=2,
        epochs=2,
        checkpoint_blocks=1,
        learning_rate=0.001,
        warmup_steps=0,
    )
    kwargs = dict(
        revision=cfg["revision"],
        cache_dir=os.environ["LLADA_TEST_CACHE"],
        trust_remote_code=True,
        local_files_only=True,
    )
    config = AutoConfig.from_pretrained(cfg["model"], **kwargs)
    config.d_model, config.n_heads, config.n_kv_heads, config.n_layers = 32, 4, 4, 2
    config.mlp_hidden_size, config.mlp_ratio = 64, 2
    config.init_device = "cpu"
    config._name_or_path = str(
        Path(os.environ["LLADA_TEST_CACHE"])
        / "models--GSAI-ML--LLaDA-8B-Instruct"
        / "snapshots"
        / cfg["revision"]
    )
    torch.manual_seed(42)
    base = AutoModel.from_config(config, trust_remote_code=True).to(dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], **kwargs)
    base_path = tmp_path / "base"
    if rank == 0:
        base.save_pretrained(base_path)
        tokenizer.save_pretrained(base_path)
    if world > 1:
        torch.distributed.barrier()
    base.config._name_or_path = str(base_path)

    items = []
    for row in range(2):
        item = {
            "is_unsafe": row,
            "has_response": row,
            "prompt_is_unsafe": 0,
            "response_is_unsafe": 1,
            "prompt_label_mask": 1,
            "response_label_mask": 1,
            "example_index": row,
        }
        for branch, verdict in (("safe", 7), ("unsafe", 8)):
            item[f"gen_{branch}_input_ids"] = torch.tensor([1, 2, 3, 4, verdict])
            item[f"gen_{branch}_block_ids"] = torch.tensor([0, 1, 1, 2, 0])
            item[f"gen_{branch}_real_length"] = 5
        item.update(
            hazard_input_ids=item["gen_unsafe_input_ids"],
            hazard_block_ids=item["gen_unsafe_block_ids"],
            hazard_real_length=5,
        )
        items.append(item)
    batch = collate_fn(items, pad_id=0)
    examples = [
        {"agent_response": None, "index": 0},
        {"agent_response": "response", "index": 1},
    ]
    manifest = tmp_path / "manifest.json"
    if rank == 0:
        manifest.write_text("{}")
    if world > 1:
        torch.distributed.barrier()
    output = tmp_path / "run"
    args = SimpleNamespace(max_updates=2, data=manifest)
    with monkeypatch.context() as patch:
        patch.setattr(train.AutoModel, "from_pretrained", lambda *a, **kw: base)
        patch.setattr(
            train.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
        )
        patch.setattr(
            train,
            "SafetyDataset",
            lambda rows, *a, **kw: [items[row["index"]] for row in rows],
        )
        train._train(
            args,
            cfg,
            examples,
            {},
            output,
            rank,
            world,
            torch.device("cpu"),
        )
    if world > 1:
        parameters = torch.cat(
            [p.detach().flatten() for p in base.parameters() if p.requires_grad]
        )
        gathered = [torch.empty_like(parameters) for _ in range(world)]
        torch.distributed.all_gather(gathered, parameters)
        for other in gathered:
            torch.testing.assert_close(parameters, other, rtol=0, atol=0)
        torch.distributed.barrier()
    if rank != 0:
        return
    assert all(p.dtype == dtype for p in base.parameters() if p.requires_grad)
    assert json.loads((output / "complete.json").read_text())["updates"] == 2
    records = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert all(record["grad_norm"] > 0 for record in records)
    assert not (output / "best").exists()
    assert not list(output.glob("evaluation-*.json"))
    with torch.no_grad():
        expected = binary_metrics_from_scores(
            *collect_eval_scores(base, [batch], "cpu", t_values=(0.25, 0.5), pad_id=0)
        )
    restored, _ = load_llada_guard(
        output / "last",
        base_model=str(base_path),
        device="cpu",
        torch_dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
    )
    with torch.no_grad():
        labels, scores = collect_eval_scores(
            restored, [batch], "cpu", t_values=(0.25, 0.5), pad_id=0
        )
    assert labels == [0.0, 1.0]
    assert torch.isfinite(torch.tensor(scores)).all()
    actual = binary_metrics_from_scores(labels, scores)
    assert actual == pytest.approx(expected, abs=1e-6)


def distributed_worker(rank, directory, objective):
    directory = Path(directory)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{directory / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            exercise_training(directory, patch, objective, rank=rank, world=2)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not os.environ.get("LLADA_TEST_CACHE"),
    reason="Set LLADA_TEST_CACHE to cached pinned LLaDA code",
)
@pytest.mark.parametrize("objective", ["ranking", "generative"])
def test_two_rank_train_save_reload(tmp_path, objective):
    torch.multiprocessing.spawn(
        distributed_worker, args=(str(tmp_path), objective), nprocs=2
    )
