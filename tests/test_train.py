import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llada_guard.train import enable_checkpointing, run, validate_config
from llada_guard.training_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["paper_ranking", "generative"])
def test_release_configs(name):
    validate_config(json.loads((ROOT / "configs" / f"{name}.json").read_text()))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "workers", -1),
        ("training", "warmup_steps", 1.5),
        ("training", "learning_rate", float("nan")),
        ("training", "learning_rate", 0),
        ("training", "max_grad_norm", -1),
        ("lora", "rank", 0),
        ("lora", "dropout", 1),
    ],
)
def test_invalid_config_is_rejected(section, key, value):
    cfg = json.loads((ROOT / "configs/paper_ranking.json").read_text())
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_dry_run_checks_data_without_loading_model(tmp_path, monkeypatch, capsys):
    cfg = json.loads((ROOT / "configs/paper_ranking.json").read_text())
    cfg["training"]["world_size"] = 1
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"user_message": "Example", "is_safe": True}) + "\n")
    manifest = rows.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"sha256": file_sha256(rows)}))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    def unexpected(*args, **kwargs):
        raise AssertionError("dry-run loaded a model")

    monkeypatch.setattr("llada_guard.train.AutoModel.from_pretrained", unexpected)
    output = tmp_path / "run"
    args = SimpleNamespace(
        config=config, data=rows, output=output, dry_run=True, max_updates=None
    )
    run(args)
    assert json.loads(capsys.readouterr().out)["global_batch"] == 4
    assert not output.exists()
    rows.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(args)


def test_checkpoint_wrapping_preserves_gradients():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(2)])
            self.model = SimpleNamespace(
                transformer=SimpleNamespace(blocks=self.blocks)
            )

        def get_base_model(self):
            return self

        def forward(self, x):
            for block in self.blocks:
                x = block(x).sin()
            return x

    torch.manual_seed(42)
    model = Model()
    x = torch.randn(2, 3)
    model(x).sum().backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    enable_checkpointing(model, 2)
    model(x).sum().backward()
    for parameter, expected in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, expected)
