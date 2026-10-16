from types import SimpleNamespace

import peft
import pytest
import torch

from llada_guard import model as loading


@pytest.mark.parametrize("revision", [None, "08b83a6feb34df1a6011b80c3c00c7563e963b07"])
def test_load_adapter_preserves_revision_and_legacy_default(monkeypatch, revision):
    calls = {}
    base = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        peft.PeftConfig,
        "from_pretrained",
        lambda path: SimpleNamespace(revision=revision),
    )

    def load_base(name, **kwargs):
        calls["model"] = kwargs
        return base

    def load_tokenizer(name, **kwargs):
        calls["tokenizer"] = kwargs
        return object()

    monkeypatch.setattr(loading.AutoModel, "from_pretrained", load_base)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        peft.PeftModel, "from_pretrained", lambda model, path, **kwargs: model
    )
    loading.load_checkpoint("base", "adapter", device="cpu")
    for target in ("model", "tokenizer"):
        if revision is None:
            assert "revision" not in calls[target]
        else:
            assert calls[target]["revision"] == revision
