from llada_guard import training_sources as sources
import json
import sys
from collections import Counter

import pytest

from llada_guard.training_data import load_training_data
from scripts import prepare_paper_training as prepare
from scripts.prepare_paper_training import (
    materialize_rows,
    route_family,
    sample_balanced,
)


def test_source_loader_pins_revision(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sources, "_load_dataset", lambda *a, **kw: calls.append((a, kw))
    )
    sources.load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    assert calls == [
        (
            ("allenai/wildguardmix", "wildguardtrain"),
            {"split": "train", "revision": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"},
        )
    ]


def test_aegis_uses_binary_labels_without_categories_or_text_filtering(monkeypatch):
    monkeypatch.setattr(
        sources,
        "load_dataset",
        lambda *a, **kw: [
            {
                "prompt": " keep ",
                "response": None,
                "violated_categories": "Violence, Unknown",
                "prompt_label": "unsafe",
            },
            {
                "prompt": "exclude",
                "response": None,
                "violated_categories": "safe",
                "prompt_label": "safe",
            },
        ],
    )
    rows = materialize_rows(sources.load_aegis())
    assert len(rows) == 2
    assert rows[0]["user_message"] == "keep"
    assert all("violated_categories" not in row for row in rows)
    assert rows[0]["is_safe"] is False
    assert rows[1]["is_safe"] is True


def test_wildguard_materialization_preserves_scoped_supervision(monkeypatch):
    monkeypatch.setattr(
        sources,
        "load_dataset",
        lambda *a, **kw: [
            {
                "prompt": "unsafe request",
                "response": "refusal",
                "prompt_harm_label": "harmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "refusal",
            },
            {
                "prompt": "safe request",
                "response": "unnecessary refusal",
                "prompt_harm_label": "unharmful",
                "response_harm_label": "unharmful",
                "response_refusal_label": "refusal",
            },
        ],
    )
    raw = sources.load_wildguardtrain()
    materialized = materialize_rows(raw)
    families = {route_family(row) for row in materialized}
    assert {"response_safe", "prompt_unsafe", "prompt_safe"} <= families
    assert all(
        row["response_label"] == "safe"
        for row in materialized
        if row["agent_response"] is not None
    )


def test_missing_labels_and_blank_responses(monkeypatch):
    monkeypatch.setattr(
        sources,
        "load_dataset",
        lambda *a, **kw: [
            {
                "prompt": "empty response",
                "response": "  ",
                "prompt_label": "safe",
                "response_label": "unsafe",
            },
            {"prompt": "no labels", "violated_categories": "Violence"},
            {"prompt": "response only", "response": "answer", "response_label": "safe"},
            {"prompt": " ", "prompt_label": "safe"},
        ],
    )
    rows = materialize_rows(sources.load_aegis())
    assert len(rows) == 2
    assert rows[0]["agent_response"] is None
    assert rows[1]["is_safe"] is True


def balanced_pool():
    return materialize_rows(
        [
            {
                "user_message": f"prompt {i}",
                "agent_response": f"response {i}",
                "prompt_label": label,
                "response_label": label,
                "source": "aegis",
            }
            for label in ("safe", "unsafe")
            for i in range(10)
        ]
    )


def test_sampling_is_balanced_and_deterministic():
    rows = balanced_pool()
    selected = sample_balanced(rows, 16, 42)
    assert selected == sample_balanced(rows, 16, 42)
    assert selected != sample_balanced(rows, 16, 43)
    assert Counter(map(route_family, selected)) == dict.fromkeys(prepare.FAMILIES, 4)
    assert len({json.dumps(row, sort_keys=True) for row in selected}) == 16


@pytest.mark.parametrize("total", [0, -4, 3, 44])
def test_sampling_rejects_invalid_or_unavailable_counts(total):
    with pytest.raises(ValueError):
        sample_balanced(balanced_pool(), total, 42)


def test_prepare_writes_loadable_data_and_refuses_overwrite(monkeypatch, tmp_path):
    output = tmp_path / "train.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--output", str(output), "--target-total", "16"]
    )
    monkeypatch.setattr(prepare, "load_aegis", balanced_pool)
    monkeypatch.setattr(prepare, "load_wildguardtrain", lambda: [])
    assert prepare.main() == 0
    rows, manifest = load_training_data(output)
    assert len(rows) == 16
    assert manifest["test_overlap_filtered"] is False
    assert manifest["source_revisions"] == sources.SOURCE_REVISIONS
    with pytest.raises(FileExistsError):
        prepare.main()
