import hashlib
import json

import pytest

from llada_guard_infer.benchmark import MANIFEST, format_evaluation_table, from_source, normalize, serialize, verify


def test_hash_enforces_content_and_order():
    payload = normalize([{"user_message": "a", "is_safe": True},
                         {"user_message": "b", "is_safe": False}])
    entry = {"n": 2, "unsafe": 1, "sha256": hashlib.sha256(serialize(payload)).hexdigest()}
    verify(payload, entry)
    payload["examples"].reverse()
    payload["labels"].reverse()
    with pytest.raises(ValueError, match="content/order"):
        verify(payload, entry)


def test_toxicchat_human_filter():
    result = from_source("toxicchat", [
        {"user_input": " a ", "human_annotation": True, "toxicity": 1},
        {"user_input": "b", "human_annotation": False, "toxicity": 0},
        {"user_input": " ", "human_annotation": True, "toxicity": 0},
    ])
    assert result["labels"] == [1]
    assert result["examples"][0]["user_message"] == "a"


def test_saferlhf_sampling_is_balanced_and_deterministic():
    source = [{"prompt": str(i), "response_0": "safe", "response_1": "unsafe",
               "is_response_0_safe": True, "is_response_1_safe": False} for i in range(1200)]
    first = from_source("saferlhf", source)
    assert first == from_source("saferlhf", source)
    assert len(first["labels"]) == 2000
    assert sum(first["labels"]) == 1000


def test_manifest_has_seven_datasets():
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest["datasets"]) == 7
    for entry in manifest["datasets"].values():
        assert entry["safe"] + entry["unsafe"] == entry["n"]
        assert len(entry["sha256"]) == 64


def test_evaluation_table_excludes_single_class_from_macro():
    table = format_evaluation_table({
        "mixed": {"n": 10, "metrics": {"auprc": .9, "f1": .8, "acc_safe": .7}},
        "phtest": {"n": 5, "metrics": {"f1": 0., "acc_safe": .6}},
    })
    assert "| phtest | 5 | N/A | N/A | 60.00 |" in table
    assert "| Macro average (mixed-label datasets) | - | 90.00 | 80.00 | - |" in table
    assert "paper" not in table
    assert "paper_percent" not in MANIFEST.read_text()
