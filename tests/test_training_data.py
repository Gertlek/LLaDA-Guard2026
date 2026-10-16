import json

import pytest

from llada_guard.training_data import file_sha256, load_examples, load_training_data


def write_rows(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_hash_and_schema(tmp_path):
    rows = [{"user_message": "Example", "is_safe": True, "agent_response": None}]
    path = write_rows(tmp_path, rows)
    assert load_examples(path, file_sha256(path)) == rows
    with pytest.raises(ValueError, match="hash mismatch"):
        load_examples(path, "0" * 64)


@pytest.mark.parametrize(
    "change",
    [
        {"is_safe": "false"},
        {"user_message": ""},
        {"agent_response": ""},
        {"prompt_label": "unsafe"},
        {"prompt_label": "unknown"},
    ],
)
def test_reject_invalid_supervision(tmp_path, change):
    path = write_rows(
        tmp_path, [{"user_message": "Example", "is_safe": True, **change}]
    )
    with pytest.raises(ValueError):
        load_examples(path, file_sha256(path))


def test_preparation_manifest(tmp_path):
    rows = [{"user_message": "Example", "is_safe": True}]
    path = write_rows(tmp_path, rows)
    manifest = {"sha256": file_sha256(path)}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    assert load_training_data(path) == (rows, manifest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_training_data(path)
