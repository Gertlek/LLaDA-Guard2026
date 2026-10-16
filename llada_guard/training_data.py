"""Hash-verified, materialized training inputs. No implicit dataset downloads."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path, expected_sha256):
    path = Path(path)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Data hash mismatch: {path.name}: expected {expected_sha256}, got {actual}"
        )
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Row {line_number} must be an object")
            if (
                not isinstance(row.get("user_message"), str)
                or not row["user_message"].strip()
            ):
                raise ValueError(f"Row {line_number} must have a nonempty user_message")
            if type(row.get("is_safe")) is not bool:
                raise ValueError(f"Row {line_number} must have a boolean is_safe")
            response = row.get("agent_response")
            if response is not None and (
                not isinstance(response, str) or not response.strip()
            ):
                raise ValueError(
                    f"Row {line_number} must have a nonempty response or null"
                )
            scope = "prompt_label" if response is None else "response_label"
            label = row.get(scope)
            if label not in (None, "", "safe", "unsafe"):
                raise ValueError(f"Row {line_number} has an invalid {scope}")
            if label in ("safe", "unsafe") and (label == "safe") != row["is_safe"]:
                raise ValueError(
                    f"Row {line_number} has contradictory scoped and binary labels"
                )
            examples.append(row)
    if not examples:
        raise ValueError("Dataset is empty")
    return examples


def load_training_data(path):
    """Read training JSONL and verify the adjacent preparation manifest."""
    path = Path(path)
    manifest = json.loads(
        path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    return load_examples(path, manifest["sha256"]), manifest
