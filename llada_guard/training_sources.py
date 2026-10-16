"""Pinned training sources with binary prompt and response labels."""

from datasets import load_dataset as _load_dataset

SOURCE_REVISIONS = {
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0": "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c",
    "allenai/wildguardmix": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
}


def load_dataset(name, *args, **kwargs):
    return _load_dataset(name, *args, revision=SOURCE_REVISIONS[name], **kwargs)


def _binary_label(value: str | None) -> str:
    label = (value or "").strip().lower()
    return {"harmful": "unsafe", "unharmful": "safe"}.get(label, label)


def _load_rows(dataset, source: str, prompt_key: str, response_key: str) -> list[dict]:
    rows = []
    for row in dataset:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        rows.append(
            {
                "user_message": prompt,
                "agent_response": (row.get("response") or "").strip() or None,
                "prompt_label": _binary_label(row.get(prompt_key)),
                "response_label": _binary_label(row.get(response_key)),
                "source": source,
            }
        )
    return rows


def load_aegis(split: str = "train") -> list[dict]:
    dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split)
    return _load_rows(dataset, "aegis", "prompt_label", "response_label")


def load_wildguardtrain() -> list[dict]:
    dataset = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    return _load_rows(
        dataset, "wildguardtrain", "prompt_harm_label", "response_harm_label"
    )
