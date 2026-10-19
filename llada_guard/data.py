from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from .prompt import (
    build_prompt,
    build_prompt_parts,
    build_target,
    canonicalize_violated_categories,
)

PROMPT_BLOCK_ID = 1
RESPONSE_BLOCK_ID = 2
MASK_TOKEN_TEXT = "<|mdm_mask|>"


class SafetyDataset(Dataset):
    """Materialize the safe/unsafe prompt pairs used by masked-scope scoring."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        category_augmentation: bool = False,
        category_augmentation_prob: float = 0.0,
        prompt_only_augmentation_prob: float = 0.0,
        prompt_format: str = "aegis_taxonomy",
        gen_only: bool = False,
        skip_cls: bool = False,
    ):
        del category_augmentation, category_augmentation_prob, prompt_only_augmentation_prob
        del skip_cls
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_format = prompt_format
        self.gen_only = gen_only

    def __len__(self):
        return len(self.examples)

    def _chat_token_count(
        self, content: str, trusted_span: Optional[tuple[int, int]] = None
    ) -> int:
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids, _ = self._encode_chat(prompt_str, content, trusted_span)
        return len(ids) + len(self.tokenizer("")["input_ids"])

    def _encode_chat(
        self,
        prompt_str: str,
        content: str,
        trusted_span: Optional[tuple[int, int]] = None,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Tokenize a chat-formatted prompt without parsing special tokens in content.

        Special-token strings typed by a user or agent (e.g. ``<|eot_id|>``,
        ``<|mdm_mask|>``) stay plain text; only the chat template's own control
        tokens are parsed. ``trusted_span`` (offsets within ``content``) marks
        release-generated text, such as the synthetic mask response, whose
        control tokens are kept.
        """
        content_start = prompt_str.find(content)
        if content_start < 0:
            raise ValueError("Classification prompt not found inside chat template")
        content_end = content_start + len(content)
        segments = [(0, content_start, False)]
        if trusted_span is None:
            segments.append((content_start, content_end, True))
        else:
            trusted_start = content_start + trusted_span[0]
            trusted_end = content_start + trusted_span[1]
            segments += [
                (content_start, trusted_start, True),
                (trusted_start, trusted_end, False),
                (trusted_end, content_end, True),
            ]
        segments.append((content_end, len(prompt_str), False))

        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for start, end, split_special in segments:
            if start == end:
                continue
            encoding = self.tokenizer(
                prompt_str[start:end],
                add_special_tokens=False,
                return_offsets_mapping=True,
                split_special_tokens=split_special,
            )
            input_ids.extend(encoding["input_ids"])
            offsets.extend((s + start, e + start) for s, e in encoding["offset_mapping"])
        return input_ids, offsets

    def _gen_total_len(
        self,
        verdict_text: str,
        user_message: str,
        agent_response: Optional[str],
        categories: Optional[list[str]],
        violated_categories: Optional[list[str]] = None,
        synthetic_response: bool = False,
    ) -> int:
        prefix, conversation, suffix = build_prompt_parts(
            user_message,
            agent_response,
            categories=categories,
            prompt_format=self.prompt_format,
        )
        trusted_span = None
        if synthetic_response and agent_response is not None:
            trusted_end = len(prefix) + len(conversation)
            trusted_span = (trusted_end - len(agent_response), trusted_end)
        prompt_len = self._chat_token_count(prefix + conversation + suffix, trusted_span)
        if verdict_text == "unsafe" and violated_categories:
            label_text = build_target(
                False,
                violated_categories,
                prompt_format=self.prompt_format,
            )
        else:
            label_text = verdict_text
        label_ids = self.tokenizer(label_text, add_special_tokens=False)["input_ids"]
        eos_extra = 1 if self.tokenizer.eos_token_id is not None else 0
        return prompt_len + len(label_ids) + eos_extra

    def _truncate_to_budget(
        self,
        user_message: str,
        agent_response: Optional[str],
        categories: Optional[list[str]],
        violated_categories: Optional[list[str]] = None,
        allow_agent_drop: bool = True,
    ) -> tuple[str, Optional[str]]:
        """Trim the response first, then the user text, until the pair fits."""
        while True:
            gen_safe_total = self._gen_total_len(
                "safe",
                user_message,
                agent_response,
                categories,
                synthetic_response=not allow_agent_drop,
            )
            gen_unsafe_total = self._gen_total_len(
                "unsafe",
                user_message,
                agent_response,
                categories,
                violated_categories=violated_categories,
                synthetic_response=not allow_agent_drop,
            )
            overflow = max(gen_safe_total, gen_unsafe_total) - self.max_seq_length
            if overflow <= 0:
                return user_message, agent_response

            if agent_response and allow_agent_drop:
                agent_ids = self.tokenizer(agent_response, add_special_tokens=False)["input_ids"]
                drop = min(max(overflow, 1), len(agent_ids))
                agent_ids = agent_ids[:-drop]
                agent_response = (
                    self.tokenizer.decode(agent_ids, skip_special_tokens=True) if agent_ids else None
                )
                continue

            user_ids = self.tokenizer(user_message, add_special_tokens=False)["input_ids"]
            drop = min(max(overflow, 1), len(user_ids))
            user_ids = user_ids[:-drop]
            user_message = self.tokenizer.decode(user_ids, skip_special_tokens=True) if user_ids else ""

    def _build_gen_sequence(
        self,
        verdict_text: str,
        user_message: str,
        agent_response: Optional[str],
        categories: Optional[list[str]],
        violated_categories: Optional[list[str]] = None,
        synthetic_response: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix, conversation_text, suffix = build_prompt_parts(
            user_message,
            agent_response,
            categories=categories,
            prompt_format=self.prompt_format,
        )
        prompt_text = prefix + conversation_text + suffix
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        content_start = prompt_str.find(prompt_text)
        if content_start < 0:
            raise ValueError("Classification prompt not found inside chat template")
        conversation_start = content_start + len(prefix)

        user_span = f"User: {user_message}"
        user_start = conversation_start
        user_end = user_start + len(user_span)

        response_start = None
        response_end = None
        if agent_response is not None:
            response_span = f"\nAgent: {agent_response}"
            response_start = user_end
            response_end = response_start + len(response_span)

        trusted_span = None
        if synthetic_response and agent_response is not None:
            trusted_end = response_end - content_start
            trusted_span = (trusted_end - len(agent_response), trusted_end)
        prompt_ids, prompt_offsets = self._encode_chat(
            prompt_str, prompt_text, trusted_span
        )

        prompt_block_ids = []
        for start, end in prompt_offsets:
            in_user = end > user_start and start < user_end
            in_response = (
                response_start is not None
                and response_end is not None
                and end > response_start
                and start < response_end
            )
            if in_user:
                prompt_block_ids.append(PROMPT_BLOCK_ID)
            elif in_response:
                prompt_block_ids.append(RESPONSE_BLOCK_ID)
            else:
                prompt_block_ids.append(0)

        include_cats = verdict_text == "unsafe" and violated_categories
        if include_cats:
            label_text = build_target(
                False,
                violated_categories,
                prompt_format=self.prompt_format,
            )
        else:
            label_text = verdict_text

        label_encoding = self.tokenizer(
            label_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        label_ids = label_encoding["input_ids"]
        label_offsets = label_encoding["offset_mapping"]

        if include_cats:
            char_blocks = [0] * len(label_text)
            pos = len("unsafe") + 1
            for cat_idx, cat in enumerate(violated_categories):
                for offset in range(len(cat)):
                    if pos + offset < len(char_blocks):
                        char_blocks[pos + offset] = 3 + cat_idx
                pos += len(cat) + 1
            label_block_ids = []
            for start, end in label_offsets:
                block_id = 0
                for char_idx in range(start, min(end, len(char_blocks))):
                    block_id = max(block_id, char_blocks[char_idx])
                label_block_ids.append(block_id)
        else:
            label_block_ids = [0] * len(label_ids)

        input_ids = list(prompt_ids) + list(label_ids)
        block_ids = prompt_block_ids + label_block_ids

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            input_ids.append(eos_id)
            block_ids.append(0)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(block_ids, dtype=torch.long),
        )

    def __getitem__(self, idx):
        ex = self.examples[idx]
        user_message = str(ex["user_message"])
        agent_response = ex.get("agent_response")
        is_safe = bool(ex.get("is_safe", True))
        violated = canonicalize_violated_categories(ex.get("violated_categories", []))

        had_response = agent_response is not None
        rendered_agent_response = str(agent_response) if had_response else MASK_TOKEN_TEXT
        user_message, rendered_agent_response = self._truncate_to_budget(
            user_message,
            rendered_agent_response,
            None,
            violated_categories=violated,
            allow_agent_drop=had_response,
        )
        has_response = int(had_response and rendered_agent_response is not None)
        prompt_text = build_prompt(
            user_message,
            rendered_agent_response,
            prompt_format=self.prompt_format,
        )

        gen_safe_input_ids, gen_safe_block_ids = self._build_gen_sequence(
            "safe",
            user_message,
            rendered_agent_response,
            None,
            synthetic_response=not had_response,
        )
        gen_unsafe_input_ids, gen_unsafe_block_ids = self._build_gen_sequence(
            "unsafe",
            user_message,
            rendered_agent_response,
            None,
            synthetic_response=not had_response,
        )
        hazard_label_mask = int((not is_safe) and bool(violated))
        if hazard_label_mask:
            hazard_input_ids, hazard_block_ids = self._build_gen_sequence(
                "unsafe",
                user_message,
                rendered_agent_response,
                None,
                violated_categories=violated,
                synthetic_response=not had_response,
            )
        else:
            hazard_input_ids = gen_unsafe_input_ids
            hazard_block_ids = gen_unsafe_block_ids

        prompt_label = (ex.get("prompt_label") or "").strip().lower()
        response_label = (ex.get("response_label") or "").strip().lower()
        prompt_label_known = prompt_label in {"safe", "unsafe"}
        response_label_known = had_response and response_label in {"safe", "unsafe"}
        if not prompt_label_known and not had_response:
            prompt_label_known = True
        if not response_label_known and had_response:
            response_label_known = True

        prompt_is_unsafe = (
            int(prompt_label == "unsafe")
            if prompt_label in {"safe", "unsafe"}
            else int((not is_safe) and not had_response)
        )
        response_is_unsafe = (
            int(response_label == "unsafe")
            if response_label in {"safe", "unsafe"}
            else int((not is_safe) and had_response)
        )

        return {
            "gen_safe_input_ids": gen_safe_input_ids,
            "gen_safe_block_ids": gen_safe_block_ids,
            "gen_safe_real_length": len(gen_safe_input_ids),
            "gen_unsafe_input_ids": gen_unsafe_input_ids,
            "gen_unsafe_block_ids": gen_unsafe_block_ids,
            "gen_unsafe_real_length": len(gen_unsafe_input_ids),
            "hazard_input_ids": hazard_input_ids,
            "hazard_block_ids": hazard_block_ids,
            "hazard_real_length": len(hazard_input_ids),
            "hazard_label_mask": hazard_label_mask,
            "hazard_eval_mask": int(is_safe or bool(violated)),
            "hazard_targets": list(violated),
            "prompt_text": prompt_text,
            "is_unsafe": int(not is_safe),
            "conv_label_mask": int(ex.get("conv_label_mask", 1)),
            "prompt_is_unsafe": prompt_is_unsafe,
            "prompt_label_mask": int(prompt_label_known),
            "response_is_unsafe": response_is_unsafe,
            "response_label_mask": int(response_label_known),
            "has_response": has_response,
            "example_index": int(ex.get("example_index", idx)),
            "eval_position": int(ex.get("eval_position", -1)),
        }


def _pad_tensor_batch(
    batch: list[dict],
    ids_key: str,
    blocks_key: str,
    length_key: str,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(item[length_key] for item in batch)
    ids_list = []
    block_list = []
    lengths = []
    for item in batch:
        ids = item[ids_key]
        blocks = item[blocks_key]
        pad_len = max_len - item[length_key]
        ids_list.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        block_list.append(torch.cat([blocks, torch.zeros(pad_len, dtype=torch.long)]))
        lengths.append(item[length_key])
    return (
        torch.stack(ids_list),
        torch.stack(block_list),
        torch.tensor(lengths, dtype=torch.long),
    )


def collate_fn(
    batch,
    pad_id: int = 0,
    mask_id: int = 126336,
    gen_only: bool = False,
    skip_cls: bool = False,
):
    """Collate materialized generative sequence views."""
    del mask_id, gen_only, skip_cls

    gen_safe_ids, gen_safe_blocks, gen_safe_lengths = _pad_tensor_batch(
        batch,
        "gen_safe_input_ids",
        "gen_safe_block_ids",
        "gen_safe_real_length",
        pad_id,
    )
    gen_unsafe_ids, gen_unsafe_blocks, gen_unsafe_lengths = _pad_tensor_batch(
        batch,
        "gen_unsafe_input_ids",
        "gen_unsafe_block_ids",
        "gen_unsafe_real_length",
        pad_id,
    )
    hazard_ids, hazard_blocks, hazard_lengths = _pad_tensor_batch(
        batch,
        "hazard_input_ids",
        "hazard_block_ids",
        "hazard_real_length",
        pad_id,
    )

    return {
        "gen_safe_input_ids": gen_safe_ids,
        "gen_safe_block_ids": gen_safe_blocks,
        "gen_safe_real_lengths": gen_safe_lengths,
        "gen_unsafe_input_ids": gen_unsafe_ids,
        "gen_unsafe_block_ids": gen_unsafe_blocks,
        "gen_unsafe_real_lengths": gen_unsafe_lengths,
        "hazard_input_ids": hazard_ids,
        "hazard_block_ids": hazard_blocks,
        "hazard_real_lengths": hazard_lengths,
        "hazard_label_mask": torch.tensor(
            [item.get("hazard_label_mask", 0) for item in batch],
            dtype=torch.float32,
        ),
        "is_unsafe": torch.tensor([item["is_unsafe"] for item in batch], dtype=torch.float32),
        "conv_label_mask": torch.tensor(
            [item.get("conv_label_mask", 1) for item in batch],
            dtype=torch.float32,
        ),
        "has_response": torch.tensor([item["has_response"] for item in batch], dtype=torch.float32),
        "example_index": torch.tensor(
            [item.get("example_index", -1) for item in batch],
            dtype=torch.long,
        ),
        "prompt_is_unsafe": torch.tensor(
            [item.get("prompt_is_unsafe", 0) for item in batch],
            dtype=torch.float32,
        ),
        "prompt_label_mask": torch.tensor(
            [item.get("prompt_label_mask", 0) for item in batch],
            dtype=torch.float32,
        ),
        "response_is_unsafe": torch.tensor(
            [item.get("response_is_unsafe", 0) for item in batch],
            dtype=torch.float32,
        ),
        "response_label_mask": torch.tensor(
            [item.get("response_label_mask", 0) for item in batch],
            dtype=torch.float32,
        ),
    }


def build_eval_loader(
    examples: list[dict],
    tokenizer,
    max_seq_length: int,
    batch_size: int,
    *,
    gen_only: bool = True,
    prompt_format: str = "aegis_taxonomy",
) -> DataLoader:
    """Build the deterministic eval loader used by public inference."""
    dataset = SafetyDataset(
        examples,
        tokenizer,
        max_seq_length=max_seq_length,
        category_augmentation=False,
        prompt_format=prompt_format,
        gen_only=gen_only,
    )
    items = [dataset[idx] for idx in range(len(dataset))]
    items.sort(
        key=lambda it: (
            int(it.get("has_response", 0)),
            max(it["gen_safe_real_length"], it["gen_unsafe_real_length"]),
        )
    )
    pad_id = tokenizer.eos_token_id or 0
    return DataLoader(
        items,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=pad_id, gen_only=gen_only),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def load_xstest_response(split: str = "response_harmfulness") -> list[dict]:
    """Load the XSTest-Response response-harmfulness split."""
    ds = load_dataset("allenai/xstest-response", split=split)
    examples = []
    for row in ds:
        user_msg = (row.get("prompt") or "").strip()
        agent_resp = row.get("response")
        label = (row.get("label") or "").strip().lower()
        if not user_msg or label not in {"harmful", "unharmful"}:
            continue
        response_label = "unsafe" if label == "harmful" else "safe"
        examples.append(
            {
                "user_message": user_msg,
                "agent_response": agent_resp.strip() if agent_resp else None,
                "is_safe": response_label == "safe",
                "violated_categories": [],
                "prompt_label": "safe",
                "response_label": response_label,
                "conv_label_mask": 1,
                "source": "xstest_response",
            }
        )
    return examples


def eval_binary_labels(examples: list[dict]) -> list[int]:
    """Return unsafe=1 labels using response labels when a response is present."""
    labels: list[int] = []
    for example in examples:
        has_response = example.get("agent_response") is not None
        if has_response:
            label = (example.get("response_label") or "").strip().lower()
        else:
            label = (example.get("prompt_label") or "").strip().lower()

        if label in {"safe", "unsafe"}:
            labels.append(1 if label == "unsafe" else 0)
        else:
            labels.append(0 if example.get("is_safe", True) else 1)
    return labels
