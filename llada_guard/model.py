from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

MASK_ID = 126336


def _build_checkpoint_load_kwargs(dtype: torch.dtype, device: str) -> dict:
    """Build memory-aware kwargs for loading the base model."""
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if isinstance(device, str) and device.startswith("cuda"):
        kwargs["device_map"] = {"": device}
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def _patch_accelerate_clear_device_cache() -> None:
    """Keep PEFT compatible with Accelerate versions lacking this helper."""
    try:
        import accelerate.utils.memory as accel_memory
    except Exception:
        return

    if hasattr(accel_memory, "clear_device_cache"):
        return

    def clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accel_memory.clear_device_cache = clear_device_cache


def load_checkpoint(
    base_model_name: str,
    checkpoint_path: str,
    torch_dtype: str = "bfloat16",
    device: str = "cuda",
    is_trainable: bool = False,
):
    """Load the released PEFT adapter on top of the base LLaDA model."""
    _patch_accelerate_clear_device_cache()

    from peft import PeftConfig, PeftModel

    dtype = getattr(torch, torch_dtype)
    adapter_config = PeftConfig.from_pretrained(checkpoint_path)
    revision_kwargs = (
        {"revision": adapter_config.revision} if adapter_config.revision else {}
    )
    load_kwargs = _build_checkpoint_load_kwargs(dtype, device)
    load_kwargs.update(revision_kwargs)
    use_device_map = "device_map" in load_kwargs

    base_model = AutoModel.from_pretrained(base_model_name, **load_kwargs)
    if not use_device_map:
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=is_trainable,
    )
    if not use_device_map:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, **revision_kwargs
    )
    return model, tokenizer
