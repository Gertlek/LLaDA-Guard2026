---
base_model: GSAI-ML/LLaDA-8B-Instruct
library_name: peft
license: mit
tags:
- safety
- moderation
- lora
---

# LLaDA-Guard

Binary safety classification for prompts and responses. This repository contains
the released LoRA adapter and inference code; the LLaDA-8B-Instruct base model
downloads separately. A CUDA GPU is recommended. Loading the base model uses
`trust_remote_code=True`.

## Setup

```bash
pip install huggingface_hub
hf download Gertlek/LLaDA-Guard --local-dir LLaDA-Guard
cd LLaDA-Guard
pip install -e .
```

## Inference

Score a prompt:

```bash
python -m llada_guard_infer.infer --checkpoint . \
  --prompt "How do I bake a cake?" --json
```

To score a response, add `--response`:

```bash
python -m llada_guard_infer.infer --checkpoint . \
  --prompt "How do I bake a cake?" \
  --response "Mix the ingredients and bake until done." --json
```

Scores above zero indicate unsafe; otherwise safe. Predictions can be wrong.
