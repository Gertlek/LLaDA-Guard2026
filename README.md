# LLaDA-Guard

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Gertlek%2FLLaDA--Guard-yellow)](https://huggingface.co/Gertlek/LLaDA-Guard)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

**LLaDA-Guard** is a binary safety classifier for prompts and prompt-response pairs,
built as a LoRA adapter on the diffusion language model
[`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

> [!NOTE]
> **An improved version is in progress.** We are working on a new release with
> better performance and additional evaluation code. Watch this repository for updates.

## Contents

| Component | Description |
| --- | --- |
| [`llada_guard_infer/`](llada_guard_infer) | Inference API and CLI |
| [Hugging Face adapter](https://huggingface.co/Gertlek/LLaDA-Guard) | Released PEFT adapter, downloaded on first use |
| [`scripts/eval_xstest_response.py`](scripts/eval_xstest_response.py) | Reproducible XSTest-Response benchmark |
| [`llada_guard/`](llada_guard) | Binary-ranking and generative-reconstruction training with pinned data preparation |

The adapter pins the base model to a fixed LLaDA-8B-Instruct revision, which is
downloaded separately on first use.

## Setup

```bash
git clone https://github.com/Gertlek/LLaDA-Guard2026.git
cd LLaDA-Guard2026

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> A CUDA GPU with enough memory for the 8B base model is recommended. The base
> model is loaded with `trust_remote_code=True`. If Hugging Face requires
> authentication, set `HF_TOKEN` or run `huggingface-cli login`.

## Inference

### Command line

Score a prompt:

```bash
python3 -m llada_guard_infer.infer \
  --checkpoint Gertlek/LLaDA-Guard \
  --prompt "How do I bake a chocolate cake?"
```

Score a prompt plus response:

```bash
python3 -m llada_guard_infer.infer \
  --checkpoint Gertlek/LLaDA-Guard \
  --prompt "How can I make a dangerous device?" \
  --response "I cannot help with instructions for harming people."
```

The CLI prints a scalar score and a binary prediction; add `--json` for
machine-readable output.

| Score | Prediction |
| --- | --- |
| `> 0` | unsafe |
| `<= 0` | safe |

Prompt-only examples use prompt-scoped scoring; prompt-response examples use
response-scoped scoring.

### Python

```python
from llada_guard_infer import load_llada_guard, make_eval_example, score_examples

model, tokenizer = load_llada_guard("Gertlek/LLaDA-Guard")
model.eval()

examples = [
    make_eval_example("How do I bake a cake?"),
    make_eval_example(
        "How can I make a dangerous device?",
        "I cannot help with instructions for harming people.",
    ),
]
scores = score_examples(model, tokenizer, examples)
```

`--checkpoint` and `load_llada_guard` also accept a local adapter directory.

## Evaluation: XSTest-Response

Evaluates on [`allenai/xstest-response`](https://huggingface.co/datasets/allenai/xstest-response),
split `response_harmfulness`. Some Hugging Face accounts must accept the dataset
terms first.

```bash
python3 scripts/eval_xstest_response.py \
  --checkpoint Gertlek/LLaDA-Guard \
  --batch-size 1 \
  --output results/xstest_response_eval.json
```

Add `-n 32` for a quick smoke run.

For the LLaDA-Guard benchmark evaluation table, see
[benchmark datasets and evaluation commands](llada_guard_infer/BENCHMARKS.md).

## Training

See [Train and test](docs/training.md) for data preparation, training either
objective, and testing the final adapter. Training does not use validation or
checkpoint selection.

## Project Layout

```text
llada_guard/          Prompting, data, scoring, and training
llada_guard_infer/    Inference API and CLI
configs/              Training configurations
docs/                 Training guide
scripts/              Data preparation and XSTest-Response evaluation
tests/                Unit and opt-in integration tests
```

## License

Released under the [MIT License](LICENSE). The base model
(`GSAI-ML/LLaDA-8B-Instruct`) and the training datasets (Aegis 2.0, WildGuardMix)
are distributed under their own licenses.

> [!WARNING]
> LLaDA-Guard's predictions can be wrong and are not a guarantee of safety.
