# Train and test

## Setup

```bash
pip install -r requirements-training.lock
pip install -e .
```

## Train

Prepare the data and train a LoRA adapter:

```bash
python -m scripts.prepare_paper_training --output data/train.jsonl

torchrun --standalone --nproc_per_node=2 -m llada_guard.train \
  --config configs/paper_ranking.json --data data/train.jsonl \
  --output runs/ranking
```

Use `configs/generative.json` for masked-reconstruction training instead of
binary ranking. Both configs use LLaDA-8B-Instruct.

Preparation samples 40,000 examples evenly across prompt/response and safe/unsafe
labels from pinned Aegis and WildGuard training splits. 

The final adapter is saved to `runs/ranking/last`, with logs and run metadata
alongside it. 

## Test

Test the trained adapter separately on XSTest-Response:

```bash
python scripts/eval_xstest_response.py \
  --checkpoint runs/ranking/last --output results/xstest_response.json
```

Run code tests with `python -m pytest -q`.
