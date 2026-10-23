# Benchmark evaluation

## Datasets and sampling

| Dataset | Task | Examples | Safe / unsafe | How chosen |
| --- | --- | ---: | ---: | --- |
| ToxicChat (human-annotated test split) | Prompt | 2,853 | 2,491 / 362 | All safe/unsafe samples. |
| OpenAI Moderation | Prompt | 1,680 | 1,158 / 522 | All safe/unsafe samples. |
| XSTest | Prompt | 450 | 250 / 200 | All safe/unsafe samples. |
| SafeRLHF | Response | 2,000 | 1,000 / 1,000 | Sample 1,000 safe and 1,000 unsafe responses. |
| BeaverTails | Response | 3,021 | 1,288 / 1,733 | All safe/unsafe samples. |
| XSTest-response | Response | 446 | 368 / 78 | All safe/unsafe samples. |
| PHTest (harmless subset) | Prompt | 2,077 | 2,077 / 0 | All safe samples. |

Exact rules are under [benchmark_manifest.json](benchmark_manifest.json).

## Prepare and verify

After installing the package, run from the repository root:

```bash
python -m llada_guard_infer.benchmark prepare \
  --output data/evaluation \
  --phtest-jsonl /path/to/phtest_harmless.jsonl
```

Dataset terms or Hugging Face authentication may be required.

## Compute the evaluation table

```bash
python -m llada_guard_infer.benchmark evaluation_table \
  --cohorts data/evaluation \
  --output results/evaluation \
  --batch-size 4 --device cuda
```

The `evaluation_table` function verifies the cohorts and released adapter, then
uses this installed package's inference API. It uses three masks at `t=0.15`

The command prints the measured table and saves `evaluation_table.md`,
`evaluation_table.json`, per-example predictions.
It reports AUPRC/F1 for mixed-label datasets and safe accuracy for PHTest.

