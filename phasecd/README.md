# KernelWeave Phase C/D Handoff

This folder is the Kaggle handoff for the Phase C/D version of KernelWeave.

## What it contains
- `docs/` — phase plan, dataset spec, benchmark plan, training plan
- `scripts/` — dataset generation, benchmark harness, and training entrypoint
- `data/` — generated JSONL training data
- `store/` — kernel store and benchmark traces
- `benchmark.json` — latest benchmark output

## What to upload to Kaggle
Upload the entire `KernelWeave_PhaseCD/` folder as a zip or add it as a Kaggle dataset.

## What Kaggle needs
- `data/train.jsonl`
- `data/eval.jsonl`
- `store/` or a fresh store created from the core repo
- `scripts/train_phasec.py`
- `scripts/generate_dataset.py`
- `scripts/benchmark.py`
- `scripts/kaggle_launch.sh`
- `scripts/kaggle_train_fixed.py`

## What the model trains on
- kernel-aware prompting examples
- kernel selection examples
- kernel composition examples
- verified trace examples
- failure / rollback examples

## Important
This is a prototype handoff designed to test whether a small kernel-native architectural shift can outperform same-size baselines on repeated tool-use, agentic work, and reasoning.
