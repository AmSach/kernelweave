# Training stack

This directory now contains the **runnable, pure-Python** KernelWeave training path.

## Important clarification

KernelWeave is still a **kernel architecture** project, not a new neural network implementation.
The training code here does **not** fine-tune transformer weights.
Instead, it:

- generates synthetic training samples from kernel templates
- fits lightweight calibration models to routing / verification features
- saves stable JSON artifacts
- works in constrained environments like Kaggle without HF / CUDA wheel churn

That is deliberate. The earlier Hugging Face notebook path kept failing on API drift, VRAM limits, and CUDA dependency mismatches, so the runnable path was restructured to stay faithful to the architecture without depending on fragile external training stacks.

## What this directory contains

- `TraceGenerator` — synthetic sample generation from kernel families
- `KaggleTrainer` — pure-Python training loop and artifact writer
- `hardware.py` — safe hardware detection and conservative fallback selection
- `complete.py` — data generation, calibration training, and save/load flow

## What it produces

- `data/train.jsonl`
- `data/eval.jsonl`
- `data/summary.json`
- `final_model/model.json`

## How to use it

```python
from kernelweave.training import train_kernel_native

trainer = train_kernel_native(
    output_dir="./kernel-native-model",
    n_samples=5000,
    epochs=3,
    batch_size=4,
)
```

## Design goals

- deterministic enough to debug
- safe on Kaggle T4 / P100-class machines
- no dependency on transformers, trl, peft, or bitsandbytes for the runnable path
- keep the kernel architecture visible and explainable

## Historical note

The earlier HF/Kaggle debugging path is preserved in `docs/RESTRUCTURING_HISTORY.md`, including the compatibility fixes that ended up in the workspace rescue notebook `kernelweave_kaggle_fixed.py`.
