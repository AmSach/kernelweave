# Training Plan

## Stage 1 — Kernel-aware prompting
Train a low-sized model to output with kernel hints and kernel-aware formatting.

## Stage 2 — Kernel selection
Train the model to choose the right kernel for a prompt family.

## Stage 3 — Kernel composition
Train the model to combine kernels when a single kernel is insufficient.

## Stage 4 — Verified trace fine-tuning
Train on traces that passed verification with weights derived from:
- verifier score
- evidence completeness
- retry count
- rollback frequency

## Stage 5 — Periodic distillation
Distill frequently reused traces into the model while retaining the kernel store as external memory.
