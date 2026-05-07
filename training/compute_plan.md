# Compute plan

## Shape
- model: KernelWeave-Reasoner
- target params: about 1.7B active / MoE-based
- context: 196,608 tokens
- precision: bf16
- checkpointing: on
- packed sequences: on

## Compute strategy
1. Start on a small cluster for dry runs.
2. Validate data and eval pipeline before scaling.
3. Use activation checkpointing and gradient accumulation.
4. Prefer packed sequences and long-context curriculum.
5. Keep the skill bank and router active during training.
6. Save frequent checkpoints; promote only after eval gates pass.

## Cost control
- dedup data first
- use trace distillation to avoid waste
- keep context long only where the task needs it
- never train on junk just because it is available
