# Timeline / History

## Before restructuring
KernelWeave went through a long Kaggle training detour trying to force a Hugging Face fine-tune path onto a constrained runtime.
That path kept breaking on API drift, VRAM limits, and CUDA dependency mismatches.

## Decision
The runnable path was restructured into a pure-Python calibration/tracing prototype while preserving the kernel architecture.

## Lesson
The architecture is the product. The training stack is just one possible implementation.