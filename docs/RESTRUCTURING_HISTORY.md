# Restructuring history

This document records the Kaggle / training detour that happened **before** the current pure-Python restructuring.

KernelWeave is fundamentally a **kernel architecture** project: the point is to make reusable, verifiable reasoning kernels the primitive, so future LLMs can be built around that architecture instead of around ad hoc prompt chains. The training work was originally aimed at a Hugging Face fine-tuning flow on Kaggle, but that path kept breaking for reasons that were mostly versioning and runtime compatibility issues, not core KernelWeave logic.

## What the earlier rescue notebook did

The workspace file `kernelweave_kaggle_fixed.py` captured the first serious repair pass. It documented and handled the following Kaggle issues:

- `SFTConfig` renamed `max_seq_length` → `max_length`
- `SFTTrainer` renamed `tokenizer` → `processing_class`
- `packing` moved into `SFTConfig`
- `dataset_text_field` moved into `SFTConfig`
- bitsandbytes 4-bit quantization was unreliable on Kaggle T4
- Qwen chat formatting needed proper special-token handling
- `gradient_checkpointing` conflicted with `use_cache`
- Qwen 7B was too large for Kaggle T4, so the notebook explicitly fell back to 1.5B
- tiny eval sets could crash training, so eval was made optional
- gated Hugging Face models needed a clearer token warning

That notebook was useful, but it still depended on the HF stack, which kept breaking in different environments.

## Debugging timeline before the restructuring

| Problem | What happened | What was learned | Final outcome |
| --- | --- | --- | --- |
| Invalid model-load kwargs | `max_length` leaked into `from_pretrained()` | generation/training args do not belong in model loading | removed the arg from model init |
| Template mismatch | `KeyError: aspect_1` during sample generation | the synthetic template pool was missing variables | expanded the variable pool and formatting paths |
| Trainer API mismatch | `setup_model(quantization=...)` crashed | the trainer signature and the helper were out of sync | removed the unsupported kwarg path |
| `trl` API drift | `SFTConfig` rejected fields like `max_seq_length` | Kaggle’s installed `trl` version was newer/different | attempted compatibility shims |
| VRAM pressure | Qwen 7B OOMed on T4 | 7B was too large for Kaggle’s free tier | added 1.5B fallback |
| dtype mismatch | `BFloat16` unscale path failed | T4/P100 + bf16 was not viable | forced fp16-only fallback |
| CUDA / bitsandbytes breakage | `libnvJitLink.so.13` missing | CUDA toolchain and bitsandbytes were incompatible | stopped relying on bnb for the working path |
| optional backend imports | `AutoModelForCausalLM` / `BloomPreTrainedModel` import failures | transformers version and optional dependencies were too brittle | removed that dependency chain from the runnable path |
| install / push noise | pip resolver warnings and tracked bytecode polluted the repo | environment churn was hiding real issues | cleaned the repo state and focused on deterministic code |

## Why the restructuring happened

After the debugging loop, the pattern was obvious: the task was not “get one more transformer flag right.” The task was to make KernelWeave **work as KernelWeave**.

So the training stack was restructured into a **pure-Python** pipeline that:

- generates synthetic training samples from kernel templates
- trains lightweight calibration models instead of neural weights
- saves JSON artifacts that are stable and inspectable
- runs without HF, bitsandbytes, CUDA extensions, or model downloads
- keeps the kernel architecture intact while removing the fragile notebook-specific runtime baggage

## Current state after restructuring

The current training bundle is intentionally boring and reliable:

- `TraceGenerator` creates synthetic samples
- `KaggleTrainer` generates data, trains calibration models, and saves artifacts
- `hardware.py` only selects safe settings and base-model fallbacks; it no longer drives a transformer fine-tune path
- the package can be installed and run in constrained environments without GPU-specific wheel drama

## What this means conceptually

KernelWeave is not “an LLM that magically trained itself.”
It is a **kernel routing and verification system** whose architecture is meant to make future models more structured:

1. solve a task
2. verify it
3. turn the successful trace into a reusable kernel
4. route future prompts through that kernel
5. accumulate competence over time

That is the architectural idea worth preserving.

## Reference files

- `README.md` — current project summary
- `training/README.md` — current training-stack documentation
- `CHANGELOG.md` — version-level summary of the restructuring
