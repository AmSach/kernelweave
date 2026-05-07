# KernelWeave

A kernel routing system for language models with learning capabilities.

## What this is

A routing layer that:
- Matches prompts to stored skill kernels using **embedding similarity + calibrated scoring**
- Falls back to raw model generation when no kernel matches
- Records interaction outcomes and can auto-promote repeated successes into new kernels
- Works with any OpenAI/Anthropic/openai-compatible model backend

## What this is NOT

- **NOT a trained neural network** — there are no weights, no PyTorch/JAX
- **NOT AGI** — it's a routing layer with learning, not a general intelligence
- **NOT magic** — kernel matching requires semantic similarity to existing kernels

## Recent improvements (v0.2.0)

✅ **Embedding-based routing**: Uses sentence-transformers (all-MiniLM-L6-v2) for semantic similarity instead of pure lexical matching

✅ **Calibration model wired**: Logistic regression model now influences routing decisions (40% weight)

✅ **Lower threshold**: 0.50 instead of 0.68 — paraphrases now match

✅ **Real kernel execution**: Kernels can actually run via model backend instead of just returning plans

✅ **CLI learning loop**: `model run --kernel-store <path>` records feedback outcomes

## Paraphrase handling

**Before** (lexical only):
- "compare two artifacts and explain differences" → ✅ kernel (0.73)
- "compare two PDF documents and summarize their differences" → ❌ generate (0.67)

**After** (embedding + calibration):
- "compare two artifacts and explain differences" → ✅ kernel (0.756)
- "compare two PDF documents and summarize their differences" → ✅ kernel (0.711)
- "diff two configs" → ✅ kernel (0.604) — **this used to fail**
- "analyze differences between two files" → ✅ kernel (0.768)

## Quick start

```bash
# Initialize a kernel store
python -m kernelweave.cli init ./store

# Install sample kernels
python -m kernelweave.cli add-sample ./store

# Route a prompt
python -m kernelweave.cli plan ./store "compare two artifacts"

# Run with a model backend
python -m kernelweave.cli model run qwen0_5 "compare two artifacts" --kernel-store ./store
```

## Architecture

- **Kernel store**: JSON persistence for kernels, traces, and runtime feedback
- **Runtime router**: Embedding + calibration-based matching
- **Execution engine**: Can actually execute kernel plans via model backend
- **Auto-promotion**: 3+ successful interactions with a task family → new kernel

## Limitations

- Embeddings are loaded lazily; first call is slower
- Kernel matching still requires some semantic overlap with existing kernels
- Not all model backends support structured kernel execution
- The learning loop only works when `--kernel-store` is provided

## Status

Working prototype with real improvements over pure lexical routing. Tests pass. Paraphrases now match. The core routing is less brittle.

For the actual code, see:
- `kernelweave/runtime.py` — routing logic
- `kernelweave/kernel.py` — store + feedback + auto-promotion
- `kernelweave/calibration.py` — logistic regression calibration

## Paper draft

See `paper/main.tex` — but note the claims there are ambitious relative to current implementation.