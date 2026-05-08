# KernelWeave

A kernel routing and verification system for language models.

## What this actually is

**A caching layer for verified reasoning patterns.**

When a language model solves a task successfully, KernelWeave:
- Stores the reasoning pattern as a typed kernel
- Verifies future outputs against postconditions before caching
- Routes similar prompts to cached kernels
- Accumulates verified competence over time

## What works now

- **Semantic routing** — Embedding-based similarity + calibration scoring
- **Postcondition verification** — Checks outputs against kernel constraints
- **Feedback accumulation** — Records success/failure for each kernel
- **Auto-promotion** — High-confidence repeated successes become candidate kernels
- **Model-agnostic** — Works with any OpenAI/Anthropic/openai-compatible backend
- **Runnable training bundle** — a pure-Python Kaggle-safe calibration/tracing path that keeps the kernel architecture executable without HF/CUDA wheel drama
- **Phase C/D handoff** — mirrored under `phasecd/` in this repo for Kaggle training, benchmark, and export work

## What doesn't work yet

### Real self-compilation
Kernels are extracted from prompts + response text, not from observed reasoning traces. The model's actual chain-of-thought, tool calls, and intermediate states aren't captured. That requires structured generation or a separate observation layer.

### Kernel composition
If you have a "structured comparison" kernel and an "evidence extraction" kernel, there's no mechanism to combine them. Composition over a kernel algebra is an open research problem.

### Constrained generation
The kernel informs the model via system prompt, but doesn't constrain the output space during generation. The model can still output anything — verification happens after the fact. A frontier system would use kernels as structured decoding constraints during token generation.

## Real contribution

**"Postcondition verification as a routing signal."**

The novel part isn't the routing (that's retrieval) or the kernels (that's program synthesis). It's using verification against formal postconditions to decide whether to trust cached reasoning for future prompts.

Testable claim: For repeated task families, routing + verification beats vanilla RAG on output quality and cost.

## What’s in `phasecd/`

- phase plan
- dataset spec
- benchmark harness
- Kaggle training entrypoint
- generated phase data
- benchmark output

## Status

Working prototype. Not frontier. Not a trained model. A useful piece of infrastructure for LLM-based systems with repeated tasks.

For the full pre-restructure debugging trail, see `docs/RESTRUCTURING_HISTORY.md`.
