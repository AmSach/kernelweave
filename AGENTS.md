# KernelWeave workspace notes

- Keep the repo honest: if an implementation is a prototype, say prototype.
- Preserve backward compatibility for stored kernels and sample traces when possible.
- Prefer deterministic, dependency-light code.
- Keep retrieval explainable: surface match scores, evidence coverage, conflict handling, and fallback reasons.
- When improving the runtime, update tests and docs together.
- Use the store as the source of truth for kernel metadata, search index, and execution feedback.

## LLM module clarification

The `kernelweave.llm` module is a **routing and simulation layer**, not a neural network.

Critical facts:
- There are **NO model weights** in this repository
- There is **NO PyTorch, JAX, or tensor framework** — check `pyproject.toml`, it has zero dependencies
- The `KernelWeaveLLM` class routes prompts to skill kernels stored as JSON, it does NOT run inference
- The "compact frontier preset" and "reasoning frontier preset" are **architecture specifications**, not trained models
- `LLMConfig` defines what a model *would* look like if trained, not what currently exists

When documenting or explaining this code:
- Never imply that the LLM module performs actual neural inference
- Always clarify that parameter counts are estimates from specs, not measurements of trained weights
- The "LLM" naming is legacy — the module is fundamentally a routing layer
- The actual working components are: kernel store, kernel compiler, runtime, skill bank, calibration

## What IS implemented and working

- Kernel store: JSON-based persistence for kernels and traces
- Kernel compiler: trace → kernel transformation with schemas and tests
- Runtime: prompt → kernel matching and execution
- Skill bank: routing layer for stored skill kernels
- Calibration: confidence learning from examples
- CLI: working command-line interface
- Tests: regression suite for the above

These are real, working Python with zero external dependencies.
