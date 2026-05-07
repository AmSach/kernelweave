# Changelog

## 0.1.1
- **Honesty fixes**: Clarified that this is NOT a neural network implementation
  - Added comprehensive docstrings explaining no weights, no PyTorch/JAX, no actual inference
  - Renamed "preset" to "spec" throughout to clarify these are architecture specifications
  - Updated README with explicit "What this IS vs. IS NOT" section
  - Updated AGENTS.md with LLM module clarification
  - Updated training/README.md to clarify simulation vs actual training
- Updated `KernelWeaveLLM` class docstring to clarify it's a routing layer, not a neural model
- Added `compact_frontier_spec()` and `reasoner_frontier_spec()` methods, deprecated old names
- Updated all documentation to reflect the actual state of the codebase

## 0.1.0
- Initial full prototype for KernelWeave
- Kernel compiler, store, runtime, CLI, docs, and paper draft
