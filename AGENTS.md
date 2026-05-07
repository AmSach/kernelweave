# KernelWeave workspace notes

- Keep the repo honest: if an implementation is a prototype, say prototype.
- Preserve backward compatibility for stored kernels and sample traces when possible.
- Prefer deterministic, dependency-light code.
- Keep retrieval explainable: surface match scores, evidence coverage, conflict handling, and fallback reasons.
- When improving the runtime, update tests and docs together.
- Use the store as the source of truth for kernel metadata, search index, and execution feedback.
