# Architecture

KernelWeave has four layers:

1. **Trace capture** — store solved behaviours as structured events.
2. **Kernel compilation** — distil a trace into a reusable, typed kernel.
3. **Kernel runtime** — select kernels for future prompts.
4. **Regression gate** — reject kernels that fail evidence or output tests.

The key idea is that competence becomes executable, not merely remembered.

```mermaid
flowchart LR
  P[Prompt] --> R[Kernel runtime]
  R -->|match| K[Kernel]
  R -->|no match| G[Raw generation]
  G --> T[Trace]
  T --> C[Kernel compiler]
  C --> S[Kernel store]
  S --> R
```
