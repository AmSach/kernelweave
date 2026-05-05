# Kernel compilation algorithm

Given a trace of a successful task, KernelWeave compiles a kernel with:

- input schema
- output schema
- preconditions
- postconditions
- rollback rules
- evidence requirements
- executable steps

## Compilation objective
Let a trace be `τ = (e_1, ..., e_n)` and a candidate kernel be `k`.
We define a score:

```latex
J(k; \tau) = \lambda_1 E(k) + \lambda_2 R(k) + \lambda_3 C(k) - \lambda_4 F(k)
```

where:
- `E(k)` = evidence completeness
- `R(k)` = runtime reuse value
- `C(k)` = compression of trace into executable structure
- `F(k)` = failure risk

## Operational rule
A kernel may enter the active store only if it passes its tests.
If a test fails, the kernel is demoted or rolled back.
