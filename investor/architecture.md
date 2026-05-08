# Architecture

KernelWeave is a kernel-native execution layer for LLMs.

## Core loop
1. Prompt enters the system.
2. Router checks whether an existing kernel can handle it.
3. If yes, the kernel executes or composes.
4. If no, the base model generates a fresh solution.
5. Verifier checks the output against postconditions.
6. Successful traces compile into new kernels.
7. Future prompts become cheaper and more reliable.

## Why it matters
Most LLM apps throw away successful work after every completion. KernelWeave keeps it.

## Architecture blocks
- Trace capture
- Kernel compiler
- Kernel store
- Router
- Verifier
- Promotion loop
- Optional model integration

## Product claim
Competence becomes executable, not just remembered.