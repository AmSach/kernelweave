# KernelWeave

A prototype for **self-compiling language models**: the model is not just queried, it is observed, distilled, tested, and upgraded with reusable skill kernels.

## What it does
- records solved tasks as typed execution traces
- distils traces into reusable kernels with preconditions and postconditions
- composes kernels before falling back to raw generation
- verifies kernels with regression tests
- exposes a small CLI and a simple local API
- ships with a paper draft and demo data

## Core claim
LLMs should not only remember text. They should accumulate **verified competence**.

A successful behaviour is converted into a kernel:
- input schema
- action plan
- evidence requirements
- rollback policy
- output contract
- tests

The next time the same task family appears, the system tries the kernel first.

## Prototype status
This repo is a full prototype, not a stub:
- working CLI
- working kernel store
- deterministic scoring and selection
- regression test harness
- sample kernels and traces
- publication-ready paper draft scaffold

## Usage

```bash
python -m kernelweave.cli --help
```

## Design files
- `file 'docs/ARCHITECTURE.md'`
- `file 'docs/ALGORITHM.md'`
- `file 'docs/DEPLOYMENT.md'`
- `file 'paper/main.tex'`

## Paper thesis
KernelWeave turns repeatable success into executable structure, so capability compounds instead of resetting every prompt.
