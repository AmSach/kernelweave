# KernelWeave

A prototype for **self-compiling language models**: the model is not just queried, it is observed, distilled, tested, and upgraded with reusable skill kernels.

## What it does
- records solved tasks as typed execution traces
- distils traces into reusable kernels with preconditions and postconditions
- composes kernels before falling back to raw generation
- verifies kernels with regression tests
- exposes a small CLI and a local execution layer
- ships with a paper draft and demo data
- includes a compact frontier LLM shell with large context, MoE, and internal skill-bank routing
- promotes successful traces into new kernels automatically
- calibrates kernel confidence from training examples instead of relying on a single hand-tuned score

## Core claim
LLMs should not only remember text. They should accumulate **verified competence**.

## Full LLM architecture
- **Tokenizer**: byte-BPE with byte fallback and larger merge budget
- **Transformer**: decoder-only core with GQA, RMSNorm, flash attention, MoE, and kernel routing hooks
- **Context**: expanded long-context target with compression/retrieval strategy
- **Training**: staged curriculum with trace distillation, self-play, and reasoning traces
- **Inference**: lower temperature, multi-pass reasoning, verification, and kernel/skill reuse
- **Skill bank**: internal, built into the architecture, not an external `skills.md`

## Internal skill bank
KernelWeave includes an internal skill layer that stores reusable skill kernels as structured objects:
- input schema
- action plan
- evidence requirements
- rollback policy
- output contract
- tests
- confidence and drift penalty

The model can route to those skills during inference, can execute them through the runtime, and can also promote traces into new skills.

## Prototype status
This repo is a full prototype, not a stub:
- working CLI
- working kernel store
- deterministic scoring and selection
- calibrated runtime confidence
- execution engine for kernel plans
- regression test harness
- five sample kernels and traces
- publication-ready paper draft scaffold
- tested compact reasoning-oriented LLM config
- internal skill-bank routing integrated into the architecture

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

## Model sizes
- compact frontier preset: 1536 d_model, 16 layers, 12 heads, 4 KV heads, 4 experts, 131072 context
- reasoning frontier preset: 2048 d_model, 24 layers, 16 heads, 4 KV heads, 8 experts, 196608 context
- built-in skill bank: yes, internal to the architecture

## Agent planner
- stepwise task decomposition
- evidence-first planning
- built-in skill bank, not an external skills file
- curiosity questions for ambiguous prompts
- kernel promotion from successful traces
