# KernelWeave

A prototype for **kernel routing and skill compilation**: routes prompts to reusable skill kernels stored as JSON objects. This is NOT a neural network — see the "What this IS vs. IS NOT" section below.

## What this IS vs. IS NOT

### ✅ What this IS:
- A **kernel routing system** that matches prompts to stored skill kernels
- A **kernel compiler** that turns execution traces into reusable, typed kernels
- A **skill bank** that stores kernels with schemas, preconditions, postconditions, evidence requirements, rollback rules, and tests
- A **calibration layer** that learns confidence from training examples
- A **runtime engine** that executes kernel plans and falls back when evidence is insufficient
- An **architecture specification system** that estimates parameter counts for hypothetical model configurations

### ❌ What this IS NOT:
- **NOT a trained neural network** — there are no weights, no checkpoints, no learned parameters
- **NOT an inference engine** — cannot run forward passes through learned representations
- **NOT a model training system** — no PyTorch, JAX, or any tensor framework
- **NOT a language model** — does not generate text from learned distributions
- The "LLM" module naming is legacy — it's a routing layer, not a neural model

## What it does
- records solved tasks as typed execution traces
- distils traces into reusable kernels with preconditions and postconditions
- composes kernels before falling back to raw generation (routing decision, not neural generation)
- verifies kernels with regression tests
- exposes a small CLI and a local execution layer
- ships with a paper draft and demo data
- includes **architecture specifications** for hypothetical model configurations (NOT trained models)
- promotes successful traces into new kernels automatically
- calibrates kernel confidence from training examples instead of relying on a single hand-tuned score

## Core claim
LLMs should not only remember text. They should accumulate **verified competence**.
This prototype demonstrates the kernel routing layer that would plug into a trained model.

## Target architecture specification (NOT implemented)
These are **specifications for hypothetical model configurations**, not trained models:

- **Tokenizer**: byte-BPE with byte fallback and larger merge budget (spec only)
- **Transformer**: decoder-only core with GQA, RMSNorm, flash attention, MoE, and kernel routing hooks (spec only)
- **Context**: expanded long-context target with compression/retrieval strategy (spec only)
- **Training**: staged curriculum with trace distillation, self-play, and reasoning traces (plan only)
- **Inference**: lower temperature, multi-pass reasoning, verification, and kernel/skill reuse (plan only)
- **Skill bank**: internal, built into the architecture specification (spec only)

To actually train a model with these specs, you would need to:
1. Implement the transformer architecture in PyTorch/JAX
2. Train on the curriculum described in `training/`
3. Load the trained weights
4. Plug the trained model into this kernel routing layer

## Internal skill bank
KernelWeave includes an internal skill layer that stores reusable skill kernels as structured objects:
- input schema
- action plan
- evidence requirements
- rollback policy
- output contract
- tests
- confidence and drift penalty

The routing layer can route to those skills, execute them through the runtime, and promote traces into new skills.

## Prototype status
This repo is a full prototype for the **kernel routing system**, not for a trained model:
- working CLI
- working kernel store
- deterministic scoring and selection
- calibrated runtime confidence
- execution engine for kernel plans
- regression test harness
- sample kernels and traces
- publication-ready paper draft scaffold
- architecture specification configs with parameter estimation
- internal skill-bank routing working with JSON kernel objects

## Usage

```bash
python -m kernelweave.cli --help
```

## Pluggable model presets

KernelWeave can now point at other models instead of pretending to be one itself.

Preset files live in `models/` and can be copied or edited for other endpoints:
- `models/qwen0_5.json`
- `models/openai-gpt-4o-mini.json`
- `models/anthropic-claude-3-5-sonnet.json`

CLI examples:
```bash
python -m kernelweave.cli model list
python -m kernelweave.cli model show qwen0_5
python -m kernelweave.cli model run qwen0_5 "hello" --mock --mock-response "ok"
```

For real calls, set the matching API key in your environment and pick the preset ID you want.

## Design files
- `file 'docs/ARCHITECTURE.md'`
- `file 'docs/ALGORITHM.md'`
- `file 'docs/DEPLOYMENT.md'`
- `file 'paper/main.tex'`

## Paper thesis
KernelWeave turns repeatable success into executable structure, so capability compounds instead of resetting every prompt.

## Architecture specification sizes (NOT trained models)
These are **config specs** that estimate parameter counts:

- compact frontier spec: 1536 d_model, 16 layers, 12 heads, 4 KV heads, 4 experts, ~1.2B params estimated
- reasoning frontier spec: 2048 d_model, 24 layers, 16 heads, 4 KV heads, 8 experts, ~2.4B params estimated
- These are NOT trained checkpoints — just architecture specifications

## Agent planner
- stepwise task decomposition
- evidence-first planning
- built-in skill bank routing (via JSON kernels, not neural embeddings)
- curiosity questions for ambiguous prompts
- kernel promotion from successful traces

## For researchers
If you want to build on this:
1. Use the kernel routing layer as-is
2. Train a model with your preferred architecture
3. Connect the trained model to the routing layer
4. Let the kernel system compound capability on top of your base model