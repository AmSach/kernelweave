# KernelWeave

A self-compiling kernel routing system for language models.

## What this is

A routing layer that:
- Matches prompts to stored skill kernels using **embeddings + calibration**
- **Executes kernels** against a real backend and **verifies outputs** against postconditions
- **Extracts kernels automatically** from successful model responses
- Records feedback and **auto-promotes** high-confidence patterns into new kernels
- Plugs into any OpenAI/Anthropic/open-source model via presets

## The closed loop (now real)

1. **Route** — Match prompt to kernel using embeddings + calibrated confidence
2. **Execute** — If kernel matches, run it through the model with kernel-aware prompting
3. **Verify** — Check output against postconditions and evidence requirements
4. **Record** — Log feedback with success/failure score
5. **Learn** — After 3+ successful runs on same task family, **auto-promote** a new kernel
6. **Reuse** — Next time, the new kernel is available for routing

## What's working

```bash
# Initialize a kernel store
python -m kernelweave.cli init ./store

# Add sample kernels
python -m kernelweave.cli add-sample ./store

# Run with model backend + auto-compile
python -m kernelweave.cli model run qwen0_5 "compare two artifacts" \\
  --kernel-store ./store \\
  --auto-compile

# The kernel candidate gets created automatically
python -m kernelweave.cli list ./store
```

## Execution + verification

When a kernel matches:
1. Backend executes the kernel plan with structured system prompt
2. Output is verified against postconditions (keyword matching)
3. Evidence requirements are checked
4. Result is scored and recorded as feedback
5. High-confidence repeated successes trigger auto-promotion

## Model-generated kernels

The `--auto-compile` flag:
- Captures the model's reasoning process from agent plan
- Extracts it as a trace (plan → steps → evidence → verification → decision)
- Compiles into a kernel candidate (status: "candidate")
- Stores it for future routing

This is the **self-compiling** part: the model's successful problem-solving becomes reusable structure.

## Components

- `kernelweave/kernel.py` — Kernel store + feedback recording + auto-promotion
- `kernelweave/runtime.py` — Routing, execution, verification
- `kernelweave/calibration.py` — Logistic regression confidence model
- `kernelweave/compiler.py` — Trace → kernel compilation
- `kernelweave/llm/model.py` — Wrapper that closes the loop
- `models/` — Presets for Qwen, OpenAI, Anthropic

## What this still isn't

- NOT a neural network
- NOT training weights
- NOT replacing the base model

It's a **compounding capability layer** on top of any model. The model does the work; KernelWeave remembers and reuses it.