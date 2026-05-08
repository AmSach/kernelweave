# KernelWeave Phase Plan

## Phase A — Runtime kernel system
- trace capture
- kernel compiler
- kernel store
- router
- verifier
- promotion loop

## Phase B — Learning signal from kernels
- successful traces become supervised data
- failures become negative examples
- verifier score becomes a training weight
- promotion history becomes a curriculum signal

## Phase C — Model integration
- low-sized model learns kernel-aware prompting
- then learns kernel selection
- then learns kernel composition
- then gets fine-tuned on verified traces

## Phase D — Memory becomes the primitive
- retrieve kernels instead of stuffing context
- execute kernels instead of re-deriving everything
- distill repeated patterns into weights periodically
