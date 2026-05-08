# Cost Model

## Hypothesis
Repeated tasks cost less when the system reuses verified kernels instead of regenerating from scratch.

## Cost drivers
- prompt tokens
- completion tokens
- retrieval overhead
- verification overhead
- model size
- fallback frequency

## Savings model
For repeated task families:
- first run pays the full generation cost
- later runs pay mostly routing + verification
- repeated work should asymptotically approach near-zero model calls for covered cases
