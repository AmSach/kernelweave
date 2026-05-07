# Data plan

## Dataset buckets
- `data/raw/general_text/`
- `data/raw/code/`
- `data/raw/math/`
- `data/raw/reasoning/`
- `data/raw/dialogue/`
- `data/raw/tool_traces/`
- `data/raw/skills/`

## Cleaning rules
- deduplicate aggressively
- remove obviously low quality spam
- keep provenance and licence metadata
- separate reasoning traces from normal dialogue
- separate skill kernels from raw traces

## Token counting
Track:
- raw tokens
- cleaned tokens
- unique tokens
- discarded tokens
- per-source contribution

## Promotion rule
A trace can become a skill kernel only if it passes tests and improves eval.
