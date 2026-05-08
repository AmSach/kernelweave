# Dataset Specification

## Dataset families
1. prompt_kernel_selection
2. prompt_kernel_prompting
3. trace_verification_pairs
4. trace_failure_pairs
5. kernel_composition_pairs
6. retrieval_memory_pairs
7. distillation_examples

## Row schema
Each row should be one JSON object with:
- `id`
- `phase`
- `task_family`
- `prompt`
- `kernel_id`
- `kernel_name`
- `kernel_steps`
- `kernel_preconditions`
- `kernel_postconditions`
- `kernel_evidence_requirements`
- `kernel_rollback`
- `response`
- `verified`
- `verification_score`
- `confidence`
- `weight`
- `target`
- `notes`

## Targets
- `prompting`
- `selection`
- `composition`
- `trace_finetune`
- `memory_retrieval`
- `distillation`
