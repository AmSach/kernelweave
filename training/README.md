# Training stack

This directory is the practical training playbook for KernelWeave.

## Important clarification

The "training" described here is for a **hypothetical model** that would use the KernelWeave kernel routing layer. There are:
- **NO trained weights** in this repository
- **NO PyTorch/JAX training scripts** that actually run
- **NO datasets** — only specifications for what datasets would be needed

The `Trainer` class in `kernelweave.llm.train` simulates training metrics for planning purposes. It does not actually train a neural network.

## What this directory contains

- **Architecture specifications**: configs defining what a model would look like
- **Training plans**: curriculum descriptions and hyperparameter settings
- **Simulation scripts**: synthetic training runs for metric estimation
- **Data plans**: specifications for what datasets would be needed

## To actually train a model

You would need to:

1. Implement the transformer architecture in PyTorch/JAX using the specs in `kernelweave/llm/config.py`
2. Acquire or build the datasets described in `training/data_plan.md`
3. Write actual training scripts (not included in this repo)
4. Train following the curriculum in `training/curriculum.md`
5. Export checkpoints and connect to the kernel routing layer

## Hypothetical training flow (for planning)

1. Build your dataset directories as listed in `kernelweave/llm/manifest.py`.
2. Create a config file from the specification:

```bash
python - <<'PY'
from kernelweave.llm import LLMConfig
from pathlib import Path
cfg = LLMConfig.reasoner_frontier_spec()
cfg.save(Path('training/config.json'))
print('wrote architecture specification to training/config.json')
PY
```

3. Validate the manifest and training plan (simulation only):

```bash
python -m kernelweave.cli llm architecture training/config.json
python -m kernelweave.cli llm bank training/config.json
```

4. Run a synthetic dry-run simulation:

```bash
python -m kernelweave.cli llm train-sim training/config.json 64
```

5. Only after that, implement actual training in PyTorch/JAX and point your trainer at the same config.

## Practical order (for actual training, not implemented here)
- data quality and dedup
- mix foundation text + code + math + reasoning
- add trace distillation
- add tool traces
- increase context length
- promote skill kernels
- hard-evaluate before scaling further


## Production-grade startup (planning notes)
- create separate folders for cleaned data, eval data, and trace data
- log token counts for every dataset before a run
- snapshot the config before each stage
- promote only checkpoints that beat the eval gate
- keep kernel-bank exports alongside checkpoints
- never let a stage advance if eval regresses

## Minimum recommended scripts (not included)
- `training/train.sh` — would run actual PyTorch/JAX training
- `training/evaluate.sh` — would evaluate trained checkpoints
- `training/build_release.sh` — would package trained model
- `training/config.json` — architecture specification

## What IS included
- Architecture specification configs
- Training simulation (`Trainer.step()` for synthetic metrics)
- Curriculum descriptions
- Dataset source specifications
- Evaluation plan

These are planning artifacts, not runnable training code.

## Public dataset training tutorial
- See `training/quickstart_public_datasets.md` for the practical version.
