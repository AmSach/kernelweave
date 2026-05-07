# Training stack

This directory is the practical training playbook for KernelWeave.

## Start training
1. Build your dataset directories as listed in `kernelweave/llm/manifest.py`.
2. Create a config file from the preset:

```bash
python - <<'PY'
from kernelweave.llm import LLMConfig
from pathlib import Path
cfg = LLMConfig.reasoner_frontier()
cfg.save(Path('training/config.json'))
print('wrote training/config.json')
PY
```

3. Validate the manifest and training plan:

```bash
python -m kernelweave.cli llm architecture training/config.json
python -m kernelweave.cli llm bank training/config.json
```

4. Run a synthetic dry-run first:

```bash
python -m kernelweave.cli llm train-sim training/config.json 64
```

5. Only after that, point your actual trainer at the same config.

## Practical order
- data quality and dedup
- mix foundation text + code + math + reasoning
- add trace distillation
- add tool traces
- increase context length
- promote skill kernels
- hard-evaluate before scaling further


## Production-grade startup
- create separate folders for cleaned data, eval data, and trace data
- log token counts for every dataset before a run
- snapshot the config before each stage
- promote only checkpoints that beat the eval gate
- keep kernel-bank exports alongside checkpoints
- never let a stage advance if eval regresses

## Minimum recommended scripts
- `training/train.sh`
- `training/evaluate.sh`
- `training/build_release.sh`
- `training/config.json`


## Public dataset training tutorial
- See `training/quickstart_public_datasets.md` for the practical version.
