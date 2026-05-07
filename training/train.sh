#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
from kernelweave.llm import LLMConfig
cfg = LLMConfig.reasoner_frontier()
Path('training').mkdir(exist_ok=True)
cfg.save(Path('training/config.json'))
print('saved training/config.json')
PY
python -m kernelweave.cli llm architecture training/config.json
python -m kernelweave.cli llm bank training/config.json
python -m kernelweave.cli llm train-sim training/config.json 64 > training/simulated_training.json
