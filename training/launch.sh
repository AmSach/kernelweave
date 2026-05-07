#!/usr/bin/env bash
set -euo pipefail
python -m kernelweave.cli llm architecture training/config.json
python -m kernelweave.cli llm bank training/config.json
python -m kernelweave.cli llm train-sim training/config.json 64 > training/simulated_training.json
python -m kernelweave.cli llm forward training/config.json "design a safe tool-using agent for long context reasoning"
