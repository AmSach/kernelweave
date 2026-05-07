#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python -m kernelweave.cli llm forward training/config.json "compare two artifacts and explain the differences"
