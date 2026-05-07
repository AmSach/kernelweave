#!/usr/bin/env bash
set -euo pipefail
./paper/build.sh
zip -r kernelweave-release.zip README.md training kernelweave paper docs tests pyproject.toml kernelweave-neurips-paper.pdf -x '*/__pycache__/*' '*/.git/*' '*/build/*'
