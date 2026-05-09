#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def find_repo() -> Path:
    env_vars = ['KERNELWEAVE_CORE_REPO', 'KERNELWEAVE_REPO']
    candidates = []
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))
    candidates.extend([
        ROOT.parent,
        ROOT.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path('/home/.z/workspaces/con_d7kcq1mzmfXZeHOQ/kernelweave'),
        Path('/home/workspace/kernelweave'),
    ])
    for candidate in Path('/home/.z/workspaces').glob('*/kernelweave'):
        candidates.append(candidate)
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / 'kernelweave').is_dir():
            return candidate
    raise SystemExit('KernelWeave repo not found. Set KERNELWEAVE_CORE_REPO to the repo root.')

REPO = find_repo()
sys.path.insert(0, str(REPO))

from kernelweave.kernel import KernelStore
from kernelweave.kernels import install_kernel_library
from kernelweave.runtime import KernelRuntime

CASES = [
    'compare two artifacts and summarize the differences',
    'write a poem about the moon',
    'summarize the architecture and cost model',
    'find all TODOs in a codebase',
    'convert JSON to YAML without losing structure',
]


def main() -> None:
    parser = argparse.ArgumentParser(description='KernelWeave benchmark harness')
    parser.add_argument('--store', default='./store')
    parser.add_argument('--ensure-samples', action='store_true')
    args = parser.parse_args()

    store = KernelStore(Path(args.store))
    if args.ensure_samples and not store.list_kernels():
        install_kernel_library(store)
    runtime = KernelRuntime(store)
    results = [runtime.run(prompt) for prompt in CASES]
    print(json.dumps({'summary': store.summary(), 'cases': results}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
