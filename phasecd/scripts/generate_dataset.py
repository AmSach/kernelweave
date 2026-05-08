#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def find_repo() -> Path:
    env = os.environ.get('KERNELWEAVE_REPO')
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path('/home/.z/workspaces/con_d7kcq1mzmfXZeHOQ/kernelweave'))
    candidates.append(Path('/home/workspace/kernelweave'))
    for candidate in Path('/home/.z/workspaces').glob('*/kernelweave'):
        candidates.append(candidate)
    for candidate in candidates:
        if (candidate / 'kernelweave').is_dir():
            return candidate
    raise SystemExit('KernelWeave repo not found. Set KERNELWEAVE_REPO to the repo root.')


REPO = find_repo()
sys.path.insert(0, str(REPO))

from kernelweave.kernel import KernelStore, TraceEvent
from kernelweave.kernels import install_kernel_library
from kernelweave.compiler import compile_trace_to_kernel
from kernelweave.compose import compose_sequence
from kernelweave.verifier import HeuristicVerifier

DEFAULT_OUT = ROOT / 'data'
DEFAULT_STORE = ROOT / 'store'
VERIFIER = HeuristicVerifier()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def kernel_prompt(kernel) -> str:
    pre = '; '.join(kernel.preconditions[:3])
    post = '; '.join(kernel.postconditions[:3])
    return f"Use the {kernel.task_family} kernel. Preconditions: {pre}. Postconditions: {post}."


def make_rows(store: KernelStore) -> list[dict[str, Any]]:
    kernels = [store.get_kernel(item['kernel_id']) for item in store.list_kernels()]
    rows: list[dict[str, Any]] = []

    for kernel in kernels:
        prompt = f"{kernel.task_family}: {kernel.description}"
        response = f"Kernel {kernel.kernel_id} handles {kernel.task_family}."
        verification = VERIFIER.verify(response, kernel.postconditions, kernel.evidence_requirements)
        rows.append({
            'id': f'{kernel.kernel_id}-prompting',
            'phase': 'C',
            'task_family': kernel.task_family,
            'prompt': prompt,
            'kernel_id': kernel.kernel_id,
            'kernel_name': kernel.name,
            'kernel_steps': kernel.steps,
            'kernel_preconditions': kernel.preconditions,
            'kernel_postconditions': kernel.postconditions,
            'kernel_evidence_requirements': kernel.evidence_requirements,
            'kernel_rollback': kernel.rollback,
            'response': kernel_prompt(kernel),
            'verified': verification.passed,
            'verification_score': verification.score,
            'confidence': kernel.status.confidence,
            'weight': round(1.0 + verification.score + kernel.status.confidence, 4),
            'target': 'prompting',
            'notes': 'kernel-aware prompting sample',
        })

        rows.append({
            'id': f'{kernel.kernel_id}-selection',
            'phase': 'C',
            'task_family': kernel.task_family,
            'prompt': f"Choose the best kernel for: {prompt}",
            'candidates': [item['kernel_id'] for item in store.list_kernels()[:5]],
            'kernel_id': kernel.kernel_id,
            'kernel_name': kernel.name,
            'response': kernel.kernel_id,
            'verified': True,
            'verification_score': kernel.status.confidence,
            'confidence': kernel.status.confidence,
            'weight': round(1.0 + kernel.status.confidence, 4),
            'target': 'selection',
            'notes': 'kernel selection classification sample',
        })

        rows.append({
            'id': f'{kernel.kernel_id}-trace',
            'phase': 'B',
            'task_family': kernel.task_family,
            'prompt': prompt,
            'kernel_id': kernel.kernel_id,
            'kernel_name': kernel.name,
            'kernel_steps': kernel.steps,
            'kernel_preconditions': kernel.preconditions,
            'kernel_postconditions': kernel.postconditions,
            'kernel_evidence_requirements': kernel.evidence_requirements,
            'kernel_rollback': kernel.rollback,
            'response': response,
            'verified': verification.passed,
            'verification_score': verification.score,
            'confidence': kernel.status.confidence,
            'weight': round(1.0 + verification.score + kernel.status.confidence, 4),
            'target': 'trace_finetune',
            'notes': 'verified trace supervision sample',
        })

        rows.append({
            'id': f'{kernel.kernel_id}-memory',
            'phase': 'D',
            'task_family': kernel.task_family,
            'prompt': f"Retrieve and execute memory for {prompt}",
            'kernel_id': kernel.kernel_id,
            'kernel_name': kernel.name,
            'response': f"Retrieve kernel {kernel.kernel_id} and execute its steps.",
            'verified': True,
            'verification_score': kernel.status.confidence,
            'confidence': kernel.status.confidence,
            'weight': round(1.25 + kernel.status.confidence, 4),
            'target': 'memory_retrieval',
            'notes': 'memory primitive sample',
        })

    for kernel_a, kernel_b in list(combinations(kernels[:8], 2))[:12]:
        composite = compose_sequence(kernel_a, kernel_b)
        rows.append({
            'id': f'{kernel_a.kernel_id}-{kernel_b.kernel_id}-compose',
            'phase': 'C',
            'task_family': composite.kernel.task_family,
            'prompt': f"Compose {kernel_a.task_family} then {kernel_b.task_family}.",
            'kernel_id': composite.kernel.kernel_id,
            'kernel_name': composite.kernel.name,
            'kernel_steps': composite.kernel.steps,
            'kernel_preconditions': composite.kernel.preconditions,
            'kernel_postconditions': composite.kernel.postconditions,
            'kernel_evidence_requirements': composite.kernel.evidence_requirements,
            'kernel_rollback': composite.kernel.rollback,
            'response': f"Compose {kernel_a.kernel_id} then {kernel_b.kernel_id}.",
            'verified': composite.is_valid(),
            'verification_score': composite.kernel.status.confidence,
            'confidence': composite.kernel.status.confidence,
            'weight': round(1.0 + composite.kernel.status.confidence, 4),
            'target': 'composition',
            'notes': 'kernel composition sample',
        })

    for kernel in kernels[:12]:
        rows.append({
            'id': f'{kernel.kernel_id}-distill',
            'phase': 'D',
            'task_family': kernel.task_family,
            'prompt': f'Distill the repeated pattern from {kernel.name}.',
            'kernel_id': kernel.kernel_id,
            'kernel_name': kernel.name,
            'response': f'{kernel.task_family} => {kernel.description}',
            'verified': True,
            'verification_score': kernel.status.confidence,
            'confidence': kernel.status.confidence,
            'weight': round(1.1 + kernel.status.confidence, 4),
            'target': 'distillation',
            'notes': 'periodic distillation sample',
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate KernelWeave Phase C/D dataset')
    parser.add_argument('--store', default=str(DEFAULT_STORE))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    store = KernelStore(Path(args.store))
    if not store.list_kernels():
        install_kernel_library(store)

    rows = make_rows(store)
    rows.sort(key=lambda row: (row['phase'], row['target'], row['id']))
    split = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split]
    eval_rows = rows[split:]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'train.jsonl', train_rows)
    write_jsonl(out / 'eval.jsonl', eval_rows)

    meta = {
        'rows_total': len(rows),
        'train_rows': len(train_rows),
        'eval_rows': len(eval_rows),
        'kernels': len(store.list_kernels()),
        'targets': sorted({row['target'] for row in rows}),
        'store': str(Path(args.store).resolve()),
        'repo': str(REPO.resolve()),
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'output_dir': str(out.resolve()), 'meta': meta}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
