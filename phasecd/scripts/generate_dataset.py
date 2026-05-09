#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def find_repo() -> Path:
    env_vars = ["KERNELWEAVE_CORE_REPO", "KERNELWEAVE_REPO"]
    candidates: list[Path] = []
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            ROOT.parent,
            ROOT.parent.parent,
            Path.cwd(),
            Path.cwd().parent,
            Path("/home/.z/workspaces/con_d7kcq1mzmfXZeHOQ/kernelweave"),
            Path("/home/workspace/kernelweave"),
        ]
    )
    for candidate in Path("/home/.z/workspaces").glob("*/kernelweave"):
        candidates.append(candidate)
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "kernelweave").is_dir():
            return candidate
    raise SystemExit("KernelWeave repo not found. Set KERNELWEAVE_CORE_REPO to the repo root.")


REPO = find_repo()
sys.path.insert(0, str(REPO))

from kernelweave.kernel import KernelStore, TraceEvent
from kernelweave.kernels import install_kernel_library
from kernelweave.compiler import compile_trace_to_kernel
from kernelweave.compose import compose_sequence
from kernelweave.runtime import KernelRuntime
from kernelweave.verifier import HeuristicVerifier

DEFAULT_OUT = ROOT / "data"
DEFAULT_STORE = ROOT / "store"
VERIFIER = HeuristicVerifier()

GENERAL_PROMPTS = [
    "compare two artifacts and summarize the differences",
    "find all TODOs in a codebase",
    "convert JSON to YAML without losing structure",
    "debug a failing training script and explain the fix",
    "summarize the architecture and cost model",
    "write a concise plan for repeated task automation",
    "extract the key facts from this document",
    "turn a rough idea into a structured execution plan",
    "choose the safest kernel for a tool-based task",
    "repair an inference plan that violates a postcondition",
    "compose two verified routines into one workflow",
    "retrieve reusable memory for a repeated task family",
    "classify the task family for this prompt",
    "explain why a result passed verification",
    "describe rollback steps for a failed execution",
    "generate a follow-up task checklist",
    "evaluate whether the result is safe to cache",
    "summarize the data flow for a task family",
    "decide whether a prompt should use memory or generation",
    "trace the execution path for a repeated workflow",
]

PROMPT_TEMPLATES = [
    "Use the {family} kernel to solve: {prompt}",
    "Kernel-aware prompting for {family}: {prompt}",
    "Given the kernel contract, respond to: {prompt}",
    "Execute the verified plan for: {prompt}",
    "Use memory and verification for: {prompt}",
]

SELECTION_TEMPLATES = [
    "Choose the best kernel for: {prompt}",
    "Which kernel should route this prompt: {prompt}",
    "Select the correct kernel family for: {prompt}",
    "Pick the most relevant verified kernel for: {prompt}",
]

TRACE_TEMPLATES = [
    "Trace a successful execution for: {prompt}",
    "Show the verified steps for: {prompt}",
    "Record the kernel trace for: {prompt}",
    "Produce a verified execution trace for: {prompt}",
]

MEMORY_TEMPLATES = [
    "Retrieve memory for: {prompt}",
    "Use kernel memory instead of re-deriving: {prompt}",
    "Find the reusable kernel for: {prompt}",
    "Query memory before generating for: {prompt}",
]

COMPOSITION_TEMPLATES = [
    "Compose {family_a} then {family_b} for: {prompt}",
    "Plan a composed kernel workflow for: {prompt}",
    "Merge the verified kernels for: {prompt}",
    "Chain the two best kernels for: {prompt}",
]

DISTILLATION_TEMPLATES = [
    "Distill the repeated pattern from {kernel_name}.",
    "Compress the reusable behavior of {kernel_name}.",
    "Summarize the kernel logic for {kernel_name}.",
    "Extract the training signal from {kernel_name}.",
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def kernel_prompt(kernel) -> str:
    pre = "; ".join(kernel.preconditions[:3])
    post = "; ".join(kernel.postconditions[:3])
    return f"Use the {kernel.task_family} kernel. Preconditions: {pre}. Postconditions: {post}."


def make_kernel_rows(store: KernelStore, per_kernel: int, rng: random.Random) -> list[dict[str, Any]]:
    kernels = [store.get_kernel(item["kernel_id"]) for item in store.list_kernels()]
    rows: list[dict[str, Any]] = []

    for kernel in kernels:
        for i in range(per_kernel):
            prompt_base = rng.choice(GENERAL_PROMPTS)
            prompt = rng.choice(PROMPT_TEMPLATES).format(family=kernel.task_family, prompt=prompt_base)
            selection_prompt = rng.choice(SELECTION_TEMPLATES).format(prompt=prompt_base)
            trace_prompt = rng.choice(TRACE_TEMPLATES).format(prompt=prompt_base)
            memory_prompt = rng.choice(MEMORY_TEMPLATES).format(prompt=prompt_base)
            distill_prompt = rng.choice(DISTILLATION_TEMPLATES).format(kernel_name=kernel.name)

            response_prompting = json.dumps(
                {
                    "mode": "kernel",
                    "kernel_id": kernel.kernel_id,
                    "kernel_name": kernel.name,
                    "reason": f"matches {kernel.task_family}",
                    "plan": kernel.steps[:8],
                },
                sort_keys=True,
            )
            verification = VERIFIER.verify(response_prompting, kernel.postconditions, kernel.evidence_requirements)
            rows.append(
                {
                    "id": f"{kernel.kernel_id}-prompting-{i}",
                    "phase": "C",
                    "task_family": kernel.task_family,
                    "prompt": prompt,
                    "kernel_id": kernel.kernel_id,
                    "kernel_name": kernel.name,
                    "kernel_steps": kernel.steps,
                    "kernel_preconditions": kernel.preconditions,
                    "kernel_postconditions": kernel.postconditions,
                    "kernel_evidence_requirements": kernel.evidence_requirements,
                    "kernel_rollback": kernel.rollback,
                    "response": response_prompting,
                    "verified": verification.passed,
                    "verification_score": verification.score,
                    "confidence": kernel.status.confidence,
                    "weight": round(1.0 + verification.score + kernel.status.confidence, 4),
                    "target": "prompting",
                    "notes": "kernel-aware prompting sample",
                }
            )

            rows.append(
                {
                    "id": f"{kernel.kernel_id}-selection-{i}",
                    "phase": "C",
                    "task_family": kernel.task_family,
                    "prompt": selection_prompt,
                    "kernel_id": kernel.kernel_id,
                    "kernel_name": kernel.name,
                    "response": json.dumps(
                        {
                            "mode": "kernel",
                            "kernel_id": kernel.kernel_id,
                            "task_family": kernel.task_family,
                            "confidence": kernel.status.confidence,
                        },
                        sort_keys=True,
                    ),
                    "verified": True,
                    "verification_score": kernel.status.confidence,
                    "confidence": kernel.status.confidence,
                    "weight": round(1.0 + kernel.status.confidence, 4),
                    "target": "selection",
                    "notes": "kernel selection classification sample",
                }
            )

            rows.append(
                {
                    "id": f"{kernel.kernel_id}-trace-{i}",
                    "phase": "B",
                    "task_family": kernel.task_family,
                    "prompt": trace_prompt,
                    "kernel_id": kernel.kernel_id,
                    "kernel_name": kernel.name,
                    "kernel_steps": kernel.steps,
                    "kernel_preconditions": kernel.preconditions,
                    "kernel_postconditions": kernel.postconditions,
                    "kernel_evidence_requirements": kernel.evidence_requirements,
                    "kernel_rollback": kernel.rollback,
                    "response": json.dumps(
                        {
                            "trace": kernel.source_trace_ids[:1],
                            "steps": kernel.steps[:8],
                            "postconditions": kernel.postconditions[:4],
                        },
                        sort_keys=True,
                    ),
                    "verified": True,
                    "verification_score": kernel.status.confidence,
                    "confidence": kernel.status.confidence,
                    "weight": round(1.0 + kernel.status.confidence, 4),
                    "target": "trace_finetune",
                    "notes": "verified trace supervision sample",
                }
            )

            rows.append(
                {
                    "id": f"{kernel.kernel_id}-memory-{i}",
                    "phase": "D",
                    "task_family": kernel.task_family,
                    "prompt": memory_prompt,
                    "kernel_id": kernel.kernel_id,
                    "kernel_name": kernel.name,
                    "response": json.dumps(
                        {
                            "mode": "memory",
                            "kernel_id": kernel.kernel_id,
                            "kernel_name": kernel.name,
                            "retrieval": "retrieve kernel and execute its steps",
                        },
                        sort_keys=True,
                    ),
                    "verified": True,
                    "verification_score": kernel.status.confidence,
                    "confidence": kernel.status.confidence,
                    "weight": round(1.2 + kernel.status.confidence, 4),
                    "target": "memory_retrieval",
                    "notes": "memory primitive sample",
                }
            )

            rows.append(
                {
                    "id": f"{kernel.kernel_id}-distill-{i}",
                    "phase": "D",
                    "task_family": kernel.task_family,
                    "prompt": distill_prompt,
                    "kernel_id": kernel.kernel_id,
                    "kernel_name": kernel.name,
                    "response": json.dumps(
                        {
                            "distillation": kernel.description,
                            "kernel_id": kernel.kernel_id,
                            "task_family": kernel.task_family,
                        },
                        sort_keys=True,
                    ),
                    "verified": True,
                    "verification_score": kernel.status.confidence,
                    "confidence": kernel.status.confidence,
                    "weight": round(1.15 + kernel.status.confidence, 4),
                    "target": "distillation",
                    "notes": "periodic distillation sample",
                }
            )

    return rows


def make_general_rows(store: KernelStore, general_count: int, rng: random.Random) -> list[dict[str, Any]]:
    runtime = KernelRuntime(store)
    runtime.preload_embeddings(GENERAL_PROMPTS)
    rows: list[dict[str, Any]] = []
    prompts = [rng.choice(GENERAL_PROMPTS) for _ in range(general_count)]
    runtime.preload_embeddings(prompts)

    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            print(f"[general_rows] {i}/{general_count}", flush=True)
        decision = runtime.evaluate_prompt(prompt)
        result = runtime.run(prompt)
        rows.append(
            {
                "id": f"general-{i}",
                "phase": "A",
                "task_family": decision.mode,
                "prompt": prompt,
                "kernel_id": decision.kernel_id,
                "response": json.dumps(result, sort_keys=True),
                "verified": bool(result.get("mode") == "kernel" or result.get("mode") == "generate"),
                "verification_score": float(result.get("confidence", 0.0)),
                "confidence": float(result.get("confidence", 0.0)),
                "weight": round(1.0 + float(result.get("confidence", 0.0)), 4),
                "target": "general",
                "notes": "general routing / reasoning sample",
            }
        )

    return rows


def make_composition_rows(store: KernelStore, composition_count: int, rng: random.Random) -> list[dict[str, Any]]:
    kernels = [store.get_kernel(item["kernel_id"]) for item in store.list_kernels()]
    rows: list[dict[str, Any]] = []
    if len(kernels) < 2:
        return rows
    pairs = list(combinations(kernels[:10], 2))
    for i, (kernel_a, kernel_b) in enumerate(pairs[:composition_count]):
        composite = compose_sequence(kernel_a, kernel_b)
        prompt_base = rng.choice(GENERAL_PROMPTS)
        rows.append(
            {
                "id": f"compose-{i}",
                "phase": "C",
                "task_family": composite.kernel.task_family,
                "prompt": rng.choice(COMPOSITION_TEMPLATES).format(
                    family_a=kernel_a.task_family,
                    family_b=kernel_b.task_family,
                    prompt=prompt_base,
                ),
                "kernel_id": composite.kernel.kernel_id,
                "kernel_name": composite.kernel.name,
                "kernel_steps": composite.kernel.steps,
                "kernel_preconditions": composite.kernel.preconditions,
                "kernel_postconditions": composite.kernel.postconditions,
                "kernel_evidence_requirements": composite.kernel.evidence_requirements,
                "kernel_rollback": composite.kernel.rollback,
                "response": json.dumps(
                    {
                        "composition": [kernel_a.kernel_id, kernel_b.kernel_id],
                        "result_kernel": composite.kernel.kernel_id,
                        "valid": composite.is_valid(),
                    },
                    sort_keys=True,
                ),
                "verified": composite.is_valid(),
                "verification_score": composite.kernel.status.confidence,
                "confidence": composite.kernel.status.confidence,
                "weight": round(1.0 + composite.kernel.status.confidence, 4),
                "target": "composition",
                "notes": "kernel composition sample",
            }
        )
    return rows


def make_rows(store: KernelStore, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    # Bigger general set, because tiny data is cosplay.
    general_count = 1200
    per_kernel = 16
    composition_count = 60
    rows: list[dict[str, Any]] = []
    rows.extend(make_general_rows(store, general_count=general_count, rng=rng))
    rows.extend(make_kernel_rows(store, per_kernel=per_kernel, rng=rng))
    rows.extend(make_composition_rows(store, composition_count=composition_count, rng=rng))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KernelWeave Phase C/D dataset")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    store = KernelStore(Path(args.store))
    if not store.list_kernels():
        install_kernel_library(store)

    rows = make_rows(store, seed=args.seed)
    rows.sort(key=lambda row: (row["phase"], row["target"], row["id"]))
    split = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split]
    eval_rows = rows[split:]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)

    meta = {
        "rows_total": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "kernels": len(store.list_kernels()),
        "targets": sorted({row["target"] for row in rows}),
        "store": str(Path(args.store).resolve()),
        "repo": str(REPO.resolve()),
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(out.resolve()), "meta": meta}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
