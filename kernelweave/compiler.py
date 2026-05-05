from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any
import math

from .kernel import Kernel, KernelStatus, TraceEvent


@dataclass
class CompilationStats:
    trace_length: int
    distinct_tools: int
    evidence_count: int
    compression_gain: float
    confidence: float


def _kernel_id(task_family: str, summary: str) -> str:
    digest = sha256(f"{task_family}:{summary}".encode("utf-8")).hexdigest()
    return f"kw-{digest[:16]}"


def score_kernel(events: list[TraceEvent]) -> CompilationStats:
    trace_length = len(events)
    distinct_tools = len({event.payload.get("tool") for event in events if event.kind == "tool" and event.payload.get("tool")})
    evidence_count = sum(1 for event in events if event.kind in {"evidence", "observation", "verification"})
    compression_gain = max(0.0, (trace_length + evidence_count) / max(1, distinct_tools + 1) - 1.0)
    confidence = 1.0 / (1.0 + math.exp(-0.8 * (evidence_count - 2)))
    return CompilationStats(trace_length, distinct_tools, evidence_count, compression_gain, confidence)


def compile_trace_to_kernel(trace_id: str, task_family: str, description: str, events: list[TraceEvent], expected_output: dict[str, Any]) -> Kernel:
    stats = score_kernel(events)
    summary = description.strip() or task_family
    kernel_id = _kernel_id(task_family, summary)

    steps = []
    preconditions = []
    postconditions = []
    rollback = []
    evidence_requirements = []
    tests = []

    for idx, event in enumerate(events, start=1):
        if event.kind == "tool":
            steps.append({"step": idx, "action": "tool", "tool": event.payload.get("tool"), "args": event.payload.get("args", {})})
        elif event.kind == "plan":
            steps.append({"step": idx, "action": "plan", "text": event.payload.get("text", "")})
        elif event.kind == "decision":
            steps.append({"step": idx, "action": "decision", "text": event.payload.get("text", "")})
        elif event.kind == "verification":
            evidence_requirements.append(event.payload.get("text", "verification"))
        elif event.kind == "evidence":
            evidence_requirements.append(event.payload.get("text", "evidence"))
        elif event.kind == "failure":
            rollback.append(event.payload.get("text", "rollback on failure"))

    preconditions.extend([
        f"task belongs to family: {task_family}",
        "inputs match schema",
        "evidence channel available",
    ])
    postconditions.extend([
        "output schema satisfied",
        "all required evidence recorded",
        "rollback not triggered",
    ])
    rollback.extend([
        "if evidence is insufficient, return to planning",
        "if a tool returns contradictory data, invalidate kernel",
    ])
    evidence_requirements.extend([
        "source trace logged",
        "final response conforms to expected output",
    ])
    tests.append({
        "name": "output-shape",
        "input": {"task_family": task_family},
        "expected": expected_output,
    })
    tests.append({
        "name": "evidence-completeness",
        "input": {"trace_id": trace_id},
        "expected": {"min_evidence": max(2, stats.evidence_count)},
    })

    return Kernel(
        kernel_id=kernel_id,
        name=f"{task_family} kernel",
        task_family=task_family,
        description=summary,
        input_schema={"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]},
        preconditions=preconditions,
        postconditions=postconditions,
        steps=steps,
        rollback=rollback,
        evidence_requirements=evidence_requirements,
        tests=tests,
        status=KernelStatus(
            state="candidate" if stats.confidence < 0.82 else "verified",
            confidence=round(stats.confidence, 4),
            failures=0,
            passes=1,
        ),
        source_trace_ids=[trace_id],
    )
