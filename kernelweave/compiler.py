from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any
import math

from .kernel import Kernel, KernelStatus, TraceEvent
from .metrics import clamp, cosine_similarity, jaccard_similarity, sigmoid


@dataclass
class CompilationStats:
    trace_length: int
    distinct_tools: int
    evidence_count: int
    compression_gain: float
    confidence: float
    matchfulness: float


def _kernel_id(task_family: str, summary: str) -> str:
    digest = sha256(f"{task_family}:{summary}".encode("utf-8")).hexdigest()
    return f"kw-{digest[:16]}"


def score_kernel(events: list[TraceEvent], task_family: str = "", description: str = "") -> CompilationStats:
    trace_length = len(events)
    distinct_tools = len(
        {
            event.payload.get("tool")
            for event in events
            if event.kind == "tool" and event.payload.get("tool")
        }
    )
    evidence_count = sum(1 for event in events if event.kind in {"evidence", "observation", "verification"})
    compression_gain = max(0.0, (trace_length + evidence_count) / max(1, distinct_tools + 1) - 1.0)
    raw_confidence = 0.55 * sigmoid(0.8 * (evidence_count - 2)) + 0.25 * sigmoid(0.4 * (trace_length - 3)) + 0.20 * sigmoid(compression_gain - 0.5)
    matchfulness = clamp(0.5 * jaccard_similarity(task_family, description) + 0.5 * cosine_similarity(task_family, description))
    confidence = clamp(raw_confidence)
    return CompilationStats(trace_length, distinct_tools, evidence_count, compression_gain, confidence, matchfulness)


def compile_trace_to_kernel(
    trace_id: str,
    task_family: str,
    description: str,
    events: list[TraceEvent],
    expected_output: dict[str, Any],
    backend: Any | None = None,
) -> Kernel:
    stats = score_kernel(events, task_family=task_family, description=description)
    summary = description.strip() or task_family
    kernel_id = _kernel_id(task_family, summary)

    steps: list[dict[str, Any]] = []
    preconditions: list[str] = []
    postconditions: list[str] = []
    rollback: list[str] = []
    evidence_requirements: list[str] = []
    tests: list[dict[str, Any]] = []

    for idx, event in enumerate(events, start=1):
        if event.kind == "tool":
            steps.append(
                {
                    "step": idx,
                    "action": "tool",
                    "tool": event.payload.get("tool"),
                    "args": event.payload.get("args", {}),
                }
            )
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
        elif event.kind == "observation":
            steps.append({"step": idx, "action": "observation", "text": event.payload.get("text", "")})

    preconditions.extend(
        [
            f"task belongs to family: {task_family}",
            "inputs match schema",
            "evidence channel available",
            "trace corresponds to a solved task",
        ]
    )
    postconditions.extend(
        [
            "output schema satisfied",
            "all required evidence recorded",
            "rollback not triggered",
            "tests passed on admission",
        ]
    )

    if backend is not None:
        try:
            import json
            import re
            
            events_summary = "\n".join([f"- {e.kind}: {e.payload.get('text', e.payload.get('tool', ''))}" for e in events])
            prompt = f"""Analyze this execution trace for task family '{task_family}' and description '{description}'.
Extract:
1. Preconditions: What must be true BEFORE this kernel can run?
2. Postconditions: What is guaranteed to be true AFTER this kernel runs?

Trace Events:
{events_summary}

Return JSON with fields 'preconditions' (list of strings) and 'postconditions' (list of strings). Output ONLY valid JSON.
"""
            response = backend.generate(prompt, system_prompt="You are an AI that extracts kernel contracts from execution traces. Output ONLY valid JSON.")
            text = response.text.strip()
            
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found")
            
            extracted_pre = parsed.get("preconditions", [])
            extracted_post = parsed.get("postconditions", [])
            
            preconditions.extend([f"extracted: {p}" for p in extracted_pre])
            postconditions.extend([f"extracted: {p}" for p in extracted_post])
            
        except Exception:
            pass
    rollback.extend(
        [
            "if evidence is insufficient, return to planning",
            "if a tool returns contradictory data, invalidate kernel",
            "if schema inference is unstable, demote kernel confidence",
        ]
    )
    evidence_requirements.extend(["source trace logged", "final response conforms to expected output"])
    tests.append(
        {
            "name": "output-shape",
            "input": {"task_family": task_family},
            "expected": expected_output,
        }
    )
    tests.append(
        {
            "name": "evidence-completeness",
            "input": {"trace_id": trace_id},
            "expected": {"min_evidence": max(2, stats.evidence_count)},
        }
    )
    tests.append(
        {
            "name": "confidence-floor",
            "input": {"task_family": task_family},
            "expected": {"min_confidence": round(stats.confidence, 4)},
        }
    )

    state = "verified" if stats.confidence >= 0.7 else "candidate"
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
            state=state,
            confidence=round(stats.confidence, 4),
            failures=0,
            passes=1,
        ),
        source_trace_ids=[trace_id],
    )
