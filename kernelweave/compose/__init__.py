"""Kernel composition algebra — combining kernels into higher-level behaviors.

This is the GENUINELY HARD part: given two verified kernels, how do you
compose them into a new kernel that preserves the verification guarantees?

Key challenges:
1. Ordering: Which kernel runs first? Does order matter?
2. Data flow: How does output from kernel A become input to kernel B?
3. Conflict: What if preconditions of B contradict postconditions of A?
4. Evidence: How is evidence accumulated across the composition?
5. Rollback: If B fails, do we rollback A?

This module provides:
- Composition operators (sequence, parallel, conditional, loop)
- Conflict detection between kernels
- Evidence accumulation across compositions
- Rollback propagation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from enum import Enum
import json

from ..kernel import Kernel, KernelStatus
from ..metrics import jaccard_similarity, semantic_similarity, clamp


class CompositionType(Enum):
    SEQUENCE = "sequence"      # A then B
    PARALLEL = "parallel"      # A and B simultaneously
    CONDITIONAL = "conditional"  # if X then A else B
    LOOP = "loop"             # repeat A while condition
    CHOICE = "choice"         # A or B (pick best match)


class ConflictType(Enum):
    PRECONDITION = "precondition"   # B.precondition conflicts with A.postcondition
    OUTPUT_SCHEMA = "output_schema"  # Output shapes don't match
    EVIDENCE = "evidence"           # Evidence requirements conflict
    TEMPORAL = "temporal"           # Ordering conflicts


@dataclass
class Conflict:
    """A detected conflict between kernels."""
    conflict_type: ConflictType
    kernel_a: str
    kernel_b: str
    description: str
    resolution: str | None = None
    severity: float = 1.0  # 0 = minor, 1 = blocking


@dataclass
class CompositionResult:
    """Result of composing kernels."""
    kernel: Kernel
    conflicts: list[Conflict]
    composition_type: CompositionType
    evidence_accumulation: list[str]
    rollback_chain: list[str]
    
    def is_valid(self) -> bool:
        return all(c.severity < 1.0 for c in self.conflicts)


def detect_conflicts(kernel_a: Kernel, kernel_b: Kernel) -> list[Conflict]:
    """Detect conflicts between two kernels.
    
    This is crucial for safe composition — we need to know if
    kernel B's preconditions can be satisfied after kernel A runs.
    """
    conflicts = []
    
    # Check precondition/postcondition compatibility
    a_postconditions = set(s.lower() for s in kernel_a.postconditions)
    b_preconditions = set(s.lower() for s in kernel_b.preconditions)
    
    for precond in kernel_b.preconditions:
        precond_lower = precond.lower()
        
        # Check if precondition is explicitly violated
        for postcond in kernel_a.postconditions:
            postcond_lower = postcond.lower()
            
            # Negation conflicts: "X not triggered" vs "X required"
            if "not " in postcond_lower and precond_lower.replace("not ", "") in postcond_lower:
                conflicts.append(Conflict(
                    conflict_type=ConflictType.PRECONDITION,
                    kernel_a=kernel_a.kernel_id,
                    kernel_b=kernel_b.kernel_id,
                    description=f"Postcondition '{postcond}' conflicts with precondition '{precond}'",
                    resolution="Remove negation conflict or reorder kernels",
                    severity=1.0,
                ))
    
    # Check output schema compatibility for sequence
    if kernel_a.output_schema.get("type") == "object" and kernel_b.input_schema.get("type") == "object":
        a_props = set(kernel_a.output_schema.get("properties", {}).keys())
        b_required = set(kernel_b.input_schema.get("required", []))
        
        missing = b_required - a_props
        if missing:
            conflicts.append(Conflict(
                conflict_type=ConflictType.OUTPUT_SCHEMA,
                kernel_a=kernel_a.kernel_id,
                kernel_b=kernel_b.kernel_id,
                description=f"Kernel B requires fields {missing} not provided by kernel A",
                resolution=f"Add adapter kernel to provide {missing}",
                severity=0.5,  # Can be fixed with adapter
            ))
    
    # Check evidence requirements
    a_evidence = set(kernel_a.evidence_requirements)
    b_evidence = set(kernel_b.evidence_requirements)
    
    # Look for contradictory evidence requirements
    for ev_a in a_evidence:
        ev_a_lower = ev_a.lower()
        for ev_b in b_evidence:
            ev_b_lower = ev_b.lower()
            
            # Contradictory: "X is sufficient" vs "X is insufficient"
            if ("sufficient" in ev_a_lower and "insufficient" in ev_b_lower) or \
               ("insufficient" in ev_a_lower and "sufficient" in ev_b_lower):
                conflicts.append(Conflict(
                    conflict_type=ConflictType.EVIDENCE,
                    kernel_a=kernel_a.kernel_id,
                    kernel_b=kernel_b.kernel_id,
                    description=f"Contradictory evidence: '{ev_a}' vs '{ev_b}'",
                    resolution="Resolve evidence contradiction in composition",
                    severity=0.7,
                ))
    
    return conflicts


def compose_sequence(
    kernel_a: Kernel,
    kernel_b: Kernel,
    name: str | None = None,
    resolve_conflicts: bool = True,
) -> CompositionResult:
    """Compose two kernels in sequence: A then B.
    
    The resulting kernel:
    - Has A's preconditions + B's preconditions (minus those satisfied by A)
    - Has B's postconditions (A's are intermediate)
    - Has A's steps followed by B's steps
    - Accumulates evidence from both
    """
    conflicts = detect_conflicts(kernel_a, kernel_b)
    
    # If blocking conflicts and not resolving, fail
    if any(c.severity >= 1.0 for c in conflicts) and not resolve_conflicts:
        raise ValueError(f"Blocking conflicts detected: {[c.description for c in conflicts if c.severity >= 1.0]}")
    
    # Generate composite kernel
    composite_id = f"comp-{kernel_a.kernel_id[:8]}-{kernel_b.kernel_id[:8]}"
    
    # Merge preconditions
    a_post_set = set(s.lower() for s in kernel_a.postconditions)
    b_pre_filtered = [
        pre for pre in kernel_b.preconditions
        if pre.lower() not in a_post_set  # Remove preconditions satisfied by A
    ]
    
    preconditions = kernel_a.preconditions + b_pre_filtered
    preconditions = list(dict.fromkeys(preconditions))  # Dedupe, preserve order
    
    # Use B's postconditions (A's are intermediate)
    postconditions = kernel_b.postconditions.copy()
    postconditions.append(f"intermediate: {kernel_a.name} completed successfully")
    
    # Merge steps with sequence markers
    steps = []
    offset = 0
    for step in kernel_a.steps:
        step_copy = step.copy()
        step_copy["step"] = offset + step.get("step", 1)
        step_copy["phase"] = "first"
        steps.append(step_copy)
        offset += 1
    
    for step in kernel_b.steps:
        step_copy = step.copy()
        step_copy["step"] = offset + step.get("step", 1)
        step_copy["phase"] = "second"
        steps.append(step_copy)
        offset += 1
    
    # Merge evidence
    evidence = list(dict.fromkeys(kernel_a.evidence_requirements + kernel_b.evidence_requirements))
    evidence.append(f"sequence: {kernel_a.kernel_id} → {kernel_b.kernel_id}")
    
    # Merge rollback: if B fails, may need to rollback A
    rollback = kernel_b.rollback.copy()
    rollback.append(f"if phase 2 fails, consider rolling back {kernel_a.name}")
    rollback.extend(kernel_a.rollback)
    
    # Build output schema
    output_schema = kernel_b.output_schema.copy()
    if "properties" not in output_schema:
        output_schema["properties"] = {}
    output_schema["properties"]["_intermediate"] = kernel_a.output_schema
    
    # Compute combined confidence
    combined_confidence = min(kernel_a.status.confidence, kernel_b.status.confidence) * 0.95
    
    kernel = Kernel(
        kernel_id=composite_id,
        name=name or f"{kernel_a.name} → {kernel_b.name}",
        task_family=f"sequence({kernel_a.task_family}, {kernel_b.task_family})",
        description=f"Sequential composition of {kernel_a.name} and {kernel_b.name}",
        input_schema=kernel_a.input_schema,
        output_schema=output_schema,
        preconditions=preconditions,
        postconditions=postconditions,
        steps=steps,
        rollback=rollback,
        evidence_requirements=evidence,
        tests=kernel_a.tests + kernel_b.tests,
        status=KernelStatus(
            state="composed",
            confidence=round(combined_confidence, 4),
            failures=0,
            passes=kernel_a.status.passes + kernel_b.status.passes,
        ),
        source_trace_ids=kernel_a.source_trace_ids + kernel_b.source_trace_ids,
    )
    
    return CompositionResult(
        kernel=kernel,
        conflicts=conflicts,
        composition_type=CompositionType.SEQUENCE,
        evidence_accumulation=evidence,
        rollback_chain=[kernel_b.kernel_id, kernel_a.kernel_id],
    )


def compose_parallel(
    kernel_a: Kernel,
    kernel_b: Kernel,
    name: str | None = None,
    merge_strategy: Literal["intersection", "union", "custom"] = "union",
) -> CompositionResult:
    """Compose two kernels in parallel: A and B simultaneously.
    
    The resulting kernel:
    - Merges preconditions from both
    - Merges postconditions from both
    - Runs both step sequences
    - Requires both to succeed
    """
    conflicts = detect_conflicts(kernel_a, kernel_b)
    
    composite_id = f"par-{kernel_a.kernel_id[:8]}-{kernel_b.kernel_id[:8]}"
    
    # Merge preconditions (union)
    preconditions = list(dict.fromkeys(kernel_a.preconditions + kernel_b.preconditions))
    
    # Merge postconditions based on strategy
    if merge_strategy == "union":
        postconditions = list(dict.fromkeys(kernel_a.postconditions + kernel_b.postconditions))
    elif merge_strategy == "intersection":
        postconditions = [
            p for p in kernel_a.postconditions
            if any(jaccard_similarity(p, q) > 0.5 for q in kernel_b.postconditions)
        ]
    else:
        postconditions = kernel_a.postconditions + kernel_b.postconditions
    
    # Interleave steps
    steps = []
    max_len = max(len(kernel_a.steps), len(kernel_b.steps))
    for i in range(max_len):
        if i < len(kernel_a.steps):
            step = kernel_a.steps[i].copy()
            step["step"] = len(steps) + 1
            step["branch"] = "A"
            steps.append(step)
        if i < len(kernel_b.steps):
            step = kernel_b.steps[i].copy()
            step["step"] = len(steps) + 1
            step["branch"] = "B"
            steps.append(step)
    
    # Merge evidence
    evidence = list(dict.fromkeys(kernel_a.evidence_requirements + kernel_b.evidence_requirements))
    evidence.append(f"parallel: {kernel_a.kernel_id} ∥ {kernel_b.kernel_id}")
    
    # Rollback: either failing requires rollback
    rollback = kernel_a.rollback + kernel_b.rollback
    
    # Output schema: merge properties
    output_schema = {"type": "object", "properties": {}, "required": []}
    output_schema["properties"]["_a"] = kernel_a.output_schema
    output_schema["properties"]["_b"] = kernel_b.output_schema
    
    # Combined confidence (parallel is riskier)
    combined_confidence = min(kernel_a.status.confidence, kernel_b.status.confidence) * 0.85
    
    kernel = Kernel(
        kernel_id=composite_id,
        name=name or f"{kernel_a.name} ∥ {kernel_b.name}",
        task_family=f"parallel({kernel_a.task_family}, {kernel_b.task_family})",
        description=f"Parallel composition of {kernel_a.name} and {kernel_b.name}",
        input_schema={"type": "object", "properties": {
            "_a": kernel_a.input_schema,
            "_b": kernel_b.input_schema,
        }},
        output_schema=output_schema,
        preconditions=preconditions,
        postconditions=postconditions,
        steps=steps,
        rollback=rollback,
        evidence_requirements=evidence,
        tests=kernel_a.tests + kernel_b.tests,
        status=KernelStatus(
            state="composed",
            confidence=round(combined_confidence, 4),
            failures=0,
            passes=min(kernel_a.status.passes, kernel_b.status.passes),
        ),
        source_trace_ids=kernel_a.source_trace_ids + kernel_b.source_trace_ids,
    )
    
    return CompositionResult(
        kernel=kernel,
        conflicts=conflicts,
        composition_type=CompositionType.PARALLEL,
        evidence_accumulation=evidence,
        rollback_chain=[kernel_a.kernel_id, kernel_b.kernel_id],
    )


def compose_conditional(
    kernel_a: Kernel,
    kernel_b: Kernel,
    condition: str,
    name: str | None = None,
) -> CompositionResult:
    """Compose two kernels conditionally: if condition then A else B.
    
    The condition is a predicate over the input.
    """
    conflicts = detect_conflicts(kernel_a, kernel_b)
    
    composite_id = f"cond-{kernel_a.kernel_id[:8]}-{kernel_b.kernel_id[:8]}"
    
    # Preconditions: condition check + both kernel preconditions
    preconditions = [f"condition evaluable: {condition}"]
    preconditions.extend(kernel_a.preconditions)
    preconditions.extend(kernel_b.preconditions)
    preconditions = list(dict.fromkeys(preconditions))
    
    # Postconditions: either A or B succeeded
    postconditions = [
        f"if {condition} then {p}" for p in kernel_a.postconditions
    ] + [
        f"if not {condition} then {p}" for p in kernel_b.postconditions
    ]
    postconditions.append("exactly one branch executed")
    
    # Steps with condition check
    steps = [
        {"step": 1, "action": "condition", "condition": condition},
    ]
    for step in kernel_a.steps:
        step_copy = step.copy()
        step_copy["step"] = len(steps) + 1
        step_copy["branch"] = "then"
        steps.append(step_copy)
    for step in kernel_b.steps:
        step_copy = step.copy()
        step_copy["step"] = len(steps) + 1
        step_copy["branch"] = "else"
        steps.append(step_copy)
    
    # Confidence is weighted average
    combined_confidence = (kernel_a.status.confidence + kernel_b.status.confidence) / 2 * 0.9
    
    kernel = Kernel(
        kernel_id=composite_id,
        name=name or f"if {condition[:20]} then {kernel_a.name} else {kernel_b.name}",
        task_family=f"conditional({kernel_a.task_family}, {kernel_b.task_family})",
        description=f"Conditional composition: {condition}",
        input_schema=kernel_a.input_schema,
        output_schema={"type": "object", "properties": {
            "branch": {"type": "string"},
            "result": {"type": "object"},
        }},
        preconditions=preconditions,
        postconditions=postconditions,
        steps=steps,
        rollback=kernel_a.rollback + kernel_b.rollback,
        evidence_requirements=kernel_a.evidence_requirements + kernel_b.evidence_requirements,
        tests=kernel_a.tests + kernel_b.tests,
        status=KernelStatus(
            state="composed",
            confidence=round(combined_confidence, 4),
            failures=0,
            passes=min(kernel_a.status.passes, kernel_b.status.passes),
        ),
        source_trace_ids=kernel_a.source_trace_ids + kernel_b.source_trace_ids,
    )
    
    return CompositionResult(
        kernel=kernel,
        conflicts=conflicts,
        composition_type=CompositionType.CONDITIONAL,
        evidence_accumulation=kernel_a.evidence_requirements + kernel_b.evidence_requirements,
        rollback_chain=[kernel_a.kernel_id, kernel_b.kernel_id],
    )


def compose_loop(
    kernel: Kernel,
    condition: str,
    max_iterations: int = 10,
    name: str | None = None,
) -> CompositionResult:
    """Compose a kernel as a loop: repeat kernel while condition.
    
    This is DANGEROUS because:
    - Evidence accumulates differently each iteration
    - Rollback becomes complex (which iteration's effects?)
    - Confidence degrades with iterations
    """
    composite_id = f"loop-{kernel.kernel_id[:8]}"
    
    # Preconditions: condition evaluable + kernel preconditions
    preconditions = [
        f"loop condition evaluable: {condition}",
        f"max iterations: {max_iterations}",
    ] + kernel.preconditions
    
    # Postconditions: condition false OR max iterations
    postconditions = kernel.postconditions + [
        f"loop terminated: {condition} is false OR iterations >= {max_iterations}",
        "all iterations completed successfully",
    ]
    
    # Steps: condition check + kernel steps, repeated
    steps = [
        {"step": 1, "action": "loop_check", "condition": condition},
    ]
    for i in range(max_iterations):
        for step in kernel.steps:
            step_copy = step.copy()
            step_copy["step"] = len(steps) + 1
            step_copy["iteration"] = i
            steps.append(step_copy)
        steps.append({"step": len(steps) + 1, "action": "loop_check", "condition": condition, "iteration": i})
    
    # Confidence degrades with iterations
    combined_confidence = kernel.status.confidence * (0.9 ** max_iterations)
    
    loop_kernel = Kernel(
        kernel_id=composite_id,
        name=name or f"loop({kernel.name})",
        task_family=f"loop({kernel.task_family})",
        description=f"Loop composition: repeat {kernel.name} while {condition}",
        input_schema=kernel.input_schema,
        output_schema={
            "type": "object",
            "properties": {
                "iterations": {"type": "integer"},
                "results": {"type": "array"},
            },
        },
        preconditions=preconditions,
        postconditions=postconditions,
        steps=steps,
        rollback=[
            f"rollback all iteration effects",
            f"if iteration N fails, rollback iterations 1..N",
        ] + kernel.rollback,
        evidence_requirements=kernel.evidence_requirements + [
            "iteration count recorded",
            "termination condition logged",
        ],
        tests=kernel.tests + [
            {"name": "loop-termination", "input": {}, "expected": {"max_iterations": max_iterations}},
        ],
        status=KernelStatus(
            state="composed",
            confidence=round(combined_confidence, 4),
            failures=0,
            passes=kernel.status.passes,
        ),
        source_trace_ids=kernel.source_trace_ids,
    )
    
    return CompositionResult(
        kernel=loop_kernel,
        conflicts=[],
        composition_type=CompositionType.LOOP,
        evidence_accumulation=kernel.evidence_requirements * max_iterations,
        rollback_chain=[kernel.kernel_id] * max_iterations,
    )


class CompositionBuilder:
    """Fluent API for building complex compositions."""
    
    def __init__(self, base_kernel: Kernel):
        self.kernel = base_kernel
        self.operations: list[tuple[CompositionType, Kernel, dict]] = []
    
    def then(self, kernel: Kernel, **kwargs) -> "CompositionBuilder":
        """Add sequential composition."""
        self.operations.append((CompositionType.SEQUENCE, kernel, kwargs))
        return self
    
    def parallel(self, kernel: Kernel, **kwargs) -> "CompositionBuilder":
        """Add parallel composition."""
        self.operations.append((CompositionType.PARALLEL, kernel, kwargs))
        return self
    
    def conditional(self, kernel_a: Kernel, kernel_b: Kernel, condition: str, **kwargs) -> "CompositionBuilder":
        """Add conditional composition."""
        self.operations.append((CompositionType.CONDITIONAL, (kernel_a, kernel_b), {"condition": condition, **kwargs}))
        return self
    
    def loop(self, condition: str, max_iterations: int = 10, **kwargs) -> "CompositionBuilder":
        """Add loop composition."""
        self.operations.append((CompositionType.LOOP, self.kernel, {"condition": condition, "max_iterations": max_iterations, **kwargs}))
        return self
    
    def build(self) -> CompositionResult:
        """Build the final composed kernel."""
        result = None
        current = self.kernel
        
        for op_type, operand, kwargs in self.operations:
            if op_type == CompositionType.SEQUENCE:
                result = compose_sequence(current, operand, **kwargs)
            elif op_type == CompositionType.PARALLEL:
                result = compose_parallel(current, operand, **kwargs)
            elif op_type == CompositionType.CONDITIONAL:
                kernel_a, kernel_b = operand
                result = compose_conditional(kernel_a, kernel_b, kwargs["condition"], **{k: v for k, v in kwargs.items() if k != "condition"})
            elif op_type == CompositionType.LOOP:
                result = compose_loop(current, kwargs["condition"], kwargs.get("max_iterations", 10), **{k: v for k, v in kwargs.items() if k not in ("condition", "max_iterations")})
            
            if result:
                current = result.kernel
        
        return result or CompositionResult(
            kernel=self.kernel,
            conflicts=[],
            composition_type=CompositionType.SEQUENCE,
            evidence_accumulation=[],
            rollback_chain=[],
        )
