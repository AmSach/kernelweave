"""Failure semantics for parallel kernel composition.

Models what happens when one branch of parallel composition fails:
- Isolation: Does the other branch continue?
- Rollback: Do we undo partial results?
- Aggregation: How do we combine partial successes?

Three strategies:
1. ALL_OR_NOTHING: All must succeed, otherwise full rollback
2. BEST_EFFORT: Continue with successful branches, ignore failures
3. CIRCUIT_BREAKER: Fail fast on first error, preserve successful state
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from enum import Enum

from ..kernel import Kernel, KernelStatus


class FailureStrategy(Enum):
    """How to handle failures in parallel composition."""
    ALL_OR_NOTHING = "all_or_nothing"  # All must succeed
    BEST_EFFORT = "best_effort"        # Use successful branches
    CIRCUIT_BREAKER = "circuit_breaker"  # Fail fast


@dataclass
class BranchResult:
    """Result from one branch of parallel execution."""
    branch_id: str
    kernel_id: str
    success: bool
    output: dict[str, Any]
    evidence: list[str]
    errors: list[str]
    duration_ms: float
    rollback_actions: list[str] = field(default_factory=list)


@dataclass
class ParallelExecutionResult:
    """Result from parallel kernel execution."""
    success: bool
    branches: list[BranchResult]
    successful_branches: list[BranchResult]
    failed_branches: list[BranchResult]
    aggregated_output: dict[str, Any]
    combined_evidence: list[str]
    rollback_chain: list[str]
    strategy_used: FailureStrategy
    failure_reason: str | None = None


class ParallelFailureHandler:
    """Handle failures in parallel kernel execution.
    
    Usage:
        handler = ParallelFailureHandler(strategy=FailureStrategy.ALL_OR_NOTHING)
        
        results = [
            BranchResult(branch_id="A", success=True, output={"x": 1}, ...),
            BranchResult(branch_id="B", success=False, output={}, ...),
        ]
        
        final = handler.aggregate(results)
        # With ALL_OR_NOTHING: success=False, rollback triggered
        # With BEST_EFFORT: success=True, uses branch A's output
    """
    
    def __init__(
        self,
        strategy: FailureStrategy = FailureStrategy.ALL_OR_NOTHING,
        min_successes: int = 1,
        timeout_ms: float | None = None,
    ):
        self.strategy = strategy
        self.min_successes = min_successes
        self.timeout_ms = timeout_ms
    
    def aggregate(
        self,
        branches: list[BranchResult],
        kernel_a: Kernel | None = None,
        kernel_b: Kernel | None = None,
    ) -> ParallelExecutionResult:
        """Aggregate branch results according to failure strategy."""
        successful = [b for b in branches if b.success]
        failed = [b for b in branches if not b.success]
        
        # Determine overall success based on strategy
        overall_success = self._compute_success(successful, failed)
        
        # Aggregate outputs
        aggregated = self._aggregate_outputs(successful)
        
        # Combine evidence
        combined_evidence = []
        for branch in successful:
            combined_evidence.extend(branch.evidence)
        
        # Build rollback chain if needed
        rollback_chain = []
        if not overall_success:
            rollback_chain = self._build_rollback_chain(branches)
        
        # Determine failure reason
        failure_reason = None
        if not overall_success:
            failure_reason = self._diagnose_failure(failed, successful)
        
        return ParallelExecutionResult(
            success=overall_success,
            branches=branches,
            successful_branches=successful,
            failed_branches=failed,
            aggregated_output=aggregated,
            combined_evidence=combined_evidence,
            rollback_chain=rollback_chain,
            strategy_used=self.strategy,
            failure_reason=failure_reason,
        )
    
    def _compute_success(
        self,
        successful: list[BranchResult],
        failed: list[BranchResult],
    ) -> bool:
        """Determine overall success based on strategy."""
        if self.strategy == FailureStrategy.ALL_OR_NOTHING:
            # All must succeed
            return len(failed) == 0 and len(successful) > 0
        
        elif self.strategy == FailureStrategy.BEST_EFFORT:
            # At least min_successes must succeed
            return len(successful) >= self.min_successes
        
        elif self.strategy == FailureStrategy.CIRCUIT_BREAKER:
            # Fail on first error (checked during execution, not here)
            # For aggregation, same as all_or_nothing
            return len(failed) == 0
        
        return False
    
    def _aggregate_outputs(
        self,
        successful: list[BranchResult],
    ) -> dict[str, Any]:
        """Aggregate outputs from successful branches."""
        if not successful:
            return {}
        
        # Single successful branch: use its output directly
        if len(successful) == 1:
            return successful[0].output.copy()
        
        # Multiple branches: merge outputs
        merged: dict[str, Any] = {}
        for branch in successful:
            for key, value in branch.output.items():
                if key in merged:
                    # Key conflict: create list
                    if not isinstance(merged[key], list):
                        merged[key] = [merged[key]]
                    merged[key].append(value)
                else:
                    merged[key] = value
        
        return merged
    
    def _build_rollback_chain(
        self,
        branches: list[BranchResult],
    ) -> list[str]:
        """Build rollback actions for failed execution."""
        rollback = []
        
        for branch in branches:
            if branch.success:
                # Need to undo successful branch
                rollback.append(f"rollback {branch.branch_id}: undo {branch.kernel_id}")
                rollback.extend(branch.rollback_actions)
            else:
                # Failed branch: record failure reason
                rollback.append(f"failed {branch.branch_id}: {'; '.join(branch.errors)}")
        
        return rollback
    
    def _diagnose_failure(
        self,
        failed: list[BranchResult],
        successful: list[BranchResult],
    ) -> str:
        """Diagnose why the execution failed."""
        if not failed and not successful:
            return "no branches executed"
        
        if failed:
            errors = []
            for branch in failed:
                for error in branch.errors[:3]:  # Top 3 errors
                    errors.append(f"{branch.branch_id}: {error}")
            return "; ".join(errors[:5])  # Top 5 errors
        
        if self.strategy == FailureStrategy.ALL_OR_NOTHING:
            return f"strategy requires all success, but {len(failed)} branches failed"
        
        return f"insufficient successes: {len(successful)} < {self.min_successes}"


def compose_parallel_with_failure(
    kernel_a: Kernel,
    kernel_b: Kernel,
    strategy: FailureStrategy = FailureStrategy.ALL_OR_NOTHING,
    task_family: str = "",
    description: str = "",
) -> tuple[Kernel, ParallelFailureHandler]:
    """Compose two kernels in parallel with explicit failure handling.
    
    Returns:
        - Composite kernel
        - Failure handler for execution
    """
    from . import CompositionType, CompositionResult, detect_conflicts
    
    # Detect conflicts
    conflicts = detect_conflicts(kernel_a, kernel_b)
    
    # Build composite steps
    steps = []
    
    # Parallel execution marker
    steps.append({
        "step": 1,
        "action": "parallel_start",
        "branches": [kernel_a.kernel_id, kernel_b.kernel_id],
        "strategy": strategy.value,
    })
    
    # Branch A steps
    for i, step in enumerate(kernel_a.steps, start=2):
        steps.append({**step, "branch": "A", "step": i})
    
    # Branch B steps (offset)
    offset = len(kernel_a.steps) + 2
    for i, step in enumerate(kernel_b.steps, start=offset):
        steps.append({**step, "branch": "B", "step": i})
    
    # Merge marker
    steps.append({
        "step": len(steps) + 1,
        "action": "parallel_merge",
        "strategy": strategy.value,
        "min_successes": 1 if strategy == FailureStrategy.BEST_EFFORT else 2,
    })
    
    # Combine postconditions with failure handling
    postconditions = kernel_a.postconditions + kernel_b.postconditions
    postconditions.append(f"parallel execution handled by {strategy.value}")
    if strategy == FailureStrategy.ALL_OR_NOTHING:
        postconditions.append("all branches succeeded")
    elif strategy == FailureStrategy.BEST_EFFORT:
        postconditions.append("at least one branch succeeded")
    
    # Combined rollback
    rollback = kernel_a.rollback + kernel_b.rollback
    if strategy == FailureStrategy.ALL_OR_NOTHING:
        rollback.insert(0, "if any branch fails, rollback all successful branches")
    elif strategy == FailureStrategy.BEST_EFFORT:
        rollback.insert(0, "if a branch fails, continue with successful branches")
    
    # Create composite kernel
    composite_id = f"par-{kernel_a.kernel_id[:8]}-{kernel_b.kernel_id[:8]}"
    
    # Compute confidence based on strategy and conflicts
    base_confidence = min(kernel_a.status.confidence, kernel_b.status.confidence)
    conflict_penalty = len(conflicts) * 0.1
    strategy_penalty = 0.05 if strategy == FailureStrategy.BEST_EFFORT else 0.0
    
    composite = Kernel(
        kernel_id=composite_id,
        name=f"parallel({kernel_a.name}, {kernel_b.name})",
        task_family=task_family or f"{kernel_a.task_family}+{kernel_b.task_family}",
        description=description or f"Parallel composition with {strategy.value}",
        input_schema=kernel_a.input_schema,
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        preconditions=kernel_a.preconditions + kernel_b.preconditions,
        postconditions=postconditions,
        steps=steps,
        rollback=rollback,
        evidence_requirements=kernel_a.evidence_requirements + kernel_b.evidence_requirements,
        tests=kernel_a.tests + kernel_b.tests,
        status=KernelStatus(
            state="candidate",
            confidence=max(0.0, base_confidence - conflict_penalty - strategy_penalty),
            failures=0,
            passes=min(kernel_a.status.passes, kernel_b.status.passes),
        ),
        source_trace_ids=kernel_a.source_trace_ids + kernel_b.source_trace_ids,
    )
    
    handler = ParallelFailureHandler(strategy=strategy)
    
    return composite, handler


# Convenience functions for each strategy

def compose_parallel_strict(
    kernel_a: Kernel,
    kernel_b: Kernel,
    task_family: str = "",
    description: str = "",
) -> tuple[Kernel, ParallelFailureHandler]:
    """Parallel composition with ALL_OR_NOTHING semantics."""
    return compose_parallel_with_failure(
        kernel_a, kernel_b,
        strategy=FailureStrategy.ALL_OR_NOTHING,
        task_family=task_family,
        description=description,
    )


def compose_parallel_best_effort(
    kernel_a: Kernel,
    kernel_b: Kernel,
    task_family: str = "",
    description: str = "",
) -> tuple[Kernel, ParallelFailureHandler]:
    """Parallel composition with BEST_EFFORT semantics."""
    return compose_parallel_with_failure(
        kernel_a, kernel_b,
        strategy=FailureStrategy.BEST_EFFORT,
        task_family=task_family,
        description=description,
    )


def compose_parallel_circuit_breaker(
    kernel_a: Kernel,
    kernel_b: Kernel,
    task_family: str = "",
    description: str = "",
) -> tuple[Kernel, ParallelFailureHandler]:
    """Parallel composition with CIRCUIT_BREAKER semantics."""
    return compose_parallel_with_failure(
        kernel_a, kernel_b,
        strategy=FailureStrategy.CIRCUIT_BREAKER,
        task_family=task_family,
        description=description,
    )
