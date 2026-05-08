"""Tests for parallel failure semantics."""
import pytest
from kernelweave.compose import (
    FailureStrategy,
    BranchResult,
    ParallelFailureHandler,
    compose_parallel_with_failure,
    compose_parallel_strict,
    compose_parallel_best_effort,
)
from kernelweave import Kernel, KernelStatus


def make_kernel(name: str, confidence: float = 0.8) -> Kernel:
    """Helper to create test kernels."""
    return Kernel(
        kernel_id=f"test-{name}",
        name=name,
        task_family="test",
        description="test kernel",
        input_schema={"type": "object"},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        preconditions=["input provided"],
        postconditions=["output valid"],
        steps=[{"step": 1, "action": "test"}],
        rollback=[],
        evidence_requirements=[],
        tests=[],
        status=KernelStatus(state="candidate", confidence=confidence, failures=0, passes=1),
        source_trace_ids=[],
    )


def test_failure_strategy_enum():
    """Test failure strategy values."""
    assert FailureStrategy.ALL_OR_NOTHING.value == "all_or_nothing"
    assert FailureStrategy.BEST_EFFORT.value == "best_effort"
    assert FailureStrategy.CIRCUIT_BREAKER.value == "circuit_breaker"


def test_branch_result_creation():
    """Test branch result dataclass."""
    result = BranchResult(
        branch_id="A",
        kernel_id="test-kernel",
        success=True,
        output={"x": 1},
        evidence=["test evidence"],
        errors=[],
        duration_ms=100.0,
    )
    
    assert result.success
    assert result.output == {"x": 1}
    assert len(result.evidence) == 1


def test_handler_all_or_nothing_success():
    """Test ALL_OR_NOTHING strategy with all successes."""
    handler = ParallelFailureHandler(strategy=FailureStrategy.ALL_OR_NOTHING)
    
    branches = [
        BranchResult("A", "k1", True, {"x": 1}, [], [], 100),
        BranchResult("B", "k2", True, {"y": 2}, [], [], 100),
    ]
    
    result = handler.aggregate(branches)
    
    assert result.success
    assert len(result.successful_branches) == 2
    assert len(result.failed_branches) == 0


def test_handler_all_or_nothing_failure():
    """Test ALL_OR_NOTHING strategy with one failure."""
    handler = ParallelFailureHandler(strategy=FailureStrategy.ALL_OR_NOTHING)
    
    branches = [
        BranchResult("A", "k1", True, {"x": 1}, [], [], 100),
        BranchResult("B", "k2", False, {}, [], ["error"], 100),
    ]
    
    result = handler.aggregate(branches)
    
    assert not result.success
    assert len(result.successful_branches) == 1
    assert len(result.failed_branches) == 1
    assert len(result.rollback_chain) > 0


def test_handler_best_effort_partial_success():
    """Test BEST_EFFORT strategy with partial success."""
    handler = ParallelFailureHandler(strategy=FailureStrategy.BEST_EFFORT)
    
    branches = [
        BranchResult("A", "k1", True, {"x": 1}, [], [], 100),
        BranchResult("B", "k2", False, {}, [], ["error"], 100),
    ]
    
    result = handler.aggregate(branches)
    
    assert result.success  # Succeeds because one branch succeeded
    assert len(result.successful_branches) == 1
    assert result.aggregated_output == {"x": 1}


def test_handler_best_effort_all_fail():
    """Test BEST_EFFORT strategy when all branches fail."""
    handler = ParallelFailureHandler(strategy=FailureStrategy.BEST_EFFORT)
    
    branches = [
        BranchResult("A", "k1", False, {}, [], ["error1"], 100),
        BranchResult("B", "k2", False, {}, [], ["error2"], 100),
    ]
    
    result = handler.aggregate(branches)
    
    assert not result.success
    assert result.failure_reason is not None


def test_handler_aggregate_outputs():
    """Test output aggregation from multiple branches."""
    handler = ParallelFailureHandler(strategy=FailureStrategy.BEST_EFFORT)
    
    branches = [
        BranchResult("A", "k1", True, {"x": 1, "shared": "a"}, [], [], 100),
        BranchResult("B", "k2", True, {"y": 2, "shared": "b"}, [], [], 100),
    ]
    
    result = handler.aggregate(branches)
    
    assert "x" in result.aggregated_output
    assert "y" in result.aggregated_output
    # Conflict: shared key becomes list
    assert "shared" in result.aggregated_output


def test_compose_parallel_with_failure():
    """Test parallel composition with failure handler."""
    kernel_a = make_kernel("A")
    kernel_b = make_kernel("B")
    
    composite, handler = compose_parallel_with_failure(
        kernel_a, kernel_b,
        strategy=FailureStrategy.BEST_EFFORT,
    )
    
    assert composite.kernel_id.startswith("par-")
    assert "parallel" in composite.name.lower()
    assert handler.strategy == FailureStrategy.BEST_EFFORT
    
    # Check steps include parallel markers
    step_types = [s.get("action") for s in composite.steps]
    assert "parallel_start" in step_types
    assert "parallel_merge" in step_types


def test_compose_parallel_strict():
    """Test strict parallel composition."""
    kernel_a = make_kernel("A")
    kernel_b = make_kernel("B")
    
    composite, handler = compose_parallel_strict(kernel_a, kernel_b)
    
    assert handler.strategy == FailureStrategy.ALL_OR_NOTHING
    assert "all branches succeeded" in composite.postconditions


def test_compose_parallel_best_effort():
    """Test best-effort parallel composition."""
    kernel_a = make_kernel("A")
    kernel_b = make_kernel("B")
    
    composite, handler = compose_parallel_best_effort(kernel_a, kernel_b)
    
    assert handler.strategy == FailureStrategy.BEST_EFFORT
    assert "at least one branch succeeded" in composite.postconditions


def test_rollback_chain_built():
    """Test rollback chain is built on failure."""
    handler = ParallelFailureHandler(strategy=FailureStrategy.ALL_OR_NOTHING)
    
    branches = [
        BranchResult("A", "k1", True, {"x": 1}, [], [], 100, rollback_actions=["undo_a"]),
        BranchResult("B", "k2", False, {}, [], ["error"], 100),
    ]
    
    result = handler.aggregate(branches)
    
    assert not result.success
    assert len(result.rollback_chain) > 0
    # Should include rollback for successful branch
    assert any("rollback A" in action or "undo_a" in action for action in result.rollback_chain)


def test_failure_diagnosis():
    """Test failure reason diagnosis."""
    handler = ParallelFailureHandler(strategy=FailureStrategy.ALL_OR_NOTHING)
    
    branches = [
        BranchResult("A", "k1", False, {}, [], ["timeout", "connection refused"], 100),
        BranchResult("B", "k2", False, {}, [], ["invalid input"], 100),
    ]
    
    result = handler.aggregate(branches)
    
    assert result.failure_reason is not None
    # Should mention errors
    assert "timeout" in result.failure_reason or "connection refused" in result.failure_reason
