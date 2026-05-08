"""Testing task family kernels."""
from kernelweave import Kernel, KernelStatus

TEST_DEBUGGING_KERNEL = Kernel(
    kernel_id="kw-test-debug-001",
    name="Test Debugging Kernel",
    task_family="test debugging",
    description="Debug failing tests and identify root causes",
    input_schema={"type": "object", "properties": {"test_output": {"type": "string"}, "code": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"failure_reason": {"type": "string"}, "fix": {"type": "string"}}},
    preconditions=[
        "test failure output is provided",
        "test code is accessible",
        "code under test is accessible",
    ],
    postconditions=[
        "failure reason is specific",
        "fix suggestion targets actual cause",
        "output mentions test name and assertion",
        "expected vs actual is shown",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse test failure output"},
        {"step": 2, "action": "tool", "tool": "test_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "assertion_analyzer", "args": {}},
        {"step": 4, "action": "tool", "tool": "code_diff", "args": {}},
        {"step": 5, "action": "evidence", "text": "assertion details captured"},
        {"step": 6, "action": "verification", "text": "fix resolves failure"},
    ],
    rollback=["if test flaky, report flakiness", "if assertion unclear, show context"],
    evidence_requirements=["Test name", "Assertion type", "Expected vs actual values"],
    tests=[{"name": "assertion-failure", "input": {"test_output": "AssertionError: 1 != 2"}, "expected": {"has_failure_reason": True}}],
    status=KernelStatus(state="verified", confidence=0.86, failures=0, passes=8),
    source_trace_ids=["trace-test-debug-001"],
)

COVERAGE_ANALYSIS_KERNEL = Kernel(
    kernel_id="kw-coverage-analysis-001",
    name="Coverage Analysis Kernel",
    task_family="coverage analysis",
    description="Analyze test coverage and identify gaps",
    input_schema={"type": "object", "properties": {"code": {"type": "string"}, "coverage_report": {"type": "object"}}},
    output_schema={"type": "object", "properties": {"uncovered": {"type": "array"}, "recommendations": {"type": "array"}}},
    preconditions=[
        "coverage report is available",
        "code is instrumented or coverage collected",
        "minimum coverage threshold is defined",
    ],
    postconditions=[
        "uncovered lines are listed",
        "recommendations target specific gaps",
        "output includes coverage percentage",
        "critical paths are prioritized",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse coverage report"},
        {"step": 2, "action": "tool", "tool": "coverage_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "gap_analyzer", "args": {}},
        {"step": 4, "action": "tool", "tool": "priority_ranker", "args": {}},
        {"step": 5, "action": "evidence", "text": "gaps identified"},
        {"step": 6, "action": "verification", "text": "recommendations address gaps"},
    ],
    rollback=["if no coverage report, suggest running tests with coverage", "if format unknown, try common formats"],
    evidence_requirements=["Line coverage percentage", "Branch coverage if available", "Uncovered functions"],
    tests=[{"name": "low-coverage", "input": {"coverage_report": {"percentage": 50}}, "expected": {"has_recommendations": True}}],
    status=KernelStatus(state="verified", confidence=0.83, failures=0, passes=6),
    source_trace_ids=["trace-coverage-analysis-001"],
)
