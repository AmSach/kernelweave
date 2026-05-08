"""Debugging task family kernels."""
from kernelweave import Kernel, KernelStatus

ERROR_DIAGNOSIS_KERNEL = Kernel(
    kernel_id="kw-error-diagnose-001",
    name="Error Diagnosis Kernel",
    task_family="error diagnosis",
    description="Diagnose errors from stack traces, logs, or behavior",
    input_schema={"type": "object", "properties": {"error": {"type": "string"}, "context": {"type": "object"}}},
    output_schema={"type": "object", "properties": {"diagnosis": {"type": "string"}, "fix_suggestions": {"type": "array"}}},
    preconditions=[
        "error message or stack trace is provided",
        "context includes relevant code",
        "language/runtime is identifiable",
    ],
    postconditions=[
        "root cause is identified or hypotheses given",
        "fix suggestions are actionable",
        "output mentions specific lines/files",
        "severity is assessed",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse error and extract key information"},
        {"step": 2, "action": "tool", "tool": "trace_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "error_matcher", "args": {}},
        {"step": 4, "action": "tool", "tool": "code_inspector", "args": {}},
        {"step": 5, "action": "evidence", "text": "error pattern matched"},
        {"step": 6, "action": "verification", "text": "diagnosis consistent with trace"},
    ],
    rollback=["if error unknown, search documentation", "if context insufficient, request more info"],
    evidence_requirements=["Error type classified", "Similar issues found", "Fix success rate from history"],
    tests=[{"name": "null-pointer", "input": {"error": "NullPointerException at line 42"}, "expected": {"diagnosis_type": "null_access"}}],
    status=KernelStatus(state="verified", confidence=0.88, failures=0, passes=9),
    source_trace_ids=["trace-error-diagnose-001"],
)

LOG_ANALYSIS_KERNEL = Kernel(
    kernel_id="kw-log-analysis-001",
    name="Log Analysis Kernel",
    task_family="log analysis",
    description="Analyze logs to find patterns, anomalies, or root causes",
    input_schema={"type": "object", "properties": {"logs": {"type": "string"}, "query": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"findings": {"type": "array"}, "timeline": {"type": "array"}}},
    preconditions=[
        "logs are in recognizable format",
        "time range is specified or inferred",
        "log source is known",
    ],
    postconditions=[
        "findings are time-ordered",
        "anomalies are highlighted",
        "output includes log snippets",
        "correlations are identified",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse and normalize log format"},
        {"step": 2, "action": "tool", "tool": "log_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "pattern_detector", "args": {}},
        {"step": 4, "action": "tool", "tool": "anomaly_detector", "args": {}},
        {"step": 5, "action": "evidence", "text": "patterns extracted"},
        {"step": 6, "action": "verification", "text": "findings supported by logs"},
    ],
    rollback=["if format unknown, try common formats", "if logs corrupted, report lines skipped"],
    evidence_requirements=["Log lines processed", "Error rate computed", "Anomaly score"],
    tests=[{"name": "error-spike", "input": {"logs": "ERROR at 10:00\nERROR at 10:01\nERROR at 10:02"}, "expected": {"has_findings": True}}],
    status=KernelStatus(state="verified", confidence=0.85, failures=0, passes=7),
    source_trace_ids=["trace-log-analysis-001"],
)
