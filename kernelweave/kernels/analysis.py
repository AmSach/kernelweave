"""Analysis task family kernels."""
from kernelweave import Kernel, KernelStatus, TraceEvent

CODE_ANALYSIS_KERNEL = Kernel(
    kernel_id="kw-code-analysis-001",
    name="Code Analysis Kernel",
    task_family="code analysis",
    description="Analyze code structure, patterns, and quality",
    input_schema={"type": "object", "properties": {"code": {"type": "string"}, "language": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"analysis": {"type": "object"}, "score": {"type": "number"}}},
    preconditions=[
        "code is valid syntax in the specified language",
        "language is one of: python, javascript, typescript, java, go, rust",
        "code length is reasonable (< 10,000 lines)",
        "analysis focuses on structure, not runtime behavior",
    ],
    postconditions=[
        "analysis includes complexity metrics",
        "analysis identifies potential issues",
        "output mentions specific code sections",
        "score is between 0 and 100",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse code into AST"},
        {"step": 2, "action": "tool", "tool": "ast_parser", "args": {"language": "auto"}},
        {"step": 3, "action": "tool", "tool": "complexity_analyzer", "args": {}},
        {"step": 4, "action": "tool", "tool": "pattern_detector", "args": {}},
        {"step": 5, "action": "evidence", "text": "complexity score computed"},
        {"step": 6, "action": "verification", "text": "all sections covered"},
    ],
    rollback=["if parsing fails, report syntax error", "if language unsupported, fallback to text analysis"],
    evidence_requirements=["AST generated", "complexity metrics calculated", "patterns identified"],
    tests=[{"name": "python-analysis", "input": {"language": "python"}, "expected": {"score_range": [0, 100]}}],
    status=KernelStatus(state="verified", confidence=0.85, failures=0, passes=5),
    source_trace_ids=["trace-code-analysis-001"],
)

SECURITY_AUDIT_KERNEL = Kernel(
    kernel_id="kw-security-audit-001",
    name="Security Audit Kernel",
    task_family="security audit",
    description="Identify security vulnerabilities and risks in code",
    input_schema={"type": "object", "properties": {"code": {"type": "string"}, "context": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"vulnerabilities": {"type": "array"}, "risk_level": {"type": "string"}}},
    preconditions=[
        "code is accessible for analysis",
        "audit scope is clearly defined",
        "context includes deployment environment",
        "security checklist is available",
    ],
    postconditions=[
        "vulnerabilities are categorized by severity",
        "each vulnerability has remediation suggestion",
        "risk level is one of: low, medium, high, critical",
        "no false positives in critical findings",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Scan for known vulnerability patterns"},
        {"step": 2, "action": "tool", "tool": "sql_injection_scanner", "args": {}},
        {"step": 3, "action": "tool", "tool": "xss_scanner", "args": {}},
        {"step": 4, "action": "tool", "tool": "auth_checker", "args": {}},
        {"step": 5, "action": "evidence", "text": "each finding has line numbers"},
        {"step": 6, "action": "verification", "text": "severity classification correct"},
    ],
    rollback=["if code is minified, request source", "if dependencies unknown, skip transitive analysis"],
    evidence_requirements=["OWASP Top 10 coverage", "line numbers for all findings", "CWE references where applicable"],
    tests=[{"name": "sql-injection", "input": {"code": "SELECT * FROM users WHERE id="}, "expected": {"vulnerabilities_count": ">= 1"}}],
    status=KernelStatus(state="verified", confidence=0.90, failures=0, passes=8),
    source_trace_ids=["trace-security-audit-001"],
)

PERFORMANCE_PROFILING_KERNEL = Kernel(
    kernel_id="kw-perf-profile-001",
    name="Performance Profiling Kernel",
    task_family="performance profiling",
    description="Identify performance bottlenecks and optimization opportunities",
    input_schema={"type": "object", "properties": {"code": {"type": "string"}, "runtime_data": {"type": "object"}}},
    output_schema={"type": "object", "properties": {"bottlenecks": {"type": "array"}, "recommendations": {"type": "array"}}},
    preconditions=[
        "code is executable or has runtime traces",
        "performance requirements are specified",
        "measurement context is available",
    ],
    postconditions=[
        "bottlenecks have time complexity analysis",
        "recommendations are prioritized by impact",
        "output includes estimated speedup",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Identify hot paths"},
        {"step": 2, "action": "tool", "tool": "complexity_analyzer", "args": {"mode": "time"}},
        {"step": 3, "action": "tool", "tool": "memory_profiler", "args": {}},
        {"step": 4, "action": "evidence", "text": "profile data captured"},
        {"step": 5, "action": "verification", "text": "recommendations tested"},
    ],
    rollback=["if no runtime data, use static analysis only", "if profiling unavailable, estimate from complexity"],
    evidence_requirements=["Time complexity for hot paths", "Memory allocation patterns", "IO operation counts"],
    tests=[{"name": "loop-analysis", "input": {"code": "for i in range(n): for j in range(n):"}, "expected": {"complexity": "O(n^2)"}}],
    status=KernelStatus(state="verified", confidence=0.82, failures=0, passes=4),
    source_trace_ids=["trace-perf-profile-001"],
)
