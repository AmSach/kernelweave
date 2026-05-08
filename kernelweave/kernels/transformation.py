"""Transformation task family kernels."""
from kernelweave import Kernel, KernelStatus

FORMAT_CONVERSION_KERNEL = Kernel(
    kernel_id="kw-format-convert-001",
    name="Format Conversion Kernel",
    task_family="format conversion",
    description="Convert data between formats (JSON, YAML, CSV, XML, etc.)",
    input_schema={"type": "object", "properties": {"data": {"type": "string"}, "from_format": {"type": "string"}, "to_format": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"converted": {"type": "string"}, "lossy": {"type": "boolean"}}},
    preconditions=[
        "source format is detected or specified",
        "target format supports data structure",
        "data is valid in source format",
    ],
    postconditions=[
        "converted data is valid in target format",
        "lossy conversion is flagged",
        "structure is preserved where possible",
        "no data corruption",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse source format into intermediate representation"},
        {"step": 2, "action": "tool", "tool": "format_detector", "args": {}},
        {"step": 3, "action": "tool", "tool": "parser", "args": {"format": "auto"}},
        {"step": 4, "action": "tool", "tool": "serializer", "args": {"format": "target"}},
        {"step": 5, "action": "evidence", "text": "roundtrip successful"},
        {"step": 6, "action": "verification", "text": "output valid"},
    ],
    rollback=["if format unsupported, suggest alternatives", "if data malformed, report errors"],
    evidence_requirements=["Source format confidence", "Target format validation", "Data loss warnings"],
    tests=[{"name": "json-to-yaml", "input": {"data": '{"a": 1}', "from_format": "json", "to_format": "yaml"}, "expected": {"valid_yaml": True}}],
    status=KernelStatus(state="verified", confidence=0.92, failures=0, passes=14),
    source_trace_ids=["trace-format-convert-001"],
)

REFACTORING_KERNEL = Kernel(
    kernel_id="kw-refactor-001",
    name="Refactoring Kernel",
    task_family="refactoring",
    description="Apply automated refactorings to code",
    input_schema={"type": "object", "properties": {"code": {"type": "string"}, "refactoring": {"type": "string"}, "target": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"refactored": {"type": "string"}, "changes": {"type": "array"}}},
    preconditions=[
        "code is syntactically valid",
        "refactoring type is supported",
        "target (function, variable, etc.) is specified",
        "tests exist or are generated",
    ],
    postconditions=[
        "behavior is preserved (tests pass)",
        "code style is maintained",
        "changes are minimal and correct",
        "no syntax errors",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Identify refactoring scope"},
        {"step": 2, "action": "tool", "tool": "ast_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "refactoring_engine", "args": {}},
        {"step": 4, "action": "tool", "tool": "test_runner", "args": {}},
        {"step": 5, "action": "evidence", "text": "tests pass after refactor"},
        {"step": 6, "action": "verification", "text": "no behavioral changes"},
    ],
    rollback=["if tests fail, show diff and rollback", "if refactoring unsupported, suggest manual approach"],
    evidence_requirements=["Change count", "Test results before/after", "Complexity change"],
    tests=[{"name": "rename-var", "input": {"code": "x = 1", "refactoring": "rename", "target": "x->y"}, "expected": {"refactored_contains": "y = 1"}}],
    status=KernelStatus(state="verified", confidence=0.80, failures=1, passes=8),
    source_trace_ids=["trace-refactor-001"],
)

MIGRATION_KERNEL = Kernel(
    kernel_id="kw-migrate-001",
    name="Migration Kernel",
    task_family="migration",
    description="Migrate code between versions, frameworks, or platforms",
    input_schema={"type": "object", "properties": {"code": {"type": "string"}, "from": {"type": "string"}, "to": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"migrated": {"type": "string"}, "breaking_changes": {"type": "array"}}},
    preconditions=[
        "source version/framework is identified",
        "target version/framework is specified",
        "migration path is known",
        "backup or version control exists",
    ],
    postconditions=[
        "migrated code is valid in target",
        "breaking changes are documented",
        "migration guide is provided",
        "tests pass after migration",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Analyze migration requirements"},
        {"step": 2, "action": "tool", "tool": "version_detector", "args": {}},
        {"step": 3, "action": "tool", "tool": "migration_mapper", "args": {}},
        {"step": 4, "action": "tool", "tool": "code_transformer", "args": {}},
        {"step": 5, "action": "evidence", "text": "all known patterns migrated"},
        {"step": 6, "action": "verification", "text": "syntax valid in target"},
    ],
    rollback=["if migration fails, show partial results", "if target unsupported, suggest alternatives"],
    evidence_requirements=["Changes categorized by type", "Manual steps identified", "Risk assessment"],
    tests=[{"name": "python2-to-3", "input": {"code": "print 'hello'", "from": "python2", "to": "python3"}, "expected": {"migrated_contains": "print("}}],
    status=KernelStatus(state="verified", confidence=0.75, failures=2, passes=5),
    source_trace_ids=["trace-migrate-001"],
)
