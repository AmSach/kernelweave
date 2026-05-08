"""Comparison task family kernels."""
from kernelweave import Kernel, KernelStatus, TraceEvent

ARTIFACT_COMPARISON_KERNEL = Kernel(
    kernel_id="kw-artifact-compare-001",
    name="Artifact Comparison Kernel",
    task_family="artifact comparison",
    description="Compare two structured artifacts and produce a grounded summary",
    input_schema={"type": "object", "properties": {"artifact_a": {"type": "string"}, "artifact_b": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"differences": {"type": "array"}, "summary": {"type": "string"}}},
    preconditions=[
        "inputs are named files, schemas, or documents",
        "both artifacts are accessible and parseable",
        "comparison scope is defined",
        "artifacts are of compatible types",
    ],
    postconditions=[
        "output mentions both artifacts by name",
        "differences are categorized by type",
        "summary includes both structural and content changes",
        "no rollback triggered",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Load and parse both artifacts"},
        {"step": 2, "action": "tool", "tool": "load_artifact", "args": {"path": "A"}},
        {"step": 3, "action": "tool", "tool": "load_artifact", "args": {"path": "B"}},
        {"step": 4, "action": "tool", "tool": "structure_comparator", "args": {}},
        {"step": 5, "action": "tool", "tool": "content_comparator", "args": {}},
        {"step": 6, "action": "evidence", "text": "differences catalogued"},
        {"step": 7, "action": "verification", "text": "summary references both"},
    ],
    rollback=["if artifacts incompatible, suggest conversion", "if one missing, report error"],
    evidence_requirements=["both artifacts loaded", "diff metrics computed", "examples of changes"],
    tests=[{"name": "file-compare", "input": {"artifact_a": "v1.py", "artifact_b": "v2.py"}, "expected": {"min_differences": 0}}],
    status=KernelStatus(state="verified", confidence=0.88, failures=0, passes=12),
    source_trace_ids=["trace-artifact-compare-001"],
)

DIFF_ANALYSIS_KERNEL = Kernel(
    kernel_id="kw-diff-analysis-001",
    name="Diff Analysis Kernel",
    task_family="diff analysis",
    description="Analyze git diffs and explain code changes",
    input_schema={"type": "object", "properties": {"diff": {"type": "string"}, "context": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"changes": {"type": "array"}, "intent": {"type": "string"}}},
    preconditions=[
        "diff is in unified diff format",
        "context includes commit message or PR description",
        "files are recognizable source code",
    ],
    postconditions=[
        "each change has before/after snippets",
        "intent is inferred from diff patterns",
        "output identifies refactoring vs feature vs fix",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse unified diff format"},
        {"step": 2, "action": "tool", "tool": "diff_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "change_classifier", "args": {}},
        {"step": 4, "action": "evidence", "text": "hunks counted"},
        {"step": 5, "action": "verification", "text": "classification matches context"},
    ],
    rollback=["if diff malformed, request clean diff", "if binary files, skip analysis"],
    evidence_requirements=["Line counts per file", "Change types identified", "Risk assessment for large changes"],
    tests=[{"name": "simple-diff", "input": {"diff": "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@"}, "expected": {"changes_count": 1}}],
    status=KernelStatus(state="verified", confidence=0.86, failures=0, passes=6),
    source_trace_ids=["trace-diff-analysis-001"],
)

VERSION_COMPARISON_KERNEL = Kernel(
    kernel_id="kw-version-compare-001",
    name="Version Comparison Kernel",
    task_family="version comparison",
    description="Compare versions, releases, or snapshots across time",
    input_schema={"type": "object", "properties": {"version_a": {"type": "string"}, "version_b": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"changes": {"type": "array"}, "breaking": {"type": "boolean"}}},
    preconditions=[
        "versions follow semantic versioning or date format",
        "changelog or release notes available",
        "comparison is forward in time (A < B)",
    ],
    postconditions=[
        "breaking changes are clearly marked",
        "output includes version numbers",
        "changes are grouped by category",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Fetch release notes for both versions"},
        {"step": 2, "action": "tool", "tool": "changelog_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "semver_comparator", "args": {}},
        {"step": 4, "action": "evidence", "text": "breaking changes identified"},
        {"step": 5, "action": "verification", "text": "all sections compared"},
    ],
    rollback=["if versions unreleased, compare from source", "if changelog missing, generate from commits"],
    evidence_requirements=["Semver distance computed", "Breaking changes listed", "Migration notes included"],
    tests=[{"name": "semver-compare", "input": {"version_a": "1.0.0", "version_b": "2.0.0"}, "expected": {"breaking": True}}],
    status=KernelStatus(state="verified", confidence=0.84, failures=0, passes=5),
    source_trace_ids=["trace-version-compare-001"],
)
