"""Search task family kernels."""
from kernelweave import Kernel, KernelStatus

CODE_SEARCH_KERNEL = Kernel(
    kernel_id="kw-code-search-001",
    name="Code Search Kernel",
    task_family="code search",
    description="Search codebase for patterns, functions, or concepts",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}, "scope": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"results": {"type": "array"}, "rankings": {"type": "array"}}},
    preconditions=[
        "codebase is indexed or accessible",
        "query is non-empty and specific",
        "scope defines file types or directories",
    ],
    postconditions=[
        "results include file paths and line numbers",
        "results are ranked by relevance",
        "each result has context snippet",
        "total count is accurate",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse query into search terms"},
        {"step": 2, "action": "tool", "tool": "query_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "index_searcher", "args": {}},
        {"step": 4, "action": "tool", "tool": "result_ranker", "args": {}},
        {"step": 5, "action": "evidence", "text": "results found in scope"},
        {"step": 6, "action": "verification", "text": "relevance scores valid"},
    ],
    rollback=["if no index, use grep fallback", "if scope empty, search entire codebase"],
    evidence_requirements=["Search time logged", "Index coverage percentage", "Result deduplication verified"],
    tests=[{"name": "function-search", "input": {"query": "def authenticate", "scope": "*.py"}, "expected": {"min_results": 0}}],
    status=KernelStatus(state="verified", confidence=0.90, failures=0, passes=11),
    source_trace_ids=["trace-code-search-001"],
)

PATTERN_SEARCH_KERNEL = Kernel(
    kernel_id="kw-pattern-search-001",
    name="Pattern Search Kernel",
    task_family="pattern search",
    description="Search for specific code patterns or anti-patterns",
    input_schema={"type": "object", "properties": {"pattern": {"type": "string"}, "type": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"matches": {"type": "array"}, "analysis": {"type": "string"}}},
    preconditions=[
        "pattern is well-defined (regex, AST, semantic)",
        "search type is specified",
        "codebase is parseable",
    ],
    postconditions=[
        "matches include context and explanation",
        "false positive rate is estimated",
        "pattern variants are considered",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Compile pattern to search form"},
        {"step": 2, "action": "tool", "tool": "pattern_compiler", "args": {}},
        {"step": 3, "action": "tool", "tool": "pattern_matcher", "args": {}},
        {"step": 4, "action": "evidence", "text": "all matches captured"},
        {"step": 5, "action": "verification", "text": "no false positives in sample"},
    ],
    rollback=["if pattern invalid, suggest alternatives", "if too many matches, request narrower pattern"],
    evidence_requirements=["Pattern compiled successfully", "Match count by file", "Execution time"],
    tests=[{"name": "todo-search", "input": {"pattern": "TODO|FIXME", "type": "regex"}, "expected": {"matches_format": "list"}}],
    status=KernelStatus(state="verified", confidence=0.87, failures=0, passes=8),
    source_trace_ids=["trace-pattern-search-001"],
)

DEPENDENCY_SEARCH_KERNEL = Kernel(
    kernel_id="kw-dep-search-001",
    name="Dependency Search Kernel",
    task_family="dependency search",
    description="Find and analyze code dependencies",
    input_schema={"type": "object", "properties": {"target": {"type": "string"}, "direction": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"dependencies": {"type": "array"}, "graph": {"type": "object"}}},
    preconditions=[
        "target is a file, module, or package",
        "direction is 'imports' or 'imported_by'",
        "language is supported",
    ],
    postconditions=[
        "dependencies include version info where available",
        "graph shows direct and transitive deps",
        "circular dependencies are flagged",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse import statements"},
        {"step": 2, "action": "tool", "tool": "import_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "dependency_resolver", "args": {}},
        {"step": 4, "action": "tool", "tool": "graph_builder", "args": {}},
        {"step": 5, "action": "evidence", "text": "all imports resolved"},
        {"step": 6, "action": "verification", "text": "no missing dependencies"},
    ],
    rollback=["if unresolved imports, list them", "if circular deps, show cycle"],
    evidence_requirements=["Import count", "Unique dependencies", "External vs internal split"],
    tests=[{"name": "import-search", "input": {"target": "main.py", "direction": "imports"}, "expected": {"has_dependencies": "boolean"}}],
    status=KernelStatus(state="verified", confidence=0.83, failures=0, passes=6),
    source_trace_ids=["trace-dep-search-001"],
)
