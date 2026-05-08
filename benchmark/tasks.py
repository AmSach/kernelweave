"""Real ToolBench-style tasks (not synthetic templates).

These are realistic tasks that match actual ToolBench/AgentBench format:
- Multi-step reasoning required
- Tool use implied
- Clear success criteria
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolBenchTask:
    """A single ToolBench task."""
    task_id: str
    instruction: str
    category: str
    expected_tools: list[str]
    success_criteria: dict[str, Any]
    difficulty: str  # easy, medium, hard
    kernel_family: str | None  # Which kernel family should match (if any)

# 50 realistic tasks across all kernel families
REAL_TOOLBENCH_TASKS = [
    # === ANALYSIS TASKS (10) ===
    ToolBenchTask(
        task_id="tb-analyze-001",
        instruction="Analyze the codebase in /src and identify code quality issues, focusing on complexity and maintainability.",
        category="code analysis",
        expected_tools=["ast_parser", "complexity_analyzer", "pattern_detector"],
        success_criteria={"min_issues": 3, "has_complexity_metrics": True},
        difficulty="medium",
        kernel_family="code analysis",
    ),
    ToolBenchTask(
        task_id="tb-analyze-002",
        instruction="Scan the authentication module for security vulnerabilities according to OWASP Top 10.",
        category="security audit",
        expected_tools=["sql_injection_scanner", "xss_scanner", "auth_checker"],
        success_criteria={"vulnerabilities_found": True, "severity_classified": True},
        difficulty="hard",
        kernel_family="security audit",
    ),
    ToolBenchTask(
        task_id="tb-analyze-003",
        instruction="Profile the data processing pipeline and identify the top 3 performance bottlenecks.",
        category="performance profiling",
        expected_tools=["complexity_analyzer", "memory_profiler"],
        success_criteria={"bottlenecks_identified": 3, "has_time_complexity": True},
        difficulty="medium",
        kernel_family="performance profiling",
    ),
    ToolBenchTask(
        task_id="tb-analyze-004",
        instruction="Review the React components for best practices violations and suggest improvements.",
        category="code analysis",
        expected_tools=["ast_parser", "pattern_detector"],
        success_criteria={"issues_found": True, "has_specific_lines": True},
        difficulty="medium",
        kernel_family="code analysis",
    ),
    ToolBenchTask(
        task_id="tb-analyze-005",
        instruction="Check the API endpoints for security issues, focusing on input validation and authentication.",
        category="security audit",
        expected_tools=["input_validator_checker", "auth_checker"],
        success_criteria={"issues_found": True, "endpoints_covered": True},
        difficulty="hard",
        kernel_family="security audit",
    ),
    ToolBenchTask(
        task_id="tb-analyze-006",
        instruction="Analyze memory usage patterns in the image processing module and suggest optimizations.",
        category="performance profiling",
        expected_tools=["memory_profiler", "complexity_analyzer"],
        success_criteria={"memory_issues_found": True, "has_recommendations": True},
        difficulty="medium",
        kernel_family="performance profiling",
    ),
    ToolBenchTask(
        task_id="tb-analyze-007",
        instruction="Review the database queries for N+1 problems and suggest optimizations.",
        category="performance profiling",
        expected_tools=["query_analyzer", "complexity_analyzer"],
        success_criteria={"n1_issues_found": True, "has_solutions": True},
        difficulty="medium",
        kernel_family="performance profiling",
    ),
    ToolBenchTask(
        task_id="tb-analyze-008",
        instruction="Audit the payment processing code for PCI DSS compliance issues.",
        category="security audit",
        expected_tools=["compliance_checker", "data_flow_analyzer"],
        success_criteria={"compliance_issues_found": True, "pci_relevant": True},
        difficulty="hard",
        kernel_family="security audit",
    ),
    ToolBenchTask(
        task_id="tb-analyze-009",
        instruction="Analyze the code structure and identify functions with high cyclomatic complexity.",
        category="code analysis",
        expected_tools=["ast_parser", "complexity_analyzer"],
        success_criteria={"complex_functions_found": True, "has_metrics": True},
        difficulty="easy",
        kernel_family="code analysis",
    ),
    ToolBenchTask(
        task_id="tb-analyze-010",
        instruction="Identify dead code and unused imports across the project.",
        category="code analysis",
        expected_tools=["unused_detector", "import_analyzer"],
        success_criteria={"dead_code_found": True, "unused_imports_found": True},
        difficulty="easy",
        kernel_family="code analysis",
    ),
    
    # === COMPARISON TASKS (10) ===
    ToolBenchTask(
        task_id="tb-compare-001",
        instruction="Compare main.py and main_refactored.py and summarize the key differences.",
        category="artifact comparison",
        expected_tools=["load_artifact", "structure_comparator", "content_comparator"],
        success_criteria={"both_files_mentioned": True, "differences_categorized": True},
        difficulty="easy",
        kernel_family="artifact comparison",
    ),
    ToolBenchTask(
        task_id="tb-compare-002",
        instruction="Analyze the git diff between feature-branch and main, categorizing changes by type.",
        category="diff analysis",
        expected_tools=["diff_parser", "change_classifier"],
        success_criteria={"changes_categorized": True, "has_line_numbers": True},
        difficulty="medium",
        kernel_family="diff analysis",
    ),
    ToolBenchTask(
        task_id="tb-compare-003",
        instruction="Compare API responses between v1.0 and v2.0, identifying breaking changes.",
        category="version comparison",
        expected_tools=["semver_comparator", "changelog_parser"],
        success_criteria={"breaking_changes_identified": True, "both_versions_mentioned": True},
        difficulty="medium",
        kernel_family="version comparison",
    ),
    ToolBenchTask(
        task_id="tb-compare-004",
        instruction="Compare the schemas of two PostgreSQL databases and list the differences.",
        category="artifact comparison",
        expected_tools=["load_artifact", "structure_comparator"],
        success_criteria={"both_schemas_mentioned": True, "structural_diffs": True},
        difficulty="medium",
        kernel_family="artifact comparison",
    ),
    ToolBenchTask(
        task_id="tb-compare-005",
        instruction="Compare Dockerfile.dev and Dockerfile.prod and explain the differences.",
        category="artifact comparison",
        expected_tools=["load_artifact", "content_comparator"],
        success_criteria={"both_files_mentioned": True, "purpose_diffs": True},
        difficulty="easy",
        kernel_family="artifact comparison",
    ),
    ToolBenchTask(
        task_id="tb-compare-006",
        instruction="Analyze changes between version 2.3.1 and 3.0.0, highlighting migration requirements.",
        category="version comparison",
        expected_tools=["semver_comparator", "changelog_parser"],
        success_criteria={"migration_required": True, "breaking_changes": True},
        difficulty="hard",
        kernel_family="version comparison",
    ),
    ToolBenchTask(
        task_id="tb-compare-007",
        instruction="Compare the test coverage reports before and after the refactoring.",
        category="artifact comparison",
        expected_tools=["load_artifact", "content_comparator"],
        success_criteria={"both_reports_mentioned": True, "coverage_diff": True},
        difficulty="easy",
        kernel_family="artifact comparison",
    ),
    ToolBenchTask(
        task_id="tb-compare-008",
        instruction="Diff the config files between staging and production environments.",
        category="diff analysis",
        expected_tools=["diff_parser", "change_classifier"],
        success_criteria={"env_diffs_identified": True, "security_notes": True},
        difficulty="medium",
        kernel_family="diff analysis",
    ),
    ToolBenchTask(
        task_id="tb-compare-009",
        instruction="Compare package.json between the frontend and backend projects.",
        category="artifact comparison",
        expected_tools=["load_artifact", "structure_comparator"],
        success_criteria={"both_files_mentioned": True, "dependency_diffs": True},
        difficulty="easy",
        kernel_family="artifact comparison",
    ),
    ToolBenchTask(
        task_id="tb-compare-010",
        instruction="Compare the OpenAPI specs of two microservices and identify incompatible endpoints.",
        category="version comparison",
        expected_tools=["spec_parser", "semver_comparator"],
        success_criteria={"incompatible_endpoints": True, "both_services_mentioned": True},
        difficulty="hard",
        kernel_family="version comparison",
    ),
    
    # === GENERATION TASKS (10) ===
    ToolBenchTask(
        task_id="tb-gen-001",
        instruction="Generate a Python function that validates email addresses with comprehensive edge case handling.",
        category="code generation",
        expected_tools=["spec_parser", "code_generator", "syntax_checker"],
        success_criteria={"valid_python": True, "handles_edge_cases": True},
        difficulty="medium",
        kernel_family="code generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-002",
        instruction="Generate unit tests for the calculate_tax function, including edge cases.",
        category="test generation",
        expected_tools=["function_extractor", "edge_case_analyzer", "test_generator"],
        success_criteria={"tests_runnable": True, "edge_cases_covered": True},
        difficulty="medium",
        kernel_family="test generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-003",
        instruction="Generate API documentation for the user service endpoints in markdown format.",
        category="documentation generation",
        expected_tools=["docstring_extractor", "example_generator", "doc_formatter"],
        success_criteria={"all_endpoints_documented": True, "markdown_format": True},
        difficulty="medium",
        kernel_family="documentation generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-004",
        instruction="Generate a Docker Compose configuration for a web app with PostgreSQL and Redis.",
        category="config generation",
        expected_tools=["schema_resolver", "config_generator", "config_validator"],
        success_criteria={"valid_yaml": True, "all_services_defined": True},
        difficulty="medium",
        kernel_family="config generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-005",
        instruction="Generate a TypeScript interface for the User API response schema.",
        category="code generation",
        expected_tools=["spec_parser", "code_generator", "syntax_checker"],
        success_criteria={"valid_typescript": True, "all_fields_covered": True},
        difficulty="easy",
        kernel_family="code generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-006",
        instruction="Generate integration tests for the payment processing flow.",
        category="test generation",
        expected_tools=["function_extractor", "test_generator"],
        success_criteria={"tests_runnable": True, "flow_covered": True},
        difficulty="hard",
        kernel_family="test generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-007",
        instruction="Generate a README for the authentication library with installation and usage examples.",
        category="documentation generation",
        expected_tools=["structure_analyzer", "metadata_extractor", "readme_generator"],
        success_criteria={"has_install_section": True, "has_examples": True},
        difficulty="medium",
        kernel_family="documentation generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-008",
        instruction="Generate a GitHub Actions workflow for CI/CD with testing and deployment stages.",
        category="config generation",
        expected_tools=["schema_resolver", "config_generator"],
        success_criteria={"valid_yaml": True, "all_stages_defined": True},
        difficulty="medium",
        kernel_family="config generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-009",
        instruction="Generate a GraphQL schema for a blog API with users, posts, and comments.",
        category="code generation",
        expected_tools=["spec_parser", "code_generator"],
        success_criteria={"valid_graphql": True, "all_types_defined": True},
        difficulty="medium",
        kernel_family="code generation",
    ),
    ToolBenchTask(
        task_id="tb-gen-010",
        instruction="Generate JSDoc comments for all exported functions in the utils module.",
        category="documentation generation",
        expected_tools=["docstring_extractor", "doc_formatter"],
        success_criteria={"all_functions_documented": True, "jsdoc_format": True},
        difficulty="easy",
        kernel_family="documentation generation",
    ),
    
    # === SEARCH TASKS (8) ===
    ToolBenchTask(
        task_id="tb-search-001",
        instruction="Search the codebase for all uses of deprecated functions.",
        category="code search",
        expected_tools=["query_parser", "index_searcher", "result_ranker"],
        success_criteria={"results_with_locations": True, "ranked_by_relevance": True},
        difficulty="medium",
        kernel_family="code search",
    ),
    ToolBenchTask(
        task_id="tb-search-002",
        instruction="Find all TODO and FIXME comments across the project.",
        category="pattern search",
        expected_tools=["pattern_compiler", "pattern_matcher"],
        success_criteria={"all_todos_found": True, "with_context": True},
        difficulty="easy",
        kernel_family="pattern search",
    ),
    ToolBenchTask(
        task_id="tb-search-003",
        instruction="Find all files that import the deprecated 'request' library.",
        category="dependency search",
        expected_tools=["import_parser", "dependency_resolver"],
        success_criteria={"all_importers_found": True, "has_file_paths": True},
        difficulty="easy",
        kernel_family="dependency search",
    ),
    ToolBenchTask(
        task_id="tb-search-004",
        instruction="Search for SQL queries that might be vulnerable to injection.",
        category="pattern search",
        expected_tools=["pattern_compiler", "pattern_matcher"],
        success_criteria={"vulnerable_queries_found": True, "has_line_numbers": True},
        difficulty="medium",
        kernel_family="pattern search",
    ),
    ToolBenchTask(
        task_id="tb-search-005",
        instruction="Find all async functions that don't have proper error handling.",
        category="code search",
        expected_tools=["query_parser", "index_searcher"],
        success_criteria={"missing_handlers_found": True, "with_examples": True},
        difficulty="medium",
        kernel_family="code search",
    ),
    ToolBenchTask(
        task_id="tb-search-006",
        instruction="Identify all direct dependencies of the auth module.",
        category="dependency search",
        expected_tools=["import_parser", "dependency_resolver"],
        success_criteria={"dependencies_listed": True, "has_versions": True},
        difficulty="easy",
        kernel_family="dependency search",
    ),
    ToolBenchTask(
        task_id="tb-search-007",
        instruction="Find all uses of 'eval()' or 'exec()' in the codebase (security risk).",
        category="pattern search",
        expected_tools=["pattern_compiler", "pattern_matcher"],
        success_criteria={"all_uses_found": True, "security_risk_noted": True},
        difficulty="easy",
        kernel_family="pattern search",
    ),
    ToolBenchTask(
        task_id="tb-search-008",
        instruction="Search for functions longer than 50 lines.",
        category="code search",
        expected_tools=["query_parser", "index_searcher"],
        success_criteria={"long_functions_found": True, "with_line_counts": True},
        difficulty="easy",
        kernel_family="code search",
    ),
    
    # === TRANSFORMATION TASKS (6) ===
    ToolBenchTask(
        task_id="tb-transform-001",
        instruction="Convert the CSV data file to JSON format.",
        category="format conversion",
        expected_tools=["format_detector", "parser", "serializer"],
        success_criteria={"valid_json": True, "data_preserved": True},
        difficulty="easy",
        kernel_family="format conversion",
    ),
    ToolBenchTask(
        task_id="tb-transform-002",
        instruction="Refactor the 'process_data' function to use async/await instead of callbacks.",
        category="refactoring",
        expected_tools=["ast_parser", "refactoring_engine", "test_runner"],
        success_criteria={"behavior_preserved": True, "uses_async": True},
        difficulty="medium",
        kernel_family="refactoring",
    ),
    ToolBenchTask(
        task_id="tb-transform-003",
        instruction="Migrate the Express.js routes to Fastify format.",
        category="migration",
        expected_tools=["version_detector", "migration_mapper", "code_transformer"],
        success_criteria={"fastify_format": True, "behavior_preserved": True},
        difficulty="hard",
        kernel_family="migration",
    ),
    ToolBenchTask(
        task_id="tb-transform-004",
        instruction="Convert the XML configuration to YAML.",
        category="format conversion",
        expected_tools=["format_detector", "parser", "serializer"],
        success_criteria={"valid_yaml": True, "structure_preserved": True},
        difficulty="easy",
        kernel_family="format conversion",
    ),
    ToolBenchTask(
        task_id="tb-transform-005",
        instruction="Rename the variable 'data' to 'userData' throughout the user module.",
        category="refactoring",
        expected_tools=["ast_parser", "refactoring_engine"],
        success_criteria={"renamed_correctly": True, "no_missed_occurrences": True},
        difficulty="medium",
        kernel_family="refactoring",
    ),
    ToolBenchTask(
        task_id="tb-transform-006",
        instruction="Migrate the Python 2.7 code to Python 3.",
        category="migration",
        expected_tools=["version_detector", "migration_mapper", "code_transformer"],
        success_criteria={"python3_compatible": True, "print_fixed": True},
        difficulty="medium",
        kernel_family="migration",
    ),
    
    # === DEBUGGING TASKS (4) ===
    ToolBenchTask(
        task_id="tb-debug-001",
        instruction="Diagnose this error: 'TypeError: Cannot read property \"id\" of undefined' in user.js line 42.",
        category="error diagnosis",
        expected_tools=["trace_parser", "error_matcher", "code_inspector"],
        success_criteria={"root_cause_found": True, "fix_suggested": True},
        difficulty="medium",
        kernel_family="error diagnosis",
    ),
    ToolBenchTask(
        task_id="tb-debug-002",
        instruction="Analyze the application logs from the last hour and identify any anomalies.",
        category="log analysis",
        expected_tools=["log_parser", "pattern_detector", "anomaly_detector"],
        success_criteria={"anomalies_found": True, "timeline_provided": True},
        difficulty="medium",
        kernel_family="log analysis",
    ),
    ToolBenchTask(
        task_id="tb-debug-003",
        instruction="Debug the failing test: 'Expected 200 but got 500' in the login endpoint test.",
        category="error diagnosis",
        expected_tools=["test_parser", "assertion_analyzer", "code_diff"],
        success_criteria={"failure_reason_found": True, "fix_suggested": True},
        difficulty="medium",
        kernel_family="error diagnosis",
    ),
    ToolBenchTask(
        task_id="tb-debug-004",
        instruction="Find the cause of the memory leak from the provided heap dump logs.",
        category="log analysis",
        expected_tools=["log_parser", "pattern_detector"],
        success_criteria={"leak_source_found": True, "fix_suggested": True},
        difficulty="hard",
        kernel_family="log analysis",
    ),
    
    # === TESTING & DOCUMENTATION TASKS (4) ===
    ToolBenchTask(
        task_id="tb-test-001",
        instruction="Debug the flaky test in test_auth.py that fails intermittently.",
        category="test debugging",
        expected_tools=["test_parser", "assertion_analyzer"],
        success_criteria={"flakiness_cause_found": True, "fix_suggested": True},
        difficulty="medium",
        kernel_family="test debugging",
    ),
    ToolBenchTask(
        task_id="tb-test-002",
        instruction="Analyze test coverage and identify the most critical uncovered paths.",
        category="coverage analysis",
        expected_tools=["coverage_parser", "gap_analyzer", "priority_ranker"],
        success_criteria={"gaps_identified": True, "prioritized": True},
        difficulty="medium",
        kernel_family="coverage analysis",
    ),
    ToolBenchTask(
        task_id="tb-doc-001",
        instruction="Generate OpenAPI documentation for the REST API from the Express route handlers.",
        category="api documentation",
        expected_tools=["spec_parser", "example_generator", "doc_formatter"],
        success_criteria={"all_endpoints_documented": True, "openapi_format": True},
        difficulty="medium",
        kernel_family="api documentation",
    ),
    ToolBenchTask(
        task_id="tb-doc-002",
        instruction="Create a comprehensive README for the CLI tool with installation, usage, and examples.",
        category="readme generation",
        expected_tools=["structure_analyzer", "metadata_extractor", "readme_generator"],
        success_criteria={"has_install": True, "has_usage": True, "has_examples": True},
        difficulty="medium",
        kernel_family="readme generation",
    ),
]


def get_tasks_by_category(category: str) -> list[ToolBenchTask]:
    """Filter tasks by category."""
    return [t for t in REAL_TOOLBENCH_TASKS if t.category == category]


def get_tasks_by_difficulty(difficulty: str) -> list[ToolBenchTask]:
    """Filter tasks by difficulty."""
    return [t for t in REAL_TOOLBENCH_TASKS if t.difficulty == difficulty]


def get_tasks_for_kernel(kernel_family: str) -> list[ToolBenchTask]:
    """Get all tasks that should match a specific kernel family."""
    return [t for t in REAL_TOOLBENCH_TASKS if t.kernel_family == kernel_family]


def task_summary() -> dict:
    """Summary of the task library."""
    categories = {}
    difficulties = {}
    kernel_families = {}
    
    for t in REAL_TOOLBENCH_TASKS:
        categories[t.category] = categories.get(t.category, 0) + 1
        difficulties[t.difficulty] = difficulties.get(t.difficulty, 0) + 1
        if t.kernel_family:
            kernel_families[t.kernel_family] = kernel_families.get(t.kernel_family, 0) + 1
    
    return {
        "total_tasks": len(REAL_TOOLBENCH_TASKS),
        "categories": categories,
        "difficulties": difficulties,
        "kernel_families": kernel_families,
    }
