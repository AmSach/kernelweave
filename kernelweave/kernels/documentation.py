"""Documentation task family kernels."""
from kernelweave import Kernel, KernelStatus

API_DOCS_KERNEL = Kernel(
    kernel_id="kw-api-docs-001",
    name="API Documentation Kernel",
    task_family="api documentation",
    description="Generate API documentation from code or specs",
    input_schema={"type": "object", "properties": {"api_spec": {"type": "string"}, "format": {"type": "string"}}},
    output_schema={"type": "object", "properties": {"docs": {"type": "string"}, "examples": {"type": "array"}}},
    preconditions=[
        "API spec is provided (OpenAPI, GraphQL, etc.) or extractable",
        "target format is specified",
        "authentication requirements are known",
    ],
    postconditions=[
        "all endpoints are documented",
        "request/response examples are included",
        "authentication is explained",
        "format matches specification",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Parse API specification"},
        {"step": 2, "action": "tool", "tool": "spec_parser", "args": {}},
        {"step": 3, "action": "tool", "tool": "example_generator", "args": {}},
        {"step": 4, "action": "tool", "tool": "doc_formatter", "args": {}},
        {"step": 5, "action": "evidence", "text": "all endpoints covered"},
        {"step": 6, "action": "verification", "text": "examples are valid"},
    ],
    rollback=["if spec incomplete, document available parts", "if format unknown, use markdown"],
    evidence_requirements=["Endpoint count", "Example coverage", "Auth method documented"],
    tests=[{"name": "openapi-docs", "input": {"api_spec": "openapi: 3.0", "format": "markdown"}, "expected": {"has_endpoints": True}}],
    status=KernelStatus(state="verified", confidence=0.87, failures=0, passes=7),
    source_trace_ids=["trace-api-docs-001"],
)

README_GENERATION_KERNEL = Kernel(
    kernel_id="kw-readme-gen-001",
    name="README Generation Kernel",
    task_family="readme generation",
    description="Generate README files from project structure and metadata",
    input_schema={"type": "object", "properties": {"project_path": {"type": "string"}, "metadata": {"type": "object"}}},
    output_schema={"type": "object", "properties": {"readme": {"type": "string"}, "sections": {"type": "array"}}},
    preconditions=[
        "project has recognizable structure",
        "package metadata exists (package.json, setup.py, etc.)",
        "entry points are identifiable",
    ],
    postconditions=[
        "sections include: install, usage, api, contributing",
        "installation instructions are accurate",
        "examples are runnable",
        "license is correctly identified",
    ],
    steps=[
        {"step": 1, "action": "plan", "text": "Scan project structure"},
        {"step": 2, "action": "tool", "tool": "structure_analyzer", "args": {}},
        {"step": 3, "action": "tool", "tool": "metadata_extractor", "args": {}},
        {"step": 4, "action": "tool", "tool": "readme_generator", "args": {}},
        {"step": 5, "action": "evidence", "text": "all sections generated"},
        {"step": 6, "action": "verification", "text": "commands are valid"},
    ],
    rollback=["if no metadata, generate from structure", "if unclear purpose, request description"],
    evidence_requirements=["Install command verified", "Usage examples tested", "License detected"],
    tests=[{"name": "python-readme", "input": {"project_path": "/tmp/project"}, "expected": {"has_install_section": True}}],
    status=KernelStatus(state="verified", confidence=0.80, failures=0, passes=5),
    source_trace_ids=["trace-readme-gen-001"],
)
