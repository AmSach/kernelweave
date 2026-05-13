from kernelweave import Kernel, KernelStatus
from kernelweave.compose import compose_sequence

# Define two mock kernels
kernel_a = Kernel(
    kernel_id="extract-data",
    name="Extract Data",
    task_family="data-extraction",
    description="Extracts key data from text",
    input_schema={"type": "object"},
    output_schema={"type": "object", "properties": {"data": {"type": "array"}}},
    preconditions=["input text available"],
    postconditions=["data field exists in output"],
    steps=[{"step": 1, "action": "extract"}],
    rollback=["clear extracted data"],
    evidence_requirements=["data extracted"],
    tests=[],
    status=KernelStatus(state="verified", confidence=0.9, failures=0, passes=10),
    source_trace_ids=[],
)

kernel_b = Kernel(
    kernel_id="format-output",
    name="Format Output",
    task_family="formatting",
    description="Formats data into a specific structure",
    input_schema={"type": "object", "required": ["data"]},
    output_schema={"type": "object", "properties": {"formatted_result": {"type": "string"}}},
    preconditions=["data field exists in input"],
    postconditions=["formatted_result field exists in output"],
    steps=[{"step": 1, "action": "format"}],
    rollback=["clear formatted result"],
    evidence_requirements=["result formatted"],
    tests=[],
    status=KernelStatus(state="verified", confidence=0.95, failures=0, passes=15),
    source_trace_ids=[],
)

print("Composing kernels...")
result = compose_sequence(kernel_a, kernel_b)

print(f"Composed Kernel ID: {result.kernel.kernel_id}")
print(f"Composed Name: {result.kernel.name}")
print(f"Preconditions: {result.kernel.preconditions}")
print(f"Postconditions: {result.kernel.postconditions}")
print(f"Steps: {len(result.kernel.steps)}")
print(f"Confidence: {result.kernel.status.confidence}")
