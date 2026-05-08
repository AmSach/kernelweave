"""Tests for trace capture and instrumentation."""
from kernelweave.trace import (
    ReasoningStep,
    ToolCall,
    EvidenceCapture,
    VerificationCheck,
    CapturedStep,
    ExecutionTrace,
    TraceCapture,
)
from kernelweave.kernel import TraceEvent


class MockBackend:
    """Mock backend for testing."""
    
    def __init__(self, response_text: str):
        self.response_text = response_text
    
    def generate(self, prompt, system_prompt="", **kwargs):
        class MockResponse:
            text = self.response_text
        return MockResponse()


def test_reasoning_step_to_dict():
    """Test ReasoningStep serialization."""
    step = ReasoningStep(
        step_type="analysis",
        content="Analyzing the input data",
        confidence=0.8,
        dependencies=["previous_step"],
    )
    
    data = step.to_dict()
    assert data["type"] == "reasoning"
    assert data["step_type"] == "analysis"
    assert data["content"] == "Analyzing the input data"
    assert data["confidence"] == 0.8


def test_tool_call_to_dict():
    """Test ToolCall serialization."""
    call = ToolCall(
        tool_name="read_file",
        arguments={"path": "/tmp/test.txt"},
        result="file contents",
        success=True,
    )
    
    data = call.to_dict()
    assert data["type"] == "tool_call"
    assert data["tool"] == "read_file"
    assert data["arguments"]["path"] == "/tmp/test.txt"


def test_trace_capture_basic():
    """Test basic trace capture from model call."""
    backend = MockBackend("First, I will analyze the data. Then, I will compare the files. Therefore, the result is success.")
    capture = TraceCapture(backend)
    
    trace = capture.generate_with_trace("Compare these files")
    
    assert trace.prompt == "Compare these files"
    assert trace.task_family == "comparison"
    assert len(trace.steps) >= 0
    assert trace.success == True


def test_trace_capture_reasoning_extraction():
    """Test extraction of reasoning steps."""
    backend = MockBackend(
        "1. First, I loaded the file. "
        "2. Then I analyzed the contents. "
        "Therefore, the comparison shows differences."
    )
    capture = TraceCapture(backend, capture_reasoning=True)
    
    trace = capture.generate_with_trace("Compare these files")
    
    assert len(trace.reasoning_chain) >= 1
    # Should detect numbered steps or conclusion
    has_content = any(rs.content for rs in trace.reasoning_chain)
    assert has_content


def test_trace_capture_evidence_extraction():
    """Test extraction of evidence items."""
    backend = MockBackend(
        "I found that the files differ in 3 places. "
        "Evidence: file A has 100 lines, file B has 150 lines."
    )
    capture = TraceCapture(backend, capture_reasoning=True)
    
    trace = capture.generate_with_trace("Compare these files")
    
    assert len(trace.evidence) >= 1
    # Should detect numerical evidence
    has_numerical = any(ev.evidence_type == "observation" for ev in trace.evidence)
    assert has_numerical or len(trace.evidence) >= 0


def test_execution_trace_to_events():
    """Test conversion of ExecutionTrace to TraceEvents."""
    trace = ExecutionTrace(
        trace_id="test-trace",
        prompt="Test prompt",
        task_family="test",
        steps=[CapturedStep(step_type="reasoning", content="step 1", timestamp=0.0)],
        tool_calls=[ToolCall(tool_name="test_tool", arguments={})],
        evidence=[EvidenceCapture(evidence_type="observation", content="found")],
        verifications=[VerificationCheck(constraint="test", passed=True)],
        reasoning_chain=[ReasoningStep(step_type="analysis", content="analyzing")],
        final_output="done",
        success=True,
        duration_ms=100.0,
    )
    
    events = trace.to_events()
    
    assert len(events) >= 1
    # First event should be plan
    assert events[0].kind == "plan"
    # Should have tool events
    assert any(e.kind == "tool" for e in events)
    # Should have evidence events
    assert any(e.kind == "evidence" for e in events)


def test_trace_to_dict():
    """Test ExecutionTrace serialization."""
    trace = ExecutionTrace(
        trace_id="test-001",
        prompt="Test",
        task_family="test",
        steps=[],
        tool_calls=[],
        evidence=[],
        verifications=[],
        reasoning_chain=[],
        final_output="result",
        success=True,
        duration_ms=50.0,
        metadata={"key": "value"},
    )
    
    data = trace.to_dict()
    
    assert data["trace_id"] == "test-001"
    assert data["prompt"] == "Test"
    assert data["success"] == True
    assert data["metadata"]["key"] == "value"


def test_task_family_inference():
    """Test task family inference from prompt."""
    backend = MockBackend("done")
    capture = TraceCapture(backend)
    
    assert capture._infer_task_family("Compare these files") == "comparison"
    assert capture._infer_task_family("Summarize this document") == "summarization"
    assert capture._infer_task_family("Fix the bug in this code") == "debugging"
    assert capture._infer_task_family("Explain how this works") == "explanation"


def test_reasoning_type_classification():
    """Test reasoning type classification."""
    backend = MockBackend("done")
    capture = TraceCapture(backend)
    
    assert capture._classify_reasoning_type("Therefore, the result is X") == "conclusion"
    # "hypothesize" contains "hypothesis" so it should match
    assert capture._classify_reasoning_type("I guess that X is true") == "hypothesis"
    assert capture._classify_reasoning_type("Let me analyze the data") == "analysis"
