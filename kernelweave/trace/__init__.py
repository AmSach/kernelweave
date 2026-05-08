"""Trace instrumentation for capturing actual model reasoning behavior.

Wrap model calls, capture chain-of-thought, tool calls, intermediate states.
Convert to TraceEvents for kernel compilation. No synthetic agent plans -
the model's real reasoning path becomes the kernel.
"""
from .instrument import TraceRecorder, TraceContext
from .events import ReasoningStep, ToolCall, EvidenceCapture, VerificationCheck
from .capture import (
    CapturedStep,
    ExecutionTrace,
    TraceCapture,
    StreamingTraceCapture,
)

__all__ = [
    # Events
    "ReasoningStep",
    "ToolCall",
    "EvidenceCapture",
    "VerificationCheck",
    # Capture
    "CapturedStep",
    "ExecutionTrace",
    "TraceCapture",
    "StreamingTraceCapture",
]
