"""Legacy trace instrumentation module.

Re-exports from capture.py for backward compatibility.
"""
from .capture import TraceCapture, ExecutionTrace

# Legacy aliases
TraceRecorder = TraceCapture
TraceContext = ExecutionTrace

__all__ = ["TraceRecorder", "TraceContext"]
