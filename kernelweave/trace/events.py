"""Event types for trace instrumentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ReasoningStep:
    """A reasoning step captured from model output."""
    
    def __init__(
        self,
        step_type: str,  # "analysis", "hypothesis", "inference", "conclusion"
        content: str,
        confidence: float = 0.0,
        dependencies: list[str] | None = None,
    ):
        self.step_type = step_type
        self.content = content
        self.confidence = confidence
        self.dependencies = dependencies or []
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "reasoning",
            "step_type": self.step_type,
            "content": self.content,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
        }


@dataclass
class ToolCall:
    """A tool call captured from model behavior."""
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    success: bool = True
    timestamp: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "tool": self.tool_name,
            "arguments": self.arguments,
            "result": str(self.result) if self.result else None,
            "success": self.success,
        }


@dataclass
class EvidenceCapture:
    """Evidence gathered during reasoning."""
    evidence_type: str  # "observation", "fact", "inference", "external"
    content: str
    source: str = ""
    reliability: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "evidence",
            "evidence_type": self.evidence_type,
            "content": self.content,
            "source": self.source,
            "reliability": self.reliability,
        }


@dataclass
class VerificationCheck:
    """A verification step against constraints."""
    constraint: str
    passed: bool
    evidence: str = ""
    notes: str = ""
    
    def to_dict(self) -> Any:
        return {
            "type": "verification",
            "constraint": self.constraint,
            "passed": self.passed,
            "evidence": self.evidence,
            "notes": self.notes,
        }
