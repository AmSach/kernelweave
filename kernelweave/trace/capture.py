"""Trace instrumentation — capture ACTUAL model behavior, not synthesized plans.

The key insight: kernel compilation should come from REAL model execution traces,
not from prompting the model to "plan" and then compiling that plan.

This module wraps model calls and captures:
  - Chain-of-thought reasoning (actual token sequences)
  - Tool calls and their results
  - Evidence gathering
  - Verification steps
  - Decision points

The captured trace becomes the kernel source, not a synthesized abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from datetime import datetime, timezone
import json
import time
import re

from .events import ReasoningStep, ToolCall, EvidenceCapture, VerificationCheck
from ..kernel import TraceEvent


class ModelBackendProtocol(Protocol):
    """Protocol for model backends that can be instrumented."""
    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> Any: ...
    def generate_stream(self, prompt: str, system_prompt: str = "", **kwargs) -> Any: ...


@dataclass
class CapturedStep:
    """A single step in the execution trace."""
    step_type: str
    content: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_trace_event(self) -> TraceEvent:
        return TraceEvent(kind=self.step_type, payload={
            "text": self.content,
            "timestamp": self.timestamp,
            **self.metadata,
        })


@dataclass
class ExecutionTrace:
    """Complete execution trace from a model call."""
    trace_id: str
    prompt: str
    task_family: str
    steps: list[CapturedStep]
    tool_calls: list[ToolCall]
    evidence: list[EvidenceCapture]
    verifications: list[VerificationCheck]
    reasoning_chain: list[ReasoningStep]
    final_output: str
    success: bool
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "task_family": self.task_family,
            "steps": [{"type": s.step_type, "content": s.content, "timestamp": s.timestamp} for s in self.steps],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "evidence": [ev.to_dict() for ev in self.evidence],
            "verifications": [v.to_dict() for v in self.verifications],
            "reasoning_chain": [rs.to_dict() for rs in self.reasoning_chain],
            "final_output": self.final_output,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }
    
    def to_events(self) -> list[TraceEvent]:
        """Convert to TraceEvent list for kernel compilation."""
        events = []
        
        # Add plan from prompt
        events.append(TraceEvent(kind="plan", payload={"text": self.prompt[:500]}))
        
        # Add reasoning steps
        for rs in self.reasoning_chain:
            events.append(TraceEvent(kind="decision", payload={
                "text": rs.content,
                "step_type": rs.step_type,
                "confidence": rs.confidence,
            }))
        
        # Add tool calls
        for tc in self.tool_calls:
            events.append(TraceEvent(kind="tool", payload={
                "tool": tc.tool_name,
                "args": tc.arguments,
                "result": str(tc.result) if tc.result else None,
                "success": tc.success,
            }))
        
        # Add evidence
        for ev in self.evidence:
            events.append(TraceEvent(kind="evidence", payload={
                "text": ev.content,
                "type": ev.evidence_type,
                "source": ev.source,
            }))
        
        # Add verifications
        for v in self.verifications:
            events.append(TraceEvent(kind="verification", payload={
                "text": v.constraint,
                "passed": v.passed,
                "evidence": v.evidence,
            }))
        
        # Add final decision
        events.append(TraceEvent(kind="decision", payload={"text": self.final_output[:500]}))
        
        return events


class TraceCapture:
    """Wrap a model backend and capture execution traces.
    
    Usage:
        backend = OpenAIBackend(...)
        capture = TraceCapture(backend)
        
        # Make a call
        trace = capture.generate_with_trace("compare these files")
        
        # Compile trace to kernel
        kernel = compile_trace_to_kernel(
            trace.trace_id,
            trace.task_family,
            trace.prompt,
            trace.to_events(),
            {"result": trace.final_output}
        )
    """
    
    def __init__(
        self,
        backend: ModelBackendProtocol | None = None,
        trace_id_prefix: str = "trace",
        capture_tools: bool = True,
        capture_reasoning: bool = True,
    ):
        self.backend = backend
        self.trace_id_prefix = trace_id_prefix
        self.capture_tools = capture_tools
        self.capture_reasoning = capture_reasoning
        self._call_counter = 0
    
    @classmethod
    def standalone(cls, trace_id_prefix: str = "trace") -> "TraceCapture":
        """Create a capture instance without a backend for manual trace building."""
        return cls(backend=None, trace_id_prefix=trace_id_prefix)
    
    def capture_execution(
        self,
        prompt: str,
        steps: list[ReasoningStep | ToolCall | EvidenceCapture | VerificationCheck],
        model_response: str = "",
        task_family: str = "",
    ) -> ExecutionTrace:
        """Manually build an execution trace from captured steps.
        
        Useful when you have raw model output and want to structure it.
        """
        start_time = time.time()
        trace_id = self._generate_trace_id()
        
        if not task_family:
            task_family = self._infer_task_family(prompt)
        
        # Categorize steps
        reasoning_chain = [s for s in steps if isinstance(s, ReasoningStep)]
        tool_calls = [s for s in steps if isinstance(s, ToolCall)]
        evidence = [s for s in steps if isinstance(s, EvidenceCapture)]
        verifications = [s for s in steps if isinstance(s, VerificationCheck)]
        
        # Build captured steps
        captured_steps: list[CapturedStep] = []
        for i, step in enumerate(steps):
            step_type = "reasoning"
            content = ""
            metadata = {}
            
            if isinstance(step, ReasoningStep):
                step_type = step.step_type
                content = step.content
                metadata = {"confidence": step.confidence}
            elif isinstance(step, ToolCall):
                step_type = "tool_call"
                content = f"{step.tool_name}({json.dumps(step.arguments)})"
                metadata = {"tool": step.tool_name, "args": step.arguments, "success": step.success}
            elif isinstance(step, EvidenceCapture):
                step_type = "evidence"
                content = step.content
                metadata = {"type": step.evidence_type, "source": step.source}
            elif isinstance(step, VerificationCheck):
                step_type = "verification"
                content = step.constraint
                metadata = {"passed": step.passed, "evidence": step.evidence}
            
            captured_steps.append(CapturedStep(
                step_type=step_type,
                content=content,
                timestamp=start_time + i * 0.1,
                metadata=metadata,
            ))
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ExecutionTrace(
            trace_id=trace_id,
            prompt=prompt,
            task_family=task_family,
            steps=captured_steps,
            tool_calls=tool_calls,
            evidence=evidence,
            verifications=verifications,
            reasoning_chain=reasoning_chain,
            final_output=model_response,
            success=len(verifications) == 0 or all(v.passed for v in verifications),
            duration_ms=duration_ms,
        )
    
    def _generate_trace_id(self) -> str:
        self._call_counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{self.trace_id_prefix}-{ts}-{self._call_counter:04d}"
    
    def generate_with_trace(
        self,
        prompt: str,
        system_prompt: str = "",
        task_family: str = "",
        **kwargs,
    ) -> ExecutionTrace:
        """Generate a response and capture the full execution trace."""
        start_time = time.time()
        trace_id = self._generate_trace_id()
        
        # Determine task family if not provided
        if not task_family:
            task_family = self._infer_task_family(prompt)
        
        steps: list[CapturedStep] = []
        tool_calls: list[ToolCall] = []
        evidence: list[EvidenceCapture] = []
        verifications: list[VerificationCheck] = []
        reasoning_chain: list[ReasoningStep] = []
        
        # Make the actual call
        response = self.backend.generate(prompt, system_prompt=system_prompt, **kwargs)
        
        # Extract content
        output_text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse the output for reasoning patterns
        if self.capture_reasoning:
            reasoning_chain = self._extract_reasoning(output_text)
            evidence = self._extract_evidence(output_text)
            verifications = self._extract_verifications(output_text, system_prompt)
        
        # Capture tool calls if present in response
        if self.capture_tools and hasattr(response, 'tool_calls'):
            for tc in response.tool_calls:
                tool_calls.append(ToolCall(
                    tool_name=tc.function.name if hasattr(tc, 'function') else tc.get('name', 'unknown'),
                    arguments=tc.function.arguments if hasattr(tc, 'function') else tc.get('arguments', {}),
                    result=None,  # Result comes from execution
                    success=True,
                    timestamp=time.time(),
                ))
        
        # Create steps from captured content
        for rs in reasoning_chain:
            steps.append(CapturedStep(
                step_type="reasoning",
                content=rs.content,
                timestamp=start_time + len(steps) * 0.1,
                metadata={"step_type": rs.step_type, "confidence": rs.confidence},
            ))
        
        for tc in tool_calls:
            steps.append(CapturedStep(
                step_type="tool_call",
                content=f"Call {tc.tool_name}({json.dumps(tc.arguments)})",
                timestamp=tc.timestamp,
                metadata={"tool": tc.tool_name, "args": tc.arguments, "success": tc.success},
            ))
        
        for ev in evidence:
            steps.append(CapturedStep(
                step_type="evidence",
                content=ev.content,
                timestamp=start_time + len(steps) * 0.1,
                metadata={"type": ev.evidence_type, "source": ev.source},
            ))
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ExecutionTrace(
            trace_id=trace_id,
            prompt=prompt,
            task_family=task_family,
            steps=steps,
            tool_calls=tool_calls,
            evidence=evidence,
            verifications=verifications,
            reasoning_chain=reasoning_chain,
            final_output=output_text,
            success=True,  # Would need external validation to determine
            duration_ms=duration_ms,
            metadata={
                "backend": type(self.backend).__name__,
                "system_prompt_length": len(system_prompt),
            },
        )
    
    def _infer_task_family(self, prompt: str) -> str:
        """Infer task family from prompt content."""
        prompt_lower = prompt.lower()
        
        if "compare" in prompt_lower or "difference" in prompt_lower:
            return "comparison"
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return "summarization"
        if "analyze" in prompt_lower or "analysis" in prompt_lower:
            return "analysis"
        if "write" in prompt_lower or "generate" in prompt_lower:
            return "generation"
        if "fix" in prompt_lower or "debug" in prompt_lower or "error" in prompt_lower:
            return "debugging"
        if "explain" in prompt_lower:
            return "explanation"
        if "list" in prompt_lower or "enumerate" in prompt_lower:
            return "listing"
        
        # Fallback: first 6 words
        words = prompt.split()[:6]
        return " ".join(words) if words else "general"
    
    def _extract_reasoning(self, text: str) -> list[ReasoningStep]:
        """Extract reasoning steps from model output."""
        steps = []
        
        # Pattern 1: Numbered steps
        numbered = re.findall(r'(?:^|\n)\s*(\d+)\.\s+(.+?)(?=(?:\n\s*\d+\.)|$)', text, re.DOTALL)
        for num, content in numbered:
            step_type = self._classify_reasoning_type(content)
            steps.append(ReasoningStep(
                step_type=step_type,
                content=content.strip(),
                confidence=0.8 if step_type == "conclusion" else 0.7,
            ))
        
        # Pattern 2: "First, ... Then, ... Finally, ..."
        temporal = re.findall(r'(?:first|then|next|finally|after that)[,\s]+(.+?)(?=\.|,|\n)', text, re.IGNORECASE)
        for content in temporal:
            steps.append(ReasoningStep(
                step_type="inference",
                content=content.strip(),
                confidence=0.7,
            ))
        
        # Pattern 3: "Because ... Therefore ..."
        causal = re.findall(r'(?:because|since|given that)\s+(.+?)(?=,|therefore|so|thus|\n)', text, re.IGNORECASE)
        for content in causal:
            steps.append(ReasoningStep(
                step_type="hypothesis",
                content=content.strip(),
                confidence=0.75,
            ))
        
        # Pattern 4: Conclusions
        conclusions = re.findall(r'(?:therefore|thus|so|hence|in conclusion|to summarize)[,\s]+(.+?)(?=\.|\n|$)', text, re.IGNORECASE)
        for content in conclusions:
            steps.append(ReasoningStep(
                step_type="conclusion",
                content=content.strip(),
                confidence=0.85,
            ))
        
        return steps
    
    def _classify_reasoning_type(self, text: str) -> str:
        """Classify the type of reasoning in a step."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["conclude", "therefore", "thus", "result"]):
            return "conclusion"
        if any(kw in text_lower for kw in ["hypothesis", "guess", "assume", "suppose"]):
            return "hypothesis"
        if any(kw in text_lower for kw in ["analyze", "examine", "look at", "consider"]):
            return "analysis"
        if any(kw in text_lower for kw in ["evidence", "found", "observed", "data shows"]):
            return "inference"
        
        return "inference"
    
    def _extract_evidence(self, text: str) -> list[EvidenceCapture]:
        """Extract evidence items from model output."""
        evidence = []
        
        # Pattern: "evidence: X" or "found that X"
        explicit = re.findall(r'(?:evidence|found|observed|noted|detected)[:\s]+(.+?)(?=\.|\n|$)', text, re.IGNORECASE)
        for content in explicit:
            evidence.append(EvidenceCapture(
                evidence_type="observation",
                content=content.strip(),
                source="model_output",
                reliability=0.9,
            ))
        
        # Pattern: Quotes and citations
        quotes = re.findall(r'"([^"]+)"', text)
        for quote in quotes:
            if len(quote) > 10:  # Skip short quoted strings
                evidence.append(EvidenceCapture(
                    evidence_type="fact",
                    content=quote,
                    source="quoted",
                    reliability=0.85,
                ))
        
        # Pattern: Numbers and statistics
        stats = re.findall(r'(\d+(?:\.\d+)?%?(?:\s*(?:percent|times|items|files|lines))?)', text)
        for stat in stats:
            evidence.append(EvidenceCapture(
                evidence_type="observation",
                content=stat,
                source="numerical",
                reliability=0.95,
            ))
        
        return evidence
    
    def _extract_verifications(self, output: str, system_prompt: str) -> list[VerificationCheck]:
        """Extract verification steps from output."""
        verifications = []
        
        # Check if output addresses system prompt requirements
        if "postconditions" in system_prompt.lower() or "requirements" in system_prompt.lower():
            # Extract requirements from system prompt
            reqs = re.findall(r'(?:must|should|need to|required)[:\s]+(.+?)(?=\.|\n|$)', system_prompt, re.IGNORECASE)
            
            for req in reqs:
                # Check if requirement is satisfied in output
                req_lower = req.lower()
                output_lower = output.lower()
                
                passed = any(kw in output_lower for kw in req_lower.split()[:3])
                
                verifications.append(VerificationCheck(
                    constraint=req,
                    passed=passed,
                    evidence=output[:200] if passed else "",
                    notes="checked against output" if passed else "not found in output",
                ))
        
        # Pattern: "verified X" / "checked X"
        explicit = re.findall(r'(?:verified|checked|confirmed|validated)[:\s]+(.+?)(?=\.|\n|$)', output, re.IGNORECASE)
        for content in explicit:
            verifications.append(VerificationCheck(
                constraint=content.strip(),
                passed=True,
                evidence=content.strip(),
                notes="explicitly verified in output",
            ))
        
        return verifications


class StreamingTraceCapture(TraceCapture):
    """Capture traces from streaming model responses."""
    
    def generate_stream_with_trace(
        self,
        prompt: str,
        system_prompt: str = "",
        task_family: str = "",
        **kwargs,
    ) -> tuple[ExecutionTrace, str]:
        """Generate streaming response and capture trace."""
        start_time = time.time()
        trace_id = self._generate_trace_id()
        
        if not task_family:
            task_family = self._infer_task_family(prompt)
        
        # Collect streaming tokens
        tokens: list[str] = []
        
        for chunk in self.backend.generate_stream(prompt, system_prompt=system_prompt, **kwargs):
            token = chunk.text if hasattr(chunk, 'text') else str(chunk)
            tokens.append(token)
        
        output_text = "".join(tokens)
        
        # Parse the complete output
        reasoning_chain = self._extract_reasoning(output_text) if self.capture_reasoning else []
        evidence = self._extract_evidence(output_text) if self.capture_reasoning else []
        verifications = self._extract_verifications(output_text, system_prompt)
        
        # Build steps
        steps = [
            CapturedStep(
                step_type="token_stream",
                content=token,
                timestamp=start_time + i * 0.01,
            )
            for i, token in enumerate(tokens)
        ]
        
        duration_ms = (time.time() - start_time) * 1000
        
        trace = ExecutionTrace(
            trace_id=trace_id,
            prompt=prompt,
            task_family=task_family,
            steps=steps,
            tool_calls=[],
            evidence=evidence,
            verifications=verifications,
            reasoning_chain=reasoning_chain,
            final_output=output_text,
            success=True,
            duration_ms=duration_ms,
        )
        
        return trace, output_text
