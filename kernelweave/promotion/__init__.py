"""Auto-Promotion: Automatically create kernels from successful traces.

REVOLUTIONARY: System self-improves by promoting successful traces to kernels.

Process:
1. Execute prompt
2. Verify output
3. If successful and high confidence, promote trace to kernel
4. Add kernel to store
5. Queue trace for training

Usage:
    from kernelweave.promotion import AutoPromoter
    
    promoter = AutoPromoter(kernel_store)
    
    # After successful execution
    if result.verification.passed:
        kernel = promoter.promote(trace)
        
    # Check promotion queue
    traces = promoter.get_training_queue()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import json
import time
import hashlib

from ..kernel import Kernel, KernelStatus, KernelStore
from ..compiler import compile_trace_to_kernel
from ..trace import ExecutionTrace


@dataclass
class PromotionConfig:
    """Configuration for auto-promotion."""
    min_confidence: float = 0.7
    min_evidence_count: int = 2
    require_verification: bool = True
    promotion_cooldown_hours: int = 24
    max_kernels_per_family: int = 10
    auto_train: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "min_evidence_count": self.min_evidence_count,
            "require_verification": self.require_verification,
            "promotion_cooldown_hours": self.promotion_cooldown_hours,
            "max_kernels_per_family": self.max_kernels_per_family,
            "auto_train": self.auto_train,
        }


@dataclass
class PromotedKernel:
    """Record of a promoted kernel."""
    kernel_id: str
    source_trace_id: str
    task_family: str
    confidence: float
    promoted_at: float
    training_queued: bool
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "source_trace_id": self.source_trace_id,
            "task_family": self.task_family,
            "confidence": self.confidence,
            "promoted_at": self.promoted_at,
            "training_queued": self.training_queued,
        }


class AutoPromoter:
    """Automatically promote successful traces to kernels.
    
    REVOLUTIONARY: The system self-improves by learning from successful
    executions. This is the key to continuous improvement.
    
    Usage:
        promoter = AutoPromoter(kernel_store)
        
        # After execution
        if result.success and result.confidence > 0.7:
            kernel = promoter.promote(trace)
            if kernel:
                print(f"New kernel created: {kernel.kernel_id}")
    """
    
    def __init__(
        self,
        kernel_store: KernelStore,
        config: PromotionConfig | None = None,
    ):
        self.store = kernel_store
        self.config = config or PromotionConfig()
        
        # Track promotions
        self.promoted: list[PromotedKernel] = []
        
        # Training queue
        self.training_queue: list[ExecutionTrace] = []
        
        # Cooldown tracking (family -> last promotion time)
        self._last_promotion: dict[str, float] = {}
    
    def should_promote(self, trace: ExecutionTrace) -> bool:
        """Check if a trace should be promoted to a kernel.
        
        Criteria:
        - Verification passed
        - Confidence above threshold
        - Enough evidence
        - Not on cooldown for this task family
        - Not too many kernels for this family
        """
        # Check verification
        if self.config.require_verification and not trace.success:
            return False
        
        # Check confidence
        if trace.confidence < self.config.min_confidence:
            return False
        
        # Check evidence count
        evidence_count = len(trace.verification_result.get("evidence_found", []))
        if evidence_count < self.config.min_evidence_count:
            return False
        
        # Check cooldown
        if trace.kernel_id:
            # This was a kernel hit, not a new trace
            return False
        
        # Check task family cooldown
        task_family = self._infer_task_family(trace.prompt)
        last_promotion = self._last_promotion.get(task_family, 0)
        hours_since = (time.time() - last_promotion) / 3600
        
        if hours_since < self.config.promotion_cooldown_hours:
            return False
        
        # Check max kernels per family
        kernels_in_family = self._count_kernels_in_family(task_family)
        if kernels_in_family >= self.config.max_kernels_per_family:
            return False
        
        return True
    
    def promote(self, trace: ExecutionTrace) -> Kernel | None:
        """Promote a trace to a kernel.
        
        Returns:
            Kernel if promoted, None if not eligible
        """
        if not self.should_promote(trace):
            return None
        
        # Create kernel from trace
        kernel = self._create_kernel_from_trace(trace)
        
        # Add to store
        self.store.add_kernel(kernel)
        
        # Record promotion
        promoted = PromotedKernel(
            kernel_id=kernel.kernel_id,
            source_trace_id=trace.trace_id,
            task_family=kernel.task_family,
            confidence=trace.confidence,
            promoted_at=time.time(),
            training_queued=self.config.auto_train,
        )
        
        self.promoted.append(promoted)
        
        # Update cooldown
        self._last_promotion[kernel.task_family] = time.time()
        
        # Queue for training
        if self.config.auto_train:
            self.training_queue.append(trace)
        
        return kernel
    
    def _create_kernel_from_trace(self, trace: ExecutionTrace) -> Kernel:
        """Create a kernel from an execution trace."""
        # Convert trace to events
        from ..kernel import TraceEvent
        
        events = []
        
        # Add plan
        events.append(TraceEvent(
            kind="plan",
            payload={"text": trace.prompt[:500]},
        ))
        
        # Add steps
        for step in trace.steps:
            events.append(TraceEvent(
                kind=step.get("action", "step"),
                payload=step,
            ))
        
        # Add verification
        if trace.verification_result.get("passed"):
            events.append(TraceEvent(
                kind="verification",
                payload={"text": "output verified"},
            ))
        
        # Compile
        kernel = compile_trace_to_kernel(
            trace_id=trace.trace_id,
            task_family=self._infer_task_family(trace.prompt),
            description=trace.prompt[:200],
            events=events,
            expected_output={"result": trace.execution_output[:256]},
        )
        
        # Set confidence from trace
        kernel.status.confidence = trace.confidence
        kernel.status.state = "candidate"
        
        return kernel
    
    def _infer_task_family(self, prompt: str) -> str:
        """Infer task family from prompt."""
        prompt_lower = prompt.lower()
        
        if "compare" in prompt_lower:
            return "comparison"
        if "analyze" in prompt_lower:
            return "analysis"
        if "find" in prompt_lower or "search" in prompt_lower:
            return "search"
        if "fix" in prompt_lower or "debug" in prompt_lower:
            return "debugging"
        if "generate" in prompt_lower or "write" in prompt_lower:
            return "generation"
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return "summarization"
        if "convert" in prompt_lower or "transform" in prompt_lower:
            return "transformation"
        if "test" in prompt_lower:
            return "testing"
        if "document" in prompt_lower or "readme" in prompt_lower:
            return "documentation"
        
        # Fallback: first 4 words
        words = prompt.split()[:4]
        return " ".join(words).lower() if words else "general"
    
    def _count_kernels_in_family(self, task_family: str) -> int:
        """Count kernels in a task family."""
        count = 0
        for item in self.store.list_kernels():
            kernel = self.store.get_kernel(item["kernel_id"])
            if kernel.task_family == task_family:
                count += 1
        return count
    
    def get_training_queue(self) -> list[ExecutionTrace]:
        """Get traces queued for training."""
        return self.training_queue.copy()
    
    def clear_training_queue(self) -> list[ExecutionTrace]:
        """Clear and return training queue."""
        traces = self.training_queue.copy()
        self.training_queue.clear()
        return traces
    
    def get_promotion_history(self) -> list[PromotedKernel]:
        """Get history of promoted kernels."""
        return self.promoted.copy()
    
    def get_statistics(self) -> dict[str, Any]:
        """Get promotion statistics."""
        return {
            "total_promotions": len(self.promoted),
            "training_queue_size": len(self.training_queue),
            "families_with_kernels": len(set(p.task_family for p in self.promoted)),
            "avg_confidence": sum(p.confidence for p in self.promoted) / max(1, len(self.promoted)),
        }
    
    def save_state(self, path: Path) -> None:
        """Save promotion state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config.to_dict(),
            "promoted": [p.to_dict() for p in self.promoted],
            "last_promotion": self._last_promotion,
        }
        
        path.write_text(json.dumps(state, indent=2))
    
    def load_state(self, path: Path) -> None:
        """Load promotion state from file."""
        path = Path(path)
        if not path.exists():
            return
        
        state = json.loads(path.read_text())
        
        self.config = PromotionConfig(**state.get("config", {}))
        self.promoted = [PromotedKernel(**p) for p in state.get("promoted", [])]
        self._last_promotion = state.get("last_promotion", {})
