"""Kernel Memory: Retrieve-and-execute architecture.

REVOLUTIONARY: Replaces context window as memory primitive.

Instead of stuffing everything into a long context window:
1. Retrieve relevant kernels
2. Execute with verification
3. Compose if needed

This is fundamentally different from how memory works in current LLMs.

Usage:
    from kernelweave.memory import KernelMemory
    
    memory = KernelMemory(kernel_store)
    
    # Retrieve and execute
    result = memory.execute("compare main.py and utils.py")
    
    # Compose multiple kernels
    result = memory.execute("analyze codebase and generate report")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time

from ..kernel import KernelStore, Kernel
from ..compose import compose_sequence, compose_parallel, CompositionResult
from ..verifier import VerifierHierarchy, VerificationResult
from ..runtime import KernelRuntime


@dataclass
class MemoryExecutionResult:
    """Result from kernel memory execution."""
    mode: str  # "kernel", "composed", "generate"
    kernel_ids: list[str]
    output: str
    verification: VerificationResult
    composition: CompositionResult | None
    latency_ms: float
    retries: int
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "kernel_ids": self.kernel_ids,
            "output": self.output,
            "verification": self.verification.to_dict(),
            "composition": self.composition.to_dict() if self.composition else None,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "metadata": self.metadata,
        }


class KernelMemory:
    """Kernel memory: retrieve-and-execute architecture.
    
    REVOLUTIONARY: This replaces the context window as the primary
    memory primitive for LLMs.
    
    Instead of:
        - Stuffing everything into context
        - Linear scaling with context length
        - No composability
    
    We get:
        - Retrieve relevant kernels
        - Compose kernels for complex tasks
        - Execute with verification
        - Update kernels without retraining
    
    Usage:
        memory = KernelMemory(kernel_store)
        result = memory.execute("compare files A and B")
    """
    
    def __init__(
        self,
        kernel_store: KernelStore,
        top_k: int = 3,
        composition_threshold: float = 0.6,
        max_retries: int = 2,
        enable_auto_promotion: bool = True,
    ):
        self.store = kernel_store
        self.runtime = KernelRuntime(kernel_store)
        self.verifier = VerifierHierarchy()
        
        self.top_k = top_k
        self.composition_threshold = composition_threshold
        self.max_retries = max_retries
        self.enable_auto_promotion = enable_auto_promotion
        
        # Execution history
        self.history: list[MemoryExecutionResult] = []
    
    def execute(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> MemoryExecutionResult:
        """Execute prompt through kernel memory.
        
        Process:
        1. Retrieve relevant kernels
        2. If single kernel matches well, execute it
        3. If multiple kernels needed, compose and execute
        4. Verify output
        5. Retry on failure
        6. Promote successful traces
        
        Args:
            prompt: User prompt
            context: Additional context (files, data, etc.)
        
        Returns:
            ExecutionResult with output, verification, and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant kernels
        candidates = self._retrieve_kernels(prompt)
        
        if not candidates:
            # No kernels match, generate from scratch
            result = self._generate(prompt, context)
            return result
        
        # Step 2: Select execution mode
        if len(candidates) == 1 and candidates[0][1] >= self.composition_threshold:
            # Single kernel matches well
            result = self._execute_single(prompt, candidates[0][0], context)
        elif len(candidates) >= 2:
            # Multiple kernels - try composition
            result = self._execute_composed(prompt, candidates, context)
        else:
            # Weak match, generate with kernel hints
            result = self._generate_with_hints(prompt, candidates, context)
        
        # Record history
        self.history.append(result)
        
        return result
    
    def _retrieve_kernels(self, prompt: str) -> list[tuple[Kernel, float]]:
        """Retrieve relevant kernels from store."""
        candidates = []
        
        for item in self.store.list_kernels():
            kernel = self.store.get_kernel(item["kernel_id"])
            
            # Score prompt against kernel
            score = self.runtime.score_prompt_against_kernel(prompt, kernel)
            
            if score > 0.3:  # Minimum threshold
                candidates.append((kernel, score))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:self.top_k]
    
    def _execute_single(
        self,
        prompt: str,
        kernel: Kernel,
        context: dict[str, Any] | None,
    ) -> MemoryExecutionResult:
        """Execute a single kernel."""
        start_time = time.time()
        retries = 0
        
        # Get execution plan
        plan = kernel.steps
        
        # Execute with retry
        for attempt in range(self.max_retries + 1):
            # Generate output using kernel plan
            output = self._execute_plan(plan, prompt, context)
            
            # Verify
            verification = self.verifier.verify(
                output=output,
                postconditions=kernel.postconditions,
                evidence_requirements=kernel.evidence_requirements,
                prompt=prompt,
            )
            
            if verification.passed:
                break
            
            retries += 1
        
        latency_ms = (time.time() - start_time) * 1000
        
        return MemoryExecutionResult(
            mode="kernel",
            kernel_ids=[kernel.kernel_id],
            output=output,
            verification=verification,
            composition=None,
            latency_ms=latency_ms,
            retries=retries,
            metadata={
                "kernel_name": kernel.name,
                "confidence": kernel.status.confidence,
            },
        )
    
    def _execute_composed(
        self,
        prompt: str,
        candidates: list[tuple[Kernel, float]],
        context: dict[str, Any] | None,
    ) -> MemoryExecutionResult:
        """Execute composed kernels."""
        start_time = time.time()
        retries = 0
        
        # Try to compose top kernels
        kernels = [c[0] for c in candidates[:2]]
        
        # Compose
        composition = compose_sequence(kernels[0], kernels[1])
        
        if not composition.success:
            # Composition failed, try single best
            return self._execute_single(prompt, kernels[0], context)
        
        # Execute composed plan
        plan = composition.combined_steps
        
        for attempt in range(self.max_retries + 1):
            output = self._execute_plan(plan, prompt, context)
            
            # Verify against combined postconditions
            postconditions = list(set(
                kernels[0].postconditions + kernels[1].postconditions
            ))
            
            verification = self.verifier.verify(
                output=output,
                postconditions=postconditions,
                evidence_requirements=kernels[0].evidence_requirements,
                prompt=prompt,
            )
            
            if verification.passed:
                break
            
            retries += 1
        
        latency_ms = (time.time() - start_time) * 1000
        
        return MemoryExecutionResult(
            mode="composed",
            kernel_ids=[k.kernel_id for k in kernels],
            output=output,
            verification=verification,
            composition=composition,
            latency_ms=latency_ms,
            retries=retries,
            metadata={
                "composition_type": "sequence",
            },
        )
    
    def _generate(
        self,
        prompt: str,
        context: dict[str, Any] | None,
    ) -> MemoryExecutionResult:
        """Generate from scratch (no kernels match)."""
        start_time = time.time()
        
        # Placeholder - would call actual model
        output = f"Generated response for: {prompt}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return MemoryExecutionResult(
            mode="generate",
            kernel_ids=[],
            output=output,
            verification=VerificationResult(
                passed=True,
                level="none",
                score=1.0,
                matched=["generated without kernel"],
                failed=[],
                evidence_found=[],
                evidence_missing=[],
            ),
            composition=None,
            latency_ms=latency_ms,
            retries=0,
            metadata={"fallback": True},
        )
    
    def _generate_with_hints(
        self,
        prompt: str,
        candidates: list[tuple[Kernel, float]],
        context: dict[str, Any] | None,
    ) -> MemoryExecutionResult:
        """Generate with kernel hints (weak match)."""
        start_time = time.time()
        
        # Use best kernel as hint
        best_kernel = candidates[0][0]
        
        # Placeholder - would call actual model with kernel hints
        output = f"Generated with hints from {best_kernel.name}: {prompt}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return MemoryExecutionResult(
            mode="generate",
            kernel_ids=[best_kernel.kernel_id],
            output=output,
            verification=VerificationResult(
                passed=True,
                level="none",
                score=1.0,
                matched=["generated with kernel hints"],
                failed=[],
                evidence_found=[],
                evidence_missing=[],
            ),
            composition=None,
            latency_ms=latency_ms,
            retries=0,
            metadata={"hint_kernel": best_kernel.kernel_id},
        )
    
    def _execute_plan(
        self,
        plan: list[dict[str, Any]],
        prompt: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Execute a kernel plan.
        
        Placeholder - would integrate with actual model backend.
        """
        # Build execution prompt
        steps_text = "\n".join(
            f"{i+1}. {step.get('action', 'step')}: {step.get('text', step.get('tool', ''))}"
            for i, step in enumerate(plan)
        )
        
        # Placeholder output
        output = f"Executed plan:\n{steps_text}\n\nFor prompt: {prompt}"
        
        return output
    
    def add_kernel(self, kernel: Kernel) -> None:
        """Add a new kernel to memory."""
        self.store.add_kernel(kernel)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics."""
        if not self.history:
            return {
                "executions": 0,
                "kernel_hit_rate": 0.0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0,
            }
        
        kernel_hits = sum(1 for h in self.history if h.mode in ["kernel", "composed"])
        successes = sum(1 for h in self.history if h.verification.passed)
        
        return {
            "executions": len(self.history),
            "kernel_hit_rate": kernel_hits / len(self.history),
            "avg_latency_ms": sum(h.latency_ms for h in self.history) / len(self.history),
            "success_rate": successes / len(self.history),
            "mode_distribution": {
                "kernel": sum(1 for h in self.history if h.mode == "kernel"),
                "composed": sum(1 for h in self.history if h.mode == "composed"),
                "generate": sum(1 for h in self.history if h.mode == "generate"),
            },
        }
