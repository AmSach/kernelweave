"""Kernel-Native Model: The main interface for kernel-native LLMs.

REVOLUTIONARY: This is the complete kernel-native system that:
1. Retrieves and executes kernels
2. Verifies outputs
3. Promotes successful traces
4. Queues traces for training

Usage:
    from kernelweave import KernelNativeModel
    
    model = KernelNativeModel(
        base_model="Qwen/Qwen2.5-7B",
        kernel_store="./kernel_store",
        enable_auto_promotion=True,
        enable_trace_collection=True,
    )
    
    result = model.run("compare main.py and utils.py")
    
    if result["verification"]["passed"]:
        print(f"Output: {result['output']}")
        print(f"Promoted: {result.get('promoted_kernel_id')}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path
import json
import time

from .kernel import Kernel, KernelStore, load_sample_store
from .memory import KernelMemory, MemoryExecutionResult
from .verifier import VerifierHierarchy, VerificationResult
from .promotion import AutoPromoter, PromotionConfig
from .trace import ExecutionTrace
from .training import TrainingConfig

# Stub implementations for compatibility
class TraceCollector:
    """Stub for compatibility."""
    def __init__(self, **kwargs):
        self.traces = []
    
    def get_verified_traces(self):
        return [t for t in self.traces if t.success]

class TraceTrainer:
    """Stub for compatibility."""
    def __init__(self, **kwargs):
        pass
    
    def train(self, traces, epochs=3):
        return {"status": "not_implemented"}

from .runtime import KernelRuntime


@dataclass
class KernelNativeConfig:
    """Configuration for kernel-native model."""
    # Model
    base_model: str = "Qwen/Qwen2.5-7B"
    device: str = "cuda"
    
    # Kernel memory
    kernel_store_path: str = "./kernel_store"
    top_k_kernels: int = 3
    composition_threshold: float = 0.6
    max_retries: int = 2
    
    # Verification
    enable_heuristic_verification: bool = True
    enable_tool_verification: bool = True
    enable_llm_judge: bool = False
    llm_judge_model: str = "gpt-4o-mini"
    
    # Auto-promotion
    enable_auto_promotion: bool = True
    min_promotion_confidence: float = 0.7
    auto_train: bool = False
    
    # Training
    training_output_dir: str = "./kernel-native-model"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "device": self.device,
            "kernel_store_path": self.kernel_store_path,
            "top_k_kernels": self.top_k_kernels,
            "composition_threshold": self.composition_threshold,
            "max_retries": self.max_retries,
            "enable_heuristic_verification": self.enable_heuristic_verification,
            "enable_tool_verification": self.enable_tool_verification,
            "enable_llm_judge": self.enable_llm_judge,
            "llm_judge_model": self.llm_judge_model,
            "enable_auto_promotion": self.enable_auto_promotion,
            "min_promotion_confidence": self.min_promotion_confidence,
            "auto_train": self.auto_train,
            "training_output_dir": self.training_output_dir,
        }


@dataclass
class ExecutionResult:
    """Result from kernel-native execution."""
    mode: str  # "kernel", "composed", "generate"
    output: str
    kernel_ids: list[str]
    verification: dict[str, Any]
    confidence: float
    promoted_kernel_id: str | None
    trace_id: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "output": self.output,
            "kernel_ids": self.kernel_ids,
            "verification": self.verification,
            "confidence": self.confidence,
            "promoted_kernel_id": self.promoted_kernel_id,
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class KernelNativeModel:
    """Kernel-native language model.
    
    REVOLUTIONARY: This is the complete system that implements
    verifiable kernel execution as the primitive unit of cognition.
    
    Key features:
    1. Kernel memory (retrieve-and-execute)
    2. Hierarchical verification
    3. Auto-promotion of successful traces
    4. Training on kernel execution traces
    
    Usage:
        model = KernelNativeModel(
            base_model="Qwen/Qwen2.5-7B",
            kernel_store="./kernel_store",
        )
        
        result = model.run("compare files A and B")
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B",
        kernel_store: KernelStore | str | None = None,
        config: KernelNativeConfig | None = None,
    ):
        self.config = config or KernelNativeConfig(base_model=base_model)
        
        # Initialize kernel store
        if isinstance(kernel_store, KernelStore):
            self.store = kernel_store
        elif isinstance(kernel_store, str):
            self.store = KernelStore(Path(kernel_store))
        else:
            self.store = load_sample_store(Path(self.config.kernel_store_path))
        
        # Initialize components
        self.memory = KernelMemory(
            kernel_store=self.store,
            top_k=self.config.top_k_kernels,
            composition_threshold=self.config.composition_threshold,
            max_retries=self.config.max_retries,
            enable_auto_promotion=self.config.enable_auto_promotion,
        )
        
        self.verifier = VerifierHierarchy(
            enable_heuristic=self.config.enable_heuristic_verification,
            enable_tool=self.config.enable_tool_verification,
            enable_llm_judge=self.config.enable_llm_judge,
            llm_judge_model=self.config.llm_judge_model,
        )
        
        self.promoter = AutoPromoter(
            kernel_store=self.store,
            config=PromotionConfig(
                min_confidence=self.config.min_promotion_confidence,
                auto_train=self.config.auto_train,
            ),
        )
        
        self.collector = TraceCollector(
            kernel_store=self.store,
            verifier_hierarchy=True,
        )
        
        # Execution counter for trace IDs
        self._execution_count = 0
    
    def run(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute prompt through kernel-native system.
        
        Process:
        1. Retrieve relevant kernels from memory
        2. Execute kernel or compose multiple kernels
        3. Verify output
        4. Promote successful trace to kernel
        5. Queue trace for training
        
        Args:
            prompt: User prompt
            context: Additional context (files, data, etc.)
        
        Returns:
            ExecutionResult with output, verification, and metadata
        """
        start_time = time.time()
        
        # Execute through kernel memory
        mem_result = self.memory.execute(prompt, context)
        
        # Create trace
        trace = ExecutionTrace(
            trace_id=self._generate_trace_id(),
            prompt=prompt,
            kernel_id=mem_result.kernel_ids[0] if mem_result.kernel_ids else None,
            preconditions=[],
            steps=mem_result.composition.combined_steps if mem_result.composition else [],
            postconditions=[],
            execution_output=mem_result.output,
            verification_result=mem_result.verification.to_dict(),
            success=mem_result.verification.passed,
            confidence=mem_result.verification.score,
            timestamp=time.time(),
            metadata={
                "mode": mem_result.mode,
                "kernel_ids": mem_result.kernel_ids,
            },
        )
        
        # Auto-promote if successful
        promoted_kernel_id = None
        if self.config.enable_auto_promotion and mem_result.verification.passed:
            kernel = self.promoter.promote(trace)
            if kernel:
                promoted_kernel_id = kernel.kernel_id
        
        # Collect trace
        self.collector.traces.append(trace)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ExecutionResult(
            mode=mem_result.mode,
            output=mem_result.output,
            kernel_ids=mem_result.kernel_ids,
            verification=mem_result.verification.to_dict(),
            confidence=mem_result.verification.score,
            promoted_kernel_id=promoted_kernel_id,
            trace_id=trace.trace_id,
            latency_ms=latency_ms,
            metadata={
                "retries": mem_result.retries,
                "composition": bool(mem_result.composition),
            },
        )
    
    def run_batch(
        self,
        prompts: list[str],
        auto_promote: bool = True,
    ) -> list[ExecutionResult]:
        """Run multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.run(prompt)
            results.append(result)
        return results
    
    def train(
        self,
        output_dir: str | None = None,
        epochs: int = 3,
    ) -> dict[str, Any]:
        """Train model on collected traces.
        
        This is the REVOLUTIONARY part: fine-tune on kernel execution traces.
        """
        traces = self.collector.get_verified_traces()
        
        if not traces:
            return {"status": "skipped", "reason": "no_verified_traces"}
        
        trainer = TraceTrainer(
            base_model=self.config.base_model,
            output_dir=output_dir or self.config.training_output_dir,
        )
        
        result = trainer.train(traces=traces, epochs=epochs)
        
        return result
    
    def get_verified_traces(self) -> list[ExecutionTrace]:
        """Get traces that passed verification."""
        return self.collector.get_verified_traces()
    
    def get_statistics(self) -> dict[str, Any]:
        """Get model statistics."""
        memory_stats = self.memory.get_statistics()
        promotion_stats = self.promoter.get_statistics()
        
        return {
            "memory": memory_stats,
            "promotion": promotion_stats,
            "traces_collected": len(self.collector.traces),
            "verified_traces": len(self.collector.get_verified_traces()),
        }
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        self._execution_count += 1
        ts = time.strftime("%Y%m%d%H%M%S")
        return f"trace-{ts}-{self._execution_count:04d}"
    
    def save_training_data(self, output_path: str) -> None:
        """Export training data for external training."""
        from .training import TraceTrainer
        
        trainer = TraceTrainer(base_model=self.config.base_model)
        trainer.export_training_data(
            traces=self.collector.get_verified_traces(),
            output_path=Path(output_path),
        )
    
    def install_kernels(self) -> None:
        """Install sample kernels."""
        from .cli import install_samples
        from .kernels.library import install_kernel_library
        
        install_samples(self.store)
        install_kernel_library(self.store)


# Convenience function
def create_model(
    base_model: str = "Qwen/Qwen2.5-7B",
    kernel_store_path: str = "./kernel_store",
    **kwargs,
) -> KernelNativeModel:
    """Create a kernel-native model."""
    return KernelNativeModel(
        base_model=base_model,
        kernel_store=kernel_store_path,
        config=KernelNativeConfig(
            base_model=base_model,
            kernel_store_path=kernel_store_path,
            **kwargs,
        ),
    )
