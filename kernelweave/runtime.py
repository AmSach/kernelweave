from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from .kernel import Kernel, KernelStore


@dataclass
class RuntimeDecision:
    mode: str
    kernel_id: str | None
    confidence: float
    reason: str


class KernelRuntime:
    def __init__(self, store: KernelStore):
        self.store = store

    def evaluate_prompt(self, prompt: str) -> RuntimeDecision:
        kernels = self.store.list_kernels()
        best = None
        best_score = -1.0
        prompt_tokens = set(prompt.lower().split())
        for item in kernels:
            kernel = self.store.get_kernel(item["kernel_id"])
            task_tokens = set(kernel.task_family.lower().split()) | set(kernel.description.lower().split())
            overlap = len(prompt_tokens & task_tokens)
            score = overlap + kernel.status.confidence
            if score > best_score:
                best_score = score
                best = kernel
        if best is None:
            return RuntimeDecision(mode="generate", kernel_id=None, confidence=0.0, reason="no kernels available")
        if best_score >= 2.0:
            return RuntimeDecision(mode="kernel", kernel_id=best.kernel_id, confidence=best.status.confidence, reason=f"matched {best.task_family}")
        return RuntimeDecision(mode="generate", kernel_id=best.kernel_id, confidence=best.status.confidence, reason="match too weak")

    def run(self, prompt: str) -> dict[str, Any]:
        decision = self.evaluate_prompt(prompt)
        if decision.mode == "kernel" and decision.kernel_id:
            kernel = self.store.get_kernel(decision.kernel_id)
            return {
                "mode": "kernel",
                "kernel_id": kernel.kernel_id,
                "kernel_name": kernel.name,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "plan": kernel.steps,
                "preconditions": kernel.preconditions,
                "postconditions": kernel.postconditions,
            }
        return {
            "mode": "generate",
            "kernel_id": decision.kernel_id,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "prompt": prompt,
            "fallback": "raw-model-generation",
        }


def plan_for_prompt(store: KernelStore, prompt: str) -> str:
    runtime = KernelRuntime(store)
    return json.dumps(runtime.run(prompt), indent=2, sort_keys=True)
