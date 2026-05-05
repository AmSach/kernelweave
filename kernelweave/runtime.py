from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from .kernel import KernelStore
from .metrics import clamp, cosine_similarity, coverage, jaccard_similarity


@dataclass
class RuntimeDecision:
    mode: str
    kernel_id: str | None
    confidence: float
    reason: str
    score: float


class KernelRuntime:
    def __init__(self, store: KernelStore):
        self.store = store

    def score_prompt_against_kernel(self, prompt: str, kernel) -> float:
        similarity = 0.6 * cosine_similarity(prompt, f"{kernel.task_family} {kernel.description}") + 0.4 * jaccard_similarity(prompt, kernel.description)
        evidence_bonus = coverage(kernel.evidence_requirements, prompt)
        confidence_bonus = kernel.status.confidence
        risk_penalty = 0.15 if kernel.status.state != "verified" else 0.0
        score = 1.35 * similarity + 0.45 * evidence_bonus + 0.60 * confidence_bonus - risk_penalty
        return clamp(score)

    def evaluate_prompt(self, prompt: str) -> RuntimeDecision:
        kernels = self.store.list_kernels()
        best = None
        best_score = -1.0
        for item in kernels:
            kernel = self.store.get_kernel(item["kernel_id"])
            score = self.score_prompt_against_kernel(prompt, kernel)
            if score > best_score:
                best_score = score
                best = kernel
        if best is None:
            return RuntimeDecision(mode="generate", kernel_id=None, confidence=0.0, reason="no kernels available", score=0.0)
        if best_score >= 0.68:
            return RuntimeDecision(mode="kernel", kernel_id=best.kernel_id, confidence=best.status.confidence, reason=f"matched {best.task_family}", score=best_score)
        return RuntimeDecision(mode="generate", kernel_id=best.kernel_id, confidence=best.status.confidence, reason="match too weak", score=best_score)

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
                "score": decision.score,
                "plan": kernel.steps,
                "preconditions": kernel.preconditions,
                "postconditions": kernel.postconditions,
                "rollback": kernel.rollback,
            }
        return {
            "mode": "generate",
            "kernel_id": decision.kernel_id,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "score": decision.score,
            "prompt": prompt,
            "fallback": "raw-model-generation",
        }


def plan_for_prompt(store: KernelStore, prompt: str) -> str:
    runtime = KernelRuntime(store)
    return json.dumps(runtime.run(prompt), indent=2, sort_keys=True)
