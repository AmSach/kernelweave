from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from .kernel import KernelStore
from .metrics import clamp, cosine_similarity, coverage, jaccard_similarity, semantic_similarity
from .calibration import predict_runtime_confidence, runtime_features


@dataclass
class RuntimeDecision:
    mode: str
    kernel_id: str | None
    confidence: float
    reason: str
    score: float


@dataclass
class VerificationResult:
    passed: bool
    score: float
    matched_conditions: list[str]
    failed_conditions: list[str]
    evidence_found: list[str]
    evidence_missing: list[str]


def verify_output_against_postconditions(output: str, postconditions: list[str], evidence_requirements: list[str] | None = None) -> VerificationResult:
    """Verify model output against kernel postconditions and evidence requirements."""
    output_lower = output.lower()
    matched = []
    failed = []
    
    for condition in postconditions:
        condition_lower = condition.lower()
        keywords = re.findall(r'\b\w{3,}\b', condition_lower)
        if any(kw in output_lower for kw in keywords):
            matched.append(condition)
        else:
            failed.append(condition)
    
    evidence_found = []
    evidence_missing = []
    if evidence_requirements:
        for req in evidence_requirements:
            req_lower = req.lower()
            keywords = re.findall(r'\b\w{3,}\b', req_lower)
            if any(kw in output_lower for kw in keywords):
                evidence_found.append(req)
            else:
                evidence_missing.append(req)
    
    base_score = len(matched) / max(1, len(postconditions))
    evidence_score = len(evidence_found) / max(1, len(evidence_requirements)) if evidence_requirements else 1.0
    total_score = 0.7 * base_score + 0.3 * evidence_score
    
    return VerificationResult(
        passed=len(failed) == 0 and total_score >= 0.5,
        score=clamp(total_score),
        matched_conditions=matched,
        failed_conditions=failed,
        evidence_found=evidence_found,
        evidence_missing=evidence_missing
    )


class KernelRuntime:
    def __init__(self, store: KernelStore, use_embeddings: bool = True):
        self.store = store
        self.use_embeddings = use_embeddings
        self._embedder = None

    def _embed_text(self, text: str) -> list[float] | None:
        if not self.use_embeddings:
            return None
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            emb = self._embedder.encode([text], convert_to_numpy=True)
            return emb[0].tolist()
        except Exception:
            return None

    def _embedding_similarity(self, prompt: str, kernel_text: str) -> float | None:
        prompt_emb = self._embed_text(prompt)
        kernel_emb = self._embed_text(kernel_text)
        if prompt_emb is None or kernel_emb is None:
            return None
        import math
        dot = sum(a * b for a, b in zip(prompt_emb, kernel_emb))
        norm_a = math.sqrt(sum(a * a for a in prompt_emb))
        norm_b = math.sqrt(sum(b * b for b in kernel_emb))
        if norm_a == 0 or norm_b == 0:
            return None
        return dot / (norm_a * norm_b)

    def score_prompt_against_kernel(self, prompt: str, kernel) -> float:
        kernel_text = f"{kernel.task_family} {kernel.description}"
        embedding_sim = self._embedding_similarity(prompt, kernel_text)
        if embedding_sim is not None:
            similarity = embedding_sim
        else:
            similarity = 0.6 * cosine_similarity(prompt, kernel_text) + 0.4 * jaccard_similarity(prompt, kernel.description)
        evidence_bonus = coverage(kernel.evidence_requirements, prompt)
        confidence_bonus = kernel.status.confidence
        risk_penalty = 0.15 if kernel.status.state != "verified" else 0.0
        base_score = 1.35 * similarity + 0.45 * evidence_bonus + 0.60 * confidence_bonus - risk_penalty
        calibrated = predict_runtime_confidence(prompt, kernel)
        return clamp(0.60 * base_score + 0.40 * calibrated)

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
        if best_score >= 0.50:
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
                "evidence_requirements": kernel.evidence_requirements,
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


class ExecutionEngine:
    def __init__(self, store: KernelStore | None = None, backend: Any = None):
        self.store = store
        self.backend = backend

    def execute_kernel(self, kernel, prompt: str) -> dict[str, Any]:
        """Execute a kernel against the backend and return results with verification."""
        if self.backend is None:
            return {
                "executed": False,
                "error": "no backend available",
                "mode": "simulated",
            }
        
        steps_text = "\n".join(
            f"{i+1}. {step.get('action', 'step')}: {step.get('text', step.get('tool', ''))}"
            for i, step in enumerate(kernel.steps)
        )
        
        system_prompt = (
            f"Execute the following kernel plan for task family '{kernel.task_family}':\n"
            f"{steps_text}\n\n"
            f"Preconditions: {', '.join(kernel.preconditions)}\n"
            f"Evidence requirements: {', '.join(kernel.evidence_requirements)}\n"
            f"Expected output: satisfy the postconditions: {', '.join(kernel.postconditions)}\n\n"
            f"After completing the plan, explicitly state what evidence you found and verify each postcondition."
        )
        
        try:
            response = self.backend.generate(prompt, system_prompt=system_prompt)
            output_text = response.text
            
            verification = verify_output_against_postconditions(
                output_text,
                kernel.postconditions,
                kernel.evidence_requirements
            )
            
            return {
                "executed": True,
                "mode": "real",
                "response_text": output_text,
                "verification": {
                    "passed": verification.passed,
                    "score": verification.score,
                    "matched_conditions": verification.matched_conditions,
                    "failed_conditions": verification.failed_conditions,
                    "evidence_found": verification.evidence_found,
                    "evidence_missing": verification.evidence_missing,
                },
                "plan": kernel.steps,
            }
        except Exception as e:
            return {
                "executed": False,
                "mode": "error",
                "error": str(e),
            }

    def execute_plan(self, plan: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Execute a routing plan, running kernels through the backend if available."""
        if self.store is None:
            return {
                "mode": plan.get("mode", "generate"),
                "kernel_id": plan.get("kernel_id"),
                "reason": plan.get("reason", "no store available"),
                "confidence": plan.get("confidence", 0.0),
                "prompt": prompt,
                "execution": "no-store",
            }
        
        mode = plan.get("mode", "generate")
        kernel_id = plan.get("kernel_id")
        
        if mode == "kernel" and kernel_id and self.backend is not None:
            kernel = self.store.get_kernel(kernel_id)
            execution_result = self.execute_kernel(kernel, prompt)
            
            if execution_result.get("executed"):
                verification = execution_result.get("verification", {})
                success = verification.get("passed", False)
                score = verification.get("score", 0.0)
                
                if self.store is not None:
                    self.store.record_runtime_feedback(
                        prompt=prompt,
                        kernel_id=kernel_id,
                        mode="kernel",
                        reason="executed and verified",
                        confidence=score,
                        evidence_debt=1.0 - score,
                        task_family=kernel.task_family,
                        response_text=execution_result.get("response_text", ""),
                        observed={
                            "success": success,
                            "verification_score": score,
                            "matched_conditions": verification.get("matched_conditions", []),
                            "failed_conditions": verification.get("failed_conditions", []),
                        }
                    )
            
            return {
                "mode": "kernel",
                "kernel_id": kernel_id,
                "confidence": plan.get("confidence", 0.0),
                "prompt": prompt,
                **execution_result,
            }
        
        runtime = KernelRuntime(self.store)
        result = runtime.run(prompt)
        result["requested_plan"] = plan
        return result


def plan_for_prompt(store: KernelStore, prompt: str) -> str:
    runtime = KernelRuntime(store)
    return json.dumps(runtime.run(prompt), indent=2, sort_keys=True)
