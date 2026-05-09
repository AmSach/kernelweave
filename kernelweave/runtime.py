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
    """Verify model output against kernel postconditions using semantic matching."""
    matched = []
    failed = []
    
    # Use semantic similarity instead of keyword matching
    for condition in postconditions:
        # Handle special postconditions that are about absence/negation
        if "not" in condition.lower() or "no " in condition.lower():
            # For negative conditions, check that the problematic thing isn't present
            # Extract the positive concept from negative condition
            positive_concept = condition.lower().replace("not ", "").replace("no ", "").strip()
            # If the positive concept appears strongly, condition fails
            # Otherwise it passes
            similarity = semantic_similarity(output, positive_concept)
            if similarity < 0.3:  # Low similarity to problematic concept = pass
                matched.append(condition)
            else:
                failed.append(condition)
        else:
            # For positive conditions, check semantic similarity
            similarity = semantic_similarity(output, condition)
            if similarity >= 0.25:  # Threshold for semantic match
                matched.append(condition)
            else:
                failed.append(condition)
    
    evidence_found = []
    evidence_missing = []
    if evidence_requirements:
        for req in evidence_requirements:
            # Use semantic similarity for evidence too
            similarity = semantic_similarity(output, req)
            coverage_score = coverage([req], output)
            # Combine semantic similarity with coverage
            combined = 0.6 * similarity + 0.4 * coverage_score
            if combined >= 0.2:
                evidence_found.append(req)
            else:
                evidence_missing.append(req)
    
    base_score = len(matched) / max(1, len(postconditions))
    evidence_score = len(evidence_found) / max(1, len(evidence_requirements)) if evidence_requirements else 1.0
    total_score = 0.7 * base_score + 0.3 * evidence_score
    
    return VerificationResult(
        passed=len(failed) == 0 and total_score >= 0.3,  # Lower threshold since we're using semantic matching
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
        self._embed_cache: dict[str, list[float] | None] = {}

    def preload_embeddings(self, texts: list[str]) -> None:
        if not self.use_embeddings or not texts:
            return
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            unseen = [text for text in texts if text not in self._embed_cache]
            if not unseen:
                return
            emb = self._embedder.encode(unseen, convert_to_numpy=True)
            for text, vector in zip(unseen, emb):
                self._embed_cache[text] = vector.tolist()
        except Exception:
            for text in texts:
                self._embed_cache.setdefault(text, None)

    def _embed_text(self, text: str) -> list[float] | None:
        if not self.use_embeddings:
            return None
        if text in self._embed_cache:
            return self._embed_cache[text]
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            emb = self._embedder.encode([text], convert_to_numpy=True)
            vector = emb[0].tolist()
            self._embed_cache[text] = vector
            return vector
        except Exception:
            self._embed_cache[text] = None
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
        
        # Check preconditions during routing (not just verification)
        precondition_penalty = self._check_routing_preconditions(prompt, kernel)
        
        base_score = 1.35 * similarity + 0.45 * evidence_bonus + 0.60 * confidence_bonus - risk_penalty - precondition_penalty
        calibrated = predict_runtime_confidence(prompt, kernel)
        return clamp(0.60 * base_score + 0.40 * calibrated)
    
    def _check_routing_preconditions(self, prompt: str, kernel) -> float:
        """Check preconditions during routing to prevent false positives."""
        penalty = 0.0
        
        # Check for artifact-scoping preconditions
        for precond in kernel.preconditions:
            if "files" in precond.lower() or "documents" in precond.lower() or "artifacts" in precond.lower():
                # Kernel requires file/document inputs
                # Check if prompt mentions files or documents
                prompt_lower = prompt.lower()
                file_indicators = ["file", "document", ".py", ".md", ".txt", ".json", ".yaml", 
                                   ".yml", ".csv", ".xml", "report", "config", "dockerfile",
                                   "version", "v1", "v2", "artifact"]
                has_file_indicator = any(ind in prompt_lower for ind in file_indicators)
                if not has_file_indicator:
                    penalty += 0.3  # Significant penalty for missing artifact indicators
        
        return penalty

    def _check_preconditions(self, prompt: str, kernel) -> bool:
        """Check if prompt satisfies kernel preconditions.
        
        This prevents false positives where a prompt matches the task family
        semantically but violates the kernel's requirements.
        
        Example: "Compare apples and oranges" matches comparison kernel
        semantically but violates "inputs are named files, schemas, or documents".
        """
        if not kernel.preconditions:
            return True
        
        prompt_lower = prompt.lower()
        
        for condition in kernel.preconditions:
            # Extract key requirements from preconditions
            if "named files" in condition.lower() or "files" in condition.lower():
                # Check if prompt references actual files
                if not any(kw in prompt_lower for kw in ["file", ".py", ".js", ".json", ".yaml", ".md", ".txt", "/"]):
                    return False
            elif "schema" in condition.lower():
                if not any(kw in prompt_lower for kw in ["schema", "json", "structure", "fields"]):
                    return False
            elif "documents" in condition.lower():
                if not any(kw in prompt_lower for kw in ["document", "doc", "file", "text"]):
                    return False
        
        return True

    def evaluate_prompt(self, prompt: str) -> RuntimeDecision:
        """Evaluate prompt with composition fallback.
        
        If no single kernel scores above threshold but two kernels
        together cover the prompt's semantic space, compose them.
        """
        kernels = self.store.list_kernels()
        candidates = []
        
        for item in kernels:
            kernel = self.store.get_kernel(item["kernel_id"])
            if not self._check_preconditions(prompt, kernel):
                continue
            score = self.score_prompt_against_kernel(prompt, kernel)
            if score > 0.3:  # Lower threshold for composition candidates
                candidates.append((kernel, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Try single kernel match first
        if candidates and candidates[0][1] >= 0.50:
            best = candidates[0][0]
            return RuntimeDecision(
                mode="kernel",
                kernel_id=best.kernel_id,
                confidence=best.status.confidence,
                reason=f"matched {best.task_family}",
                score=candidates[0][1],
            )
        
        # Composition fallback: if top 2 kernels together cover prompt
        if len(candidates) >= 2:
            from .compose import compose_sequence
            k1, s1 = candidates[0]
            k2, s2 = candidates[1]
            combined_score = 0.6 * max(s1, s2) + 0.4 * min(s1, s2)
            if combined_score >= 0.45:
                composite = compose_sequence(k1, k2)
                return RuntimeDecision(
                    mode="kernel",
                    kernel_id=composite.kernel.kernel_id,
                    confidence=composite.kernel.status.confidence,
                    reason=f"composed {k1.task_family} + {k2.task_family}",
                    score=combined_score,
                )
        
        # Fall through to generate
        return RuntimeDecision(
            mode="generate",
            kernel_id=candidates[0][0].kernel_id if candidates else None,
            confidence=candidates[0][0].status.confidence if candidates else 0.0,
            reason="no kernel match above threshold",
            score=candidates[0][1] if candidates else 0.0,
        )

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
