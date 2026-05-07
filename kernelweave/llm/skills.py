from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json

from ..calibration import predict_runtime_confidence
from ..compiler import compile_trace_to_kernel
from ..kernel import Kernel, TraceEvent
from ..metrics import clamp, conflict_terms, jaccard_similarity, normalize_text, semantic_similarity


@dataclass
class SkillKernel:
    kernel_id: str
    name: str
    task_family: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    preconditions: list[str]
    postconditions: list[str]
    steps: list[dict[str, Any]]
    rollback: list[str]
    evidence_requirements: list[str]
    tests: list[dict[str, Any]]
    confidence: float
    source_trace_ids: list[str]
    tags: list[str] = field(default_factory=list)
    rationale: str = ""
    version: int = 1
    drift_penalty: float = 0.0
    active: bool = True

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["digest"] = self.digest()
        return payload

    @classmethod
    def from_kernel(cls, kernel: Kernel) -> "SkillKernel":
        return cls(
            kernel_id=kernel.kernel_id,
            name=kernel.name,
            task_family=kernel.task_family,
            description=kernel.description,
            input_schema=kernel.input_schema,
            output_schema=kernel.output_schema,
            preconditions=list(kernel.preconditions),
            postconditions=list(kernel.postconditions),
            steps=list(kernel.steps),
            rollback=list(kernel.rollback),
            evidence_requirements=list(kernel.evidence_requirements),
            tests=list(kernel.tests),
            confidence=kernel.status.confidence,
            source_trace_ids=list(kernel.source_trace_ids),
            tags=list(kernel.tags or []),
            rationale=kernel.rationale,
            version=kernel.version,
            drift_penalty=kernel.status.drift_penalty,
            active=kernel.status.state not in {"retired", "rejected"},
        )

    def to_kernel(self) -> Kernel:
        from ..kernel import KernelStatus

        return Kernel(
            kernel_id=self.kernel_id,
            name=self.name,
            task_family=self.task_family,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            preconditions=list(self.preconditions),
            postconditions=list(self.postconditions),
            steps=list(self.steps),
            rollback=list(self.rollback),
            evidence_requirements=list(self.evidence_requirements),
            tests=list(self.tests),
            status=KernelStatus(
                state="verified" if self.active else "retired",
                confidence=self.confidence,
                failures=0,
                passes=1,
                drift_penalty=self.drift_penalty,
                version=self.version,
            ),
            source_trace_ids=list(self.source_trace_ids),
            version=self.version,
            tags=list(self.tags),
            rationale=self.rationale,
        )

    def score_prompt(self, prompt: str) -> dict[str, float]:
        semantic = 0.45 * semantic_similarity(prompt, self.task_family) + 0.25 * jaccard_similarity(prompt, self.description)
        evidence = 0.30 * semantic_similarity(prompt, " ".join(self.evidence_requirements)) + 0.70 * clamp(len(self.evidence_requirements) / 6.0)
        novelty = clamp(1.0 - jaccard_similarity(normalize_text(prompt), normalize_text(self.description)))
        confidence = clamp(self.confidence)
        drift = clamp(self.drift_penalty)
        conflicts = conflict_terms(prompt) & (conflict_terms(self.task_family) | conflict_terms(self.description))
        score = clamp(0.50 * semantic + 0.15 * evidence + 0.18 * confidence + 0.10 * novelty - 0.18 * drift - 0.12 * len(conflicts))
        debt = clamp(1.0 - evidence + drift * 0.5 + 0.10 * len(conflicts))
        calibrated = predict_runtime_confidence(prompt, self.to_kernel())
        return {"score": score, "evidence": evidence, "debt": debt, "semantic": semantic, "novelty": novelty, "confidence": confidence, "calibrated": calibrated}


@dataclass
class SkillRoute:
    mode: str
    kernel_id: str | None
    confidence: float
    reason: str
    evidence_debt: float
    selected_kernel: str | None = None
    curiosity_questions: list[str] = field(default_factory=list)


@dataclass
class SkillBankManifest:
    kernels: list[dict[str, Any]]
    summary: dict[str, Any]


class SkillKernelBank:
    def __init__(self, kernels: Iterable[SkillKernel] | None = None):
        self.kernels: dict[str, SkillKernel] = {}
        for kernel in kernels or []:
            self.register(kernel)

    def register(self, kernel: SkillKernel) -> SkillKernel:
        self.kernels[kernel.kernel_id] = kernel
        return kernel

    def import_kernel(self, kernel: Kernel) -> SkillKernel:
        return self.register(SkillKernel.from_kernel(kernel))

    def import_from_store(self, store: Any) -> None:
        for item in store.list_kernels():
            kernel = store.get_kernel(item["kernel_id"])
            self.import_kernel(kernel)

    def promote_trace(
        self,
        trace_id: str,
        task_family: str,
        description: str,
        events: list[TraceEvent],
        expected_output: dict[str, Any],
        tags: list[str] | None = None,
        rationale: str = "",
    ) -> SkillKernel:
        kernel = compile_trace_to_kernel(trace_id, task_family, description, events, expected_output)
        skill = SkillKernel.from_kernel(kernel)
        if tags:
            skill.tags = list(tags)
        skill.rationale = rationale or skill.rationale
        return self.register(skill)

    def list(self) -> list[SkillKernel]:
        return sorted(self.kernels.values(), key=lambda item: item.kernel_id)

    def summary(self) -> dict[str, Any]:
        kernels = self.list()
        avg_conf = sum(item.confidence for item in kernels) / max(1, len(kernels))
        return {
            "kernels": len(kernels),
            "average_confidence": round(avg_conf, 4),
            "active_kernels": sum(1 for item in kernels if item.active),
            "task_families": sorted({item.task_family for item in kernels}),
            "top_kernels": [item.task_family for item in kernels[:5]],
        }

    def curiosity_questions(self, prompt: str) -> list[str]:
        tokens = [token for token in normalize_text(prompt).split() if token]
        if not tokens:
            return ["What is the actual goal?", "What evidence is missing?", "What would a safe fallback look like?"]
        head = tokens[:6]
        return [
            f"What hidden constraint is implied by: {' '.join(head)}?",
            f"What evidence would prove the prompt about {' '.join(head[:3])}?",
            f"What is the cheapest safe answer if {' '.join(head[:2])} turns out ambiguous?",
        ]

    def route(self, prompt: str) -> SkillRoute:
        if not self.kernels:
            return SkillRoute(mode="generate", kernel_id=None, confidence=0.0, reason="no internal skills", evidence_debt=1.0)

        best_kernel: SkillKernel | None = None
        best_stats: dict[str, float] | None = None
        best_score = -1.0
        alternatives: list[tuple[float, SkillKernel, dict[str, float]]] = []
        for kernel in self.kernels.values():
            if not kernel.active:
                continue
            stats = kernel.score_prompt(prompt)
            score = 0.65 * stats["score"] + 0.35 * stats["calibrated"]
            alternatives.append((score, kernel, stats))
            if score > best_score:
                best_score = score
                best_kernel = kernel
                best_stats = stats

        if best_kernel is None or best_stats is None:
            return SkillRoute(mode="generate", kernel_id=None, confidence=0.0, reason="no active skills", evidence_debt=1.0)

        close_matches = [item for item in alternatives if abs(item[0] - best_score) <= 0.04]
        if len(close_matches) > 1:
            best_score -= 0.04 * (len(close_matches) - 1)
        if any((conflict_terms(prompt) & (conflict_terms(kernel.task_family) | conflict_terms(kernel.description))) for _, kernel, _ in close_matches):
            best_score -= 0.08

        should_execute = best_score >= 0.18 and best_stats["debt"] <= 0.85 and best_stats["confidence"] >= 0.30
        reason = f"matched {best_kernel.task_family}" if should_execute else "match too weak or evidence sparse"
        mode = "skill" if should_execute else "generate"
        return SkillRoute(
            mode=mode,
            kernel_id=best_kernel.kernel_id,
            confidence=clamp(best_score),
            reason=reason,
            evidence_debt=best_stats["debt"],
            selected_kernel=best_kernel.kernel_id,
            curiosity_questions=[] if should_execute else self.curiosity_questions(prompt),
        )

    def promote_from_trace(self, trace_id: str, task_family: str, description: str, events: list[TraceEvent], expected_output: dict[str, Any], tags: list[str] | None = None, rationale: str = "") -> SkillKernel:
        return self.promote_trace(trace_id, task_family, description, events, expected_output, tags=tags, rationale=rationale)

    def promote_from_store_trace(self, trace_id: str, store: Any, task_family: str, description: str, events: list[TraceEvent], expected_output: dict[str, Any]) -> SkillKernel:
        skill = self.promote_trace(trace_id, task_family, description, events, expected_output)
        if hasattr(store, "record_runtime_feedback"):
            store.record_runtime_feedback(description or task_family, skill.kernel_id, "skill", "promoted from trace", skill.confidence, 0.0, observed={"trace_id": trace_id})
        return skill

    def demote(self, kernel_id: str, reason: str = "") -> None:
        kernel = self.kernels[kernel_id]
        kernel.active = False
        if reason:
            kernel.rationale = f"{kernel.rationale}\n{reason}".strip()

    def promote(self, kernel_id: str) -> None:
        self.kernels[kernel_id].active = True

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = SkillBankManifest(kernels=[kernel.to_dict() for kernel in self.list()], summary=self.summary())
        path.write_text(json.dumps(payload.__dict__, indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "SkillKernelBank":
        payload = json.loads(path.read_text())
        kernels = []
        for item in payload.get("kernels", []):
            skill = SkillKernel(
                kernel_id=item["kernel_id"],
                name=item["name"],
                task_family=item["task_family"],
                description=item["description"],
                input_schema=item["input_schema"],
                output_schema=item["output_schema"],
                preconditions=list(item["preconditions"]),
                postconditions=list(item["postconditions"]),
                steps=list(item["steps"]),
                rollback=list(item["rollback"]),
                evidence_requirements=list(item["evidence_requirements"]),
                tests=list(item["tests"]),
                confidence=float(item.get("confidence", 0.0)),
                source_trace_ids=list(item.get("source_trace_ids", [])),
                tags=list(item.get("tags", [])),
                rationale=str(item.get("rationale", "")),
                version=int(item.get("version", 1)),
                drift_penalty=float(item.get("drift_penalty", 0.0)),
                active=bool(item.get("active", True)),
            )
            kernels.append(skill)
        return cls(kernels)
