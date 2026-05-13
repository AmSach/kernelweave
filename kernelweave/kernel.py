from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

from .metrics import normalize_text


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    payload: dict[str, Any]


@dataclass
class KernelStatus:
    state: str
    confidence: float
    failures: int
    passes: int


@dataclass
class RuntimeFeedback:
    feedback_id: str
    sequence: int
    prompt: str
    task_family: str
    kernel_id: str | None
    mode: str
    reason: str
    confidence: float
    evidence_debt: float
    observed: dict[str, Any]
    response_text: str = ""
    success: bool = False
    auto_promoted_kernel_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeFeedback":
        return cls(
            feedback_id=str(data["feedback_id"]),
            sequence=int(data.get("sequence", 0)),
            prompt=str(data.get("prompt", "")),
            task_family=str(data.get("task_family", "")),
            kernel_id=data.get("kernel_id"),
            mode=str(data.get("mode", "")),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
            evidence_debt=float(data.get("evidence_debt", 0.0)),
            observed=dict(data.get("observed", {})),
            response_text=str(data.get("response_text", "")),
            success=bool(data.get("success", False)),
            auto_promoted_kernel_id=data.get("auto_promoted_kernel_id"),
        )


@dataclass
class Kernel:
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
    status: KernelStatus
    source_trace_ids: list[str]
    version: int = 2

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["digest"] = self.digest()
        return payload

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "Kernel":
        data = json.loads(path.read_text())
        status = KernelStatus(**data["status"])
        return cls(
            kernel_id=data["kernel_id"],
            name=data["name"],
            task_family=data["task_family"],
            description=data["description"],
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            preconditions=list(data.get("preconditions", [])),
            postconditions=list(data.get("postconditions", [])),
            steps=list(data.get("steps", [])),
            rollback=list(data.get("rollback", [])),
            evidence_requirements=list(data.get("evidence_requirements", [])),
            tests=list(data.get("tests", [])),
            status=status,
            source_trace_ids=list(data.get("source_trace_ids", [])),
            version=int(data.get("version", 2)),
        )


class KernelStore:
    def __init__(self, root: Path):
        self.root = root
        self.kernels_dir = root / "kernels"
        self.traces_dir = root / "traces"
        self.feedback_dir = root / "feedback"
        self.index_path = root / "index.json"
        self.feedback_index_path = root / "feedback_index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.kernels_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_json_file(self.index_path, {"kernels": [], "traces": []})
        self._ensure_json_file(self.feedback_index_path, {"feedback": []})

    def _ensure_json_file(self, path: Path, default: dict[str, Any]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            path.write_text(json.dumps(default, indent=2, sort_keys=True))
            return
        try:
            json.loads(path.read_text())
        except Exception:
            path.write_text(json.dumps(default, indent=2, sort_keys=True))

    def _read_index(self) -> dict[str, Any]:
        try:
            data = json.loads(self.index_path.read_text())
            if not isinstance(data, dict):
                raise ValueError("index must be an object")
            data.setdefault("kernels", [])
            data.setdefault("traces", [])
            return data
        except Exception:
            default = {"kernels": [], "traces": []}
            self.index_path.write_text(json.dumps(default, indent=2, sort_keys=True))
            return default

    def _write_index(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True))

    def _read_feedback_index(self) -> dict[str, Any]:
        try:
            data = json.loads(self.feedback_index_path.read_text())
            if not isinstance(data, dict):
                raise ValueError("feedback index must be an object")
            data.setdefault("feedback", [])
            return data
        except Exception:
            default = {"feedback": []}
            self.feedback_index_path.write_text(json.dumps(default, indent=2, sort_keys=True))
            return default

    def _write_feedback_index(self, index: dict[str, Any]) -> None:
        self.feedback_index_path.write_text(json.dumps(index, indent=2, sort_keys=True))

    def add_kernel(self, kernel: Kernel) -> Path:
        path = self.kernels_dir / f"{kernel.kernel_id}.json"
        kernel.save(path)
        index = self._read_index()
        kernels = {item["kernel_id"]: item for item in index.get("kernels", [])}
        kernels[kernel.kernel_id] = {
            "kernel_id": kernel.kernel_id,
            "name": kernel.name,
            "task_family": kernel.task_family,
            "digest": kernel.digest(),
            "version": kernel.version,
            "status": kernel.status.state,
            "path": str(path.relative_to(self.root)),
        }
        index["kernels"] = sorted(kernels.values(), key=lambda item: item["kernel_id"])
        self._write_index(index)
        return path

    def list_kernels(self) -> list[dict[str, Any]]:
        return self._read_index().get("kernels", [])

    def get_kernel(self, kernel_id: str) -> Kernel:
        return Kernel.load(self.kernels_dir / f"{kernel_id}.json")

    def add_trace(self, trace_id: str, trace: dict[str, Any]) -> Path:
        path = self.traces_dir / f"{trace_id}.json"
        path.write_text(json.dumps(trace, indent=2, sort_keys=True))
        index = self._read_index()
        traces = {item["trace_id"]: item for item in index.get("traces", [])}
        traces[trace_id] = {"trace_id": trace_id, "path": str(path.relative_to(self.root))}
        index["traces"] = sorted(traces.values(), key=lambda item: item["trace_id"])
        self._write_index(index)
        return path

    def list_traces(self) -> list[dict[str, Any]]:
        return self._read_index().get("traces", [])

    def list_feedback(self) -> list[dict[str, Any]]:
        return self._read_feedback_index().get("feedback", [])

    def _feedback_records_for_task_family(self, task_family: str) -> list[RuntimeFeedback]:
        records = []
        for item in self.list_feedback():
            if item.get("task_family") != task_family:
                continue
            records.append(self.get_feedback(item["feedback_id"]))
        return records

    def get_feedback(self, feedback_id: str) -> RuntimeFeedback:
        return RuntimeFeedback.from_dict(json.loads((self.feedback_dir / f"{feedback_id}.json").read_text()))

    def _normalize_task_family(self, prompt: str, task_family: str = "") -> str:
        if task_family.strip():
            return task_family.strip()
        tokens = normalize_text(prompt).split()
        if not tokens:
            return "general"
        return " ".join(tokens[:6])

    def _save_feedback(self, feedback: RuntimeFeedback) -> None:
        path = self.feedback_dir / f"{feedback.feedback_id}.json"
        path.write_text(json.dumps(feedback.to_dict(), indent=2, sort_keys=True))
        index = self._read_feedback_index()
        records = {item["feedback_id"]: item for item in index.get("feedback", [])}
        records[feedback.feedback_id] = {
            "feedback_id": feedback.feedback_id,
            "sequence": feedback.sequence,
            "task_family": feedback.task_family,
            "kernel_id": feedback.kernel_id,
            "mode": feedback.mode,
            "confidence": feedback.confidence,
            "evidence_debt": feedback.evidence_debt,
            "success": feedback.success,
            "auto_promoted_kernel_id": feedback.auto_promoted_kernel_id,
            "path": str(path.relative_to(self.root)),
        }
        index["feedback"] = sorted(records.values(), key=lambda item: (item.get("sequence", 0), item["feedback_id"]))
        self._write_feedback_index(index)

    def _maybe_auto_promote_feedback(self, feedback: RuntimeFeedback) -> str | None:
        if feedback.mode == "training":
            return None
        if not feedback.success:
            return None
        group = self._feedback_records_for_task_family(feedback.task_family)
        recent = [item for item in group if item.success][-3:]
        if len(recent) < 3:
            return None
        avg_confidence = sum(item.confidence for item in recent) / len(recent)
        avg_debt = sum(item.evidence_debt for item in recent) / len(recent)
        if avg_confidence < 0.75 or avg_debt > 0.35:
            return None

        from .compiler import compile_trace_to_kernel

        family = feedback.task_family
        summary = f"Auto-promoted reusable routine for {family}"
        evidence_blob = "; ".join(sorted({item.reason for item in recent if item.reason})) or "repeated successful interactions"
        response_text = feedback.response_text or feedback.reason or "success"
        trace_id = f"feedback-{hashlib.sha256(('|'.join(item.feedback_id for item in recent) + family).encode('utf-8')).hexdigest()[:16]}"
        events = [
            TraceEvent(kind="plan", payload={"text": recent[-1].prompt}),
            TraceEvent(kind="evidence", payload={"text": evidence_blob}),
            TraceEvent(kind="verification", payload={"text": f"avg_confidence={avg_confidence:.3f}; avg_evidence_debt={avg_debt:.3f}"}),
            TraceEvent(kind="decision", payload={"text": response_text}),
        ]
        kernel = compile_trace_to_kernel(trace_id, family, summary, events, expected_output={"result": response_text[:256]})
        kernel.source_trace_ids = [item.feedback_id for item in recent]
        self.add_kernel(kernel)
        for item in recent:
            item.auto_promoted_kernel_id = kernel.kernel_id
            self._save_feedback(item)
        return kernel.kernel_id

    def record_runtime_feedback(
        self,
        prompt: str,
        kernel_id: str | None,
        mode: str,
        reason: str,
        confidence: float,
        evidence_debt: float,
        observed: dict[str, Any] | None = None,
        task_family: str = "",
        response_text: str = "",
    ) -> dict[str, Any]:
        observed = dict(observed or {})
        index = self._read_feedback_index()
        sequence = len(index.get("feedback", [])) + 1
        family = self._normalize_task_family(prompt, task_family or str(observed.get("task_family", "")))
        feedback_id = hashlib.sha256(
            f"{sequence}:{prompt}:{kernel_id or ''}:{mode}:{family}".encode("utf-8")
        ).hexdigest()[:16]
        success_hint = observed.get("success")
        success = bool(success_hint) if success_hint is not None else (mode != "training" and confidence >= 0.75 and evidence_debt <= 0.35 and bool(response_text.strip()))
        feedback = RuntimeFeedback(
            feedback_id=feedback_id,
            sequence=sequence,
            prompt=prompt,
            task_family=family,
            kernel_id=kernel_id,
            mode=mode,
            reason=reason,
            confidence=confidence,
            evidence_debt=evidence_debt,
            observed=observed,
            response_text=response_text,
            success=success,
        )
        self._save_feedback(feedback)
        promoted_kernel_id = self._maybe_auto_promote_feedback(feedback)
        if promoted_kernel_id is not None:
            feedback.auto_promoted_kernel_id = promoted_kernel_id
            self._save_feedback(feedback)
        return feedback.to_dict()

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "kernels": len(self.list_kernels()),
            "traces": len(self.list_traces()),
            "feedback": len(self.list_feedback()),
        }


def load_sample_store(root: Path) -> KernelStore:
    return KernelStore(root)
