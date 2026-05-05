from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json


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
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            preconditions=list(data["preconditions"]),
            postconditions=list(data["postconditions"]),
            steps=list(data["steps"]),
            rollback=list(data["rollback"]),
            evidence_requirements=list(data["evidence_requirements"]),
            tests=list(data["tests"]),
            status=status,
            source_trace_ids=list(data["source_trace_ids"]),
            version=int(data.get("version", 2)),
        )


class KernelStore:
    def __init__(self, root: Path):
        self.root = root
        self.kernels_dir = root / "kernels"
        self.traces_dir = root / "traces"
        self.index_path = root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.kernels_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"kernels": [], "traces": []}, indent=2))

    def _read_index(self) -> dict[str, Any]:
        return json.loads(self.index_path.read_text())

    def _write_index(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True))

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

    def summary(self) -> dict[str, Any]:
        return {"root": str(self.root), "kernels": len(self.list_kernels()), "traces": len(self.list_traces())}


def load_sample_store(root: Path) -> KernelStore:
    return KernelStore(root)
