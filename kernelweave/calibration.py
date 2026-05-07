from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
import json
import math

from .metrics import clamp, conflict_terms, infer_task_family, semantic_profile, semantic_similarity, jaccard_similarity, coverage


@dataclass
class CalibrationExample:
    features: dict[str, float]
    label: float
    note: str = ""


@dataclass
class CalibrationModel:
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    feature_order: tuple[str, ...] = ()

    def _dot(self, features: dict[str, float]) -> float:
        score = self.bias
        for key in self.feature_order:
            score += self.weights.get(key, 0.0) * float(features.get(key, 0.0))
        return score

    def predict(self, features: dict[str, float]) -> float:
        return clamp(1.0 / (1.0 + math.exp(-self._dot(features))))

    def fit(self, examples: Iterable[CalibrationExample], epochs: int = 400, learning_rate: float = 0.08, l2: float = 0.002) -> "CalibrationModel":
        example_list = list(examples)
        if not example_list:
            raise ValueError("at least one calibration example is required")
        features = sorted({key for example in example_list for key in example.features})
        self.feature_order = tuple(features)
        self.weights = {key: 0.0 for key in self.feature_order}
        self.bias = 0.0

        for _ in range(epochs):
            grad_w = {key: 0.0 for key in self.feature_order}
            grad_b = 0.0
            for example in example_list:
                target = clamp(float(example.label))
                prediction = self.predict(example.features)
                error = prediction - target
                grad_b += error
                for key in self.feature_order:
                    grad_w[key] += error * float(example.features.get(key, 0.0))
            step = learning_rate / len(example_list)
            self.bias -= step * grad_b
            for key in self.feature_order:
                regularised = grad_w[key] / len(example_list) + l2 * self.weights[key]
                self.weights[key] -= learning_rate * regularised
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"bias": self.bias, "weights": dict(self.weights), "feature_order": list(self.feature_order)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationModel":
        return cls(weights={str(k): float(v) for k, v in data.get("weights", {}).items()}, bias=float(data.get("bias", 0.0)), feature_order=tuple(data.get("feature_order", [])))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


@dataclass
class RuntimeCalibrationInput:
    semantic: float
    description_semantic: float
    coverage: float
    confidence: float
    drift: float
    conflict_count: float
    support: float
    length_ratio: float
    intent_match: float

    def to_features(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "description_semantic": self.description_semantic,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "drift": self.drift,
            "conflict_count": self.conflict_count,
            "support": self.support,
            "length_ratio": self.length_ratio,
            "intent_match": self.intent_match,
        }


@dataclass
class CompileCalibrationInput:
    trace_length: float
    distinct_tools: float
    evidence_count: float
    compression_gain: float
    reuse_value: float
    failure_risk: float
    conflict_count: float
    evidence_density: float

    def to_features(self) -> dict[str, float]:
        return {
            "trace_length": self.trace_length,
            "distinct_tools": self.distinct_tools,
            "evidence_count": self.evidence_count,
            "compression_gain": self.compression_gain,
            "reuse_value": self.reuse_value,
            "failure_risk": self.failure_risk,
            "conflict_count": self.conflict_count,
            "evidence_density": self.evidence_density,
        }


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "training"
RUNTIME_CALIBRATION_FILE = CALIBRATION_DIR / "runtime_calibration.json"
COMPILE_CALIBRATION_FILE = CALIBRATION_DIR / "compile_calibration.json"


def load_calibration_examples(path: Path) -> list[CalibrationExample]:
    payload = json.loads(path.read_text())
    examples: list[CalibrationExample] = []
    for item in payload:
        if "features" not in item:
            raise ValueError(f"calibration example missing features: {item}")
        examples.append(CalibrationExample(features={str(k): float(v) for k, v in item["features"].items()}, label=float(item["label"]), note=str(item.get("note", ""))))
    return examples


def write_calibration_examples(path: Path, examples: Iterable[CalibrationExample]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"features": example.features, "label": example.label, "note": example.note}
        for example in examples
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


@lru_cache(maxsize=1)
def default_runtime_model() -> CalibrationModel:
    if RUNTIME_CALIBRATION_FILE.exists():
        model = CalibrationModel()
        model.fit(load_calibration_examples(RUNTIME_CALIBRATION_FILE))
        return model
    return CalibrationModel(weights={"semantic": 2.0, "description_semantic": 1.6, "coverage": 1.2, "confidence": 1.1, "support": 0.8, "intent_match": 0.9, "drift": -1.5, "conflict_count": -1.8, "length_ratio": 0.1}, bias=-1.1, feature_order=("semantic", "description_semantic", "coverage", "confidence", "drift", "conflict_count", "support", "length_ratio", "intent_match"))


@lru_cache(maxsize=1)
def default_compile_model() -> CalibrationModel:
    if COMPILE_CALIBRATION_FILE.exists():
        model = CalibrationModel()
        model.fit(load_calibration_examples(COMPILE_CALIBRATION_FILE))
        return model
    return CalibrationModel(weights={"trace_length": 0.6, "distinct_tools": 0.9, "evidence_count": 1.1, "compression_gain": 0.7, "reuse_value": 1.0, "failure_risk": -1.4, "conflict_count": -2.0, "evidence_density": 0.8}, bias=-0.8, feature_order=("trace_length", "distinct_tools", "evidence_count", "compression_gain", "reuse_value", "failure_risk", "conflict_count", "evidence_density"))


def runtime_features(prompt: str, kernel: Any) -> RuntimeCalibrationInput:
    prompt_profile = semantic_profile(prompt)
    kernel_profile = semantic_profile(f"{kernel.task_family} {kernel.description}")
    prompt_terms = set(prompt_profile.terms)
    kernel_terms = set(kernel_profile.terms)
    conflict_set = conflict_terms(prompt) & (conflict_terms(kernel.task_family) | conflict_terms(kernel.description))
    return RuntimeCalibrationInput(
        semantic=semantic_similarity(prompt, kernel.task_family),
        description_semantic=semantic_similarity(prompt, kernel.description),
        coverage=coverage(kernel.evidence_requirements, prompt),
        confidence=float(getattr(getattr(kernel, "status", kernel), "confidence", 0.0)),
        drift=float(getattr(getattr(kernel, "status", kernel), "drift_penalty", 0.0)),
        conflict_count=float(len(conflict_set)),
        support=jaccard_similarity(" ".join(prompt_terms), " ".join(kernel_terms)),
        length_ratio=min(1.0, max(0.0, len(prompt_terms) / max(1, len(kernel_terms)) / 2.0)),
        intent_match=1.0 if prompt_profile.intent != "general" and prompt_profile.intent == kernel_profile.intent else 0.0,
    )


def compile_features(stats: Any) -> CompileCalibrationInput:
    trace_length = float(getattr(stats, "trace_length", 0.0))
    distinct_tools = float(getattr(stats, "distinct_tools", 0.0))
    evidence_count = float(getattr(stats, "evidence_count", 0.0))
    compression_gain = float(getattr(stats, "compression_gain", 0.0))
    reuse_value = float(getattr(stats, "reuse_value", 0.0))
    failure_risk = float(getattr(stats, "failure_risk", 0.0))
    conflict_count = float(getattr(stats, "conflict_count", 0.0))
    evidence_density = evidence_count / max(1.0, trace_length)
    return CompileCalibrationInput(
        trace_length=min(1.0, trace_length / 16.0),
        distinct_tools=min(1.0, distinct_tools / 8.0),
        evidence_count=min(1.0, evidence_count / 8.0),
        compression_gain=min(1.0, compression_gain / 4.0),
        reuse_value=clamp(reuse_value),
        failure_risk=clamp(failure_risk),
        conflict_count=min(1.0, conflict_count / 4.0),
        evidence_density=min(1.0, evidence_density),
    )


def predict_runtime_confidence(prompt: str, kernel: Any) -> float:
    model = default_runtime_model()
    return model.predict(runtime_features(prompt, kernel).to_features())


def predict_compile_confidence(stats: Any) -> float:
    model = default_compile_model()
    return model.predict(compile_features(stats).to_features())


def calibration_summary() -> dict[str, Any]:
    runtime_model = default_runtime_model()
    compile_model = default_compile_model()
    return {
        "runtime": runtime_model.to_dict(),
        "compile": compile_model.to_dict(),
        "runtime_examples": RUNTIME_CALIBRATION_FILE.exists(),
        "compile_examples": COMPILE_CALIBRATION_FILE.exists(),
    }
