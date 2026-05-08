from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..calibration import CalibrationExample, CalibrationModel

TRAINING_DEPS: list[str] = []


def _ensure_deps() -> None:
    return


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class TrainingConfig:
    base_model: str = "KernelWeave/Synthetic"
    output_dir: str = "./kernel-native-model"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.08
    warmup_steps: int = 12
    max_grad_norm: float = 1.0
    n_train_samples: int = 5000
    n_eval_samples: int = 500
    max_input_length: int = 512
    max_output_length: int = 256
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSample:
    prompt: str
    response: str
    task_family: str
    kernel_id: str | None
    verification_passed: bool
    confidence: float
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSnapshot:
    epoch: int
    train_loss: float
    eval_loss: float
    eval_accuracy: float
    avg_confidence: float
    kernel_reuse_rate: float
    prompt_length_mean: float
    response_length_mean: float
    checkpoint_health: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingReport:
    model_name: str
    output_dir: str
    hardware: dict[str, Any]
    config: dict[str, Any]
    train_samples: int
    eval_samples: int
    family_counts: dict[str, int]
    history: list[dict[str, Any]]
    runtime_calibration: dict[str, Any]
    compile_calibration: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceGenerator:
    def __init__(self, kernel_store: Any = None):
        self.kernel_store = kernel_store
        self._task_templates = self._build_task_templates()
        self._response_templates = self._build_response_templates()
        self._pools = self._build_pools()

    def _build_task_templates(self) -> dict[str, list[str]]:
        return {
            "comparison": [
                "Compare {file_a} and {file_b}.",
                "What are the key differences between {item_a} and {item_b}?",
            ],
            "analysis": [
                "Analyze {file} and identify potential {issue_type}.",
                "Review {code} for {issue_type}.",
            ],
            "search": [
                "Find all {file_type} files in {directory}.",
                "Search for {pattern} in all files.",
            ],
            "generation": [
                "Generate a {format} report from {source}.",
                "Create a {format} summary of {topic}.",
            ],
            "debugging": [
                "Fix the bug in {file} that causes {error}.",
                "Debug {file} and resolve {error}.",
            ],
            "summarization": [
                "Summarize {document}.",
                "Create a summary of {document}.",
            ],
            "transformation": [
                "Convert {input} from {format_a} to {format_b}.",
                "Transform {input} into {format_b}.",
            ],
            "testing": [
                "Write tests for {module}.",
                "Create test cases for {function}.",
            ],
        }

    def _build_response_templates(self) -> dict[str, list[str]]:
        return {
            "comparison": [
                "I found {n} key differences between {file_a} and {file_b}: {diff_1}; {diff_2}; {diff_3}.",
                "Comparing {file_a} and {file_b}, the main difference is {aspect}.",
            ],
            "analysis": [
                "Analysis of {file}: {issue_1}. Recommendation: {rec_1}.",
                "I identified {n} {issue_type} in {file}: {issue_1}; {issue_2}.",
            ],
            "search": [
                "Found {n} {file_type} files in {directory}: {file_1}, {file_2}, {file_3}.",
                "Search results for '{pattern}': {file_1}, {file_2}, {file_3}.",
            ],
            "generation": [
                "# {title}\n\n{content}\n\n## Summary\n\n{summary}",
                "**Report:** {content}\n\n**Key point:** {point_1}",
            ],
            "debugging": [
                "Fixed {file}: {fix}. Root cause: {cause}.",
                "Debugging complete for {file}. Applied fix: {fix}.",
            ],
            "summarization": [
                "**Summary of {document}:** {summary}",
                "{document} mainly covers {topic}. Key point: {point_1}.",
            ],
            "transformation": [
                "Transformation complete from {format_a} to {format_b}.",
                "```{format_b}\n{output}\n```",
            ],
            "testing": [
                "```python\n{test_code}\n```\nCoverage: {coverage}",
                "Generated {n} test cases for {module}.",
            ],
        }

    def _build_pools(self) -> dict[str, list[str]]:
        return {
            "files": ["main.py", "utils.py", "config.yaml", "app.js", "index.ts", "README.md", "test.py"],
            "directories": ["src", "lib", "tests", "docs", "config", "scripts"],
            "file_types": ["Python", "JavaScript", "TypeScript", "config", "test", "documentation"],
            "formats": ["JSON", "YAML", "Markdown", "HTML", "CSV", "XML"],
            "patterns": ["TODO", "FIXME", "deprecated", "unused", "import", "function"],
            "issues": ["bugs", "security issues", "performance problems", "style issues", "unused imports"],
            "errors": ["TypeError", "ValueError", "ImportError", "AttributeError", "SyntaxError"],
            "diffs": ["different variable names", "different function signatures", "different import structure", "different class hierarchy", "different error handling"],
            "aspects": ["error handling", "performance", "code style", "architecture", "documentation", "testing"],
            "features": ["async support", "type hints", "comprehensive tests", "detailed docs", "modular design"],
            "issue_texts": ["unused variable on line 42", "missing type hint", "potential null pointer", "deprecated API usage", "missing error handling"],
            "rec_texts": ["add type hints", "implement error handling", "refactor to use modern API", "add unit tests", "improve documentation"],
            "severities": ["low", "medium", "high", "critical"],
            "good_texts": ["good test coverage", "clean code structure", "proper documentation", "modern Python practices"],
            "file_names": ["main.py", "utils.py", "config.py", "test_main.py", "helpers.py", "constants.py"],
            "titles": ["Analysis Report", "Code Review Summary", "Performance Audit", "Security Assessment", "Technical Documentation"],
            "contents": ["This document provides a comprehensive analysis.", "The following report details the key findings.", "After thorough review, we identified opportunities.", "This analysis covers the major aspects."],
            "summaries": ["Key findings indicate significant improvements needed.", "Overall, the codebase is well-structured.", "Several optimization opportunities were identified.", "The analysis reveals a solid foundation."],
            "points": ["Improved performance metrics", "Enhanced security posture", "Better code maintainability", "Clearer documentation", "More comprehensive tests"],
            "fixes": ["added null check before access", "fixed variable name typo", "added missing import", "corrected function signature", "added proper error handling"],
            "causes": ["null reference without check", "variable naming inconsistency", "missing module import", "incompatible type conversion", "unhandled edge case"],
            "diff_snippets": ["- old_value\n+ new_value", "- if x:\n+ if x is not None:", "- import old\n+ import new", "- def foo()\n+ def foo(x: int)"],
            "documents": ["the technical report", "the codebase documentation", "the project README", "the architecture overview", "the API documentation"],
            "findings": ["improved code quality", "enhanced security", "better performance", "clearer documentation", "more robust error handling"],
            "outputs": ['{"result": "success"}', "name: value\nother: data", "# Result\n\nSuccess", "<result>success</result>"],
            "stat_texts": ["5 fields converted", "10 records processed", "structure preserved", "format validated"],
            "test_codes": ["def test_function():\n    assert True", "def test_edge_case():\n    result = process(None)\n    assert result is not None", "def test_integration():\n    output = integrate(input_data)\n    validate(output)"],
            "coverages": ["85%", "92%", "78%", "95%", "88%"],
            "codes": ["def process(data):\n    return data", "class Handler:\n    def handle(self, x):\n        return x * 2", "async def fetch():\n    return await get()"],
            "modules": ["main", "utils", "handler", "processor", "converter"],
            "functions": ["process_data", "validate_input", "transform_output", "handle_request", "parse_config"],
            "functionalities": ["edge cases", "error handling", "input validation", "output formatting", "integration points"],
            "scenarios": ["empty input", "null values", "large datasets", "concurrent access", "network failure"],
            "subjects": ["the codebase", "the architecture", "the API design", "the database schema", "the deployment pipeline"],
            "tasks": ["generate report", "create summary", "analyze performance", "review security", "document changes"],
        }

    def _variables(self, family: str) -> dict[str, str]:
        pools = self._pools
        n = str(random.randint(1, 5))
        return {
            "file": random.choice(pools["files"]),
            "file_a": random.choice(pools["files"]),
            "file_b": random.choice(pools["files"]),
            "item_a": random.choice(pools["files"]),
            "item_b": random.choice(pools["files"]),
            "directory": random.choice(pools["directories"]),
            "file_type": random.choice(pools["file_types"]),
            "format": random.choice(pools["formats"]),
            "format_a": random.choice(pools["formats"]),
            "format_b": random.choice(pools["formats"]),
            "pattern": random.choice(pools["patterns"]),
            "issue_type": random.choice(pools["issues"]),
            "error": random.choice(pools["errors"]),
            "n": n,
            "code": random.choice(pools["codes"]),
            "module": random.choice(pools["modules"]),
            "function": random.choice(pools["functions"]),
            "functionality": random.choice(pools["functionalities"]),
            "scenario": random.choice(pools["scenarios"]),
            "source": random.choice(pools["files"]),
            "subject": random.choice(pools["subjects"]),
            "task": random.choice(pools["tasks"]),
            "diff_1": random.choice(pools["diffs"]),
            "diff_2": random.choice(pools["diffs"]),
            "diff_3": random.choice(pools["diffs"]),
            "aspect": random.choice(pools["aspects"]),
            "aspect_1": random.choice(pools["aspects"]),
            "aspect_2": random.choice(pools["aspects"]),
            "aspect_3": random.choice(pools["aspects"]),
            "feature_a": random.choice(pools["features"]),
            "feature_b": random.choice(pools["features"]),
            "issue_1": random.choice(pools["issue_texts"]),
            "issue_2": random.choice(pools["issue_texts"]),
            "line_1": str(random.randint(10, 100)),
            "line_2": str(random.randint(10, 100)),
            "rec_1": random.choice(pools["rec_texts"]),
            "rec_2": random.choice(pools["rec_texts"]),
            "severity": random.choice(pools["severities"]),
            "good_1": random.choice(pools["good_texts"]),
            "file_1": random.choice(pools["file_names"]),
            "file_2": random.choice(pools["file_names"]),
            "file_3": random.choice(pools["file_names"]),
            "n_1": str(random.randint(1, 10)),
            "n_2": str(random.randint(1, 10)),
            "n_3": str(random.randint(1, 10)),
            "total": str(random.randint(5, 50)),
            "title": random.choice(pools["titles"]),
            "content": random.choice(pools["contents"]),
            "summary": random.choice(pools["summaries"]),
            "point_1": random.choice(pools["points"]),
            "point_2": random.choice(pools["points"]),
            "point_3": random.choice(pools["points"]),
            "topic": random.choice(pools["aspects"]),
            "fix": random.choice(pools["fixes"]),
            "diff": random.choice(pools["diff_snippets"]),
            "cause": random.choice(pools["causes"]),
            "document": random.choice(pools["documents"]),
            "findings": random.choice(pools["findings"]),
            "input": random.choice(pools["files"]),
            "output": random.choice(pools["outputs"]),
            "stats": random.choice(pools["stat_texts"]),
            "tests": random.choice(pools["test_codes"]),
            "test_code": random.choice(pools["test_codes"]),
            "coverage": random.choice(pools["coverages"]),
            "result": "task completed successfully",
        }

    def generate_samples(self, n_samples: int = 1000, seed: int = 42) -> list[TrainingSample]:
        random.seed(seed)
        families = list(self._task_templates)
        samples: list[TrainingSample] = []
        for _ in range(n_samples):
            family = random.choice(families)
            variables = self._variables(family)
            prompt = random.choice(self._task_templates[family]).format(**variables)
            response = random.choice(self._response_templates[family]).format(**variables)
            verification_passed = random.random() > 0.3
            confidence = 0.58 + random.random() * 0.37
            weight = 1.0 + (0.5 if verification_passed else 0.0) + (0.25 if family in {"comparison", "analysis", "debugging"} else 0.0)
            samples.append(
                TrainingSample(
                    prompt=prompt,
                    response=response,
                    task_family=family,
                    kernel_id=None,
                    verification_passed=verification_passed,
                    confidence=confidence,
                    weight=weight,
                )
            )
        return samples

    def family_counts(self, samples: list[TrainingSample]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample.task_family] = counts.get(sample.task_family, 0) + 1
        return counts


class KaggleTrainer:
    def __init__(
        self,
        base_model: str = "KernelWeave/Synthetic",
        output_dir: str = "./kernel-native-model",
        config: TrainingConfig | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or TrainingConfig(base_model=base_model, output_dir=output_dir)
        self.base_model = self.config.base_model
        from .hardware import detect_hardware, apply_hardware_profile, _resolve_base_model

        self.hardware = detect_hardware()
        self.base_model = _resolve_base_model(self.base_model, self.hardware)
        self.config.base_model = self.base_model
        apply_hardware_profile(self.config, self.hardware)
        self._generator = TraceGenerator()
        self._runtime_model = CalibrationModel()
        self._compile_model = CalibrationModel()
        self._train_data: list[TrainingSample] = []
        self._eval_data: list[TrainingSample] = []
        self._history: list[TrainingSnapshot] = []
        self._trained = False
        self._setup = False
        self._report: TrainingReport | None = None
        _ensure_deps()
        print("KaggleTrainer initialized")
        print(f"  Base model: {self.base_model}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Hardware: {self.hardware.gpu_name} x{self.hardware.gpu_count}")
        if self.hardware.safe_mode:
            print("  Mode: safe pure-Python training")

    def generate_training_data(self, n_samples: int = 5000, seed: int | None = None) -> None:
        seed = self.config.seed if seed is None else seed
        print(f"\nGenerating {n_samples} training samples...")
        samples = self._generator.generate_samples(n_samples=n_samples, seed=seed)
        n_eval = min(max(1, int(n_samples * 0.1)), self.config.n_eval_samples)
        self._eval_data = samples[:n_eval]
        self._train_data = samples[n_eval:]
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(data_dir / "train.jsonl", self._train_data)
        self._write_jsonl(data_dir / "eval.jsonl", self._eval_data)
        summary = {
            "train_samples": len(self._train_data),
            "eval_samples": len(self._eval_data),
            "family_counts": self._generator.family_counts(samples),
            "base_model": self.base_model,
        }
        (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"✓ Generated {len(self._train_data)} train samples")
        print(f"✓ Generated {len(self._eval_data)} eval samples")
        print(f"✓ Saved to {data_dir}")

    def setup_model(self) -> None:
        self._runtime_model = CalibrationModel()
        self._compile_model = CalibrationModel()
        self._setup = True
        print(f"\nPreparing pure-Python training state for: {self.base_model}")
        print("✓ Calibration learners ready")

    def train(self, epochs: int = 3, batch_size: int = 4) -> None:
        if not self._train_data:
            self.generate_training_data(n_samples=self.config.n_train_samples, seed=self.config.seed)
        if not self._setup:
            self.setup_model()
        batch_size = max(1, min(batch_size, self.config.batch_size))
        train_examples = [CalibrationExample(features=self._runtime_features(sample), label=1.0 if sample.verification_passed else 0.0, note=sample.task_family) for sample in self._train_data]
        eval_examples = [CalibrationExample(features=self._runtime_features(sample), label=1.0 if sample.verification_passed else 0.0, note=sample.task_family) for sample in self._eval_data]
        compile_examples = [CalibrationExample(features=self._compile_features(sample), label=1.0 if sample.weight >= 1.5 else 0.0, note=sample.task_family) for sample in self._train_data]
        self._history = []
        rng = random.Random(self.config.seed)
        for epoch in range(1, max(1, epochs) + 1):
            shuffled = train_examples[:]
            rng.shuffle(shuffled)
            sample_count = max(1, len(shuffled))
            bootstrapped = [shuffled[rng.randrange(sample_count)] for _ in range(sample_count)]
            self._runtime_model.fit(bootstrapped, epochs=60, learning_rate=self.config.learning_rate, l2=0.003)
            self._compile_model.fit(compile_examples, epochs=60, learning_rate=self.config.learning_rate * 0.75, l2=0.003)
            metrics = self._evaluate(self._runtime_model, eval_examples)
            compile_metrics = self._evaluate(self._compile_model, compile_examples[: max(1, len(compile_examples) // 5)])
            snapshot = TrainingSnapshot(
                epoch=epoch,
                train_loss=self._loss(self._runtime_model, bootstrapped),
                eval_loss=metrics["loss"],
                eval_accuracy=metrics["accuracy"],
                avg_confidence=metrics["avg_confidence"],
                kernel_reuse_rate=compile_metrics["avg_confidence"],
                prompt_length_mean=metrics["prompt_length_mean"],
                response_length_mean=metrics["response_length_mean"],
                checkpoint_health="healthy" if metrics["accuracy"] >= 0.65 else "needs-attention",
            )
            self._history.append(snapshot)
            print(
                f"Epoch {epoch}/{epochs} | loss={snapshot.train_loss:.3f} | eval_loss={snapshot.eval_loss:.3f} | "
                f"eval_acc={snapshot.eval_accuracy:.3f} | confidence={snapshot.avg_confidence:.3f}"
            )
        self._trained = True
        self._report = self._build_report()
        print("✓ Training complete")

    def save_model(self) -> None:
        if not self._trained:
            raise RuntimeError("No trained model to save. Run train() first.")
        save_path = self.output_dir / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)
        self._runtime_model.save(save_path / "runtime_calibration.json")
        self._compile_model.save(save_path / "compile_calibration.json")
        report = self._build_report()
        (save_path / "model.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        (save_path / "README.md").write_text(self._model_readme(report))
        print(f"✓ Model saved to {save_path}")
        print(f"\nTo inspect: {save_path / 'model.json'}")

    def push_to_hub(self, repo_name: str) -> Path:
        if not self._trained:
            raise RuntimeError("Train the bundle before exporting it.")
        export = self.output_dir / "final_model" / "hub_export.json"
        payload = {
            "repo_name": repo_name,
            "bundle_path": str((self.output_dir / "final_model").resolve()),
            "notes": ["This is a pure-Python KernelWeave bundle, not HF weights."],
        }
        export.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Prepared export metadata at {export}")
        return export

    def describe(self) -> dict[str, Any]:
        report = self._build_report() if self._report is None else self._report
        return report.to_dict()

    def _write_jsonl(self, path: Path, samples: list[TrainingSample]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")

    def _runtime_features(self, sample: TrainingSample) -> dict[str, float]:
        prompt_tokens = _tokenize(sample.prompt)
        response_tokens = _tokenize(sample.response)
        prompt_set = set(prompt_tokens)
        response_set = set(response_tokens)
        overlap = len(prompt_set & response_set) / max(1, len(prompt_set | response_set))
        prompt_len = min(len(prompt_tokens) / 64.0, 1.0)
        response_len = min(len(response_tokens) / 96.0, 1.0)
        has_code = 1.0 if any(marker in sample.response for marker in ("```", "def ", "class ", "{", "}", "import ")) else 0.0
        has_numbers = 1.0 if any(ch.isdigit() for ch in sample.prompt + sample.response) else 0.0
        family_flags = {
            "comparison": 1.0 if sample.task_family == "comparison" else 0.0,
            "analysis": 1.0 if sample.task_family == "analysis" else 0.0,
            "search": 1.0 if sample.task_family == "search" else 0.0,
            "generation": 1.0 if sample.task_family == "generation" else 0.0,
            "debugging": 1.0 if sample.task_family == "debugging" else 0.0,
            "summarization": 1.0 if sample.task_family == "summarization" else 0.0,
            "transformation": 1.0 if sample.task_family == "transformation" else 0.0,
            "testing": 1.0 if sample.task_family == "testing" else 0.0,
        }
        return {
            "prompt_length": prompt_len,
            "response_length": response_len,
            "overlap": overlap,
            "has_code": has_code,
            "has_numbers": has_numbers,
            "confidence": _clamp01(sample.confidence),
            "weight": _clamp01(sample.weight / 3.0),
            "verification_hint": 1.0 if sample.verification_passed else 0.0,
            **family_flags,
        }

    def _compile_features(self, sample: TrainingSample) -> dict[str, float]:
        features = self._runtime_features(sample)
        features["prompt_length"] = _clamp01(features["prompt_length"] * 0.75 + 0.1)
        features["response_length"] = _clamp01(features["response_length"] * 0.9 + 0.05)
        features["weight"] = _clamp01(sample.weight / 2.5)
        return features

    def _loss(self, model: CalibrationModel, examples: list[CalibrationExample]) -> float:
        if not examples:
            return 0.0
        eps = 1e-8
        total = 0.0
        for example in examples:
            prediction = _clamp01(model.predict(example.features))
            target = _clamp01(example.label)
            total += -(target * math.log(prediction + eps) + (1.0 - target) * math.log(1.0 - prediction + eps))
        return total / len(examples)

    def _evaluate(self, model: CalibrationModel, examples: list[CalibrationExample]) -> dict[str, float]:
        if not examples:
            return {"loss": 0.0, "accuracy": 0.0, "avg_confidence": 0.0, "prompt_length_mean": 0.0, "response_length_mean": 0.0}
        predictions = [_clamp01(model.predict(example.features)) for example in examples]
        targets = [_clamp01(example.label) for example in examples]
        loss = self._loss(model, examples)
        accuracy = sum((pred >= 0.5) == (target >= 0.5) for pred, target in zip(predictions, targets)) / len(examples)
        avg_confidence = _safe_mean(predictions)
        prompt_length_mean = _safe_mean([example.features.get("prompt_length", 0.0) for example in examples])
        response_length_mean = _safe_mean([example.features.get("response_length", 0.0) for example in examples])
        return {
            "loss": loss,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "prompt_length_mean": prompt_length_mean,
            "response_length_mean": response_length_mean,
        }

    def _build_report(self) -> TrainingReport:
        runtime = self._runtime_model.to_dict()
        compile_model = self._compile_model.to_dict()
        return TrainingReport(
            model_name=self.base_model,
            output_dir=str(self.output_dir),
            hardware=self.hardware.to_dict(),
            config=self.config.to_dict(),
            train_samples=len(self._train_data),
            eval_samples=len(self._eval_data),
            family_counts=self._generator.family_counts(self._train_data + self._eval_data),
            history=[snapshot.to_dict() for snapshot in self._history],
            runtime_calibration=runtime,
            compile_calibration=compile_model,
            notes=[
                "This bundle is pure Python and does not depend on transformers, peft, trl, or bitsandbytes.",
                "It trains calibration models over synthetic KernelWeave traces and exports JSON artifacts.",
            ],
        )

    def _model_readme(self, report: TrainingReport) -> str:
        return "\n".join(
            [
                "# KernelWeave synthetic training bundle",
                "",
                "This output contains pure-Python calibration artifacts, not neural network weights.",
                "",
                f"- Base model tag: `{report.model_name}`",
                f"- Train samples: `{report.train_samples}`",
                f"- Eval samples: `{report.eval_samples}`",
                f"- Hardware: `{report.hardware.get('gpu_name', 'CPU')}`",
                "",
                "Files:",
                "- `model.json` — bundle metadata and metrics",
                "- `runtime_calibration.json` — runtime confidence model",
                "- `compile_calibration.json` — compile confidence model",
            ]
        )


def train_kernel_native(
    base_model: str = "KernelWeave/Synthetic",
    output_dir: str = "./kernel-native-model",
    n_samples: int = 5000,
    epochs: int = 3,
    batch_size: int | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> KaggleTrainer:
    from .hardware import detect_hardware, apply_hardware_profile, _resolve_base_model

    profile = detect_hardware()
    resolved_base_model = _resolve_base_model(base_model, profile)
    config = TrainingConfig(
        base_model=resolved_base_model,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size or profile.batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation,
        max_input_length=max(64, profile.max_seq_length // 2),
        max_output_length=max(64, profile.max_seq_length // 2),
        seed=seed,
        **kwargs,
    )
    apply_hardware_profile(config, profile)
    trainer = KaggleTrainer(base_model=resolved_base_model, output_dir=output_dir, config=config)
    trainer.generate_training_data(n_samples=n_samples, seed=seed)
    trainer.setup_model()
    trainer.train(epochs=epochs, batch_size=batch_size or profile.batch_size)
    trainer.save_model()
    return trainer


def auto_train(
    base_model: str = "KernelWeave/Synthetic",
    output_dir: str = "./kernel-native-model",
    n_samples: int = 5000,
    epochs: int = 3,
    **kwargs: Any,
) -> KaggleTrainer:
    return train_kernel_native(base_model=base_model, output_dir=output_dir, n_samples=n_samples, epochs=epochs, **kwargs)
