from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from .config import LLMConfig


@dataclass
class DatasetSource:
    name: str
    path: str
    kind: str
    share: float
    license: str
    notes: list[str] = field(default_factory=list)
    min_examples: int = 0
    max_examples: int = 0
    risk: str = "low"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset source name must not be empty")
        if not self.path.strip():
            raise ValueError("dataset source path must not be empty")
        if self.share < 0.0:
            raise ValueError("dataset source share must be non-negative")
        if self.min_examples < 0 or self.max_examples < 0:
            raise ValueError("example counts must be non-negative")
        if self.max_examples and self.max_examples < self.min_examples:
            raise ValueError("max_examples must be >= min_examples")


@dataclass
class CheckpointSpec:
    step: int
    label: str
    save_optimizer_state: bool = True
    save_skill_bank: bool = True
    save_eval_snapshot: bool = True
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.step < 0:
            raise ValueError("checkpoint step must be non-negative")
        if not self.label.strip():
            raise ValueError("checkpoint label must not be empty")


@dataclass
class EvaluationSpec:
    name: str
    kind: str
    metric: str
    target: float
    split: str
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("evaluation name must not be empty")
        if not self.kind.strip():
            raise ValueError("evaluation kind must not be empty")
        if not self.metric.strip():
            raise ValueError("evaluation metric must not be empty")
        if not self.split.strip():
            raise ValueError("evaluation split must not be empty")


@dataclass
class TrainingManifest:
    project_name: str
    model_name: str
    target_params_billion: float
    target_context_tokens: int
    dataset_sources: list[DatasetSource]
    checkpoints: list[CheckpointSpec]
    evaluations: list[EvaluationSpec]
    stages: list[str]
    notes: list[str] = field(default_factory=list)
    built_in_skill_bank: bool = True
    built_in_agent_planning: bool = True
    built_in_curiosity: bool = True
    built_in_feedback_loop: bool = True
    version: str = "1.0"
    max_training_days: int = 0
    max_tokens_seen: int = 0

    def validate(self) -> None:
        if not self.project_name.strip():
            raise ValueError("project_name must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.target_params_billion <= 0:
            raise ValueError("target_params_billion must be positive")
        if self.target_context_tokens <= 0:
            raise ValueError("target_context_tokens must be positive")
        if not self.dataset_sources:
            raise ValueError("dataset_sources must not be empty")
        if not self.checkpoints:
            raise ValueError("checkpoints must not be empty")
        if not self.evaluations:
            raise ValueError("evaluations must not be empty")
        if not self.stages:
            raise ValueError("stages must not be empty")
        for source in self.dataset_sources:
            source.validate()
        for checkpoint in self.checkpoints:
            checkpoint.validate()
        for evaluation in self.evaluations:
            evaluation.validate()
        if self.max_training_days < 0:
            raise ValueError("max_training_days must be non-negative")
        if self.max_tokens_seen < 0:
            raise ValueError("max_tokens_seen must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "TrainingManifest":
        data = json.loads(path.read_text())
        return cls(
            project_name=data["project_name"],
            model_name=data["model_name"],
            target_params_billion=float(data["target_params_billion"]),
            target_context_tokens=int(data["target_context_tokens"]),
            dataset_sources=[DatasetSource(**item) for item in data.get("dataset_sources", [])],
            checkpoints=[CheckpointSpec(**item) for item in data.get("checkpoints", [])],
            evaluations=[EvaluationSpec(**item) for item in data.get("evaluations", [])],
            stages=list(data.get("stages", [])),
            notes=list(data.get("notes", [])),
            built_in_skill_bank=bool(data.get("built_in_skill_bank", True)),
            built_in_agent_planning=bool(data.get("built_in_agent_planning", True)),
            built_in_curiosity=bool(data.get("built_in_curiosity", True)),
            built_in_feedback_loop=bool(data.get("built_in_feedback_loop", True)),
            version=str(data.get("version", "1.0")),
            max_training_days=int(data.get("max_training_days", 0)),
            max_tokens_seen=int(data.get("max_tokens_seen", 0)),
        )

    def summary(self) -> dict[str, Any]:
        self.validate()
        total_share = sum(source.share for source in self.dataset_sources)
        return {
            "project_name": self.project_name,
            "model_name": self.model_name,
            "target_params_billion": self.target_params_billion,
            "target_context_tokens": self.target_context_tokens,
            "dataset_sources": len(self.dataset_sources),
            "checkpoints": len(self.checkpoints),
            "evaluations": len(self.evaluations),
            "stages": list(self.stages),
            "total_dataset_share": round(total_share, 4),
            "built_in_skill_bank": self.built_in_skill_bank,
            "built_in_agent_planning": self.built_in_agent_planning,
            "built_in_curiosity": self.built_in_curiosity,
            "built_in_feedback_loop": self.built_in_feedback_loop,
        }


def config_model_params_billion(config: LLMConfig) -> float:
    t = config.transformer
    emb = t.d_model * config.tokenizer.vocab_size
    attn = 4 * t.n_layers * t.d_model * t.d_model // max(1, t.n_heads)
    ffn = 2 * t.n_layers * t.d_model * t.d_ff
    out = t.d_model * config.tokenizer.vocab_size
    moe_active = 0
    if t.use_moe:
        expert_ffn = int(t.d_model * t.d_ff * t.expert_ffn_multiplier)
        moe_active = t.n_layers * t.top_k_experts * expert_ffn
    total = emb + attn + ffn + out + moe_active
    if t.tie_embeddings:
        total -= out
    return total / 1_000_000_000


def default_frontier_manifest(config: LLMConfig | None = None) -> TrainingManifest:
    config = config or LLMConfig.reasoner_frontier_spec()
    config.validate()
    return TrainingManifest(
        project_name="KernelWeave",
        model_name=config.name,
        target_params_billion=round(config_model_params_billion(config), 3),
        target_context_tokens=config.inference.max_context_tokens,
        dataset_sources=[
            DatasetSource(name="general_text", path="datasets/general_text", kind="web+book", share=0.22, license="mixed", min_examples=12_000_000, max_examples=30_000_000, notes=["Cleaned and deduplicated.", "General language stability."], risk="medium"),
            DatasetSource(name="code", path="datasets/code", kind="code", share=0.14, license="open-source", min_examples=6_000_000, max_examples=18_000_000, notes=["Prefer high-quality repos and docs.", "Must exclude generated junk."], risk="medium"),
            DatasetSource(name="math", path="datasets/math", kind="math", share=0.16, license="mixed", min_examples=3_000_000, max_examples=10_000_000, notes=["Proofs, derivations, and worked examples."], risk="low"),
            DatasetSource(name="reasoning", path="datasets/reasoning", kind="reasoning", share=0.22, license="mixed", min_examples=5_000_000, max_examples=15_000_000, notes=["Reasoning traces should be distilled, not dumped raw."], risk="high"),
            DatasetSource(name="dialogue", path="datasets/dialogue", kind="dialogue", share=0.10, license="mixed", min_examples=4_000_000, max_examples=12_000_000, notes=["Preference-aligned assistant interactions."], risk="medium"),
            DatasetSource(name="tool_traces", path="datasets/tool_traces", kind="agent", share=0.12, license="internal+open", min_examples=3_000_000, max_examples=10_000_000, notes=["Stepwise tool use and recovery behaviour."], risk="high"),
            DatasetSource(name="skills", path="datasets/skill_kernels", kind="internal", share=0.04, license="internal", min_examples=750_000, max_examples=3_000_000, notes=["Promoted kernels and regression passing traces."], risk="low"),
        ],
        checkpoints=[
            CheckpointSpec(step=0, label="bootstrap", notes=["Tokenizer, config, and routing smoke test."], save_optimizer_state=False),
            CheckpointSpec(step=2_000, label="foundation", notes=["Early loss and contamination checks."]),
            CheckpointSpec(step=20_000, label="reasoning", notes=["Reasoning trace quality and self-consistency checks."]),
            CheckpointSpec(step=75_000, label="tooling", notes=["Tool-use success and recovery under errors."]),
            CheckpointSpec(step=150_000, label="long-context", notes=["Extended context validation and retrieval stability."]),
            CheckpointSpec(step=300_000, label="skill-bank", notes=["Promoted skills and kernel reuse quality."]),
            CheckpointSpec(step=500_000, label="final", notes=["Full evaluation and release candidate."]),
        ],
        evaluations=[
            EvaluationSpec(name="reasoning-holdout", kind="reasoning", metric="accuracy", target=0.74, split="held_out_reasoning", notes=["Hard prompts unseen in training."]),
            EvaluationSpec(name="tool-use", kind="agent", metric="success_rate", target=0.72, split="held_out_agent", notes=["Tool calls, retries, and recovery."]),
            EvaluationSpec(name="long-context", kind="memory", metric="retrieval_accuracy", target=0.74, split="long_context", notes=["Needle-in-haystack plus multi-hop retrieval."]),
            EvaluationSpec(name="code", kind="coding", metric="pass_rate", target=0.62, split="held_out_code", notes=["Realistic code tasks and debugging."]),
            EvaluationSpec(name="safety", kind="safety", metric="reject_rate", target=0.96, split="adversarial", notes=["Refuse unsafe or policy-violating requests."]),
        ],
        stages=[
            "data_quality_and_dedup",
            "foundation_language_pretraining",
            "reasoning_trace_distillation",
            "tool_use_and_agentic_recovery",
            "long_context_optimization",
            "self_play_and_curiosity",
            "skill_bank_promotion",
            "evaluation_hardening",
        ],
        notes=[
            "KernelWeave keeps the skill bank inside the architecture, not as an external skills file.",
            "The model should practice stepwise decomposition on hard tasks and compress successes into kernels.",
            "Long context is only useful if retrieval and compression remain cheap enough to use all the time.",
            "Reasoning traces should be distilled into structured objectives, not mindlessly copied.",
        ],
        built_in_skill_bank=True,
        built_in_agent_planning=True,
        built_in_curiosity=True,
        built_in_feedback_loop=True,
        version="1.2",
        max_training_days=180,
        max_tokens_seen=0,
    )
