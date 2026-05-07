"""
Architecture specification configs for hypothetical model configurations.

IMPORTANT: These are NOT trained models.
======================================
These dataclasses define what a model architecture WOULD look like if trained.
They are used to:
  - Estimate parameter counts
  - Specify hyperparameters for a training run
  - Configure the kernel routing layer

There are NO weights, NO checkpoints, NO trained parameters.
The `compact_frontier_spec()` and `reasoner_frontier_spec()` methods return
architecture specifications, not trained models.

To actually use these specs:
1. Implement the transformer architecture in PyTorch/JAX
2. Train a model following the spec
3. Load the trained weights
4. Connect to the kernel routing layer
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class TokenizerConfig:
    """Specification for a byte-BPE tokenizer. Not a trained tokenizer."""
    kind: str = "byte-bpe"
    vocab_size: int = 32768
    lowercase: bool = False
    byte_fallback: bool = True
    special_tokens: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
    min_pair_frequency: int = 2
    max_merge_rounds: int = 12000

    def validate(self) -> None:
        if self.vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256")
        if self.min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be at least 1")
        if self.max_merge_rounds < 0:
            raise ValueError("max_merge_rounds must be non-negative")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")


@dataclass
class TransformerConfig:
    """Specification for a decoder-only transformer architecture. Not a trained model."""
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 8192
    context_length: int = 196608
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 800000.0
    activation: str = "silu"
    use_gqa: bool = True
    n_kv_heads: int = 4
    use_rmsnorm: bool = True
    use_flash_attention: bool = True
    use_moe: bool = True
    n_experts: int = 8
    top_k_experts: int = 2
    expert_ffn_multiplier: float = 1.0
    router_hidden_dim: int = 384
    tie_embeddings: bool = True
    kernel_routing_enabled: bool = True
    reasoning_router: str = "hybrid"
    curvature_memory_depth: int = 8
    retrieval_backend: str = "hybrid"
    retrieval_index_size: int = 4096
    conflict_aware_routing: bool = True

    def validate(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0 or self.d_ff <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.use_gqa and self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads when use_gqa is enabled")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.use_moe:
            if self.n_experts < 1:
                raise ValueError("n_experts must be positive when use_moe is enabled")
            if self.top_k_experts < 1:
                raise ValueError("top_k_experts must be positive when use_moe is enabled")
            if self.top_k_experts > self.n_experts:
                raise ValueError("top_k_experts cannot exceed n_experts")
            if self.expert_ffn_multiplier <= 0:
                raise ValueError("expert_ffn_multiplier must be positive")
            if self.router_hidden_dim <= 0:
                raise ValueError("router_hidden_dim must be positive")
        if self.reasoning_router not in {"evidence-aware", "kernel-first", "hybrid", "self-reflective"}:
            raise ValueError("unsupported reasoning_router")
        if self.curvature_memory_depth < 1:
            raise ValueError("curvature_memory_depth must be positive")
        if self.retrieval_backend not in {"hybrid", "dense", "sparse", "hybrid-with-feedback"}:
            raise ValueError("unsupported retrieval_backend")
        if self.retrieval_index_size < 1:
            raise ValueError("retrieval_index_size must be positive")


@dataclass
class TrainingConfig:
    """Training hyperparameters for a hypothetical training run. Not a trained checkpoint."""
    max_steps: int = 500000
    batch_size: int = 256
    micro_batch_size: int = 8
    sequence_length: int = 32768
    learning_rate: float = 2.0e-4
    warmup_steps: int = 4000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    precision: str = "bf16"
    seed: int = 42
    log_every: int = 10
    eval_every: int = 1000
    checkpoint_every: int = 2000
    data_mix: dict[str, float] = field(
        default_factory=lambda: {
            "general_text": 0.25,
            "code": 0.15,
            "math": 0.15,
            "reasoning": 0.20,
            "dialogue": 0.10,
            "tool_traces": 0.10,
            "skills": 0.05,
        }
    )
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    gradient_accumulation_steps: int = 2
    activation_checkpointing: bool = True
    packed_sequences: bool = True
    trace_distillation_fraction: float = 0.25
    reasoning_trace_fraction: float = 0.25
    curriculum_boundaries: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072)
    self_play_fraction: float = 0.10
    curiosity_fraction: float = 0.10
    rerank_fraction: float = 0.05
    synthetic_fraction: float = 0.10
    kernel_feedback_fraction: float = 0.10

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if not self.data_mix:
            raise ValueError("data_mix must not be empty")
        total = sum(self.data_mix.values())
        if total <= 0:
            raise ValueError("data_mix weights must sum to something positive")
        if len(self.optimizer_betas) != 2:
            raise ValueError("optimizer_betas must have two values")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        for fraction_name in ("trace_distillation_fraction", "reasoning_trace_fraction", "self_play_fraction", "curiosity_fraction", "rerank_fraction", "synthetic_fraction", "kernel_feedback_fraction"):
            value = getattr(self, fraction_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{fraction_name} must be in [0, 1]")
        if self.trace_distillation_fraction + self.reasoning_trace_fraction + self.self_play_fraction + self.curiosity_fraction + self.rerank_fraction + self.synthetic_fraction + self.kernel_feedback_fraction > 1.0:
            raise ValueError("specialized data fractions must sum to at most 1")
        if not self.curriculum_boundaries:
            raise ValueError("curriculum_boundaries must not be empty")
        if any(boundary <= 0 for boundary in self.curriculum_boundaries):
            raise ValueError("curriculum_boundaries must be positive")
        if tuple(sorted(self.curriculum_boundaries)) != self.curriculum_boundaries:
            self.curriculum_boundaries = tuple(sorted(self.curriculum_boundaries))


@dataclass
class InferenceConfig:
    """Inference parameters for the routing layer. Actual inference requires a trained model."""
    max_new_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.92
    top_k: int = 32
    repetition_penalty: float = 1.08
    cache_backend: str = "paged-kv"
    kernel_reuse_enabled: bool = True
    system_prompt: str = "You are a careful, exact, high-agency reasoning assistant."
    max_context_tokens: int = 196608
    min_kernel_confidence: float = 0.25
    reasoning_passes: int = 4
    verification_passes: int = 3
    self_consistency_samples: int = 5
    long_context_strategy: str = "compress-retrieve-route"
    curiosity_enabled: bool = True
    curiosity_budget: int = 3
    conflict_aware_routing: bool = True
    retrieval_blend: float = 0.58

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if not 0.0 < self.min_kernel_confidence <= 1.0:
            raise ValueError("min_kernel_confidence must be in (0, 1]")
        if self.reasoning_passes < 1:
            raise ValueError("reasoning_passes must be at least 1")
        if self.verification_passes < 0:
            raise ValueError("verification_passes must be non-negative")
        if self.self_consistency_samples < 1:
            raise ValueError("self_consistency_samples must be at least 1")
        if self.long_context_strategy not in {"compress-retrieve-route", "dense-only", "memory-first"}:
            raise ValueError("unsupported long_context_strategy")
        if self.curiosity_budget < 0:
            raise ValueError("curiosity_budget must be non-negative")
        if not isinstance(self.curiosity_enabled, bool):
            raise ValueError("curiosity_enabled must be boolean")
        if not isinstance(self.conflict_aware_routing, bool):
            raise ValueError("conflict_aware_routing must be boolean")
        if not 0.0 <= self.retrieval_blend <= 1.0:
            raise ValueError("retrieval_blend must be in [0, 1]")


@dataclass
class LLMConfig:
    """
    Architecture specification for a hypothetical language model.
    
    This is NOT a trained model. It defines:
    - What the architecture would look like if implemented
    - Hyperparameters for training
    - Parameter count estimates (from architecture spec, not trained weights)
    
    Use `compact_frontier_spec()` or `reasoner_frontier_spec()` to get
    predefined architecture specifications.
    """
    
    name: str = "KernelWeave-Reasoner-Spec"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    memory_budget_tokens: int = 196608
    safety_mode: str = "strict"
    notes: list[str] = field(
        default_factory=lambda: [
            "ARCHITECTURE SPECIFICATION, NOT A TRAINED MODEL.",
            "This config estimates parameter counts for a hypothetical architecture.",
            "To train a model: implement in PyTorch/JAX and follow training/ docs.",
            "Kernel reuse routing is built into the specification.",
        ]
    )
    architecture_version: str = "reasoner-spec-v4"
    model_family: str = "decoder-only"
    built_in_skill_bank: bool = True
    built_in_curiosity_loop: bool = True
    built_in_self_training: bool = True
    built_in_feedback_loop: bool = True

    def validate(self) -> None:
        self.tokenizer.validate()
        self.transformer.validate()
        self.training.validate()
        self.inference.validate()
        if self.memory_budget_tokens <= 0:
            raise ValueError("memory_budget_tokens must be positive")
        if self.safety_mode not in {"strict", "balanced", "open"}:
            raise ValueError("safety_mode must be strict, balanced, or open")
        if self.model_family not in {"decoder-only", "encoder-decoder", "mixture-of-experts"}:
            raise ValueError("unsupported model_family")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def compact_frontier_spec(cls) -> "LLMConfig":
        """
        Compact architecture specification (~1.2B params estimated).
        
        This is NOT a trained model. It's a config that estimates what
        a compact model architecture would look like if trained.
        """
        return cls(
            name="KernelWeave-Compact-Frontier-Spec",
            tokenizer=TokenizerConfig(vocab_size=32768, max_merge_rounds=8000),
            transformer=TransformerConfig(
                d_model=1536,
                n_layers=16,
                n_heads=12,
                d_ff=4096,
                context_length=131072,
                rope_theta=500000.0,
                n_kv_heads=4,
                use_moe=True,
                n_experts=4,
                top_k_experts=1,
                expert_ffn_multiplier=1.0,
                router_hidden_dim=256,
                tie_embeddings=True,
                kernel_routing_enabled=True,
                reasoning_router="hybrid",
                curvature_memory_depth=8,
                retrieval_backend="hybrid",
                retrieval_index_size=2048,
                conflict_aware_routing=True,
            ),
            training=TrainingConfig(
                max_steps=350000,
                batch_size=128,
                micro_batch_size=8,
                sequence_length=16384,
                learning_rate=2.5e-4,
                warmup_steps=2000,
                activation_checkpointing=True,
                packed_sequences=True,
                trace_distillation_fraction=0.25,
                reasoning_trace_fraction=0.25,
                self_play_fraction=0.10,
                curiosity_fraction=0.10,
                rerank_fraction=0.05,
                synthetic_fraction=0.10,
                kernel_feedback_fraction=0.10,
            ),
            inference=InferenceConfig(
                max_new_tokens=1024,
                temperature=0.65,
                top_p=0.92,
                top_k=32,
                repetition_penalty=1.08,
                max_context_tokens=131072,
                min_kernel_confidence=0.30,
                reasoning_passes=3,
                verification_passes=2,
                self_consistency_samples=3,
                curiosity_enabled=True,
                curiosity_budget=3,
                long_context_strategy="compress-retrieve-route",
                conflict_aware_routing=True,
                retrieval_blend=0.58,
            ),
            memory_budget_tokens=131072,
            safety_mode="strict",
            notes=[
                "ARCHITECTURE SPEC, NOT TRAINED MODEL.",
                "Compact spec for architecture planning and parameter estimation.",
                "Built-in skill bank routing is part of the spec, not a trained component.",
                "Long context target exists in the spec; actual implementation needs training.",
            ],
            architecture_version="compact-frontier-spec-v4",
            model_family="decoder-only",
            built_in_skill_bank=True,
            built_in_curiosity_loop=True,
            built_in_self_training=True,
            built_in_feedback_loop=True,
        )

    @classmethod
    def reasoner_frontier_spec(cls) -> "LLMConfig":
        """
        Full-scale architecture specification (~2.4B params estimated).
        
        This is NOT a trained model. It's a config that estimates what
        a reasoning-focused model architecture would look like if trained.
        """
        return cls()

    # Keep backward compatibility aliases
    @classmethod
    def compact_frontier(cls) -> "LLMConfig":
        """Deprecated: Use compact_frontier_spec() instead."""
        return cls.compact_frontier_spec()

    @classmethod
    def reasoner_frontier(cls) -> "LLMConfig":
        """Deprecated: Use reasoner_frontier_spec() instead."""
        return cls.reasoner_frontier_spec()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        tokenizer = TokenizerConfig(**data.get("tokenizer", {}))
        transformer = TransformerConfig(**data.get("transformer", {}))
        training = TrainingConfig(**data.get("training", {}))
        inference = InferenceConfig(**data.get("inference", {}))
        return cls(
            name=data.get("name", "KernelWeave-Reasoner"),
            tokenizer=tokenizer,
            transformer=transformer,
            training=training,
            inference=inference,
            memory_budget_tokens=int(data.get("memory_budget_tokens", 196608)),
            safety_mode=str(data.get("safety_mode", "strict")),
            notes=list(data.get("notes", [])),
            architecture_version=str(data.get("architecture_version", "reasoner-v4")),
            model_family=str(data.get("model_family", "decoder-only")),
            built_in_skill_bank=bool(data.get("built_in_skill_bank", True)),
            built_in_curiosity_loop=bool(data.get("built_in_curiosity_loop", True)),
            built_in_self_training=bool(data.get("built_in_self_training", True)),
            built_in_feedback_loop=bool(data.get("built_in_feedback_loop", True)),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "LLMConfig":
        return cls.from_dict(json.loads(path.read_text()))
