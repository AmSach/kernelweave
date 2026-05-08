from .kernel import Kernel, KernelStatus, KernelStore, TraceEvent, load_sample_store
from .kernels import install_kernel_library, ALL_KERNELS, kernel_summary
from .compiler import CompilationStats, compile_trace_to_kernel, score_kernel
from .runtime import KernelRuntime, plan_for_prompt
from .cli import install_samples
from .metrics import clamp, cosine_similarity, coverage, jaccard_similarity, normalize_text, sigmoid
from .llm import (
    AnthropicBackend,
    AgentPlan,
    AgentPlanner,
    AgentStep,
    AgentTrace,
    ForwardTrace,
    InferenceConfig,
    KernelWeaveLLM,
    LLMConfig,
    ModelBackend,
    ModelCatalog,
    ModelPreset,
    ModelResponse,
    MockBackend,
    OpenAICompatibleBackend,
    ParameterSummary,
    SimpleTokenizer,
    SkillBankManifest,
    SkillKernel,
    SkillKernelBank,
    SkillRoute,
    TokenizerConfig,
    TokenizerReport,
    TrainingConfig,
    TrainingPlan,
    TrainingSnapshot,
    TransformerConfig,
    backend_from_preset,
    run_preset,
)

# Constrained generation
from .constrained import (
    ConstrainedDecoder,
    ConstrainedResponse,
    ConstrainedGrammar,
    ConstrainedTokenSampler,
    postconditions_to_grammar,
    validate_against_grammar,
)

# Trace capture
from .trace import (
    ReasoningStep,
    ToolCall,
    EvidenceCapture,
    VerificationCheck,
    ExecutionTrace,
    TraceCapture,
)

# Composition
from .compose import (
    compose_sequence,
    compose_parallel,
    compose_conditional,
    compose_loop,
    detect_conflicts,
    CompositionBuilder,
)

# REVOLUTIONARY: Training
from .training import (
    ExecutionTrace as TrainingTrace,
    TraceCollector,
    TraceTrainer,
    TrainingConfig as TrainingPipelineConfig,
)

# REVOLUTIONARY: Verifier
from .verifier import (
    VerifierHierarchy,
    VerificationResult,
    HeuristicVerifier,
    ToolExecutionVerifier,
    LLMJudgeVerifier,
    verify_output,
)

# REVOLUTIONARY: Memory
from .memory import (
    KernelMemory,
    MemoryExecutionResult,
)

# REVOLUTIONARY: Promotion
from .promotion import (
    AutoPromoter,
    PromotionConfig,
    PromotedKernel,
)

# REVOLUTIONARY: Model
from .model import (
    KernelNativeModel,
    KernelNativeConfig,
    ExecutionResult,
    create_model,
)

__all__ = [
    "Kernel",
    "KernelStatus",
    "KernelStore",
    "TraceEvent",
    "load_sample_store",
    "install_samples",
    "CompilationStats",
    "compile_trace_to_kernel",
    "score_kernel",
    "KernelRuntime",
    "plan_for_prompt",
    "clamp",
    "cosine_similarity",
    "coverage",
    "jaccard_similarity",
    "normalize_text",
    "sigmoid",
    "ConstrainedDecoder",
    "ConstrainedResponse",
    "ConstrainedGrammar",
    "ConstrainedTokenSampler",
    "postconditions_to_grammar",
    "validate_against_grammar",
    "ReasoningStep",
    "ToolCall",
    "EvidenceCapture",
    "VerificationCheck",
    "ExecutionTrace",
    "TraceCapture",
    "compose_sequence",
    "compose_parallel",
    "compose_conditional",
    "compose_loop",
    "detect_conflicts",
    "CompositionBuilder",
    "InferenceConfig",
    "LLMConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "TransformerConfig",
    "ForwardTrace",
    "KernelWeaveLLM",
    "ParameterSummary",
    "SimpleTokenizer",
    "TokenizerReport",
    "TrainingPlan",
    "TrainingSnapshot",
    "SkillKernel",
    "SkillKernelBank",
    "SkillRoute",
    "SkillBankManifest",
    "AgentPlanner",
    "AgentPlan",
    "AgentStep",
    "AgentTrace",
    "ModelBackend",
    "ModelCatalog",
    "ModelPreset",
    "ModelResponse",
    "MockBackend",
    "OpenAICompatibleBackend",
    "AnthropicBackend",
    "backend_from_preset",
    "run_preset",
    
    # REVOLUTIONARY
    "TrainingTrace",
    "TraceCollector",
    "TraceTrainer",
    "TrainingPipelineConfig",
    "VerifierHierarchy",
    "VerificationResult",
    "HeuristicVerifier",
    "ToolExecutionVerifier",
    "LLMJudgeVerifier",
    "verify_output",
    "KernelMemory",
    "MemoryExecutionResult",
    "AutoPromoter",
    "PromotionConfig",
    "PromotedKernel",
    "KernelNativeModel",
    "KernelNativeConfig",
    "ExecutionResult",
    "create_model",
]