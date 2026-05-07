"""
LLM architecture specification and kernel routing layer.

CRITICAL DISCLAIMER: This module does NOT implement a neural network.
===============================================================
There are:
  - NO model weights
  - NO PyTorch, JAX, or any tensor framework
  - NO actual inference forward pass through learned parameters
  - NO training loop with backpropagation

What this module ACTUALLY does:
  - Define architecture specifications (TransformerConfig, etc.)
  - Estimate parameter counts from those specifications
  - Route prompts to skill kernels stored as JSON objects
  - Simulate what a trained model "would" do based on kernel matching

The `KernelWeaveLLM` class is a routing and simulation layer, not a trained model.
It estimates how many parameters an architecture would have if it existed,
but it cannot run inference because there are no weights to compute with.

The "compact frontier preset" and "reasoning frontier preset" are architecture
specifications, not trained checkpoints. They describe what a model could look
like, not what currently exists.

For the actual working parts of KernelWeave, see:
  - kernelweave.kernel: kernel store and compilation
  - kernelweave.runtime: execution engine for kernel plans
  - kernelweave.calibration: confidence calibration from examples
  - kernelweave.skills: skill kernel bank routing

These are real, working Python that route between stored JSON skill objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .config import LLMConfig
from .tokenizer import SimpleTokenizer
from .agent import AgentPlanner
from .providers import ModelBackend, ModelResponse
from ..calibration import predict_runtime_confidence
from ..kernel import KernelStore
from ..runtime import ExecutionEngine, KernelRuntime
from ..metrics import clamp, conflict_terms, jaccard_similarity, normalize_text, semantic_similarity, sigmoid
from .skills import SkillKernelBank


@dataclass
class ParameterSummary:
    total_parameters: int
    attention_parameters: int
    feedforward_parameters: int
    embedding_parameters: int
    output_parameters: int
    moe_active_parameters: int


@dataclass
class ForwardTrace:
    prompt: str
    prompt_tokens: list[int]
    compiled_prompt: dict[str, Any]
    chosen_route: str
    kernel_plan: dict[str, Any] | None
    generated_preview: str
    curiosity_questions: list[str]
    agent_plan: dict[str, Any]
    execution: dict[str, Any]


class KernelWeaveLLM:
    """
    Kernel routing and simulation layer — not a trained neural network.

    This class can also wrap a real model backend. In that mode it routes
    a prompt through KernelWeave first, then calls the backend with a
    kernel-aware system prompt.
    """
    
    def __init__(
        self,
        config: LLMConfig,
        tokenizer: SimpleTokenizer | None = None,
        kernel_store: KernelStore | None = None,
        backend: ModelBackend | None = None,
    ):
        self.config = config
        self.config.validate()
        self.tokenizer = tokenizer or SimpleTokenizer(self.config.tokenizer)
        self.kernel_store = kernel_store
        self.backend = backend
        self.skill_bank = SkillKernelBank()
        if kernel_store is not None:
            self.skill_bank.import_from_store(kernel_store)
        self.agent_planner = AgentPlanner(self.skill_bank)
        self.summary = self._estimate_parameters()
        self.last_state: dict[str, Any] = {"tokens": [], "trace": []}
        self.executor = ExecutionEngine(kernel_store, backend=backend) if kernel_store is not None else None

    def set_backend(self, backend: ModelBackend | None) -> None:
        self.backend = backend

    def _estimate_parameters(self) -> ParameterSummary:
        t = self.config.transformer
        emb = t.d_model * self.config.tokenizer.vocab_size
        attn = 4 * t.n_layers * t.d_model * t.d_model // max(1, t.n_heads)
        ffn = 2 * t.n_layers * t.d_model * t.d_ff
        out = t.d_model * self.config.tokenizer.vocab_size
        moe_active = 0
        if t.use_moe:
            expert_ffn = int(t.d_model * t.d_ff * t.expert_ffn_multiplier)
            moe_active = t.n_layers * t.top_k_experts * expert_ffn
        total = emb + attn + ffn + out + moe_active
        if t.tie_embeddings:
            total -= out
        return ParameterSummary(total, attn, ffn, emb, out, moe_active)

    def parameter_megabytes(self, bytes_per_param: float = 2.0) -> float:
        return self.summary.total_parameters * bytes_per_param / (1024 * 1024)

    def parameter_billion(self) -> float:
        return self.summary.total_parameters / 1_000_000_000

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "architecture_version": self.config.architecture_version,
            "model_family": self.config.model_family,
            "transformer": self.config.transformer.__dict__,
            "tokenizer": self.config.tokenizer.__dict__,
            "training": self.config.training.__dict__,
            "inference": self.config.inference.__dict__,
            "parameter_summary": self.summary.__dict__,
            "parameter_megabytes_bf16": round(self.parameter_megabytes(2.0), 2),
            "parameter_megabytes_int8": round(self.parameter_megabytes(1.0), 2),
            "parameter_billion": round(self.parameter_billion(), 3),
            "skill_bank": self.skill_bank.summary(),
            "agent_planner_max_steps": self.agent_planner.max_steps,
        }

    def route_prompt(self, prompt: str) -> dict[str, Any]:
        words = prompt.split()
        intent = "analysis" if len(words) > 14 else "response"
        pressure = clamp(len(words) / max(1, self.config.inference.max_new_tokens))
        base_confidence = sigmoid(2.2 - pressure * 4.0)
        kernel_plan = None
        route = "generate"
        curiosity_questions: list[str] = []
        agent_plan: dict[str, Any] = self.agent_planner.plan(prompt).to_dict()

        if self.config.inference.curiosity_enabled:
            curiosity_questions = agent_plan.get("curiosity_questions", [])[: self.config.inference.curiosity_budget]

        if self.config.inference.kernel_reuse_enabled:
            if self.kernel_store is not None:
                runtime = KernelRuntime(self.kernel_store)
                decision = runtime.evaluate_prompt(prompt)
                base_confidence = max(base_confidence, decision.confidence)
                route = decision.mode
                if decision.mode == "kernel" and decision.kernel_id:
                    kernel_plan = self.kernel_store.get_kernel(decision.kernel_id).to_dict()
            else:
                routed = self.skill_bank.route(prompt)
                base_confidence = max(base_confidence, routed.confidence)
                route = routed.mode
                if routed.mode == "skill" and routed.kernel_id:
                    kernel_plan = self.skill_bank.kernels[routed.kernel_id].to_dict()
                if routed.curiosity_questions:
                    curiosity_questions = routed.curiosity_questions

        if route == "generate" and agent_plan.get("strategy") == "stepwise":
            route = "agent"

        execution = {}
        if self.executor is not None:
            plan = {"mode": route, "kernel_id": kernel_plan.get("kernel_id") if kernel_plan else None, "reason": "route selection", "confidence": base_confidence, "evidence_debt": 1.0, "prompt": prompt}
            execution = self.executor.execute_plan(plan, prompt)

        return {
            "intent": intent,
            "confidence": base_confidence,
            "context_tokens": min(len(words) * 4, self.config.memory_budget_tokens),
            "routing": route,
            "kernel_plan": kernel_plan,
            "curiosity_questions": curiosity_questions,
            "agent_plan": agent_plan,
            "execution": execution,
        }

    def _kernel_system_prompt(self, routing: dict[str, Any], extra_system_prompt: str = "") -> str:
        parts: list[str] = []
        if extra_system_prompt.strip():
            parts.append(extra_system_prompt.strip())
        parts.append(self.config.inference.system_prompt.strip())
        parts.append("KernelWeave routing layer active. Follow the kernel contract and respect evidence gates.")
        parts.append(f"Routing mode: {routing.get('routing', 'generate')}.")
        if routing.get("kernel_plan"):
            parts.append("Kernel plan JSON:\n" + json.dumps(routing["kernel_plan"], indent=2, sort_keys=True))
        if routing.get("agent_plan"):
            parts.append("Agent plan JSON:\n" + json.dumps(routing["agent_plan"], indent=2, sort_keys=True))
        questions = routing.get("curiosity_questions") or []
        if questions:
            parts.append("Curiosity questions:\n- " + "\n- ".join(questions))
        return "\n\n".join(part for part in parts if part)

    def respond(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        auto_compile: bool = False,
    ) -> dict[str, Any]:
        routing = self.route_prompt(prompt)
        trace = self.forward(prompt)
        if self.backend is None:
            return {
                "mode": "routing-only",
                "routing": routing,
                "trace": trace,
                "response": None,
                "text": "",
            }
        effective_system_prompt = self._kernel_system_prompt(routing, extra_system_prompt=system_prompt)
        response = self.backend.generate(
            prompt,
            system_prompt=effective_system_prompt,
            temperature=temperature if temperature is not None else self.config.inference.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.inference.max_new_tokens,
        )
        
        kernel_candidate = None
        if auto_compile and self.kernel_store is not None and routing.get("routing") in {"generate", "agent"}:
            kernel_candidate = self._extract_kernel_from_response(prompt, response.text, routing)
            if kernel_candidate:
                self.kernel_store.add_kernel(kernel_candidate)
        
        if self.kernel_store is not None:
            success = routing.get("routing") in {"kernel", "skill"} or bool(kernel_candidate)
            self.kernel_store.record_runtime_feedback(
                prompt=prompt,
                kernel_id=kernel_candidate.kernel_id if kernel_candidate else None,
                mode=routing.get("routing", "generate"),
                reason="model response",
                confidence=routing.get("confidence", 0.0),
                evidence_debt=1.0 - routing.get("confidence", 0.0),
                task_family=routing.get("agent_plan", {}).get("steps", [{}])[0].get("title", "") if routing.get("routing") == "agent" else "",
                response_text=response.text,
                observed={"success": success, "auto_compiled": bool(kernel_candidate)}
            )
        
        return {
            "mode": "hybrid",
            "routing": routing,
            "trace": trace,
            "response": response.to_dict(),
            "text": response.text,
            "system_prompt": effective_system_prompt,
            "kernel_candidate": kernel_candidate.to_dict() if kernel_candidate else None,
        }

    def _extract_kernel_from_response(self, prompt: str, response_text: str, routing: dict[str, Any]) -> Any:
        """Extract a kernel candidate from a successful model response."""
        from ..kernel import TraceEvent
        from ..compiler import compile_trace_to_kernel
        from ..metrics import infer_task_family
        import hashlib
        
        agent_plan = routing.get("agent_plan", {})
        steps = agent_plan.get("steps", [])
        if not steps and not response_text.strip():
            return None
        
        # Derive task family from prompt semantics, not step titles
        task_family = infer_task_family(prompt)
        description = prompt.strip()[:200]
        
        # Override with agent plan's objective if available and more specific
        if agent_plan.get("strategy") and len(steps) > 1:
            # Use the agent's overall intent, not individual step titles
            task_family = agent_plan.get("strategy", task_family)
            first_objective = steps[0].get("objective", "") if steps else ""
            if first_objective and len(first_objective) > len(task_family):
                description = first_objective
        
        trace_id = f"auto-{hashlib.sha256(prompt.encode()).hexdigest()[:12]}"
        
        events = [
            TraceEvent(kind="plan", payload={"text": description}),
        ]
        
        for step in steps[:5]:
            events.append(TraceEvent(
                kind="tool" if step.get("tools") else "plan",
                payload={"text": step.get("title", ""), "tool": step.get("tools", [""])[0] if step.get("tools") else ""}
            ))
        
        events.extend([
            TraceEvent(kind="evidence", payload={"text": f"completed task: {task_family}"}),
            TraceEvent(kind="verification", payload={"text": response_text[:200]}),
            TraceEvent(kind="decision", payload={"text": response_text[:500]}),
        ])
        
        kernel = compile_trace_to_kernel(
            trace_id=trace_id,
            task_family=task_family,
            description=description,
            events=events,
            expected_output={"result": response_text[:256]}
        )
        kernel.status.state = "candidate"
        return kernel

    def forward(self, prompt: str) -> dict[str, Any]:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        routing = self.route_prompt(prompt)
        preview = self._generate_preview(prompt, routing)
        trace = ForwardTrace(
            prompt=prompt,
            prompt_tokens=token_ids,
            compiled_prompt=routing,
            chosen_route=routing["routing"],
            kernel_plan=routing.get("kernel_plan"),
            generated_preview=preview,
            curiosity_questions=routing.get("curiosity_questions", []),
            agent_plan=routing.get("agent_plan", {}),
            execution=routing.get("execution", {}),
        )
        self.last_state = {"tokens": token_ids, "trace": trace.__dict__}
        return trace.__dict__

    def _generate_preview(self, prompt: str, routing: dict[str, Any]) -> str:
        words = prompt.strip().split()
        if not words:
            return ""
        if routing["routing"] in {"kernel", "skill"} and routing.get("kernel_plan"):
            steps = routing["kernel_plan"].get("steps", [])
            if steps:
                head = steps[0]
                return f"KernelWeaveLLM would follow kernel step: {head.get('action', 'plan')}"
        if routing["routing"] == "agent" and routing.get("agent_plan"):
            steps = routing["agent_plan"].get("steps", [])
            if steps:
                head = steps[0]
                return f"Agent plan starts with: {head.get('title', 'plan')}"
        if len(words) <= 8:
            return " | ".join(words[:8])
        return " ".join(words[: min(len(words), 24)])

    def similarity_report(self, left: str, right: str) -> dict[str, float]:
        left_n = normalize_text(left)
        right_n = normalize_text(right)
        canonical_conflict = conflict_terms(left) & conflict_terms(right)
        sem = semantic_similarity(left_n, right_n)
        j = jaccard_similarity(left_n, right_n)
        conflict_penalty = 0.10 * len(canonical_conflict)
        return {"jaccard": j, "semantic": clamp(sem - conflict_penalty), "length_ratio": clamp(len(left.split()) / max(1, len(right.split())) / 2.0), "confidence": sigmoid(1.2 * sem + 0.6 * j - conflict_penalty)}

    def save_summary(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.describe(), indent=2, sort_keys=True))
        return path
