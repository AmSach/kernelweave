from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .skills import SkillKernelBank, SkillRoute
from ..metrics import clamp, conflict_terms, jaccard_similarity, normalize_text, semantic_similarity, sigmoid


@dataclass
class AgentStep:
    index: int
    title: str
    objective: str
    evidence_needed: list[str]
    tools: list[str]
    verification: list[str]
    risk: str = "low"


@dataclass
class AgentPlan:
    prompt: str
    complexity: str
    estimated_steps: int
    confidence: float
    skill_route: dict[str, Any]
    curiosity_questions: list[str]
    steps: list[AgentStep]
    rationale: str
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [asdict(step) for step in self.steps]
        return payload


@dataclass
class AgentTrace:
    prompt: str
    plan: AgentPlan
    observations: list[str]
    draft_answer: str
    verification_notes: list[str]
    final_answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "plan": self.plan.to_dict(),
            "observations": list(self.observations),
            "draft_answer": self.draft_answer,
            "verification_notes": list(self.verification_notes),
            "final_answer": self.final_answer,
        }


class AgentPlanner:
    def __init__(self, skill_bank: SkillKernelBank | None = None, max_steps: int = 8):
        self.skill_bank = skill_bank or SkillKernelBank()
        self.max_steps = max(4, max_steps)

    def _complexity(self, prompt: str) -> tuple[str, float]:
        normalized = normalize_text(prompt)
        tokens = normalized.split()
        token_count = len(tokens)
        keywords = {
            "design",
            "build",
            "implement",
            "derive",
            "prove",
            "optimize",
            "train",
            "evaluate",
            "compare",
            "debug",
            "plan",
            "analyze",
            "reason",
            "agent",
            "tool",
            "architecture",
            "frontier",
            "execute",
            "trace",
            "compile",
            "promote",
        }
        keyword_hits = sum(1 for token in tokens if token in keywords)
        punctuation_bonus = min(3, prompt.count("?") + prompt.count(":"))
        structure_bonus = 1.0 if any(mark in prompt for mark in ["\n", "- ", "1.", "2.", "3."]) else 0.0
        complexity_score = clamp((token_count / 180.0) + keyword_hits * 0.12 + punctuation_bonus * 0.05 + structure_bonus * 0.08)
        if complexity_score < 0.25:
            return "simple", complexity_score
        if complexity_score < 0.55:
            return "moderate", complexity_score
        if complexity_score < 0.82:
            return "complex", complexity_score
        return "frontier", complexity_score

    def _base_steps(self, prompt: str, skill_route: SkillRoute, complexity: str) -> list[AgentStep]:
        steps = [
            AgentStep(
                index=1,
                title="Clarify the mission",
                objective="Restate the user goal in operational terms and identify the final deliverable.",
                evidence_needed=["problem statement", "output format", "success criteria"],
                tools=["planner"],
                verification=["can the task be answered in one line?", "what must be preserved exactly?"],
            ),
            AgentStep(
                index=2,
                title="Collect constraints",
                objective="Extract hard constraints, hidden assumptions, and any missing information.",
                evidence_needed=["constraints", "limits", "required context"],
                tools=["evidence-check"],
                verification=["does the answer depend on a missing fact?"],
            ),
            AgentStep(
                index=3,
                title="Search internal skills",
                objective="Look for a built-in skill kernel or reusable routine that matches the task family.",
                evidence_needed=["skill family match", "confidence threshold"],
                tools=["skill-bank", "kernel-match", "trace-compilation"],
                verification=["is there a safe reusable kernel?"],
            ),
            AgentStep(
                index=4,
                title="Break the task apart",
                objective="Split the task into manageable subproblems and choose the hardest first.",
                evidence_needed=["subtask list", "dependency order"],
                tools=["decomposer"],
                verification=["are subtasks ordered correctly?"],
            ),
            AgentStep(
                index=5,
                title="Solve and cross-check",
                objective="Produce a draft solution, then test it against the constraints and alternatives.",
                evidence_needed=["draft answer", "alternative path", "sanity check"],
                tools=["reasoner", "critic"],
                verification=["does the draft violate any constraint?"],
            ),
            AgentStep(
                index=6,
                title="Assemble the final answer",
                objective="Write the output in the requested style and with the requested level of detail.",
                evidence_needed=["final format", "tone", "mandatory details"],
                tools=["writer"],
                verification=["does the answer directly satisfy the request?"],
            ),
        ]

        if complexity in {"complex", "frontier"}:
            steps.extend(
                [
                    AgentStep(
                        index=7,
                        title="Run a counterexample pass",
                        objective="Check whether the answer fails under a plausible counterexample or edge case.",
                        evidence_needed=["counterexample", "edge case"],
                        tools=["critic", "counterexample"],
                        verification=["can the answer survive a contradiction test?"],
                        risk="medium",
                    ),
                    AgentStep(
                        index=8,
                        title="Compress the reasoning trace",
                        objective="Summarize the logic into a compact, reusable chain without losing correctness.",
                        evidence_needed=["reasoning trace", "compressed summary"],
                        tools=["compressor", "trace-compiler"],
                        verification=["did compression remove any essential step?"],
                        risk="low",
                    ),
                ]
            )

        if complexity == "frontier":
            steps.append(
                AgentStep(
                    index=9,
                    title="Promote the lesson into memory",
                    objective="If the task family is recurring, promote the successful pattern into the internal skill bank.",
                    evidence_needed=["successful trace", "repeated task family", "passing gates"],
                    tools=["skill-promoter", "kernel-store"],
                    verification=["should this become a reusable kernel?"],
                    risk="low",
                )
            )

        return steps[: self.max_steps]

    def plan(self, prompt: str) -> AgentPlan:
        skill_route = self.skill_bank.route(prompt)
        complexity, complexity_score = self._complexity(prompt)
        curiosity = self.skill_bank.curiosity_questions(prompt)
        if skill_route.reason == "no internal skills":
            skill_route = SkillRoute(
                mode=skill_route.mode,
                kernel_id=skill_route.kernel_id,
                confidence=skill_route.confidence,
                reason="fallback to general planning",
                evidence_debt=skill_route.evidence_debt,
                selected_kernel=skill_route.selected_kernel,
                curiosity_questions=curiosity,
            )
        steps = self._base_steps(prompt, skill_route, complexity)
        semantic_pressure = 0.5 * semantic_similarity(prompt, skill_route.reason) + 0.5 * jaccard_similarity(prompt, skill_route.reason)
        confidence = clamp(0.25 + complexity_score * 0.25 + skill_route.confidence * 0.20 + semantic_pressure * 0.10 + (0.15 if skill_route.mode == "skill" else 0.0))
        strategy = "direct"
        if complexity in {"complex", "frontier"}:
            strategy = "stepwise"
        if skill_route.mode == "skill":
            strategy = "skill-first"
        rationale = (
            f"Complexity={complexity}; matched kernel={skill_route.kernel_id or 'none'}; "
            f"evidence_debt={skill_route.evidence_debt:.2f}; stepwise reasoning enabled."
        )
        return AgentPlan(
            prompt=prompt,
            complexity=complexity,
            estimated_steps=len(steps),
            confidence=confidence,
            skill_route={
                "mode": skill_route.mode,
                "kernel_id": skill_route.kernel_id,
                "confidence": skill_route.confidence,
                "reason": skill_route.reason,
                "evidence_debt": skill_route.evidence_debt,
                "selected_kernel": skill_route.selected_kernel,
            },
            curiosity_questions=curiosity,
            steps=steps,
            rationale=rationale,
            strategy=strategy,
        )

    def trace(self, prompt: str, draft_answer: str = "", final_answer: str = "") -> AgentTrace:
        plan = self.plan(prompt)
        observations = [
            f"complexity={plan.complexity}",
            f"skill_route={plan.skill_route['mode']}",
            f"estimated_steps={plan.estimated_steps}",
        ]
        verification_notes = [
            "check constraints before finishing",
            "prefer exactness over fluency when the two conflict",
        ]
        return AgentTrace(
            prompt=prompt,
            plan=plan,
            observations=observations,
            draft_answer=draft_answer,
            verification_notes=verification_notes,
            final_answer=final_answer or draft_answer,
        )
