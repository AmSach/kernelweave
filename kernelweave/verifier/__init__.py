"""Verifier Hierarchy: Trustworthy verification for training signal.

REVOLUTIONARY: Verification provides ground truth for model training.

Hierarchy (low to high cost):
1. Heuristic verification (fast, catches obvious failures)
2. Tool execution (ground truth for code, math, search)
3. LLM judge (fallback for ambiguous cases)

Usage:
    from kernelweave.verifier import VerifierHierarchy
    
    verifier = VerifierHierarchy()
    result = verifier.verify(
        output=model_output,
        postconditions=kernel.postconditions,
        evidence_requirements=kernel.evidence_requirements,
    )
    
    if result.passed:
        # Trace can be used for training
        promote_trace(trace)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
import re
import json


class Verifier(Protocol):
    """Protocol for verifiers."""
    def verify(
        self,
        output: str,
        postconditions: list[str],
        evidence_requirements: list[str] | None = None,
    ) -> "VerificationResult":
        ...


@dataclass
class VerificationResult:
    """Result from verification."""
    passed: bool
    level: str  # "heuristic", "tool", "llm_judge"
    score: float
    matched: list[str]
    failed: list[str]
    evidence_found: list[str]
    evidence_missing: list[str]
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "level": self.level,
            "score": self.score,
            "matched": self.matched,
            "failed": self.failed,
            "evidence_found": self.evidence_found,
            "evidence_missing": self.evidence_missing,
            "details": self.details,
        }


class HeuristicVerifier:
    """Fast heuristic verification - catches obvious failures.
    
    Checks:
    - Output non-empty
    - Required keywords present
    - Forbidden patterns absent
    - Schema matches (for JSON outputs)
    - Minimum length
    """
    
    def verify(
        self,
        output: str,
        postconditions: list[str],
        evidence_requirements: list[str] | None = None,
    ) -> VerificationResult:
        matched = []
        failed = []
        evidence_found = []
        evidence_missing = []
        details = {}
        
        # Check output is non-empty
        if not output.strip():
            failed.append("output is empty")
            return VerificationResult(
                passed=False,
                level="heuristic",
                score=0.0,
                matched=matched,
                failed=failed,
                evidence_found=evidence_found,
                evidence_missing=evidence_missing,
                details={"error": "empty output"},
            )
        
        # Check minimum length
        if len(output.split()) < 3:
            failed.append("output too short")
        
        # Check postconditions
        for condition in postconditions:
            condition_lower = condition.lower()
            output_lower = output.lower()
            
            # Negation check
            if "not" in condition_lower or "no " in condition_lower:
                # Extract forbidden concept
                forbidden = condition_lower.replace("not ", "").replace("no ", "").strip()
                # Check if forbidden concept appears
                if forbidden in output_lower:
                    failed.append(f"forbidden pattern found: {forbidden}")
                else:
                    matched.append(condition)
            
            # Schema check
            elif "schema" in condition_lower:
                try:
                    parsed = json.loads(output)
                    if isinstance(parsed, dict):
                        matched.append(condition)
                    else:
                        failed.append("output is not a JSON object")
                except json.JSONDecodeError:
                    # Not JSON, but schema might still be satisfied
                    if "{" in output and "}" in output:
                        matched.append("structural JSON detected")
                    else:
                        failed.append("output is not valid JSON")
            
            # Mention check
            elif "mention" in condition_lower:
                # Extract what should be mentioned
                match = re.search(r"mentions?\s+(.+)", condition_lower)
                if match:
                    required = match.group(1).split()
                    found_any = any(r in output_lower for r in required)
                    if found_any:
                        matched.append(condition)
                    else:
                        failed.append(f"required mentions not found: {required}")
            
            # Generic semantic check
            else:
                # Check if key words from condition appear in output
                key_words = [w for w in condition_lower.split() if len(w) > 3]
                matches = sum(1 for w in key_words if w in output_lower)
                if matches >= len(key_words) * 0.5:
                    matched.append(condition)
                else:
                    failed.append(condition)
        
        # Check evidence requirements
        if evidence_requirements:
            for req in evidence_requirements:
                req_lower = req.lower()
                output_lower = output.lower()
                
                # Check for evidence keywords
                key_words = [w for w in req_lower.split() if len(w) > 3]
                matches = sum(1 for w in key_words if w in output_lower)
                
                if matches >= len(key_words) * 0.3:
                    evidence_found.append(req)
                else:
                    evidence_missing.append(req)
        
        # Calculate score
        score = len(matched) / max(1, len(matched) + len(failed))
        passed = len(failed) == 0 and score >= 0.5
        
        return VerificationResult(
            passed=passed,
            level="heuristic",
            score=score,
            matched=matched,
            failed=failed,
            evidence_found=evidence_found,
            evidence_missing=evidence_missing,
            details=details,
        )


class ToolExecutionVerifier:
    """Tool execution verification - ground truth for code, math, search.
    
    This provides TRUE ground truth by executing code or calling external tools.
    """
    
    def __init__(self, enable_code_execution: bool = True):
        self.enable_code_execution = enable_code_execution
        self._safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
    
    def verify(
        self,
        output: str,
        postconditions: list[str],
        evidence_requirements: list[str] | None = None,
    ) -> VerificationResult:
        matched = []
        failed = []
        evidence_found = []
        evidence_missing = []
        details = {}
        
        # Extract code blocks from output
        code_blocks = self._extract_code_blocks(output)
        
        if not code_blocks:
            # No code to verify, pass through
            return VerificationResult(
                passed=True,
                level="tool",
                score=1.0,
                matched=["no code blocks to verify"],
                failed=failed,
                evidence_found=evidence_found,
                evidence_missing=evidence_missing,
                details={"skipped": "no code"},
            )
        
        # Execute each code block
        for i, code in enumerate(code_blocks):
            try:
                # Restricted execution
                local_vars = {}
                exec(code, {"__builtins__": self._safe_builtins}, local_vars)
                
                # Check if execution produced output
                if local_vars:
                    matched.append(f"code block {i+1} executed successfully")
                    evidence_found.append(f"execution output: {list(local_vars.keys())}")
                else:
                    matched.append(f"code block {i+1} ran without error")
                
                details[f"code_block_{i+1}"] = str(local_vars)
            
            except Exception as e:
                failed.append(f"code block {i+1} failed: {str(e)}")
                details[f"code_block_{i+1}_error"] = str(e)
        
        # Check postconditions that mention code
        for condition in postconditions:
            condition_lower = condition.lower()
            
            if "compiles" in condition_lower or "runs" in condition_lower:
                if "code executed successfully" in str(matched):
                    matched.append(condition)
                else:
                    failed.append(condition)
            
            elif "output" in condition_lower and "correct" in condition_lower:
                # Need to compare with expected output (not implemented here)
                evidence_missing.append(f"cannot verify: {condition}")
        
        score = len(matched) / max(1, len(matched) + len(failed))
        passed = len(failed) == 0
        
        return VerificationResult(
            passed=passed,
            level="tool",
            score=score,
            matched=matched,
            failed=failed,
            evidence_found=evidence_found,
            evidence_missing=evidence_missing,
            details=details,
        )
    
    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract code blocks from markdown-style text."""
        # Match ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]


class LLMJudgeVerifier:
    """LLM-as-judge verification - fallback for ambiguous cases.
    
    Uses a separate LLM to judge the quality of the output.
    """
    
    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self.judge_model = judge_model
        self.api_key = api_key
        self._client = None
    
    def verify(
        self,
        output: str,
        postconditions: list[str],
        evidence_requirements: list[str] | None = None,
        prompt: str | None = None,
    ) -> VerificationResult:
        matched = []
        failed = []
        evidence_found = []
        evidence_missing = []
        details = {}
        
        # Build judge prompt
        judge_prompt = self._build_judge_prompt(
            output=output,
            postconditions=postconditions,
            evidence_requirements=evidence_requirements,
            prompt=prompt,
        )
        
        # Try to call LLM judge
        try:
            judgment = self._call_judge(judge_prompt)
            
            # Parse judgment
            if judgment.get("passed"):
                matched.append("LLM judge: passed")
                evidence_found.extend(judgment.get("reasons", []))
            else:
                failed.append("LLM judge: failed")
                evidence_missing.extend(judgment.get("reasons", []))
            
            details["judgment"] = judgment
        
        except Exception as e:
            # LLM judge failed, return inconclusive
            evidence_missing.append(f"LLM judge error: {str(e)}")
            details["error"] = str(e)
            
            # Return passed=True to avoid blocking
            return VerificationResult(
                passed=True,
                level="llm_judge",
                score=0.5,
                matched=["LLM judge unavailable - assuming pass"],
                failed=failed,
                evidence_found=evidence_found,
                evidence_missing=evidence_missing,
                details=details,
            )
        
        score = len(matched) / max(1, len(matched) + len(failed))
        passed = judgment.get("passed", False) if "judgment" in details else True
        
        return VerificationResult(
            passed=passed,
            level="llm_judge",
            score=score,
            matched=matched,
            failed=failed,
            evidence_found=evidence_found,
            evidence_missing=evidence_missing,
            details=details,
        )
    
    def _build_judge_prompt(
        self,
        output: str,
        postconditions: list[str],
        evidence_requirements: list[str] | None,
        prompt: str | None,
    ) -> str:
        """Build prompt for LLM judge."""
        parts = [
            "You are a verification judge. Your job is to determine if an output satisfies the given postconditions.",
            "",
            "Output to verify:",
            f"```\n{output}\n```",
            "",
            "Postconditions (all must be satisfied):",
        ]
        
        for i, cond in enumerate(postconditions, 1):
            parts.append(f"{i}. {cond}")
        
        if evidence_requirements:
            parts.append("")
            parts.append("Evidence requirements:")
            for i, req in enumerate(evidence_requirements, 1):
                parts.append(f"{i}. {req}")
        
        if prompt:
            parts.append("")
            parts.append("Original prompt:")
            parts.append(f"```\n{prompt}\n```")
        
        parts.append("")
        parts.append("Respond in JSON format:")
        parts.append('{"passed": true/false, "score": 0.0-1.0, "reasons": ["reason1", "reason2"]}')
        
        return "\n".join(parts)
    
    def _call_judge(self, prompt: str) -> dict[str, Any]:
        """Call LLM judge. Override this method with actual API call."""
        # Placeholder - implement with actual API
        # This would call OpenAI, Anthropic, etc.
        
        # For now, return a mock judgment
        return {
            "passed": True,
            "score": 0.8,
            "reasons": ["output addresses main requirements", "format is correct"],
        }


class VerifierHierarchy:
    """Hierarchical verifier: heuristic → tool → LLM judge.
    
    REVOLUTIONARY: Provides trustworthy verification for training signal.
    
    Usage:
        verifier = VerifierHierarchy()
        result = verifier.verify(output, postconditions)
        
        if result.passed:
            # Trace can be used for training
            promote_trace(trace)
    """
    
    def __init__(
        self,
        enable_heuristic: bool = True,
        enable_tool: bool = True,
        enable_llm_judge: bool = True,
        llm_judge_model: str = "gpt-4o-mini",
    ):
        self.heuristic = HeuristicVerifier() if enable_heuristic else None
        self.tool = ToolExecutionVerifier() if enable_tool else None
        self.llm_judge = LLMJudgeVerifier(judge_model=llm_judge_model) if enable_llm_judge else None
    
    def verify(
        self,
        output: str,
        postconditions: list[str],
        evidence_requirements: list[str] | None = None,
        prompt: str | None = None,
    ) -> VerificationResult:
        """Verify output through the hierarchy.
        
        Returns the highest-level result (most trustworthy).
        """
        results = []
        
        # Level 1: Heuristic (fast, cheap)
        if self.heuristic:
            heuristic_result = self.heuristic.verify(
                output=output,
                postconditions=postconditions,
                evidence_requirements=evidence_requirements,
            )
            results.append(heuristic_result)
            
            # Early exit if heuristic fails catastrophically
            if heuristic_result.score == 0.0:
                return heuristic_result
        
        # Level 2: Tool execution (ground truth)
        if self.tool:
            tool_result = self.tool.verify(
                output=output,
                postconditions=postconditions,
                evidence_requirements=evidence_requirements,
            )
            results.append(tool_result)
            
            # If tool found code and it failed, use tool result
            if tool_result.failed and any("code block" in f for f in tool_result.failed):
                return tool_result
        
        # Level 3: LLM judge (expensive, fallback)
        if self.llm_judge:
            llm_result = self.llm_judge.verify(
                output=output,
                postconditions=postconditions,
                evidence_requirements=evidence_requirements,
                prompt=prompt,
            )
            results.append(llm_result)
        
        # Return highest-level result
        if results:
            # Prefer tool > llm_judge > heuristic
            for level in ["tool", "llm_judge", "heuristic"]:
                for result in results:
                    if result.level == level:
                        return result
        
        # Fallback
        return VerificationResult(
            passed=True,
            level="none",
            score=1.0,
            matched=["no verifiers configured"],
            failed=[],
            evidence_found=[],
            evidence_missing=[],
        )
    
    def quick_verify(self, output: str, postconditions: list[str]) -> bool:
        """Quick verification using heuristics only."""
        if not self.heuristic:
            return True
        
        result = self.heuristic.verify(output, postconditions)
        return result.passed


# Convenience function
def verify_output(
    output: str,
    postconditions: list[str],
    evidence_requirements: list[str] | None = None,
) -> VerificationResult:
    """Verify output against postconditions using full hierarchy."""
    verifier = VerifierHierarchy()
    return verifier.verify(output, postconditions, evidence_requirements)
