"""Output quality scoring for benchmark evaluation.

Evaluates output quality across multiple dimensions:
- Correctness: Does the output address the task?
- Completeness: Are all required elements present?
- Clarity: Is the output understandable?
- Actionability: Can the output be used directly?
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class QualityScore:
    """Quality score across multiple dimensions."""
    correctness: float  # 0-1: Does output address the task?
    completeness: float  # 0-1: Are all required elements present?
    clarity: float  # 0-1: Is output understandable?
    actionability: float  # 0-1: Can output be used directly?
    overall: float  # Weighted average
    
    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "actionability": self.actionability,
            "overall": self.overall,
        }


def score_output(
    output: str,
    task: dict[str, Any],
    success_criteria: dict[str, Any],
) -> QualityScore:
    """Score the quality of an output.
    
    Args:
        output: The generated output
        task: The original task
        success_criteria: Criteria for success
    
    Returns:
        QualityScore with dimension scores
    """
    output_lower = output.lower()
    instruction = task.get("instruction", "").lower()
    
    # Correctness: Does output address the task?
    correctness = score_correctness(output_lower, instruction, success_criteria)
    
    # Completeness: Are all required elements present?
    completeness = score_completeness(output_lower, success_criteria)
    
    # Clarity: Is output understandable?
    clarity = score_clarity(output)
    
    # Actionability: Can output be used directly?
    actionability = score_actionability(output_lower, success_criteria)
    
    # Overall: Weighted average
    overall = (
        0.35 * correctness +
        0.25 * completeness +
        0.20 * clarity +
        0.20 * actionability
    )
    
    return QualityScore(
        correctness=correctness,
        completeness=completeness,
        clarity=clarity,
        actionability=actionability,
        overall=overall,
    )


def score_correctness(output: str, instruction: str, criteria: dict) -> float:
    """Score whether output correctly addresses the task."""
    score = 0.0
    
    # Task type matching
    if "compare" in instruction:
        # Should mention both items being compared
        if "both" in output or "and" in output:
            score += 0.3
        # Should have differences section
        if "differ" in output or "change" in output:
            score += 0.3
        # Should be specific
        if any(c.isdigit() for c in output):
            score += 0.2
    
    elif "analyze" in instruction or "review" in instruction:
        # Should identify issues
        if "issue" in output or "problem" in output or "concern" in output:
            score += 0.3
        # Should be specific
        if "line" in output or "function" in output or "class" in output:
            score += 0.3
        # Should have recommendations
        if "recommend" in output or "suggest" in output or "should" in output:
            score += 0.2
    
    elif "generate" in instruction or "create" in instruction:
        # Should produce code/text
        if "def " in output or "function" in output or "class " in output:
            score += 0.4
        # Should be complete
        if "```" in output or len(output) > 100:
            score += 0.3
    
    elif "search" in instruction or "find" in instruction:
        # Should list results
        if "found" in output or "result" in output:
            score += 0.3
        # Should have locations
        if ".py" in output or "line" in output:
            score += 0.3
        # Should be ranked/counted
        if any(c.isdigit() for c in output):
            score += 0.2
    
    elif "debug" in instruction or "fix" in instruction:
        # Should identify cause
        if "cause" in output or "reason" in output or "because" in output:
            score += 0.4
        # Should suggest fix
        if "fix" in output or "solution" in output or "change" in output:
            score += 0.3
    
    else:
        # Generic correctness
        if len(output) > 50:
            score += 0.3
        if any(kw in output for kw in ["result", "output", "complete"]):
            score += 0.3
    
    return min(1.0, score)


def score_completeness(output: str, criteria: dict) -> float:
    """Score whether all required elements are present."""
    score = 0.0
    total_elements = 0
    
    # Check for required elements in criteria
    if "both_files_mentioned" in criteria:
        total_elements += 1
        # Check if output mentions both items
        if output.count("file") >= 2 or "both" in output:
            score += 1
    
    if "differences_categorized" in criteria:
        total_elements += 1
        if "structural" in output or "content" in output or "functional" in output:
            score += 1
    
    if "has_complexity_metrics" in criteria:
        total_elements += 1
        if "complexity" in output or "cyclomatic" in output:
            score += 1
    
    if "vulnerabilities_found" in criteria:
        total_elements += 1
        if "vulnerability" in output or "security" in output:
            score += 1
    
    if "tests_runnable" in criteria:
        total_elements += 1
        if "assert" in output or "test" in output:
            score += 1
    
    if "valid_json" in criteria or "valid_yaml" in criteria:
        total_elements += 1
        if "{" in output or ":" in output:
            score += 1
    
    # Default: check for basic structure
    if total_elements == 0:
        total_elements = 3
        if len(output) > 50:
            score += 1
        if output.count("\n") >= 2:
            score += 1
        if any(c.isdigit() for c in output):
            score += 1
    
    return score / max(1, total_elements)


def score_clarity(output: str) -> float:
    """Score whether output is clear and understandable."""
    score = 0.0
    
    # Length check (not too short, not too verbose)
    if 50 < len(output) < 500:
        score += 0.3
    elif len(output) >= 500:
        score += 0.2  # Long outputs lose some clarity
    
    # Structure check
    if "\n\n" in output:  # Paragraph breaks
        score += 0.2
    if any(marker in output for marker in ["1.", "-", "*", "•"]):  # Lists
        score += 0.2
    
    # Formatting check
    if "```" in output:  # Code blocks
        score += 0.2
    elif output.count("\n") > 3:  # Multi-line output
        score += 0.1
    
    # Technical term check (should explain or use clearly)
    jargon = ["refactor", "optimize", "implement", "architecture"]
    if any(term in output.lower() for term in jargon):
        score += 0.1  # Uses technical terms
    
    return min(1.0, score)


def score_actionability(output: str, criteria: dict) -> float:
    """Score whether output can be used directly."""
    score = 0.0
    
    # Code output
    if "def " in output or "function " in output:
        score += 0.4
        if "return" in output:
            score += 0.2  # Complete function
    
    # Commands
    if "$ " in output or "npm " in output or "pip " in output:
        score += 0.3
    
    # Specific suggestions
    if "change line" in output or "replace" in output or "modify" in output:
        score += 0.3
    
    # File references
    if ".py" in output or ".js" in output:
        score += 0.2
    
    # Steps/instructions
    if "step 1" in output or "first" in output:
        score += 0.2
    
    # Default: check for concrete information
    if len([c for c in output if c.isdigit()]) > 0:
        score += 0.1
    
    return min(1.0, score)


def compare_outputs(
    outputs: dict[str, str],
    task: dict[str, Any],
    success_criteria: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compare multiple outputs for the same task.
    
    Args:
        outputs: Dict mapping system name to output
        task: The original task
        success_criteria: Criteria for success
    
    Returns:
        Dict mapping system name to quality scores
    """
    scores = {}
    for system_name, output in outputs.items():
        quality = score_output(output, task, success_criteria)
        scores[system_name] = quality.to_dict()
    
    return scores
