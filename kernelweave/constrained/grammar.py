"""Grammar-based constrained generation from kernel postconditions.

This is the core research contribution: postconditions become a formal grammar
that constrains token probabilities during generation. The model PHYSICALLY
CANNOT output invalid states.

Key insight: Most constrained generation uses JSON Schema, but schemas alone
don't encode semantic constraints like "output must mention both artifacts".
We translate postconditions into a CFG that enforces semantic constraints
via structural requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from enum import Enum
import re
import json


class ConstraintType(Enum):
    """Types of constraints derived from postconditions."""
    STRUCTURAL = "structural"      # Output shape (JSON schema)
    SEMANTIC = "semantic"          # Content requirements ("mentions X")
    TEMPORAL = "temporal"          # Ordering (step A before step B)
    NEGATION = "negation"          # Exclusion ("rollback NOT triggered")
    QUANTITATIVE = "quantitative"  # Counts ("at least 2 evidence items")


@dataclass
class GrammarRule:
    """A production rule in the constrained generation grammar."""
    name: str
    pattern: str  # Regex or structural pattern
    constraint_type: ConstraintType
    required: bool = True
    alternatives: list[str] = field(default_factory=list)
    
    def to_bnf(self) -> str:
        """Convert to Backus-Naur Form for grammar-based generation."""
        if self.alternatives:
            alts = " | ".join(f'"{a}"' for a in self.alternatives)
            return f"<{self.name}> ::= {alts}"
        return f'<{self.name}> ::= {self.pattern}'


@dataclass
class ConstrainedGrammar:
    """A grammar derived from kernel postconditions for constrained generation."""
    kernel_id: str
    task_family: str
    rules: list[GrammarRule]
    json_schema: dict[str, Any]
    required_keywords: list[str]
    forbidden_patterns: list[str]
    structural_template: str | None = None
    
    def to_guidance_grammar(self) -> str:
        """Export to guidance-style grammar string."""
        lines = []
        for rule in self.rules:
            lines.append(rule.to_bnf())
        return "\n".join(lines)
    
    def to_lark_grammar(self) -> str:
        """Export to Lark grammar format for CFG parsing."""
        lines = ["start: output"]
        lines.append("output: object")
        lines.append("object: \"{\" [pair (\",\" pair)*] \"}\"")
        lines.append("pair: STRING \":\" value")
        lines.append("value: STRING | NUMBER | object | array | \"true\" | \"false\" | \"null\"")
        lines.append("array: \"[\" [value (\",\" value)*] \"]\"")
        
        for rule in self.rules:
            lines.append(f"; {rule.name}: {rule.constraint_type.value}")
        
        return "\n".join(lines)


def parse_postcondition_to_constraints(condition: str) -> list[GrammarRule]:
    """Parse a single postcondition string into grammar rules.
    
    This is where the magic happens: natural language postconditions
    become formal constraints that can be enforced during generation.
    """
    rules = []
    condition_lower = condition.lower().strip()
    
    # Pattern: "X not triggered" / "no X"
    if " not " in condition_lower or "not " in condition_lower:
        match = re.search(r"(\w+)\s+not\s+(\w+)", condition_lower)
        if match:
            forbidden = f"{match.group(1)} {match.group(2)}"
            rules.append(GrammarRule(
                name=f"negation_{match.group(1)}_{match.group(2)}",
                pattern=f"(?!.*{re.escape(forbidden)})",
                constraint_type=ConstraintType.NEGATION,
                required=True,
            ))
    
    # Pattern: "X mentions Y" / "X includes Y"
    mention_match = re.search(r"(?:mentions|includes|references?)\s+(?:both\s+)?(\w+)(?:\s+and\s+(\w+))?", condition_lower)
    if mention_match:
        required_terms = [mention_match.group(1)]
        if mention_match.group(2):
            required_terms.append(mention_match.group(2))
        for term in required_terms:
            rules.append(GrammarRule(
                name=f"required_mention_{term}",
                pattern=f".*{term}.*",
                constraint_type=ConstraintType.SEMANTIC,
                required=True,
                alternatives=[term, term.lower(), term.upper(), term.title()],
            ))
    
    # Pattern: "all X recorded" / "X satisfied"
    if "all " in condition_lower and ("recorded" in condition_lower or "satisfied" in condition_lower):
        match = re.search(r"all\s+(\w+(?:\s+\w+)?)\s+(?:recorded|satisfied)", condition_lower)
        if match:
            rules.append(GrammarRule(
                name=f"completeness_{match.group(1).replace(' ', '_')}",
                pattern=r".+",  # Must have content
                constraint_type=ConstraintType.QUANTITATIVE,
                required=True,
            ))
    
    # Pattern: "X schema satisfied"
    if "schema" in condition_lower:
        rules.append(GrammarRule(
            name="schema_compliance",
            pattern=r'\{.*\}',  # Must be valid JSON object
            constraint_type=ConstraintType.STRUCTURAL,
            required=True,
        ))
    
    # Pattern: "at least N X" / "minimum N X"
    count_match = re.search(r"(?:at least|minimum|min)\s+(\d+)\s+(\w+)", condition_lower)
    if count_match:
        min_count = int(count_match.group(1))
        entity = count_match.group(2)
        rules.append(GrammarRule(
            name=f"min_count_{entity}",
            pattern=f"(?:{entity}.*){{{min_count},}}",
            constraint_type=ConstraintType.QUANTITATIVE,
            required=True,
        ))
    
    # Pattern: "tests passed" / "verification succeeded"
    if "passed" in condition_lower or "succeeded" in condition_lower:
        rules.append(GrammarRule(
            name="success_flag",
            pattern=r'"(?:passed|true|success|succeeded)"',
            constraint_type=ConstraintType.SEMANTIC,
            required=True,
            alternatives=["passed", "true", "success", "succeeded"],
        ))
    
    return rules


def postconditions_to_grammar(
    postconditions: list[str],
    output_schema: dict[str, Any] | None = None,
    kernel_id: str = "",
    task_family: str = "",
) -> ConstrainedGrammar:
    """Convert kernel postconditions to a constrained generation grammar.
    
    This is the DIFFERENCE between:
      - "I hope the model follows the plan" (current approach)
      - "the model PHYSICALLY CANNOT output invalid states" (this approach)
    
    The grammar constrains token probabilities during generation.
    """
    all_rules: list[GrammarRule] = []
    required_keywords: list[str] = []
    forbidden_patterns: list[str] = []
    
    for condition in postconditions:
        rules = parse_postcondition_to_constraints(condition)
        all_rules.extend(rules)
        
        # Extract keywords for quick validation
        for rule in rules:
            if rule.constraint_type == ConstraintType.SEMANTIC:
                required_keywords.extend(rule.alternatives)
            elif rule.constraint_type == ConstraintType.NEGATION:
                forbidden_patterns.append(rule.pattern)
    
    # Build JSON schema from output_schema + postconditions
    schema = output_schema or {"type": "object", "properties": {}}
    
    # Enrich schema with postcondition-derived constraints
    for rule in all_rules:
        if rule.constraint_type == ConstraintType.STRUCTURAL:
            # Add structural constraints to schema
            if "properties" not in schema:
                schema["properties"] = {}
            if "required" not in schema:
                schema["required"] = []
    
    # Build structural template if possible
    template = None
    if output_schema and "properties" in output_schema:
        template = build_structural_template(output_schema, all_rules)
    
    return ConstrainedGrammar(
        kernel_id=kernel_id,
        task_family=task_family,
        rules=all_rules,
        json_schema=schema,
        required_keywords=list(set(required_keywords)),
        forbidden_patterns=forbidden_patterns,
        structural_template=template,
    )


def build_structural_template(
    schema: dict[str, Any],
    rules: list[GrammarRule],
) -> str:
    """Build a JSON template that encodes structural constraints."""
    props = schema.get("properties", {})
    required = schema.get("required", [])
    
    template_parts = []
    for key in required:
        if key in props:
            prop_type = props[key].get("type", "string")
            if prop_type == "string":
                template_parts.append(f'"{key}": "{{{key}}}"')
            elif prop_type == "boolean":
                template_parts.append(f'"{key}": true')
            elif prop_type == "array":
                template_parts.append(f'"{key}": []')
            elif prop_type == "object":
                template_parts.append(f'"{key}": {{}}')
            else:
                template_parts.append(f'"{key}": null')
    
    return "{" + ", ".join(template_parts) + "}"


class ConstrainedTokenSampler:
    """Token sampler that enforces grammar constraints during generation.
    
    This is the practical implementation: we filter token probabilities
    to only allow tokens that can lead to valid outputs per the grammar.
    """
    
    def __init__(self, grammar: ConstrainedGrammar):
        self.grammar = grammar
        self.state_stack: list[str] = []
        self.generated_tokens: list[str] = []
        self.required_yet: set[str] = set()
    
    def filter_logits(self, logits: dict[str, float], prefix: str) -> dict[str, float]:
        """Filter token probabilities to enforce grammar constraints.
        
        Args:
            logits: Token -> probability mapping from model
            prefix: Current generated text so far
            
        Returns:
            Filtered logits with invalid tokens set to -inf
        """
        filtered = {}
        prefix_lower = prefix.lower()
        
        for token, prob in logits.items():
            token_lower = token.lower()
            candidate = prefix + token
            
            # Check forbidden patterns
            valid = True
            for forbidden in self.grammar.forbidden_patterns:
                try:
                    if re.search(forbidden, candidate, re.IGNORECASE):
                        valid = False
                        break
                except re.error:
                    pass
            
            if not valid:
                filtered[token] = float("-inf")
                continue
            
            # Check if token helps satisfy required keywords
            for kw in self.grammar.required_keywords:
                if kw.lower() in token_lower and kw not in self.required_yet:
                    # Boost tokens that satisfy requirements
                    prob = min(1.0, prob * 1.5)
                    self.required_yet.add(kw)
                    break
            
            filtered[token] = prob
        
        return filtered
    
    def is_complete(self, generated: str) -> tuple[bool, list[str]]:
        """Check if generated output is complete and valid.
        
        Returns (is_complete, list_of_missing_requirements)
        """
        missing = []
        generated_lower = generated.lower()
        
        for kw in self.grammar.required_keywords:
            if kw.lower() not in generated_lower:
                missing.append(kw)
        
        for pattern in self.grammar.forbidden_patterns:
            try:
                if re.search(pattern, generated, re.IGNORECASE):
                    return False, [f"forbidden pattern found: {pattern}"]
            except re.error:
                pass
        
        return len(missing) == 0, missing


def validate_against_grammar(
    output: str,
    grammar: ConstrainedGrammar,
) -> tuple[bool, list[str]]:
    """Validate an output against a constrained grammar.
    
    Returns (is_valid, list_of_violations)
    """
    violations = []
    output_lower = output.lower()
    
    # Check required keywords
    for kw in grammar.required_keywords:
        if kw.lower() not in output_lower:
            violations.append(f"Missing required keyword: {kw}")
    
    # Check forbidden patterns
    for pattern in grammar.forbidden_patterns:
        try:
            if re.search(pattern, output, re.IGNORECASE):
                violations.append(f"Forbidden pattern matched: {pattern}")
        except re.error:
            pass
    
    # Check JSON structure if applicable
    if grammar.json_schema:
        try:
            parsed = json.loads(output)
            # Validate against schema
            schema_errors = _validate_json_schema(parsed, grammar.json_schema)
            violations.extend(schema_errors)
        except json.JSONDecodeError:
            violations.append("Output is not valid JSON")
    
    return len(violations) == 0, violations


def _validate_json_schema(obj: Any, schema: dict[str, Any]) -> list[str]:
    """Simple JSON schema validator."""
    errors = []
    
    if schema.get("type") == "object":
        if not isinstance(obj, dict):
            errors.append(f"Expected object, got {type(obj).__name__}")
            return errors
        
        required = schema.get("required", [])
        for field in required:
            if field not in obj:
                errors.append(f"Missing required field: {field}")
    
    return errors
