"""Convert kernel postconditions to JSON Schema for structured decoding."""
from __future__ import annotations

import json
import re
from typing import Any


def postconditions_to_schema(postconditions: list[str], output_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert postcondition strings to a JSON Schema for structured output.
    
    This transforms natural language postconditions into a structured schema
    that can be used with constrained generation APIs (OpenAI structured outputs,
    Anthropic tool use, guidance grammars).
    
    Example:
        postconditions = [
            "output schema satisfied",
            "all required evidence recorded",
            "comparison mentions both artifacts"
        ]
        
    Returns a schema that enforces these constraints.
    """
    properties = {}
    required = []
    
    # Extract the base output schema if provided
    if output_schema and "properties" in output_schema:
        properties = dict(output_schema.get("properties", {}))
        required = list(output_schema.get("required", []))
    
    # Parse postconditions into schema constraints
    for condition in postconditions:
        condition_lower = condition.lower().strip()
        
        # Evidence constraints
        if "evidence" in condition_lower:
            prop_name = "evidence_found"
            if prop_name not in properties:
                properties[prop_name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Evidence items satisfying: {condition}"
                }
                required.append(prop_name)
        
        # Schema satisfaction
        elif "schema" in condition_lower or "output" in condition_lower:
            # This constrains the overall output structure
            if "result" not in properties:
                properties["result"] = {
                    "type": "string",
                    "description": "Primary output result"
                }
                required.append("result")
        
        # Verification/rollback constraints
        elif "rollback" in condition_lower:
            prop_name = "rollback_triggered"
            if prop_name not in properties:
                properties[prop_name] = {
                    "type": "boolean",
                    "description": f"Whether rollback was triggered: {condition}",
                    "const": False  # Must be false for success
                }
                required.append(prop_name)
        
        # Tests/checks
        elif "test" in condition_lower or "passed" in condition_lower:
            prop_name = "tests_passed"
            if prop_name not in properties:
                properties[prop_name] = {
                    "type": "boolean",
                    "description": f"All tests passed: {condition}",
                    "const": True
                }
                required.append(prop_name)
        
        # Task-specific constraints (extract key entities)
        else:
            # Extract key nouns/phrases as properties
            key_terms = _extract_key_terms(condition)
            for term in key_terms:
                prop_name = _term_to_property_name(term)
                if prop_name not in properties:
                    properties[prop_name] = {
                        "type": "string",
                        "description": f"Required by postcondition: {condition}"
                    }
                    if "mention" in condition_lower or "include" in condition_lower:
                        required.append(prop_name)
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(set(required)),
        "additionalProperties": True,  # Allow extra fields
    }
    
    return schema


def _extract_key_terms(text: str) -> list[str]:
    """Extract key terms from a postcondition that should appear in output."""
    # Remove common filler words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "must", "shall",
                 "can", "need", "dare", "ought", "used", "to", "of", "in",
                 "for", "on", "with", "at", "by", "from", "as", "into",
                 "through", "during", "before", "after", "above", "below",
                 "between", "under", "again", "further", "then", "once",
                 "all", "any", "both", "each", "few", "more", "most", "other",
                 "some", "such", "no", "nor", "not", "only", "own", "same",
                 "so", "than", "too", "very", "satisfy", "satisfied", "output"}
    
    # Extract words (3+ chars, not stopwords)
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    terms = [w for w in words if w not in stopwords]
    
    return list(dict.fromkeys(terms))  # Preserve order, remove duplicates


def _term_to_property_name(term: str) -> str:
    """Convert a term to a valid JSON Schema property name."""
    # Simple snake_case conversion
    return term.replace(" ", "_").replace("-", "_")


def validate_output(output: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate an output against a schema.
    
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in output:
            errors.append(f"Missing required field: {field}")
            continue
        
        value = output[field]
        field_schema = schema.get("properties", {}).get(field, {})
        
        # Check const constraints
        if "const" in field_schema:
            if value != field_schema["const"]:
                errors.append(f"Field {field} must be {field_schema['const']}, got {value}")
        
        # Check type
        expected_type = field_schema.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field {field} must be string, got {type(value).__name__}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field {field} must be boolean, got {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Field {field} must be array, got {type(value).__name__}")
    
    return len(errors) == 0, errors
