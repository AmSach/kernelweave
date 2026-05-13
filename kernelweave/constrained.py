"""
Structured decoding — constrain model outputs to valid kernel states.

Converts kernel postconditions into JSON schemas and uses constrained generation
to ensure model outputs are physically valid according to the kernel's contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import json
import re


@dataclass
class ConstrainedOutput:
    """Result of constrained generation."""
    text: str
    parsed: dict[str, Any]
    valid: bool
    validation_errors: list[str]
    attempts: int
    schema_used: dict[str, Any]


def postconditions_to_schema(postconditions: list[str], output_schema: dict[str, Any] | None = None, backend: Any | None = None) -> dict[str, Any]:
    """
    Convert postcondition strings into a JSON schema.
    
    If backend is provided, uses LLM to generate a precise schema.
    Otherwise, falls back to best-effort regex translation.
    """
    if backend is not None:
        try:
            import json
            import re
            
            prompt = f"""Convert these natural language postconditions into a JSON schema.
Postconditions:
{json.dumps(postconditions, indent=2)}

Return ONLY valid JSON representing the JSON schema (type: "object"). Include properties, types, and required fields based on the postconditions.
"""
            response = backend.generate(prompt, system_prompt="You are an expert at converting natural language constraints into JSON schemas. Output ONLY valid JSON.")
            text = response.text.strip()
            
            try:
                schema = json.loads(text)
                if "type" not in schema:
                    schema["type"] = "object"
                return schema
            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    return json.loads(json_match.group())
                # Fallback to regex if LLM output is unparsable
        except Exception:
            pass # Fallback to regex

    # Original regex fallback
    properties = {}
    required = []
    
    if output_schema and "properties" in output_schema:
        properties = dict(output_schema["properties"])
    
    for condition in postconditions:
        cond_lower = condition.lower()
        
        # Extract field names from conditions
        field_match = re.search(r"(\w+)\s+(?:must\s+be|is|should\s+be)", cond_lower)
        if field_match:
            field_name = field_match.group(1)
            if field_name not in properties:
                properties[field_name] = {"type": "string"}
            required.append(field_name)
        
        # Detect type requirements
        if "number" in cond_lower or "numeric" in cond_lower or "count" in cond_lower:
            field_match = re.search(r"(\w+)\s+(?:must\s+be|is|should\s+be)", cond_lower)
            if field_match:
                properties[field_match.group(1)] = {"type": "number"}
        elif "boolean" in cond_lower or "true or false" in cond_lower:
            field_match = re.search(r"(\w+)\s+(?:must\s+be|is|should\s+be)", cond_lower)
            if field_match:
                properties[field_match.group(1)] = {"type": "boolean"}
        elif "list" in cond_lower or "array" in cond_lower:
            field_match = re.search(r"(\w+)\s+(?:must\s+be|is|should\s+be)", cond_lower)
            if field_match:
                properties[field_match.group(1)] = {"type": "array", "items": {"type": "string"}}
        
        # Detect enum values
        enum_match = re.search(r"(?:must\s+be|is|should\s+be)\s+(.+?)(?:\s+and|\s*$|\s*\.)", cond_lower)
        if enum_match:
            values_str = enum_match.group(1)
            values = [v.strip().strip('"\'') for v in re.split(r'\s+or\s+|\s*,\s*', values_str)]
            if len(values) > 1 and all(len(v) < 50 for v in values):
                if field_match:
                    properties[field_match.group(1)] = {"type": "string", "enum": values}
    
    # Ensure result field exists
    if "result" not in properties:
        properties["result"] = {"type": "string", "description": "The primary output"}
    if "result" not in required:
        required.append("result")
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(set(required)),
        "additionalProperties": True,
    }
    
    return schema


def generate_with_retry(
    backend: Any,
    prompt: str,
    schema: dict[str, Any],
    max_attempts: int = 3,
    temperature: float = 0.7,
    system_prompt: str = "",
    validator: Callable[[dict], bool] | None = None,
) -> ConstrainedOutput:
    """
    Generate with schema validation and retry.
    
    Attempts to generate output that matches the schema. On failure,
    retries with adjusted prompts.
    """
    attempts = 0
    errors = []
    
    for attempt in range(max_attempts):
        attempts += 1
        
        # Adjust system prompt based on attempt
        if attempt == 0:
            sp = system_prompt
        else:
            sp = f"{system_prompt}\n\nPrevious output was invalid. Errors: {'; '.join(errors[-3:])}\nEnsure output is valid JSON matching the required schema."
        
        try:
            response = backend.generate(
                prompt,
                system_prompt=sp,
                temperature=temperature,
            )
            text = response.text.strip()
            
            # Try to parse as JSON
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        errors.append(f"Not valid JSON on attempt {attempt+1}")
                        continue
                else:
                    errors.append(f"No JSON found in output on attempt {attempt+1}")
                    continue
            
            # Validate against schema (basic check)
            validation_errors = validate_against_schema(parsed, schema)
            
            if validation_errors:
                errors.extend(validation_errors)
                continue
            
            # Custom validator if provided
            if validator and not validator(parsed):
                errors.append("Custom validation failed")
                continue
            
            return ConstrainedOutput(
                text=text,
                parsed=parsed,
                valid=True,
                validation_errors=[],
                attempts=attempts,
                schema_used=schema,
            )
            
        except Exception as e:
            errors.append(str(e))
    
    # All attempts failed - return last attempt
    return ConstrainedOutput(
        text=text if 'text' in locals() else "",
        parsed=parsed if 'parsed' in locals() else {},
        valid=False,
        validation_errors=errors,
        attempts=attempts,
        schema_used=schema,
    )


def validate_against_schema(obj: dict, schema: dict) -> list[str]:
    """Basic schema validation - check required fields and types."""
    errors = []
    
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    for field in required:
        if field not in obj:
            errors.append(f"Missing required field: {field}")
    
    for key, value in obj.items():
        if key in properties:
            prop_spec = properties[key]
            expected_type = prop_spec.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Field {key} should be string, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field {key} should be number, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Field {key} should be boolean, got {type(value).__name__}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"Field {key} should be array, got {type(value).__name__}")
            
            # Check enum
            if "enum" in prop_spec and value not in prop_spec["enum"]:
                errors.append(f"Field {key} must be one of {prop_spec['enum']}, got {value}")
    
    return errors


class ConstrainedGenerator:
    """
    Generator that enforces kernel constraints on outputs.
    
    Uses structured decoding to ensure outputs are valid according to
    the kernel's postconditions and output schema.
    """
    
    def __init__(self, backend: Any, use_guidance: bool = True):
        self.backend = backend
        self.use_guidance = use_guidance
        self._guidance_model = None
    
    def _init_guidance(self):
        """Initialize guidance model for constrained generation."""
        if not self.use_guidance:
            return None
        try:
            import guidance
            # Use the same model as backend
            if hasattr(self.backend, 'preset'):
                model_name = self.backend.preset.model
                if 'gpt' in model_name.lower() or 'openai' in model_name.lower():
                    return guidance.models.OpenAI(model_name)
            return None
        except ImportError:
            return None
    
    def generate_constrained(
        self,
        prompt: str,
        kernel: Any,
        max_attempts: int = 3,
    ) -> ConstrainedOutput:
        """
        Generate output constrained by kernel's postconditions.
        
        The kernel's output_schema and postconditions are converted to
        a JSON schema that constrains generation.
        """
        schema = postconditions_to_schema(
            kernel.postconditions,
            kernel.output_schema,
            backend=self.backend
        )
        
        steps_text = "\n".join(
            f"{i+1}. {step.get('action', 'step')}: {step.get('text', step.get('tool', ''))}"
            for i, step in enumerate(kernel.steps)
        )
        
        system_prompt = f"""Execute the kernel plan for {kernel.task_family}:

{steps_text}

Output a JSON object with fields matching these requirements:
{json.dumps(schema.get('properties', {}), indent=2)}

Required fields: {', '.join(schema.get('required', []))}

Evidence requirements: {', '.join(kernel.evidence_requirements)}
Postconditions to satisfy: {', '.join(kernel.postconditions)}

Output ONLY valid JSON, no additional text."""

        return generate_with_retry(
            self.backend,
            prompt,
            schema,
            max_attempts=max_attempts,
            system_prompt=system_prompt,
        )
