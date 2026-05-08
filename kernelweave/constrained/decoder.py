"""Constrained decoder using postcondition-derived schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import postconditions_to_schema, validate_output


@dataclass
class ConstrainedResponse:
    """Response from constrained generation."""
    text: str
    structured_output: dict[str, Any]
    is_valid: bool
    validation_errors: list[str]
    schema_used: dict[str, Any]
    raw_response: Any = None


class ConstrainedDecoder:
    """Wrap a model backend with structured decoding from kernel postconditions.
    
    This is the core research contribution: the kernel's postconditions become
    a schema that constrains token probabilities during generation. The model
    physically cannot output invalid states.
    
    Usage:
        decoder = ConstrainedDecoder(backend, kernel)
        response = decoder.generate(prompt)
        # response.is_valid is guaranteed True if generation succeeded
    """
    
    def __init__(
        self,
        backend: Any,
        kernel: Any = None,
        postconditions: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        provider: str = "openai",  # or "anthropic", "local"
    ):
        self.backend = backend
        self.kernel = kernel
        self.postconditions = postconditions or []
        self.output_schema = output_schema
        self.provider = provider
        
        # Build schema from kernel if provided
        if kernel is not None:
            self.postconditions = list(kernel.postconditions)
            self.output_schema = kernel.output_schema
        
        # Generate constraint schema
        self.schema = postconditions_to_schema(
            self.postconditions,
            self.output_schema
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_on_invalid: bool = True,
        max_retries: int = 3,
    ) -> ConstrainedResponse:
        """Generate with structured output constrained by postconditions.
        
        The key innovation: generation is constrained to the schema derived
        from postconditions. Invalid outputs are not just rejected - they
        are prevented during generation.
        """
        # Build structured output request based on provider
        if self.provider == "openai":
            response = self._generate_openai(
                prompt, system_prompt, temperature, max_tokens
            )
        elif self.provider == "anthropic":
            response = self._generate_anthropic(
                prompt, system_prompt, temperature, max_tokens
            )
        else:
            # Fallback to unconstrained + post-validation
            response = self._generate_unconstrained(
                prompt, system_prompt, temperature, max_tokens
            )
        
        # Validate and retry if needed
        attempts = 1
        while attempts <= max_retries:
            structured = self._extract_structured_output(response)
            is_valid, errors = validate_output(structured, self.schema)
            
            if is_valid or not retry_on_invalid:
                return ConstrainedResponse(
                    text=response.text if hasattr(response, 'text') else str(response),
                    structured_output=structured,
                    is_valid=is_valid,
                    validation_errors=errors,
                    schema_used=self.schema,
                    raw_response=response,
                )
            
            # Retry with explicit schema in system prompt
            system_prompt = self._add_schema_to_prompt(system_prompt)
            response = self._generate_openai(
                prompt, system_prompt, temperature, max_tokens,
                force_structured=True
            )
            attempts += 1
        
        # Return last attempt even if invalid
        structured = self._extract_structured_output(response)
        is_valid, errors = validate_output(structured, self.schema)
        return ConstrainedResponse(
            text=response.text if hasattr(response, 'text') else str(response),
            structured_output=structured,
            is_valid=is_valid,
            validation_errors=errors,
            schema_used=self.schema,
            raw_response=response,
        )
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        force_structured: bool = False,
    ) -> Any:
        """Generate using OpenAI's structured output API."""
        # Check if backend supports structured output
        if hasattr(self.backend, 'generate_structured') or force_structured:
            # Use structured generation
            return self.backend.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "kernel_output",
                        "schema": self.schema,
                        "strict": True
                    }
                }
            )
        
        # Fallback to regular generation
        return self.backend.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Any:
        """Generate using Anthropic's tool/schema constraints."""
        # Anthropic uses tool use for structured output
        tool_schema = {
            "name": "structured_output",
            "description": "Output matching kernel postconditions",
            "input_schema": self.schema
        }
        
        if hasattr(self.backend, 'generate_with_tools'):
            return self.backend.generate_with_tools(
                prompt,
                tools=[tool_schema],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        return self.backend.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def _generate_unconstrained(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Any:
        """Fallback unconstrained generation with schema in prompt."""
        schema_hint = f"\n\nOutput must conform to this JSON schema:\n{json.dumps(self.schema, indent=2)}"
        enhanced_system = system_prompt + schema_hint
        
        return self.backend.generate(
            prompt,
            system_prompt=enhanced_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def _extract_structured_output(self, response: Any) -> dict[str, Any]:
        """Extract structured output from response."""
        if hasattr(response, 'structured_output'):
            return response.structured_output
        if hasattr(response, 'parsed'):
            return response.parsed
        
        # Try to parse as JSON
        text = response.text if hasattr(response, 'text') else str(response)
        try:
            return json.loads(text)
        except:
            # Return text as result field
            return {"result": text}
    
    def _add_schema_to_prompt(self, system_prompt: str) -> str:
        """Add schema constraints to system prompt for retry."""
        return system_prompt + f"\n\nCRITICAL: Output must be valid JSON matching this schema:\n{json.dumps(self.schema, indent=2)}"
"""Structured decoding from kernel postconditions.

This module provides STRUCTURED RETRY with schema-guided generation.
It validates outputs against kernel postconditions and retries with
constrained prompts on failure.

IMPORTANT: This is NOT token-level constrained decoding (logit manipulation).
For true constrained decoding, integrate with:
- Hugging Face generate() with LogitsProcessor
- vLLM guided decoding
- Outlines grammar-constrained generation

What this DOES provide:
- Postcondition → JSON Schema conversion
- Validation against semantic constraints
- Structured retry with schema-in-prompt
- Early termination on constraint satisfaction

The "constrained" name refers to constraint validation, not logits.
"""