"""Structured decoding from kernel postconditions.

The kernel's postconditions become a JSON Schema that constrains generation.
This is the difference between "I hope the model follows the plan" and 
"the model physically cannot output invalid states."
"""
from .schema import postconditions_to_schema, validate_output
from .decoder import ConstrainedDecoder, ConstrainedResponse
from .grammar import (
    ConstraintType,
    GrammarRule,
    ConstrainedGrammar,
    parse_postcondition_to_constraints,
    postconditions_to_grammar,
    ConstrainedTokenSampler,
    validate_against_grammar,
)
from .token_level import (
    FiniteStateConstraint,
    ConstrainedGenerator,
    LogitsProcessorConstraint,
    create_constrained_pipeline,
    OUTLINES_AVAILABLE,
)

__all__ = [
    # Schema
    "postconditions_to_schema",
    "validate_output",
    # Decoder
    "ConstrainedDecoder",
    "ConstrainedResponse",
    # Grammar
    "ConstraintType",
    "GrammarRule",
    "ConstrainedGrammar",
    "parse_postcondition_to_constraints",
    "postconditions_to_grammar",
    "ConstrainedTokenSampler",
    "validate_against_grammar",
    # Token-level constrained generation
    "FiniteStateConstraint",
    "ConstrainedGenerator",
    "LogitsProcessorConstraint",
    "create_constrained_pipeline",
    "OUTLINES_AVAILABLE",
]
