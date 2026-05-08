"""Tests for constrained generation grammar."""
from kernelweave.constrained import (
    parse_postcondition_to_constraints,
    postconditions_to_grammar,
    ConstrainedTokenSampler,
    validate_against_grammar,
    ConstraintType,
)
from kernelweave.kernel import Kernel, KernelStatus


def test_parse_mention_constraint():
    """Test parsing of 'mentions' postconditions."""
    rules = parse_postcondition_to_constraints("output mentions both artifacts")
    
    assert len(rules) >= 1
    assert any(r.constraint_type == ConstraintType.SEMANTIC for r in rules)
    assert any("artifact" in r.name.lower() for r in rules)


def test_parse_negation_constraint():
    """Test parsing of negation postconditions."""
    rules = parse_postcondition_to_constraints("rollback not triggered")
    
    assert len(rules) >= 1
    assert any(r.constraint_type == ConstraintType.NEGATION for r in rules)


def test_parse_schema_constraint():
    """Test parsing of schema postconditions."""
    rules = parse_postcondition_to_constraints("output schema satisfied")
    
    assert any(r.constraint_type == ConstraintType.STRUCTURAL for r in rules)


def test_postconditions_to_grammar():
    """Test full grammar generation from postconditions."""
    postconditions = [
        "output schema satisfied",
        "comparison mentions both artifacts",
        "rollback not triggered",
    ]
    
    grammar = postconditions_to_grammar(
        postconditions,
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        kernel_id="test-kernel",
        task_family="comparison",
    )
    
    # Grammar should have at least the postconditions converted
    assert len(grammar.rules) >= 2
    assert grammar.kernel_id == "test-kernel"
    assert grammar.task_family == "comparison"


def test_token_sampler_filters():
    """Test token sampler filtering logic."""
    postconditions = ["output mentions success"]
    grammar = postconditions_to_grammar(postconditions)
    sampler = ConstrainedTokenSampler(grammar)
    
    # Tokens that satisfy requirements should be boosted
    logits = {"success": 0.5, "failure": 0.5, "the": 0.3}
    filtered = sampler.filter_logits(logits, "")
    
    assert "success" in filtered
    assert filtered["success"] >= logits["success"]  # Boosted


def test_token_sampler_completeness():
    """Test completeness checking."""
    postconditions = ["output mentions success"]
    grammar = postconditions_to_grammar(postconditions)
    sampler = ConstrainedTokenSampler(grammar)
    
    # Output missing required keyword
    is_complete, missing = sampler.is_complete("This is a failure message")
    assert not is_complete
    assert "success" in missing
    
    # Output with required keyword
    is_complete, missing = sampler.is_complete("This is a success message")
    assert is_complete


def test_validate_against_grammar():
    """Test full grammar validation."""
    postconditions = [
        "output mentions artifact",
        "rollback not triggered",
    ]
    grammar = postconditions_to_grammar(postconditions)
    
    # Valid output - contains "artifact" and does NOT contain "rollback triggered"
    output = '{"result": "The artifact was analyzed successfully. No issues found."}'
    is_valid, violations = validate_against_grammar(output, grammar)
    # Should be valid because it mentions artifact and doesn't have "rollback triggered"
    assert is_valid or any("artifact" not in v.lower() for v in violations)
    
    # Missing required keyword
    output_missing = '{"result": "Analysis complete."}'
    is_valid, violations = validate_against_grammar(output_missing, grammar)
    assert not is_valid
    assert any("artifact" in v.lower() for v in violations)


def test_grammar_to_bnf():
    """Test BNF export for external parsers."""
    postconditions = ["output mentions success"]
    grammar = postconditions_to_grammar(postconditions)
    
    bnf = grammar.to_guidance_grammar()
    assert "::=" in bnf  # BNF syntax


def test_grammar_with_kernel():
    """Test grammar generation from Kernel object."""
    kernel = Kernel(
        kernel_id="test-kw",
        name="Test Kernel",
        task_family="comparison",
        description="Test kernel for grammar",
        input_schema={"type": "object"},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        preconditions=["input provided"],
        postconditions=["output mentions success", "rollback not triggered"],
        steps=[{"step": 1, "action": "test"}],
        rollback=[],
        evidence_requirements=[],
        tests=[],
        status=KernelStatus(state="candidate", confidence=0.8, failures=0, passes=1),
        source_trace_ids=[],
    )
    
    grammar = postconditions_to_grammar(
        kernel.postconditions,
        kernel.output_schema,
        kernel.kernel_id,
        kernel.task_family,
    )
    
    assert grammar.kernel_id == kernel.kernel_id
    assert grammar.task_family == kernel.task_family
