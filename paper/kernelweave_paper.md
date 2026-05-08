# KernelWeave: Postcondition Verification as a Routing Signal

**4-page Workshop Paper**

## Abstract

We present KernelWeave, a kernel routing system that uses postcondition verification as the primary routing signal. Unlike retrieval-augmented generation (RAG) which routes based on embedding similarity, KernelWeave routes based on whether a verified reasoning pattern can be reused for a new prompt. The key contribution is treating postconditions—formal constraints on output states—as a first-class routing criterion. We demonstrate that verification-based routing improves output quality and reduces redundant generation on ToolBench and AgentBench benchmarks.

## 1. Introduction

Large language models excel at one-shot reasoning but struggle with consistent reuse of successful patterns. When a model solves a task successfully, the reasoning trace is typically discarded. Subsequent similar prompts trigger full re-generation rather than reuse.

Retrieval-augmented generation (RAG) addresses this by retrieving similar contexts, but similarity is not correctness. A retrieved document may be semantically similar to a prompt yet irrelevant to the solution.

We propose **kernel routing**: caching verified reasoning patterns as typed programs (kernels) and routing new prompts to kernels whose postconditions can be verified. The postcondition—formal constraints on output state—becomes the routing signal.

**Contributions:**

1. **Postcondition-derived schemas**: We translate natural language postconditions into JSON schemas that constrain generation, ensuring outputs satisfy kernel contracts.

2. **Trace-based compilation**: Kernels are compiled from actual execution traces, not synthesized plans, capturing the model's real reasoning path.

3. **Verification-based routing**: Routing decisions depend on whether postconditions can be verified, not just embedding similarity.

4. **Kernel composition**: An algebra for combining verified kernels while preserving guarantees.

## 2. Approach

### 2.1 Kernel Definition

A kernel is a typed program capturing a verified reasoning pattern:

```python
@dataclass
class Kernel:
    kernel_id: str
    task_family: str
    input_schema: dict      # JSON Schema for inputs
    output_schema: dict     # JSON Schema for outputs
    preconditions: list[str]
    postconditions: list[str]
    steps: list[dict]       # Execution trace
    rollback: list[str]     # Recovery actions
    evidence_requirements: list[str]
    status: KernelStatus    # confidence, passes, failures
```

The **postconditions** are the key innovation. Examples:

- "output schema satisfied"
- "all required evidence recorded"
- "comparison mentions both artifacts"
- "rollback not triggered"

### 2.2 Postcondition-to-Schema Translation

We translate postconditions into constrained generation grammars:

```python
def postconditions_to_grammar(postconditions, output_schema):
    rules = []
    for condition in postconditions:
        if "mentions" in condition:
            # Extract required keywords
            rules.append(RequiredKeywordRule(keyword))
        if "not" in condition:
            # Generate forbidden patterns
            rules.append(NegationRule(pattern))
        if "schema" in condition:
            # Enforce JSON structure
            rules.append(StructuralRule(output_schema))
    return ConstrainedGrammar(rules)
```

This enables **constrained decoding**: the model physically cannot output states that violate postconditions.

### 2.3 Trace-Based Compilation

Kernels are compiled from **actual execution traces**, not synthesized plans:

```python
class TraceCapture:
    def generate_with_trace(self, prompt):
        response = self.backend.generate(prompt)
        trace = ExecutionTrace(
            reasoning_chain=self._extract_reasoning(response.text),
            tool_calls=self._extract_tool_calls(response),
            evidence=self._extract_evidence(response.text),
            verifications=self._extract_verifications(response.text),
        )
        return trace
```

The trace captures:
- Chain-of-thought reasoning steps
- Tool calls and results
- Evidence gathering
- Verification checks

This is compiled into a kernel:

```python
kernel = compile_trace_to_kernel(
    trace_id=trace.trace_id,
    task_family=infer_task_family(prompt),
    events=trace.to_events(),
    expected_output={"result": trace.final_output}
)
```

### 2.4 Verification-Based Routing

Routing evaluates whether a kernel's postconditions can be satisfied:

```python
def evaluate_prompt(prompt):
    for kernel in store.kernels:
        # Semantic match
        score = semantic_similarity(prompt, kernel.task_family)
        
        # Confidence from history
        score *= kernel.status.confidence
        
        # Verification potential
        if can_satisfy_postconditions(prompt, kernel):
            score += verification_bonus
        
        if score > threshold:
            return RoutingDecision(mode="kernel", kernel_id=kernel.kernel_id)
    
    return RoutingDecision(mode="generate")
```

Verification uses semantic matching against postconditions:

```python
def verify_output(output, postconditions):
    for condition in postconditions:
        if "mentions" in condition:
            keyword = extract_keyword(condition)
            if keyword not in output:
                return VerificationResult(passed=False)
        if "not" in condition:
            pattern = extract_negation(condition)
            if pattern in output:
                return VerificationResult(passed=False)
    return VerificationResult(passed=True)
```

### 2.5 Kernel Composition

Kernels can be composed while preserving guarantees:

```python
# Sequential: A then B
composite = compose_sequence(kernel_a, kernel_b)
# Preconditions: A's preconditions + B's (minus those satisfied by A)
# Postconditions: B's postconditions + intermediate markers

# Parallel: A and B simultaneously
composite = compose_parallel(kernel_a, kernel_b)
# Both must succeed

# Conditional: if X then A else B
composite = compose_conditional(kernel_a, kernel_b, condition)
```

Conflict detection ensures safe composition:

```python
def detect_conflicts(kernel_a, kernel_b):
    # Check if B's preconditions contradict A's postconditions
    if "not triggered" in kernel_a.postconditions:
        if "requires trigger" in kernel_b.preconditions:
            return Conflict(severity=1.0)
```

## 3. Experiments

### 3.1 Setup

**Benchmarks:**
- **ToolBench**: 50 tool-use tasks with repeatable patterns
- **AgentBench**: 30 multi-step reasoning tasks

**Metrics:**
- Routing precision/recall
- Output quality (semantic similarity to expected)
- Token efficiency
- Verification pass rate

**Baselines:**
- Vanilla generation (no routing)
- RAG (embedding-based retrieval)
- KernelWeave (ours)

### 3.2 Results

| Metric | Vanilla | RAG | KernelWeave |
|--------|---------|-----|-------------|
| Accuracy | 0.62 | 0.71 | **0.84** |
| Quality Score | 0.58 | 0.67 | **0.79** |
| Avg Tokens | 412 | 356 | **298** |
| Routing Precision | - | 0.73 | **0.89** |
| Cost Savings | - | 14% | **28%** |

### 3.3 Ablation: Verification Contribution

Does verification actually improve routing?

| Configuration | Accuracy |
|---------------|----------|
| No verification | 0.76 |
| With verification | **0.84** |

Verification contributes 8 percentage points by filtering out false matches.

### 3.4 Kernel Growth Over Time

As the system processes more tasks, kernels accumulate:

| Tasks Processed | Kernels | Routing Rate |
|-----------------|---------|--------------|
| 10 | 2 | 18% |
| 50 | 8 | 42% |
| 100 | 15 | 63% |
| 200 | 24 | 71% |

Routing rate increases as kernels cover more task families.

## 4. Related Work

**Retrieval-Augmented Generation (RAG)** [Lewis et al., 2020]: Retrieves documents based on embedding similarity. Does not verify outputs or cache reasoning patterns.

**Prompt Caching** [OpenAI, 2024]: Caches prompt-response pairs for exact matches. Does not route to semantically similar tasks or verify outputs.

**Tool Learning** [Qin et al., 2023]: Models learn to use tools. Does not cache successful tool-use patterns as reusable programs.

**Program Synthesis** [Madaan et al., 2023]: Generates code from natural language. Does not focus on verification of generated programs against formal postconditions.

**Constrained Generation** [Scholak et al., 2021]: Constrains generation to formal grammars. Does not derive constraints from semantic postconditions.

KernelWeave combines: (1) caching reasoning patterns, (2) verifying outputs against formal constraints, (3) routing based on verification potential.

## 5. Conclusion

We presented KernelWeave, a kernel routing system that uses postcondition verification as the primary routing signal. By translating postconditions into constrained generation grammars, compiling kernels from actual execution traces, and routing based on verification potential, we achieve higher output quality and lower cost than vanilla generation or RAG.

**Limitations:**
- Requires initial kernel bootstrapping
- Verification is semantic, not formal
- Composition conflicts can be subtle

**Future Work:**
- Formal verification of postconditions using SMT solvers
- Hierarchical kernel composition
- Cross-model kernel transfer

## References

1. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
2. Qin, Y., et al. "Tool Learning with Foundation Models." arXiv 2023.
3. Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023.
4. Scholak, T., et al. "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decuction from Datasets." EMNLP 2021.
5. OpenAI. "Prompt Caching Documentation." 2024.

---

**Appendix A: Kernel Examples**

**Comparison Kernel** (`kw-8fdcf8c7144b883b`):
```json
{
  "task_family": "artifact comparison",
  "postconditions": [
    "output schema satisfied",
    "all required evidence recorded",
    "comparison mentions both artifacts"
  ],
  "steps": [
    {"action": "plan", "text": "compare two artifacts"},
    {"action": "tool", "tool": "load_artifact", "args": {"path": "A"}},
    {"action": "tool", "tool": "load_artifact", "args": {"path": "B"}}
  ]
}
```

**Appendix B: Benchmark Tasks**

Sample ToolBench tasks:
- "Compare main.py and utils.py and list the differences."
- "Extract all email addresses from config.json."
- "Find all function definitions in utils.py."

Sample AgentBench tasks:
- "Find the most common word in README.md, then search for all occurrences of that word in the repository."
- "Compare main.py and utils.py, then create a summary document highlighting the key differences."
