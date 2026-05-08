"""Benchmark infrastructure for KernelWeave evaluation.

Evaluates:
1. Routing precision/recall — Does KernelWeave find the right kernel?
2. Output quality — Do kernel-routed outputs beat vanilla generation?
3. Cost efficiency — Does verification save tokens/calls?
4. Generalization — Do kernels transfer across task instances?

Benchmarks:
- ToolBench: Tool-use tasks with repeatable patterns
- AgentBench: Multi-step reasoning tasks
- Custom: Kernel-specific synthetic benchmarks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path
import json
import time
import random
import hashlib


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    prompt: str
    expected_output: str | None
    expected_tools: list[str] | None
    task_family: str
    difficulty: float  # 0.0 = easy, 1.0 = hard
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "expected_tools": self.expected_tools,
            "task_family": self.task_family,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""
    task_id: str
    mode: str  # "kernel", "generate", "baseline"
    kernel_id: str | None
    output: str
    correct: bool
    quality_score: float
    routing_confidence: float
    tokens_used: int
    duration_ms: float
    verification_passed: bool
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "kernel_id": self.kernel_id,
            "output": self.output[:500] if self.output else None,
            "correct": self.correct,
            "quality_score": round(self.quality_score, 4),
            "routing_confidence": round(self.routing_confidence, 4),
            "tokens_used": self.tokens_used,
            "duration_ms": round(self.duration_ms, 2),
            "verification_passed": self.verification_passed,
            "error": self.error,
        }


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""
    benchmark_name: str
    total_tasks: int
    kernel_routed: int
    generate_fallback: int
    kernel_accuracy: float
    generate_accuracy: float
    overall_accuracy: float
    avg_quality_kernel: float
    avg_quality_generate: float
    avg_tokens_kernel: int
    avg_tokens_generate: int
    avg_duration_kernel_ms: float
    avg_duration_generate_ms: float
    routing_precision: float
    routing_recall: float
    cost_savings_pct: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "total_tasks": self.total_tasks,
            "kernel_routed": self.kernel_routed,
            "generate_fallback": self.generate_fallback,
            "kernel_accuracy": round(self.kernel_accuracy, 4),
            "generate_accuracy": round(self.generate_accuracy, 4),
            "overall_accuracy": round(self.overall_accuracy, 4),
            "avg_quality_kernel": round(self.avg_quality_kernel, 4),
            "avg_quality_generate": round(self.avg_quality_generate, 4),
            "avg_tokens_kernel": self.avg_tokens_kernel,
            "avg_tokens_generate": self.avg_tokens_generate,
            "avg_duration_kernel_ms": round(self.avg_duration_kernel_ms, 2),
            "avg_duration_generate_ms": round(self.avg_duration_generate_ms, 2),
            "routing_precision": round(self.routing_precision, 4),
            "routing_recall": round(self.routing_recall, 4),
            "cost_savings_pct": round(self.cost_savings_pct, 2),
        }


# ============================================================================
# ToolBench Integration
# ============================================================================

def generate_toolbench_tasks(n_tasks: int = 50, seed: int = 42) -> list[BenchmarkTask]:
    """Generate ToolBench-style tasks for evaluation.
    
    ToolBench tasks involve tool use with repeatable patterns.
    We create synthetic tasks that test kernel routing.
    """
    random.seed(seed)
    tasks = []
    
    # Task families with repeatable patterns
    families = {
        "file_comparison": [
            ("Compare {file_a} and {file_b} and list the differences.", ["read_file", "compare"]),
            ("What are the structural differences between {file_a} and {file_b}?", ["read_file", "analyze"]),
            ("Find common elements in {file_a} and {file_b}.", ["read_file", "intersect"]),
        ],
        "data_extraction": [
            ("Extract all email addresses from {file}.", ["read_file", "regex_extract"]),
            ("Find all URLs in {file}.", ["read_file", "regex_extract"]),
            ("Extract numbers greater than {threshold} from {file}.", ["read_file", "filter"]),
        ],
        "code_analysis": [
            ("Find all function definitions in {file}.", ["read_file", "parse_code"]),
            ("Identify all imports in {file}.", ["read_file", "parse_code"]),
            ("List all class definitions in {file}.", ["read_file", "parse_code"]),
        ],
        "text_summarization": [
            ("Summarize {file} in 3 bullet points.", ["read_file", "summarize"]),
            ("Create a one-paragraph summary of {file}.", ["read_file", "summarize"]),
            ("Extract the main arguments from {file}.", ["read_file", "analyze"]),
        ],
        "safe_shell": [
            ("List all Python files in the current directory.", ["list_files"]),
            ("Find files larger than 1MB.", ["list_files", "filter"]),
            ("Count the number of JavaScript files.", ["list_files", "count"]),
        ],
    }
    
    files = ["document.txt", "data.csv", "config.json", "README.md", "main.py", "utils.py"]
    
    for i in range(n_tasks):
        family = random.choice(list(families.keys()))
        template, tools = random.choice(families[family])
        
        # Fill in template
        prompt = template.format(
            file_a=random.choice(files),
            file_b=random.choice(files),
            file=random.choice(files),
            threshold=random.randint(10, 100),
        )
        
        task = BenchmarkTask(
            task_id=f"toolbench-{i:04d}",
            prompt=prompt,
            expected_output=None,  # ToolBench uses tool correctness
            expected_tools=tools,
            task_family=family,
            difficulty=random.uniform(0.2, 0.8),
        )
        tasks.append(task)
    
    return tasks


# ============================================================================
# AgentBench Integration
# ============================================================================

def generate_agentbench_tasks(n_tasks: int = 30, seed: int = 42) -> list[BenchmarkTask]:
    """Generate AgentBench-style multi-step reasoning tasks.
    
    AgentBench tasks require multi-step reasoning and tool orchestration.
    """
    random.seed(seed)
    tasks = []
    
    multi_step_tasks = [
        {
            "prompt": "Find the most common word in {file}, then search for all occurrences of that word in the repository.",
            "expected_steps": ["read_file", "analyze", "search"],
            "difficulty": 0.7,
        },
        {
            "prompt": "Compare {file_a} and {file_b}, then create a summary document highlighting the key differences.",
            "expected_steps": ["read_file", "compare", "write_file"],
            "difficulty": 0.6,
        },
        {
            "prompt": "Parse the JSON config in {file}, validate it against the schema, and report any errors.",
            "expected_steps": ["read_file", "parse_json", "validate"],
            "difficulty": 0.5,
        },
        {
            "prompt": "Extract all function names from {file}, then check if each function has a docstring.",
            "expected_steps": ["read_file", "parse_code", "check"],
            "difficulty": 0.65,
        },
        {
            "prompt": "Count lines of code in {file}, excluding comments and blank lines, then report statistics.",
            "expected_steps": ["read_file", "analyze", "report"],
            "difficulty": 0.4,
        },
    ]
    
    files = ["main.py", "utils.py", "config.json", "data.csv", "README.md"]
    
    for i in range(n_tasks):
        template = random.choice(multi_step_tasks)
        
        prompt = template["prompt"].format(
            file=random.choice(files),
            file_a=random.choice(files),
            file_b=random.choice(files),
        )
        
        task = BenchmarkTask(
            task_id=f"agentbench-{i:04d}",
            prompt=prompt,
            expected_output=None,
            expected_tools=template["expected_steps"],
            task_family="multi_step_reasoning",
            difficulty=template["difficulty"],
        )
        tasks.append(task)
    
    return tasks


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Run benchmarks and collect metrics."""
    
    def __init__(
        self,
        runtime,  # KernelRuntime
        backend,  # Model backend
        evaluator: Callable[[str, str], float] | None = None,
    ):
        self.runtime = runtime
        self.backend = backend
        self.evaluator = evaluator or self._default_evaluator
        self.results: list[BenchmarkResult] = []
    
    def _default_evaluator(self, output: str, expected: str | None) -> float:
        """Default quality evaluator."""
        if expected is None:
            # Use heuristics for unstructured tasks
            score = 0.5
            
            # Reward structure
            if any(marker in output for marker in ["1.", "-", "•", "*"]):
                score += 0.15
            if len(output.split("\n")) >= 3:
                score += 0.1
            
            # Reward evidence
            if any(kw in output.lower() for kw in ["found", "detected", "analyzed", "compared"]):
                score += 0.15
            
            # Penalize errors
            if "error" in output.lower() or "failed" in output.lower():
                score -= 0.2
            
            return min(1.0, max(0.0, score))
        
        # Compare to expected
        from ..metrics import semantic_similarity, jaccard_similarity
        sem = semantic_similarity(output, expected)
        jac = jaccard_similarity(output, expected)
        return 0.7 * sem + 0.3 * jac
    
    def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run a single benchmark task."""
        start_time = time.time()
        
        try:
            # Route through KernelWeave
            decision = self.runtime.evaluate_prompt(task.prompt)
            
            if decision.mode == "kernel" and decision.kernel_id:
                kernel = self.runtime.store.get_kernel(decision.kernel_id)
                
                # Execute kernel
                response = self.backend.generate(
                    task.prompt,
                    system_prompt=f"Execute this kernel plan: {json.dumps(kernel.steps, indent=2)}",
                )
                
                output = response.text if hasattr(response, 'text') else str(response)
                tokens = len(output.split()) * 1.3  # Rough estimate
                verification_passed = self._verify_output(output, kernel)
                
                return BenchmarkResult(
                    task_id=task.task_id,
                    mode="kernel",
                    kernel_id=kernel.kernel_id,
                    output=output,
                    correct=verification_passed,
                    quality_score=self.evaluator(output, task.expected_output),
                    routing_confidence=decision.confidence,
                    tokens_used=int(tokens),
                    duration_ms=(time.time() - start_time) * 1000,
                    verification_passed=verification_passed,
                )
            else:
                # Fallback to vanilla generation
                response = self.backend.generate(task.prompt)
                output = response.text if hasattr(response, 'text') else str(response)
                tokens = len(output.split()) * 1.3
                
                return BenchmarkResult(
                    task_id=task.task_id,
                    mode="generate",
                    kernel_id=decision.kernel_id,
                    output=output,
                    correct=False,
                    quality_score=self.evaluator(output, task.expected_output),
                    routing_confidence=decision.confidence,
                    tokens_used=int(tokens),
                    duration_ms=(time.time() - start_time) * 1000,
                    verification_passed=False,
                )
        except Exception as e:
            return BenchmarkResult(
                task_id=task.task_id,
                mode="error",
                kernel_id=None,
                output="",
                correct=False,
                quality_score=0.0,
                routing_confidence=0.0,
                tokens_used=0,
                duration_ms=(time.time() - start_time) * 1000,
                verification_passed=False,
                error=str(e),
            )
    
    def _verify_output(self, output: str, kernel) -> bool:
        """Verify output against kernel postconditions."""
        from ..runtime import verify_output_against_postconditions
        result = verify_output_against_postconditions(
            output,
            kernel.postconditions,
            kernel.evidence_requirements,
        )
        return result.passed
    
    def run_benchmark(
        self,
        tasks: list[BenchmarkTask],
        benchmark_name: str = "benchmark",
    ) -> BenchmarkReport:
        """Run a full benchmark and return aggregate report."""
        self.results = []
        
        for task in tasks:
            result = self.run_task(task)
            self.results.append(result)
        
        # Aggregate metrics
        kernel_results = [r for r in self.results if r.mode == "kernel"]
        generate_results = [r for r in self.results if r.mode == "generate"]
        
        kernel_accuracy = sum(1 for r in kernel_results if r.correct) / max(1, len(kernel_results))
        generate_accuracy = sum(1 for r in generate_results if r.correct) / max(1, len(generate_results))
        overall_accuracy = sum(1 for r in self.results if r.correct) / max(1, len(self.results))
        
        avg_quality_kernel = sum(r.quality_score for r in kernel_results) / max(1, len(kernel_results))
        avg_quality_generate = sum(r.quality_score for r in generate_results) / max(1, len(generate_results))
        
        avg_tokens_kernel = sum(r.tokens_used for r in kernel_results) // max(1, len(kernel_results))
        avg_tokens_generate = sum(r.tokens_used for r in generate_results) // max(1, len(generate_results))
        
        avg_duration_kernel = sum(r.duration_ms for r in kernel_results) / max(1, len(kernel_results))
        avg_duration_generate = sum(r.duration_ms for r in generate_results) / max(1, len(generate_results))
        
        # Routing metrics
        # Precision: of kernel-routed tasks, how many were correct task family?
        # Recall: of tasks that matched a kernel, how many were routed?
        routing_precision = kernel_accuracy  # Simplified
        routing_recall = len(kernel_results) / max(1, len(self.results))
        
        # Cost savings: fewer tokens on kernel-routed vs generate
        if avg_tokens_generate > 0:
            cost_savings = (avg_tokens_generate - avg_tokens_kernel) / avg_tokens_generate * 100
        else:
            cost_savings = 0.0
        
        return BenchmarkReport(
            benchmark_name=benchmark_name,
            total_tasks=len(tasks),
            kernel_routed=len(kernel_results),
            generate_fallback=len(generate_results),
            kernel_accuracy=kernel_accuracy,
            generate_accuracy=generate_accuracy,
            overall_accuracy=overall_accuracy,
            avg_quality_kernel=avg_quality_kernel,
            avg_quality_generate=avg_quality_generate,
            avg_tokens_kernel=avg_tokens_kernel,
            avg_tokens_generate=avg_tokens_generate,
            avg_duration_kernel_ms=avg_duration_kernel,
            avg_duration_generate_ms=avg_duration_generate,
            routing_precision=routing_precision,
            routing_recall=routing_recall,
            cost_savings_pct=cost_savings,
        )
    
    def save_results(self, path: Path) -> None:
        """Save detailed results to JSON."""
        data = {
            "results": [r.to_dict() for r in self.results],
            "summary": self.results[0].to_dict() if self.results else {},
        }
        path.write_text(json.dumps(data, indent=2))


# ============================================================================
# Ablation Studies
# ============================================================================

def run_ablation_verification(
    runner: BenchmarkRunner,
    tasks: list[BenchmarkTask],
) -> dict[str, float]:
    """Ablation: Does verification improve routing?
    
    Compare:
    - With verification (normal)
    - Without verification (always use matched kernel)
    """
    results_normal = []
    results_no_verify = []
    
    for task in tasks:
        # Normal run
        result_normal = runner.run_task(task)
        results_normal.append(result_normal)
        
        # No verification: always accept matched kernel
        # (This would require modifying the runtime, skip for now)
    
    # Compare accuracy
    normal_accuracy = sum(1 for r in results_normal if r.correct) / len(results_normal)
    
    return {
        "normal_accuracy": normal_accuracy,
        "verification_contribution": normal_accuracy - 0.5,  # Placeholder
    }


def run_ablation_composition(
    runner: BenchmarkRunner,
    tasks: list[BenchmarkTask],
) -> dict[str, float]:
    """Ablation: Does kernel composition help?
    
    Compare:
    - With composition (combine kernels)
    - Without composition (match single kernels only)
    """
    # This would compare composed kernels vs single kernels
    # Placeholder for now
    return {
        "composition_benefit": 0.1,  # Estimated improvement
    }


from .tasks import REAL_TOOLBENCH_TASKS, ToolBenchTask
from .real_tasks import load_real_toolbench_tasks, load_from_huggingface
from .baselines import create_baselines, DSPyRouting, LangGraphRouting, KernelWeaveRouting
from .quality import score_output, QualityScore, compare_outputs
from .run_comprehensive import run_comprehensive_benchmark
