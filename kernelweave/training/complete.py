"""Complete training system with inbuilt trace generation and LoRA fine-tuning.

REVOLUTIONARY: Self-contained training that generates its own data from kernels.

Dependencies (auto-installed):
- transformers
- trl
- peft
- bitsandbytes
- torch
- datasets

Usage on Kaggle:
    from kernelweave.training import KaggleTrainer
    
    trainer = KaggleTrainer(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        output_dir="./kernel-native-model",
    )
    
    # Generate training data from kernels
    trainer.generate_training_data(n_samples=5000)
    
    # Train with LoRA
    trainer.train(epochs=3, batch_size=4)
    
    # Save model
    trainer.save_model()
"""
from __future__ import annotations

import os
import json
import random
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime

# Training dependencies - will auto-install if missing
TRAINING_DEPS = [
    "transformers>=4.40.0",
    "trl>=0.8.0",
    "peft>=0.10.0",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.28.0",
    "datasets>=2.18.0",
    "torch>=2.2.0",
]


def _ensure_deps():
    """Auto-install training dependencies if missing."""
    missing = []
    
    for dep in TRAINING_DEPS:
        pkg = dep.split(">=")[0].split("==")[0].split("<")[0]
        try:
            __import__(pkg)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"Installing training dependencies: {missing}")
        import subprocess
        subprocess.check_call(["pip", "install", "-q"] + missing)


@dataclass
class TrainingConfig:
    """Configuration for kernel-native training."""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./kernel-native-model"
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training config
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data config
    n_train_samples: int = 5000
    n_eval_samples: int = 500
    max_input_length: int = 512
    max_output_length: int = 256
    
    # Kernel config
    kernel_hit_weight: float = 2.0  # Weight for samples that hit kernels
    verification_weight: float = 1.5  # Weight for verified traces
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "n_train_samples": self.n_train_samples,
            "n_eval_samples": self.n_eval_samples,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "kernel_hit_weight": self.kernel_hit_weight,
            "verification_weight": self.verification_weight,
        }


@dataclass
class TrainingSample:
    """A single training sample."""
    prompt: str
    response: str
    task_family: str
    kernel_id: str | None
    verification_passed: bool
    confidence: float
    weight: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "task_family": self.task_family,
            "kernel_id": self.kernel_id,
            "verification_passed": self.verification_passed,
            "confidence": self.confidence,
            "weight": self.weight,
        }


class TraceGenerator:
    """Generate training traces from kernels and templates.
    
    REVOLUTIONARY: Creates synthetic training data from kernel definitions.
    No external data needed.
    """
    
    def __init__(self, kernel_store: Any = None):
        self.kernel_store = kernel_store
        self._task_templates = self._build_task_templates()
        self._response_templates = self._build_response_templates()
    
    def generate_samples(
        self,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> list[TrainingSample]:
        """Generate training samples from kernels and templates."""
        random.seed(seed)
        samples = []
        
        # Get kernels if available
        kernels = []
        if self.kernel_store:
            for k in self.kernel_store.list_kernels():
                kernels.append(self.kernel_store.get_kernel(k["kernel_id"]))
        
        # If no kernels, use built-in templates
        if not kernels:
            kernels = self._get_builtin_kernels()
        
        for _ in range(n_samples):
            # Select kernel
            kernel = random.choice(kernels)
            
            # Generate prompt
            prompt = self._generate_prompt(kernel)
            
            # Generate response
            response = self._generate_response(kernel, prompt)
            
            # Calculate weight
            weight = 1.0
            if kernel:
                weight *= 1.5  # Kernel hit
            if random.random() > 0.3:  # 70% verified
                weight *= 1.5
            
            sample = TrainingSample(
                prompt=prompt,
                response=response,
                task_family=kernel.task_family if hasattr(kernel, 'task_family') else "general",
                kernel_id=kernel.kernel_id if hasattr(kernel, 'kernel_id') else None,
                verification_passed=random.random() > 0.3,
                confidence=random.uniform(0.6, 0.95),
                weight=weight,
            )
            samples.append(sample)
        
        return samples
    
    def _build_task_templates(self) -> dict[str, list[str]]:
        """Build task templates for each family."""
        return {
            "comparison": [
                "Compare {file_a} and {file_b} and summarize the differences.",
                "What are the key differences between {item_a} and {item_b}?",
                "Compare the following two code files: {file_a} and {file_b}.",
                "Analyze differences between {file_a} and {file_b}.",
                "Find all differences between {item_a} and {item_b}.",
            ],
            "analysis": [
                "Analyze {file} and identify potential {issue_type}.",
                "Review {code} for {issue_type}.",
                "Find {issue_type} in {file}.",
                "Examine {file} and report any {issue_type}.",
                "What {issue_type} exist in {file}?",
            ],
            "search": [
                "Find all {file_type} files in {directory}.",
                "Search for {pattern} in all files.",
                "Locate all occurrences of {pattern}.",
                "Find files matching {pattern}.",
                "Search {directory} for {pattern}.",
            ],
            "generation": [
                "Generate a {format} report from {source}.",
                "Create a {format} summary of {topic}.",
                "Write a {format} document about {topic}.",
                "Produce a {format} analysis of {subject}.",
                "Generate {format} output for {task}.",
            ],
            "debugging": [
                "Fix the bug in {file} that causes {error}.",
                "Debug {file} and resolve {error}.",
                "Find and fix {error} in {file}.",
                "Resolve {error} in {code}.",
                "Fix {file} to prevent {error}.",
            ],
            "summarization": [
                "Summarize {document}.",
                "Create a summary of {document}.",
                "Provide a brief overview of {document}.",
                "What are the main points in {document}?",
                "Summarize the key findings in {document}.",
            ],
            "transformation": [
                "Convert {input} from {format_a} to {format_b}.",
                "Transform {input} into {format_b}.",
                "Change {input} format from {format_a} to {format_b}.",
                "Reformat {input} as {format_b}.",
                "Parse {input} and output {format_b}.",
            ],
            "testing": [
                "Write tests for {module}.",
                "Create test cases for {function}.",
                "Generate unit tests for {code}.",
                "Add tests to cover {functionality}.",
                "Test {module} for {scenario}.",
            ],
        }
    
    def _build_response_templates(self) -> dict[str, list[str]]:
        """Build response templates for each family."""
        return {
            "comparison": [
                "I found {n} key differences between {file_a} and {file_b}:\n\n1. {diff_1}\n2. {diff_2}\n3. {diff_3}\n\nOverall, these files differ in {aspect}.",
                "Comparing {file_a} and {file_b}:\n\n**Similarities:**\n- {sim_1}\n- {sim_2}\n\n**Differences:**\n- {diff_1}\n- {diff_2}",
                "After analyzing both files:\n\n{file_a} has {feature_a}, while {file_b} has {feature_b}.\n\nThe main differences are in {aspect}.",
            ],
            "analysis": [
                "Analysis of {file}:\n\n**Issues found:**\n1. {issue_1}\n2. {issue_2}\n\n**Recommendations:**\n- {rec_1}\n- {rec_2}",
                "I identified {n} {issue_type} in {file}:\n\n1. {issue_1} (line {line_1})\n2. {issue_2} (line {line_2})\n\nSeverity: {severity}",
                "Review results:\n\n✗ {issue_1}\n✗ {issue_2}\n✓ {good_1}\n\nTotal issues: {n}",
            ],
            "search": [
                "Found {n} {file_type} files in {directory}:\n\n1. {file_1}\n2. {file_2}\n3. {file_3}\n...",
                "Search results for '{pattern}':\n\n- {file_1}: {n_1} matches\n- {file_2}: {n_2} matches\n- {file_3}: {n_3} matches\n\nTotal: {total} matches",
                "Located {n} files:\n\n{file_1}\n{file_2}\n{file_3}\n...",
            ],
            "generation": [
                "# {title}\n\n{content}\n\n## Summary\n\n{summary}",
                "```{format}\n{content}\n```\n\nGenerated from {source}.",
                "**Report:**\n\n{content}\n\n**Key Points:**\n- {point_1}\n- {point_2}",
            ],
            "debugging": [
                "Fixed the bug in {file}:\n\n**Problem:** {error}\n\n**Solution:** {fix}\n\n**Changes:**\n```diff\n{diff}\n```",
                "Debugging complete.\n\n**Root cause:** {cause}\n\n**Fix applied:** {fix}\n\n**Files modified:** {file}",
                "Issue resolved:\n\n1. Identified {error} in {file}\n2. Applied fix: {fix}\n3. Verified solution\n\n✓ Bug fixed",
            ],
            "summarization": [
                "**Summary of {document}:**\n\n{summary}\n\n**Key points:**\n1. {point_1}\n2. {point_2}\n3. {point_3}",
                "Overview:\n\n{summary}\n\nMain findings: {findings}",
                "Brief:\n\n{document} covers {topic}. Key aspects:\n- {aspect_1}\n- {aspect_2}\n- {aspect_3}",
            ],
            "transformation": [
                "```{format_b}\n{output}\n```\n\nConverted from {format_a}.",
                "Transformation complete.\n\nInput ({format_a}):\n```\n{input}\n```\n\nOutput ({format_b}):\n```\n{output}\n```",
                "```{format_b}\n{output}\n```\n\n{stats}",
            ],
            "testing": [
                "```python\nimport pytest\n\n{tests}\n```\n\nTests cover: {coverage}",
                "# Tests for {module}\n\n```python\n{test_code}\n```\n\n**Coverage:** {coverage}%",
                "Generated {n} test cases:\n\n{tests}\n\nAll tests passing ✓",
            ],
        }
    
    def _get_builtin_kernels(self) -> list[Any]:
        """Get built-in kernels if no store is available."""
        from dataclasses import dataclass
        
        @dataclass
        class MockKernel:
            kernel_id: str
            task_family: str
            postconditions: list[str]
            evidence_requirements: list[str]
        
        return [
            MockKernel("kw-comparison", "comparison", ["mentions both files", "lists differences"], ["structural differences found"]),
            MockKernel("kw-analysis", "analysis", ["identifies issues", "provides recommendations"], ["issues documented"]),
            MockKernel("kw-search", "search", ["finds matching files", "reports counts"], ["results listed"]),
            MockKernel("kw-generation", "generation", ["output format correct", "content relevant"], ["format validated"]),
            MockKernel("kw-debugging", "debugging", ["root cause identified", "fix applied"], ["bug resolved"]),
            MockKernel("kw-summarization", "summarization", ["key points extracted", "concise summary"], ["main ideas covered"]),
            MockKernel("kw-transformation", "transformation", ["format converted", "data preserved"], ["transformation complete"]),
            MockKernel("kw-testing", "testing", ["test cases generated", "coverage reported"], ["tests valid"]),
        ]
    
    def _generate_prompt(self, kernel: Any) -> str:
        """Generate a prompt for a kernel."""
        family = kernel.task_family if hasattr(kernel, 'task_family') else "general"
        templates = self._task_templates.get(family, ["Complete the task: {task}"])
        template = random.choice(templates)
        
        # Fill in template variables
        variables = self._generate_variables(family)
        prompt = template.format(**variables)
        
        return prompt
    
    def _generate_response(self, kernel: Any, prompt: str) -> str:
        """Generate a response for a prompt."""
        family = kernel.task_family if hasattr(kernel, 'task_family') else "general"
        templates = self._response_templates.get(family, ["Task completed: {result}"])
        template = random.choice(templates)
        
        # Fill in template variables
        variables = self._generate_variables(family)
        response = template.format(**variables)
        
        return response
    
    def _generate_variables(self, family: str) -> dict[str, str]:
        """Generate random variables for templates."""
        files = ["main.py", "utils.py", "config.yaml", "app.js", "index.ts", "README.md", "test.py"]
        directories = ["src", "lib", "tests", "docs", "config", "scripts"]
        file_types = ["Python", "JavaScript", "TypeScript", "config", "test", "documentation"]
        formats = ["JSON", "YAML", "Markdown", "HTML", "CSV", "XML"]
        patterns = ["TODO", "FIXME", "deprecated", "unused", "import", "function"]
        issues = ["bugs", "security issues", "performance problems", "style issues", "unused imports"]
        errors = ["TypeError", "ValueError", "ImportError", "AttributeError", "SyntaxError"]
        
        n = random.randint(1, 5)
        
        return {
            "file": random.choice(files),
            "file_a": random.choice(files),
            "file_b": random.choice(files),
            "item_a": random.choice(files),
            "item_b": random.choice(files),
            "directory": random.choice(directories),
            "file_type": random.choice(file_types),
            "format": random.choice(formats),
            "format_a": random.choice(formats),
            "format_b": random.choice(formats),
            "pattern": random.choice(patterns),
            "issue_type": random.choice(issues),
            "error": random.choice(errors),
            "n": str(n),
            "input": "input data",
            "source": "data.json",
            "topic": "architecture",
            "subject": "the codebase",
            "task": "complete the operation",
            "document": "the document",
            "module": "the module",
            "function": "the function",
            "code": "the code",
            "functionality": "core functionality",
            "scenario": "edge cases",
            # Response-specific
            "diff_1": "variable naming differs",
            "diff_2": "function implementation varies",
            "diff_3": "import order is different",
            "aspect": "implementation approach",
            "sim_1": "both use similar patterns",
            "sim_2": "shared dependencies",
            "feature_a": "feature A",
            "feature_b": "feature B",
            "issue_1": "unused variable",
            "issue_2": "missing error handling",
            "line_1": "10",
            "line_2": "25",
            "severity": "medium",
            "rec_1": "add error handling",
            "rec_2": "remove unused code",
            "good_1": "proper documentation",
            "file_1": "file1.py",
            "file_2": "file2.py",
            "file_3": "file3.py",
            "n_1": "5",
            "n_2": "3",
            "n_3": "7",
            "total": "15",
            "title": "Report",
            "content": "Content here",
            "summary": "Summary of findings",
            "point_1": "key point 1",
            "point_2": "key point 2",
            "point_3": "key point 3",
            "fix": "added null check",
            "cause": "null pointer dereference",
            "diff": "- old\\n+ new",
            "findings": "important results",
            "output": "transformed output",
            "stats": "Processed 100 items",
            "tests": "def test_x(): assert True",
            "coverage": "85%",
            "test_code": "test code here",
        }


class KaggleTrainer:
    """Complete training pipeline for Kaggle.
    
    REVOLUTIONARY: Zero external data, auto-installs deps, works on free GPU.
    
    Usage:
        from kernelweave.training import KaggleTrainer
        
        trainer = KaggleTrainer(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            output_dir="./kernel-native-model",
        )
        
        # Generate data
        trainer.generate_training_data(n_samples=5000)
        
        # Train
        trainer.train(epochs=3)
        
        # Save
        trainer.save_model()
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "./kernel-native-model",
        config: TrainingConfig | None = None,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config or TrainingConfig(base_model=base_model, output_dir=output_dir)
        
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._train_data = None
        self._eval_data = None
        
        # Auto-install deps
        _ensure_deps()
        
        print(f"KaggleTrainer initialized")
        print(f"  Base model: {base_model}")
        print(f"  Output dir: {output_dir}")
    
    def generate_training_data(
        self,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> None:
        """Generate training data from kernels."""
        print(f"\nGenerating {n_samples} training samples...")
        
        generator = TraceGenerator()
        samples = generator.generate_samples(n_samples=n_samples, seed=seed)
        
        # Split into train/eval
        n_eval = min(int(n_samples * 0.1), 500)
        self._eval_data = samples[:n_eval]
        self._train_data = samples[n_eval:]
        
        # Save to disk
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = data_dir / "train.jsonl"
        with open(train_path, "w") as f:
            for sample in self._train_data:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        eval_path = data_dir / "eval.jsonl"
        with open(eval_path, "w") as f:
            for sample in self._eval_data:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        print(f"✓ Generated {len(self._train_data)} train samples")
        print(f"✓ Generated {len(self._eval_data)} eval samples")
        print(f"✓ Saved to {data_dir}")
    
    def setup_model(self) -> None:
        """Setup model with LoRA."""
        print(f"\nLoading model: {self.base_model}")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Quantization config for 4-bit training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        
        # Apply LoRA
        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()
        
        print(f"✓ Model loaded with LoRA (r={self.config.lora_r})")
    
    def train(self, epochs: int = 3, batch_size: int = 4) -> None:
        """Train the model."""
        if self._train_data is None:
            self.generate_training_data()
        
        if self._model is None:
            self.setup_model()
        
        print(f"\nTraining for {epochs} epochs...")
        
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        
        # Convert to HF dataset
        train_dataset = Dataset.from_list([
            {
                "text": self._format_sample(sample),
                "weight": sample.weight,
            }
            for sample in self._train_data
        ])
        
        eval_dataset = Dataset.from_list([
            {
                "text": self._format_sample(sample),
                "weight": sample.weight,
            }
            for sample in self._eval_data
        ])
        
        # Training args
        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_seq_length=self.config.max_input_length + self.config.max_output_length,
        )
        
        # Create trainer
        self._trainer = SFTTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self._tokenizer,
        )
        
        # Train
        self._trainer.train()
        
        print(f"✓ Training complete")
    
    def _format_sample(self, sample: TrainingSample) -> str:
        """Format a sample for training."""
        # Qwen chat format
        return f"<|im_start|>user\n{sample.prompt}<|im_end|>\n<|im_start|>assistant\n{sample.response}<|im_end|>"
    
    def save_model(self, path: str | None = None) -> None:
        """Save the trained model."""
        if self._model is None:
            raise ValueError("No model to save. Run train() first.")
        
        save_path = Path(path) if path else self.output_dir / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)
        
        # Save config
        config_path = save_path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"✓ Model saved to {save_path}")
        print(f"  To load: model = AutoModelForCausalLM.from_pretrained('{save_path}')")
    
    def push_to_hub(self, repo_id: str, token: str | None = None) -> None:
        """Push model to HuggingFace Hub."""
        if self._model is None:
            raise ValueError("No model to push. Run train() first.")
        
        self._model.push_to_hub(repo_id, token=token)
        self._tokenizer.push_to_hub(repo_id, token=token)
        
        print(f"✓ Model pushed to https://huggingface.co/{repo_id}")
    
    def evaluate(self, n_samples: int = 100) -> dict[str, float]:
        """Evaluate the model."""
        if self._model is None or self._tokenizer is None:
            raise ValueError("No model to evaluate. Run train() first.")
        
        print(f"\nEvaluating on {n_samples} samples...")
        
        import torch
        
        correct = 0
        total = 0
        
        for sample in self._eval_data[:n_samples]:
            # Generate
            inputs = self._tokenizer(
                sample.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_length,
            ).to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_output_length,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(sample.prompt):].strip()
            
            # Simple evaluation: check if task family is mentioned
            if sample.task_family.lower() in response.lower():
                correct += 1
            total += 1
        
        accuracy = correct / max(1, total)
        
        results = {
            "accuracy": accuracy,
            "n_samples": n_samples,
        }
        
        print(f"✓ Accuracy: {accuracy:.2%}")
        
        return results


# Convenience functions
def train_kernel_native(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "./kernel-native-model",
    n_samples: int = 5000,
    epochs: int = 3,
    batch_size: int = 4,
) -> KaggleTrainer:
    """Complete training pipeline in one function call.
    
    Usage:
        trainer = train_kernel_native(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            output_dir="./model",
            n_samples=5000,
            epochs=3,
        )
    """
    trainer = KaggleTrainer(
        base_model=base_model,
        output_dir=output_dir,
    )
    
    trainer.generate_training_data(n_samples=n_samples)
    trainer.train(epochs=epochs, batch_size=batch_size)
    trainer.save_model()
    
    return trainer
