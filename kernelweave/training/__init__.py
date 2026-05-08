"""Kernel-native training: Fine-tune models on kernel execution traces.

This is the REVOLUTIONARY component: kernels influence weights.

Training objective:
    Loss = λ₁·TokenLoss + λ₂·ExecutionLoss + λ₃·VerificationLoss

Usage:
    from kernelweave.training import TraceTrainer
    
    trainer = TraceTrainer(
        base_model="Qwen/Qwen2.5-7B",
        kernel_store="./kernel_store",
    )
    
    trainer.train(
        traces=verified_traces,
        epochs=3,
        output_dir="./kernel-native-model",
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from pathlib import Path
import json
import time

# Training dependencies (optional - graceful fallback if not available)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object
    DataLoader = object


@dataclass
class ExecutionTrace:
    """A complete kernel execution trace for training."""
    trace_id: str
    prompt: str
    kernel_id: str | None
    preconditions: list[str]
    steps: list[dict[str, Any]]
    postconditions: list[str]
    execution_output: str
    verification_result: dict[str, Any]
    success: bool
    confidence: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_training_sample(self) -> dict[str, Any]:
        """Convert trace to training sample format."""
        # Format for token loss
        token_text = f"<|user|>\n{self.prompt}\n<|assistant|<\n{self.execution_output}"
        
        # Format for execution loss
        execution_text = "\n".join(
            f"Step {i+1}: {step.get('action', 'step')} - {step.get('text', step.get('tool', ''))}"
            for i, step in enumerate(self.steps)
        )
        
        # Format for verification loss
        verification_text = "\n".join(
            f"✓ {cond}" if self.verification_result.get("passed", False) else f"✗ {cond}"
            for cond in self.postconditions
        )
        
        return {
            "trace_id": self.trace_id,
            "token_text": token_text,
            "execution_text": execution_text,
            "verification_text": verification_text,
            "success": self.success,
            "confidence": self.confidence,
            "kernel_id": self.kernel_id,
        }
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "kernel_id": self.kernel_id,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "postconditions": self.postconditions,
            "execution_output": self.execution_output,
            "verification_result": self.verification_result,
            "success": self.success,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionTrace":
        return cls(**data)


@dataclass
class TrainingConfig:
    """Configuration for kernel-native training."""
    base_model: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "./kernel-native-model"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_seq_length: int = 2048
    
    # Loss weights (the revolutionary part)
    token_loss_weight: float = 1.0
    execution_loss_weight: float = 0.5
    verification_loss_weight: float = 0.3
    
    # Hardware
    device: str = "cuda"
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # Logging
    log_every: int = 10
    save_every: int = 500
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "token_loss_weight": self.token_loss_weight,
            "execution_loss_weight": self.execution_loss_weight,
            "verification_loss_weight": self.verification_loss_weight,
            "device": self.device,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "log_every": self.log_every,
            "save_every": self.save_every,
        }


class KernelTraceDataset(Dataset):
    """Dataset of kernel execution traces for training."""
    
    def __init__(
        self,
        traces: list[ExecutionTrace],
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.traces = traces
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.traces)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        trace = self.traces[idx]
        sample = trace.to_training_sample()
        
        # Tokenize
        encoding = self.tokenizer(
            sample["token_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone(),
            "success": sample["success"],
            "confidence": sample["confidence"],
        }


class TraceTrainer:
    """Train models on kernel execution traces.
    
    This is the REVOLUTIONARY component: kernels influence weights.
    
    The training objective combines three losses:
    1. Token loss: Standard next-token prediction
    2. Execution loss: Did the model execute the kernel steps correctly?
    3. Verification loss: Did the postconditions hold?
    
    Usage:
        trainer = TraceTrainer(
            base_model="Qwen/Qwen2.5-7B",
            output_dir="./kernel-native-model",
        )
        
        trainer.train(traces=verified_traces)
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B",
        output_dir: str = "./kernel-native-model",
        config: TrainingConfig | None = None,
    ):
        self.base_model_name = base_model
        self.output_dir = Path(output_dir)
        self.config = config or TrainingConfig(base_model=base_model, output_dir=output_dir)
        
        self.model = None
        self.tokenizer = None
        self.device = self.config.device
        
    def load_model(self) -> None:
        """Load base model and tokenizer."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and Transformers required for training. "
                "Install with: pip install torch transformers"
            )
        
        print(f"Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        
        print(f"✓ Model loaded on {self.device}")
    
    def train(
        self,
        traces: list[ExecutionTrace],
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """Fine-tune model on kernel execution traces.
        
        Args:
            traces: List of verified execution traces
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        
        Returns:
            Training metrics
        """
        if self.model is None:
            self.load_model()
        
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate
        
        # Filter to successful traces only
        successful_traces = [t for t in traces if t.success]
        print(f"Training on {len(successful_traces)} successful traces (filtered from {len(traces)} total)")
        
        if len(successful_traces) == 0:
            print("⚠ No successful traces to train on. Skipping training.")
            return {"status": "skipped", "reason": "no_successful_traces"}
        
        # Create dataset
        dataset = KernelTraceDataset(
            traces=successful_traces,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.log_every,
            save_steps=self.config.save_every,
            save_total_limit=3,
            fp16=self.device == "cuda",
            report_to="none",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        print(f"\nStarting training:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Traces: {len(successful_traces)}")
        print()
        
        start_time = time.time()
        trainer.train()
        duration = time.time() - start_time
        
        # Save
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✓ Training complete in {duration:.1f}s")
        print(f"✓ Model saved to {self.output_dir}")
        
        # Save training metadata
        metadata = {
            "base_model": self.base_model_name,
            "traces_used": len(successful_traces),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "duration_seconds": duration,
            "loss_weights": {
                "token": self.config.token_loss_weight,
                "execution": self.config.execution_loss_weight,
                "verification": self.config.verification_loss_weight,
            },
        }
        
        metadata_path = self.output_dir / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        return {
            "status": "success",
            "traces_used": len(successful_traces),
            "duration_seconds": duration,
            "output_dir": str(self.output_dir),
        }
    
    def export_training_data(
        self,
        traces: list[ExecutionTrace],
        output_path: Path,
    ) -> None:
        """Export traces to JSONL for external training.
        
        Use this if you want to train on a different platform.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as f:
            for trace in traces:
                if trace.success:
                    sample = trace.to_training_sample()
                    f.write(json.dumps(sample) + "\n")
        
        print(f"✓ Exported {len(traces)} traces to {output_path}")


class TraceCollector:
    """Collect execution traces from model runs.
    
    Usage:
        collector = TraceCollector(kernel_store)
        traces = collector.collect_from_prompts(
            prompts=["compare files", "find TODOs"],
            model=my_model,
            n_iterations=10,
        )
    """
    
    def __init__(
        self,
        kernel_store: Any,
        verifier_hierarchy: bool = True,
    ):
        self.kernel_store = kernel_store
        self.verifier_hierarchy = verifier_hierarchy
        self.traces: list[ExecutionTrace] = []
    
    def collect_from_prompts(
        self,
        prompts: list[str],
        model: Any,
        n_iterations: int = 1,
    ) -> list[ExecutionTrace]:
        """Run prompts through model and collect traces."""
        for prompt in prompts:
            for i in range(n_iterations):
                result = model.run(prompt)
                
                trace = ExecutionTrace(
                    trace_id=f"trace-{time.time()}-{i}",
                    prompt=prompt,
                    kernel_id=result.get("kernel_id"),
                    preconditions=result.get("preconditions", []),
                    steps=result.get("plan", []),
                    postconditions=result.get("postconditions", []),
                    execution_output=result.get("output", ""),
                    verification_result=result.get("verification", {}),
                    success=result.get("verification", {}).get("passed", False),
                    confidence=result.get("confidence", 0.0),
                    timestamp=time.time(),
                )
                
                self.traces.append(trace)
        
        return self.traces
    
    def get_verified_traces(self) -> list[ExecutionTrace]:
        """Get only traces that passed verification."""
        return [t for t in self.traces if t.success]
    
    def save_traces(self, output_path: Path) -> None:
        """Save traces to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as f:
            for trace in self.traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        
        print(f"✓ Saved {len(self.traces)} traces to {output_path}")


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kernel-native training")
    parser.add_argument("command", choices=["train", "collect", "export"])
    parser.add_argument("--traces", type=Path, help="Path to traces JSONL")
    parser.add_argument("--prompts", type=Path, help="Path to prompts JSONL")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output", type=Path, default=Path("./kernel-native-model"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    if args.command == "train":
        if not args.traces:
            print("Error: --traces required for training")
            return
        
        # Load traces
        traces = []
        with args.traces.open() as f:
            for line in f:
                traces.append(ExecutionTrace.from_dict(json.loads(line)))
        
        # Train
        trainer = TraceTrainer(
            base_model=args.base_model,
            output_dir=str(args.output),
        )
        
        trainer.train(
            traces=traces,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    
    elif args.command == "export":
        if not args.traces:
            print("Error: --traces required for export")
            return
        
        traces = []
        with args.traces.open() as f:
            for line in f:
                traces.append(ExecutionTrace.from_dict(json.loads(line)))
        
        trainer = TraceTrainer(base_model=args.base_model)
        trainer.export_training_data(traces, args.output / "training_data.jsonl")


if __name__ == "__main__":
    main()
