#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def resolve_repo() -> Path:
    env = os.environ.get("KERNELWEAVE_CORE_REPO")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            ROOT.parent / "kernelweave",
            ROOT.parent / "kernelweave-core",
            Path.cwd() / "kernelweave",
            Path.cwd() / "kernelweave-core",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit("Set KERNELWEAVE_CORE_REPO or clone the main KernelWeave repo beside this handoff folder.")


CORE_REPO = resolve_repo()
sys.path.insert(0, str(CORE_REPO))

from kernelweave.kernel import KernelStore
from kernelweave.kernels import install_kernel_library
from kernelweave.runtime import KernelRuntime

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


DEFAULT_BASE_MODEL = os.environ.get("KERNELWEAVE_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_LR = float(os.environ.get("KERNELWEAVE_LR", "3e-4"))
DEFAULT_EPOCHS = int(os.environ.get("KERNELWEAVE_EPOCHS", "4"))
DEFAULT_MAX_LEN = int(os.environ.get("KERNELWEAVE_MAX_LEN", "768"))
DEFAULT_BATCH = int(os.environ.get("KERNELWEAVE_BATCH_SIZE", "1"))
DEFAULT_GRAD_ACCUM = int(os.environ.get("KERNELWEAVE_GRAD_ACCUM", "8"))


class ProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps:
            pct = 100.0 * float(state.global_step) / float(state.max_steps)
            print(f"[train {pct:6.2f}%] step {state.global_step}/{state.max_steps}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        total_epochs = float(args.num_train_epochs)
        epoch = float(state.epoch or 0.0)
        pct = 100.0 * min(epoch, total_epochs) / total_epochs
        print(f"[epoch {epoch:4.2f}/{total_epochs:4.2f}] {pct:6.2f}% complete", flush=True)


def build_text(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return f"### User:\n{prompt}\n\n### Assistant:\n{response}"


def load_phase_dataset(data_dir: Path, tokenizer, max_len: int):
    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / "eval.jsonl"
    if not train_path.exists() or not eval_path.exists():
        raise SystemExit(f"Missing dataset files in {data_dir}. Run scripts/generate_dataset.py first.")

    def _prep(example: dict[str, Any]) -> dict[str, Any]:
        text = build_text(tokenizer, example["prompt"], example["response"])
        return tokenizer(text, truncation=True, max_length=max_len)

    train = load_dataset("json", data_files=str(train_path), split="train")
    eval_ = load_dataset("json", data_files=str(eval_path), split="train")

    train = train.map(_prep, remove_columns=train.column_names)
    eval_ = eval_.map(_prep, remove_columns=eval_.column_names)
    return train, eval_


def prepare_model(base_model: str, max_len: int):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(os.environ.get("KERNELWEAVE_LORA_R", "16")),
        lora_alpha=int(os.environ.get("KERNELWEAVE_LORA_ALPHA", "32")),
        lora_dropout=float(os.environ.get("KERNELWEAVE_LORA_DROPOUT", "0.05")),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def build_report(history: list[dict[str, Any]], output_dir: Path, model_name: str, base_model: str, train_rows: int, eval_rows: int, data_dir: Path, phase_results: list[dict[str, Any]]):
    report = {
        "model_name": model_name,
        "base_model": base_model,
        "output_dir": str(output_dir),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "history": history,
        "phase_results": phase_results,
        "data_dir": str(data_dir),
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def benchmark_model(store_root: Path, prompts: list[str], output_dir: Path):
    store = KernelStore(store_root)
    if not store.list_kernels():
        install_kernel_library(store)
    runtime = KernelRuntime(store)
    results = [runtime.run(prompt) for prompt in prompts]
    payload = {"summary": store.summary(), "prompts": prompts, "results": results}
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelWeave Phase C trainer")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "phasec" / "best_adapter"))
    parser.add_argument("--store-dir", default=str(ROOT / "store"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    model, tokenizer = prepare_model(args.base_model, args.max_len)
    train_ds, eval_ds = load_phase_dataset(data_dir, tokenizer, args.max_len)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, args.batch_size)))
    total_steps = steps_per_epoch * args.epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_num_workers=2,
        run_name="kernelweave-phasec",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    history: list[dict[str, Any]] = []

    class EvalRecorder(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            record = {"epoch": float(state.epoch or 0.0), "metrics": metrics or {}}
            history.append(record)
            print(f"[eval] epoch={record['epoch']:.2f} metrics={json.dumps(metrics or {}, sort_keys=True)}", flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[ProgressCallback(), EvalRecorder()],
    )

    print(f"Base model: {args.base_model}")
    print(f"Train rows: {len(train_ds)} | Eval rows: {len(eval_ds)}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(f"Max length: {args.max_len}")
    print("Starting training...")
    trainer.train()

    best_dir = output_dir / "best_adapter"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    phase_results = benchmark_model(Path(args.store_dir), [
        "compare two artifacts and summarize the differences",
        "find all TODOs in a codebase",
        "convert JSON to YAML without losing structure",
        "debug a failing training script and explain the fix",
        "summarize the architecture and cost model",
    ], output_dir)

    report = build_report(
        history=history,
        output_dir=output_dir,
        model_name=str(best_dir),
        base_model=args.base_model,
        train_rows=len(train_ds),
        eval_rows=len(eval_ds),
        data_dir=data_dir,
        phase_results=phase_results["results"],
    )

    print(json.dumps({
        "saved_model": str(best_dir),
        "training_report": str(output_dir / "training_report.json"),
        "benchmark": str(output_dir / "benchmark.json"),
        "report": report,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
