#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/AmSach/kernelweave.git"
DEFAULT_REPO_DIR = Path("/kaggle/working/kernelweave")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("KW_OUTPUT_DIR", "/kaggle/working/kernelweave_llm_bundle"))
DEFAULT_EXPORT_DIR = Path(os.environ.get("KW_EXPORT_DIR", "/kaggle/working/kernelweave_llm_export"))
DEFAULT_BASE_MODEL = os.environ.get("KW_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_FALLBACK_MODELS = [m.strip() for m in os.environ.get("KW_FALLBACK_MODELS", "TinyLlama/TinyLlama-1.1B-Chat-v1.0,sshleifer/tiny-gpt2").split(",") if m.strip()]
DEFAULT_SAMPLES = int(os.environ.get("KW_SAMPLES", "6000"))
DEFAULT_EPOCHS = int(os.environ.get("KW_EPOCHS", "3"))
DEFAULT_BATCH = int(os.environ.get("KW_BATCH", "1"))
DEFAULT_GRAD_ACCUM = int(os.environ.get("KW_GRAD_ACCUM", "8"))
DEFAULT_LR = float(os.environ.get("KW_LR", "2e-4"))
DEFAULT_MAX_LEN = int(os.environ.get("KW_MAX_LEN", "512"))
DEFAULT_SEED = int(os.environ.get("KW_SEED", "42"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("KW_MAX_NEW_TOKENS", "192"))
SYSTEM_PROMPT = (
    "You are KernelWeave, a verification-driven kernel router. "
    "Prefer the smallest verified kernel that satisfies the request. "
    "Use exact, grounded reasoning. Do not invent evidence."
)
COMMON_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "Wqkv",
    "query_key_value",
    "c_attn",
]


@dataclass
class KernelSpec:
    kernel_id: str
    name: str
    task_family: str
    description: str
    preconditions: list[str]
    postconditions: list[str]
    evidence_requirements: list[str]
    rollback: list[str]
    steps: list[dict[str, Any]]
    confidence: float


@dataclass
class Example:
    prompt: str
    response: str
    task_family: str
    target: str
    kernel_id: str | None = None
    kernel_name: str | None = None


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_deps() -> None:
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers>=4.44.0"),
        ("datasets", "datasets>=2.20.0"),
        ("peft", "peft>=0.12.0"),
        ("accelerate", "accelerate>=0.33.0"),
        ("sentencepiece", "sentencepiece"),
        ("safetensors", "safetensors"),
    ]
    for import_name, pip_name in packages:
        try:
            __import__(import_name)
        except Exception:
            run([sys.executable, "-m", "pip", "install", "-q", pip_name])


def locate_repo() -> Path:
    env = os.environ.get("KERNELWEAVE_REPO")
    if env:
        repo = Path(env).expanduser().resolve()
        if (repo / "kernelweave").is_dir():
            return repo
    for candidate in [DEFAULT_REPO_DIR, Path("/kaggle/input/kernelweave"), Path("/kaggle/input/kernelweave-repo"), Path("/workspace/kernelweave")]:
        if (candidate / "kernelweave").is_dir():
            return candidate.resolve()
    if DEFAULT_REPO_DIR.exists():
        return DEFAULT_REPO_DIR.resolve()
    print("[setup] cloning KernelWeave repo")
    run(["git", "clone", "--depth", "1", REPO_URL, str(DEFAULT_REPO_DIR)])
    return DEFAULT_REPO_DIR.resolve()


def load_kernel_specs(repo_root: Path) -> list[KernelSpec]:
    kernels_dir = repo_root / "phasecd" / "store" / "kernels"
    specs: list[KernelSpec] = []
    for path in sorted(kernels_dir.glob("*.json")):
        data = json.loads(path.read_text())
        specs.append(
            KernelSpec(
                kernel_id=data["kernel_id"],
                name=data["name"],
                task_family=data["task_family"],
                description=data.get("description", ""),
                preconditions=list(data.get("preconditions", [])),
                postconditions=list(data.get("postconditions", [])),
                evidence_requirements=list(data.get("evidence_requirements", [])),
                rollback=list(data.get("rollback", [])),
                steps=list(data.get("steps", [])),
                confidence=float(data.get("status", {}).get("confidence", 0.0)),
            )
        )
    if not specs:
        raise SystemExit(f"No kernel specs found in {kernels_dir}")
    return specs


PROMPT_TEMPLATES = [
    "Route this task to the best kernel. Task: {task}. Return JSON with kernel_id, kernel_name, reason, plan, and verification_plan.",
    "Kernel contract:\n- family: {family}\n- name: {name}\n- preconditions: {preconditions}\n- postconditions: {postconditions}\n- evidence: {evidence}\n\nTask: {task}\nReturn a concise JSON plan.",
    "Given the kernel contract below, choose whether to reuse it or fall back to generation.\nFamily: {family}\nKernel: {name}\nTask: {task}",
    "Use the most relevant verified kernel. Task: {task}. Mention the kernel id and the evidence you would check.",
]

MEMORY_TEMPLATES = [
    "Retrieve the reusable kernel for: {task}. Explain why it should be reused instead of regenerated.",
    "Memory mode for this task: {task}. Return a short plan that references the stored kernel.",
    "For the request below, act like a kernel memory system. Task: {task}",
]

TRACE_TEMPLATES = [
    "Reconstruct a successful trace for: {task}. Include steps, evidence, and the verification outcome.",
    "Given the kernel contract, produce a trace-style answer for: {task}",
]

DISTILL_TEMPLATES = [
    "Distill the repeated pattern in this kernel family for: {task}",
    "Compress the reusable behavior into a short instruction for: {task}",
]

COMPOSE_TEMPLATES = [
    "Compose these two kernels into one plan: {left_family} + {right_family}. Task: {task}",
    "Chain the best kernels you can to solve: {task}. Mention both kernel ids if possible.",
]

BENCHMARK_CASES = [
    {
        "name": "compare-artifacts",
        "family": "comparison",
        "prompt": "Compare main.py and utils.py and summarize the differences clearly.",
    },
    {
        "name": "find-todos",
        "family": "search",
        "prompt": "Find all TODOs in a codebase and explain where they matter most.",
    },
    {
        "name": "debug-failure",
        "family": "debugging",
        "prompt": "Debug a failing training script and explain the fix.",
    },
    {
        "name": "json-to-yaml",
        "family": "transformation",
        "prompt": "Convert this JSON to YAML without losing structure: {\"name\": \"alice\", \"age\": 30}",
    },
    {
        "name": "summarize-architecture",
        "family": "summarization",
        "prompt": "Summarize the architecture and cost model of a kernel router in 5 bullet points.",
    },
    {
        "name": "generate-tests",
        "family": "testing",
        "prompt": "Write tests for a Python function that parses config files and handles missing values.",
    },
    {
        "name": "doc-gen",
        "family": "documentation generation",
        "prompt": "Draft README text for a kernel routing system that uses verification and memory.",
    },
    {
        "name": "security-audit",
        "family": "security audit",
        "prompt": "Audit this API route design for security issues and give concrete fixes.",
    },
]

TASK_PROMPTS = {
    "comparison": [
        "Compare {a} and {b}.",
        "What are the key differences between {a} and {b}?",
        "Compare {a} with {b} and keep the answer grounded.",
    ],
    "analysis": [
        "Analyze {item} for likely issues.",
        "Review {item} and identify the main risks.",
    ],
    "search": [
        "Find all files like {pattern} in {directory}.",
        "Search the codebase for {pattern} under {directory}.",
    ],
    "generation": [
        "Generate a {format} report from {source}.",
        "Create a {format} summary of {topic}.",
    ],
    "debugging": [
        "Fix the bug in {file} that causes {error}.",
        "Debug {file} and resolve {error}.",
    ],
    "summarization": [
        "Summarize {document}.",
        "Create a summary of {document}.",
    ],
    "transformation": [
        "Convert {input} from {format_a} to {format_b}.",
        "Transform {input} into {format_b}.",
    ],
    "testing": [
        "Write tests for {module}.",
        "Create test cases for {function}.",
    ],
    "documentation generation": [
        "Draft documentation for {subject}.",
        "Write a README section for {subject}.",
    ],
    "security audit": [
        "Audit {subject} for security risks.",
        "Review {subject} and identify vulnerabilities.",
    ],
    "code analysis": [
        "Analyze the code in {subject}.",
        "Inspect {subject} and summarize the code quality.",
    ],
    "artifact comparison": [
        "Compare {left} and {right} and summarize differences.",
        "Analyze structural and content differences between {left} and {right}.",
    ],
    "format conversion": [
        "Convert {input} from {format_a} to {format_b}.",
        "Transform {input} into a valid {format_b} representation.",
    ],
    "test generation": [
        "Generate tests for {subject}.",
        "Write a small test suite for {subject}.",
    ],
    "readme generation": [
        "Write a README for {subject}.",
        "Create concise documentation for {subject}.",
    ],
    "version comparison": [
        "Compare version {a} and {b}.",
        "Explain the differences between version {a} and {b}.",
    ],
}

GENERIC_TASKS = [
    "compare two artifacts and summarize the differences",
    "find all TODOs in a codebase",
    "convert JSON to YAML without losing structure",
    "debug a failing training script and explain the fix",
    "summarize the architecture and cost model",
    "write tests for a config parser",
    "document a reusable automation system",
    "audit an API route for security issues",
    "generate a changelog from structured diffs",
    "extract key facts from a document with evidence",
]

ALPHABET = ["A", "B", "C", "D", "E", "F"]


def family_to_prompt_family(family: str) -> str:
    return family.lower().strip()


def build_task_for_kernel(kernel: KernelSpec, rng: random.Random) -> str:
    family = family_to_prompt_family(kernel.task_family)
    pre = "; ".join(kernel.preconditions[:2]) or "no special preconditions"
    post = "; ".join(kernel.postconditions[:2]) or "no special postconditions"
    evidence = "; ".join(kernel.evidence_requirements[:2]) or "no special evidence"
    template_pool = PROMPT_TEMPLATES + MEMORY_TEMPLATES + TRACE_TEMPLATES + DISTILL_TEMPLATES
    template = rng.choice(template_pool)
    generic_task = rng.choice(GENERIC_TASKS)
    return template.format(
        family=family,
        name=kernel.name,
        preconditions=pre,
        postconditions=post,
        evidence=evidence,
        task=generic_task,
    )


def synthesize_examples(kernels: list[KernelSpec], n_samples: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    family_groups: dict[str, list[KernelSpec]] = {}
    for kernel in kernels:
        family_groups.setdefault(kernel.task_family, []).append(kernel)

    while len(examples) < n_samples:
        kernel = rng.choice(kernels)
        family = kernel.task_family
        task = build_task_for_kernel(kernel, rng)

        if rng.random() < 0.2 and len(family_groups.get(family, [])) > 1:
            other = rng.choice([k for k in family_groups[family] if k.kernel_id != kernel.kernel_id])
            prompt = COMPOSE_TEMPLATES[rng.randrange(len(COMPOSE_TEMPLATES))].format(
                left_family=kernel.task_family,
                right_family=other.task_family,
                task=task,
            )
            response = json.dumps(
                {
                    "mode": "composed",
                    "kernel_ids": [kernel.kernel_id, other.kernel_id],
                    "reason": f"composes {kernel.task_family} and {other.task_family}",
                    "plan": [step.get("text", step.get("tool", "step")) for step in (kernel.steps[:3] + other.steps[:3])],
                    "verification_plan": list(dict.fromkeys(kernel.evidence_requirements + other.evidence_requirements))[:5],
                },
                sort_keys=True,
            )
            examples.append(Example(prompt=prompt, response=response, task_family=family, target="composition", kernel_id=kernel.kernel_id, kernel_name=kernel.name))
            continue

        route_payload = {
            "mode": "kernel",
            "kernel_id": kernel.kernel_id,
            "kernel_name": kernel.name,
            "reason": f"matches {kernel.task_family}",
            "plan": [step.get("text", step.get("tool", "step")) for step in kernel.steps[:6]],
            "verification_plan": list(kernel.evidence_requirements[:4]),
        }
        prompt = PROMPT_TEMPLATES[rng.randrange(len(PROMPT_TEMPLATES))].format(
            family=family,
            name=kernel.name,
            preconditions="; ".join(kernel.preconditions[:2]) or "none",
            postconditions="; ".join(kernel.postconditions[:2]) or "none",
            evidence="; ".join(kernel.evidence_requirements[:2]) or "none",
            task=task,
        )
        examples.append(Example(prompt=prompt, response=json.dumps(route_payload, sort_keys=True), task_family=family, target="routing", kernel_id=kernel.kernel_id, kernel_name=kernel.name))

        if len(examples) >= n_samples:
            break

        prompt = MEMORY_TEMPLATES[rng.randrange(len(MEMORY_TEMPLATES))].format(task=task)
        response = json.dumps(
            {
                "mode": "memory",
                "kernel_id": kernel.kernel_id,
                "kernel_name": kernel.name,
                "retrieval": "retrieve kernel and execute its steps",
                "why": kernel.description,
            },
            sort_keys=True,
        )
        examples.append(Example(prompt=prompt, response=response, task_family=family, target="memory", kernel_id=kernel.kernel_id, kernel_name=kernel.name))

        if len(examples) >= n_samples:
            break

        prompt = TRACE_TEMPLATES[rng.randrange(len(TRACE_TEMPLATES))].format(task=task)
        response = json.dumps(
            {
                "trace": [kernel.kernel_id],
                "steps": [step.get("text", step.get("tool", "step")) for step in kernel.steps[:6]],
                "evidence": kernel.evidence_requirements[:4],
                "postconditions": kernel.postconditions[:4],
            },
            sort_keys=True,
        )
        examples.append(Example(prompt=prompt, response=response, task_family=family, target="trace", kernel_id=kernel.kernel_id, kernel_name=kernel.name))

        if len(examples) >= n_samples:
            break

        prompt = DISTILL_TEMPLATES[rng.randrange(len(DISTILL_TEMPLATES))].format(task=task)
        response = json.dumps(
            {
                "distillation": kernel.description,
                "kernel_id": kernel.kernel_id,
                "task_family": kernel.task_family,
            },
            sort_keys=True,
        )
        examples.append(Example(prompt=prompt, response=response, task_family=family, target="distillation", kernel_id=kernel.kernel_id, kernel_name=kernel.name))

    return examples[:n_samples]


def split_examples(examples: list[Example], eval_ratio: float = 0.1) -> tuple[list[Example], list[Example]]:
    split = max(1, int(len(examples) * (1.0 - eval_ratio)))
    return examples[:split], examples[split:]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_text(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{prompt}\n\n### Assistant:\n{response}"


def build_dataset(tokenizer, examples: list[Example], max_len: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex in examples:
        text = build_text(tokenizer, ex.prompt, ex.response)
        rows.append(
            {
                "text": text,
                "task_family": ex.task_family,
                "target": ex.target,
                "kernel_id": ex.kernel_id,
                "kernel_name": ex.kernel_name,
            }
        )
    return rows


def choose_lora_targets(model) -> list[str]:
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in COMMON_LORA_TARGETS:
            suffixes.add(leaf)
    if suffixes:
        return sorted(suffixes)
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "linear" in cls_name and name.split(".")[-1] not in {"lm_head", "embed_tokens"}:
            suffixes.add(name.split(".")[-1])
    return sorted(suffixes)[:8] if suffixes else ["q_proj", "v_proj"]


def gen(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 192) -> tuple[str, int, int, float]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{prompt}\n\n### Assistant:\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    output_ids = out[0][prompt_tokens:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return text, prompt_tokens, int(output_ids.shape[0]), elapsed


def score_case(case: dict[str, str], response: str) -> float:
    text = response.strip().lower()
    family = case["family"]
    if family in {"comparison", "artifact comparison", "version comparison"}:
        return float(any(w in text for w in ["compare", "difference", "both", "structur", "content"]))
    if family in {"search"}:
        return float(any(w in text for w in ["found", "search", "file", "match", "todo"]))
    if family in {"debugging", "security audit"}:
        return float(any(w in text for w in ["fix", "cause", "risk", "vulnerab", "bug", "patch"]))
    if family in {"transformation", "format conversion"}:
        return float(any(w in text for w in ["json", "yaml", "conversion", "structure", "```" ]))
    if family in {"testing", "test generation"}:
        return float("assert" in text or "def test_" in text)
    if family in {"summarization", "documentation generation", "readme generation"}:
        return float(len(text.split()) >= 30)
    if family in {"code analysis"}:
        return float(any(w in text for w in ["analysis", "issue", "quality", "architecture"]))
    return float(len(text) > 20)


def make_report_table(rows: list[dict[str, Any]]) -> str:
    headers = ["case", "model", "score", "tokens", "seconds", "cost_proxy"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['model']} | {row['score']:.2f} | {row['tokens']} | {row['seconds']:.1f} | {row['cost_proxy']:.1f} |"
        )
    return "\n".join(lines)


def export_bundle(bundle_dir: Path, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists():
        archive_root = shutil.make_archive(str(export_dir / "kernelweave_llm_bundle"), "zip", root_dir=bundle_dir)
        return Path(archive_root)
    raise SystemExit(f"Bundle directory missing: {bundle_dir}")


def setup_model(base_model: str, device: str, fallback_models: list[str] | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    candidates = [base_model]
    for model_name in fallback_models or DEFAULT_FALLBACK_MODELS:
        if model_name and model_name not in candidates:
            candidates.append(model_name)

    last_error: Exception | None = None
    for model_name in candidates:
        try:
            print(f"[model] loading {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
            )
            model.to(device)
            model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print(f"[model] ready: {model_name}")
            return model, tokenizer, model_name
        except Exception as exc:
            last_error = exc
            print(f"[model] failed: {model_name}: {exc}")
            _release_torch()
            continue

    raise SystemExit(f"Unable to load any model candidate. Last error: {last_error}")


def build_training_arguments(output_dir: Path, epochs: int, batch_size: int, grad_accum: int, lr: float) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "logging_steps": 10,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "fp16": True,
        "bf16": False,
        "max_grad_norm": 1.0,
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "dataloader_num_workers": 0,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def train_lora(base_model: str, output_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], epochs: int, batch_size: int, grad_accum: int, lr: float, max_len: int, seed: int):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, set_seed

    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer, loaded_model_name = setup_model(base_model, device)
    lora_targets = choose_lora_targets(model)
    print(f"[lora] targets={lora_targets}")
    print(f"[lora] base_model={loaded_model_name}")

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    def attach_text(example):
        return {
            "text": build_text(tokenizer, example["prompt"], example["response"])
        }

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_len)
        out["labels"] = out["input_ids"].copy()
        return out

    train_text_ds = train_ds.map(attach_text)
    eval_text_ds = eval_ds.map(attach_text)
    train_tok = train_text_ds.map(tokenize, remove_columns=train_text_ds.column_names)
    eval_tok = eval_text_ds.map(tokenize, remove_columns=eval_text_ds.column_names)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"[train] step={state.global_step} logs={json.dumps(logs, sort_keys=True)}", flush=True)

    args = build_training_arguments(output_dir, epochs, batch_size, grad_accum, lr)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[LogCallback()],
    )

    print(f"[train] samples={len(train_tok)} eval={len(eval_tok)}")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    return tokenizer, model, trainer.model, adapter_dir, merged_dir


def _release_torch() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _load_model(base_model: str, device: str, path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model_path = path or base_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def evaluate_models(base_model: str, merged_dir: Path, benchmark_dir: Path, device: str, max_new_tokens: int):
    tokenizer, base = _load_model(base_model, device)
    base_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"cases": []}
    for case in BENCHMARK_CASES:
        text, prompt_tokens, out_tokens, elapsed = gen(base, tokenizer, case["prompt"], device, max_new_tokens=max_new_tokens)
        score = score_case(case, text)
        cost_proxy = float(prompt_tokens + out_tokens)
        base_rows.append({
            "case": case["name"],
            "model": "base",
            "score": score,
            "tokens": prompt_tokens + out_tokens,
            "seconds": elapsed,
            "cost_proxy": cost_proxy,
        })
        detail["cases"].append({
            "case": case["name"],
            "prompt": case["prompt"],
            "results": {
                "base": {
                    "text": text,
                    "score": score,
                    "tokens": prompt_tokens + out_tokens,
                    "seconds": elapsed,
                    "cost_proxy": cost_proxy,
                }
            }
        })
    del base
    del tokenizer
    _release_torch()

    tokenizer, tuned = _load_model(base_model, device, str(merged_dir))
    tuned_rows: list[dict[str, Any]] = []
    for idx, case in enumerate(BENCHMARK_CASES):
        text, prompt_tokens, out_tokens, elapsed = gen(tuned, tokenizer, case["prompt"], device, max_new_tokens=max_new_tokens)
        score = score_case(case, text)
        cost_proxy = float(prompt_tokens + out_tokens)
        tuned_rows.append({
            "case": case["name"],
            "model": "tuned",
            "score": score,
            "tokens": prompt_tokens + out_tokens,
            "seconds": elapsed,
            "cost_proxy": cost_proxy,
        })
        detail["cases"][idx]["results"]["tuned"] = {
            "text": text,
            "score": score,
            "tokens": prompt_tokens + out_tokens,
            "seconds": elapsed,
            "cost_proxy": cost_proxy,
        }
    del tuned
    del tokenizer
    _release_torch()

    all_rows = base_rows + tuned_rows
    base_quality = sum(r["score"] for r in base_rows) / len(base_rows)
    tuned_quality = sum(r["score"] for r in tuned_rows) / len(tuned_rows)
    base_cost = sum(r["cost_proxy"] for r in base_rows) / len(base_rows)
    tuned_cost = sum(r["cost_proxy"] for r in tuned_rows) / len(tuned_rows)

    summary = {
        "base_quality": base_quality,
        "tuned_quality": tuned_quality,
        "quality_delta": tuned_quality - base_quality,
        "base_cost_proxy": base_cost,
        "tuned_cost_proxy": tuned_cost,
        "cost_delta": tuned_cost - base_cost,
        "cases": len(BENCHMARK_CASES),
    }

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "benchmark.json").write_text(json.dumps({"summary": summary, **detail}, indent=2, sort_keys=True))
    (benchmark_dir / "benchmark.md").write_text(
        "# KernelWeave Benchmark\n\n"
        + make_report_table(all_rows)
        + f"\n\n## Summary\n\n- Base quality: {base_quality:.2f}\n- Tuned quality: {tuned_quality:.2f}\n- Quality delta: {tuned_quality - base_quality:+.2f}\n- Base cost proxy: {base_cost:.1f}\n- Tuned cost proxy: {tuned_cost:.1f}\n- Cost delta: {tuned_cost - base_cost:+.1f}\n"
    )
    return summary, all_rows


def local_smoke_test(merged_dir: Path, base_model: str, device: str, max_new_tokens: int) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(merged_dir), trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32).to(device)
    model.eval()
    cases = BENCHMARK_CASES[:3]
    rows = []
    for case in cases:
        text, prompt_tokens, out_tokens, elapsed = gen(model, tokenizer, case["prompt"], device, max_new_tokens=max_new_tokens)
        rows.append({
            "case": case["name"],
            "output": text,
            "tokens": prompt_tokens + out_tokens,
            "seconds": elapsed,
            "score": score_case(case, text),
        })
    return rows


def realtime_demo(base_model: str, merged_dir: Path, prompts: list[str], output_dir: Path, device: str, max_new_tokens: int) -> list[dict[str, Any]]:
    base_tokenizer, base = _load_model(base_model, device)
    rows: list[dict[str, Any]] = []
    md_lines = ["# KernelWeave realtime demo", ""]
    for i, prompt in enumerate(prompts, 1):
        base_text, base_prompt_tokens, base_out_tokens, base_seconds = gen(base, base_tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        row = {
            "index": i,
            "prompt": prompt,
            "base": {
                "text": base_text,
                "tokens": base_prompt_tokens + base_out_tokens,
                "seconds": base_seconds,
                "score": _score_realtime(prompt, base_text),
            },
            "tuned": {},
        }
        rows.append(row)
        md_lines.extend([
            f"## Prompt {i}",
            "",
            prompt,
            "",
            f"**Base** ({row['base']['tokens']} tok, {row['base']['seconds']:.1f}s, score {row['base']['score']:.2f})",
            "",
            row["base"]["text"] or "_empty_",
            "",
        ])
    del base
    del base_tokenizer
    _release_torch()

    tuned_tokenizer, tuned = _load_model(base_model, device, str(merged_dir))
    for i, prompt in enumerate(prompts, 1):
        tuned_text, tuned_prompt_tokens, tuned_out_tokens, tuned_seconds = gen(tuned, tuned_tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        rows[i - 1]["tuned"] = {
            "text": tuned_text,
            "tokens": tuned_prompt_tokens + tuned_out_tokens,
            "seconds": tuned_seconds,
            "score": _score_realtime(prompt, tuned_text),
        }
        md_lines.extend([
            f"**Tuned** ({rows[i - 1]['tuned']['tokens']} tok, {rows[i - 1]['tuned']['seconds']:.1f}s, score {rows[i - 1]['tuned']['score']:.2f})",
            "",
            rows[i - 1]["tuned"]["text"] or "_empty_",
            "",
            "---",
            "",
        ])
    del tuned
    del tuned_tokenizer
    _release_torch()

    (output_dir / "realtime_demo.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    (output_dir / "realtime_demo.md").write_text("\n".join(md_lines))
    print("\n======== REALTIME DEMO ========")
    for row in rows:
        print(f"\n[PROMPT {row['index']}] {row['prompt']}")
        print(f"[BASE ] score={row['base']['score']:.2f} tok={row['base']['tokens']} sec={row['base']['seconds']:.1f}")
        print(row["base"]["text"])
        print(f"[TUNED] score={row['tuned']['score']:.2f} tok={row['tuned']['tokens']} sec={row['tuned']['seconds']:.1f}")
        print(row["tuned"]["text"])
    print("==============================")
    return rows


def _score_realtime(prompt: str, response: str) -> float:
    text = response.strip().lower()
    prompt_text = prompt.lower()
    if any(k in prompt_text for k in ["compare", "difference"]):
        return float(any(w in text for w in ["compare", "difference", "both", "structur", "content"]))
    if any(k in prompt_text for k in ["todo", "search", "find"]):
        return float(any(w in text for w in ["found", "search", "match", "todo", "file"]))
    if any(k in prompt_text for k in ["convert", "yaml", "json"]):
        return float(any(w in text for w in ["json", "yaml", "structure", "conversion", "```" ]))
    if any(k in prompt_text for k in ["test", "tests"]):
        return float("assert" in text or "def test_" in text)
    if any(k in prompt_text for k in ["audit", "security", "debug"]):
        return float(any(w in text for w in ["fix", "risk", "cause", "bug", "vulnerab"]))
    return float(len(text) > 20)


def build_demo_report(repo_root: Path, output_dir: Path, export_zip: Path, summary: dict[str, Any], rows: list[dict[str, Any]], samples: int, epochs: int, base_model: str) -> Path:
    report = {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "export_zip": str(export_zip),
        "base_model": base_model,
        "samples": samples,
        "epochs": epochs,
        "summary": summary,
        "cases": rows,
        "notes": [
            "This trains a real Hugging Face LLM with LoRA on kernel-aware synthetic data.",
            "Export includes both LoRA adapter and merged full model.",
            "Benchmark compares baseline vs tuned quality and a simple cost proxy.",
        ],
    }
    path = output_dir / "demo_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelWeave LLM product-style training + benchmark")
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--realtime-demo", action="store_true")
    parser.add_argument("--realtime-prompts", type=int, default=3)
    args = parser.parse_args()

    ensure_deps()
    repo_root = args.repo.resolve() if args.repo else locate_repo()
    output_dir = args.output_dir.resolve()
    export_dir = args.export_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        print("======== ENV ========")
        print(f"Python  : {sys.version.split()[0]}")
        print(f"Torch   : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPUs    : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  {props.total_memory // (1024**2)} MiB")
        else:
            print("GPUs    : 0")
    except Exception as exc:
        print(f"[warn] torch probe failed: {exc}")

    kernels = load_kernel_specs(repo_root)
    examples = synthesize_examples(kernels, args.samples, args.seed)
    train_examples, eval_examples = split_examples(examples, eval_ratio=0.1)

    # Persist the synthetic dataset.
    dataset_dir = output_dir / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(dataset_dir / "train.jsonl", [e.__dict__ for e in train_examples])
    save_jsonl(dataset_dir / "eval.jsonl", [e.__dict__ for e in eval_examples])
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "samples": len(examples),
                "train": len(train_examples),
                "eval": len(eval_examples),
                "kernels": len(kernels),
                "base_model": args.base_model,
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.benchmark_only:
        summary = {"mode": "benchmark_only"}
        benchmark_dir = output_dir / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        (benchmark_dir / "benchmark.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    try:
        # Train the actual LLM adapter.
        tokenizer, base_model_obj, tuned_model_obj, adapter_dir, merged_dir = train_lora(
            base_model=args.base_model,
            output_dir=output_dir,
            train_rows=[e.__dict__ for e in train_examples],
            eval_rows=[e.__dict__ for e in eval_examples],
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_len=args.max_len,
            seed=args.seed,
        )

        # Benchmark baseline vs tuned on the same cases.
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        summary, rows = evaluate_models(args.base_model, merged_dir, output_dir / "benchmark", device, args.max_new_tokens)

        # Optional smoke test on the merged model.
        smoke_rows: list[dict[str, Any]] = []
        if args.smoke_only:
            smoke_rows = local_smoke_test(merged_dir, args.base_model, device, args.max_new_tokens)
            (output_dir / "smoke.json").write_text(json.dumps(smoke_rows, indent=2, sort_keys=True))

        realtime_rows: list[dict[str, Any]] = []
        if args.realtime_demo:
            realtime_prompts = [case["prompt"] for case in BENCHMARK_CASES[: max(1, args.realtime_prompts)]]
            realtime_rows = realtime_demo(args.base_model, merged_dir, realtime_prompts, output_dir, device, args.max_new_tokens)

        # Bundle exports.
        bundle_dir = output_dir / "bundle"
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dataset_dir, bundle_dir / "data")
        shutil.copytree(adapter_dir, bundle_dir / "adapter")
        shutil.copytree(merged_dir, bundle_dir / "merged_model")
        shutil.copytree(output_dir / "benchmark", bundle_dir / "benchmark")
        if (output_dir / "realtime_demo.json").exists():
            shutil.copy2(output_dir / "realtime_demo.json", bundle_dir / "realtime_demo.json")
        if (output_dir / "realtime_demo.md").exists():
            shutil.copy2(output_dir / "realtime_demo.md", bundle_dir / "realtime_demo.md")
        bundle_meta = {
            "repo_root": str(repo_root),
            "base_model": args.base_model,
            "loaded_model": args.base_model,
            "samples": args.samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "max_len": args.max_len,
            "summary": summary,
            "realtime_demo": args.realtime_demo,
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(bundle_meta, indent=2, sort_keys=True))
        export_zip = Path(shutil.make_archive(str(export_dir / "kernelweave_llm_bundle"), "zip", root_dir=bundle_dir))

        report_path = build_demo_report(repo_root, output_dir, export_zip, summary, rows, args.samples, args.epochs, args.base_model)

        print("\n======== EXPORT ========")
        print(f"Data dir     : {dataset_dir}")
        print(f"Adapter dir   : {adapter_dir}")
        print(f"Merged model  : {merged_dir}")
        print(f"Benchmark dir : {output_dir / 'benchmark'}")
        print(f"Bundle zip    : {export_zip}")
        print(f"Report        : {report_path}")
        print("========================")
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception:
        error_log = output_dir / "error.log"
        error_log.write_text(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
