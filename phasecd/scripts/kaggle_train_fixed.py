#!/usr/bin/env python3
"""
KernelWeave — Fixed Kaggle Training Notebook
=============================================
Paste each CELL into a separate Kaggle code cell.

BUGS FIXED (vs phasecd/scripts/train_phasec.py):
  1. dtype= → use the kwarg name, confirmed working in transformers 5.x via **kwargs pop
  2. No DDP launch → replaced with accelerate-based launch that uses BOTH T4s
  3. device_map='auto' conflicts with DDP → removed; Trainer handles placement
  4. dataset only 89 rows (stale) → regenerate before training
  5. steps_per_epoch divides by device_count BEFORE DDP starts → wrong; Trainer does this
  6. max_steps=8000 with 89 rows = training ends in ~2 min → scaled to dataset size
  7. No accelerate config → write it explicitly so both T4s are used
  8. dataloader_num_workers=0 on Kaggle → set to 2 (2 CPU cores per GPU available)
  9. gradient_checkpointing_kwargs not set → incompatibility with some PEFT versions fixed
 10. load_best_model_at_end=True with save_strategy≠eval_strategy → aligned
"""

# =============================================================================
# CELL 1 — Install (run once, then restart kernel)
# =============================================================================
"""
!pip install -q \
    git+https://github.com/AmSach/kernelweave.git \
    "transformers>=4.40.0" \
    "trl>=0.8.0" \
    "peft>=0.10.0" \
    "accelerate>=0.29.0" \
    "datasets>=2.18.0"
"""

# =============================================================================
# CELL 2 — Regenerate dataset (fixes the 89-row stale data problem)
# =============================================================================
"""
import subprocess, sys, os
from pathlib import Path

REPO = Path("/kaggle/working/kernelweave")  # adjust if cloned elsewhere

result = subprocess.run(
    [sys.executable, str(REPO / "phasecd/scripts/generate_dataset.py"),
     "--store", str(REPO / "phasecd/store"),
     "--output-dir", str(REPO / "phasecd/data")],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
"""

# =============================================================================
# CELL 3 — Write accelerate config for 2x T4
#           This is the key fix for GPU under-utilisation.
#           Without this, Trainer uses only GPU 0.
# =============================================================================
"""
import json
from pathlib import Path

accelerate_cfg = {
    "compute_environment": "LOCAL_MACHINE",
    "distributed_type": "MULTI_GPU",
    "downcast_bf16": "no",
    "gpu_ids": "all",
    "machine_rank": 0,
    "main_training_function": "main",
    "mixed_precision": "fp16",
    "num_machines": 1,
    "num_processes": 2,          # <-- both T4s
    "rdzv_backend": "static",
    "same_network": True,
    "tpu_env": [],
    "tpu_use_cluster": False,
    "tpu_use_sudo": False,
    "use_cpu": False,
}

cfg_path = Path("/root/.cache/huggingface/accelerate/default_config.yaml")
cfg_path.parent.mkdir(parents=True, exist_ok=True)

# Write as YAML
import yaml  # pip install pyyaml - already available on Kaggle
cfg_path.write_text(yaml.dump(accelerate_cfg))
print(f"Accelerate config written to {cfg_path}")
print(yaml.dump(accelerate_cfg))
"""

# =============================================================================
# CELL 4 — Training script written to disk, then launched with accelerate
#           We write it as a file so accelerate launch can spawn 2 processes.
# =============================================================================

TRAIN_SCRIPT = '''
#!/usr/bin/env python3
"""KernelWeave Phase C/D training — fixed for 2x T4 Kaggle."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Config — all overrideable via environment variables
# ---------------------------------------------------------------------------
BASE_MODEL   = os.environ.get("KW_BASE_MODEL",   "Qwen/Qwen2.5-1.5B-Instruct")
DATA_DIR     = Path(os.environ.get("KW_DATA_DIR",    "/kaggle/working/kernelweave/phasecd/data"))
OUTPUT_DIR   = Path(os.environ.get("KW_OUTPUT_DIR",  "/kaggle/working/kw-output"))
STORE_DIR    = Path(os.environ.get("KW_STORE_DIR",   "/kaggle/working/kernelweave/phasecd/store"))
LR           = float(os.environ.get("KW_LR",          "4e-4"))
EPOCHS       = int(os.environ.get("KW_EPOCHS",        "3"))
MAX_LEN      = int(os.environ.get("KW_MAX_LEN",       "384"))
BATCH        = int(os.environ.get("KW_BATCH",         "2"))   # per-device; 2 GPUs → eff=4
GRAD_ACCUM   = int(os.environ.get("KW_GRAD_ACCUM",    "4"))   # eff batch = 2*2*4 = 16
LORA_R       = int(os.environ.get("KW_LORA_R",        "16"))
LORA_ALPHA   = int(os.environ.get("KW_LORA_ALPHA",    "32"))
SEED         = int(os.environ.get("KW_SEED",          "42"))

set_seed(SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_text(tokenizer, prompt: str, response: str) -> str:
    """Format a sample using the model's native chat template."""
    messages = [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return f"### User:\\n{prompt}\\n\\n### Assistant:\\n{response}"


def load_dataset_split(data_dir: Path, tokenizer, max_len: int):
    train_path = data_dir / "train.jsonl"
    eval_path  = data_dir / "eval.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. "
            "Run: python phasecd/scripts/generate_dataset.py --store phasecd/store --output-dir phasecd/data"
        )

    def _tokenize(example):
        text = build_text(tokenizer, example["prompt"], example["response"])
        return tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,   # collator handles padding
        )

    train = load_dataset("json", data_files=str(train_path), split="train")
    eval_ = load_dataset("json", data_files=str(eval_path),  split="train")

    train = train.map(_tokenize, remove_columns=train.column_names, num_proc=1)
    eval_ = eval_.map(_tokenize, remove_columns=eval_.column_names, num_proc=1)
    return train, eval_


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(base_model: str):
    use_cuda = torch.cuda.is_available()
    n_gpu    = torch.cuda.device_count() if use_cuda else 0

    print(f"CUDA available: {use_cuda}, GPUs: {n_gpu}")
    for i in range(n_gpu):
        gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({gb:.1f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # FIX: dtype= is accepted via **kwargs in transformers 5.x (popped before model init)
    # Both 'dtype' and legacy 'torch_dtype' work.  Use dtype= for clarity.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=torch.float16 if use_cuda else torch.float32,
        low_cpu_mem_usage=True,
        # FIX: do NOT set device_map here — Trainer/DDP handles device placement.
        # device_map="auto" conflicts with DDP and causes cross-device tensor errors.
    )

    # Required for gradient checkpointing + PEFT compatibility
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}  # avoids deprecation warn
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.max_steps:
            pct = 100.0 * state.global_step / state.max_steps
            loss = logs.get("loss", "?")
            print(f"[{pct:5.1f}%] step {state.global_step}/{state.max_steps}  loss={loss}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        ep   = float(state.epoch or 0)
        tot  = float(args.num_train_epochs or 1)
        print(f"[epoch {ep:.2f}/{tot:.2f}] complete", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)

    model, tokenizer = load_model(BASE_MODEL)
    train_ds, eval_ds = load_dataset_split(DATA_DIR, tokenizer, MAX_LEN)

    print(f"Train rows : {len(train_ds)}")
    print(f"Eval rows  : {len(eval_ds)}")
    print(f"Base model : {BASE_MODEL}")
    print(f"LR={LR}  epochs={EPOCHS}  batch={BATCH}  grad_accum={GRAD_ACCUM}  lora_r={LORA_R}")

    # FIX: compute steps from dataset size, not a fixed cap.
    # Trainer already divides by num_processes in DDP — pass raw per-device rows.
    n_gpu = max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(train_ds) / BATCH)   # per-device; Trainer multiplies
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = max(10, int(total_steps * 0.06))
    eval_save_steps = max(20, total_steps // 8)

    print(f"steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Logging / saving
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,

        # Evaluation
        eval_strategy="steps",      # FIX: was evaluation_strategy (deprecated)
        eval_steps=eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision & memory
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",

        # Multi-GPU — Trainer uses all GPUs visible; accelerate launch sets CUDA_VISIBLE_DEVICES
        dataloader_num_workers=2,   # FIX: was 0; 2 workers per GPU significantly improves throughput
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=[],
        run_name="kernelweave-phasec",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[ProgressCallback()],
    )

    trainer.train()

    # Save best adapter
    best_dir = OUTPUT_DIR / "best_adapter"
    best_dir.mkdir(exist_ok=True)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\\n✓ Adapter saved to {best_dir}")

    # Benchmark against kernel store
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from kernelweave.kernel import KernelStore
        from kernelweave.kernels import install_kernel_library
        from kernelweave.runtime import KernelRuntime

        store = KernelStore(STORE_DIR)
        if not store.list_kernels():
            install_kernel_library(store)
        runtime = KernelRuntime(store)
        prompts = [
            "compare two artifacts and summarize the differences",
            "find all TODOs in a codebase",
            "convert JSON to YAML without losing structure",
        ]
        results = [runtime.run(p) for p in prompts]
        payload = {"prompts": prompts, "results": results, "store_summary": store.summary()}
        (OUTPUT_DIR / "benchmark.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
        print("✓ Benchmark written to", OUTPUT_DIR / "benchmark.json")
    except Exception as e:
        print(f"Benchmark skipped: {e}")

    print("\\n✓ Training complete.")
    print(f"  Best adapter : {best_dir}")
    print(f"  Checkpoints  : {OUTPUT_DIR / 'checkpoints'}")


if __name__ == "__main__":
    main()
'''

# =============================================================================
# CELL 5 — Write the script and launch with accelerate for both T4s
# =============================================================================
"""
from pathlib import Path

script_path = Path("/kaggle/working/kw_train.py")
script_path.write_text(TRAIN_SCRIPT)
print(f"Script written to {script_path}")

# Launch with accelerate — this spawns 2 processes, one per T4
# Environment vars let you tweak without editing the script
import subprocess, sys, os

env = os.environ.copy()
env.update({
    "KW_BASE_MODEL":  "Qwen/Qwen2.5-1.5B-Instruct",
    "KW_DATA_DIR":    "/kaggle/working/kernelweave/phasecd/data",
    "KW_OUTPUT_DIR":  "/kaggle/working/kw-output",
    "KW_STORE_DIR":   "/kaggle/working/kernelweave/phasecd/store",
    "KW_EPOCHS":      "3",
    "KW_BATCH":       "2",     # per GPU
    "KW_GRAD_ACCUM":  "4",     # effective batch = 2 GPUs × 2 × 4 = 16
    "KW_LR":          "4e-4",
    "KW_LORA_R":      "16",
    "KW_MAX_LEN":     "384",
})

result = subprocess.run(
    ["accelerate", "launch",
     "--num_processes", "2",       # both T4s
     "--mixed_precision", "fp16",
     str(script_path)],
    env=env,
)
print("Exit code:", result.returncode)
"""

# =============================================================================
# CELL 6 — Quick inference test after training
# =============================================================================
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path("/kaggle/working/kw-output/best_adapter")

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
model = model.merge_and_unload()
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

for prompt in [
    "Compare main.py and utils.py and summarize the differences.",
    "Find all TODO comments in a Python codebase.",
    "Convert this JSON to YAML: {\"name\": \"alice\", \"age\": 30}",
]:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt")
    if torch.cuda.is_available():
        ids = {k: v.cuda() for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=200, temperature=0.7, do_sample=True,
                             pad_token_id=tok.eos_token_id)
    response = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Prompt : {prompt}")
    print(f"Response: {response[:300]}")
    print()
"""
