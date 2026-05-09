#!/usr/bin/env bash
# KernelWeave — Kaggle 2x T4 launch script
# Run this from /kaggle/working/kernelweave
#
# Usage:
#   chmod +x phasecd/scripts/kaggle_launch.sh
#   ./phasecd/scripts/kaggle_launch.sh
#
# Or with custom settings:
#   KW_EPOCHS=5 KW_LR=3e-4 ./phasecd/scripts/kaggle_launch.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "Repo root: $REPO_ROOT"

# Step 1 — regenerate dataset from kernel store
echo ""
echo "=== Regenerating dataset ==="
python3 "$REPO_ROOT/phasecd/scripts/generate_dataset.py" \
    --store "$REPO_ROOT/phasecd/store" \
    --output-dir "$REPO_ROOT/phasecd/data"

# Step 2 — launch with accelerate for 2x T4
echo ""
echo "=== Launching training on 2x T4 ==="
export KW_BASE_MODEL="${KW_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export KW_DATA_DIR="${KW_DATA_DIR:-$REPO_ROOT/phasecd/data}"
export KW_OUTPUT_DIR="${KW_OUTPUT_DIR:-/kaggle/working/kw-output}"
export KW_STORE_DIR="${KW_STORE_DIR:-$REPO_ROOT/phasecd/store}"
export KW_EPOCHS="${KW_EPOCHS:-3}"
export KW_BATCH="${KW_BATCH:-2}"         # per GPU; 2 GPUs x 2 x grad_accum=4 → eff batch=16
export KW_GRAD_ACCUM="${KW_GRAD_ACCUM:-4}"
export KW_LR="${KW_LR:-4e-4}"
export KW_LORA_R="${KW_LORA_R:-16}"
export KW_MAX_LEN="${KW_MAX_LEN:-384}"
export KERNELWEAVE_CORE_REPO="$REPO_ROOT"

accelerate launch \
    --num_processes 2 \
    --mixed_precision fp16 \
    "$REPO_ROOT/phasecd/scripts/train_phasec.py" \
    --data-dir "$KW_DATA_DIR" \
    --output-dir "$KW_OUTPUT_DIR" \
    --store-dir "$KW_STORE_DIR" \
    --base-model "$KW_BASE_MODEL" \
    --epochs "$KW_EPOCHS" \
    --batch-size "$KW_BATCH" \
    --grad-accum "$KW_GRAD_ACCUM" \
    --learning-rate "$KW_LR" \
    --max-len "$KW_MAX_LEN"

echo ""
echo "=== Done. Model saved to $KW_OUTPUT_DIR/best_adapter ==="
