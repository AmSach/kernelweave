"""Kernel-native training."""

import os

if (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")) and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .complete import TrainingConfig, TrainingSample, TraceGenerator, KaggleTrainer, train_kernel_native, _ensure_deps, TRAINING_DEPS
from .hardware import detect_hardware, auto_train, HardwareProfile, apply_hardware_profile, _resolve_base_model

__all__ = [
    "TrainingConfig",
    "TrainingSample",
    "TraceGenerator",
    "KaggleTrainer",
    "train_kernel_native",
    "auto_train",
    "detect_hardware",
    "HardwareProfile",
    "apply_hardware_profile",
    "TRAINING_DEPS",
    "_ensure_deps",
]
