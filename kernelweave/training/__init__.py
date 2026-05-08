"""Kernel-native training: Fine-tune models on kernel execution traces.

REVOLUTIONARY: Self-contained training that generates its own data from kernels.

Quick Start (Kaggle):
    from kernelweave.training import train_kernel_native
    
    trainer = train_kernel_native(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        output_dir="./kernel-native-model",
        n_samples=5000,
        epochs=3,
    )
"""
from .complete import (
    TrainingConfig,
    TrainingSample,
    TraceGenerator,
    KaggleTrainer,
    _ensure_deps,
    TRAINING_DEPS,
)
from .hardware import (
    detect_hardware,
    auto_train,
    train_kernel_native,
    HardwareProfile,
    apply_hardware_profile,
)

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