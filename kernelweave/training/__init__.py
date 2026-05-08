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
    train_kernel_native,
    TRAINING_DEPS,
    _ensure_deps,
)

__all__ = [
    "TrainingConfig",
    "TrainingSample",
    "TraceGenerator",
    "KaggleTrainer",
    "train_kernel_native",
    "TRAINING_DEPS",
    "_ensure_deps",
]