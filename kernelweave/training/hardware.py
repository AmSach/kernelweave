from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from .complete import TrainingConfig


@dataclass
class HardwareProfile:
    gpu_name: str
    gpu_count: int
    vram_per_gpu_gb: float
    total_vram_gb: float
    compute_capability: str
    batch_size: int
    gradient_accumulation: int
    lora_r: int
    max_seq_length: int
    quantization: str
    gradient_checkpointing: bool
    safe_mode: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "vram_per_gpu_gb": self.vram_per_gpu_gb,
            "total_vram_gb": self.total_vram_gb,
            "compute_capability": self.compute_capability,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "lora_r": self.lora_r,
            "max_seq_length": self.max_seq_length,
            "quantization": self.quantization,
            "gradient_checkpointing": self.gradient_checkpointing,
            "safe_mode": self.safe_mode,
        }


def _resolve_base_model(base_model: str, profile: HardwareProfile) -> str:
    if profile.safe_mode:
        return os.environ.get("KERNELWEAVE_SAFE_BASE_MODEL", "KernelWeave/Synthetic")
    return base_model


def detect_hardware() -> HardwareProfile:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            compute_cap = f"{props.major}.{props.minor}"
            if vram_gb >= 24:
                batch_size = 4
                grad_accum = 2
                lora_r = 16
                max_seq = 1024
                grad_ckpt = True
            else:
                batch_size = 2
                grad_accum = 4
                lora_r = 8
                max_seq = 512
                grad_ckpt = True
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                total_vram_gb=vram_gb * gpu_count,
                compute_capability=compute_cap,
                batch_size=batch_size,
                gradient_accumulation=grad_accum,
                lora_r=lora_r,
                max_seq_length=max_seq,
                quantization="fp16",
                gradient_checkpointing=grad_ckpt,
                safe_mode=True,
            )
    except Exception:
        pass

    return HardwareProfile(
        gpu_name="CPU",
        gpu_count=0,
        vram_per_gpu_gb=0.0,
        total_vram_gb=0.0,
        compute_capability="0.0",
        batch_size=1,
        gradient_accumulation=16,
        lora_r=4,
        max_seq_length=256,
        quantization="fp16",
        gradient_checkpointing=True,
        safe_mode=True,
    )


def apply_hardware_profile(config: TrainingConfig, profile: HardwareProfile) -> TrainingConfig:
    config.batch_size = min(config.batch_size, profile.batch_size)
    config.gradient_accumulation_steps = max(config.gradient_accumulation_steps, profile.gradient_accumulation)
    config.lora_r = min(config.lora_r, profile.lora_r)
    config.max_input_length = min(config.max_input_length, profile.max_seq_length // 2)
    config.max_output_length = min(config.max_output_length, profile.max_seq_length // 2)
    return config


def auto_train(
    base_model: str = "KernelWeave/Synthetic",
    output_dir: str = "./kernel-native-model",
    n_samples: int = 5000,
    epochs: int = 3,
    **kwargs: Any,
) -> Any:
    from .complete import train_kernel_native
    return train_kernel_native(base_model=base_model, output_dir=output_dir, n_samples=n_samples, epochs=epochs, **kwargs)
