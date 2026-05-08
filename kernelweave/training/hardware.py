"""Auto-detect hardware and optimize training parameters."""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

# Import TrainingConfig for use in functions
from .complete import TrainingConfig


@dataclass
class HardwareProfile:
    """Detected hardware profile."""
    gpu_name: str
    gpu_count: int
    vram_per_gpu_gb: float
    total_vram_gb: float
    compute_capability: str
    
    # Optimal training settings
    batch_size: int
    gradient_accumulation: int
    lora_r: int
    max_seq_length: int
    quantization: str  # "4bit", "8bit", "fp16"
    gradient_checkpointing: bool
    
    def to_dict(self):
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
        }


def detect_hardware() -> HardwareProfile:
    """Auto-detect GPU hardware and return optimal settings.
    
    Supports:
    - Kaggle T4 (2x) - Free tier
    - Kaggle P100
    - Colab T4
    - Colab A100
    - Local RTX 3090/4090
    - Local A100/H100
    """
    
    # Check if CUDA available
    try:
        import torch
        if not torch.cuda.is_available():
            return _get_cpu_profile()
    except ImportError:
        return _get_cpu_profile()
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    
    # Get VRAM
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    compute_cap = f"{props.major}.{props.minor}"
    
    # Determine optimal settings based on GPU
    profile = _get_gpu_profile(gpu_name, gpu_count, vram_gb, compute_cap)
    
    print(f"\n{'='*60}")
    print(f"AUTO-DETECTED HARDWARE")
    print(f"{'='*60}")
    print(f"  GPU:          {gpu_name}")
    print(f"  Count:        {gpu_count}")
    print(f"  VRAM/GPU:     {vram_gb:.1f} GB")
    print(f"  Total VRAM:   {vram_gb * gpu_count:.1f} GB")
    print(f"  Compute:      {compute_cap}")
    print(f"\nOPTIMIZED SETTINGS")
    print(f"{'='*60}")
    print(f"  Batch size:             {profile.batch_size}")
    print(f"  Gradient accumulation:  {profile.gradient_accumulation}")
    print(f"  LoRA rank:              {profile.lora_r}")
    print(f"  Max sequence length:    {profile.max_seq_length}")
    print(f"  Quantization:           {profile.quantization}")
    print(f"  Gradient checkpointing: {profile.gradient_checkpointing}")
    print(f"{'='*60}\n")
    
    return profile


def _get_gpu_profile(gpu_name: str, gpu_count: int, vram_gb: float, compute_cap: str) -> HardwareProfile:
    """Get optimal profile for specific GPU."""
    
    gpu_lower = gpu_name.lower()
    total_vram = vram_gb * gpu_count
    
    # NVIDIA A100 (40GB/80GB)
    if "a100" in gpu_lower:
        if vram_gb >= 70:  # A100 80GB
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                total_vram_gb=total_vram,
                compute_capability=compute_cap,
                batch_size=8,
                gradient_accumulation=2,
                lora_r=64,
                max_seq_length=2048,
                quantization="fp16",  # No quantization needed
                gradient_checkpointing=False,  # Not needed with 80GB
            )
        else:  # A100 40GB
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                total_vram_gb=total_vram,
                compute_capability=compute_cap,
                batch_size=6,
                gradient_accumulation=2,
                lora_r=32,
                max_seq_length=1536,
                quantization="fp16",
                gradient_checkpointing=False,
            )
    
    # NVIDIA H100
    if "h100" in gpu_lower:
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vram_per_gpu_gb=vram_gb,
            total_vram_gb=total_vram,
            compute_capability=compute_cap,
            batch_size=8,
            gradient_accumulation=2,
            lora_r=64,
            max_seq_length=2048,
            quantization="fp16",
            gradient_checkpointing=False,
        )
    
    # NVIDIA V100 (16GB/32GB)
    if "v100" in gpu_lower:
        if vram_gb >= 30:  # V100 32GB
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                total_vram_gb=total_vram,
                compute_capability=compute_cap,
                batch_size=4,
                gradient_accumulation=4,
                lora_r=24,
                max_seq_length=1024,
                quantization="fp16",
                gradient_checkpointing=False,
            )
        else:  # V100 16GB
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                total_vram_gb=total_vram,
                compute_capability=compute_cap,
                batch_size=2,
                gradient_accumulation=8,
                lora_r=16,
                max_seq_length=768,
                quantization="fp16",
                gradient_checkpointing=True,
            )
    
    # NVIDIA P100 (Kaggle)
    if "p100" in gpu_lower:
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vram_per_gpu_gb=vram_gb,
            total_vram_gb=total_vram,
            compute_capability=compute_cap,
            batch_size=4,
            gradient_accumulation=4,
            lora_r=16,
            max_seq_length=768,
            quantization="fp16",
            gradient_checkpointing=True,
        )
    
    # NVIDIA T4 (Kaggle free tier - usually 2x T4)
    if "t4" in gpu_lower:
        if gpu_count >= 2:
            # Kaggle gives 2x T4 (total 30GB)
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                total_vram_gb=total_vram,
                compute_capability=compute_cap,
                batch_size=4,
                gradient_accumulation=4,
                lora_r=16,
                max_seq_length=768,
                quantization="4bit",
                gradient_checkpointing=True,
            )
        else:
            # Single T4 (Colab free tier)
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                total_vram_gb=total_vram,
                compute_capability=compute_cap,
                batch_size=2,
                gradient_accumulation=8,
                lora_r=8,
                max_seq_length=512,
                quantization="4bit",
                gradient_checkpointing=True,
            )
    
    # NVIDIA RTX 3090 (24GB)
    if "3090" in gpu_lower:
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vram_per_gpu_gb=vram_gb,
            total_vram_gb=total_vram,
            compute_capability=compute_cap,
            batch_size=4,
            gradient_accumulation=4,
            lora_r=24,
            max_seq_length=1024,
            quantization="8bit",
            gradient_checkpointing=True,
        )
    
    # NVIDIA RTX 4090 (24GB)
    if "4090" in gpu_lower:
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vram_per_gpu_gb=vram_gb,
            total_vram_gb=total_vram,
            compute_capability=compute_cap,
            batch_size=4,
            gradient_accumulation=4,
            lora_r=32,
            max_seq_length=1024,
            quantization="fp16",  # Can do fp16 with 24GB
            gradient_checkpointing=True,
        )
    
    # NVIDIA L4 (24GB - newer GPU)
    if "l4" in gpu_lower:
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vram_per_gpu_gb=vram_gb,
            total_vram_gb=total_vram,
            compute_capability=compute_cap,
            batch_size=4,
            gradient_accumulation=4,
            lora_r=24,
            max_seq_length=1024,
            quantization="8bit",
            gradient_checkpointing=True,
        )
    
    # A10G (24GB - AWS)
    if "a10" in gpu_lower:
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vram_per_gpu_gb=vram_gb,
            total_vram_gb=total_vram,
            compute_capability=compute_cap,
            batch_size=4,
            gradient_accumulation=4,
            lora_r=24,
            max_seq_length=1024,
            quantization="8bit",
            gradient_checkpointing=True,
        )
    
    # Default: Conservative settings for unknown GPU
    return HardwareProfile(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        vram_per_gpu_gb=vram_gb,
        total_vram_gb=total_vram,
        compute_capability=compute_cap,
        batch_size=2,
        gradient_accumulation=8,
        lora_r=8,
        max_seq_length=512,
        quantization="4bit",
        gradient_checkpointing=True,
    )


def _get_cpu_profile() -> HardwareProfile:
    """Get profile for CPU-only (fallback)."""
    print("\n⚠ No GPU detected - using CPU (training will be very slow)")
    
    return HardwareProfile(
        gpu_name="CPU",
        gpu_count=0,
        vram_per_gpu_gb=0,
        total_vram_gb=0,
        compute_capability="0.0",
        batch_size=1,
        gradient_accumulation=16,
        lora_r=4,
        max_seq_length=256,
        quantization="8bit",
        gradient_checkpointing=True,
    )


def apply_hardware_profile(config: 'TrainingConfig', profile: HardwareProfile) -> 'TrainingConfig':
    """Apply hardware profile to training config."""
    config.batch_size = profile.batch_size
    config.gradient_accumulation_steps = profile.gradient_accumulation
    config.lora_r = profile.lora_r
    config.max_input_length = profile.max_seq_length // 2
    config.max_output_length = profile.max_seq_length // 2
    
    return config


# Convenience function for auto-training
def auto_train(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "./kernel-native-model",
    n_samples: int = 5000,
    epochs: int = 3,
    **kwargs,
) -> 'KaggleTrainer':
    """Auto-detect hardware and train with optimal settings."""
    # Import here to avoid circular dependency
    from .complete import KaggleTrainer
    
    # Detect hardware
    profile = detect_hardware()
    
    # Create config with optimal settings
    config = TrainingConfig(
        base_model=base_model,
        output_dir=output_dir,
        batch_size=profile.batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation,
        lora_r=profile.lora_r,
        max_input_length=profile.max_seq_length // 2,
        max_output_length=profile.max_seq_length // 2,
        **kwargs,
    )
    
    # Create trainer
    trainer = KaggleTrainer(
        base_model=base_model,
        output_dir=output_dir,
        config=config,
    )
    
    # Generate data
    trainer.generate_training_data(n_samples=n_samples)
    
    # Setup model with appropriate quantization
    trainer.setup_model(quantization=profile.quantization)
    
    # Train
    trainer.train(epochs=epochs, batch_size=config.batch_size)
    
    # Save
    trainer.save_model()
    
    return trainer


# Backwards compatibility
def train_kernel_native(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "./kernel-native-model",
    n_samples: int = 5000,
    epochs: int = 3,
    batch_size: Optional[int] = None,
    **kwargs,
) -> 'KaggleTrainer':
    """Train a kernel-native model with auto-detection."""
    # Import here to avoid circular dependency
    from .complete import KaggleTrainer
    
    if batch_size is not None:
        # Manual override
        profile = detect_hardware()
        profile.batch_size = batch_size
        
        config = TrainingConfig(
            base_model=base_model,
            output_dir=output_dir,
            batch_size=batch_size,
            **kwargs,
        )
        
        trainer = KaggleTrainer(
            base_model=base_model,
            output_dir=output_dir,
            config=config,
        )
        
        trainer.generate_training_data(n_samples=n_samples)
        trainer.setup_model(quantization=profile.quantization)
        trainer.train(epochs=epochs, batch_size=batch_size)
        trainer.save_model()
        
        return trainer
    else:
        # Auto-detect
        return auto_train(
            base_model=base_model,
            output_dir=output_dir,
            n_samples=n_samples,
            epochs=epochs,
            **kwargs,
        )