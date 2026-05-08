#!/usr/bin/env python3
"""
STANDALONE TRAINING SCRIPT FOR KAGGLE
Copy-paste this entire file into a Kaggle cell - NO INSTALL NEEDED

Usage:
    %run standalone_train.py
    
Or copy-paste the entire content into a cell.
"""

import os
import sys
import random
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================

def ensure_deps():
    """Auto-install all required dependencies."""
    deps = [
        "torch",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.29.0",
        "trl>=0.8.0",
        "datasets",
    ]
    
    for dep in deps:
        try:
            __import__(dep.split(">=")[0].split("==")[0])
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], check=True)
    
    print("✓ All dependencies installed")

# ============================================================================
# HARDWARE DETECTION
# ============================================================================

@dataclass
class HardwareProfile:
    gpu_name: str
    gpu_count: int
    vram_per_gpu: float
    compute_capability: float
    
    batch_size: int
    gradient_accumulation: int
    lora_r: int
    max_seq_length: int
    use_4bit: bool
    use_gradient_checkpointing: bool

def detect_hardware() -> HardwareProfile:
    """Detect GPU and optimize settings."""
    ensure_deps()
    
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. This script requires CUDA.")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    
    # Get VRAM
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / (1024**3)
    compute = float(f"{props.major}.{props.minor}")
    
    # Optimize based on GPU
    if "A100" in gpu_name:
        # A100 40GB or 80GB
        batch_size = 8
        grad_accum = 2
        lora_r = 64
        max_seq = 2048
        use_4bit = False
        grad_ckpt = False
    elif "V100" in gpu_name:
        # V100 16GB
        batch_size = 4
        grad_accum = 4
        lora_r = 32
        max_seq = 1024
        use_4bit = True
        grad_ckpt = True
    elif "T4" in gpu_name:
        # T4 16GB (Kaggle free tier)
        batch_size = 4
        grad_accum = 4
        lora_r = 16
        max_seq = 768
        use_4bit = True
        grad_ckpt = True
    elif "P100" in gpu_name:
        # P100 16GB
        batch_size = 4
        grad_accum = 4
        lora_r = 16
        max_seq = 768
        use_4bit = True
        grad_ckpt = True
    else:
        # Unknown GPU - use conservative settings
        batch_size = 2
        grad_accum = 8
        lora_r = 16
        max_seq = 512
        use_4bit = True
        grad_ckpt = True
    
    return HardwareProfile(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        vram_per_gpu=vram,
        compute_capability=compute,
        batch_size=batch_size,
        gradient_accumulation=grad_accum,
        lora_r=lora_r,
        max_seq_length=max_seq,
        use_4bit=use_4bit,
        use_gradient_checkpointing=grad_ckpt,
    )

# ============================================================================
# TEMPLATE GENERATION (BULLETPROOF)
# ============================================================================

class TemplateGenerator:
    """Generate training samples from templates - ALL VARIABLES INCLUDED."""
    
    def __init__(self):
        self._task_templates = self._build_task_templates()
        self._response_templates = self._build_response_templates()
        self._all_variables = self._build_all_variables()
    
    def _build_task_templates(self) -> dict[str, list[str]]:
        """Task prompt templates."""
        return {
            "comparison": [
                "Compare {file_a} and {file_b} and summarize the differences.",
                "What are the key differences between {item_a} and {item_b}?",
            ],
            "analysis": [
                "Analyze {file} and identify potential {issue_type}.",
                "Review {code} for {issue_type}.",
            ],
            "search": [
                "Find all {file_type} files in {directory}.",
                "Search for {pattern} in all files.",
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
        }
    
    def _build_response_templates(self) -> dict[str, list[str]]:
        """Response templates - ONLY using safe variables."""
        return {
            "comparison": [
                "I found {n} key differences between {file_a} and {file_b}.\n\nOverall, these files differ in {aspect}.",
                "Comparing {file_a} and {file_b}:\n\nThe main differences are in {aspect}.",
            ],
            "analysis": [
                "Analysis of {file}:\n\n**Issues found:**\n1. {issue}\n\n**Recommendations:**\n- {recommendation}",
                "I identified {issue_type} in {file}.\n\nSeverity: {severity}",
            ],
            "search": [
                "Found {n} {file_type} files in {directory}.",
                "Search results for '{pattern}':\n\nTotal: {n} matches",
            ],
            "generation": [
                "# {title}\n\n{content}\n\n## Summary\n\n{summary}",
                "**Report:**\n\n{content}",
            ],
            "debugging": [
                "Fixed the bug in {file}:\n\n**Problem:** {error}\n\n**Solution:** {fix}",
                "Debugging complete.\n\n**Root cause:** {cause}\n\n**Fix applied:** {fix}",
            ],
            "summarization": [
                "**Summary of {document}:**\n\n{summary}\n\n**Key points:**\n1. {point}\n2. {another_point}",
                "Overview:\n\n{summary}\n\nMain findings: {findings}",
            ],
            "transformation": [
                "Transformation complete.\n\nConverted from {format_a} to {format_b}.",
                "```{format_b}\n{output}\n```",
            ],
            "testing": [
                "```python\n{test_code}\n```\n\nTests cover: {coverage}",
                "Generated {n} test cases.\n\nAll tests passing.",
            ],
        }
    
    def _build_all_variables(self) -> dict[str, list[str]]:
        """Complete variable pools for ALL templates."""
        return {
            # Files and directories
            "files": ["main.py", "utils.py", "config.yaml", "app.js", "index.ts", "README.md", "test.py"],
            "directories": ["src", "lib", "tests", "docs", "config", "scripts"],
            "file_types": ["Python", "JavaScript", "TypeScript", "config", "test", "documentation"],
            "formats": ["JSON", "YAML", "Markdown", "HTML", "CSV", "XML"],
            "patterns": ["TODO", "FIXME", "deprecated", "unused", "import", "function"],
            "issues": ["bugs", "security issues", "performance problems", "style issues", "unused imports"],
            "errors": ["TypeError", "ValueError", "ImportError", "AttributeError", "SyntaxError"],
            
            # Aspects and features
            "aspects": ["error handling", "performance", "code style", "architecture", "documentation", "testing"],
            "features": ["async support", "type hints", "comprehensive tests", "detailed docs", "modular design"],
            "severities": ["low", "medium", "high", "critical"],
            
            # Code elements
            "codes": ["def process(data): return data", "class Handler: pass", "async def fetch(): pass"],
            "modules": ["main", "utils", "handler", "processor", "converter"],
            "functions": ["process_data", "validate_input", "transform_output", "handle_request", "parse_config"],
            
            # Content
            "titles": ["Analysis Report", "Code Review Summary", "Performance Audit", "Security Assessment"],
            "contents": ["This document provides a comprehensive analysis...", "The following report details the key findings..."],
            "summaries": ["Key findings indicate improvements needed.", "Overall, the codebase is well-structured."],
            "points": ["Improved performance", "Enhanced security", "Better maintainability", "Clearer documentation"],
            "findings": ["improved code quality", "enhanced security", "better performance"],
            
            # Debugging
            "fixes": ["added null check", "fixed variable name", "added missing import", "corrected signature"],
            "causes": ["null reference", "naming inconsistency", "missing import", "type mismatch"],
            
            # Documents
            "documents": ["the technical report", "the codebase documentation", "the project README", "the architecture overview"],
            
            # Transformation
            "outputs": ['{"result": "success"}', "name: value\nother: data", "# Result\n\nSuccess"],
            
            # Testing
            "test_codes": ["def test_function(): assert True", "def test_edge_case(): pass"],
            "coverages": ["85%", "92%", "78%", "95%", "88%"],
            
            # Issues and recommendations
            "issue_texts": ["unused variable", "missing type hint", "potential null pointer", "deprecated API"],
            "rec_texts": ["add type hints", "implement error handling", "refactor code", "add tests"],
        }
    
    def _generate_vars(self) -> dict[str, str]:
        """Generate a complete set of variables - GUARANTEED to have ALL needed keys."""
        v = self._all_variables
        n = str(random.randint(1, 5))
        
        return {
            # Basic
            "file": random.choice(v["files"]),
            "file_a": random.choice(v["files"]),
            "file_b": random.choice(v["files"]),
            "item_a": random.choice(v["files"]),
            "item_b": random.choice(v["files"]),
            "directory": random.choice(v["directories"]),
            "file_type": random.choice(v["file_types"]),
            "format": random.choice(v["formats"]),
            "format_a": random.choice(v["formats"]),
            "format_b": random.choice(v["formats"]),
            "pattern": random.choice(v["patterns"]),
            "issue_type": random.choice(v["issues"]),
            "error": random.choice(v["errors"]),
            "n": n,
            
            # Code
            "code": random.choice(v["codes"]),
            "module": random.choice(v["modules"]),
            "function": random.choice(v["functions"]),
            
            # Content
            "title": random.choice(v["titles"]),
            "content": random.choice(v["contents"]),
            "summary": random.choice(v["summaries"]),
            "point": random.choice(v["points"]),
            "another_point": random.choice(v["points"]),
            "findings": random.choice(v["findings"]),
            
            # Aspects
            "aspect": random.choice(v["aspects"]),
            "severity": random.choice(v["severities"]),
            
            # Debugging
            "fix": random.choice(v["fixes"]),
            "cause": random.choice(v["causes"]),
            
            # Documents
            "document": random.choice(v["documents"]),
            
            # Transformation
            "input": random.choice(v["files"]),
            "output": random.choice(v["outputs"]),
            
            # Testing
            "test_code": random.choice(v["test_codes"]),
            "coverage": random.choice(v["coverages"]),
            
            # Issues
            "issue": random.choice(v["issue_texts"]),
            "recommendation": random.choice(v["rec_texts"]),
            
            # Source
            "source": random.choice(v["files"]),
            "topic": random.choice(v["aspects"]),
        }
    
    def generate_samples(self, n_samples: int = 100) -> list[dict[str, str]]:
        """Generate training samples."""
        samples = []
        families = list(self._task_templates.keys())
        
        for _ in range(n_samples):
            family = random.choice(families)
            vars_dict = self._generate_vars()
            
            # Generate prompt
            prompt_template = random.choice(self._task_templates[family])
            prompt = prompt_template.format(**vars_dict)
            
            # Generate response
            response_template = random.choice(self._response_templates[family])
            response = response_template.format(**vars_dict)
            
            samples.append({
                "prompt": prompt,
                "response": response,
                "family": family,
            })
        
        return samples

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class KernelTrainer:
    """Complete training pipeline."""
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "./kernel-native-model",
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.hardware = detect_hardware()
        
        self._model = None
        self._tokenizer = None
        self._trainer = None
        
        self._print_hardware_info()
    
    def _print_hardware_info(self):
        """Print hardware profile."""
        print("=" * 60)
        print("AUTO-DETECTED HARDWARE")
        print("=" * 60)
        print(f"  GPU:          {self.hardware.gpu_name}")
        print(f"  Count:        {self.hardware.gpu_count}")
        print(f"  VRAM/GPU:     {self.hardware.vram_per_gpu:.1f} GB")
        print(f"  Compute:      {self.hardware.compute_capability}")
        print()
        print("OPTIMIZED SETTINGS")
        print("=" * 60)
        print(f"  Batch size:             {self.hardware.batch_size}")
        print(f"  Gradient accumulation:  {self.hardware.gradient_accumulation}")
        print(f"  LoRA rank:              {self.hardware.lora_r}")
        print(f"  Max sequence length:    {self.hardware.max_seq_length}")
        print(f"  Quantization:           {'4bit' if self.hardware.use_4bit else 'fp16'}")
        print(f"  Gradient checkpointing: {self.hardware.use_gradient_checkpointing}")
        print("=" * 60)
        print()
    
    def setup_model(self):
        """Load model with optimized settings."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        
        print(f"Loading model: {self.base_model}")
        
        # Quantization config
        if self.hardware.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # max_length is NOT a valid param here - removed
        )
        
        # Gradient checkpointing
        if self.hardware.use_gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
        
        # LoRA
        lora_config = LoraConfig(
            r=self.hardware.lora_r,
            lora_alpha=self.hardware.lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()
        
        print("✓ Model loaded")
    
    def generate_data(self, n_samples: int = 5000):
        """Generate training data."""
        print(f"\nGenerating {n_samples} training samples...")
        
        generator = TemplateGenerator()
        samples = generator.generate_samples(n_samples)
        
        # Format for Qwen chat
        formatted_data = []
        for sample in samples:
            text = f"