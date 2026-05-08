#!/usr/bin/env python3
"""
Kaggle Setup Script - Run this first in Kaggle notebook.

Usage in Kaggle:
    !python kaggle_setup.py
    
Or copy-paste the cells below into Kaggle notebook.
"""

print("""
================================================================================
                    KERNELWEAVE KAGGLE SETUP
================================================================================

Step 1: Install KernelWeave
----------------------------
!pip install -q git+https://github.com/amsach/kernelweave.git


Step 2: Verify Installation
----------------------------
from kernelweave import train_kernel_native, TRAINING_DEPS
print("✓ KernelWeave installed")
print(f"Training dependencies: {TRAINING_DEPS}")


Step 3: Train Model (Zero External Data)
-----------------------------------------
from kernelweave import train_kernel_native

trainer = train_kernel_native(
    base_model="Qwen/Qwen2.5-7B-Instruct",  # or "meta-llama/Llama-3.2-3B-Instruct"
    output_dir="./kernel-native-model",
    n_samples=5000,  # Auto-generated from kernels
    epochs=3,
    batch_size=4,  # Works on T4 GPU
)

# Model saved to ./kernel-native-model/final_model


Step 4: Use Model
-----------------
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./kernel-native-model/final_model")
tokenizer = AutoTokenizer.from_pretrained("./kernel-native-model/final_model")

prompt = "Compare main.py and utils.py"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))


Step 5: Push to Hub (Optional)
-------------------------------
from huggingface_hub import login
login()  # Enter your HF token

trainer.push_to_hub("your-username/kernel-native-qwen-7b")


================================================================================
                    QUICK START - COPY PASTE
================================================================================

# Cell 1: Install
!pip install -q git+https://github.com/amsach/kernelweave.git transformers trl peft bitsandbytes accelerate

# Cell 2: Train
from kernelweave import train_kernel_native

trainer = train_kernel_native(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./kernel-native-model",
    n_samples=5000,
    epochs=3,
)

# Cell 3: Test
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./kernel-native-model/final_model")
tokenizer = AutoTokenizer.from_pretrained("./kernel-native-model/final_model")

prompt = "Compare main.py and utils.py and summarize the differences."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

================================================================================
""")

if __name__ == "__main__":
    # Auto-run setup
    import subprocess
    import sys
    
    print("\nInstalling KernelWeave...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/amsach/kernelweave.git"
    ])
    
    print("\nInstalling training dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q"
    ] + TRAINING_DEPS)
    
    print("\n✓ Setup complete!")
    print("\nNow run:")
    print("    from kernelweave import train_kernel_native")
    print("    trainer = train_kernel_native()")
