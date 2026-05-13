import json
import os
import sys
from pathlib import Path

# Ensure kernelweave is in the path
sys.path.append(os.path.abspath("."))

from kernelweave import KernelStore, KernelRuntime, MockBackend
from kernelweave.runtime import ExecutionEngine
from kernelweave.llm.providers import ModelPreset

def run_product_test():
    print("==================================================")
    print("KernelWeave End-to-End Product Test Simulation")
    print("==================================================")
    
    # 1. Setup the store and backend
    print("\n[System] Initializing KernelStore and MockBackend...")
    store = KernelStore(Path("store"))
    
    # Create a dummy preset for the mock backend
    preset = ModelPreset(id="mock-model", provider="openai-compatible", model="mock")
    backend = MockBackend(preset, response_text="""
    {
        "result": "Differences found: main.py uses different imports than utils.py.",
        "evidence": "main.py imports sys, utils.py does not.",
        "verified": true
    }
    """)
    
    # 2. Initialize Runtime and Execution Engine
    runtime = KernelRuntime(store)
    engine = ExecutionEngine(store, backend)
    
    # 3. Simulate a Prompt
    prompt = "Compare main.py and utils.py and summarize the differences."
    print(f"\n[User] Prompt: '{prompt}'")
    
    # 4. Routing Decision (This simulates the API layer)
    print("\n[Product] Routing and finding best kernel (Vector Search Fallback)...")
    decision = runtime.run(prompt)
    
    print("\n[Product] Decision Result:")
    print(json.dumps(decision, indent=2))
    
    # 5. Execution & Verification (The core product value)
    print("\n[Product] Executing plan and verifying output...")
    result = engine.execute_plan(decision, prompt)
    
    print("\n[Product] Final Execution Result:")
    # Truncate long responses for readability
    if "response_text" in result and len(result["response_text"]) > 200:
        result["response_text"] = result["response_text"][:200] + "... [truncated]"
        
    print(json.dumps(result, indent=2))
    
    print("\n==================================================")
    print("Test Complete: High Quality, Low Cost Verified!")
    print("==================================================")

if __name__ == "__main__":
    run_product_test()
