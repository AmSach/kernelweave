"""
KernelWeave Interactive Shell for Ollama
========================================

Launch this script to chat with KernelWeave attached to your local Ollama instance.
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

# Force UTF-8 on Windows
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure kernelweave is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kernelweave.kernel import KernelStore, TraceEvent
from kernelweave.runtime import ExecutionEngine, KernelRuntime
from kernelweave.compiler import compile_trace_to_kernel
from kernelweave.verifier import VerifierHierarchy
from kernelweave.llm.providers import ModelPreset, OllamaBackend

# ── Colors ─────────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"


def print_banner(model_name: str):
    print(f"\n{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD} 🕸️  KernelWeave Interactive Shell{RESET}")
    print(f" {DIM}Attached to Ollama ({model_name}){RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")
    print("Type your prompt to invoke the KernelWeave router and engine.")
    print("Type 'exit' or 'quit' to close. Type 'kernels' to list store.")
    print()


def make_backend(model: str) -> OllamaBackend:
    preset = ModelPreset(
        id="interactive-ollama",
        provider="ollama",
        model=model,
        base_url="http://127.0.0.1:11434",
        temperature=0.4,
        max_tokens=2048,
        timeout_seconds=120,
    )
    return OllamaBackend(preset, json_mode=False)


def check_ollama(backend: OllamaBackend) -> bool:
    print(f"{DIM}Checking Ollama connectivity...{RESET}", end="", flush=True)
    try:
        backend.generate("test", max_tokens=1)
        print(f"\r{GREEN}[OK]{RESET} Ollama is responding!    ")
        return True
    except Exception as e:
        print(f"\r{RED}[FAIL]{RESET} Cannot reach Ollama: {e}")
        print(f"\n{YELLOW}Please ensure Ollama is running (`ollama serve`) and the model exists.{RESET}\n")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KernelWeave Interactive Shell")
    parser.add_argument("--model", type=str, default="gemma4:e2b", help="Ollama model to use")
    parser.add_argument("--store", type=str, default="store", help="Path to KernelStore")
    args = parser.parse_args()

    backend = make_backend(args.model)
    
    print_banner(args.model)
    
    if not check_ollama(backend):
        sys.exit(1)
        
    store_path = Path(args.store)
    store = KernelStore(store_path)
    runtime = KernelRuntime(store, use_embeddings=False)
    engine = ExecutionEngine(store, backend)
    verifier = VerifierHierarchy(backend=backend)
    
    print(f"{DIM}Store loaded: {store.summary()}{RESET}\n")

    while True:
        try:
            prompt = input(f"\n{BOLD}{GREEN}User > {RESET}")
        except (KeyboardInterrupt, EOFError):
            print()
            break
            
        prompt = prompt.strip()
        if not prompt:
            continue
            
        if prompt.lower() in ("exit", "quit"):
            break
            
        if prompt.lower() == "kernels":
            kernels = store.list_kernels()
            print(f"\n{YELLOW}Kernels ({len(kernels)}):{RESET}")
            for k in kernels:
                print(f"  - {k['kernel_id']} [{k['task_family']}] (confidence: {k['confidence']:.2f})")
            continue
            
        # ── Simple File Reader Tool ────────────────────────────────
        # Detect files mentioned in the prompt and inject their content
        words = prompt.split()
        injected_content = ""
        for word in words:
            # Clean up word from common punctuation
            clean_word = word.strip(".,;:\"'?!()[]{}")
            if "." in clean_word and not clean_word.startswith("."):
                possible_path = Path(clean_word)
                if possible_path.exists() and possible_path.is_file():
                    try:
                        print(f"{DIM}Reading file: {clean_word}...{RESET}")
                        with open(possible_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        injected_content += f"\n\n=== Content of {clean_word} ===\n{content}\n"
                    except Exception as e:
                        print(f"{DIM}Failed to read {clean_word}: {e}{RESET}")
                        
        if injected_content:
            original_prompt = prompt
            prompt = f"The user is asking a question about these files. Answer the user prompt based on the file contents provided below.\n\nUser Prompt: {prompt}{injected_content}"
            print(f"{DIM}Injected file context into prompt.{RESET}")
            
        # 1. Routing
        print(f"{DIM}Routing...{RESET}", end="\r")
        t0 = time.time()
        plan = runtime.run(prompt)
        print(f"{DIM}[Router] Mode: {plan['mode']} | Kernel: {plan.get('kernel_id', 'none')} | Score: {plan.get('score', 0):.2f}{RESET}")

        # 2. Execution
        # Overwrite the routing line to clean up terminal
        print(" " * 80, end="\r")
        print(f"{DIM}Executing with {args.model}...{RESET}", end="\r")
        
        result = engine.execute_plan(plan, prompt)
        response_text = result.get("response_text", "")
        
        # The runtime engine defers raw generation when no kernel matches
        # We must perform the generation manually for the interactive shell fallback
        if not response_text and plan.get("mode") == "generate":
            print(f"{DIM}Falling back to raw generation...{RESET}", end="\r")
            try:
                resp = backend.generate(prompt)
                response_text = resp.text
            except Exception as e:
                response_text = f"Error during generation: {e}"
        
        # Clear the executing line
        print(" " * 80, end="\r")
        print(f"{BOLD}{MAGENTA}KernelWeave > {RESET}")
        print(f"{response_text}\n")
        
        # 3. Verification & Self-Compilation
        if response_text:
            print(f"{DIM}Verifying output...{RESET}", end="\r")
            postconditions = plan.get("postconditions", ["output addresses the prompt directly"])
            v_result = verifier.verify(response_text, postconditions, prompt=prompt)
            status_color = GREEN if v_result.passed else YELLOW
            status_text = "PASS" if v_result.passed else "WARN"
            print(f"{DIM}[Verifier] Status: {status_color}{status_text}{RESET}{DIM} | Score: {v_result.score:.2f} ({v_result.level}){RESET}")
            
            if plan['mode'] == 'generate' and len(response_text) > 20:
                # Self-compile
                print(f"{DIM}Self-compiling new kernel...{RESET}", end="\r")
                events = [
                    TraceEvent(kind="plan", payload={"text": prompt}),
                    TraceEvent(kind="execution", payload={"text": response_text[:200] + "..."}),
                    TraceEvent(kind="verification", payload={"text": f"completed generation with score {v_result.score:.2f}"}),
                ]
                
                # Try to guess a better task family if possible
                task_family = "general query"
                if "compare" in prompt.lower() or "diff" in prompt.lower():
                    task_family = "artifact comparison"
                elif "code" in prompt.lower() or "python" in prompt.lower() or "function" in prompt.lower():
                    task_family = "code generation"
                
                try:
                    new_kernel = compile_trace_to_kernel(
                        trace_id=f"trace-{int(time.time())}",
                        task_family=task_family,
                        description=prompt[:100],
                        events=events,
                        expected_output={"result": "generated text response"},
                        backend=backend,
                    )
                    store.add_kernel(new_kernel)
                    print(f"{DIM}[Compiler] New kernel saved: {new_kernel.kernel_id} [{task_family}] (confidence: {new_kernel.status.confidence:.2f}){RESET}")
                except Exception as e:
                    print(f"{DIM}[Compiler] Failed to self-compile: {e}{RESET}")


if __name__ == "__main__":
    main()
