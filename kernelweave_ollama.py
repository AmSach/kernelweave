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

# ── Tools for ReAct Loop ───────────────────────────────────────────
def tool_list_dir(path="."):
    import os
    try:
        return json.dumps(os.listdir(path))
    except Exception as e:
        return str(e)

def tool_read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return str(e)

def tool_write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "File written successfully."
    except Exception as e:
        return str(e)

def tool_run_command(command):
    import subprocess
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return str(e)

def tool_register_tool(name, code):
    import os
    from pathlib import Path
    import importlib.util
    
    try:
        tools_dir = Path("custom_tools")
        tools_dir.mkdir(exist_ok=True)
        
        file_path = tools_dir / f"{name}.py"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for the function by name or fallback to 'execute'
        if hasattr(module, name):
            func = getattr(module, name)
        elif hasattr(module, "execute"):
            func = getattr(module, "execute")
        else:
            return f"Error: Module must contain a function named '{name}' or 'execute'."
            
        # Add to global TOOLS dictionary
        TOOLS[name] = func
        return f"Tool '{name}' registered successfully and is now available in this session!"
    except Exception as e:
        return f"Failed to register tool: {e}"

TOOLS = {
    "list_dir": tool_list_dir,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_command": tool_run_command,
    "register_tool": tool_register_tool
}


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


def make_backend(provider: str, model: str, base_url: str) -> Any:
    preset = ModelPreset(
        id="interactive-session",
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=0.4,
        max_tokens=2048,
        timeout_seconds=30,
    )
    if provider == "ollama":
        return OllamaBackend(preset, json_mode=False)
    else:
        return OpenAICompatibleBackend(preset)

def check_connectivity(backend: Any) -> bool:
    try:
        backend.generate("test", max_tokens=1)
        return True
    except Exception:
        return False

def get_ollama_models(url="http://127.0.0.1:11434"):
    import urllib.request
    import json
    try:
        req = urllib.request.Request(f"{url}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))
            return [m['name'] for m in data.get('models', [])]
    except Exception:
        return []

def get_openai_models(url="http://127.0.0.1:1234/v1"):
    import urllib.request
    import json
    try:
        req = urllib.request.Request(f"{url}/models")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))
            return [m['id'] for m in data.get('data', [])]
    except Exception:
        return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KernelWeave Interactive Shell")
    parser.add_argument("--model", type=str, default="gemma4:e2b", help="Model name to use")
    parser.add_argument("--store", type=str, default="store", help="Path to KernelStore")
    args = parser.parse_args()

    print_banner(args.model)
    
    # Interactive Selector
    print(f"{BOLD}Select your LLM provider setup:{RESET}")
    print("1. Ollama (Default port 11434)")
    print("2. LM Studio (Default port 1234)")
    print("3. Custom Endpoint & Model")
    print("Press Enter to auto-scan all defaults.")
    
    selected_model = args.model
    endpoints = []
    
    try:
        choice = input(f"\n{BOLD}Enter choice (1-3): {RESET}").strip()
        
        if choice == "1":
            print(f"{DIM}Fetching models from Ollama...{RESET}")
            models = get_ollama_models()
            if models:
                print(f"\n{YELLOW}Available Ollama Models:{RESET}")
                for i, m in enumerate(models):
                    print(f"  {i+1}. {m}")
                m_choice = input(f"{BOLD}Select model number (default 1): {RESET}").strip()
                if m_choice.isdigit() and 1 <= int(m_choice) <= len(models):
                    selected_model = models[int(m_choice)-1]
                else:
                    selected_model = models[0] if models else args.model
            else:
                print(f"{YELLOW}No models found or cannot reach Ollama. Using default.{RESET}")
                selected_model = args.model
            endpoints = [("ollama", "http://127.0.0.1:11434", selected_model)]
            
        elif choice == "2":
            print(f"{DIM}Fetching models from LM Studio...{RESET}")
            models = get_openai_models()
            if models:
                print(f"\n{YELLOW}Available LM Studio Models:{RESET}")
                for i, m in enumerate(models):
                    print(f"  {i+1}. {m}")
                m_choice = input(f"{BOLD}Select model number (default 1): {RESET}").strip()
                if m_choice.isdigit() and 1 <= int(m_choice) <= len(models):
                    selected_model = models[int(m_choice)-1]
                else:
                    selected_model = models[0] if models else args.model
            else:
                print(f"{YELLOW}No models found or cannot reach LM Studio. Using default.{RESET}")
                selected_model = args.model
            endpoints = [("openai-compatible", "http://127.0.0.1:1234/v1", selected_model)]
            
        elif choice == "3":
            selected_model = input(f"{BOLD}Enter model name: {RESET}").strip()
            custom_url = input(f"{BOLD}Enter API Base URL (e.g., http://localhost:11434): {RESET}").strip()
            custom_provider = "ollama" if "11434" in custom_url and "v1" not in custom_url else "openai-compatible"
            endpoints = [(custom_provider, custom_url, selected_model)]
            
        else:
            print(f"{DIM}Auto-scanning default local endpoints...{RESET}")
            endpoints = [
                ("ollama", "http://127.0.0.1:11434", args.model),
                ("openai-compatible", "http://127.0.0.1:11434/v1", args.model),
                ("openai-compatible", "http://127.0.0.1:1234/v1", args.model),
            ]
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)
        
    backend = None
    connected = False
    
    for provider, url, model in endpoints:
        print(f"{DIM}Trying {provider} at {url} with model '{model}'...{RESET}", end="", flush=True)
        test_backend = make_backend(provider, model, url)
        if check_connectivity(test_backend):
            print(f"\r{GREEN}[OK]{RESET} Connected to {provider} at {url}    ")
            backend = test_backend
            connected = True
            args.model = model # Update args.model for display purposes
            break
        else:
            print(f"\r{YELLOW}[FAIL]{RESET} {provider} at {url} not responding.    ")
            
    if not connected:
        print(f"\n{RED}[ERROR]{RESET} No running LLM server detected on default ports.")
        print(f"{YELLOW}Please ensure Ollama or LM Studio is running.{RESET}")
        
        # Interactive fallback
        try:
            choice = input(f"\n{BOLD}Would you like to enter a custom endpoint URL? (y/n): {RESET}").strip().lower()
            if choice == 'y':
                custom_url = input(f"{BOLD}Enter API Base URL (e.g., http://localhost:11434): {RESET}").strip()
                custom_provider = "ollama" if "11434" in custom_url and "v1" not in custom_url else "openai-compatible"
                print(f"{DIM}Trying custom endpoint...{RESET}", end="", flush=True)
                test_backend = make_backend(custom_provider, args.model, custom_url)
                if check_connectivity(test_backend):
                    print(f"\r{GREEN}[OK]{RESET} Connected successfully!    ")
                    backend = test_backend
                else:
                    print(f"\r{RED}[FAIL]{RESET} Custom endpoint also failed. Exiting.")
                    sys.exit(1)
            else:
                print(f"{YELLOW}Exiting. Please start your LLM server and try again.{RESET}")
                sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print()
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
            print(f"{DIM}Falling back to ReAct tool loop...{RESET}", end="\r")
            
            system_prompt = (
                "You are a helpful assistant with access to tools. You can use tools by outputting a JSON object with 'tool' and 'args' fields. For example:\n"
                "```json\n"
                "{\n"
                "  \"tool\": \"read_file\",\n"
                "  \"args\": {\"path\": \"file.txt\"}\n"
                "}\n"
                "```\n"
                "Available tools:\n"
                "- `list_dir(path=\".\")`: List directory contents.\n"
                "- `read_file(path)`: Read file content.\n"
                "- `write_file(path, content)`: Write file content.\n"
                "- `run_command(command)`: Run a terminal command.\n"
                "- `register_tool(name, code)`: Create a new tool. 'code' must be a Python string containing a function named 'name' or 'execute'.\n\n"
                "If you have enough information to answer, just answer normally. Do not use tools if you don't need to. "
                "Always output valid JSON when using a tool. After receiving tool output, continue answering or use another tool."
            )
            
            conversation = prompt
            max_iterations = 5
            
            try:
                for _ in range(max_iterations):
                    resp = backend.generate(conversation, system_prompt=system_prompt)
                    text = resp.text.strip()
                    
                    # Check if model wants to use a tool
                    if "```json" in text or (text.startswith("{") and "tool" in text):
                        # Extract JSON
                        try:
                            if "```json" in text:
                                json_str = text.split("```json")[1].split("```")[0].strip()
                            else:
                                json_str = text.strip()
                                
                            tool_call = json.loads(json_str)
                            tool_name = tool_call.get("tool")
                            tool_args = tool_call.get("args", {})
                            
                            if tool_name in TOOLS:
                                print(f"{DIM}Invoking tool {tool_name} with {tool_args}...{RESET}")
                                tool_result = TOOLS[tool_name](**tool_args)
                                conversation += f"\n\nObservation (Result of {tool_name}):\n{tool_result}\n\nContinue with your task."
                                print(f"{DIM}Tool executed. Continuing...{RESET}", end="\r")
                                continue
                            else:
                                conversation += f"\n\nObservation: Tool '{tool_name}' not found."
                                continue
                        except Exception as e:
                            # Not valid JSON or failed to parse, assume it's just a text response
                            response_text = text
                            break
                    else:
                        response_text = text
                        break
                        
                if not response_text:
                    response_text = "No final answer produced within iteration limit."
                    
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
