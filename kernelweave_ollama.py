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
from kernelweave.llm.providers import ModelPreset, OllamaBackend, OpenAICompatibleBackend

def ensure_dependency(package_name, import_name):
    import importlib
    import subprocess
    import sys
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"\033[93m[Setup] Installing missing dependency: {package_name}...\033[0m")
        print(f"\033[93m[Setup] This may take 1-2 minutes for large packages. Please do not close the window...\033[0m")
        try:
            # Check if pip is available
            try:
                subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
            except Exception:
                print(f"\033[93m[Setup] pip not found in venv. Attempting to install pip...\033[0m")
                try:
                    import urllib.request
                    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
                    subprocess.run([sys.executable, "get-pip.py"], check=True)
                    print(f"\033[92m[Setup] pip installed successfully in venv!\033[0m")
                except Exception as e:
                    print(f"\033[91m[Setup] Failed to install pip: {e}\033[0m")
                
            # Try running via the current python executable
            subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
            print(f"\033[92m[Setup] Successfully installed {package_name}!\033[0m")
            return True
        except Exception:
            try:
                # Fallback to system pip if python -m pip fails
                subprocess.run(["pip", "install", package_name], check=True)
                print(f"\033[92m[Setup] Successfully installed {package_name} via system pip!\033[0m")
                return True
            except Exception as e:
                print(f"\033[91m[Setup] Failed to install {package_name}: {e}\033[0m")
                print(f"\033[93m[Setup] Please run 'pip install {package_name}' manually.\033[0m")
                return False

HAS_DDG = ensure_dependency("duckduckgo-search", "duckduckgo_search")
if HAS_DDG:
    try:
        import importlib
        importlib.invalidate_caches()
        from duckduckgo_search import DDGS
    except ImportError:
        print("\033[93m[Setup] duckduckgo-search installed but not visible to current process. Falling back to scraper for this session.\033[0m")
        HAS_DDG = False

# Ensure sentence-transformers is installed for vector matching in router
ensure_dependency("sentence-transformers", "sentence_transformers")

HAS_PLAYWRIGHT = ensure_dependency("playwright", "playwright")

def tool_browser_browse(url):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=False)
            except Exception:
                print("[Setup] Installing Chromium binaries...")
                import subprocess
                import sys
                subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                browser = p.chromium.launch(headless=False)
            
            # Stealth: Set realistic user agent
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            context = browser.new_context(user_agent=user_agent)
            page = context.new_page()
            
            # Stealth: Hide navigator.webdriver flag to bypass bot detection
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            print(f"[Browser] Navigating to {url}...")
            page.goto(url, wait_until="domcontentloaded")
            
            import time
            time.sleep(2)
            title = page.title()
            content = page.evaluate("() => document.body.innerText")
            
            # Log and check for captchas
            print(f"[Browser] Title: {title}")
            print(f"[Browser] Content snippet: {content[:100].strip()}")
            
            if "captcha" in content.lower() or "robot" in content.lower() or "verify you are human" in content.lower():
                print(f"[Browser] WARNING: Captcha or bot detection detected on {url}!")
                return f"Browser error: Access denied or Captcha detected on {url}. Please try another source."
                
            browser.close()
            return f"Successfully browsed {url}.\nTitle: {title}\nContent Snippet: {content[:500]}..."
    except Exception as e:
        print(f"[Browser] Error: {e}")
        return f"Browser error: {e}"

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

def tool_write_file(path=None, content="", filename=None):
    actual_path = path or filename
    if not actual_path:
        return "Error: path or filename required."
    try:
        from pathlib import Path
        Path(actual_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(actual_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File '{actual_path}' written successfully."
    except Exception as e:
        return str(e)

def tool_run_command(command, shell=True):
    import subprocess
    try:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True)
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

def get_ollama_embedding(text, model="gemma4:e2b"):
    import urllib.request
    import json
    url = "http://127.0.0.1:11434/api/embeddings"
    body = {"model": model, "prompt": text}
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get("embedding", [])
    except Exception:
        return []

def cosine_similarity(v1, v2):
    import math
    if not v1 or not v2:
        return 0
    dot_product = sum(a*b for a,b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a*a for a in v1))
    magnitude2 = math.sqrt(sum(b*b for b in v2))
    if not magnitude1 or not magnitude2:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def print_box(title, text, color="\033[96m"):
    lines = text.split('\n')
    width = max(len(line) for line in lines) + 4
    width = max(width, len(title) + 6)
    if width > 80: width = 80 # Cap width
    reset = "\033[0m"
    print(color + "┌" + "─" * (width-2) + "┐" + reset)
    print(color + f"│  {BOLD}{title}{reset}{color}" + " " * (width - len(title) - 4) + "│" + reset)
    print(color + "├" + "─" * (width-2) + "┤" + reset)
    for line in lines:
        # Wrap long lines if needed, or just truncate for the box display
        truncated_line = line[:width-4]
        print(color + "│  " + reset + truncated_line + " " * (width - len(truncated_line) - 4) + color + "│" + reset)
    print(color + "└" + "─" * (width-2) + "┘" + reset)

def get_relevant_memory(query, history, top_k=3):
    if not history:
        return ""
        
    print(f"\033[2m[Memory] Searching past conversation via embeddings...\033[0m")
    query_vec = get_ollama_embedding(query)
    if not query_vec:
        # Fallback to last few messages if embedding fails
        return "\n".join(history[-top_k*2:])
        
    scored_memories = []
    for msg in history:
        msg_vec = get_ollama_embedding(msg)
        if msg_vec:
            score = cosine_similarity(query_vec, msg_vec)
            scored_memories.append((score, msg))
            
    # Sort by score descending
    scored_memories.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K
    relevant = [msg for score, msg in scored_memories[:top_k]]
    return "\n---\n".join(relevant)

def log_conversation(prompt, response):
    import datetime
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "conversations.jsonl"
        
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt": prompt,
            "response": response
        }
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"\033[2m[Log Error] Failed to save log: {e}\033[0m")

def tool_web_search(query):
    import urllib.request
    import urllib.parse
    import re
    import json
    
    print(f"{DIM}Searching the web for: {query}...{RESET}")
    results = []
    
    try:
        # Priority 1: Use official duckduckgo_search package if available
        if HAS_DDG:
            try:
                with DDGS() as ddgs:
                    ddg_results = ddgs.text(query, max_results=10)
                    for r in ddg_results:
                        if 'body' in r:
                            results.append(r['body'])
                    print(f"{DIM}Fetched {len(results)} results via duckduckgo_search.{RESET}")
            except Exception as e:
                print(f"{DIM}duckduckgo_search failed ({e}), falling back to scraper...{RESET}")
                
        # Priority 2: Fallback to scraping
        if not results:
            url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({'q': query})
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read().decode('utf-8')
                    
                # Extract snippets using more generic regexes
                snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
                if not snippets:
                    # Try finding text inside result__snippet divs or similar
                    snippets = re.findall(r'<[^>]+class="[^"]*snippet[^"]*"[^>]*>(.*?)</[^>]+>', html, re.DOTALL)
                
                for snippet in snippets:
                    clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    if clean_snippet and len(clean_snippet) > 10:
                        results.append(clean_snippet)
                        
                print(f"{DIM}Fetched {len(results)} results via scraper.{RESET}")
            except Exception as e:
                return f"Search failed during scraping: {e}"
                
        if not results:
            return ("No results found. DuckDuckGo may be rate-limiting the scraper.\n"
                    "Tip: Run 'pip install duckduckgo-search' to enable the robust official API.")
            
        # 2. Vector Embedding Search (RAG)
        # Embed the query
        query_vec = get_ollama_embedding(query)
        
        if not query_vec:
            # Fallback to simple return if embeddings fail
            return "\n\n".join(results[:5])
            
        # Embed results and score them
        scored_results = []
        print(f"{DIM}Ranking results with vector embeddings...{RESET}")
        for res in results[:10]: # Score top 10
            res_vec = get_ollama_embedding(res)
            if res_vec:
                score = cosine_similarity(query_vec, res_vec)
                scored_results.append((score, res))
            else:
                scored_results.append((0, res))
                
        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Format output
        output = []
        for score, res in scored_results[:5]: # Return top 5 ranked
            output.append(f"[Relevance: {score:.2f}] {res}")
            
        return "\n\n".join(output)
        
    except Exception as e:
        return f"Search failed: {e}"

TOOLS = {
    "list_dir": tool_list_dir,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_command": tool_run_command,
    "register_tool": tool_register_tool,
    "web_search": tool_web_search,
    "browser_browse": tool_browser_browse
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

def wrap_with_streaming(backend, provider, url, model):
    original_generate = backend.generate
    
    def streaming_generate(prompt, system_prompt="", **kwargs):
        # Fall back to original for JSON mode or if specific overrides are requested
        if kwargs.get("json_mode") or kwargs.get("json_schema") or kwargs.get("max_tokens") == 1:
            return original_generate(prompt, system_prompt=system_prompt, **kwargs)
            
        import urllib.request
        import json
        
        combined_prompt = prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{prompt}"
            
        if provider == "ollama":
            body = {
                "model": model,
                "prompt": combined_prompt,
                "stream": True,
                "options": {"temperature": 0.4}
            }
            target_url = f"{url.rstrip('/')}/api/generate"
        else: # openai-compatible
            body = {
                "model": model,
                "messages": [],
                "temperature": 0.4,
                "stream": True
            }
            if system_prompt:
                body["messages"].append({"role": "system", "content": system_prompt})
            body["messages"].append({"role": "user", "content": prompt})
            target_url = f"{url.rstrip('/')}/chat/completions"
            
        req = urllib.request.Request(target_url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
        
        full_text = ""
        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    if line:
                        line_text = line.decode('utf-8').strip()
                        if provider == "ollama":
                            payload = json.loads(line_text)
                            chunk = payload.get("response", "")
                            full_text += chunk
                            print(chunk, end="", flush=True)
                            if payload.get("done", False):
                                break
                        else: # openai-compatible
                            if line_text.startswith("data: "):
                                data_str = line_text[6:]
                                if data_str == "[DONE]":
                                    break
                                payload = json.loads(data_str)
                                choices = payload.get("choices", [])
                                if choices:
                                    chunk = choices[0].get("delta", {}).get("content", "")
                                    full_text += chunk
                                    print(chunk, end="", flush=True)
            print() # New line at the end
            from kernelweave.llm.providers import ModelResponse
            return ModelResponse(text=full_text, provider=provider, model=model, raw={"streamed": True}, usage={})
        except Exception as e:
            # Fallback to original non-streaming call if stream fails
            return original_generate(prompt, system_prompt=system_prompt, **kwargs)
            
    backend.generate = streaming_generate
    return backend

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
        
    # Wrap backend with streaming for the interactive session
    if connected:
        backend = wrap_with_streaming(backend, provider, url, model)
    else:
        backend = wrap_with_streaming(backend, custom_provider, custom_url, args.model)
        
    store_path = Path(args.store)
    store = KernelStore(store_path)
    
    # Enable embeddings for smart kernel matching
    print(f"{DIM}Initializing vector embeddings for router...{RESET}")
    runtime = KernelRuntime(store, use_embeddings=True)
    
    engine = ExecutionEngine(store, backend)
    verifier = VerifierHierarchy(backend=backend)
    
    print(f"{DIM}Store loaded: {store.summary()}{RESET}\n")

    conversation_history = []

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
                "- `register_tool(name, code)`: Create a new tool. 'code' must be a Python string containing a function named 'name' or 'execute'.\n"
                "- `web_search(query)`: Search the web for information. Uses vector embeddings to rank relevance.\n\n"
                "If you have enough information to answer, just answer normally. Do not use tools if you don't need to. "
                "Always output valid JSON when using a tool. After receiving tool output, continue answering or use another tool."
            )
            
            relevant_mem = get_relevant_memory(prompt, conversation_history)
            conversation = f"System: Relevant past context:\n{relevant_mem}\n\nUser: {prompt}"
            max_iterations = 5
            
            try:
                for _ in range(max_iterations):
                    print(f"{DIM}Thinking...{RESET}", end="\r")
                    
                    import urllib.request
                    url = "http://127.0.0.1:11434/api/generate"
                    body = {"model": args.model, "prompt": f"{system_prompt}\n\n{conversation}", "stream": True}
                    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
                    
                    text = ""
                    first_token = True
                    with urllib.request.urlopen(req, timeout=30) as response:
                        for line in response:
                            if line:
                                chunk = json.loads(line.decode('utf-8'))
                                token = chunk.get("response", "")
                                if first_token and token.strip():
                                    print(" " * 20, end="\r") # Clear "Thinking..."
                                    first_token = False
                                print(token, end="", flush=True)
                                text += token
                    print() # Newline after stream
                    
                    # Check if model wants to use a tool
                    if "```json" in text or (text.strip().startswith("{") and "tool" in text):
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
                                print_box("Tool Execution", f"Invoking {tool_name} with {tool_args}", color=CYAN)
                                tool_result = TOOLS[tool_name](**tool_args)
                                print_box("Tool Result", str(tool_result), color=GREEN)
                                
                                conversation += f"\n\nObservation (Result of {tool_name}):\n{tool_result}\n\nContinue with your task."
                                continue
                            else:
                                print_box("Error", f"Tool '{tool_name}' not found.", color=RED)
                                conversation += f"\n\nObservation: Tool '{tool_name}' not found."
                                continue
                        except Exception as e:
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
        print_box("KernelWeave OS", response_text, color=MAGENTA)
        
        # Log conversation
        log_conversation(prompt, response_text)
        
        # Append to history
        conversation_history.append(f"User: {prompt}")
        conversation_history.append(f"Assistant: {response_text}")
        
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
