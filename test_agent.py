"""
Test Script for Rupert's Agentic Loop
=====================================
This script simulates the ReAct loop directly to test tasks without the interactive shell.
"""
import os
import sys
import json
import time
import re
import urllib.request

# Ensure kernelweave is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kernelweave_ollama import TOOLS, get_ollama_models
from kernelweave.kernel import KernelStore
from pathlib import Path

def extract_json(text):
    start = text.find('{')
    if start == -1:
        return None
    count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return text[start:i+1]
    return None

def run_test(prompt, model_name="granite4.1:8b"):
    print(f"Starting test with prompt: '{prompt}' using model: {model_name}")
    
    # Initialize
    store = KernelStore(Path("store"))
    runtime = KernelRuntime(store, use_embeddings=True)
    
    # Routing
    plan = runtime.run(prompt)
    print(f"[Router] Mode: {plan['mode']} | Kernel: {plan.get('kernel_id', 'none')}")
    
    system_prompt = (
        "You are Rupert, an advanced autonomous AI operating system.\n"
        "You must use tools by outputting a JSON object. For example:\n"
        "```json\n"
        "{\n"
        "  \"tool\": \"web_search\",\n"
        "  \"args\": {\"query\": \"latest news\"}\n"
        "}\n"
        "```\n"
        "Available tools: `browser_browse`, `web_search`, `read_file`, `write_file`, `list_dir`, `run_command`.\n"
        "Be extremely concise. Output JSON immediately if you need a tool."
    )
    
    conversation = f"User: {prompt}"
    max_iterations = 3
    
    url = "http://127.0.0.1:11434/api/generate"
    
    try:
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            print("Thinking...", end="\r")
            
            body = {"model": model_name, "prompt": f"{system_prompt}\n\n{conversation}", "stream": True}
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
            
            text = ""
            first_token = True
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        token = chunk.get("response", "")
                        
                        # Check for reasoning/thinking fields
                        reasoning = chunk.get("reasoning_content", "") or chunk.get("thinking", "")
                        if reasoning:
                            token = reasoning
                            
                        if first_token and token.strip():
                            print(" " * 20, end="\r") # Clear "Thinking..."
                            first_token = False
                        print(token, end="", flush=True)
                        text += token
            print() # Newline after stream
            
            # Check if model wants to use a tool
            json_str = extract_json(text)
            if json_str:
                try:
                    tool_call = json.loads(json_str)
                    tool_name = tool_call.get("tool")
                    tool_args = tool_call.get("args", {})
                    
                    if tool_name in TOOLS:
                        print(f"\n[Tool Execution] Invoking {tool_name} with {tool_args}")
                        tool_result = TOOLS[tool_name](**tool_args)
                        
                        # Fallback for failed search
                        if tool_name == "web_search" and ("No results found" in tool_result or "failed" in tool_result.lower()):
                            print("[Fallback] Web search failed. Trying browser_browse...")
                            if "browser_browse" in TOOLS:
                                import urllib.parse
                                query_encoded = urllib.parse.quote(tool_args.get("query", ""))
                                tool_result = TOOLS["browser_browse"](url=f"https://www.google.com/search?q={query_encoded}")
                                print(f"[Fallback Result] {tool_result}")
                        
                        print(f"[Tool Result] {tool_result}")
                        
                        conversation += f"\n\nObservation (Result of {tool_name}):\n{tool_result}\n\nContinue with your task."
                        continue
                    else:
                        print(f"\n[Error] Tool '{tool_name}' not found.")
                        conversation += f"\n\nObservation: Tool '{tool_name}' not found."
                        continue
                except Exception as e:
                    print(f"\n[Error] Failed to execute tool: {e}")
                    conversation += f"\n\nObservation: Error parsing or executing tool: {e}. Please retry with valid JSON."
                    continue
            else:
                print("\nNo tool call detected. Task complete or model answered directly.")
                break
                
    except Exception as e:
        print(f"\nError during test: {e}")

if __name__ == "__main__":
    # Test with the researcher task
    prompt = "fetch me some news about Generative AI breakthroughs 2026 and save it to a file named ai_report.txt"
    # Use the model the user preferred or default
    model = "granite4.1:8b" 
    
    # Check available models
    models = get_ollama_models()
    if models:
        print(f"Available models: {models}")
        if "granite4.1:8b" not in models:
            model = models[0]
            print(f"Defaulting to available model: {model}")
            
    run_test(prompt, model_name=model)
