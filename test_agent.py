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
import urllib.parse

# Ensure kernelweave is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kernelweave_ollama import TOOLS, get_ollama_models
from kernelweave.kernel import KernelStore
from pathlib import Path
from kernelweave.runtime import KernelRuntime

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
        "You are Rupert, an advanced autonomous AI operating system running on Windows.\n"
        "You must use tools by outputting a JSON object. You MUST include a 'thought' field for reasoning and a 'plan' list for multi-step tasks. For example:\n"
        "{\n"
        "  \"thought\": \"I need to search the web to find the latest news.\",\n"
        "  \"plan\": [\"Search the web\", \"Read articles\", \"Summarize\"],\n"
        "  \"tool\": \"web_search\",\n"
        "  \"args\": {\"query\": \"latest news\"}\n"
        "}\n"
        "Available tools: `browser_browse`, `web_search`, `read_file`, `write_file`, `list_dir`, `run_command`.\n"
        "CRITICAL RULES:\n"
        "1. You are on Windows. Do NOT use Unix commands like `source`, `cat <<EOF`, or `ls`. Use Windows equivalents or use provided tools.\n"
        "2. To create or edit files, ALWAYS use the `write_file` tool. Do NOT use `echo` or `cat` in `run_command` to write files.\n"
        "3. Output the JSON block IMMEDIATELY. Do not put any text before or after it.\n"
        "4. After writing or editing a file, you MUST verify it works by running it or reading it!"
    )
    
    conversation = (
        "User: Search the web for quantum computing.\n"
        "Rupert: {\"thought\": \"I need to search the web.\", \"plan\": [\"Search\"], \"tool\": \"web_search\", \"args\": {\"query\": \"quantum computing\"}}\n"
        f"User: {prompt}"
    )
    max_iterations = 10
    
    url = "http://127.0.0.1:11434/api/generate"
    
    try:
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            print("Thinking...", end="\r")
            
            body = {"model": model_name, "prompt": f"{system_prompt}\n\n{conversation}", "stream": True, "format": "json"}
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
                        
                        if len(text) > 4000:
                            print("\n\033[93m[Warning] Model output too long (>4000 chars), truncating...\033[0m")
                            break
            print() # Newline after stream
            
            # Check if model wants to use a tool
            json_str = extract_json(text)
            if json_str:
                try:
                    tool_call = json.loads(json_str)
                    tool_name = tool_call.get("tool")
                    tool_args = tool_call.get("args", {})
                    
                    # Fix for string arguments instead of dict
                    if isinstance(tool_args, str):
                        if tool_name == "run_command":
                            tool_args = {"command": tool_args}
                        elif tool_name in ["read_file", "list_dir"]:
                            tool_args = {"path": tool_args}
                        elif tool_name == "browser_browse":
                            tool_args = {"url": tool_args}
                    
                    if tool_name in TOOLS:
                        print(f"\n[Tool Execution] Invoking {tool_name} with {tool_args}")
                        tool_result = TOOLS[tool_name](**tool_args)
                        
                        # Fallback for failed search
                        if tool_name == "web_search" and ("No results found" in tool_result or "failed" in tool_result.lower()):
                            print("[Fallback] Web search failed. Trying browser_browse...")
                            if "browser_browse" in TOOLS:
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
    tasks = [
        "There is a file named 'broken_script.py' in the current directory. It has a bug (division by zero). You must read the file, run it using run_command to see the error, fix the bug by updating the file with write_file, and run it again to verify it works!",
        "Act as a quantitative trader. Search for 'latest Bitcoin price USD', extract the price, calculate a mock moving average, and write a trading signal (BUY/SELL) to 'trading_signal.txt' based on whether the price is above or below $60,000.",
        "Act as a UI/UX designer. Create a CSS file named 'modern_theme.css' with a glassmorphism design (backdrop-filter: blur(10px), semi-transparent background). Then read the file to verify it was written correctly.",
        "Act as a data analyst. Create a CSV file named 'sales_data.csv' with columns: Product, Units, Price. Put 3 rows of data. Then read the file and calculate the total revenue (Units * Price) for all products and output it to 'revenue_summary.txt'."
    ]
    
    # Use the model the user preferred or default
    model = "granite4.1:8b" 
    
    # Check available models
    models = get_ollama_models()
    if models:
        print(f"Available models: {models}")
        if "granite4.1:8b" not in models:
            model = models[0]
            print(f"Defaulting to available model: {model}")
            
    for i, task in enumerate(tasks):
        print(f"\n==================================================")
        print(f"RUNNING TASK {i+1}/{len(tasks)}")
        print(f"Task: {task}")
        print(f"==================================================")
        try:
            run_test(task, model_name=model)
        except Exception as e:
            print(f"Task {i+1} failed with error: {e}")
