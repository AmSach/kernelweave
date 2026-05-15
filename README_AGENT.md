# KernelWeave Autonomous Agent (Ollama Integration)

This directory contains an autonomous agent loop that integrates the KernelWeave architecture with local LLMs via Ollama. It implements a ReAct (Reasoning + Acting) loop with tool use, JSON enforcement, and state tracking.

## Files
- `kernelweave_ollama.py`: The main interactive CLI loop. It routes prompts to local models, handles tool execution, and self-compiles traces into kernels.
- `test_agent.py`: A test runner script with a suite of complex tasks (Full-Stack, Trader, Designer, Debugger) to evaluate model performance.

## Features
- **JSON Enforcement**: Forces models to output structured JSON for tool calls.
- **State Tracker**: Detects repeating actions and injects warnings to break loops.
- **Robust Tools**: Tools like `read_file`, `write_file`, and `list_dir` are tolerant of common model hallucinations (e.g., `file_path` vs `path`).
- **Web Search with Fallbacks**: If DuckDuckGo fails, it falls back to **Bing via Playwright** to bypass Captchas!

## How to Run
1. Ensure Ollama is running locally with a model (e.g., `granite4.1:8b` or `gemma4:e4b`).
2. Run the interactive loop:
   ```bash
   python kernelweave_ollama.py --model granite4.1:8b
   ```
3. Run the test suite:
   ```bash
   python test_agent.py
   ```

## Shifting to Codex / Handover
If you are shifting this work to Codex or another system:
- The core logic for tool use and ReAct loop is in `kernelweave_ollama.py` lines 750-830.
- The tools are defined in lines 125-160.
- You can copy these tools and the loop logic to any other project to give a local LLM file-system and web access!
