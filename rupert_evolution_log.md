# Rupert Evolution Log

This file tracks the autonomous testing and evolution of Rupert OS to make it a true "frontier-grade" assistant.

## Vision
To evolve Rupert from a prototype shell into a reliable, autonomous co-worker capable of handling complex, sequential tasks across different domains (Research, Coding, Trading/Data).

## Current Status (2026-05-14)
- **GUI**: Rebuilt as Rupert OS with 4 panels (Prompt, Kernel, Tool, Command).
- **Engine**: ReAct loop implemented (max 3 iterations).
- **Tools**: `web_search`, `write_file`, `read_file`, `run_command`, `list_dir`, `browser_browse` (Playwright).
- **Issues**: Playwright installation failed in the user's specific venv due to missing `pip`. Fallback to system `pip` implemented in code but needs verification. Scraper returning 0 results.

---

## Evolution Plan

### Phase 1: Robustness & Reliability (Current)
- [ ] Verify Playwright installation via system pip fallback.
- [ ] Fix the web search scraper fallback or ensure Playwright works for search.
- [ ] Increase ReAct loop limit or make it dynamic.

### Phase 2: Domain Simulations & Testing
I will simulate tasks for different personas and log the results here.

#### 1. The Researcher
- **Task**: Fetch the latest news on "Generative AI breakthroughs 2026", summarize them, and save a report.
- **Expected Outcome**: Rupert uses search/browser, processes text, and writes a clean MD file.
- **Status**: Failed (Gaps identified: Brittle search, fragile JSON parsing).

#### 2. The Coder
- **Task**: Create a Python script to calculate Fibonacci numbers with a CLI, write tests, and run them.
- **Expected Outcome**: Rupert writes code, runs it, fixes errors, and verifies success.
- **Status**: Failed (Gaps identified: Model used 'filename' instead of 'path').

#### 3. The Trader/Data Analyst
- **Task**: Fetch mock stock data, calculate a moving average, and output a recommendation.
- **Expected Outcome**: Rupert handles data processing and logic.
- **Status**: Pending.

---

## Execution Logs

### Test 1: Researcher Task Simulation (Initial Attempt)
- **Task**: Fetch news about Generative AI breakthroughs 2026.
- **Result**: **FAILED**.
- **Details**:
  1. `web_search` returned 0 results because the scraper was rate-limited.
  2. The model (`granite4.1:8b`) outputted invalid JSON in Iteration 2 (extra closing brace).
  3. The script crashed on JSON error instead of feeding it back to the model.
- **Gaps Identified**:
  - Need robust JSON parsing (fix malformed JSON).
  - Need automatic fallback to `browser_browse` if `web_search` fails.
  - Need to feed errors back to the model in the loop!

### Test 2: Coder Task Simulation (Initial Attempt)
- **Task**: Write fibonacci.py and test it.
- **Result**: **FAILED**.
- **Details**:
  1. The model used `write_file` with argument `filename` instead of `path`.
  2. The tool threw an error. The model failed to correct itself across 3 iterations.
- **Fix Applied**:
  - Upgraded `tool_write_file` in `kernelweave_ollama.py` to accept both `path` and `filename` as aliases!

### Test 3: Super Complex Full-Stack Task (Initial Attempt)
- **Task**: Create a full-stack web app, run server, and verify with browser.
- **Result**: **FAILED**.
- **Details**:
  1. The model tried to write to `test_app/requirements.txt` but the directory `test_app` didn't exist!
  2. The tool failed with `FileNotFoundError` ([Errno 2] No such file or directory).
  3. The model failed to create the directory and kept trying to write the file.
- **Fix Applied**:
  - Upgraded `tool_write_file` in `kernelweave_ollama.py` to **automatically create parent directories**!

### Test 4: Super Complex Full-Stack Task (Second Attempt)
- **Task**: Create a full-stack web app, run server, and verify with browser.
- **Result**: **PARTIAL SUCCESS** (Gaps identified: Hallucinated arguments).
- **Details**:
  1. The model successfully wrote `requirements.txt`, `app.py`, `test_app.py`, `setup.py`, and `templates/index.html`!
  2. It successfully ran `pip install` in the venv!
  3. It failed in Iteration 9 because it added `"shell": true` to `run_command` arguments.
  4. It ran out of iterations (max 10) before running the server or browser.
- **Fix Applied**:
  - Upgraded `tool_run_command` in `kernelweave_ollama.py` to accept `shell` as an optional argument!
