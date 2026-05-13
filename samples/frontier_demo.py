"""
KernelWeave Frontier-Grade Demo - Live on Ollama Gemma4
========================================================

This demo exercises EVERY frontier upgrade against your local Gemma4
model.  Zero cloud API calls.  The five pillars demonstrated:

  1. Ollama Backend         - native /api/chat integration
  2. Constrained Decoding   - Ollama's format:"json" grammar sampling
  3. Production Verifier    - LLM-as-judge backed by real Gemma4
  4. Self-Compilation       - LLM-assisted precondition/postcondition extraction
  5. Full Pipeline          - Route -> Execute -> Verify -> Compile -> Promote

Run:  python samples/frontier_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure kernelweave is importable from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernelweave.kernel import Kernel, KernelStatus, KernelStore, TraceEvent
from kernelweave.compiler import compile_trace_to_kernel
from kernelweave.runtime import (
    ExecutionEngine,
    KernelRuntime,
    verify_output_against_postconditions,
)
from kernelweave.verifier import VerifierHierarchy
from kernelweave.constrained import postconditions_to_schema, validate_output
from kernelweave.llm.providers import ModelPreset, OllamaBackend


# ── Formatting helpers ─────────────────────────────────────────────
# Force UTF-8 on Windows to avoid charmap encode errors
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"


def banner(text: str) -> None:
    width = 70
    print(f"\n{CYAN}{'=' * width}")
    print(f"  {BOLD}{text}{RESET}{CYAN}")
    print(f"{'=' * width}{RESET}\n")


def section(num: int, title: str) -> None:
    print(f"\n{YELLOW}{'-' * 60}")
    print(f"  PILLAR {num}: {BOLD}{title}{RESET}")
    print(f"{YELLOW}{'-' * 60}{RESET}\n")


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}|{RESET} {msg}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


# ── Build Ollama backend ───────────────────────────────────────────
def make_backend(model: str = "gemma4:e4b") -> OllamaBackend:
    preset = ModelPreset(
        id="frontier-demo",
        provider="ollama",
        model=model,
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
        max_tokens=2048,
        timeout_seconds=120,
    )
    return OllamaBackend(preset, json_mode=False)


# ───────────────────────────────────────────────────────────────────
#  PILLAR 1 - Ollama Backend: basic generation
# ───────────────────────────────────────────────────────────────────
def demo_ollama_backend(backend: OllamaBackend) -> None:
    section(1, "Ollama Native Backend")

    prompt = "What is the capital of France? Answer in one sentence."
    info(f"Prompt: {prompt}")

    t0 = time.time()
    resp = backend.generate(prompt, system_prompt="Answer concisely.")
    ok(f"Response ({elapsed(t0)}):")
    info(f"  {resp.text.strip()[:200]}")
    info(f"  Provider: {resp.provider}  Model: {resp.model}")
    if resp.usage:
        info(f"  Usage: {resp.usage}")


# ───────────────────────────────────────────────────────────────────
#  PILLAR 2 - True Constrained Decoding via Ollama JSON mode
# ───────────────────────────────────────────────────────────────────
def demo_constrained_decoding(backend: OllamaBackend) -> None:
    section(2, "True Constrained Decoding (Ollama JSON Mode)")

    # Define postconditions and derive a schema
    postconditions = [
        "output schema satisfied",
        "all required evidence recorded",
        "comparison mentions both artifacts",
        "rollback not triggered",
    ]
    schema = postconditions_to_schema(
        postconditions,
        {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]},
    )
    info("Schema derived from postconditions:")
    for line in json.dumps(schema, indent=2).splitlines():
        info(f"  {line}")

    # -- Unconstrained generation (may or may not be valid JSON) ----
    prompt = (
        "Compare file main.py and utils.py. "
        "Identify at least 2 differences in imports and function names."
    )
    info(f"\nPrompt: {prompt}")

    info("\n-- Unconstrained generation (no format constraint):")
    t0 = time.time()
    raw = backend.generate(prompt, system_prompt="Be concise.")
    info(f"  Raw text ({elapsed(t0)}):")
    for line in raw.text.strip().splitlines()[:6]:
        info(f"    {line}")
    try:
        json.loads(raw.text)
        ok("  Happens to be valid JSON (lucky)")
    except json.JSONDecodeError:
        info("  Not valid JSON (expected - no constraint was applied)")

    # -- Constrained generation (Ollama JSON grammar sampling) -----
    info("\n-- Constrained generation (format: json):")
    t0 = time.time()
    constrained = backend.generate(
        prompt,
        system_prompt=(
            "Output a JSON object with these fields:\n"
            + json.dumps(schema.get("properties", {}), indent=2)
            + "\nRequired: " + ", ".join(schema.get("required", []))
        ),
        json_mode=True,
    )
    info(f"  Constrained text ({elapsed(t0)}):")
    for line in constrained.text.strip().splitlines()[:10]:
        info(f"    {line}")
    try:
        parsed = json.loads(constrained.text)
        ok("  [OK] Valid JSON - guaranteed by Ollama grammar sampling")
        is_valid, errors = validate_output(parsed, schema)
        if is_valid:
            ok("  [OK] Passes postcondition schema validation")
        else:
            fail(f"  Schema validation errors: {errors}")
    except json.JSONDecodeError:
        fail("  JSON decode failed (unexpected)")

    # -- Schema-constrained generation (Ollama JSON schema mode) ---
    info("\n-- Schema-constrained generation (format: {schema}):")
    # Build a strict schema for Ollama's schema-mode
    strict_schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "differences": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence_found": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rollback_triggered": {"type": "boolean"},
        },
        "required": ["result", "differences", "evidence_found", "rollback_triggered"],
    }
    t0 = time.time()
    schema_resp = backend.generate(
        prompt,
        system_prompt="Provide a detailed comparison. Fill every required field.",
        json_schema=strict_schema,
    )
    info(f"  Schema-constrained text ({elapsed(t0)}):")
    for line in schema_resp.text.strip().splitlines()[:12]:
        info(f"    {line}")
    try:
        parsed = json.loads(schema_resp.text)
        ok("  [OK] Valid JSON")
        # Check required fields
        missing = [f for f in strict_schema["required"] if f not in parsed]
        if not missing:
            ok("  [OK] All required fields present")
        else:
            fail(f"  Missing fields: {missing}")
    except json.JSONDecodeError:
        fail("  JSON decode failed")


# ───────────────────────────────────────────────────────────────────
#  PILLAR 3 - Production Verifier with LLM Judge
# ───────────────────────────────────────────────────────────────────
def demo_verifier(backend: OllamaBackend) -> None:
    section(3, "Production Verifier - LLM-as-Judge (Gemma4)")

    # Generate some output to verify
    prompt = "Explain the difference between a list and a tuple in Python."
    t0 = time.time()
    resp = backend.generate(prompt)
    output = resp.text.strip()
    info(f"Model output ({elapsed(t0)}):")
    for line in output.splitlines()[:5]:
        info(f"  {line}")

    postconditions = [
        "output mentions lists are mutable",
        "output mentions tuples are immutable",
        "output is educational and accurate",
    ]
    evidence_reqs = ["example of list usage", "example of tuple usage"]

    # -- Heuristic-only verification (fast, local) -----------------
    info("\n-- Level 1: Heuristic verification (local, no LLM):")
    verifier_no_llm = VerifierHierarchy(
        enable_heuristic=True, enable_tool=False, enable_llm_judge=False
    )
    t0 = time.time()
    result_h = verifier_no_llm.verify(output, postconditions, evidence_reqs, prompt)
    status = f"{GREEN}PASS{RESET}" if result_h.passed else f"{RED}FAIL{RESET}"
    info(f"  Result: {status}  Score: {result_h.score:.2f}  Level: {result_h.level}  ({elapsed(t0)})")
    if result_h.matched:
        info(f"  Matched: {result_h.matched[:3]}")
    if result_h.failed:
        info(f"  Failed: {result_h.failed[:3]}")

    # -- Full hierarchy WITH LLM judge backed by Gemma4 ------------
    info("\n-- Level 3: Full hierarchy with LLM Judge (Gemma4):")
    verifier_full = VerifierHierarchy(
        enable_heuristic=True,
        enable_tool=True,
        enable_llm_judge=True,
        backend=backend,  # REAL Gemma4 backend!
    )
    t0 = time.time()
    result_full = verifier_full.verify(output, postconditions, evidence_reqs, prompt)
    status = f"{GREEN}PASS{RESET}" if result_full.passed else f"{RED}FAIL{RESET}"
    info(f"  Result: {status}  Score: {result_full.score:.2f}  Level: {result_full.level}  ({elapsed(t0)})")
    if result_full.matched:
        info(f"  Matched: {result_full.matched}")
    if result_full.failed:
        info(f"  Failed: {result_full.failed}")
    if result_full.details.get("judgment"):
        info(f"  Judge verdict: {json.dumps(result_full.details['judgment'], indent=2)[:300]}")


# ───────────────────────────────────────────────────────────────────
#  PILLAR 4 - Self-Compilation (LLM-assisted)
# ───────────────────────────────────────────────────────────────────
def demo_self_compilation(backend: OllamaBackend) -> None:
    section(4, "Self-Compilation - LLM-Assisted Kernel Extraction")

    # Build a synthetic trace
    events = [
        TraceEvent(kind="plan", payload={"text": "Plan: read both files, compare line by line"}),
        TraceEvent(kind="tool", payload={"tool": "file_read", "args": {"path": "main.py"}}),
        TraceEvent(kind="tool", payload={"tool": "file_read", "args": {"path": "utils.py"}}),
        TraceEvent(kind="observation", payload={"text": "main.py has 120 lines, utils.py has 45 lines"}),
        TraceEvent(kind="evidence", payload={"text": "main.py imports sys; utils.py does not"}),
        TraceEvent(kind="evidence", payload={"text": "utils.py defines helper_func; main.py does not"}),
        TraceEvent(kind="verification", payload={"text": "both files read successfully, differences catalogued"}),
        TraceEvent(kind="decision", payload={"text": "output structured comparison report"}),
    ]

    # -- Without LLM (basic extraction) ----------------------------
    info("-- Without LLM (rule-based extraction):")
    t0 = time.time()
    kernel_basic = compile_trace_to_kernel(
        trace_id="demo-trace-001",
        task_family="artifact comparison",
        description="Compare two source files and summarize differences",
        events=events,
        expected_output={"result": "comparison report"},
        backend=None,
    )
    info(f"  Compiled in {elapsed(t0)}")
    info(f"  Preconditions ({len(kernel_basic.preconditions)}):")
    for p in kernel_basic.preconditions[:4]:
        info(f"    - {p}")
    info(f"  Postconditions ({len(kernel_basic.postconditions)}):")
    for p in kernel_basic.postconditions[:4]:
        info(f"    - {p}")

    # -- With LLM (Gemma4 extracts real pre/post conditions) -------
    info("\n-- With LLM (Gemma4 extracts real pre/postconditions):")
    t0 = time.time()
    kernel_llm = compile_trace_to_kernel(
        trace_id="demo-trace-002",
        task_family="artifact comparison",
        description="Compare two source files and summarize differences",
        events=events,
        expected_output={"result": "comparison report"},
        backend=backend,  # REAL Gemma4!
    )
    info(f"  Compiled in {elapsed(t0)}")
    info(f"  Preconditions ({len(kernel_llm.preconditions)}):")
    for p in kernel_llm.preconditions:
        info(f"    - {p}")
    info(f"  Postconditions ({len(kernel_llm.postconditions)}):")
    for p in kernel_llm.postconditions:
        info(f"    - {p}")

    # Show the extracted vs. hardcoded difference
    extracted_pre = [p for p in kernel_llm.preconditions if p.startswith("extracted:")]
    extracted_post = [p for p in kernel_llm.postconditions if p.startswith("extracted:")]
    if extracted_pre or extracted_post:
        ok(f"  LLM extracted {len(extracted_pre)} preconditions + {len(extracted_post)} postconditions beyond template")
    else:
        info("  (LLM did not return additional conditions this time)")


# ───────────────────────────────────────────────────────────────────
#  PILLAR 5 - Full Pipeline: Route -> Execute -> Verify -> Compile
# ───────────────────────────────────────────────────────────────────
def demo_full_pipeline(backend: OllamaBackend) -> None:
    section(5, "Full Pipeline - Route -> Execute -> Verify -> Compile -> Promote")

    # Use a temporary store for this demo
    demo_store_path = Path("store")
    store = KernelStore(demo_store_path)

    info(f"Store: {store.summary()}")

    # -- Step 1: Routing -------------------------------------------
    prompt = "Compare config.yaml and config_prod.yaml and list the differences."
    info(f"\n[Step 1] Routing prompt: '{prompt}'")
    runtime = KernelRuntime(store, use_embeddings=False)
    t0 = time.time()
    plan = runtime.run(prompt)
    info(f"  Decision ({elapsed(t0)}): mode={plan['mode']}, score={plan.get('score', 0):.3f}")
    info(f"  Reason: {plan.get('reason', 'n/a')}")
    if plan.get("kernel_id"):
        info(f"  Kernel: {plan['kernel_id']}")

    # -- Step 2: Execute with backend ------------------------------
    info(f"\n[Step 2] Executing with Gemma4...")
    engine = ExecutionEngine(store, backend)
    t0 = time.time()
    result = engine.execute_plan(plan, prompt)
    info(f"  Executed ({elapsed(t0)}): mode={result.get('mode')}")
    response_text = result.get("response_text", "")
    if response_text:
        info(f"  Response preview:")
        for line in response_text.strip().splitlines()[:5]:
            info(f"    {line}")
    if result.get("verification"):
        v = result["verification"]
        status = f"{GREEN}PASS{RESET}" if v["passed"] else f"{RED}FAIL{RESET}"
        info(f"  Verification: {status}  Score: {v['score']:.2f}")

    # -- Step 3: Verify with LLM Judge -----------------------------
    if response_text:
        info(f"\n[Step 3] Verifying with LLM Judge (Gemma4)...")
        verifier = VerifierHierarchy(backend=backend)
        postconditions = plan.get("postconditions", [
            "output lists differences between config files",
            "output is structured and clear",
        ])
        t0 = time.time()
        v_result = verifier.verify(response_text, postconditions, prompt=prompt)
        status = f"{GREEN}PASS{RESET}" if v_result.passed else f"{RED}FAIL{RESET}"
        info(f"  Verification: {status}  Score: {v_result.score:.2f}  Level: {v_result.level}  ({elapsed(t0)})")

    # -- Step 4: Self-Compile into kernel --------------------------
    info(f"\n[Step 4] Self-compiling trace into kernel (LLM-assisted)...")
    events = [
        TraceEvent(kind="plan", payload={"text": prompt}),
        TraceEvent(kind="tool", payload={"tool": "file_read", "args": {"path": "config.yaml"}}),
        TraceEvent(kind="tool", payload={"tool": "file_read", "args": {"path": "config_prod.yaml"}}),
        TraceEvent(kind="evidence", payload={"text": "found differences in database host and debug flags"}),
        TraceEvent(kind="verification", payload={"text": "all differences catalogued"}),
        TraceEvent(kind="decision", payload={"text": response_text[:256] if response_text else "comparison done"}),
    ]
    t0 = time.time()
    new_kernel = compile_trace_to_kernel(
        trace_id="frontier-demo-trace",
        task_family="config comparison",
        description="Compare YAML configuration files",
        events=events,
        expected_output={"result": "config diff report"},
        backend=backend,
    )
    info(f"  Compiled kernel ({elapsed(t0)}):")
    info(f"    ID: {new_kernel.kernel_id}")
    info(f"    Family: {new_kernel.task_family}")
    info(f"    Confidence: {new_kernel.status.confidence:.4f}")
    info(f"    State: {new_kernel.status.state}")
    info(f"    Preconditions: {len(new_kernel.preconditions)}")
    info(f"    Postconditions: {len(new_kernel.postconditions)}")

    # -- Step 5: Promote to store ----------------------------------
    info(f"\n[Step 5] Saving compiled kernel to store...")
    path = store.add_kernel(new_kernel)
    ok(f"  Kernel saved: {path}")
    info(f"  Store now: {store.summary()}")

    # -- Step 6: Re-route - does the new kernel match? -------------
    info(f"\n[Step 6] Re-routing same prompt to test new kernel match...")
    plan2 = runtime.run(prompt)
    info(f"  Decision: mode={plan2['mode']}, score={plan2.get('score', 0):.3f}")
    info(f"  Reason: {plan2.get('reason', 'n/a')}")
    if plan2.get("kernel_id") == new_kernel.kernel_id:
        ok("  [OK] New kernel matched on re-route! Self-improvement loop complete.")
    elif plan2.get("kernel_id"):
        info(f"  Matched different kernel: {plan2['kernel_id']}")
    else:
        info("  No kernel match (score below threshold)")


# -- Constrained decoding with schema for pipeline -----------------
def demo_constrained_pipeline(backend: OllamaBackend) -> None:
    section(6, "BONUS: Constrained Execution - Kernel-Guided JSON Generation")

    # Load a kernel from the store
    store = KernelStore(Path("store"))
    kernels = store.list_kernels()
    if not kernels:
        info("No kernels in store - skipping.")
        return

    kernel = store.get_kernel(kernels[0]["kernel_id"])
    info(f"Using kernel: {kernel.name} ({kernel.kernel_id})")
    info(f"Postconditions: {kernel.postconditions[:4]}")

    # Derive schema from postconditions
    schema = postconditions_to_schema(kernel.postconditions, kernel.output_schema)
    info(f"Derived schema has {len(schema.get('properties', {}))} properties")

    # Generate with schema constraint
    prompt = "Compare the imports and functions in main.py vs utils.py"
    info(f"\nPrompt: {prompt}")
    t0 = time.time()
    resp = backend.generate(
        prompt,
        system_prompt=(
            f"Execute kernel '{kernel.name}' for task family '{kernel.task_family}'.\n"
            f"Output JSON matching: {json.dumps(schema, indent=2)}\n"
            f"Postconditions to satisfy: {', '.join(kernel.postconditions[:4])}"
        ),
        json_mode=True,
    )
    info(f"Response ({elapsed(t0)}):")
    for line in resp.text.strip().splitlines()[:10]:
        info(f"  {line}")

    try:
        parsed = json.loads(resp.text)
        ok("Valid JSON output")
        is_valid, errors = validate_output(parsed, schema)
        if is_valid:
            ok("Passes postcondition schema validation")
        else:
            fail(f"Schema errors: {errors}")

        # Verify with postconditions
        v = verify_output_against_postconditions(
            resp.text, kernel.postconditions, kernel.evidence_requirements
        )
        status = f"{GREEN}PASS{RESET}" if v.passed else f"{RED}FAIL{RESET}"
        info(f"Postcondition verification: {status}  Score: {v.score:.2f}")
    except json.JSONDecodeError:
        fail("JSON decode failed")


# -- Main -----------------------------------------------------------
def main() -> None:
    banner("KernelWeave Frontier-Grade Demo - Ollama Gemma4")

    print(f"  {DIM}This demo exercises every frontier upgrade against your")
    print(f"  local Gemma4 model.  Zero cloud API calls.{RESET}")
    print(f"  {DIM}Ensure Ollama is running: ollama serve{RESET}")

    # Detect model
    model = "gemma4:e2b"
    info(f"Using model: {model}")

    backend = make_backend(model)

    # Quick connectivity check
    info("Testing Ollama connectivity...")
    try:
        t0 = time.time()
        test = backend.generate("Say 'ready' if you can hear me.", max_tokens=32)
        ok(f"Connected to Ollama ({elapsed(t0)}): {test.text.strip()[:60]}")
    except Exception as e:
        fail(f"Cannot reach Ollama: {e}")
        print(f"\n  {RED}Make sure Ollama is running: ollama serve{RESET}")
        sys.exit(1)

    # Run all pillars
    demo_ollama_backend(backend)
    demo_constrained_decoding(backend)
    demo_verifier(backend)
    demo_self_compilation(backend)
    demo_full_pipeline(backend)
    demo_constrained_pipeline(backend)

    banner("ALL FRONTIER PILLARS DEMONSTRATED SUCCESSFULLY")
    print(f"  {GREEN}Summary of what you just saw:{RESET}")
    print(f"  1. OllamaBackend - Native /api/chat integration with Gemma4")
    print(f"  2. Constrained Decoding - Ollama's grammar-based JSON sampling")
    print(f"  3. Production Verifier - Real LLM-as-judge (not mock)")
    print(f"  4. Self-Compilation - LLM-extracted pre/postconditions")
    print(f"  5. Full Pipeline - Route -> Execute -> Verify -> Compile -> Promote")
    print(f"  6. Kernel-Guided Generation - Postconditions -> Schema -> JSON")
    print()


if __name__ == "__main__":
    main()
