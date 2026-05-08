from __future__ import annotations

import argparse
from pathlib import Path
import json

from .compiler import compile_trace_to_kernel
from .kernel import KernelStore, TraceEvent, load_sample_store
from .runtime import KernelRuntime
from .llm import LLMConfig, KernelWeaveLLM
from .llm.providers import ModelCatalog, MockBackend, backend_from_preset, run_preset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernelweave", description="Self-compiling skill kernels for LLMs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="initialise a store")
    init_p.add_argument("store", type=Path)

    add_p = sub.add_parser("add-sample", help="install sample kernels and traces")
    add_p.add_argument("store", type=Path)

    compile_p = sub.add_parser("compile", help="compile a trace into a kernel")
    compile_p.add_argument("store", type=Path)
    compile_p.add_argument("trace_id")
    compile_p.add_argument("task_family")
    compile_p.add_argument("description")

    plan_p = sub.add_parser("plan", help="choose a kernel or fallback for a prompt")
    plan_p.add_argument("store", type=Path)
    plan_p.add_argument("prompt")

    list_p = sub.add_parser("list", help="list kernels")
    list_p.add_argument("store", type=Path)

    traces_p = sub.add_parser("traces", help="list traces")
    traces_p.add_argument("store", type=Path)

    info_p = sub.add_parser("info", help="summarize store status")
    info_p.add_argument("store", type=Path)

    model_p = sub.add_parser("model", help="manage pluggable model presets")
    model_sub = model_p.add_subparsers(dest="model_cmd", required=True)

    model_list = model_sub.add_parser("list", help="list model presets")
    model_list.add_argument("--models-dir", type=Path, default=None)

    model_show = model_sub.add_parser("show", help="show a model preset")
    model_show.add_argument("preset_id")
    model_show.add_argument("--models-dir", type=Path, default=None)

    model_run = model_sub.add_parser("run", help="run a prompt against a preset")
    model_run.add_argument("preset_id")
    model_run.add_argument("prompt")
    model_run.add_argument("--models-dir", type=Path, default=None)
    model_run.add_argument("--kernel-store", type=Path, default=None)
    model_run.add_argument("--system-prompt", default="")
    model_run.add_argument("--temperature", type=float, default=None)
    model_run.add_argument("--max-tokens", type=int, default=None)
    model_run.add_argument("--mock", action="store_true")
    model_run.add_argument("--mock-response", default="")
    model_run.add_argument("--auto-compile", action="store_true", help="automatically compile successful responses into kernel candidates")

    return parser


def install_samples(store: KernelStore) -> None:
    """Install sample kernels for demonstration.
    
    Updated with expanded alias tables to reduce false negatives.
    """
    from .kernel import Kernel, KernelStatus, TraceEvent
    from .compiler import compile_trace_to_kernel
    
    # Comparison kernel - expanded aliases
    comparison_aliases = [
        "artifact comparison", "file comparison", "document comparison",
        "compare files", "compare documents", "compare artifacts",
        "diff files", "diff documents", "find differences",
        "what changed", "what differs", "how do they differ",
        "list changes", "show differences", "explain differences",
        "compare versions", "version comparison", "diff between",
        "changes between", "diverge", "divergence", "delta",
        "dockerfile comparison", "config comparison", "code comparison",
    ]
    
    comparison_trace = [
        TraceEvent(kind="plan", payload={"text": "compare two artifacts and explain differences"}),
        TraceEvent(kind="tool", payload={"tool": "load_artifact", "args": {"path": "A"}}),
        TraceEvent(kind="tool", payload={"tool": "load_artifact", "args": {"path": "B"}}),
        TraceEvent(kind="evidence", payload={"text": "differences found"}),
        TraceEvent(kind="verification", payload={"text": "summary mentions both"}),
    ]
    
    comparison_kernel = compile_trace_to_kernel(
        "trace-compare-002",
        "artifact comparison",
        "Compare two structured artifacts and produce a grounded summary.",
        comparison_trace,
        {"result": "comparison summary"},
    )
    
    # Add expanded aliases as additional task family hints
    comparison_kernel.metadata = {"aliases": comparison_aliases}
    
    # Add artifact-scoping precondition to prevent false positives
    comparison_kernel.preconditions.insert(0, "inputs are named files, schemas, or documents")
    
    comparison_kernel.status = KernelStatus(
        state="verified",
        confidence=0.62,
        failures=0,
        passes=2,
    )
    
    # Safe command kernel - expanded aliases
    command_aliases = [
        "safe shell command", "safe command execution", "shell safety",
        "run command safely", "execute command", "bash command",
        "shell command", "terminal command", "cli command",
        "command line", "run script", "execute script",
        "safe execution", "secure command", "validated command",
    ]
    
    command_trace = [
        TraceEvent(kind="plan", payload={"text": "generate safe shell command"}),
        TraceEvent(kind="tool", payload={"tool": "check_safety", "args": {"command": "placeholder"}}),
        TraceEvent(kind="verification", payload={"text": "no destructive patterns"}),
        TraceEvent(kind="evidence", payload={"text": "command validated"}),
    ]
    
    command_kernel = compile_trace_to_kernel(
        "trace-command-001",
        "safe shell command",
        "Generate a validated, non-destructive shell command.",
        command_trace,
        {"result": "safe command"},
    )
    
    command_kernel.metadata = {"aliases": command_aliases}
    command_kernel.status = KernelStatus(
        state="verified",
        confidence=0.71,
        failures=0,
        passes=1,
    )
    
    store.add_kernel(comparison_kernel)
    store.add_kernel(command_kernel)


def _load_catalog(models_dir: Path | None) -> ModelCatalog:
    if models_dir is None:
        return ModelCatalog.load_default()
    if models_dir.exists():
        return ModelCatalog.from_paths([models_dir])
    return ModelCatalog()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "model":
        catalog = _load_catalog(getattr(args, "models_dir", None))
        if args.model_cmd == "list":
            print(json.dumps([preset.to_dict() for preset in catalog.list()], indent=2, sort_keys=True))
            return
        if args.model_cmd == "show":
            print(json.dumps(catalog.get(args.preset_id).to_dict(), indent=2, sort_keys=True))
            return
        if args.model_cmd == "run":
            preset = catalog.get(args.preset_id)
            if getattr(args, "kernel_store", None) is not None:
                kernel_store = load_sample_store(args.kernel_store)
            else:
                kernel_store = None
            backend = MockBackend(preset, response_text=args.mock_response) if getattr(args, "mock", False) else backend_from_preset(preset)
            wrapper = KernelWeaveLLM(LLMConfig.reasoner_frontier_spec(), kernel_store=kernel_store, backend=backend)
            result = wrapper.respond(
                args.prompt,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            # Record feedback for learning loop
            if kernel_store is not None and result.get("text"):
                kernel_store.record_runtime_feedback(
                    prompt=args.prompt,
                    kernel_id=result.get("routing", {}).get("kernel_plan", {}).get("kernel_id") if result.get("routing", {}).get("kernel_plan") else None,
                    mode=result.get("routing", {}).get("routing", "generate"),
                    reason=result.get("routing", {}).get("routing", "generate"),
                    confidence=result.get("routing", {}).get("confidence", 0.5),
                    evidence_debt=1.0 - result.get("routing", {}).get("confidence", 0.5),
                    response_text=result.get("text", ""),
                    observed={"success": bool(result.get("text"))},
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return

    store = load_sample_store(args.store)

    if args.cmd == "init":
        print(json.dumps(store.summary(), indent=2, sort_keys=True))
        return
    if args.cmd == "add-sample":
        install_samples(store)
        print(json.dumps(store.summary(), indent=2, sort_keys=True))
        return
    if args.cmd == "compile":
        events = [
            TraceEvent(kind="plan", payload={"text": "user provided trace"}),
            TraceEvent(kind="verification", payload={"text": "trace was supplied"}),
        ]
        kernel = compile_trace_to_kernel(args.trace_id, args.task_family, args.description, events, expected_output={"result": "ok"})
        store.add_kernel(kernel)
        print(json.dumps(kernel.to_dict(), indent=2, sort_keys=True))
        return
    if args.cmd == "plan":
        runtime = KernelRuntime(store)
        print(json.dumps(runtime.run(args.prompt), indent=2, sort_keys=True))
        return
    if args.cmd == "list":
        print(json.dumps(store.list_kernels(), indent=2, sort_keys=True))
        return
    if args.cmd == "traces":
        print(json.dumps(store.list_traces(), indent=2, sort_keys=True))
        return
    if args.cmd == "info":
        print(json.dumps(store.summary(), indent=2, sort_keys=True))
        return


if __name__ == "__main__":
    main()
