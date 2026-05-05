from __future__ import annotations

import argparse
from pathlib import Path
import json

from .compiler import compile_trace_to_kernel
from .kernel import KernelStore, TraceEvent, load_sample_store
from .runtime import KernelRuntime


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

    return parser


def install_samples(store: KernelStore) -> None:
    trace = {
        "trace_id": "trace-safe-shell-001",
        "events": [
            {"kind": "plan", "payload": {"text": "classify request and build safe command"}},
            {"kind": "tool", "payload": {"tool": "parse_intent", "args": {"mode": "safety"}}},
            {"kind": "evidence", "payload": {"text": "command violates policy if destructive"}},
            {"kind": "verification", "payload": {"text": "command is non-destructive"}},
            {"kind": "decision", "payload": {"text": "emit safe shell command"}},
        ],
    }
    store.add_trace(trace["trace_id"], trace)
    events = [TraceEvent(kind=item["kind"], payload=item["payload"]) for item in trace["events"]]
    kernel = compile_trace_to_kernel(
        trace_id=trace["trace_id"],
        task_family="safe shell command synthesis",
        description="Turn plain language into a guarded shell command when the action is safe.",
        events=events,
        expected_output={"result": "safe command"},
    )
    store.add_kernel(kernel)

    trace2 = {
        "trace_id": "trace-compare-002",
        "events": [
            {"kind": "plan", "payload": {"text": "compare two artifacts and explain differences"}},
            {"kind": "tool", "payload": {"tool": "load_artifact", "args": {"path": "A"}}},
            {"kind": "tool", "payload": {"tool": "load_artifact", "args": {"path": "B"}}},
            {"kind": "evidence", "payload": {"text": "differences are structural and numerical"}},
            {"kind": "verification", "payload": {"text": "summary mentions both artifacts"}},
        ],
    }
    store.add_trace(trace2["trace_id"], trace2)
    events2 = [TraceEvent(kind=item["kind"], payload=item["payload"]) for item in trace2["events"]]
    kernel2 = compile_trace_to_kernel(
        trace_id=trace2["trace_id"],
        task_family="artifact comparison",
        description="Compare two structured artifacts and produce a grounded summary.",
        events=events2,
        expected_output={"result": "comparison summary"},
    )
    store.add_kernel(kernel2)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
