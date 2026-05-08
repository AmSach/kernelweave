#!/usr/bin/env python3
"""KernelWeave CLI - Complete kernel-native system.

Usage:
    # Initialize
    kernelweave init --store ./kernel_store
    
    # Install kernels
    kernelweave install-kernels --store ./kernel_store
    
    # Run a prompt
    kernelweave run "compare main.py and utils.py" --store ./kernel_store
    
    # Collect traces
    kernelweave collect --prompts prompts.jsonl --store ./kernel_store --output traces.jsonl
    
    # Train on traces
    kernelweave train --traces traces.jsonl --base-model Qwen/Qwen2.5-7B --output ./kernel-native-model
    
    # Serve API
    kernelweave serve --port 8080 --store ./kernel_store
    
    # Evaluate
    kernelweave eval --model ./kernel-native-model --tasks benchmark/tasks.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernelweave",
        description="Kernel-native language model framework",
    )
    
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # Init
    init_p = sub.add_parser("init", help="Initialize kernel store")
    init_p.add_argument("--store", type=Path, default=Path("./kernel_store"))
    
    # Install kernels
    install_p = sub.add_parser("install-kernels", help="Install sample kernels")
    install_p.add_argument("--store", type=Path, default=Path("./kernel_store"))
    
    # Run
    run_p = sub.add_parser("run", help="Execute prompt through kernel-native system")
    run_p.add_argument("prompt", help="Prompt to execute")
    run_p.add_argument("--store", type=Path, default=Path("./kernel_store"))
    run_p.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    run_p.add_argument("--auto-promote", action="store_true", help="Auto-promote successful traces")
    
    # Collect
    collect_p = sub.add_parser("collect", help="Collect execution traces")
    collect_p.add_argument("--prompts", type=Path, required=True, help="Path to prompts JSONL")
    collect_p.add_argument("--store", type=Path, default=Path("./kernel_store"))
    collect_p.add_argument("--output", type=Path, default=Path("traces.jsonl"))
    collect_p.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    
    # Train
    train_p = sub.add_parser("train", help="Train model on kernel execution traces")
    train_p.add_argument("--traces", type=Path, required=True, help="Path to traces JSONL")
    train_p.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    train_p.add_argument("--output", type=Path, default=Path("./kernel-native-model"))
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--learning-rate", type=float, default=5e-5)
    
    # Serve
    serve_p = sub.add_parser("serve", help="Start REST API server")
    serve_p.add_argument("--port", type=int, default=8080)
    serve_p.add_argument("--store", type=Path, default=Path("./kernel_store"))
    serve_p.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    
    # Eval
    eval_p = sub.add_parser("eval", help="Evaluate model")
    eval_p.add_argument("--model", type=Path, required=True, help="Path to trained model")
    eval_p.add_argument("--baseline", type=Path, help="Path to baseline model for comparison")
    eval_p.add_argument("--tasks", type=Path, required=True, help="Path to tasks JSONL")
    eval_p.add_argument("--output", type=Path, default=Path("eval_results.json"))
    
    # Stats
    stats_p = sub.add_parser("stats", help="Show model statistics")
    stats_p.add_argument("--store", type=Path, default=Path("./kernel_store"))
    
    # Export
    export_p = sub.add_parser("export", help="Export training data")
    export_p.add_argument("--store", type=Path, default=Path("./kernel_store"))
    export_p.add_argument("--output", type=Path, default=Path("training_data.jsonl"))
    
    return parser


def cmd_init(args):
    """Initialize kernel store."""
    from kernelweave import KernelStore
    
    store = KernelStore(args.store)
    print(f"✓ Initialized kernel store at {args.store}")
    print(f"  Kernels: {store.summary()['kernels']}")


def cmd_install_kernels(args):
    """Install sample kernels."""
    from kernelweave import KernelStore
    from kernelweave.cli import install_samples
    from kernelweave.kernels.library import install_kernel_library
    
    store = KernelStore(args.store)
    install_samples(store)
    install_kernel_library(store)
    
    print(f"✓ Installed kernels to {args.store}")
    print(f"  Total kernels: {store.summary()['kernels']}")


def cmd_run(args):
    """Run a prompt."""
    from kernelweave.model import KernelNativeModel, KernelNativeConfig
    
    config = KernelNativeConfig(
        base_model=args.base_model,
        kernel_store_path=str(args.store),
        enable_auto_promotion=args.auto_promote,
    )
    
    model = KernelNativeModel(
        base_model=args.base_model,
        kernel_store=args.store,
        config=config,
    )
    
    result = model.run(args.prompt)
    
    print("\n" + "=" * 60)
    print("EXECUTION RESULT")
    print("=" * 60)
    print(f"Mode: {result.mode}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Kernels: {result.kernel_ids}")
    print(f"Verification: {'✓ PASSED' if result.verification.get('passed') else '✗ FAILED'}")
    
    if result.promoted_kernel_id:
        print(f"Promoted: {result.promoted_kernel_id}")
    
    print("\nOUTPUT:")
    print("-" * 60)
    print(result.output[:500])
    if len(result.output) > 500:
        print(f"... ({len(result.output)} chars total)")


def cmd_collect(args):
    """Collect execution traces."""
    from kernelweave.model import KernelNativeModel, KernelNativeConfig
    from kernelweave.training import TraceCollector
    
    # Load prompts
    prompts = []
    with args.prompts.open() as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"] if isinstance(data, dict) else data)
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Create model
    config = KernelNativeConfig(
        base_model=args.base_model,
        kernel_store_path=str(args.store),
        enable_auto_promotion=True,
    )
    
    model = KernelNativeModel(
        base_model=args.base_model,
        kernel_store=args.store,
        config=config,
    )
    
    # Collect traces
    print("Collecting traces...")
    for i, prompt in enumerate(prompts):
        result = model.run(prompt)
        print(f"  [{i+1}/{len(prompts)}] mode={result.mode}, verified={result.verification.get('passed')}")
    
    # Save traces
    model.collector.save_traces(args.output)
    
    print(f"\n✓ Collected {len(model.collector.traces)} traces")
    print(f"  Verified: {len(model.collector.get_verified_traces())}")
    print(f"  Saved to: {args.output}")


def cmd_train(args):
    """Train model on traces."""
    from kernelweave.training import TraceTrainer, ExecutionTrace
    
    # Load traces
    traces = []
    with args.traces.open() as f:
        for line in f:
            traces.append(ExecutionTrace.from_dict(json.loads(line)))
    
    print(f"Loaded {len(traces)} traces")
    
    # Train
    trainer = TraceTrainer(
        base_model=args.base_model,
        output_dir=str(args.output),
    )
    
    result = trainer.train(
        traces=traces,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    print(f"\n✓ Training complete")
    print(f"  Output: {args.output}")
    print(f"  Status: {result.get('status')}")


def cmd_serve(args):
    """Start REST API server."""
    print(f"Starting server on port {args.port}...")
    print("Note: Full server implementation requires FastAPI/Flask")
    print("This is a placeholder - implement with your preferred framework")
    
    # Placeholder - would use FastAPI
    print(f"""
    Example endpoints:
    
    POST /run
    {{
        "prompt": "compare files A and B"
    }}
    
    Response:
    {{
        "mode": "kernel",
        "output": "...",
        "verification": {{"passed": true}},
        "kernel_ids": ["kw-xxx"]
    }}
    
    GET /stats
    {{
        "kernels": 22,
        "traces": 150,
        "verified": 120
    }}
    """)


def cmd_stats(args):
    """Show model statistics."""
    from kernelweave import KernelStore
    
    store = KernelStore(args.store)
    summary = store.summary()
    
    print("\n" + "=" * 60)
    print("KERNEL STORE STATISTICS")
    print("=" * 60)
    print(f"Kernels: {summary['kernels']}")
    print(f"Traces: {summary['traces']}")
    print(f"Feedback entries: {summary.get('feedback', 0)}")


def cmd_export(args):
    """Export training data."""
    from kernelweave import KernelStore
    from kernelweave.training import TraceCollector
    
    # This would load existing traces from store
    # Placeholder for now
    print("Export requires existing traces in store")
    print("Run 'kernelweave collect' first to generate traces")


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    commands = {
        "init": cmd_init,
        "install-kernels": cmd_install_kernels,
        "run": cmd_run,
        "collect": cmd_collect,
        "train": cmd_train,
        "serve": cmd_serve,
        "eval": None,  # TODO
        "stats": cmd_stats,
        "export": cmd_export,
    }
    
    if args.cmd in commands and commands[args.cmd]:
        commands[args.cmd](args)
    else:
        print(f"Command '{args.cmd}' not implemented yet")
        sys.exit(1)


if __name__ == "__main__":
    main()
