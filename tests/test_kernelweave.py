from pathlib import Path

from kernelweave import KernelRuntime, KernelStore, TraceEvent, compile_trace_to_kernel, load_sample_store
from kernelweave.cli import install_samples
from kernelweave.metrics import cosine_similarity, coverage, jaccard_similarity


def test_metrics_are_reasonable():
    assert jaccard_similarity("safe shell command", "safe command") > 0.3
    assert cosine_similarity("compare two artifacts", "compare artifacts") > 0.2
    assert coverage(["evidence required", "rollback"], "the evidence is required and rollback exists") > 0.5


def test_compile_and_roundtrip(tmp_path: Path):
    store = load_sample_store(tmp_path)
    trace = [
        TraceEvent(kind="plan", payload={"text": "compare items"}),
        TraceEvent(kind="tool", payload={"tool": "load_artifact", "args": {"path": "A"}}),
        TraceEvent(kind="evidence", payload={"text": "items differ"}),
        TraceEvent(kind="verification", payload={"text": "summary references both"}),
    ]
    kernel = compile_trace_to_kernel("trace-1", "artifact comparison", "compare artifacts", trace, {"result": "ok"})
    path = store.add_kernel(kernel)
    reloaded = store.get_kernel(kernel.kernel_id)
    assert reloaded.digest() == kernel.digest()
    assert path.exists()
    assert reloaded.status.confidence >= 0.5


def test_runtime_prefers_kernel(tmp_path: Path):
    store = load_sample_store(tmp_path)
    install_samples(store)
    runtime = KernelRuntime(store)
    result = runtime.run("please compare two artifacts and explain the differences")
    assert result["mode"] == "kernel"
    assert result["kernel_id"]


def test_runtime_falls_back_when_prompt_is_unrelated(tmp_path: Path):
    store = load_sample_store(tmp_path)
    install_samples(store)
    runtime = KernelRuntime(store)
    result = runtime.run("write a poem about the moon")
    assert result["mode"] == "generate"


def test_store_summary(tmp_path: Path):
    store = load_sample_store(tmp_path)
    install_samples(store)
    summary = store.summary()
    assert summary["kernels"] >= 2
    assert summary["traces"] >= 2
