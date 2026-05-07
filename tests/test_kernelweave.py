from pathlib import Path

from kernelweave import KernelRuntime, KernelStore, TraceEvent, compile_trace_to_kernel, load_sample_store, ModelCatalog, ModelPreset, backend_from_preset, MockBackend, KernelWeaveLLM, LLMConfig
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


def test_model_catalog_and_mock_backend(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    preset_path = model_dir / "mock.json"
    preset_path.write_text(
        '{"id":"mock","provider":"openai-compatible","model":"mock-model","base_url":"http://127.0.0.1:11434/v1","default":true}'
    )
    catalog = ModelCatalog.from_paths([model_dir])
    preset = catalog.get("mock")
    backend = MockBackend(preset, response_text="hello from mock")
    response = backend.generate("hello")
    assert response.model == "mock-model"
    assert response.text == "hello from mock"


def test_kernel_weave_llm_responds_with_mock_backend(tmp_path: Path):
    store = load_sample_store(tmp_path)
    install_samples(store)
    preset = ModelPreset(id="mock", provider="openai-compatible", model="mock-model", base_url="http://127.0.0.1:11434/v1")
    backend = MockBackend(preset, response_text="wrapped response")
    wrapper = KernelWeaveLLM(LLMConfig.compact_frontier(), kernel_store=store, backend=backend)
    result = wrapper.respond("compare two artifacts")
    assert result["mode"] == "hybrid"
    assert result["text"] == "wrapped response"
    assert result["routing"]["routing"] in {"kernel", "generate", "agent", "skill"}
