from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol
import json
import os
import urllib.error
import urllib.request


@dataclass
class ModelPreset:
    id: str
    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    system_prompt: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 60
    headers: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    context_window: int = 0
    default: bool = False

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("model preset id must not be empty")
        if not self.provider.strip():
            raise ValueError("model preset provider must not be empty")
        if self.provider not in {"openai", "openai-compatible", "anthropic", "ollama"}:
            raise ValueError("unsupported model preset provider")
        if not self.model.strip():
            raise ValueError("model name must not be empty")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.context_window < 0:
            raise ValueError("context_window must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.provider
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelPreset":
        provider = str(data.get("provider", data.get("kind", "openai-compatible")))
        return cls(
            id=str(data.get("id", data.get("name", ""))),
            provider=provider,
            model=str(data.get("model", "")),
            base_url=str(data.get("base_url", "")),
            api_key_env=str(data.get("api_key_env", "")),
            system_prompt=str(data.get("system_prompt", "")),
            temperature=float(data.get("temperature", 0.2)),
            max_tokens=int(data.get("max_tokens", data.get("max_new_tokens", 1024))),
            timeout_seconds=int(data.get("timeout_seconds", 60)),
            headers=dict(data.get("headers", {})),
            notes=list(data.get("notes", [])),
            context_window=int(data.get("context_window", 0)),
            default=bool(data.get("default", False)),
        )

    def resolve_base_url(self) -> str:
        if self.base_url.strip():
            return self.base_url.rstrip("/")
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        if self.provider == "anthropic":
            return "https://api.anthropic.com"
        if self.provider == "openai-compatible":
            return os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        if self.provider == "ollama":
            return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        raise ValueError(f"base_url is required for provider {self.provider}")

    def resolve_api_key(self) -> str | None:
        if self.api_key_env.strip():
            value = os.environ.get(self.api_key_env)
            if value:
                return value
        if self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if self.provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        if self.provider == "openai-compatible":
            return os.environ.get("OPENAI_COMPATIBLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return None


@dataclass
class ModelResponse:
    text: str
    provider: str
    model: str
    raw: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelBackend(Protocol):
    preset: ModelPreset

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        ...


class OpenAICompatibleBackend:
    def __init__(self, preset: ModelPreset):
        self.preset = preset

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        url = self.preset.resolve_base_url().rstrip("/") + "/chat/completions"
        body = {
            "model": self.preset.model,
            "messages": [],
            "temperature": self.preset.temperature if temperature is None else temperature,
            "max_tokens": self.preset.max_tokens if max_tokens is None else max_tokens,
        }
        effective_system = system_prompt or self.preset.system_prompt
        if effective_system:
            body["messages"].append({"role": "system", "content": effective_system})
        body["messages"].append({"role": "user", "content": prompt})
        headers = {
            "content-type": "application/json",
            **self.preset.headers,
        }
        api_key = self.preset.resolve_api_key()
        if api_key:
            headers.setdefault("authorization", f"Bearer {api_key}")
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.preset.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed: {exc.code} {detail}") from exc
        choices = payload.get("choices", [])
        text = ""
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict):
                    text = str(message.get("content", ""))
                if not text:
                    text = str(first.get("text", ""))
        return ModelResponse(text=text, provider=self.preset.provider, model=self.preset.model, raw=payload, usage=dict(payload.get("usage", {})))


class AnthropicBackend:
    def __init__(self, preset: ModelPreset):
        self.preset = preset

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        url = self.preset.resolve_base_url().rstrip("/") + "/v1/messages"
        body = {
            "model": self.preset.model,
            "max_tokens": self.preset.max_tokens if max_tokens is None else max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        effective_system = system_prompt or self.preset.system_prompt
        if effective_system:
            body["system"] = effective_system
        if temperature is not None:
            body["temperature"] = temperature
        elif self.preset.temperature:
            body["temperature"] = self.preset.temperature
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            **self.preset.headers,
        }
        api_key = self.preset.resolve_api_key()
        if api_key:
            headers.setdefault("x-api-key", api_key)
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.preset.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic request failed: {exc.code} {detail}") from exc
        content = payload.get("content", [])
        text = ""
        if content:
            first = content[0]
            if isinstance(first, dict):
                text = str(first.get("text", ""))
        return ModelResponse(text=text, provider=self.preset.provider, model=self.preset.model, raw=payload, usage=dict(payload.get("usage", {})))


class MockBackend:
    def __init__(self, preset: ModelPreset, response_text: str = ""):
        self.preset = preset
        self.response_text = response_text or f"mock response for {preset.id}"
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        call = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.calls.append(call)
        return ModelResponse(
            text=self.response_text,
            provider=self.preset.provider,
            model=self.preset.model,
            raw={"mock": True, "call": call},
            usage={},
        )


class OllamaBackend:
    """Backend for Ollama's native API with true constrained JSON generation.

    Uses Ollama's /api/chat endpoint with the ``format: "json"`` parameter
    for grammar-based constrained decoding.  When ``json_mode`` is True,
    Ollama constrains token sampling so that the output is guaranteed to be
    valid JSON — this is NOT retry-and-validate; it is real constrained
    decoding at the token level.
    """

    def __init__(self, preset: ModelPreset, json_mode: bool = False):
        self.preset = preset
        self.json_mode = json_mode

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Generate using Ollama's native /api/generate endpoint.

        When *json_mode* (or *json_schema*) is active, Ollama uses internal
        grammar sampling to guarantee structurally valid JSON.
        """
        url = self.preset.resolve_base_url().rstrip("/") + "/api/generate"
        
        # Combine system prompt and user prompt
        effective_system = system_prompt or self.preset.system_prompt
        combined_prompt = prompt
        if effective_system:
            combined_prompt = f"{effective_system}\n\n{prompt}"
            
        body: dict[str, Any] = {
            "model": self.preset.model,
            "prompt": combined_prompt,
            "stream": False,
            "options": {
                "temperature": self.preset.temperature if temperature is None else temperature,
            },
        }
        if max_tokens or self.preset.max_tokens:
            body["options"]["num_predict"] = max_tokens or self.preset.max_tokens

        # Determine whether to enable constrained JSON mode
        use_json = json_mode if json_mode is not None else self.json_mode
        if json_schema is not None:
            body["format"] = json_schema  # Ollama supports schema-object format
        elif use_json:
            body["format"] = "json"

        headers = {
            "content-type": "application/json",
            **self.preset.headers,
        }
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.preset.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed: {exc.code} {detail}") from exc

        text = str(payload.get("response", ""))
        usage = {}
        if "prompt_eval_count" in payload:
            usage["prompt_tokens"] = payload["prompt_eval_count"]
        if "eval_count" in payload:
            usage["completion_tokens"] = payload["eval_count"]
        return ModelResponse(
            text=text,
            provider="ollama",
            model=self.preset.model,
            raw=payload,
            usage=usage,
        )


class ModelCatalog:
    def __init__(self, presets: Iterable[ModelPreset] | None = None):
        self._presets: dict[str, ModelPreset] = {}
        for preset in presets or []:
            self.add(preset)

    def add(self, preset: ModelPreset) -> ModelPreset:
        preset.validate()
        self._presets[preset.id] = preset
        return preset

    def list(self) -> list[ModelPreset]:
        return sorted(self._presets.values(), key=lambda item: item.id)

    def get(self, preset_id: str) -> ModelPreset:
        try:
            return self._presets[preset_id]
        except KeyError as exc:
            raise KeyError(f"unknown model preset: {preset_id}") from exc

    def summary(self) -> dict[str, Any]:
        presets = self.list()
        return {
            "presets": len(presets),
            "providers": sorted({preset.provider for preset in presets}),
            "defaults": [preset.id for preset in presets if preset.default],
        }

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "ModelCatalog":
        catalog = cls()
        for path in paths:
            if path.is_dir():
                for item in sorted(path.glob("*.json")):
                    catalog.add(ModelPreset.from_dict(json.loads(item.read_text())))
            elif path.is_file() and path.suffix.lower() == ".json":
                catalog.add(ModelPreset.from_dict(json.loads(path.read_text())))
        return catalog

    @classmethod
    def default_paths(cls) -> list[Path]:
        repo_root = Path(__file__).resolve().parents[2]
        cwd = Path.cwd()
        paths = [cwd / "models", repo_root / "models"]
        unique: list[Path] = []
        for path in paths:
            if path.exists() and path not in unique:
                unique.append(path)
        return unique

    @classmethod
    def load_default(cls) -> "ModelCatalog":
        catalog = cls()
        for path in cls.default_paths():
            if path.is_dir():
                for item in sorted(path.glob("*.json")):
                    catalog.add(ModelPreset.from_dict(json.loads(item.read_text())))
        return catalog


def backend_from_preset(preset: ModelPreset) -> ModelBackend:
    if preset.provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleBackend(preset)
    if preset.provider == "anthropic":
        return AnthropicBackend(preset)
    if preset.provider == "ollama":
        return OllamaBackend(preset)
    raise ValueError(f"unsupported provider: {preset.provider}")


def run_preset(prompt: str, preset: ModelPreset, system_prompt: str = "", temperature: float | None = None, max_tokens: int | None = None) -> ModelResponse:
    backend = backend_from_preset(preset)
    return backend.generate(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)
