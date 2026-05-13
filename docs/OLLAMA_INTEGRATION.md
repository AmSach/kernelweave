# KernelWeave Local Ollama Integration

KernelWeave has a direct, seamless integration with your local **Ollama** setup, allowing for completely offline, zero-dependency, frontier-grade execution.

## Getting Started Quickly

Just like tools such as Open WebUI, you can launch KernelWeave attached natively to Ollama with a single command. 

Open your terminal or PowerShell in the `kernelweave` directory and run:

**On Windows (Batch):**
```cmd
.\kernelweave-ollama.bat
```

**On Windows (PowerShell):**
```powershell
.\kernelweave-ollama.ps1
```

*(By default, this attaches to `gemma4:e2b`. If you want to use a different model, simply pass it as an argument: `.\kernelweave-ollama.bat granite4.1:8b`)*

### What happens when you run this?
1. It launches a beautifully formatted interactive terminal shell.
2. It natively binds the **KernelWeave Router**, the **Execution Engine**, the **Production LLM Judge**, and the **Self-Compiler** directly to your Ollama `/api/generate` endpoint.
3. As you chat, the engine routes your prompts. If an existing compiled "Skill Kernel" matches, it executes it with true token-level constrained JSON decoding. If not, it generates an answer and the compiler silently extracts the trace into a new Kernel for the future.

## How it works under the hood

The integration relies on three key pillars:

1. **`OllamaBackend` (in `kernelweave/llm/providers.py`)**: 
   This connects directly to `http://127.0.0.1:11434/api/generate`. It bypasses the need for Heavy PyTorch or Transformers dependencies in the local environment.

2. **Native JSON Grammar Constraints**: 
   When a Kernel specifies a strict output schema via its postconditions, KernelWeave passes `format: "json"` alongside the structured schema directly to Ollama. Ollama's C++ inference engine enforces this at the token level, meaning it is mathematically impossible for the model to output invalid JSON structure.

3. **Production Verifier (`VerifierHierarchy`)**:
   After generation, the `LLMJudgeVerifier` calls Ollama in the background to ensure all nuanced criteria (e.g., "the tone was professional") are verified before trusting the response.

## Running the Showcase Demo

If you want to see all these features tested automatically, you can run the showcase script:
```powershell
python samples\frontier_demo.py
```
This will automatically execute 6 different benchmarks against your local Ollama setup to prove the router, the JSON constraint engine, and the self-compiler are all functioning perfectly offline.
