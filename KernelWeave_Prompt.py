#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def build_inputs(tokenizer, prompt: str, system_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
    return tokenizer(text, return_tensors="pt")


def load_model(mode: str, model_dir: Path, base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model if mode == "adapter" else str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if mode == "merged":
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()
    return tokenizer, model


def generate(model, tokenizer, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float):
    import torch

    inputs = build_inputs(tokenizer, prompt, system_prompt)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_tokens = int(inputs["input_ids"].shape[1])
    text = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelWeave prompt runner")
    parser.add_argument("--mode", choices=["merged", "adapter"], default="merged")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to merged_model/ or adapter/")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model for adapter mode")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are KernelWeave. Use verification-driven, concise, grounded answers.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer, model = load_model(args.mode, args.model_dir, args.base_model)
    output = generate(model, tokenizer, args.prompt, args.system, args.max_new_tokens, args.temperature)
    print(output)


if __name__ == "__main__":
    main()
