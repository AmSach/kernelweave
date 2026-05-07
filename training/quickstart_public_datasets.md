# Public dataset quickstart

## Good open datasets for a PC-friendly first run
- **C4 / mC4**: broad web text
- **OpenWebText**: cleaner web text
- **The Stack** or **StarCoderData**: code
- **GSM8K** / math QA sets: reasoning
- **OpenAssistant / ShareGPT-style conversations**: instruction and dialogue
- **Tool-use traces** you create yourself from solved tasks

## What to train on first
1. general text
2. instruction data
3. code
4. math
5. reasoning traces
6. tool traces
7. skill kernels

## What not to do
- don’t mix everything randomly
- don’t train on raw chat logs without cleaning
- don’t use live-changing facts as memorised pretraining targets
- don’t skip evaluation just because the loss looks cute

## Data shape
Use JSONL whenever possible:
- `{"text":"..."}` for raw text
- `{"prompt":"...","response":"..."}` for instruction data
- `{"messages":[...]}` for chat
- `{"problem":"...","steps":"...","answer":"..."}` for reasoning
- `{"prompt":"...","tool_trace":[...],"final":"..."}` for agent traces

## Folder layout
```text
datasets/
  general_text/
  code/
  math/
  reasoning/
  dialogue/
  tool_traces/
  skills/
```

## Training discipline
- dedupe and filter first
- estimate token counts before training
- cap each source share
- hold out a clean eval split
- promote only if eval improves


## Which formats to prefer
- **Best for raw text**: JSONL with `{"text": ...}`
- **Best for instruction tuning**: JSONL with `{"prompt": ..., "response": ...}`
- **Best for chat**: JSONL with `{"messages": [...]}`
- **Best for reasoning**: JSONL with `{"problem": ..., "steps": [...], "answer": ...}`
- **Best for tools**: JSONL with `{"prompt": ..., "tool_trace": [...], "final": ...}`

## How to mix them
For a first serious run on a PC, use something like:
- 40% general text
- 20% instruction/dialogue
- 15% code
- 15% reasoning
- 5% tool traces
- 5% skills / promoted kernels

## What good data looks like
- clean
- deduped
- not full of boilerplate
- not full of garbage markdown or scrape noise
- aligned to the task you want the model to be good at
- split into train / validation / eval before training

## What bad data looks like
- random dumped web pages
- repeated spam
- unfiltered auto-generated text
- answers without questions
- wrong or stale facts
- unlabeled mixtures of everything
