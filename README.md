# slm-offline-ai-agent

A ReAct agent (think → call tool → observe → repeat) that runs entirely on your laptop.
No API key, no credit card, no network after setup. The reasoning loop is hand-written —
no LangChain, no smolagents — so nothing is hidden from you.

**Six tools:** `calculator`, `write_note`, `read_note`, `list_notes`, `current_datetime`, `word_count`.

## Setup (Windows / CMD)

```cmd
winget install Ollama.Ollama
ollama --version           :: want 0.31.x or newer
ollama pull qwen3:4b       :: ~2.6 GB, one time
ollama list                :: confirm qwen3:4b is there
```

Then, in the project folder:

```cmd
uv sync
copy .env.example .env
```

No `uv`? `py -m venv .venv && .venv\Scripts\activate && pip install ollama python-dotenv pytest`

## Run

```cmd
uv run python -m src.main "what is 15%% of 1240? save the answer to tip.txt"
uv run python -m src.main --trace "read tip.txt and tell me the word count"
uv run python -m src.main                :: interactive REPL
```

`--trace` prints every tool call and result to stderr as it happens — that's the whole
point of writing the loop yourself, so watch it at least once.

## Prove it's offline

Start a REPL, run one task, then **turn on airplane mode** and run another. It keeps working.
Ollama serves the model at `127.0.0.1:11434` — "this computer only." Nothing leaves the laptop.

## Test

```cmd
uv run pytest
```

The tests script a fake model client, so the whole suite runs with no Ollama and no network.
They cover the two things worth caring about:

- **The calculator never calls `eval()`.** It parses to an AST and evaluates only the
  operator node types in `_ALLOWED_OPS`. A model's output is untrusted input — handing it
  to `eval()` is arbitrary code execution on your machine.
- **File tools can't escape `sandbox/`.** `_safe_path` resolves the path and checks
  `is_relative_to(SANDBOX_DIR)`, so `../../important_file.txt` raises instead of writing.

Also covered: a tool exception is returned to the model *as text* rather than crashing the
run (the model can then recover), unknown tool names are handled, and `MAX_STEPS` caps a
model that would otherwise call tools forever.

## Layout

```
src/config.py   every constant: model, MAX_STEPS, paths, system prompt
src/tools.py    the six tools + safe eval + path sandboxing
src/agent.py    the ReAct loop (client is injectable, which is why it's testable)
src/main.py     CLI: one-shot, REPL, --trace
tests/          the whole thing, no network required
```

## Model choice

`qwen3:4b` is the smallest one that chains multiple tool calls reliably. `llama3.2:3b`
(~2 GB) works if RAM is tight but will occasionally skip a tool call it should have made.
Swap it in `.env` (`OLLAMA_MODEL=`) — nothing else hardcodes the model.

## Where to take it next

The loop is the reusable part. Swap the tool list and you have a different agent:

- Point the tools at Toast/Strava JSON on disk → a local "ask my data" agent with zero
  cloud exposure.
- Add a `retrieve(query)` tool over a local embedding index → offline RAG, and now the
  CRAG self-grading step is just another tool call.
- Log every `(task, tool_calls, final_answer)` triple → that's your eval set for free.
