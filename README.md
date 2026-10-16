# gx — Local LLM CLI for Termux / Android

[![CI](https://github.com/Dilipkumarkv/LocalCLI/actions/workflows/ci.yml/badge.svg)](https://github.com/Dilipkumarkv/LocalCLI/actions/workflows/ci.yml)

**gx** is a lightweight, single-file CLI client for a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` running on **Termux / Android** (or any Linux/macOS box). It uses **only the Python
standard library** — no pip, no venv, no dependencies to fight on a phone.

```
$ gx ask "explain quantization in one paragraph"
...streams tokens live from your on-device model...
[87 tokens, 21.4 tok/s]
```

## Features

- **Streaming chat** — `ask` (one-shot), `chat` (interactive REPL with `/exit`, `/clear`, `/sys`)
- **Sessions** — auto-save/resume conversations (`--save`, `--load`), list/show/delete/export to Markdown
- **Personas** — switch system prompt + sampling params + pinned model (`coder`, `tutor`, `terse` built in)
- **Prompt templates** — `{{variable}}` templates, render with `gx run tpl_name key=value ...`
- **Agent mode** — autonomous tool loop (read/write/list/search files, calc) with a **validated sandbox** and human confirmation for dangerous tools
- **Benchmarks** — TTFT + tok/s, thread sweeps, thermal-throttle detection, JSONL history
- **Model manager** — list/select GGUF files with fuzzy matching; `pull` downloads with resume (HTTP Range), Hugging Face blob-URL auto-fix, disk-space checks
- **Server lifecycle** — `gx server start|stop|status` with readiness polling and log capture
- **Batch loops** — repeat a prompt, run prompts from a file, or refine until a condition is met
- **Doctor** — one-command health check of the whole Termux setup

## Requirements

- Python **3.8+** (`pkg install python`)
- `llama-server` binary (`pkg install llama-cpp`, or build llama.cpp)
- One or more `.gguf` model files

## Install (Termux)

```bash
pkg install python llama-cpp git
git clone https://github.com/Dilipkumarkv/LocalCLI.git
cp LocalCLI/gx.py $PREFIX/bin/gx
chmod +x $PREFIX/bin/gx

# get a model and go
mkdir -p ~/models
gx pull "https://huggingface.co/<user>/<repo>/blob/main/<model>.Q4_K_M.gguf"
gx model use <model>       # fuzzy match on the filename
gx server start            # starts llama-server and waits until it is ready
gx ask "hello, on-device AI"
```

gx keeps its state in `~/.gx/` (config, sessions, personas, templates, logs). Set `GX_HOME` to relocate it.

## Commands

| Command | Purpose |
|---|---|
| `gx ask "..." [--persona p]` | One-shot prompt with streaming output; stdin is merged (`cat file \| gx ask "summarize"`) |
| `gx chat [--save name] [--load name]` | Interactive chat with context guard and auto-save |
| `gx session list\|show\|rm\|export <name>` | Manage saved sessions |
| `gx persona list\|show\|add\|edit\|rm` | Manage personas |
| `gx tpl list\|show\|add\|rm` | Manage prompt templates |
| `gx run <template> key=value ...` | Render + send a template |
| `gx agent "task" [--trace] [--max-steps N] [--dir sandbox] [-y] [--save name]` | Autonomous tool-using agent |
| `gx bench [--runs N] [--sweep threads=2,4,6] [--history]` | Benchmark TTFT / generation speed |
| `gx pull <url> [--force] [--out name]` | Download GGUF with resume support (`HF_TOKEN` honored) |
| `gx model list\|use <name>` | List / select models from `~/models` |
| `gx set <key> <value>` | Update config (see below) |
| `gx server start\|stop\|status` | Manage the llama-server process |
| `gx loop "prompt" [-n 5 \| -f file \| --until text]` | Repetitive / batch runs |
| `gx doctor` | Diagnose the environment |
| `gx selftest` | Run the built-in test suite (no server needed) |

Piping with no subcommand also works: `echo "hi" | gx` sends "hi" as a prompt.

## Configuration

`~/.gx/config.json`, edited via `gx set <key> <value>`:

| Key | Default | Meaning |
|---|---|---|
| `temp` | `0.7` | Sampling temperature |
| `ctx` | `2048` | Context size passed to llama-server |
| `threads` | `4` | CPU threads passed to llama-server |
| `max_tokens` | `256` | Max tokens per generation |
| `system` | `""` | Default system prompt |
| `model` | `""` | Active GGUF filename |
| `models_dir` | `~/models` | Where GGUF files live |
| `host` / `port` | `127.0.0.1` / `8080` | llama-server endpoint |
| `agent_shell` | `true` | Allow the agent's `shell` tool (see Security) |

Environment overrides: `GX_HOME` (state dir), `GX_URL` (full base URL), `GX_HOST`, `GX_PORT`, `HF_TOKEN` (gated HF downloads).

## Security notes for `gx agent`

- The **file tools** (`read_file`, `write_file`, `list_dir`, `search_files`) are strictly confined to the sandbox directory (`--dir`, default `.`). Path traversal (`..`), absolute escapes, and symlink escapes are rejected.
- `calc` is a sandboxed AST arithmetic evaluator — no `eval()`, no attribute access.
- ⚠️ The **`shell` tool is NOT filesystem-confined**: it runs a real shell with the sandbox as working directory, so a model can reach anything your user account can. Mitigations:
  - every `shell` (and overwrite `write_file`) call asks for interactive confirmation **unless you pass `-y`/`--yes`**;
  - it can be disabled entirely: `gx set agent_shell false` (the tool then reports it is disabled and the agent falls back to file tools);
  - avoid `--yes` with untrusted prompts — small local models are easy to prompt-inject.
- Agent runs are appended to `~/.gx/agent_log.jsonl` (task, steps, tools used, outcome) for auditing.

## Testing & CI

The suite runs against a fully mocked llama-server (SSE streaming, HTTP Range downloads, scripted agent conversations) — no model or network needed:

```bash
python gx.py selftest      # or: python -m unittest test_gx -v
```

CI runs it on Ubuntu across Python 3.9–3.13 on every push/PR.

## Limitations

- Token counts use a ~4-chars-per-token heuristic (offline-friendly); the context guard is therefore approximate.
- Designed primarily for Termux/Linux; on Windows the sandbox symlink-escape test is skipped unless symlink privileges are enabled.
- `server` management expects the `llama-server` OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`).

## License

[MIT](LICENSE) © 2025 Dilip Kumar K V
