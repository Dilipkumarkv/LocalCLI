#!/usr/bin/env python3
"""gx - Lightweight CLI client for local llama.cpp llama-server on Termux / Android.

Uses ONLY the Python standard library.
"""

import argparse
import ast
import io
import json
import operator
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

# Configuration Constants
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.gx")
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "config.json")
DEFAULT_PID_FILE = os.path.join(DEFAULT_CONFIG_DIR, "server.pid")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "server.log")
DEFAULT_SESSIONS_DIR = os.path.join(DEFAULT_CONFIG_DIR, "sessions")
DEFAULT_PERSONAS_DIR = os.path.join(DEFAULT_CONFIG_DIR, "personas")
DEFAULT_TEMPLATES_DIR = os.path.join(DEFAULT_CONFIG_DIR, "templates")
DEFAULT_BENCH_HISTORY = os.path.join(DEFAULT_CONFIG_DIR, "bench_history.jsonl")
DEFAULT_AGENT_LOG = os.path.join(DEFAULT_CONFIG_DIR, "agent_log.jsonl")

DEFAULT_CONFIG: Dict[str, Any] = {
    "temp": 0.7,
    "ctx": 2048,
    "threads": 4,
    "max_tokens": 256,
    "system": "",
    "model": "",
    "models_dir": "~/models",
    "host": "127.0.0.1",
    "port": 8080,
}

BUILTIN_PERSONAS: Dict[str, Dict[str, Any]] = {
    "coder": {
        "name": "coder",
        "system": "You are an expert programming assistant. Provide concise, clean, code-first answers with minimal explanation.",
        "temp": 0.2,
        "max_tokens": 1024,
    },
    "tutor": {
        "name": "tutor",
        "system": "You are a friendly tutor. Explain concepts step by step clearly and simply, then ask exactly one check question at the end to verify understanding.",
        "temp": 0.7,
        "max_tokens": 512,
    },
    "terse": {
        "name": "terse",
        "system": "You are an extremely concise assistant. Provide direct, single-line answers whenever possible without filler.",
        "temp": 0.3,
        "max_tokens": 128,
    },
}

BUILTIN_TEMPLATES: Dict[str, str] = {
    "summarize": "Summarize the following content in {{style}} style:\n\n{{input}}",
    "explain": "Explain {{topic}} to a {{level}} audience clearly with examples.",
}


# Path & Directory Helpers


def get_config_dir() -> str:
    """Return the configuration directory path, creating it if needed."""
    cfg_dir = os.environ.get("GX_HOME", DEFAULT_CONFIG_DIR)
    os.makedirs(cfg_dir, exist_ok=True)
    return cfg_dir


def get_config_file() -> str:
    """Return path to config.json."""
    return os.path.join(get_config_dir(), "config.json")


def get_pid_file() -> str:
    """Return path to server.pid."""
    return os.path.join(get_config_dir(), "server.pid")


def get_log_file() -> str:
    """Return path to server.log."""
    return os.path.join(get_config_dir(), "server.log")


def get_sessions_dir() -> str:
    """Return path to sessions directory, creating it if needed."""
    sessions_dir = os.path.join(get_config_dir(), "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    return sessions_dir


def get_personas_dir() -> str:
    """Return path to personas directory, creating it if needed."""
    personas_dir = os.path.join(get_config_dir(), "personas")
    os.makedirs(personas_dir, exist_ok=True)
    return personas_dir


def get_templates_dir() -> str:
    """Return path to templates directory, creating it if needed."""
    templates_dir = os.path.join(get_config_dir(), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    return templates_dir


def get_bench_history_file() -> str:
    """Return path to bench_history.jsonl."""
    return os.path.join(get_config_dir(), "bench_history.jsonl")


def get_agent_log_file() -> str:
    """Return path to agent_log.jsonl."""
    return os.path.join(get_config_dir(), "agent_log.jsonl")


def append_agent_log(entry: Dict[str, Any]) -> None:
    """Append agent execution record to agent_log.jsonl."""
    log_file = get_agent_log_file()
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def validate_session_name(name: str) -> bool:
    """Validate that name contains only letters, digits, hyphens, and underscores."""
    if not name or not isinstance(name, str):
        return False
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


# Built-ins, Personas & Templates Initializers


def ensure_builtin_personas() -> None:
    """Seed built-in personas on first run if missing."""
    p_dir = get_personas_dir()
    for name, p_data in BUILTIN_PERSONAS.items():
        f_path = os.path.join(p_dir, f"{name}.json")
        if not os.path.isfile(f_path):
            try:
                with open(f_path, "w", encoding="utf-8") as f:
                    json.dump(p_data, f, indent=2)
            except Exception:
                pass


def ensure_builtin_templates() -> None:
    """Seed built-in templates on first run if missing."""
    t_dir = get_templates_dir()
    for name, t_content in BUILTIN_TEMPLATES.items():
        f_path = os.path.join(t_dir, f"{name}.txt")
        if not os.path.isfile(f_path):
            try:
                with open(f_path, "w", encoding="utf-8") as f:
                    f.write(t_content)
            except Exception:
                pass


def load_persona(name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load persona by name. Returns (persona_dict, error_message)."""
    ensure_builtin_personas()
    if not validate_session_name(name):
        return None, f"Invalid persona name '{name}'."
    f_path = os.path.join(get_personas_dir(), f"{name}.json")
    if not os.path.isfile(f_path):
        return None, f"Persona '{name}' not found."
    try:
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return None, f"Persona file '{name}.json' is corrupted."
            return data, None
    except Exception as e:
        return None, f"Error reading persona '{name}': {e}"


def save_persona(name: str, data: Dict[str, Any]) -> bool:
    """Save persona dictionary to disk."""
    if not validate_session_name(name):
        return False
    f_path = os.path.join(get_personas_dir(), f"{name}.json")
    try:
        with open(f_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        sys.stderr.write(f"Error saving persona '{name}': {e}\n")
        return False


def apply_persona(
    persona_name: str, config: Dict[str, Any], models_dir: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Apply persona defaults onto config. Checks for pinned model presence."""
    p_data, err = load_persona(persona_name)
    if err:
        return config, err

    new_cfg = dict(config)
    if p_data.get("system"):
        new_cfg["system"] = p_data["system"]
    if p_data.get("temp") is not None:
        new_cfg["temp"] = float(p_data["temp"])
    if p_data.get("max_tokens") is not None:
        new_cfg["max_tokens"] = int(p_data["max_tokens"])

    pinned_model = p_data.get("model", "").strip()
    if pinned_model:
        avail_models = list_gguf_models(models_dir)
        matched, _ = match_model(pinned_model, avail_models)
        if matched:
            new_cfg["model"] = matched
        else:
            cur_model = new_cfg.get("model", "")
            sys.stderr.write(
                f"Warning: Persona '{persona_name}' pinned model '{pinned_model}' not found in {models_dir}. Falling back to active model '{cur_model}'.\n"
            )

    return new_cfg, None


def load_template(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Load prompt template text by name."""
    ensure_builtin_templates()
    if not validate_session_name(name):
        return None, f"Invalid template name '{name}'."
    f_path = os.path.join(get_templates_dir(), f"{name}.txt")
    if not os.path.isfile(f_path):
        return None, f"Template '{name}' not found."
    try:
        with open(f_path, "r", encoding="utf-8") as f:
            return f.read(), None
    except Exception as e:
        return None, f"Error reading template '{name}': {e}"


def save_template(name: str, content: str) -> bool:
    """Save prompt template text to disk."""
    if not validate_session_name(name):
        return False
    f_path = os.path.join(get_templates_dir(), f"{name}.txt")
    try:
        with open(f_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        sys.stderr.write(f"Error saving template '{name}': {e}\n")
        return False


def extract_quantization(filename: str) -> str:
    """Extract quantization string from model filename (e.g. Q4_K_M, Q8_0, BF16)."""
    clean = os.path.basename(filename)
    if clean.lower().endswith(".gguf"):
        clean = clean[:-5]
    match = re.search(r"(q[0-9]+_[a-z0-9_]+|bf16|f16|f32|q[0-9]+_[0-9]|q[0-9]+_[a-z])", clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


# Sessions & Context Guard


def get_session_file(name: str) -> str:
    """Return absolute path to a session JSON file."""
    return os.path.join(get_sessions_dir(), f"{name}.json")


def save_session(name: str, model: str, system: str, messages: List[Dict[str, str]]) -> bool:
    """Save session to disk."""
    if not validate_session_name(name):
        return False
    filepath = get_session_file(name)
    data = {
        "name": name,
        "model": model,
        "system": system,
        "messages": messages,
        "updated_at": time.time(),
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        sys.stderr.write(f"Error saving session '{name}': {e}\n")
        return False


def load_session(name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load session from disk. Returns (data, error_message)."""
    if not validate_session_name(name):
        return None, f"Invalid session name '{name}'. Use only letters, digits, hyphens, and underscores."
    filepath = get_session_file(name)
    if not os.path.isfile(filepath):
        return None, f"Session '{name}' not found in {get_sessions_dir()}."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return None, f"Session file for '{name}' is corrupted (not a JSON object)."
            return data, None
    except json.JSONDecodeError:
        return None, f"Session file for '{name}' is corrupted (invalid JSON syntax)."
    except Exception as e:
        return None, f"Error reading session '{name}': {e}"


def estimate_token_count(messages: List[Dict[str, str]]) -> int:
    """Estimate token count based on standard ~4 characters per token heuristic."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, total_chars // 4)


def apply_context_guard(
    messages: List[Dict[str, str]], ctx_limit: int, threshold_pct: float = 0.8
) -> Tuple[List[Dict[str, str]], int]:
    """Ensure conversation history does not exceed threshold% of context limit.

    Drops oldest user/assistant pairs while preserving system messages.
    Returns (trimmed_messages, count_of_dropped_messages).
    """
    max_allowed = int(ctx_limit * threshold_pct)
    if estimate_token_count(messages) <= max_allowed:
        return messages, 0

    system_msgs = [m for m in messages if m.get("role") == "system"]
    chat_msgs = [m for m in messages if m.get("role") != "system"]

    dropped_count = 0
    while chat_msgs and estimate_token_count(system_msgs + chat_msgs) > max_allowed:
        if len(chat_msgs) >= 2 and chat_msgs[0].get("role") == "user" and chat_msgs[1].get("role") == "assistant":
            chat_msgs.pop(0)
            chat_msgs.pop(0)
            dropped_count += 2
        else:
            chat_msgs.pop(0)
            dropped_count += 1

    return system_msgs + chat_msgs, dropped_count


# Configuration & URL Resolution


def load_config() -> Dict[str, Any]:
    """Load configuration with default fallbacks."""
    cfg = dict(DEFAULT_CONFIG)
    cfg_file = get_config_file()
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    cfg.update(loaded)
        except Exception:
            pass
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Save configuration dictionary to config.json."""
    cfg_file = get_config_file()
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_base_url(
    config: Dict[str, Any],
    cli_host: Optional[str] = None,
    cli_port: Optional[int] = None,
    cli_url: Optional[str] = None,
) -> str:
    """Resolve the base URL in priority order: CLI args > Env var > Config > Default."""
    if cli_url:
        return cli_url.rstrip("/")
    env_url = os.environ.get("GX_URL")
    if env_url:
        return env_url.rstrip("/")

    host = cli_host or os.environ.get("GX_HOST") or config.get("host", "127.0.0.1")
    port = (
        cli_port
        or (int(os.environ["GX_PORT"]) if "GX_PORT" in os.environ else None)
        or config.get("port", 8080)
    )
    return f"http://{host}:{port}"


# Models Management


def get_models_dir(config: Dict[str, Any]) -> str:
    """Return resolved models directory."""
    raw = config.get("models_dir", "~/models")
    return os.path.expanduser(raw)


def list_gguf_models(models_dir: str) -> List[str]:
    """Return list of .gguf files in models_dir."""
    if not os.path.isdir(models_dir):
        return []
    try:
        entries = os.listdir(models_dir)
        return sorted(
            [e for e in entries if e.lower().endswith(".gguf") and os.path.isfile(os.path.join(models_dir, e))]
        )
    except Exception:
        return []


def match_model(query: str, available_models: List[str]) -> Tuple[Optional[str], List[str]]:
    """Match a query string to available model filenames.

    Supports exact match, case-insensitive match, and substring matches.
    Returns (best_match_filename_or_None, list_of_all_matching_candidates).
    """
    if not available_models:
        return None, []

    if query in available_models:
        return query, [query]

    q_lower = query.lower()
    q_clean = q_lower[:-5] if q_lower.endswith(".gguf") else q_lower

    exact_no_ext = [m for m in available_models if (m[:-5] if m.lower().endswith(".gguf") else m).lower() == q_clean]
    if len(exact_no_ext) == 1:
        return exact_no_ext[0], exact_no_ext

    matches = [m for m in available_models if q_clean in m.lower()]
    if len(matches) == 1:
        return matches[0], matches

    prefix_matches = [m for m in available_models if m.lower().startswith(q_clean)]
    if len(prefix_matches) == 1:
        return prefix_matches[0], prefix_matches

    return None, matches if matches else prefix_matches


def format_file_size(bytes_size: int) -> str:
    """Format bytes to human readable size (MB/GB)."""
    if bytes_size >= 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 ** 3):.2f} GB"
    if bytes_size >= 1024 * 1024:
        return f"{bytes_size / (1024 ** 2):.1f} MB"
    return f"{bytes_size / 1024:.1f} KB"


# Server Process Management & Stdin Helpers


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def get_running_server_pid() -> Optional[int]:
    """Return PID if server is actively running, else None."""
    pid_file = get_pid_file()
    if not os.path.isfile(pid_file):
        return None
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if is_pid_alive(pid):
            return pid
        try:
            os.remove(pid_file)
        except OSError:
            pass
    except Exception:
        pass
    return None


def find_llama_server_binary() -> Optional[str]:
    """Find llama-server binary in PATH or common Termux paths."""
    found = shutil.which("llama-server")
    if found:
        return found

    candidates = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        os.path.expanduser("~/llama.cpp/llama-server"),
        "/data/data/com.termux/files/usr/bin/llama-server",
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def read_piped_stdin() -> str:
    """Read stdin if data is being piped into gx without blocking."""
    try:
        if not sys.stdin.isatty():
            if hasattr(select, "select") and hasattr(sys.stdin, "fileno"):
                try:
                    fd = sys.stdin.fileno()
                    rlist, _, _ = select.select([fd], [], [], 0.0)
                    if rlist:
                        return sys.stdin.read().strip()
                    return ""
                except (io.UnsupportedOperation, ValueError, OSError):
                    content = sys.stdin.read()
                    return content.strip() if content else ""
            else:
                content = sys.stdin.read()
                return content.strip() if content else ""
    except Exception:
        return ""
    return ""


# SSE Parsing & Networking


def parse_sse_chunks(raw_bytes: bytes, buffer: str) -> Tuple[List[str], str, bool]:
    """Parse raw stream bytes for OpenAI/llama.cpp SSE data lines.

    Returns (list_of_tokens, remaining_buffer, is_done).
    """
    buffer += raw_bytes.decode("utf-8", errors="replace")
    tokens: List[str] = []
    is_done = False

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line or line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                is_done = True
                buffer = buffer.strip()
                if not buffer:
                    buffer = ""
                break
            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])
                if choices:
                    first = choices[0]
                    delta = first.get("delta", {})
                    token = delta.get("content", "")
                    if not token and "text" in first:
                        token = first["text"]
                    if token:
                        tokens.append(token)
            except json.JSONDecodeError:
                continue

    return tokens, buffer, is_done


def stream_chat_completion_detailed(
    messages: List[Dict[str, str]],
    base_url: str,
    config: Dict[str, Any],
    on_token: Optional[Callable[[str], None]] = None,
    max_tokens_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Send chat request and collect performance metrics (TTFT, prompt tok/s, gen tok/s, timings)."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    max_tokens = max_tokens_override if max_tokens_override is not None else int(config.get("max_tokens", 256))
    payload = {
        "messages": messages,
        "temperature": float(config.get("temp", 0.7)),
        "max_tokens": max_tokens,
        "stream": True,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    full_text = []
    token_count = 0
    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    buffer = ""
    server_timings: Optional[Dict[str, Any]] = None

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            while True:
                chunk = response.read(64)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            buffer = ""
                            break
                        try:
                            data = json.loads(data_str)
                            if "timings" in data:
                                server_timings = data["timings"]
                            choices = data.get("choices", [])
                            if choices:
                                first = choices[0]
                                delta = first.get("delta", {})
                                token = delta.get("content", "")
                                if not token and "text" in first:
                                    token = first["text"]
                                if token:
                                    if t_first_token is None:
                                        t_first_token = time.perf_counter()
                                    full_text.append(token)
                                    token_count += 1
                                    if on_token:
                                        on_token(token)
                        except json.JSONDecodeError:
                            continue
    except urllib.error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError) or "Connection refused" in str(e):
            raise ConnectionError(
                f"Cannot connect to llama-server at {base_url}. Make sure it is running ('gx server start')."
            ) from e
        raise ConnectionError(f"Connection error to {base_url}: {e}") from e
    except Exception as e:
        if "Connection refused" in str(e):
            raise ConnectionError(
                f"Cannot connect to llama-server at {base_url}. Make sure it is running ('gx server start')."
            ) from e
        raise

    t_end = time.perf_counter()
    elapsed = max(t_end - t_start, 0.001)
    ttft_sec = (t_first_token - t_start) if t_first_token is not None else elapsed
    ttft_ms = ttft_sec * 1000.0

    if server_timings:
        prompt_eval_tok_s = float(server_timings.get("prompt_per_second", 0.0))
        gen_tok_s = float(server_timings.get("predicted_per_second", 0.0))
        if gen_tok_s <= 0.0 and token_count > 0:
            gen_elapsed = max(t_end - (t_first_token or t_start), 0.001)
            gen_tok_s = token_count / gen_elapsed
    else:
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        est_prompt_tokens = max(1, prompt_chars // 4)
        prompt_eval_tok_s = est_prompt_tokens / ttft_sec if ttft_sec > 0 else 0.0
        gen_elapsed = max(t_end - (t_first_token or t_start), 0.001)
        gen_tok_s = token_count / gen_elapsed if gen_elapsed > 0 else (token_count / elapsed)

    return {
        "text": "".join(full_text),
        "elapsed": elapsed,
        "token_count": token_count,
        "ttft_ms": ttft_ms,
        "prompt_eval_tok_s": prompt_eval_tok_s,
        "gen_tok_s": gen_tok_s,
        "timings": server_timings,
    }


def stream_chat_completion(
    messages: List[Dict[str, str]],
    base_url: str,
    config: Dict[str, Any],
    on_token: Optional[Callable[[str], None]] = None,
) -> Tuple[str, float, int]:
    """Send chat request and stream SSE response.

    Returns (full_response_text, elapsed_seconds, token_count).
    """
    res = stream_chat_completion_detailed(messages, base_url, config, on_token=on_token)
    return res["text"], res["elapsed"], res["token_count"]


def normalize_model_url(url: str) -> str:
    """Convert Hugging Face blob URLs to resolve URLs automatically."""
    url = url.strip()
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


# Command Implementations


def cmd_ask(args: argparse.Namespace) -> int:
    """Execute one-shot prompt with streaming output."""
    config = load_config()
    models_dir = get_models_dir(config)

    persona_name = getattr(args, "persona", None)
    if persona_name:
        config, err = apply_persona(persona_name, config, models_dir)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

    base_url = get_base_url(config, args.host, args.port, getattr(args, "url", None))

    stdin_input = read_piped_stdin()
    prompt = args.prompt or ""

    if stdin_input:
        if prompt:
            prompt = f"{prompt}\n\n{stdin_input}"
        else:
            prompt = stdin_input

    if not prompt.strip():
        sys.stderr.write("Error: Empty prompt. Provide a prompt argument or pipe text via stdin.\n")
        return 1

    messages = []
    sys_prompt = config.get("system", "").strip()
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        def on_token(token: str) -> None:
            sys.stdout.write(token)
            sys.stdout.flush()

        _, elapsed, count = stream_chat_completion(messages, base_url, config, on_token=on_token)
        tok_per_sec = count / elapsed if elapsed > 0 else 0.0
        sys.stdout.write(f"\n\n[{count} tokens, {tok_per_sec:.1f} tok/s]\n")
        sys.stdout.flush()
        return 0
    except KeyboardInterrupt:
        sys.stdout.write("\n\n[Generation stopped by user]\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat loop with in-memory conversation history, session persistence, and persona support."""
    config = load_config()
    models_dir = get_models_dir(config)

    persona_name = getattr(args, "persona", None)
    if persona_name:
        config, err = apply_persona(persona_name, config, models_dir)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

    base_url = get_base_url(config, args.host, args.port, getattr(args, "url", None))

    session_name = getattr(args, "save", None)
    load_name = getattr(args, "load", None)
    sys_prompt = config.get("system", "").strip()
    messages: List[Dict[str, str]] = []

    if load_name:
        session_data, err = load_session(load_name)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1
        messages = list(session_data.get("messages", []))
        if session_data.get("system"):
            sys_prompt = session_data["system"]
        if session_data.get("model"):
            config["model"] = session_data["model"]
        if not session_name:
            session_name = load_name
        print(f"[Loaded session '{load_name}' ({len(messages)} messages)]")

    if session_name and not validate_session_name(session_name):
        sys.stderr.write(
            f"Error: Invalid session name '{session_name}'. Use only letters, digits, hyphens (-), and underscores (_).\n"
        )
        return 1

    if not messages and sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    print("gx interactive chat (/exit to quit, /clear to reset, /sys <text> to set system prompt)")
    if persona_name:
        print(f"Persona: {persona_name}")
    if session_name:
        print(f"Session: {session_name} (Auto-saving after each turn)")
    if sys_prompt:
        print(f"System:  {sys_prompt}")
    print()

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            print("Goodbye!")
            break

        if user_input == "/clear":
            messages = []
            sys_prompt = config.get("system", "").strip()
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            if session_name:
                save_session(session_name, config.get("model", ""), sys_prompt, messages)
            print("[History cleared]\n")
            continue

        if user_input.startswith("/sys"):
            new_sys = user_input[4:].strip()
            if not new_sys:
                cur_sys = next((m["content"] for m in messages if m["role"] == "system"), "(none)")
                print(f"[Current system prompt: {cur_sys}]\n")
            else:
                sys_prompt = new_sys
                config["system"] = new_sys
                messages = [m for m in messages if m["role"] != "system"]
                messages.insert(0, {"role": "system", "content": new_sys})
                if session_name:
                    save_session(session_name, config.get("model", ""), sys_prompt, messages)
                print(f"[System prompt updated: {new_sys}]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        ctx_limit = int(config.get("ctx", 2048))
        messages, dropped = apply_context_guard(messages, ctx_limit, threshold_pct=0.8)
        if dropped > 0:
            est = estimate_token_count(messages)
            print(f"[Context guard: trimmed {dropped} older message(s) to stay within 80% context limit ({est}/{ctx_limit} tokens)]")

        print("AI  > ", end="", flush=True)

        try:
            def on_token(token: str) -> None:
                sys.stdout.write(token)
                sys.stdout.flush()

            full_resp, elapsed, count = stream_chat_completion(
                messages, base_url, config, on_token=on_token
            )
            messages.append({"role": "assistant", "content": full_resp})
            tok_per_sec = count / elapsed if elapsed > 0 else 0.0
            print(f"\n[{count} tokens, {tok_per_sec:.1f} tok/s]\n")

            if session_name:
                save_session(session_name, config.get("model", ""), sys_prompt, messages)

        except KeyboardInterrupt:
            print("\n[Generation stopped]\n")
            if session_name:
                save_session(session_name, config.get("model", ""), sys_prompt, messages)
        except Exception as e:
            print(f"\nError: {e}\n")

    return 0


def cmd_session(args: argparse.Namespace) -> int:
    """Manage saved chat sessions."""
    action = getattr(args, "session_action", None)
    sessions_dir = get_sessions_dir()

    if action == "list":
        if not os.path.isdir(sessions_dir):
            print(f"No sessions directory at {sessions_dir}")
            return 0

        files = sorted(
            [f for f in os.listdir(sessions_dir) if f.endswith(".json")],
            key=lambda x: os.path.getmtime(os.path.join(sessions_dir, x)),
            reverse=True,
        )
        if not files:
            print(f"No saved sessions in {sessions_dir}")
            return 0

        print(f"{'Session':<25} {'Messages':<10} {'Last Modified':<20}")
        print("-" * 58)
        for f in files:
            name = f[:-5]
            f_path = os.path.join(sessions_dir, f)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f_path)))
            try:
                with open(f_path, "r", encoding="utf-8") as s_f:
                    data = json.load(s_f)
                    msg_count = len(data.get("messages", []))
                    count_str = str(msg_count)
            except Exception:
                count_str = "[corrupted]"
            print(f"{name:<25} {count_str:<10} {mtime:<20}")
        return 0

    if action == "show":
        name = args.name
        data, err = load_session(name)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

        print(f"Session: {name}")
        if data.get("model"):
            print(f"Model:   {data['model']}")
        if data.get("system"):
            print(f"System:  {data['system']}")
        print("=" * 50)

        for m in data.get("messages", []):
            role = m.get("role", "unknown").capitalize()
            content = m.get("content", "")
            print(f"\n[{role}]\n{content}")
        return 0

    if action == "rm":
        name = args.name
        if not validate_session_name(name):
            sys.stderr.write(f"Error: Invalid session name '{name}'.\n")
            return 1

        f_path = get_session_file(name)
        if not os.path.isfile(f_path):
            sys.stderr.write(f"Error: Session '{name}' not found.\n")
            return 1

        if not getattr(args, "yes", False):
            try:
                ans = input(f"Delete session '{name}'? [y/N]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return 0
            if ans not in ("y", "yes"):
                print("Cancelled.")
                return 0

        try:
            os.remove(f_path)
            print(f"Deleted session '{name}'.")
            return 0
        except Exception as e:
            sys.stderr.write(f"Error deleting session '{name}': {e}\n")
            return 1

    if action == "export":
        name = args.name
        data, err = load_session(name)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

        model = data.get("model", "default")
        system = data.get("system", "")

        out = [f"# Chat Session: {name}\n"]
        out.append(f"- **Model:** {model}")
        if system:
            out.append(f"- **System Prompt:** {system}")
        out.append("\n---\n")

        for m in data.get("messages", []):
            role = m.get("role", "unknown").capitalize()
            content = m.get("content", "")
            out.append(f"### {role}\n\n{content}\n")

        sys.stdout.write("\n".join(out))
        sys.stdout.flush()
        return 0

    sys.stderr.write("Usage: gx session [list|show <name>|rm <name>|export <name>]\n")
    return 1


def cmd_persona(args: argparse.Namespace) -> int:
    """Manage custom AI personas."""
    ensure_builtin_personas()
    action = getattr(args, "persona_action", None)
    p_dir = get_personas_dir()

    if action == "list":
        files = sorted([f for f in os.listdir(p_dir) if f.endswith(".json")])
        if not files:
            print("No personas found.")
            return 0
        print(f"{'Persona':<15} {'Temp':<6} {'MaxTok':<8} {'Model':<20} {'System Summary'}")
        print("-" * 75)
        for f in files:
            name = f[:-5]
            data, _ = load_persona(name)
            if data:
                temp = str(data.get("temp", "-"))
                max_tok = str(data.get("max_tokens", "-"))
                model = str(data.get("model", "-") or "-")
                sys_summary = data.get("system", "").replace("\n", " ")[:30]
                if len(data.get("system", "")) > 30:
                    sys_summary += "..."
                print(f"{name:<15} {temp:<6} {max_tok:<8} {model:<20} {sys_summary}")
        return 0

    if action == "show":
        name = args.name
        data, err = load_persona(name)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1
        print(f"Persona:    {name}")
        print(f"Temp:       {data.get('temp', 0.7)}")
        print(f"Max Tokens: {data.get('max_tokens', 256)}")
        print(f"Model:      {data.get('model', '(none)')}")
        print("System Prompt:\n" + ("=" * 40))
        print(data.get("system", ""))
        return 0

    if action in ("add", "edit"):
        name = args.name
        if not validate_session_name(name):
            sys.stderr.write(f"Error: Invalid persona name '{name}'.\n")
            return 1

        existing, _ = load_persona(name) if action == "edit" else (None, None)
        print(f"Configure persona '{name}':")
        print("Enter system prompt (multi-line, finish with a line containing only '.'):")
        sys_lines = []
        while True:
            try:
                line = input()
                if line.strip() == ".":
                    break
                sys_lines.append(line)
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return 0
        system_prompt = "\n".join(sys_lines).strip()
        if not system_prompt and existing:
            system_prompt = existing.get("system", "")

        def_temp = str(existing.get("temp", 0.7)) if existing else "0.7"
        try:
            t_input = input(f"Temperature [{def_temp}]: ").strip()
            temp_val = float(t_input) if t_input else float(def_temp)
        except ValueError:
            temp_val = float(def_temp)

        def_tok = str(existing.get("max_tokens", 256)) if existing else "256"
        try:
            tok_input = input(f"Max tokens [{def_tok}]: ").strip()
            max_tok_val = int(tok_input) if tok_input else int(def_tok)
        except ValueError:
            max_tok_val = int(def_tok)

        def_model = existing.get("model", "") if existing else ""
        model_input = input(f"Pinned model (optional) [{def_model}]: ").strip()
        model_val = model_input if model_input else def_model

        data = {
            "name": name,
            "system": system_prompt,
            "temp": temp_val,
            "max_tokens": max_tok_val,
            "model": model_val,
        }
        if save_persona(name, data):
            print(f"Saved persona '{name}'.")
            return 0
        return 1

    if action == "rm":
        name = args.name
        f_path = os.path.join(p_dir, f"{name}.json")
        if not os.path.isfile(f_path):
            sys.stderr.write(f"Error: Persona '{name}' not found.\n")
            return 1
        try:
            os.remove(f_path)
            print(f"Deleted persona '{name}'.")
            return 0
        except Exception as e:
            sys.stderr.write(f"Error deleting persona '{name}': {e}\n")
            return 1

    sys.stderr.write("Usage: gx persona [list|show <name>|add <name>|edit <name>|rm <name>]\n")
    return 1


def cmd_tpl(args: argparse.Namespace) -> int:
    """Manage reusable prompt templates."""
    ensure_builtin_templates()
    action = getattr(args, "tpl_action", None)
    t_dir = get_templates_dir()

    if action == "list":
        files = sorted([f for f in os.listdir(t_dir) if f.endswith(".txt")])
        if not files:
            print("No templates found.")
            return 0
        print(f"{'Template':<20} {'Variables'}")
        print("-" * 50)
        for f in files:
            name = f[:-4]
            content, _ = load_template(name)
            if content:
                vars_found = sorted(list(set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", content))))
                vars_str = ", ".join(f"{{{{{v}}}}}" for v in vars_found)
                print(f"{name:<20} {vars_str}")
        return 0

    if action == "show":
        name = args.name
        content, err = load_template(name)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1
        print(f"Template: {name}\n" + ("=" * 40))
        print(content)
        return 0

    if action == "add":
        name = args.name
        if not validate_session_name(name):
            sys.stderr.write(f"Error: Invalid template name '{name}'.\n")
            return 1
        print(f"Enter content for template '{name}' (finish with a line containing only '.'):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == ".":
                    break
                lines.append(line)
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return 0
        content = "\n".join(lines).strip()
        if not content:
            sys.stderr.write("Error: Template content cannot be empty.\n")
            return 1
        if save_template(name, content):
            print(f"Saved template '{name}'.")
            return 0
        return 1

    if action == "rm":
        name = args.name
        f_path = os.path.join(t_dir, f"{name}.txt")
        if not os.path.isfile(f_path):
            sys.stderr.write(f"Error: Template '{name}' not found.\n")
            return 1
        try:
            os.remove(f_path)
            print(f"Deleted template '{name}'.")
            return 0
        except Exception as e:
            sys.stderr.write(f"Error deleting template '{name}': {e}\n")
            return 1

    sys.stderr.write("Usage: gx tpl [list|show <name>|add <name>|rm <name>]\n")
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Execute a prompt template with variable substitution and optional persona."""
    ensure_builtin_templates()
    template_name = args.template
    tpl_text, err = load_template(template_name)
    if err:
        sys.stderr.write(f"Error: {err}\n")
        return 1

    placeholders = sorted(list(set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", tpl_text))))

    provided_vars: Dict[str, str] = {}
    for item in (args.vars or []):
        if "=" in item:
            k, v = item.split("=", 1)
            provided_vars[k.strip()] = v.strip()

    if "input" in placeholders and "input" not in provided_vars:
        stdin_val = read_piped_stdin()
        if stdin_val:
            provided_vars["input"] = stdin_val

    missing = [p for p in placeholders if p not in provided_vars or not str(provided_vars[p]).strip()]
    if missing:
        sys.stderr.write(f"Error: Missing required template variable(s): {', '.join(missing)}\n")
        return 1

    unused = [k for k in provided_vars if k not in placeholders]
    if unused:
        print(f"Warning: Unused variable(s): {', '.join(unused)}")

    prompt = tpl_text
    for k, v in provided_vars.items():
        prompt = prompt.replace(f"{{{{{k}}}}}", v)

    config = load_config()
    models_dir = get_models_dir(config)
    persona_name = getattr(args, "persona", None)
    if persona_name:
        config, err = apply_persona(persona_name, config, models_dir)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

    base_url = get_base_url(config, args.host, args.port, getattr(args, "url", None))

    messages = []
    sys_prompt = config.get("system", "").strip()
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        def on_token(token: str) -> None:
            sys.stdout.write(token)
            sys.stdout.flush()

        _, elapsed, count = stream_chat_completion(messages, base_url, config, on_token=on_token)
        tok_per_sec = count / elapsed if elapsed > 0 else 0.0
        sys.stdout.write(f"\n\n[{count} tokens, {tok_per_sec:.1f} tok/s]\n")
        sys.stdout.flush()
        return 0
    except KeyboardInterrupt:
        sys.stdout.write("\n\n[Generation stopped by user]\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


def run_benchmark_suite(
    runs: int,
    prompt_tokens: int,
    gen_tokens: int,
    base_url: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute benchmark runs and collect TTFT, prompt eval speed, and gen speeds."""
    words = ["word"] * max(1, prompt_tokens)
    test_prompt = "Bench: " + " ".join(words)

    messages = [{"role": "user", "content": test_prompt}]
    ttfts: List[float] = []
    prompt_speeds: List[float] = []
    gen_speeds: List[float] = []

    for _ in range(max(1, runs)):
        res = stream_chat_completion_detailed(
            messages,
            base_url,
            config,
            max_tokens_override=gen_tokens,
        )
        ttfts.append(res["ttft_ms"])
        if res["prompt_eval_tok_s"] > 0:
            prompt_speeds.append(res["prompt_eval_tok_s"])
        gen_speeds.append(res["gen_tok_s"])

    avg_gen = sum(gen_speeds) / len(gen_speeds) if gen_speeds else 0.0
    min_gen = min(gen_speeds) if gen_speeds else 0.0
    max_gen = max(gen_speeds) if gen_speeds else 0.0
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_prompt = sum(prompt_speeds) / len(prompt_speeds) if prompt_speeds else 0.0

    throttle_warning = False
    if len(gen_speeds) >= 2 and gen_speeds[0] > 0:
        drop_pct = ((gen_speeds[0] - gen_speeds[-1]) / gen_speeds[0]) * 100.0
        if drop_pct > 25.0:
            throttle_warning = True

    return {
        "runs": runs,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "avg_gen_tok_s": avg_gen,
        "min_gen_tok_s": min_gen,
        "max_gen_tok_s": max_gen,
        "avg_ttft_ms": avg_ttft,
        "avg_prompt_tok_s": avg_prompt,
        "gen_speeds": gen_speeds,
        "throttle_warning": throttle_warning,
    }


def cmd_bench(args: argparse.Namespace) -> int:
    """Benchmark local model performance and test thread sweeps."""
    config = load_config()
    history_file = get_bench_history_file()

    if getattr(args, "history", False):
        if not os.path.isfile(history_file):
            print("No benchmark history found.")
            return 0
        with open(history_file, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if not lines:
            print("No benchmark history found.")
            return 0
        print(f"{'Timestamp':<20} {'Model':<28} {'Quant':<8} {'Th':<4} {'Gen (tok/s)':<12} {'TTFT (ms)'}")
        print("-" * 80)
        for entry in lines[-10:]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("timestamp", time.time())))
            m = entry.get("model", "unknown")[:26]
            q = entry.get("quant", "-")
            th = str(entry.get("threads", "-"))
            gen_str = f"{entry.get('avg_gen_tok_s', 0.0):.1f}"
            ttft_str = f"{entry.get('avg_ttft_ms', 0.0):.1f}"
            print(f"{ts:<20} {m:<28} {q:<8} {th:<4} {gen_str:<12} {ttft_str}")
        return 0

    base_url = get_base_url(config, args.host, args.port, getattr(args, "url", None))
    model_name = config.get("model", "default")
    quant = extract_quantization(model_name)
    ctx = config.get("ctx", 2048)
    runs = args.runs if args.runs is not None else 3
    prompt_tokens = args.prompt_tokens if args.prompt_tokens is not None else 128
    gen_tokens = args.gen_tokens if args.gen_tokens is not None else 64

    sweep_spec = getattr(args, "sweep", None)
    if sweep_spec and "threads=" in sweep_spec:
        threads_str = sweep_spec.split("threads=")[1]
        try:
            thread_list = [int(t.strip()) for t in threads_str.split(",") if t.strip()]
        except ValueError:
            sys.stderr.write("Error: Invalid sweep format. Expected e.g. --sweep threads=2,4,6\n")
            return 1

        print(f"Starting threads sweep {thread_list} for model: {model_name}...")
        results = []
        orig_threads = config.get("threads", 4)

        for th in thread_list:
            config["threads"] = th
            save_config(config)
            print(f"\nRestarting server with threads={th}...")
            cmd_server(argparse.Namespace(server_action="stop"))
            cmd_server(argparse.Namespace(server_action="start"))
            time.sleep(1.0)
            res = run_benchmark_suite(runs, prompt_tokens, gen_tokens, base_url, config)
            res["threads"] = th
            results.append(res)

        config["threads"] = orig_threads
        save_config(config)

        best_speed = max(r["avg_gen_tok_s"] for r in results) if results else 0.0

        print("\n" + "=" * 65)
        print(f"Sweep Results for {model_name} ({quant})")
        print(f"{'Threads':<10} {'Avg Gen (tok/s)':<18} {'Min/Max (tok/s)':<20} {'TTFT (ms)'}")
        print("-" * 65)
        for r in results:
            is_best = " * (best)" if r["avg_gen_tok_s"] == best_speed else ""
            min_max = f"{r['min_gen_tok_s']:.1f} / {r['max_gen_tok_s']:.1f}"
            print(f"{r['threads']:<10} {r['avg_gen_tok_s']:<18.1f} {min_max:<20} {r['avg_ttft_ms']:<10.1f}{is_best}")
        print("=" * 65)
        return 0

    print(f"Running benchmark on {base_url} ({runs} runs, {prompt_tokens}p / {gen_tokens}g tokens)...")
    res = run_benchmark_suite(runs, prompt_tokens, gen_tokens, base_url, config)

    print("\n" + "=" * 55)
    print(f"Model:        {model_name}")
    print(f"Quantization: {quant}")
    print(f"Context:      {ctx} | Threads: {config.get('threads', 4)}")
    print(f"Runs:         {runs} (Prompt tokens: ~{prompt_tokens}, Gen tokens: {gen_tokens})")
    print("-" * 55)
    print(f"TTFT (Time to First Token): {res['avg_ttft_ms']:.1f} ms")
    if res['avg_prompt_tok_s'] > 0:
        print(f"Prompt Processing Speed:    {res['avg_prompt_tok_s']:.1f} tok/s")
    print(
        f"Generation Speed:           Avg: {res['avg_gen_tok_s']:.1f} tok/s | Min: {res['min_gen_tok_s']:.1f} tok/s | Max: {res['max_gen_tok_s']:.1f} tok/s"
    )

    if res["throttle_warning"]:
        speeds = res["gen_speeds"]
        drop_pct = ((speeds[0] - speeds[-1]) / speeds[0]) * 100.0
        print(
            f"\nWarning: Possible thermal throttling detected! Speed dropped from {speeds[0]:.1f} tok/s to {speeds[-1]:.1f} tok/s ({drop_pct:.1f}% drop)."
        )
    print("=" * 55)

    entry = {
        "timestamp": time.time(),
        "model": model_name,
        "quant": quant,
        "ctx": ctx,
        "threads": config.get("threads", 4),
        "avg_gen_tok_s": round(res["avg_gen_tok_s"], 2),
        "min_gen_tok_s": round(res["min_gen_tok_s"], 2),
        "max_gen_tok_s": round(res["max_gen_tok_s"], 2),
        "avg_ttft_ms": round(res["avg_ttft_ms"], 2),
        "avg_prompt_tok_s": round(res["avg_prompt_tok_s"], 2),
    }

    try:
        with open(history_file, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    if args.out:
        try:
            with open(args.out, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(entry) + "\n")
            print(f"Results saved to: {args.out}")
        except Exception as e:
            sys.stderr.write(f"Error saving results to {args.out}: {e}\n")

    return 0


# Agent System Prompt, Sandbox Tools & Execution Loop

AGENT_SYSTEM_PROMPT = """You are an autonomous AI assistant that solves tasks by executing tools.
All tool paths must be within the current sandbox directory.

You MUST respond with ONLY ONE valid JSON object per turn:
- To call a tool: {"tool": "<name>", "args": {<arguments>}}
- To give the final answer: {"final": "<answer>"}

Available Tools:
1. read_file(path) - Read first 4000 characters of a file.
   Example: {"tool": "read_file", "args": {"path": "notes.txt"}}
2. list_dir(path) - List files and directories with sizes.
   Example: {"tool": "list_dir", "args": {"path": "."}}
3. write_file(path, content) - Write text content to a file in the sandbox.
   Example: {"tool": "write_file", "args": {"path": "output.txt", "content": "Hello World"}}
4. search_files(pattern, path) - Plain-text grep across files (max 20 matches).
   Example: {"tool": "search_files", "args": {"pattern": "TODO", "path": "."}}
5. calc(expression) - Safely evaluate a math expression (no eval/attributes).
   Example: {"tool": "calc", "args": {"expression": "(12 + 34) * 5"}}
6. shell(cmd) - Run a shell command in sandbox (20s timeout, max 2000 chars output).
   Example: {"tool": "shell", "args": {"cmd": "ls -la"}}

Always reply with ONLY a single JSON object.
"""


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON dictionary object from raw text."""
    if not text:
        return None

    # First check code block contents if present
    code_block_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    for block in code_block_matches:
        block_clean = block.strip()
        try:
            val = json.loads(block_clean)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    # Find candidate { ... } substrings
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, n):
            c = text[end]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            val = json.loads(candidate)
                            if isinstance(val, dict):
                                return val
                        except Exception:
                            pass
                        break
    return None


def resolve_sandbox_path(sandbox_dir: str, user_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve and validate that a path is strictly inside the sandbox directory.

    Rejects directory traversal (..), absolute escapes, and escaping symlinks.
    Returns (resolved_absolute_path, error_message).
    """
    sandbox_root = os.path.realpath(os.path.abspath(sandbox_dir))
    if not user_path:
        return sandbox_root, None

    if os.path.isabs(user_path):
        target_path = os.path.abspath(user_path)
    else:
        target_path = os.path.abspath(os.path.join(sandbox_root, user_path))

    real_target = os.path.realpath(target_path)
    parent_dir = os.path.dirname(target_path)
    real_parent = os.path.realpath(parent_dir)

    try:
        common_target = os.path.commonpath([sandbox_root, real_target])
        common_parent = os.path.commonpath([sandbox_root, real_parent])
    except ValueError:
        return None, f"Error: Path '{user_path}' escapes sandbox directory '{sandbox_root}'."

    if common_target != sandbox_root or common_parent != sandbox_root:
        return None, f"Error: Path '{user_path}' escapes sandbox directory '{sandbox_root}'."

    return real_target, None


def tool_read_file(sandbox_dir: str, path: str) -> str:
    """Read up to 4000 characters from a sandbox file with truncation notice."""
    real_path, err = resolve_sandbox_path(sandbox_dir, path)
    if err:
        return err

    if not os.path.exists(real_path) or not os.path.isfile(real_path):
        return f"Error: File '{path}' not found."

    try:
        with open(real_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(4001)
        if len(content) > 4000:
            return content[:4000] + "\n[Truncated: only first 4000 characters shown]"
        return content
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def tool_list_dir(sandbox_dir: str, path: str = ".") -> str:
    """List names and sizes of files and directories in sandbox."""
    real_path, err = resolve_sandbox_path(sandbox_dir, path)
    if err:
        return err

    if not os.path.exists(real_path) or not os.path.isdir(real_path):
        return f"Error: Directory '{path}' not found."

    try:
        entries = sorted(os.listdir(real_path))
        if not entries:
            return "(empty directory)"
        lines = []
        for e in entries:
            ep = os.path.join(real_path, e)
            if os.path.islink(ep):
                lines.append(f"[LNK]  {e} -> {os.readlink(ep)}")
            elif os.path.isdir(ep):
                lines.append(f"[DIR]  {e}/")
            else:
                try:
                    sz = os.path.getsize(ep)
                    lines.append(f"[FILE] {e} ({format_file_size(sz)})")
                except Exception:
                    lines.append(f"[FILE] {e}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory '{path}': {e}"


def tool_write_file(
    sandbox_dir: str,
    path: str,
    content: str,
    yes: bool = False,
    confirm_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Write content to a file inside the sandbox, requiring confirmation unless --yes."""
    real_path, err = resolve_sandbox_path(sandbox_dir, path)
    if err:
        return err

    if os.path.exists(real_path):
        try:
            curr_size = os.path.getsize(real_path)
            prompt = f"Overwrite existing file '{path}' ({curr_size} bytes) with {len(content)} chars? [y/N]: "
        except Exception:
            prompt = f"Overwrite existing file '{path}' with {len(content)} chars? [y/N]: "
    else:
        prompt = f"Write {len(content)} chars to new file '{path}'? [y/N]: "

    if not yes:
        if confirm_fn is not None:
            confirmed = confirm_fn(prompt)
        else:
            try:
                ans = input(prompt).strip().lower()
                confirmed = ans in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                confirmed = False
        if not confirmed:
            return f"Tool aborted: user denied write to '{path}'."

    try:
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        with open(real_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing file '{path}': {e}"


def tool_search_files(sandbox_dir: str, pattern: str, path: str = ".") -> str:
    """Search for plain-text pattern across files in sandbox (max 20 matches)."""
    real_path, err = resolve_sandbox_path(sandbox_dir, path)
    if err:
        return err

    if not os.path.exists(real_path):
        return f"Error: Path '{path}' not found."

    matches: List[str] = []
    max_matches = 20
    reached_limit = False

    sandbox_root = os.path.realpath(os.path.abspath(sandbox_dir))

    for root, _, files in os.walk(real_path):
        if reached_limit:
            break
        for f in sorted(files):
            file_path = os.path.join(root, f)
            rel_path = os.path.relpath(file_path, sandbox_root)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
                    for line_idx, line in enumerate(file_obj, 1):
                        if pattern in line:
                            matches.append(f"{rel_path}:{line_idx}: {line.strip()[:150]}")
                            if len(matches) >= max_matches:
                                reached_limit = True
                                break
            except Exception:
                continue

    if not matches:
        return "No matches found."
    out = "\n".join(matches)
    if reached_limit:
        out += "\n[Reached limit of 20 matches]"
    return out


ALLOWED_AST_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_ast_node(node: ast.AST) -> Any:
    """Evaluate an AST expression node recursively using only allowed arithmetic operators."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
        return getattr(node, "n")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_ast_node(node.left)
        right = _safe_eval_ast_node(node.right)
        op_type = type(node.op)
        if op_type not in ALLOWED_AST_OPERATORS:
            raise ValueError(f"Disallowed operator: {op_type.__name__}")
        if op_type is ast.Pow and (isinstance(right, (int, float)) and right > 1000):
            raise ValueError("Exponent too large")
        return ALLOWED_AST_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_ast_node(node.operand)
        op_type = type(node.op)
        if op_type not in ALLOWED_AST_OPERATORS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        return ALLOWED_AST_OPERATORS[op_type](operand)
    else:
        raise ValueError(f"Disallowed syntax node: {type(node).__name__}")


def tool_calc(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression without eval()."""
    if not expression or not expression.strip():
        return "Error: Empty expression."
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_ast_node(parsed)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: Invalid arithmetic expression: {e}"


def tool_shell(
    sandbox_dir: str,
    cmd: str,
    yes: bool = False,
    confirm_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Run a shell command inside the sandbox root directory (20s timeout, max 2000 chars output)."""
    sandbox_root = os.path.realpath(os.path.abspath(sandbox_dir))
    prompt = f"Run: {cmd}? [y/N]: "

    if not yes:
        if confirm_fn is not None:
            confirmed = confirm_fn(prompt)
        else:
            try:
                ans = input(prompt).strip().lower()
                confirmed = ans in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                confirmed = False
        if not confirmed:
            return f"Tool aborted: user denied shell command '{cmd}'."

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=sandbox_root,
            capture_output=True,
            text=True,
            timeout=20,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        if not combined:
            combined = f"(command exited with status {proc.returncode})"
        if len(combined) > 2000:
            combined = combined[:2000] + "\n[Truncated: output exceeded 2000 chars]"
        return combined
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 20 seconds."
    except Exception as e:
        return f"Error executing shell command: {e}"


def cmd_agent(args: argparse.Namespace) -> int:
    """Autonomous tool-using agent loop with sandbox enforcement."""
    config = load_config()
    models_dir = get_models_dir(config)

    persona_name = getattr(args, "persona", None)
    persona_system = ""
    if persona_name:
        config, err = apply_persona(persona_name, config, models_dir)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1
        persona_system = config.get("system", "").strip()

    base_url = get_base_url(config, args.host, args.port, getattr(args, "url", None))
    sandbox_dir = getattr(args, "dir", ".") or "."
    sandbox_root = os.path.realpath(os.path.abspath(sandbox_dir))
    if not os.path.isdir(sandbox_root):
        sys.stderr.write(f"Error: Sandbox directory '{sandbox_dir}' does not exist.\n")
        return 1

    max_steps = args.max_steps if args.max_steps is not None else 6
    trace = getattr(args, "trace", False)
    yes = getattr(args, "yes", False)
    task = args.task
    save_session_name = getattr(args, "save", None)

    system_content = (persona_system + "\n\n" + AGENT_SYSTEM_PROMPT).strip() if persona_system else AGENT_SYSTEM_PROMPT

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task},
    ]

    tools_used: List[str] = []
    consecutive_invalid = 0
    outcome = "step_limit"
    step = 0

    if trace:
        print(f"Starting agent in sandbox: {sandbox_root}")
        print(f"Task: {task}\n")

    for step in range(1, max_steps + 1):
        if trace:
            print(f"--- Step {step}/{max_steps} ---")

        resp_chunks: List[str] = []

        def on_tok(t: str) -> None:
            if trace:
                sys.stdout.write(t)
                sys.stdout.flush()
            resp_chunks.append(t)

        try:
            resp_text, elapsed, tok_count = stream_chat_completion(
                messages, base_url, config, on_token=on_tok if trace else None
            )
        except Exception as e:
            sys.stderr.write(f"Agent connection error: {e}\n")
            outcome = "aborted"
            break

        if trace:
            print()

        json_obj = extract_first_json_object(resp_text)
        if not json_obj:
            consecutive_invalid += 1
            if consecutive_invalid >= 2:
                sys.stderr.write("Aborted: 2 consecutive invalid format responses.\n")
                outcome = "aborted"
                break
            messages.append({"role": "assistant", "content": resp_text})
            messages.append({"role": "user", "content": "Invalid format, reply with JSON only"})
            continue

        consecutive_invalid = 0

        if "final" in json_obj:
            final_answer = str(json_obj["final"])
            outcome = "final"
            messages.append({"role": "assistant", "content": json.dumps(json_obj)})
            if not trace:
                print(final_answer)
            else:
                print(f"\n[Final Answer]\n{final_answer}")
            break

        if "tool" in json_obj:
            tool_name = str(json_obj.get("tool", "")).strip()
            tool_args = json_obj.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tools_used.append(tool_name)

            if tool_name == "read_file":
                tool_result = tool_read_file(sandbox_root, str(tool_args.get("path", "")))
            elif tool_name == "list_dir":
                tool_result = tool_list_dir(sandbox_root, str(tool_args.get("path", ".")))
            elif tool_name == "write_file":
                tool_result = tool_write_file(
                    sandbox_root,
                    str(tool_args.get("path", "")),
                    str(tool_args.get("content", "")),
                    yes=yes,
                )
            elif tool_name == "search_files":
                tool_result = tool_search_files(
                    sandbox_root,
                    str(tool_args.get("pattern", "")),
                    str(tool_args.get("path", ".")),
                )
            elif tool_name == "calc":
                tool_result = tool_calc(str(tool_args.get("expression", "")))
            elif tool_name == "shell":
                tool_result = tool_shell(
                    sandbox_root,
                    str(tool_args.get("cmd", "")),
                    yes=yes,
                )
            else:
                tool_result = f"Error: Unknown tool '{tool_name}'."

            if trace:
                print(f"Tool [{tool_name}] output:\n{tool_result}\n")

            messages.append({"role": "assistant", "content": json.dumps(json_obj)})
            messages.append({"role": "user", "content": f"Tool '{tool_name}' result:\n{tool_result}"})
            continue

        # If JSON dict has neither 'final' nor 'tool'
        consecutive_invalid += 1
        if consecutive_invalid >= 2:
            sys.stderr.write("Aborted: 2 consecutive invalid format responses.\n")
            outcome = "aborted"
            break
        messages.append({"role": "assistant", "content": json.dumps(json_obj)})
        messages.append({"role": "user", "content": "Invalid format, reply with JSON only"})

    if outcome == "step_limit":
        print("stopped: step limit")

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": task,
        "steps": min(step, max_steps),
        "tools_used": tools_used,
        "outcome": outcome,
    }
    append_agent_log(log_entry)

    if save_session_name:
        if validate_session_name(save_session_name):
            save_session(save_session_name, config.get("model", ""), system_content, messages)
            print(f"Agent session transcript saved to '{save_session_name}'.")
        else:
            sys.stderr.write(f"Error: Invalid session name '{save_session_name}'.\n")

    return 0 if outcome == "final" else (1 if outcome == "aborted" else 0)


def cmd_pull(args: argparse.Namespace) -> int:
    """Download a GGUF model into models directory with resume support and progress tracking."""
    config = load_config()
    models_dir = get_models_dir(config)
    os.makedirs(models_dir, exist_ok=True)

    raw_url = args.url
    url = normalize_model_url(raw_url)

    if args.out:
        filename = args.out
    else:
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or filename == "/":
            filename = "model.gguf"

    target_file = os.path.join(models_dir, filename)
    part_file = target_file + ".part"

    if os.path.isfile(target_file) and not args.force:
        sys.stderr.write(
            f"Error: File '{filename}' already exists in {models_dir}.\nUse --force to overwrite.\n"
        )
        return 1

    try:
        usage = shutil.disk_usage(models_dir)
        free_bytes = usage.free
        if free_bytes < 1024 * 1024 * 1024:
            print(f"Warning: Low disk space! Only {format_file_size(free_bytes)} free on device.")
    except Exception:
        free_bytes = 10 * 1024 * 1024 * 1024

    headers = {"User-Agent": "gx-downloader/1.0"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    existing_size = 0
    if os.path.isfile(part_file):
        existing_size = os.path.getsize(part_file)
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            print(f"Resuming download from {format_file_size(existing_size)}...")

    req = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            if existing_size > 0:
                shutil.move(part_file, target_file)
                print(f"Download complete: {filename} ({format_file_size(existing_size)})")
                return 0
        sys.stderr.write(f"HTTP Error {e.code}: {e.reason}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Error connecting to download URL: {e}\n")
        return 1

    status_code = getattr(response, "status", response.getcode())
    content_len = response.headers.get("Content-Length")

    if status_code == 206:
        total_size = existing_size + int(content_len) if content_len else 0
        write_mode = "ab"
    else:
        total_size = int(content_len) if content_len else 0
        existing_size = 0
        write_mode = "wb"

    if total_size > 0:
        needed_bytes = total_size - existing_size
        if free_bytes - needed_bytes < 1024 * 1024 * 1024:
            rem = max(0, free_bytes - needed_bytes)
            print(f"Warning: Download will leave less than 1 GB of free space ({format_file_size(rem)} remaining).")

    print(f"Downloading {filename}...")
    if total_size > 0:
        print(f"Total size: {format_file_size(total_size)}")

    downloaded = existing_size
    t_start = time.perf_counter()
    session_bytes = 0
    chunk_size = 64 * 1024

    try:
        with open(part_file, write_mode) as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                session_bytes += len(chunk)

                elapsed = time.perf_counter() - t_start
                speed = session_bytes / elapsed if elapsed > 0 else 0.0
                speed_str = f"{format_file_size(int(speed))}/s"

                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    eta_sec = int((total_size - downloaded) / speed) if speed > 0 else 0
                    eta_str = f"{eta_sec // 60:02d}:{eta_sec % 60:02d}"
                    progress = f"\r[{pct:5.1f}%] {format_file_size(downloaded)} / {format_file_size(total_size)} | {speed_str} | ETA: {eta_str}"
                else:
                    progress = f"\rDownloaded {format_file_size(downloaded)} | {speed_str}"

                sys.stdout.write(progress)
                sys.stdout.flush()

        if total_size > 0 and downloaded != total_size:
            sys.stderr.write(
                f"\nError: Size mismatch! Downloaded {downloaded} bytes, expected {total_size} bytes.\n"
            )
            return 1

        shutil.move(part_file, target_file)
        sys.stdout.write(f"\nDownload complete: {target_file} ({format_file_size(downloaded)})\n")
        return 0

    except KeyboardInterrupt:
        sys.stdout.write("\n\n[Download interrupted by user]\n")
        sys.stdout.write(f"Partial file saved: {part_file}\n")
        sys.stdout.write(f"Resume anytime with: gx pull \"{raw_url}\"\n")
        return 130
    except Exception as e:
        sys.stderr.write(f"\nError downloading file: {e}\n")
        return 1
    finally:
        response.close()


def cmd_model(args: argparse.Namespace) -> int:
    """Handle model subcommands: list, use."""
    config = load_config()
    models_dir = get_models_dir(config)
    action = getattr(args, "model_action", None)

    if not os.path.isdir(models_dir):
        sys.stderr.write(f"Error: Models directory '{models_dir}' does not exist.\n")
        sys.stderr.write(f"Create it with: mkdir -p {models_dir}\n")
        return 1

    models = list_gguf_models(models_dir)

    if action == "list":
        if not models:
            print(f"No .gguf models found in {models_dir}")
            return 0

        current = config.get("model", "")
        print(f"Models in {models_dir}:")
        for m in models:
            m_path = os.path.join(models_dir, m)
            size_str = format_file_size(os.path.getsize(m_path))
            is_cur = " * (active)" if m == current or m == os.path.basename(current) else ""
            print(f"  {m:<45} {size_str:>10}{is_cur}")
        return 0

    if action == "use":
        query = args.name
        if not query:
            sys.stderr.write("Error: Please provide a model name or partial name.\n")
            return 1

        matched, candidates = match_model(query, models)
        if matched:
            config["model"] = matched
            save_config(config)
            print(f"Active model set to: {matched}")
            return 0

        if len(candidates) > 1:
            sys.stderr.write(f"Error: Multiple models match '{query}':\n")
            for c in candidates:
                sys.stderr.write(f"  - {c}\n")
            sys.stderr.write("Please specify a more specific name.\n")
            return 1

        sys.stderr.write(f"Error: No model matching '{query}' found in {models_dir}.\n")
        return 1

    sys.stderr.write("Usage: gx model [list|use <name>]\n")
    return 1


def cmd_set(args: argparse.Namespace) -> int:
    """Update a configuration key-value pair."""
    config = load_config()
    key = args.key.lower()
    value = args.value

    valid_keys = {"temp", "ctx", "threads", "max_tokens", "system", "models_dir", "host", "port", "model"}
    if key not in valid_keys:
        sys.stderr.write(f"Error: Unknown key '{key}'. Valid keys: {', '.join(sorted(valid_keys))}\n")
        return 1

    try:
        if key == "temp":
            config[key] = float(value)
        elif key in ("ctx", "threads", "max_tokens", "port"):
            config[key] = int(value)
        else:
            config[key] = value
        save_config(config)
        print(f"Set '{key}' = {config[key]}")
        return 0
    except ValueError as e:
        sys.stderr.write(f"Error: Invalid value '{value}' for key '{key}': {e}\n")
        return 1


def cmd_server(args: argparse.Namespace) -> int:
    """Manage background llama-server lifecycle (start, stop, status)."""
    action = args.server_action
    config = load_config()

    if action == "status":
        pid = get_running_server_pid()
        base_url = get_base_url(config)
        if pid:
            reachable = False
            try:
                with urllib.request.urlopen(f"{base_url}/v1/models", timeout=1.5):
                    reachable = True
            except Exception:
                pass
            status_desc = "Reachable" if reachable else "Starting/Busy"
            print(f"llama-server is RUNNING (PID: {pid}) at {base_url} [{status_desc}]")
            return 0
        print("llama-server is STOPPED")
        return 0

    if action == "stop":
        pid = get_running_server_pid()
        if not pid:
            print("llama-server is not running.")
            return 0
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.1)
                if not is_pid_alive(pid):
                    break
            if is_pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
            print(f"Stopped llama-server (PID: {pid}).")
        except Exception as e:
            sys.stderr.write(f"Error stopping process {pid}: {e}\n")
        finally:
            pid_file = get_pid_file()
            if os.path.isfile(pid_file):
                try:
                    os.remove(pid_file)
                except OSError:
                    pass
        return 0

    if action == "start":
        pid = get_running_server_pid()
        if pid:
            print(f"llama-server is already running (PID: {pid}).")
            return 0

        model_name = config.get("model", "").strip()
        if not model_name:
            sys.stderr.write("Error: No model selected. Use 'gx model list' and 'gx model use <name>'.\n")
            return 1

        models_dir = get_models_dir(config)
        model_path = os.path.join(models_dir, model_name) if not os.path.isabs(model_name) else model_name
        if not os.path.isfile(model_path):
            sys.stderr.write(f"Error: Model file '{model_path}' does not exist.\n")
            return 1

        llama_bin = find_llama_server_binary()
        if not llama_bin:
            sys.stderr.write("Error: 'llama-server' binary not found on PATH or standard locations.\n")
            sys.stderr.write("Hint: Install via 'pkg install llama-cpp' or build llama.cpp from source.\n")
            return 1

        host = config.get("host", "127.0.0.1")
        port = str(config.get("port", 8080))
        ctx = str(config.get("ctx", 2048))
        threads = str(config.get("threads", 4))

        cmd = [
            llama_bin,
            "-m",
            model_path,
            "-c",
            ctx,
            "-t",
            threads,
            "--host",
            host,
            "--port",
            port,
        ]

        log_file = get_log_file()
        try:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n--- Starting llama-server at {time.ctime()} ---\n")
                lf.flush()
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            with open(get_pid_file(), "w", encoding="utf-8") as pf:
                pf.write(str(proc.pid))

            print(f"Started llama-server (PID: {proc.pid})")
            print(f"Model: {model_name}")
            print(f"URL:   http://{host}:{port}")
            print(f"Log:   {log_file}")
            return 0
        except Exception as e:
            sys.stderr.write(f"Error launching llama-server: {e}\n")
            return 1

    sys.stderr.write("Usage: gx server [start|stop|status]\n")
    return 1


def cmd_loop(args: argparse.Namespace) -> int:
    """Execute batch or repetitive prompt loops."""
    config = load_config()
    models_dir = get_models_dir(config)

    persona_name = getattr(args, "persona", None)
    if persona_name:
        config, err = apply_persona(persona_name, config, models_dir)
        if err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

    base_url = get_base_url(config, args.host, args.port, getattr(args, "url", None))

    prompt_template = args.prompt or ""
    out_file = args.out
    until_target = args.until
    max_iter = args.max_iter if args.max_iter is not None else 5
    n_runs = args.n
    file_src = args.f

    tasks: List[str] = []
    if file_src:
        if not os.path.isfile(file_src):
            sys.stderr.write(f"Error: Input file '{file_src}' not found.\n")
            return 1
        with open(file_src, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            if "{line}" in prompt_template:
                tasks.append(prompt_template.replace("{line}", line))
            elif prompt_template:
                tasks.append(f"{prompt_template} {line}")
            else:
                tasks.append(line)
    elif n_runs:
        tasks = [prompt_template] * n_runs
    elif until_target:
        pass
    else:
        tasks = [prompt_template]

    total_tokens = 0
    total_time = 0.0
    runs_completed = 0

    print(f"Starting gx loop (Target: {base_url})")

    try:
        if until_target:
            cur_prompt = prompt_template
            iteration = 1
            while iteration <= max_iter:
                print(f"\n--- Iteration {iteration}/{max_iter} ---")
                print(f"Input: {cur_prompt}\nResponse: ", end="", flush=True)

                messages = []
                sys_prompt = config.get("system", "").strip()
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": cur_prompt})

                def on_tok(t: str) -> None:
                    sys.stdout.write(t)
                    sys.stdout.flush()

                resp, elapsed, count = stream_chat_completion(
                    messages, base_url, config, on_token=on_tok
                )
                tok_s = count / elapsed if elapsed > 0 else 0.0
                print(f"\n[{count} tokens, {tok_s:.1f} tok/s]")

                runs_completed += 1
                total_tokens += count
                total_time += elapsed

                if out_file:
                    with open(out_file, "a", encoding="utf-8") as out_f:
                        record = {
                            "iteration": iteration,
                            "input": cur_prompt,
                            "output": resp,
                            "tokens_per_sec": round(tok_s, 2),
                        }
                        out_f.write(json.dumps(record) + "\n")

                if until_target.lower() in resp.lower():
                    print(f"\n[Stopping condition met: found '{until_target}' in response]")
                    break

                cur_prompt = (
                    f"{prompt_template}\n\n"
                    f"Previous attempt output:\n{resp}\n\n"
                    f"Please refine to include or satisfy: {until_target}"
                )
                iteration += 1

        else:
            for idx, task_prompt in enumerate(tasks, 1):
                print(f"\n--- Run {idx}/{len(tasks)} ---")
                print(f"Input: {task_prompt}\nResponse: ", end="", flush=True)

                messages = []
                sys_prompt = config.get("system", "").strip()
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": task_prompt})

                def on_tok(t: str) -> None:
                    sys.stdout.write(t)
                    sys.stdout.flush()

                resp, elapsed, count = stream_chat_completion(
                    messages, base_url, config, on_token=on_tok
                )
                tok_s = count / elapsed if elapsed > 0 else 0.0
                print(f"\n[{count} tokens, {tok_per_sec:.1f} tok/s]" if 'tok_per_sec' in locals() else f"\n[{count} tokens, {tok_s:.1f} tok/s]")

                runs_completed += 1
                total_tokens += count
                total_time += elapsed

                if out_file:
                    with open(out_file, "a", encoding="utf-8") as out_f:
                        record = {
                            "iteration": idx,
                            "input": task_prompt,
                            "output": resp,
                            "tokens_per_sec": round(tok_s, 2),
                        }
                        out_f.write(json.dumps(record) + "\n")

    except KeyboardInterrupt:
        print("\n\n[Loop interrupted by user]")
    except Exception as e:
        sys.stderr.write(f"\nError in loop: {e}\n")

    avg_speed = total_tokens / total_time if total_time > 0 else 0.0
    print("\n--- Summary ---")
    print(f"Runs completed: {runs_completed}")
    print(f"Total tokens:   {total_tokens}")
    print(f"Average speed:  {avg_speed:.1f} tok/s")
    if out_file:
        print(f"Results saved:  {out_file}")

    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Perform health and environment diagnostics."""
    config = load_config()
    base_url = get_base_url(config, args.host, args.port, getattr(args, "url", None))
    models_dir = get_models_dir(config)
    sessions_dir = get_sessions_dir()
    personas_dir = get_personas_dir()
    templates_dir = get_templates_dir()
    ensure_builtin_personas()
    ensure_builtin_templates()
    all_ok = True

    print("gx Diagnostic Doctor\n" + "=" * 30)

    # 1. Python Version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 8):
        print(f" [OK]   Python version: {py_ver}")
    else:
        print(f" [FAIL] Python version: {py_ver} (< 3.8)")
        print("        Fix: pkg install python")
        all_ok = False

    # 2. Sessions Directory Writable
    if os.path.isdir(sessions_dir) and os.access(sessions_dir, os.W_OK):
        print(f" [OK]   Sessions directory is writable ({sessions_dir})")
    else:
        print(f" [FAIL] Sessions directory not writable ({sessions_dir})")
        print(f"        Fix: mkdir -p {sessions_dir} && chmod u+w {sessions_dir}")
        all_ok = False

    # 3. Sandbox / Working Directory Writable
    cwd = os.getcwd()
    if os.access(cwd, os.W_OK):
        print(f" [OK]   Sandbox directory is writable ({cwd})")
    else:
        print(f" [FAIL] Sandbox directory not writable ({cwd})")
        all_ok = False

    # 4. Personas Directory and Built-ins
    if os.path.isdir(personas_dir):
        p_files = os.listdir(personas_dir)
        builtins_ok = all(f"{b}.json" in p_files for b in BUILTIN_PERSONAS)
        if builtins_ok:
            print(f" [OK]   Personas directory and built-ins present ({len(p_files)} personas)")
        else:
            print(f" [FAIL] Some built-in personas missing in {personas_dir}")
            all_ok = False
    else:
        print(f" [FAIL] Personas directory '{personas_dir}' does not exist")
        all_ok = False

    # 4. Templates Directory and Built-ins
    if os.path.isdir(templates_dir):
        t_files = os.listdir(templates_dir)
        builtins_ok = all(f"{b}.txt" in t_files for b in BUILTIN_TEMPLATES)
        if builtins_ok:
            print(f" [OK]   Templates directory and built-ins present ({len(t_files)} templates)")
        else:
            print(f" [FAIL] Some built-in templates missing in {templates_dir}")
            all_ok = False
    else:
        print(f" [FAIL] Templates directory '{templates_dir}' does not exist")
        all_ok = False

    # 5. Models Directory & GGUF files
    if os.path.isdir(models_dir):
        ggufs = list_gguf_models(models_dir)
        if ggufs:
            print(f" [OK]   Models dir '{models_dir}' exists ({len(ggufs)} .gguf models)")
        else:
            print(f" [FAIL] Models dir '{models_dir}' has 0 .gguf files")
            print(f"        Fix: Download .gguf models into {models_dir} (e.g. gx pull <url>)")
            all_ok = False
    else:
        print(f" [FAIL] Models dir '{models_dir}' does not exist")
        print(f"        Fix: mkdir -p {models_dir} && place .gguf models there")
        all_ok = False

    # 6. Free Disk Space
    try:
        check_path = models_dir if os.path.isdir(models_dir) else get_config_dir()
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= 1.0:
            print(f" [OK]   Free disk space: {free_gb:.2f} GB available")
        else:
            print(f" [WARN] Low disk space: {free_gb:.2f} GB available (< 1.0 GB)")
    except Exception:
        pass

    # 7. llama-server binary
    llama_bin = find_llama_server_binary()
    if llama_bin:
        print(f" [OK]   llama-server found: {llama_bin}")
    else:
        print(" [FAIL] llama-server binary not found on PATH")
        print("        Fix: pkg install llama-cpp or build llama.cpp from source")
        all_ok = False

    # 8. Server reachability
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=1.5):
            print(f" [OK]   Server reachable at {base_url}")
    except Exception:
        print(f" [FAIL] Server unreachable at {base_url}")
        print("        Fix: Run 'gx server start' or check host/port settings")
        all_ok = False

    print("=" * 30)
    print("Status: " + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run unit and mock integration test suite."""
    import unittest

    test_file = os.path.join(os.path.dirname(__file__), "test_gx.py")
    if not os.path.isfile(test_file):
        test_file = "test_gx.py"

    print("Running gx selftest suite...\n")
    loader = unittest.TestLoader()
    try:
        if os.path.isfile(test_file):
            suite = loader.discover(
                start_dir=os.path.dirname(os.path.abspath(test_file)) or ".",
                pattern=os.path.basename(test_file),
            )
        else:
            suite = loader.loadTestsFromName("test_gx")
    except Exception as e:
        sys.stderr.write(f"Could not load test_gx: {e}\n")
        return 1

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="gx",
        description="Fast, zero-dependency CLI client for llama-server on Termux / Android.",
    )
    parser.add_argument("--host", help="Server host override (e.g. 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Server port override (e.g. 8080)")
    parser.add_argument("--url", help="Full base URL override (e.g. http://127.0.0.1:8080)")

    subparsers = parser.add_subparsers(dest="subcommand", title="Commands")

    # ask
    p_ask = subparsers.add_parser("ask", help="Send one-shot prompt with streaming output")
    p_ask.add_argument("prompt", nargs="?", default="", help="Prompt text (or piped via stdin)")
    p_ask.add_argument("--persona", help="Apply persona defaults (e.g. coder, tutor, terse)")

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive conversation session")
    p_chat.add_argument("--save", help="Save history to ~/.gx/sessions/<name>.json after every turn")
    p_chat.add_argument("--load", help="Resume chat from saved session <name>")
    p_chat.add_argument("--persona", help="Apply persona defaults")

    # session
    p_session = subparsers.add_parser("session", help="Manage saved chat sessions")
    p_session_sub = p_session.add_subparsers(dest="session_action")
    p_session_sub.add_parser("list", help="List saved chat sessions")

    p_show = p_session_sub.add_parser("show", help="Show conversation from a session")
    p_show.add_argument("name", help="Session name")

    p_rm = p_session_sub.add_parser("rm", help="Delete a session")
    p_rm.add_argument("name", help="Session name")
    p_rm.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    p_export = p_session_sub.add_parser("export", help="Export session as Markdown")
    p_export.add_argument("name", help="Session name")

    # persona
    p_persona = subparsers.add_parser("persona", help="Manage custom AI personas")
    p_persona_sub = p_persona.add_subparsers(dest="persona_action")
    p_persona_sub.add_parser("list", help="List available personas")

    p_pers_show = p_persona_sub.add_parser("show", help="Show persona details")
    p_pers_show.add_argument("name", help="Persona name")

    p_pers_add = p_persona_sub.add_parser("add", help="Create persona interactively")
    p_pers_add.add_argument("name", help="Persona name")

    p_pers_edit = p_persona_sub.add_parser("edit", help="Edit persona interactively")
    p_pers_edit.add_argument("name", help="Persona name")

    p_pers_rm = p_persona_sub.add_parser("rm", help="Delete a persona")
    p_pers_rm.add_argument("name", help="Persona name")

    # tpl
    p_tpl = subparsers.add_parser("tpl", help="Manage reusable prompt templates")
    p_tpl_sub = p_tpl.add_subparsers(dest="tpl_action")
    p_tpl_sub.add_parser("list", help="List templates and variable placeholders")

    p_tpl_show = p_tpl_sub.add_parser("show", help="Show template text")
    p_tpl_show.add_argument("name", help="Template name")

    p_tpl_add = p_tpl_sub.add_parser("add", help="Add template interactively")
    p_tpl_add.add_argument("name", help="Template name")

    p_tpl_rm = p_tpl_sub.add_parser("rm", help="Delete template")
    p_tpl_rm.add_argument("name", help="Template name")

    # run
    p_run = subparsers.add_parser("run", help="Run a prompt template with variable substitutions")
    p_run.add_argument("template", help="Template name to render")
    p_run.add_argument("vars", nargs="*", help="Variable key=value pairs (e.g. style=bullets)")
    p_run.add_argument("--persona", help="Apply persona defaults")

    # agent
    p_agent = subparsers.add_parser("agent", help="Autonomous tool-using agent loop")
    p_agent.add_argument("task", help="Task prompt for the agent")
    p_agent.add_argument("--max-steps", type=int, default=6, help="Maximum agent steps (default 6)")
    p_agent.add_argument("--persona", help="Apply persona defaults")
    p_agent.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt for write_file and shell")
    p_agent.add_argument("--trace", action="store_true", help="Print every step and tool output")
    p_agent.add_argument("--dir", default=".", help="Sandbox root directory (default current directory)")
    p_agent.add_argument("--save", help="Save agent session transcript to ~/.gx/sessions/<name>.json")

    # bench
    p_bench = subparsers.add_parser("bench", help="Benchmark generation and TTFT performance")
    p_bench.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default 3)")
    p_bench.add_argument("--prompt-tokens", type=int, default=128, help="Prompt token count (default 128)")
    p_bench.add_argument("--gen-tokens", type=int, default=64, help="Tokens to generate per run (default 64)")
    p_bench.add_argument("--out", help="Save benchmark result JSONL to file")
    p_bench.add_argument("--sweep", help="Sweep parameter values (e.g. threads=2,4,6)")
    p_bench.add_argument("--history", action="store_true", help="Display last 10 benchmark history runs")

    # pull
    p_pull = subparsers.add_parser("pull", help="Download GGUF model into models directory")
    p_pull.add_argument("url", help="Model URL or Hugging Face resolve/blob URL")
    p_pull.add_argument("--force", action="store_true", help="Overwrite existing file if present")
    p_pull.add_argument("--out", help="Custom output filename")

    # model
    p_model = subparsers.add_parser("model", help="List or select models")
    p_model_sub = p_model.add_subparsers(dest="model_action")
    p_model_sub.add_parser("list", help="List .gguf models in models directory")
    p_model_use = p_model_sub.add_parser("use", help="Select active model by name or substring")
    p_model_use.add_argument("name", help="Model name or partial name")

    # set
    p_set = subparsers.add_parser("set", help="Set configuration value")
    p_set.add_argument("key", help="Key name (temp, ctx, threads, max_tokens, system, etc.)")
    p_set.add_argument("value", help="Value to assign")

    # server
    p_server = subparsers.add_parser("server", help="Manage background llama-server process")
    p_server.add_argument("server_action", choices=["start", "stop", "status"], help="Action to perform")

    # loop
    p_loop = subparsers.add_parser("loop", help="Repetitive prompt execution and batch runs")
    p_loop.add_argument("prompt", nargs="?", default="", help="Prompt or template string with optional {line}")
    p_loop.add_argument("-n", type=int, help="Number of times to run prompt")
    p_loop.add_argument("-f", help="Input file to read lines from for batch execution")
    p_loop.add_argument("--until", help="Stop loop when response contains this text")
    p_loop.add_argument("--max-iter", type=int, default=5, help="Max iterations for --until (default 5)")
    p_loop.add_argument("--out", help="Output JSONL file path to save results")
    p_loop.add_argument("--persona", help="Apply persona defaults")

    # doctor
    subparsers.add_parser("doctor", help="Check Termux environment, model paths, and llama-server")

    # selftest
    subparsers.add_parser("selftest", help="Run comprehensive automated test suite")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        if not sys.stdin.isatty():
            args.prompt = ""
            return cmd_ask(args)
        parser.print_help()
        return 0

    handlers = {
        "ask": cmd_ask,
        "chat": cmd_chat,
        "session": cmd_session,
        "persona": cmd_persona,
        "tpl": cmd_tpl,
        "run": cmd_run,
        "agent": cmd_agent,
        "bench": cmd_bench,
        "pull": cmd_pull,
        "model": cmd_model,
        "set": cmd_set,
        "server": cmd_server,
        "loop": cmd_loop,
        "doctor": cmd_doctor,
        "selftest": cmd_selftest,
    }

    handler = handlers.get(args.subcommand)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
