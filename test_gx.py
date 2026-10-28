#!/usr/bin/env python3
"""Comprehensive test suite for gx CLI with Mock llama-server."""

import http.server
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import gx

MOCK_FILE_DATA = b"GGUF_MODEL_MOCK_DATA_0123456789" * 10  # 320 bytes


def _can_create_symlinks() -> bool:
    """Check whether the platform allows creating symlinks (Windows may require privileges)."""
    if os.name != "nt":
        return True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "target.txt")
            with open(target, "w") as f:
                f.write("x")
            os.symlink(target, os.path.join(tmp, "link.txt"))
        return True
    except OSError:
        return False


class MockLlamaHandler(http.server.BaseHTTPRequestHandler):
    """Mock HTTP handler emulating llama-server OpenAI API and file downloads."""

    # Records the most recent POST body so tests can assert request fidelity.
    last_request_data: Optional[Dict[str, Any]] = None

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_HEAD(self) -> None:
        if self.path.startswith("/files/mismatch.gguf"):
            self.send_response(200)
            self.send_header("Content-Length", "500")
            self.end_headers()
        elif self.path.startswith("/files/"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(MOCK_FILE_DATA)))
            self.end_headers()
        elif self.path in ("/v1/models", "/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/v1/models", "/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[{"id":"mock-model"}]}')
        elif self.path.startswith("/files/mismatch.gguf"):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", "500")
            self.end_headers()
            self.wfile.write(b"X" * 200)
        elif self.path.startswith("/files/"):
            total_len = len(MOCK_FILE_DATA)
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                try:
                    start_str = range_header[6:].split("-")[0]
                    start = int(start_str)
                    if start >= total_len:
                        self.send_response(416)
                        self.end_headers()
                        return
                    chunk = MOCK_FILE_DATA[start:]
                    self.send_response(206)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Range", f"bytes {start}-{total_len - 1}/{total_len}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                except Exception:
                    pass

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(total_len))
            self.end_headers()
            self.wfile.write(MOCK_FILE_DATA)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}
            MockLlamaHandler.last_request_data = data

            messages = data.get("messages", [])
            last_msg = messages[-1]["content"] if messages else "empty"
            task_msg = messages[1]["content"] if len(messages) > 1 else last_msg

            # Agent mock scripting
            if "AGENT_SCRIPT_SEQUENCE" in task_msg:
                if len(messages) == 2:
                    tokens = [json.dumps({"tool": "list_dir", "args": {"path": "."}})]
                elif len(messages) == 4:
                    tokens = [json.dumps({"tool": "calc", "args": {"expression": "10 * 5"}})]
                else:
                    tokens = [json.dumps({"final": "Task completed: count is 50."})]
            elif "AGENT_SCRIPT_INFINITE" in task_msg:
                tokens = [json.dumps({"tool": "calc", "args": {"expression": "1 + 1"}})]
            elif "AGENT_SCRIPT_BAD_JSON" in task_msg:
                tokens = ["Not a valid json response at all."]
            elif "AGENT_SCRIPT_WRITE_FILE" in task_msg:
                if len(messages) == 2:
                    tokens = [json.dumps({"tool": "write_file", "args": {"path": "agent_written.txt", "content": "agent payload"}})]
                else:
                    tokens = [json.dumps({"final": "Wrote file successfully."})]
            elif "AGENT_SCRIPT_SHELL" in task_msg:
                if len(messages) == 2:
                    tokens = [json.dumps({"tool": "shell", "args": {"cmd": "echo shell_executed"}})]
                else:
                    tokens = [json.dumps({"final": "Executed shell."})]
            elif "magic_word" in last_msg or "find_me" in last_msg:
                tokens = ["Found", " the", " magic_word", "!"]
            elif "unreachable_string" in last_msg:
                tokens = ["Still", " working", "..."]
            else:
                tokens = ["Echo:", " ", last_msg]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            for i, tok in enumerate(tokens):
                chunk: Dict[str, Any] = {
                    "choices": [
                        {"delta": {"content": tok}, "index": 0, "finish_reason": None}
                    ]
                }
                # Include timings object on the final token chunk
                if i == len(tokens) - 1:
                    chunk["timings"] = {
                        "prompt_n": 128,
                        "prompt_ms": 50.0,
                        "prompt_per_second": 2560.0,
                        "predicted_n": 64,
                        "predicted_ms": 1600.0,
                        "predicted_per_second": 40.0,
                    }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()


class MockLlamaServer:
    """Threaded mock llama-server on ephemeral port."""

    def __init__(self) -> None:
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), MockLlamaHandler)
        self.port = self.httpd.server_address[1]
        self.host = "127.0.0.1"
        self.url = f"http://{self.host}:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class TestGxCLI(unittest.TestCase):
    """Unit and integration tests for gx."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_server = MockLlamaServer()
        cls.mock_server.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="gx_test_")
        os.makedirs(os.path.join(self.test_dir, "models"), exist_ok=True)
        self.orig_gx_home = os.environ.get("GX_HOME")
        os.environ["GX_HOME"] = self.test_dir
        os.environ["GX_URL"] = self.mock_server.url

    def tearDown(self) -> None:
        if self.orig_gx_home:
            os.environ["GX_HOME"] = self.orig_gx_home
        else:
            os.environ.pop("GX_HOME", None)
        os.environ.pop("GX_URL", None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_config_defaults_load_save(self) -> None:
        """Test default config loading and modifying via save_config / cmd_set."""
        cfg = gx.load_config()
        self.assertEqual(cfg["temp"], 0.7)
        self.assertEqual(cfg["ctx"], 2048)
        self.assertEqual(cfg["threads"], 4)
        self.assertEqual(cfg["max_tokens"], 256)

        parser = gx.build_parser()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            args = parser.parse_args(["set", "temp", "0.2"])
            rc = gx.cmd_set(args)
        self.assertEqual(rc, 0)

        with patch("sys.stdout", captured):
            args = parser.parse_args(["set", "ctx", "4096"])
            gx.cmd_set(args)

        reloaded = gx.load_config()
        self.assertEqual(reloaded["temp"], 0.2)
        self.assertEqual(reloaded["ctx"], 4096)

    def test_model_matching_exact_and_partial(self) -> None:
        """Test model substring matching and edge cases."""
        models = [
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "llama-3.2-1b-instruct-q8_0.gguf",
            "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        ]

        match, _ = gx.match_model("llama-3.2-1b-instruct-q8_0.gguf", models)
        self.assertEqual(match, "llama-3.2-1b-instruct-q8_0.gguf")

        match, _ = gx.match_model("qwen", models)
        self.assertEqual(match, "qwen2.5-0.5b-instruct-q4_k_m.gguf")

        match, _ = gx.match_model("MISTRAL", models)
        self.assertEqual(match, "mistral-7b-instruct-v0.3.Q4_K_M.gguf")

        models_ambig = ["qwen-0.5b.gguf", "qwen-1.5b.gguf"]
        match, candidates = gx.match_model("qwen", models_ambig)
        self.assertIsNone(match)
        self.assertEqual(len(candidates), 2)

        match, candidates = gx.match_model("nonexistent", models)
        self.assertIsNone(match)
        self.assertEqual(len(candidates), 0)

    def test_quantization_extraction(self) -> None:
        """Test quantization string extraction."""
        self.assertEqual(gx.extract_quantization("qwen2.5-0.5b-instruct-q4_k_m.gguf"), "Q4_K_M")
        self.assertEqual(gx.extract_quantization("llama-3.2-1b-instruct-q8_0.gguf"), "Q8_0")
        self.assertEqual(gx.extract_quantization("model-f16.gguf"), "F16")
        self.assertEqual(gx.extract_quantization("custom.gguf"), "UNKNOWN")

    def test_sse_parsing_standard(self) -> None:
        """Test standard SSE parsing."""
        raw = (
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        tokens, buf, is_done = gx.parse_sse_chunks(raw, "")
        self.assertEqual(tokens, ["Hello", " world"])
        self.assertTrue(is_done)
        self.assertEqual(buf, "")

    def test_sse_parsing_malformed_and_split_chunks(self) -> None:
        """Test parsing SSE stream with malformed lines, comment lines, and chunk boundaries."""
        chunk1 = b'data: {"choices":[{"delta":{"content":"Good"'
        tokens1, buf1, is_done1 = gx.parse_sse_chunks(chunk1, "")
        self.assertEqual(tokens1, [])
        self.assertFalse(is_done1)
        self.assertTrue(len(buf1) > 0)

        chunk2 = b'}}]}\n\n: this is a comment\ndata: {invalid json}\ndata: {"choices":[{"delta":{"content":" morning"}}]}\n\n'
        tokens2, buf2, is_done2 = gx.parse_sse_chunks(chunk2, buf1)
        self.assertEqual(tokens2, ["Good", " morning"])
        self.assertFalse(is_done2)

        chunk3 = b"data: [DONE]\n\n"
        tokens3, buf3, is_done3 = gx.parse_sse_chunks(chunk3, buf2)
        self.assertEqual(tokens3, [])
        self.assertTrue(is_done3)

    def test_ask_command_streaming(self) -> None:
        """Test one-shot ask streaming against mock server."""
        parser = gx.build_parser()
        args = parser.parse_args(["ask", "Ping"])

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_ask(args)

        self.assertEqual(rc, 0)
        output = captured_out.getvalue()
        self.assertIn("Echo: Ping", output)
        self.assertIn("tok/s", output)

    def test_stdin_piping(self) -> None:
        """Test stdin piping into prompt."""
        parser = gx.build_parser()
        args = parser.parse_args(["ask", "Summarize:"])

        captured_out = io.StringIO()
        with patch("sys.stdin", io.StringIO("File contents from pipe")), patch("sys.stdout", captured_out):
            rc = gx.cmd_ask(args)

        self.assertEqual(rc, 0)
        output = captured_out.getvalue()
        self.assertIn("Echo: Summarize:\n\nFile contents from pipe", output)

    def test_loop_with_n(self) -> None:
        """Test gx loop with -n 3 runs and JSONL output."""
        out_jsonl = os.path.join(self.test_dir, "results.jsonl")
        parser = gx.build_parser()
        args = parser.parse_args(["loop", "Test N", "-n", "3", "--out", out_jsonl])

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_loop(args)

        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(out_jsonl))
        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0]["iteration"], 1)
        self.assertEqual(lines[0]["input"], "Test N")
        self.assertIn("Echo: Test N", lines[0]["output"])

    def test_loop_with_f(self) -> None:
        """Test gx loop with -f file batch line substitution."""
        input_file = os.path.join(self.test_dir, "inputs.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("apple\nbanana\ncherry\n")

        out_jsonl = os.path.join(self.test_dir, "batch_out.jsonl")
        parser = gx.build_parser()
        args = parser.parse_args(
            ["loop", "Translate {line} to French", "-f", input_file, "--out", out_jsonl]
        )

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_loop(args)

        self.assertEqual(rc, 0)
        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0]["input"], "Translate apple to French")
        self.assertEqual(lines[1]["input"], "Translate banana to French")
        self.assertEqual(lines[2]["input"], "Translate cherry to French")

    def test_loop_until_condition_and_max_iter(self) -> None:
        """Test gx loop --until stopping condition and max iterations."""
        parser = gx.build_parser()
        args = parser.parse_args(["loop", "Please output magic_word", "--until", "magic_word"])

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_loop(args)

        self.assertEqual(rc, 0)
        output = captured_out.getvalue()
        self.assertIn("Stopping condition met", output)

        out_jsonl = os.path.join(self.test_dir, "until_cap.jsonl")
        args_cap = parser.parse_args(
            ["loop", "Never met", "--until", "unreachable_string", "--max-iter", "3", "--out", out_jsonl]
        )
        with patch("sys.stdout", io.StringIO()):
            rc = gx.cmd_loop(args_cap)

        self.assertEqual(rc, 0)
        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 3)

    def test_error_handling_server_down(self) -> None:
        """Test clear error when server is down."""
        parser = gx.build_parser()
        args = parser.parse_args(["--port", "59999", "ask", "hello"])

        os.environ.pop("GX_URL", None)
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_ask(args)

        self.assertEqual(rc, 1)
        err_msg = captured_err.getvalue()
        self.assertIn("Cannot connect to llama-server", err_msg)
        self.assertIn("gx server start", err_msg)

    def test_error_handling_no_model_selected(self) -> None:
        """Test error when starting server without a model configured."""
        parser = gx.build_parser()
        args = parser.parse_args(["server", "start"])

        cfg = gx.load_config()
        cfg["model"] = ""
        gx.save_config(cfg)

        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_server(args)

        self.assertEqual(rc, 1)
        self.assertIn("No model selected", captured_err.getvalue())

    def test_session_save_load_roundtrip(self) -> None:
        """Test session save and load roundtrip."""
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I print in Python?"},
            {"role": "assistant", "content": "Use print('hello')."},
        ]
        ok = gx.save_session("my_session", "qwen-test.gguf", "You are a helpful assistant.", msgs)
        self.assertTrue(ok)

        data, err = gx.load_session("my_session")
        self.assertIsNone(err)
        self.assertEqual(data["name"], "my_session")
        self.assertEqual(data["model"], "qwen-test.gguf")
        self.assertEqual(len(data["messages"]), 3)

    def test_session_invalid_name_rejected(self) -> None:
        """Test rejection of invalid session names."""
        self.assertFalse(gx.validate_session_name("bad name with spaces"))
        self.assertFalse(gx.validate_session_name("../traversal"))
        self.assertFalse(gx.validate_session_name("session*"))
        self.assertTrue(gx.validate_session_name("valid-name_123"))

    def test_session_corrupt_file_handled(self) -> None:
        """Test corrupt JSON session files are handled gracefully without tracebacks."""
        sessions_dir = gx.get_sessions_dir()
        corrupt_file = os.path.join(sessions_dir, "broken.json")
        with open(corrupt_file, "w") as f:
            f.write("{invalid json content---")

        data, err = gx.load_session("broken")
        self.assertIsNone(data)
        self.assertIn("corrupted", err)

    def test_context_trimming_keeps_system_prompt(self) -> None:
        """Test context guard drops oldest user/assistant pairs while preserving system prompt."""
        sys_msg = {"role": "system", "content": "SYSTEM_PROMPT_CRITICAL"}
        old_user1 = {"role": "user", "content": "A" * 200}
        old_asst1 = {"role": "assistant", "content": "B" * 200}
        new_user2 = {"role": "user", "content": "Current question"}

        msgs = [sys_msg, old_user1, old_asst1, new_user2]
        trimmed, dropped = gx.apply_context_guard(msgs, ctx_limit=100, threshold_pct=0.8)

        self.assertEqual(dropped, 2)
        self.assertEqual(trimmed[0], sys_msg)
        self.assertEqual(trimmed[1], new_user2)

    def test_session_subcommands(self) -> None:
        """Test session list, show, export, and rm subcommands."""
        msgs = [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]
        gx.save_session("test_subcmd", "mock_model", "Helpful assistant", msgs)
        parser = gx.build_parser()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            gx.cmd_session(parser.parse_args(["session", "list"]))
        self.assertIn("test_subcmd", captured.getvalue())

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            gx.cmd_session(parser.parse_args(["session", "show", "test_subcmd"]))
        self.assertIn("Hello!", captured.getvalue())

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            gx.cmd_session(parser.parse_args(["session", "export", "test_subcmd"]))
        self.assertIn("# Chat Session: test_subcmd", captured.getvalue())

        with patch("sys.stdout", io.StringIO()):
            rc = gx.cmd_session(parser.parse_args(["session", "rm", "test_subcmd", "-y"]))
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.isfile(gx.get_session_file("test_subcmd")))

    def test_pull_blob_to_resolve_url(self) -> None:
        """Test conversion of Hugging Face blob URLs to resolve URLs."""
        blob_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-GGUF/blob/main/model.gguf"
        expected = "https://huggingface.co/Qwen/Qwen2.5-0.5B-GGUF/resolve/main/model.gguf"
        self.assertEqual(gx.normalize_model_url(blob_url), expected)

    def test_pull_download_and_resume_with_range(self) -> None:
        """Test full download and resume with HTTP Range header."""
        models_dir = os.path.join(self.test_dir, "models")
        cfg = gx.load_config()
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        part_file = os.path.join(models_dir, "test_dl.gguf.part")
        with open(part_file, "wb") as f:
            f.write(MOCK_FILE_DATA[:100])

        parser = gx.build_parser()
        args = parser.parse_args(["pull", f"{self.mock_server.url}/files/test_dl.gguf", "--out", "test_dl.gguf"])
        with patch("sys.stdout", io.StringIO()):
            rc = gx.cmd_pull(args)
        self.assertEqual(rc, 0)
        final_file = os.path.join(models_dir, "test_dl.gguf")
        self.assertEqual(os.path.getsize(final_file), len(MOCK_FILE_DATA))

    def test_pull_existing_file_refused_without_force(self) -> None:
        """Test refusal to overwrite existing model file without --force."""
        models_dir = os.path.join(self.test_dir, "models")
        existing = os.path.join(models_dir, "existing.gguf")
        with open(existing, "wb") as f:
            f.write(b"ORIGINAL")

        cfg = gx.load_config()
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        parser = gx.build_parser()
        args = parser.parse_args(["pull", f"{self.mock_server.url}/files/existing.gguf"])
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_pull(args)
        self.assertEqual(rc, 1)
        self.assertIn("already exists", captured_err.getvalue())

    def test_pull_size_mismatch_detected(self) -> None:
        """Test error when downloaded bytes do not match expected Content-Length."""
        models_dir = os.path.join(self.test_dir, "models")
        cfg = gx.load_config()
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        parser = gx.build_parser()
        args = parser.parse_args(["pull", f"{self.mock_server.url}/files/mismatch.gguf"])
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err), patch("sys.stdout", io.StringIO()):
            rc = gx.cmd_pull(args)
        self.assertEqual(rc, 1)
        self.assertIn("Size mismatch", captured_err.getvalue())

    # Phase 3 Tests: Personas, Templates & Bench

    def test_personas_builtin_and_roundtrip(self) -> None:
        """Test built-in persona seeding, custom add/show/rm, and fallback."""
        gx.ensure_builtin_personas()
        p_dir = gx.get_personas_dir()
        self.assertTrue(os.path.isfile(os.path.join(p_dir, "coder.json")))
        self.assertTrue(os.path.isfile(os.path.join(p_dir, "tutor.json")))
        self.assertTrue(os.path.isfile(os.path.join(p_dir, "terse.json")))

        # Custom persona save & load
        custom = {"name": "custom_bot", "system": "Be helpful.", "temp": 0.5, "max_tokens": 300, "model": "fake.gguf"}
        self.assertTrue(gx.save_persona("custom_bot", custom))
        loaded, err = gx.load_persona("custom_bot")
        self.assertIsNone(err)
        self.assertEqual(loaded["temp"], 0.5)

        # Apply persona with missing pinned model fallback
        cfg = gx.load_config()
        cfg["model"] = "current_active.gguf"
        models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            new_cfg, err = gx.apply_persona("custom_bot", cfg, models_dir)

        self.assertIsNone(err)
        self.assertEqual(new_cfg["temp"], 0.5)
        self.assertEqual(new_cfg["max_tokens"], 300)
        self.assertEqual(new_cfg["model"], "current_active.gguf")  # Kept fallback
        self.assertIn("Falling back", captured_err.getvalue())

        # List & Show subcommands
        parser = gx.build_parser()
        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            gx.cmd_persona(parser.parse_args(["persona", "list"]))
        self.assertIn("coder", captured_out.getvalue())
        self.assertIn("custom_bot", captured_out.getvalue())

        # Rm subcommand
        with patch("sys.stdout", io.StringIO()):
            gx.cmd_persona(parser.parse_args(["persona", "rm", "custom_bot"]))
        self.assertFalse(os.path.isfile(os.path.join(p_dir, "custom_bot.json")))

    def test_templates_builtin_render_and_validation(self) -> None:
        """Test prompt template builtins, variable substitutions, and error handling."""
        gx.ensure_builtin_templates()
        t_dir = gx.get_templates_dir()
        self.assertTrue(os.path.isfile(os.path.join(t_dir, "summarize.txt")))
        self.assertTrue(os.path.isfile(os.path.join(t_dir, "explain.txt")))

        parser = gx.build_parser()

        # 1. Success template run
        args = parser.parse_args(["run", "explain", "topic=AsyncIO", "level=expert"])
        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_run(args)
        self.assertEqual(rc, 0)
        self.assertIn("Echo: Explain AsyncIO to a expert audience", captured_out.getvalue())

        # 2. Missing variable error
        args_missing = parser.parse_args(["run", "explain", "topic=AsyncIO"])
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_run(args_missing)
        self.assertEqual(rc, 1)
        self.assertIn("Missing required template variable", captured_err.getvalue())
        self.assertIn("level", captured_err.getvalue())

        # 3. Unused variable warning
        args_unused = parser.parse_args(["run", "explain", "topic=Python", "level=kids", "extra=ignored"])
        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_run(args_unused)
        self.assertEqual(rc, 0)
        self.assertIn("Warning: Unused variable(s): extra", captured_out.getvalue())

        # 4. Input placeholder piped via stdin
        args_summarize = parser.parse_args(["run", "summarize", "style=bullet_points"])
        captured_out = io.StringIO()
        with patch("sys.stdin", io.StringIO("My raw article text")), patch("sys.stdout", captured_out):
            rc = gx.cmd_run(args_summarize)
        self.assertEqual(rc, 0)
        self.assertIn("Echo: Summarize the following content in bullet_points style:\n\nMy raw article text", captured_out.getvalue())

    def test_bench_suite_and_history(self) -> None:
        """Test benchmark execution, timings parsing, history appending, and throttle logic."""
        cfg = gx.load_config()
        cfg["model"] = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        gx.save_config(cfg)

        parser = gx.build_parser()
        bench_out = os.path.join(self.test_dir, "bench_test.jsonl")
        args = parser.parse_args(["bench", "--runs", "2", "--prompt-tokens", "10", "--gen-tokens", "20", "--out", bench_out])

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_bench(args)

        self.assertEqual(rc, 0)
        out_text = captured_out.getvalue()
        self.assertIn("Model:", out_text)
        self.assertIn("Quantization: Q4_K_M", out_text)
        self.assertIn("TTFT", out_text)
        self.assertIn("Generation Speed:", out_text)

        # Check bench history
        self.assertTrue(os.path.isfile(gx.get_bench_history_file()))
        args_hist = parser.parse_args(["bench", "--history"])
        captured_hist = io.StringIO()
        with patch("sys.stdout", captured_hist):
            rc_hist = gx.cmd_bench(args_hist)
        self.assertEqual(rc_hist, 0)
        self.assertIn("Q4_K_M", captured_hist.getvalue())

    def test_bench_throttle_warning(self) -> None:
        """Test run_benchmark_suite's >25% first-to-last speed drop detection."""
        cfg = gx.load_config()

        def fake_result(gen_tok_s: float):
            return {
                "text": "x",
                "elapsed": 1.0,
                "token_count": 1,
                "ttft_ms": 10.0,
                "prompt_eval_tok_s": 100.0,
                "gen_tok_s": gen_tok_s,
                "timings": None,
            }

        # 1. Declining speeds (40.0 -> 25.0, a 37.5% drop) -> throttle warning
        speeds = iter([40.0, 25.0])
        with patch(
            "gx.stream_chat_completion_detailed",
            lambda messages, base_url, config, on_token=None, max_tokens_override=None: fake_result(next(speeds)),
        ):
            res = gx.run_benchmark_suite(2, 8, 8, "http://127.0.0.1:1", cfg)
        self.assertTrue(res["throttle_warning"])
        self.assertEqual(res["gen_speeds"], [40.0, 25.0])

        # 2. Constant speeds -> no warning
        with patch(
            "gx.stream_chat_completion_detailed",
            lambda messages, base_url, config, on_token=None, max_tokens_override=None: fake_result(40.0),
        ):
            res_steady = gx.run_benchmark_suite(2, 8, 8, "http://127.0.0.1:1", cfg)
        self.assertFalse(res_steady["throttle_warning"])

    def test_doctor_diagnostics_phase3(self) -> None:
        """Test gx doctor checks including personas, templates, and sandbox."""
        models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "test.gguf"), "w") as f:
            f.write("dummy gguf")

        cfg = gx.load_config()
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        parser = gx.build_parser()
        args = parser.parse_args(["doctor"])

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch(
            "gx.find_llama_server_binary", return_value="/fake/llama-server"
        ):
            rc = gx.cmd_doctor(args)
        self.assertEqual(rc, 0)
        out = captured_out.getvalue()
        self.assertIn("ALL CHECKS PASSED", out)
        self.assertIn("Python version", out)
        self.assertIn("Sessions directory", out)
        self.assertIn("Sandbox directory is writable", out)
        self.assertIn("Personas directory", out)
        self.assertIn("Templates directory", out)
        self.assertIn("Models dir", out)
        self.assertIn("Free disk space", out)

        # Missing llama-server binary -> failure exit code
        captured_out2 = io.StringIO()
        with patch("sys.stdout", captured_out2), patch(
            "gx.find_llama_server_binary", return_value=None
        ):
            rc2 = gx.cmd_doctor(args)
        self.assertEqual(rc2, 1)
        self.assertIn("llama-server binary not found", captured_out2.getvalue())

    # Phase 4 Tests: Agent, Tools, Sandbox & JSON Protocol

    def test_json_extraction_formats(self) -> None:
        """Test JSON extraction from plain, code fences, chatter, malformed, and multiple objects."""
        # 1. Plain JSON
        plain = '{"tool": "calc", "args": {"expression": "2 + 2"}}'
        obj1 = gx.extract_first_json_object(plain)
        self.assertIsNotNone(obj1)
        self.assertEqual(obj1.get("tool"), "calc")

        # 2. In markdown code fences
        fenced = "```json\n{\"final\": \"All steps completed.\"}\n```"
        obj2 = gx.extract_first_json_object(fenced)
        self.assertIsNotNone(obj2)
        self.assertEqual(obj2.get("final"), "All steps completed.")

        # 3. Chatter around JSON
        chatter = "Sure! Here is your tool call:\n{\"tool\": \"read_file\", \"args\": {\"path\": \"a.txt\"}}\nLet me know!"
        obj3 = gx.extract_first_json_object(chatter)
        self.assertIsNotNone(obj3)
        self.assertEqual(obj3.get("tool"), "read_file")

        # 4. Malformed JSON
        malformed = "{\"tool\": calc, incomplete..."
        self.assertIsNone(gx.extract_first_json_object(malformed))

        # 5. Multiple JSON objects -> extracts the FIRST object
        multiple = '{"tool": "first_tool", "args": {}} and {"tool": "second_tool", "args": {}}'
        obj5 = gx.extract_first_json_object(multiple)
        self.assertIsNotNone(obj5)
        self.assertEqual(obj5.get("tool"), "first_tool")

    def test_sandbox_path_validation_and_rejections(self) -> None:
        """Test sandbox path resolution and rejection of .. and absolute escapes."""
        sandbox = os.path.join(self.test_dir, "agent_sandbox")
        os.makedirs(sandbox, exist_ok=True)
        inside_file = os.path.join(sandbox, "valid.txt")
        with open(inside_file, "w") as f:
            f.write("hello sandbox")

        # 1. Valid paths inside
        resolved, err = gx.resolve_sandbox_path(sandbox, "valid.txt")
        self.assertIsNone(err)
        self.assertEqual(resolved, os.path.realpath(inside_file))

        # 2. Escape via ..
        _, err_dotdot = gx.resolve_sandbox_path(sandbox, "../outside.txt")
        self.assertIsNotNone(err_dotdot)
        self.assertIn("escapes sandbox", err_dotdot)

        # 3. Escape via absolute path
        _, err_abs = gx.resolve_sandbox_path(sandbox, "/etc/passwd")
        self.assertIsNotNone(err_abs)
        self.assertIn("escapes sandbox", err_abs)

    @unittest.skipUnless(_can_create_symlinks(), "symlink creation requires privileges on this platform")
    def test_sandbox_symlink_escape_rejected(self) -> None:
        """Test that a symlink pointing outside the sandbox is rejected."""
        sandbox = os.path.join(self.test_dir, "agent_sandbox_link")
        os.makedirs(sandbox, exist_ok=True)

        outside_dir = tempfile.mkdtemp(prefix="outside_")
        outside_file = os.path.join(outside_dir, "secret.txt")
        with open(outside_file, "w") as f:
            f.write("secret")

        symlink_path = os.path.join(sandbox, "escape_link")
        try:
            os.symlink(outside_dir, symlink_path)
            _, err_symlink = gx.resolve_sandbox_path(sandbox, "escape_link/secret.txt")
            self.assertIsNotNone(err_symlink)
            self.assertIn("escapes sandbox", err_symlink)
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_calc_tool_safety_and_math(self) -> None:
        """Test safe AST arithmetic evaluator and rejection of code injection."""
        self.assertEqual(gx.tool_calc("2 + 3 * 4"), "14")
        self.assertEqual(gx.tool_calc("(10 - 2) / 2 ** 3"), "1")
        self.assertEqual(gx.tool_calc("-5 + 8 % 3"), "-3")
        self.assertEqual(gx.tool_calc("10 // 3"), "3")
        self.assertIn("Division by zero", gx.tool_calc("10 / 0"))

        # Injection rejections
        self.assertIn("Error", gx.tool_calc("__import__('os').system('ls')"))
        self.assertIn("Error", gx.tool_calc("open('/etc/passwd').read()"))
        self.assertIn("Error", gx.tool_calc("math.sin(1)"))

    def test_agent_scripted_loop_and_transcript_logging(self) -> None:
        """Test end-to-end agent loop with scripted tool calls, termination, and logging."""
        sandbox = os.path.join(self.test_dir, "sandbox_run")
        os.makedirs(sandbox, exist_ok=True)
        test_file = os.path.join(sandbox, "sample.txt")
        with open(test_file, "w") as f:
            f.write("sample content")

        parser = gx.build_parser()
        args = parser.parse_args(["agent", "AGENT_SCRIPT_SEQUENCE", "--dir", sandbox, "--save", "agent_sess_1"])

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_agent(args)

        self.assertEqual(rc, 0)
        out_text = captured_out.getvalue()
        self.assertIn("Task completed: count is 50.", out_text)

        # Verify agent log was appended
        log_file = gx.get_agent_log_file()
        self.assertTrue(os.path.isfile(log_file))
        with open(log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
        last_log = logs[-1]
        self.assertEqual(last_log["task"], "AGENT_SCRIPT_SEQUENCE")
        self.assertEqual(last_log["outcome"], "final")
        self.assertIn("list_dir", last_log["tools_used"])
        self.assertIn("calc", last_log["tools_used"])

        # Verify session transcript saved
        sess_file = gx.get_session_file("agent_sess_1")
        self.assertTrue(os.path.isfile(sess_file))

    def test_agent_step_limit_enforcement(self) -> None:
        """Test agent stops strictly at max-steps limit and records step_limit outcome."""
        sandbox = os.path.join(self.test_dir, "sandbox_limit")
        os.makedirs(sandbox, exist_ok=True)

        parser = gx.build_parser()
        args = parser.parse_args(["agent", "AGENT_SCRIPT_INFINITE", "--max-steps", "3", "--dir", sandbox])

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            rc = gx.cmd_agent(args)

        self.assertEqual(rc, 0)
        self.assertIn("stopped: step limit", captured_out.getvalue())

    def test_agent_consecutive_bad_json_aborts(self) -> None:
        """Test agent aborts after 2 consecutive non-JSON replies."""
        sandbox = os.path.join(self.test_dir, "sandbox_bad")
        os.makedirs(sandbox, exist_ok=True)

        parser = gx.build_parser()
        args = parser.parse_args(["agent", "AGENT_SCRIPT_BAD_JSON", "--max-steps", "5", "--dir", sandbox])

        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_agent(args)

        self.assertEqual(rc, 1)
        self.assertIn("Aborted: 2 consecutive invalid format responses", captured_err.getvalue())

    def test_agent_write_file_confirmation_and_yes_bypass(self) -> None:
        """Test confirmation refusal vs --yes bypass for write_file tool."""
        sandbox = os.path.join(self.test_dir, "sandbox_write")
        os.makedirs(sandbox, exist_ok=True)

        # 1. Denied via prompt
        res_denied = gx.tool_write_file(sandbox, "denied.txt", "data", yes=False, confirm_fn=lambda _: False)
        self.assertIn("Tool aborted: user denied", res_denied)
        self.assertFalse(os.path.isfile(os.path.join(sandbox, "denied.txt")))

        # 2. Confirmed via prompt
        res_conf = gx.tool_write_file(sandbox, "conf.txt", "data", yes=False, confirm_fn=lambda _: True)
        self.assertIn("Successfully wrote", res_conf)
        self.assertTrue(os.path.isfile(os.path.join(sandbox, "conf.txt")))

        # 3. Bypassed via yes=True
        res_yes = gx.tool_write_file(sandbox, "yes.txt", "data", yes=True)
        self.assertIn("Successfully wrote", res_yes)
        self.assertTrue(os.path.isfile(os.path.join(sandbox, "yes.txt")))

    def test_agent_shell_confirmation_and_yes_bypass(self) -> None:
        """Test confirmation refusal vs --yes bypass for shell tool."""
        sandbox = os.path.join(self.test_dir, "sandbox_sh")
        os.makedirs(sandbox, exist_ok=True)

        # 1. Denied
        res_denied = gx.tool_shell(sandbox, "echo denied", yes=False, confirm_fn=lambda _: False)
        self.assertIn("Tool aborted: user denied", res_denied)

        # 2. Confirmed
        res_conf = gx.tool_shell(sandbox, "echo confirmed_run", yes=False, confirm_fn=lambda _: True)
        self.assertIn("confirmed_run", res_conf)

        # 3. Yes flag
        res_yes = gx.tool_shell(sandbox, "echo yes_run", yes=True)
        self.assertIn("yes_run", res_yes)

    def test_shell_timeout_and_output_cap(self) -> None:
        """Test shell tool output truncation at 2000 chars and timeout handling."""
        sandbox = os.path.join(self.test_dir, "sandbox_cap")
        os.makedirs(sandbox, exist_ok=True)

        # 1. Output capping
        cmd_big = f'"{sys.executable}" -c "print(\'A\' * 3000)"'
        out = gx.tool_shell(sandbox, cmd_big, yes=True)
        self.assertIn("[Truncated: output exceeded 2000 chars]", out)
        self.assertTrue(len(out) <= 2100)


    # ---- Remediation tests: server lifecycle, agent gating, model use, sweeps, pull hardening ----

    def test_server_start_waits_for_readiness(self) -> None:
        """Test server start launches the process, polls /v1/models, and reports ready."""
        models_dir = os.path.join(self.test_dir, "models")
        with open(os.path.join(models_dir, "test.gguf"), "w") as f:
            f.write("gguf")
        cfg = gx.load_config()
        cfg["model"] = "test.gguf"
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        class FakeProc:
            pid = 424242
            returncode = None

            def poll(self):
                return None

        parser = gx.build_parser()
        args = parser.parse_args(["server", "start"])
        captured = io.StringIO()
        try:
            with patch("sys.stdout", captured), patch(
                "gx.find_llama_server_binary", return_value="/fake/llama-server"
            ), patch("gx.subprocess.Popen", return_value=FakeProc()):
                rc = gx.cmd_server(args)
        finally:
            pid_file = gx.get_pid_file()
            if os.path.isfile(pid_file):
                os.remove(pid_file)

        self.assertEqual(rc, 0)
        out = captured.getvalue()
        self.assertIn("Started llama-server (PID: 424242)", out)
        self.assertIn("Server ready at", out)
        self.assertIn(self.mock_server.url, out)

    def test_server_start_detects_early_exit(self) -> None:
        """Test server start reports an error and cleans the pid file if the process dies."""
        models_dir = os.path.join(self.test_dir, "models")
        with open(os.path.join(models_dir, "test.gguf"), "w") as f:
            f.write("gguf")
        cfg = gx.load_config()
        cfg["model"] = "test.gguf"
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        class DeadProc:
            pid = 424243
            returncode = 1

            def poll(self):
                return 1

        parser = gx.build_parser()
        args = parser.parse_args(["server", "start"])
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with patch("sys.stdout", captured_out), patch("sys.stderr", captured_err), patch(
            "gx.find_llama_server_binary", return_value="/fake/llama-server"
        ), patch("gx.subprocess.Popen", return_value=DeadProc()):
            rc = gx.cmd_server(args)

        self.assertEqual(rc, 1)
        self.assertIn("exited during startup", captured_err.getvalue())
        self.assertFalse(os.path.isfile(gx.get_pid_file()))

    def test_server_start_already_running_and_status(self) -> None:
        """Test the already-running guard and status output for running/stopped states."""
        parser = gx.build_parser()
        pid_file = gx.get_pid_file()
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))  # this test process is definitely alive

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_server(parser.parse_args(["server", "start"]))
        self.assertEqual(rc, 0)
        self.assertIn("already running", captured.getvalue())

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_server(parser.parse_args(["server", "status"]))
        self.assertEqual(rc, 0)
        self.assertIn("RUNNING", captured.getvalue())

        os.remove(pid_file)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_server(parser.parse_args(["server", "status"]))
        self.assertEqual(rc, 0)
        self.assertIn("STOPPED", captured.getvalue())

    def test_server_stop_terminates_process_and_clears_pid_file(self) -> None:
        """Test stop signals the recorded process (SIGTERM first, SIGKILL escalation) and clears the pid file."""
        parser = gx.build_parser()
        pid_file = gx.get_pid_file()

        # Scenario 1: SIGTERM stops the process -> no escalation
        with open(pid_file, "w") as f:
            f.write("424244")
        kills: List[int] = []
        alive = {"status": True}

        def fake_kill(pid, sig):
            kills.append(sig)
            alive["status"] = False  # process dies on first signal

        captured = io.StringIO()
        with patch("sys.stdout", captured), patch("gx.os.kill", fake_kill), patch(
            "gx.is_pid_alive", lambda pid: alive["status"]
        ):
            rc = gx.cmd_server(parser.parse_args(["server", "stop"]))
        self.assertEqual(rc, 0)
        self.assertIn("Stopped llama-server", captured.getvalue())
        self.assertEqual(kills, [signal.SIGTERM])
        self.assertFalse(os.path.isfile(pid_file))

        # Scenario 2: process ignores SIGTERM -> SIGKILL escalation
        with open(pid_file, "w") as f:
            f.write("424245")
        kills2: List[int] = []
        alive2 = {"status": True}
        call_count = {"n": 0}
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)

        def fake_kill2(pid, sig):
            kills2.append(sig)
            call_count["n"] += 1
            # On Windows, SIGKILL == SIGTERM, so we use call count to simulate escalation
            if call_count["n"] >= 2:
                alive2["status"] = False

        captured2 = io.StringIO()
        with patch("sys.stdout", captured2), patch("sys.stderr", io.StringIO()), patch(
            "gx.os.kill", fake_kill2
        ), patch("gx.is_pid_alive", lambda pid: alive2["status"]):
            rc2 = gx.cmd_server(parser.parse_args(["server", "stop"]))
        self.assertEqual(rc2, 0)
        self.assertIn("Stopped llama-server", captured2.getvalue())
        # On Windows, SIGKILL == SIGTERM, so escalation sends SIGTERM twice
        self.assertEqual(kills2, [signal.SIGTERM, sigkill])
        self.assertFalse(os.path.isfile(pid_file))

    def test_agent_shell_disabled_by_config(self) -> None:
        """Test agent_shell=false removes the shell tool and refuses shell calls."""
        cfg = gx.load_config()
        cfg["agent_shell"] = False
        gx.save_config(cfg)

        sandbox = os.path.join(self.test_dir, "sandbox_gated")
        os.makedirs(sandbox, exist_ok=True)
        parser = gx.build_parser()
        args = parser.parse_args([
            "agent", "AGENT_SCRIPT_SHELL", "--dir", sandbox, "--trace", "--save", "gated_sess",
        ])

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_agent(args)

        self.assertEqual(rc, 0)
        self.assertIn("shell tool is disabled by configuration", captured.getvalue())

        data, err = gx.load_session("gated_sess")
        self.assertIsNone(err)
        self.assertNotIn("6. shell", data["messages"][0]["content"])
        self.assertIn("shell tool is disabled", json.dumps(data["messages"]))

    def test_agent_shell_enabled_with_yes(self) -> None:
        """Test agent_shell=true with --yes executes the shell tool end to end."""
        sandbox = os.path.join(self.test_dir, "sandbox_shell_on")
        os.makedirs(sandbox, exist_ok=True)
        parser = gx.build_parser()
        args = parser.parse_args([
            "agent", "AGENT_SCRIPT_SHELL", "--dir", sandbox, "--trace", "--yes", "--save", "sh_on_sess",
        ])

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_agent(args)

        self.assertEqual(rc, 0)
        self.assertIn("shell_executed", captured.getvalue())

        data, err = gx.load_session("sh_on_sess")
        self.assertIsNone(err)
        self.assertIn("6. shell", data["messages"][0]["content"])

    def test_model_list_and_use(self) -> None:
        """Test model list marker, use persistence, ambiguity, and no-match errors."""
        models_dir = os.path.join(self.test_dir, "models")
        for name in ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "llama-3.2-1b-instruct-q8_0.gguf"):
            with open(os.path.join(models_dir, name), "w") as f:
                f.write("x")
        cfg = gx.load_config()
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        parser = gx.build_parser()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_model(parser.parse_args(["model", "list"]))
        self.assertEqual(rc, 0)
        self.assertIn("qwen2.5-0.5b-instruct-q4_k_m.gguf", captured.getvalue())

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_model(parser.parse_args(["model", "use", "llama"]))
        self.assertEqual(rc, 0)
        self.assertIn("Active model set to", captured.getvalue())
        self.assertEqual(gx.load_config()["model"], "llama-3.2-1b-instruct-q8_0.gguf")

        # Ambiguous substring -> error listing candidates
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_model(parser.parse_args(["model", "use", "gguf"]))
        self.assertEqual(rc, 1)
        self.assertIn("Multiple models match", captured_err.getvalue())

        # No match
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_model(parser.parse_args(["model", "use", "nonexistent-model"]))
        self.assertEqual(rc, 1)
        self.assertIn("No model matching", captured_err.getvalue())

    def test_bench_sweep_validation_and_config_restore(self) -> None:
        """Test sweep prefix validation and that the threads config is restored after a sweep."""
        parser = gx.build_parser()

        # 1. Missing threads= prefix is rejected
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_bench(parser.parse_args(["bench", "--sweep", "2,4"]))
        self.assertEqual(rc, 1)
        self.assertIn("Invalid sweep format", captured_err.getvalue())

        # 2. Sweep with faked server + benchmark restores the original threads config
        cfg = gx.load_config()
        cfg["threads"] = 3
        gx.save_config(cfg)

        def fake_suite(runs, prompt_tokens, gen_tokens, base_url, config):
            return {
                "runs": runs,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "avg_gen_tok_s": 10.0,
                "min_gen_tok_s": 9.0,
                "max_gen_tok_s": 11.0,
                "avg_ttft_ms": 50.0,
                "avg_prompt_tok_s": 100.0,
                "gen_speeds": [10.0, 10.0],
                "throttle_warning": False,
            }

        captured = io.StringIO()
        with patch("sys.stdout", captured), patch(
            "gx.cmd_server", return_value=0
        ), patch("gx.run_benchmark_suite", fake_suite):
            rc = gx.cmd_bench(parser.parse_args(["bench", "--sweep", "threads=2,4"]))
        self.assertEqual(rc, 0)
        self.assertIn("Sweep Results", captured.getvalue())
        self.assertIn("* (best)", captured.getvalue())
        self.assertEqual(gx.load_config()["threads"], 3)

        # 3. Sweep runs are recorded in bench history
        with open(gx.get_bench_history_file(), "r", encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(e.get("sweep") for e in entries))

    def test_tool_shell_timeout_message(self) -> None:
        """Test the 20s shell timeout returns a clear error instead of hanging."""
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 20)

        with patch("gx.subprocess.run", fake_run):
            res = gx.tool_shell(self.test_dir, "sleep 100", yes=True)
        self.assertIn("Error: Command timed out after 20 seconds.", res)

    def test_pull_416_finalizes_only_complete_part(self) -> None:
        """Test that a 416 finalize verifies the .part size against the remote file."""
        models_dir = os.path.join(self.test_dir, "models")
        cfg = gx.load_config()
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        part_file = os.path.join(models_dir, "complete.gguf.part")
        with open(part_file, "wb") as f:
            f.write(MOCK_FILE_DATA)  # exactly the remote size

        parser = gx.build_parser()
        args = parser.parse_args(["pull", f"{self.mock_server.url}/files/complete.gguf"])
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = gx.cmd_pull(args)
        self.assertEqual(rc, 0)
        self.assertIn("Download complete", captured.getvalue())
        self.assertTrue(os.path.isfile(os.path.join(models_dir, "complete.gguf")))
        self.assertFalse(os.path.isfile(part_file))

    def test_pull_416_discards_stale_part(self) -> None:
        """Test that a stale .part is discarded instead of being finalized."""
        models_dir = os.path.join(self.test_dir, "models")
        cfg = gx.load_config()
        cfg["models_dir"] = models_dir
        gx.save_config(cfg)

        part_file = os.path.join(models_dir, "stale.gguf.part")
        with open(part_file, "wb") as f:
            f.write(b"Z" * 400)  # larger than the remote 320-byte file

        parser = gx.build_parser()
        args = parser.parse_args(["pull", f"{self.mock_server.url}/files/stale.gguf"])
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err), patch("sys.stdout", io.StringIO()):
            rc = gx.cmd_pull(args)
        self.assertEqual(rc, 1)
        self.assertIn("discarded", captured_err.getvalue())
        self.assertFalse(os.path.isfile(part_file))
        self.assertFalse(os.path.isfile(os.path.join(models_dir, "stale.gguf")))

    def test_hf_token_scoping(self) -> None:
        """Test HF_TOKEN is attached only to Hugging Face URLs."""
        self.assertTrue(gx._is_huggingface_url("https://huggingface.co/u/m/resolve/main/x.gguf"))
        self.assertTrue(gx._is_huggingface_url("https://cdn-lfs.hf.co/x"))
        self.assertFalse(gx._is_huggingface_url(f"{self.mock_server.url}/files/x.gguf"))
        self.assertFalse(gx._is_huggingface_url("https://evil.example.com/x.gguf"))

        old_token = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "secret-token"
        try:
            seen = {}

            def fake_urlopen(req, timeout=None):
                seen["authorization"] = req.headers.get("Authorization")
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

            parser = gx.build_parser()

            # Non-HF host: token must NOT be sent
            args = parser.parse_args(["pull", f"{self.mock_server.url}/files/nope.gguf"])
            with patch("sys.stderr", io.StringIO()):
                with patch("urllib.request.urlopen", fake_urlopen):
                    rc = gx.cmd_pull(args)
            self.assertEqual(rc, 1)
            self.assertIsNone(seen.get("authorization"))

            # HF host: token IS sent
            args_hf = parser.parse_args(["pull", "https://huggingface.co/u/m/resolve/main/x.gguf"])
            with patch("sys.stderr", io.StringIO()):
                with patch("urllib.request.urlopen", fake_urlopen):
                    gx.cmd_pull(args_hf)
            self.assertEqual(seen.get("authorization"), "Bearer secret-token")
        finally:
            if old_token is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = old_token

    def test_pull_out_rejects_path_separators(self) -> None:
        """Test --out with traversal characters is rejected before any download."""
        parser = gx.build_parser()
        args = parser.parse_args(["pull", f"{self.mock_server.url}/files/x.gguf", "--out", "../evil.gguf"])
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            rc = gx.cmd_pull(args)
        self.assertEqual(rc, 1)
        self.assertIn("Invalid output filename", captured_err.getvalue())

    def test_search_files_finds_inside_and_single_file(self) -> None:
        """Test search_files finds real files and supports a single-file path argument."""
        sandbox = os.path.join(self.test_dir, "sandbox_search")
        os.makedirs(sandbox, exist_ok=True)
        with open(os.path.join(sandbox, "notes.txt"), "w") as f:
            f.write("the needle is here\n")

        res = gx.tool_search_files(sandbox, "needle", ".")
        self.assertIn("notes.txt:1:", res)

        res_file = gx.tool_search_files(sandbox, "needle", "notes.txt")
        self.assertIn("needle", res_file)

    @unittest.skipUnless(_can_create_symlinks(), "symlink creation requires privileges on this platform")
    def test_search_files_skips_escaping_file_symlinks(self) -> None:
        """Test a file symlink inside the sandbox pointing outside is not read."""
        sandbox = os.path.join(self.test_dir, "sandbox_search_link")
        os.makedirs(sandbox, exist_ok=True)
        outside_dir = tempfile.mkdtemp(prefix="outside_secret_")
        secret_file = os.path.join(outside_dir, "secret.txt")
        with open(secret_file, "w") as f:
            f.write("SECRET_NEEDLE_DO_NOT_LEAK\n")
        try:
            os.symlink(secret_file, os.path.join(sandbox, "leak.txt"))
            res = gx.tool_search_files(sandbox, "SECRET_NEEDLE_DO_NOT_LEAK", ".")
            self.assertNotIn("SECRET_NEEDLE_DO_NOT_LEAK", res)
            self.assertIn("Skipped 1 path(s) resolving outside the sandbox", res)
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_stream_flag_sent_to_server(self) -> None:
        """Test chat requests are sent with stream=True."""
        parser = gx.build_parser()
        args = parser.parse_args(["ask", "check the stream flag"])
        with patch("sys.stdout", io.StringIO()):
            gx.cmd_ask(args)
        data = MockLlamaHandler.last_request_data or {}
        self.assertTrue(data.get("stream"))

    def test_corrupt_config_warns_and_uses_defaults(self) -> None:
        """Test a corrupted config.json warns and falls back to defaults."""
        cfg_file = gx.get_config_file()
        with open(cfg_file, "w") as f:
            f.write("{corrupted json!!")
        captured_err = io.StringIO()
        with patch("sys.stderr", captured_err):
            cfg = gx.load_config()
        self.assertIn("Warning", captured_err.getvalue())
        self.assertEqual(cfg["temp"], 0.7)

    def test_set_agent_shell_bool_parsing(self) -> None:
        """Test 'gx set agent_shell' stores real booleans."""
        parser = gx.build_parser()
        with patch("sys.stdout", io.StringIO()):
            rc = gx.cmd_set(parser.parse_args(["set", "agent_shell", "false"]))
        self.assertEqual(rc, 0)
        self.assertIs(gx.load_config()["agent_shell"], False)

        with patch("sys.stdout", io.StringIO()):
            rc = gx.cmd_set(parser.parse_args(["set", "agent_shell", "yes"]))
        self.assertEqual(rc, 0)
        self.assertIs(gx.load_config()["agent_shell"], True)


if __name__ == "__main__":
    unittest.main()

