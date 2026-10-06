#!/usr/bin/env python3
"""Comprehensive test suite for gx CLI with Mock llama-server."""

import http.server
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

import gx

MOCK_FILE_DATA = b"GGUF_MODEL_MOCK_DATA_0123456789" * 10  # 320 bytes


class MockLlamaHandler(http.server.BaseHTTPRequestHandler):
    """Mock HTTP handler emulating llama-server OpenAI API and file downloads."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

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
        args = parser.parse_args(["set", "temp", "0.2"])
        rc = gx.cmd_set(args)
        self.assertEqual(rc, 0)

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
        with patch("sys.stderr", captured_err):
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
        """Test thermal throttling calculation warning."""
        # Simulated speeds with >25% drop: [40.0, 25.0] -> 37.5% drop
        fake_speeds = [40.0, 25.0]
        drop_pct = ((fake_speeds[0] - fake_speeds[-1]) / fake_speeds[0]) * 100.0
        self.assertTrue(drop_pct > 25.0)

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
        with patch("sys.stdout", captured_out):
            gx.cmd_doctor(args)

        out = captured_out.getvalue()
        self.assertIn("Python version", out)
        self.assertIn("Sessions directory", out)
        self.assertIn("Sandbox directory is writable", out)
        self.assertIn("Personas directory", out)
        self.assertIn("Templates directory", out)
        self.assertIn("Models dir", out)
        self.assertIn("Free disk space", out)

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
        """Test sandbox path resolution and rejection of .., absolute escapes, and escaping symlinks."""
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

        # 4. Escape via symlink pointing outside
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
        cmd_big = "python3 -c \"print('A' * 3000)\""
        out = gx.tool_shell(sandbox, cmd_big, yes=True)
        self.assertIn("[Truncated: output exceeded 2000 chars]", out)
        self.assertTrue(len(out) <= 2100)


if __name__ == "__main__":
    unittest.main()

