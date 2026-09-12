# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path

from eval_harness.executors.output_protocol import (
    OutputProtocol,
    OutputProtocolError,
    ParsedCursorStatus,
    ParsedOutput,
    ProtocolErrorCode,
    parse_claude_output,
    parse_codex_output,
    parse_cursor_output,
    parse_cursor_status,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "executor_protocol"


class OutputProtocolTests(unittest.TestCase):
    def assert_code(self, expected: ProtocolErrorCode, callback: object, *args: object) -> None:
        if not callable(callback):
            self.fail("callback must be callable")
        with self.assertRaises(OutputProtocolError) as raised:
            callback(*args)
        self.assertIs(raised.exception.code, expected)

    def test_codex_uses_authoritative_final_message_and_accepts_additive_fields(self) -> None:
        stdout = (FIXTURE_ROOT / "codex-success.jsonl").read_text(encoding="utf-8")
        parsed = parse_codex_output(stdout, "authoritative final text")
        self.assertEqual(parsed, ParsedOutput(OutputProtocol.CODEX, "authoritative final text"))

    def test_codex_allows_empty_authoritative_final_message(self) -> None:
        stdout = (FIXTURE_ROOT / "codex-success.jsonl").read_bytes()
        self.assertEqual(parse_codex_output(stdout, b"").output_text, "")

    def test_codex_requires_final_message_but_does_not_expose_payload(self) -> None:
        stdout = (FIXTURE_ROOT / "codex-success.jsonl").read_text(encoding="utf-8")
        self.assert_code(ProtocolErrorCode.MISSING_FINAL_MESSAGE, parse_codex_output, stdout, None)
        with self.assertRaises(OutputProtocolError) as raised:
            parse_codex_output(stdout, b"\xff")
        self.assertIs(raised.exception.code, ProtocolErrorCode.INVALID_FINAL_MESSAGE)
        self.assertNotIn("\\xff", str(raised.exception))

    def test_codex_accepts_documented_item_events_and_rejects_terminal_error(self) -> None:
        stdout = (FIXTURE_ROOT / "codex-error.jsonl").read_text(encoding="utf-8")
        self.assert_code(ProtocolErrorCode.TERMINAL_FAILURE, parse_codex_output, stdout, "answer")

    def test_codex_validates_all_documented_item_variants(self) -> None:
        stdout = (FIXTURE_ROOT / "codex-all-items.jsonl").read_text(encoding="utf-8")
        self.assertEqual(parse_codex_output(stdout, "answer").output_text, "answer")

        prefix = [
            '{"type":"thread.started","thread_id":"thread"}',
            '{"type":"turn.started"}',
        ]
        terminal = (
            '{"type":"turn.completed","usage":{"input_tokens":0,"cached_input_tokens":0,'
            '"output_tokens":0,"reasoning_output_tokens":0}}'
        )
        invalid_items = (
            '{"id":"item","type":"command_execution","command":"true","aggregated_output":"",'
            '"exit_code":true,"status":"completed"}',
            '{"id":"item","type":"file_change","changes":[],"status":"future"}',
            '{"id":"item","type":"mcp_tool_call","server":"s","tool":"t",'
            '"result":{"content":{}},"status":"completed"}',
            '{"id":"item","type":"collab_tool_call","tool":"future","sender_thread_id":"s",'
            '"receiver_thread_ids":[],"agents_states":{},"status":"completed"}',
            '{"id":"item","type":"web_search","query":"q","action":{"type":"future"}}',
            '{"id":"item","type":"todo_list","items":[{"text":"x","completed":"yes"}]}',
            '{"id":"item","type":"agent_message","text":1}',
        )
        for item in invalid_items:
            with self.subTest(item=item):
                self.assert_code(
                    ProtocolErrorCode.INVALID_SHAPE,
                    parse_codex_output,
                    "\n".join((*prefix, json.dumps({"type": "item.completed", "item": json.loads(item)}), terminal)),
                    "answer",
                )

    def test_codex_rejects_invalid_json_duplicate_keys_nonfinite_and_trailing_content(self) -> None:
        valid_prefix = '{"type":"thread.started","thread_id":"thread"}'
        self.assert_code(ProtocolErrorCode.DUPLICATE_KEY, parse_codex_output, '{"type":"x","type":"y"}', "answer")
        self.assert_code(ProtocolErrorCode.INVALID_JSON, parse_codex_output, "NaN", "answer")
        self.assert_code(ProtocolErrorCode.INVALID_JSON, parse_codex_output, '{"value":1e999}', "answer")
        self.assert_code(ProtocolErrorCode.INVALID_JSON, parse_codex_output, valid_prefix + " trailing", "answer")
        self.assert_code(ProtocolErrorCode.EMPTY_OUTPUT, parse_codex_output, "   \n", "answer")
        self.assert_code(ProtocolErrorCode.INVALID_JSON, parse_codex_output, 42, "answer")

    def test_codex_rejects_bad_shapes_and_sequences(self) -> None:
        valid_thread = '{"type":"thread.started","thread_id":"thread"}'
        valid_turn = '{"type":"turn.started"}'
        item = '{"type":"item.completed","item":{"id":"item","type":"agent_message","text":"text"}}'
        terminal = (
            '{"type":"turn.completed","usage":{"input_tokens":0,"cached_input_tokens":0,'
            '"output_tokens":0,"reasoning_output_tokens":0}}'
        )
        valid = "\n".join((valid_thread, valid_turn, item, terminal))
        self.assertEqual(parse_codex_output(valid, "answer").output_text, "answer")
        self.assert_code(ProtocolErrorCode.INVALID_SEQUENCE, parse_codex_output, valid_turn, "answer")
        self.assert_code(
            ProtocolErrorCode.INVALID_SEQUENCE, parse_codex_output, "\n".join((valid_thread, valid_turn)), "answer"
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SEQUENCE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, terminal, terminal)),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, '{"type":"item.completed"}', terminal)),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_JSON,
            parse_codex_output,
            "\n".join((valid_thread, "", terminal)),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join(("{}", valid_turn, terminal)),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SEQUENCE,
            parse_codex_output,
            "\n".join((valid_thread, terminal)),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SEQUENCE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, '{"type":"unexpected"}', terminal)),
            "answer",
        )
        failed_terminal = '{"type":"turn.failed","error":{"message":"failure"}}'
        self.assert_code(
            ProtocolErrorCode.TERMINAL_FAILURE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, failed_terminal)),
            "answer",
        )
        additive_terminal = (
            '{"type":"turn.completed","usage":{"input_tokens":0,"cached_input_tokens":0,'
            '"output_tokens":0,"reasoning_output_tokens":0},"future_field":true}'
        )
        self.assertEqual(
            parse_codex_output("\n".join((valid_thread, valid_turn, additive_terminal)), "answer").output_text,
            "answer",
        )
        missing_usage = '{"type":"turn.completed"}'
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, missing_usage)),
            "answer",
        )
        bad_agent = '{"type":"item.completed","item":{"id":"item","type":"agent_message"}}'
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, bad_agent, terminal)),
            "answer",
        )
        bad_usage = (
            '{"type":"turn.completed","usage":{"input_tokens":0,"cached_input_tokens":0,'
            '"output_tokens":0,"reasoning_output_tokens":0.5}}'
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, bad_usage)),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, '{"type":"turn.completed","usage":[]}')),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join(
                (
                    valid_thread,
                    valid_turn,
                    '{"type":"item.completed","item":{"id":"item","type":"file_change","status":"completed"}}',
                    terminal,
                )
            ),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join(
                (
                    valid_thread,
                    valid_turn,
                    '{"type":"item.completed","item":{"id":"item","type":"collab_tool_call",'
                    '"tool":"wait","sender_thread_id":"sender","receiver_thread_ids":[1],'
                    '"agents_states":{},"status":"completed"}}',
                    terminal,
                )
            ),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join(
                (
                    valid_thread,
                    valid_turn,
                    '{"type":"item.completed","item":{"id":"item","type":"collab_tool_call",'
                    '"tool":"wait","sender_thread_id":"sender","receiver_thread_ids":[],'
                    '"prompt":1,"agents_states":{},"status":"completed"}}',
                    terminal,
                )
            ),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            "\n".join(
                (
                    valid_thread,
                    valid_turn,
                    '{"type":"item.completed","item":{"id":"search","type":"web_search",'
                    '"query":"q","action":{"type":"search","queries":[1]}}}',
                    terminal,
                )
            ),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_FINAL_MESSAGE,
            parse_codex_output,
            valid,
            object(),
        )
        self.assert_code(
            ProtocolErrorCode.TERMINAL_FAILURE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, '{"type":"error","message":"failure"}')),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SEQUENCE,
            parse_codex_output,
            "\n".join((valid_thread, valid_turn, '{"type":"turn.started"}', terminal)),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SHAPE,
            parse_codex_output,
            (FIXTURE_ROOT / "codex-unknown-item.jsonl").read_text(encoding="utf-8"),
            "answer",
        )
        self.assert_code(
            ProtocolErrorCode.INVALID_SEQUENCE,
            parse_codex_output,
            (FIXTURE_ROOT / "codex-unknown-event.jsonl").read_text(encoding="utf-8"),
            "answer",
        )

    def test_claude_success_requires_terminal_fields_and_preserves_empty_result(self) -> None:
        stdout = (FIXTURE_ROOT / "claude-success.json").read_text(encoding="utf-8")
        parsed = parse_claude_output(stdout)
        self.assertIs(parsed.protocol, OutputProtocol.CLAUDE)
        self.assertEqual(parsed.output_text, "final answer")

        empty = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "num_turns": 0,
            "result": "",
            "session_id": "session",
        }
        self.assertEqual(parse_claude_output(json.dumps(empty)).output_text, "")

    def test_claude_rejects_errors_and_invalid_known_fields(self) -> None:
        success: dict[str, object] = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 1,
            "duration_api_ms": 1,
            "num_turns": 1,
            "result": "answer",
            "session_id": "session",
        }
        invalid_terminal_fields: tuple[tuple[str, object], ...] = (("is_error", True), ("error", "failure"))
        for field, value in invalid_terminal_fields:
            payload: dict[str, object] = dict(success)
            payload[field] = value
            self.assert_code(ProtocolErrorCode.TERMINAL_FAILURE, parse_claude_output, json.dumps(payload))
        unknown_error_subtype = dict(success)
        unknown_error_subtype["subtype"] = "error"
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(unknown_error_subtype))
        unknown_subtype = dict(success)
        unknown_subtype["subtype"] = "future_result"
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(unknown_subtype))
        missing_subtype = dict(success)
        del missing_subtype["subtype"]
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(missing_subtype))
        invalid_numeric_fields: tuple[tuple[str, object], ...] = (
            ("duration_ms", True),
            ("duration_api_ms", -1),
            ("num_turns", 1.5),
        )
        for field, bad_value in invalid_numeric_fields:
            numeric_payload: dict[str, object] = dict(success)
            numeric_payload[field] = bad_value
            self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(numeric_payload))
        missing = dict(success)
        del missing["result"]
        self.assert_code(ProtocolErrorCode.MISSING_FINAL_MESSAGE, parse_claude_output, json.dumps(missing))
        wrong_result = dict(success)
        wrong_result["result"] = {"text": "answer"}
        self.assert_code(ProtocolErrorCode.INVALID_FINAL_MESSAGE, parse_claude_output, json.dumps(wrong_result))
        for field, value in (
            ("api_error_status", 401),
            ("terminal_reason", "api_error"),
            ("errors", ["redacted"]),
        ):
            terminal_payload = dict(success)
            terminal_payload[field] = value
            self.assert_code(ProtocolErrorCode.TERMINAL_FAILURE, parse_claude_output, json.dumps(terminal_payload))
        documented_error_subtype = dict(success)
        documented_error_subtype["subtype"] = "error_max_turns"
        self.assert_code(ProtocolErrorCode.TERMINAL_FAILURE, parse_claude_output, json.dumps(documented_error_subtype))
        self.assert_code(
            ProtocolErrorCode.TERMINAL_FAILURE,
            parse_claude_output,
            (FIXTURE_ROOT / "claude-api-error.json").read_text(encoding="utf-8"),
        )
        invalid_terminal = dict(success)
        invalid_terminal["api_error_status"] = "401"
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(invalid_terminal))
        invalid_reason_type = dict(success)
        invalid_reason_type["terminal_reason"] = 1
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(invalid_reason_type))
        unknown_reason = dict(success)
        unknown_reason["terminal_reason"] = "future_reason"
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(unknown_reason))
        invalid_errors = dict(success)
        invalid_errors["errors"] = [{}]
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(invalid_errors))
        invalid_errors_type = dict(success)
        invalid_errors_type["errors"] = {}
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_claude_output, json.dumps(invalid_errors_type))

    def test_cursor_success_accepts_optional_request_id_and_additive_fields(self) -> None:
        stdout = (FIXTURE_ROOT / "cursor-success.json").read_bytes()
        parsed = parse_cursor_output(stdout)
        self.assertIs(parsed.protocol, OutputProtocol.CURSOR)
        self.assertEqual(parsed.output_text, "final answer")

        payload = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "result": "",
            "session_id": "session",
            "unknown": ["accepted"],
        }
        self.assertEqual(parse_cursor_output(json.dumps(payload) + "\n").output_text, "")

    def test_cursor_rejects_terminal_and_malformed_outputs(self) -> None:
        success: dict[str, object] = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 1,
            "duration_api_ms": 1,
            "result": "answer",
            "session_id": "session",
        }
        payload = dict(success)
        payload["type"] = "error"
        self.assert_code(ProtocolErrorCode.TERMINAL_FAILURE, parse_cursor_output, json.dumps(payload))
        payload = dict(success)
        payload["subtype"] = "future_result"
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_output, json.dumps(payload))
        payload = dict(success)
        payload["request_id"] = None
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_output, json.dumps(payload))
        for field, value in (
            ("api_error_status", 401),
            ("terminal_reason", "future_additive_reason"),
            ("errors", ["future additive detail"]),
            ("error", "future additive detail"),
        ):
            additive_payload = dict(success)
            additive_payload[field] = value
            self.assertEqual(parse_cursor_output(json.dumps(additive_payload)).output_text, "answer")
        self.assert_code(ProtocolErrorCode.DUPLICATE_KEY, parse_cursor_output, '{"type":"result","type":"result"}')
        self.assert_code(ProtocolErrorCode.INVALID_JSON, parse_cursor_output, json.dumps(success) + " trailing")
        self.assert_code(ProtocolErrorCode.EMPTY_OUTPUT, parse_cursor_output, "\n\t")
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_output, "[]")
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_output, json.dumps({"type": "other"}))
        self.assert_code(ProtocolErrorCode.INVALID_JSON, parse_cursor_output, b"\xff")
        invalid_bool = dict(success)
        invalid_bool["is_error"] = "false"
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_output, json.dumps(invalid_bool))

    def test_cursor_status_requires_positive_documented_account_shape(self) -> None:
        authenticated = json.loads(
            (FIXTURE_ROOT / "cursor-2026.09.10-fd3934a-status-authenticated.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            parse_cursor_status(json.dumps(authenticated)),
            ParsedCursorStatus(status="authenticated", is_authenticated=True),
        )
        for name, expected in (
            ("cursor-2026.09.10-fd3934a-status-partial.json", False),
            ("cursor-2026.09.10-fd3934a-status-unauthenticated.json", False),
            ("cursor-2026.09.10-fd3934a-status-error.json", False),
        ):
            with self.subTest(name=name):
                parsed = parse_cursor_status((FIXTURE_ROOT / name).read_bytes())
                self.assertEqual(parsed.is_authenticated, expected)
        token_only = (FIXTURE_ROOT / "cursor-2026.09.10-fd3934a-status-token-only.json").read_text(encoding="utf-8")
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, token_only)
        invalid_flags = dict(authenticated)
        invalid_flags["isAuthenticated"] = False
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, json.dumps(invalid_flags))
        invalid_partial = json.loads(
            (FIXTURE_ROOT / "cursor-2026.09.10-fd3934a-status-partial.json").read_text(encoding="utf-8")
        )
        invalid_partial["hasRefreshToken"] = True
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, json.dumps(invalid_partial))
        invalid_unauthenticated = json.loads(
            (FIXTURE_ROOT / "cursor-2026.09.10-fd3934a-status-unauthenticated.json").read_text(encoding="utf-8")
        )
        invalid_unauthenticated["hasAccessToken"] = True
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, json.dumps(invalid_unauthenticated))
        invalid_identity = dict(authenticated)
        invalid_identity["userInfo"] = {"email": "", "userId": 0}
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, json.dumps(invalid_identity))
        invalid_identity_type = dict(authenticated)
        invalid_identity_type["userInfo"] = {"email": 1}
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, json.dumps(invalid_identity_type))
        invalid_user_id_type = dict(authenticated)
        invalid_user_id_type["userInfo"] = {"userId": []}
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, json.dumps(invalid_user_id_type))
        unknown = dict(authenticated)
        unknown["status"] = "future-status"
        self.assert_code(ProtocolErrorCode.INVALID_SHAPE, parse_cursor_status, json.dumps(unknown))
        duplicate = '{"status":"unauthenticated","status":"error"}'
        self.assert_code(ProtocolErrorCode.DUPLICATE_KEY, parse_cursor_status, duplicate)
        with self.assertRaises(OutputProtocolError) as raised:
            parse_cursor_status(
                json.dumps(
                    {
                        "status": "authenticated",
                        "isAuthenticated": True,
                        "hasAccessToken": True,
                        "hasRefreshToken": True,
                        "message": "secret-token-value",
                    }
                )
            )
        self.assertNotIn("secret-token-value", str(raised.exception))

    def test_parsed_output_rejects_invalid_constructor_values(self) -> None:
        with self.assertRaises(ValueError):
            ParsedOutput("unknown", "answer")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ParsedOutput(OutputProtocol.CODEX, object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
