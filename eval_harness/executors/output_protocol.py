# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict parsers for supported executor terminal output.

Executor processes expose different machine-readable formats.  This module
normalizes only their documented successful terminal forms and fails closed on
truncated, malformed, or terminal-error output.  It deliberately does not
interpret benchmark names or infer a score from a malformed response.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class OutputProtocol(StrEnum):
    """The vendor output format accepted by a parser."""

    CODEX = "codex"
    CLAUDE = "claude"
    CURSOR = "cursor"


class ProtocolErrorCode(StrEnum):
    """Stable parser failure codes suitable for run metadata."""

    EMPTY_OUTPUT = "empty_output"
    INVALID_JSON = "invalid_json"
    DUPLICATE_KEY = "duplicate_key"
    INVALID_SHAPE = "invalid_shape"
    INVALID_SEQUENCE = "invalid_sequence"
    TERMINAL_FAILURE = "terminal_failure"
    MISSING_FINAL_MESSAGE = "missing_final_message"
    INVALID_FINAL_MESSAGE = "invalid_final_message"


class OutputProtocolError(ValueError):
    """A parser error whose message contains no vendor payload."""

    code: ProtocolErrorCode

    def __init__(self, code: ProtocolErrorCode, message: str | None = None) -> None:
        self.code = ProtocolErrorCode(code)
        # Callers may persist ``code``.  Any detail supplied by this module is
        # static and never includes a decoded vendor payload.
        super().__init__(message or self.code.value)


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    """Normalized final text together with the protocol that produced it."""

    protocol: OutputProtocol
    output_text: str

    def __post_init__(self) -> None:
        try:
            protocol = OutputProtocol(self.protocol)
        except (TypeError, ValueError) as exc:
            raise ValueError("parsed output protocol is unsupported") from exc
        if not isinstance(self.output_text, str):
            raise TypeError("parsed output text must be a string")
        object.__setattr__(self, "protocol", protocol)


class _DuplicateKey(ValueError):
    pass


class _NonFiniteNumber(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _reject_nonfinite_number(_value: str) -> Any:
    raise _NonFiniteNumber


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteNumber
    return parsed


def _decode_json_input(value: str | bytes | bytearray) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OutputProtocolError(ProtocolErrorCode.INVALID_JSON, "output is not valid UTF-8 JSON") from exc
    raise OutputProtocolError(ProtocolErrorCode.INVALID_JSON, "output must be text or UTF-8 bytes")


def _parse_json_object(value: str | bytes | bytearray, *, empty_code: ProtocolErrorCode) -> dict[str, object]:
    text = _decode_json_input(value)
    if not text.strip():
        raise OutputProtocolError(empty_code, "output is empty")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
            parse_float=_parse_finite_float,
        )
    except _DuplicateKey as exc:
        raise OutputProtocolError(ProtocolErrorCode.DUPLICATE_KEY, "output contains a duplicate JSON key") from exc
    except (json.JSONDecodeError, _NonFiniteNumber, RecursionError, TypeError, ValueError) as exc:
        raise OutputProtocolError(ProtocolErrorCode.INVALID_JSON, "output is not one valid JSON value") from exc
    if not isinstance(parsed, dict):
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "output JSON value must be an object")
    return parsed


def parse_strict_json_object(
    value: str | bytes | bytearray,
    *,
    empty_code: ProtocolErrorCode = ProtocolErrorCode.EMPTY_OUTPUT,
) -> dict[str, object]:
    """Decode one strict JSON object for an executor status response.

    The helper deliberately returns only an object and shares the duplicate
    key, non-finite number, UTF-8, and trailing-content checks used by the
    execution output parsers.  It is public so authentication preflight does
    not grow a second, less restrictive JSON decoder.
    """

    return _parse_json_object(value, empty_code=empty_code)


def _parse_json_lines(value: str | bytes | bytearray) -> list[dict[str, object]]:
    text = _decode_json_input(value)
    if not text.strip():
        raise OutputProtocolError(ProtocolErrorCode.EMPTY_OUTPUT, "output is empty")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise OutputProtocolError(ProtocolErrorCode.EMPTY_OUTPUT, "output is empty")
    parsed: list[dict[str, object]] = []
    for line in lines:
        if not line.strip():
            raise OutputProtocolError(ProtocolErrorCode.INVALID_JSON, "JSONL output contains an empty line")
        parsed.append(_parse_json_object(line, empty_code=ProtocolErrorCode.INVALID_JSON))
    return parsed


def _require_text(
    payload: Mapping[str, object],
    key: str,
    *,
    missing_code: ProtocolErrorCode = ProtocolErrorCode.INVALID_SHAPE,
    invalid_code: ProtocolErrorCode = ProtocolErrorCode.INVALID_SHAPE,
    nonempty: bool = False,
) -> str:
    if key not in payload:
        raise OutputProtocolError(missing_code, "output is missing a required text field")
    value = payload[key]
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise OutputProtocolError(invalid_code, "output contains an invalid text field")
    return value


def _require_nonnegative_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    # ``bool`` is an ``int`` subclass but is not a valid duration/count.
    if type(value) is not int or value < 0:
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "output contains an invalid numeric field")
    return value


def _require_false(payload: Mapping[str, object], key: str) -> None:
    if type(payload.get(key)) is not bool:
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "output contains an invalid boolean field")
    if payload[key] is not False:
        raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "executor reported a terminal failure")


_CODEX_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
        "turn.failed",
        "error",
    }
)
_CODEX_ITEM_TYPES = frozenset(
    {
        "agent_message",
        "reasoning",
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "collab_tool_call",
        "web_search",
        "todo_list",
        "error",
    }
)
_CODEX_COMMAND_STATUSES = frozenset({"in_progress", "completed", "failed", "declined"})
_CODEX_PATCH_STATUSES = frozenset({"in_progress", "completed", "failed"})
_CODEX_PATCH_KINDS = frozenset({"add", "delete", "update"})
_CODEX_MCP_STATUSES = frozenset({"in_progress", "completed", "failed"})
_CODEX_COLLAB_TOOLS = frozenset({"spawn_agent", "send_input", "wait", "close_agent"})
_CODEX_COLLAB_STATUSES = frozenset({"in_progress", "completed", "failed"})
_CODEX_COLLAB_AGENT_STATUSES = frozenset(
    {"pending_init", "running", "interrupted", "completed", "errored", "shutdown", "not_found"}
)
_CODEX_WEB_SEARCH_ACTIONS = frozenset({"search", "open_page", "find_in_page", "other"})
_CLAUDE_SUBTYPES = frozenset(
    {
        "success",
        "error_during_execution",
        "error_max_turns",
        "error_max_budget_usd",
        "error_max_structured_output_retries",
    }
)
_CURSOR_SUBTYPES = frozenset({"success"})
_CLAUDE_TERMINAL_REASONS = frozenset({"completed", "max_turns", "api_error", "aborted_streaming", "aborted_tools"})


def _validate_codex_event(event: Mapping[str, object]) -> str:
    def require_object(value: object, message: str) -> Mapping[str, object]:
        if not isinstance(value, dict):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, message)
        return value

    def require_array(value: object, message: str) -> list[object]:
        if not isinstance(value, list):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, message)
        return value

    def require_object_field(payload: Mapping[str, object], key: str, message: str) -> Mapping[str, object]:
        if key not in payload:
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, message)
        return require_object(payload[key], message)

    def require_array_field(payload: Mapping[str, object], key: str, message: str) -> list[object]:
        if key not in payload:
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, message)
        return require_array(payload[key], message)

    def require_enum(payload: Mapping[str, object], key: str, allowed: frozenset[str]) -> str:
        value = _require_text(payload, key, nonempty=True)
        if value not in allowed:
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "output contains an unknown enum variant")
        return value

    def require_optional_text(payload: Mapping[str, object], key: str) -> None:
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            raise OutputProtocolError(
                ProtocolErrorCode.INVALID_SHAPE, "output contains an invalid optional text field"
            )

    def require_string_array(value: object, message: str) -> None:
        for item in require_array(value, message):
            if not isinstance(item, str):
                raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, message)

    def require_usage_count(payload: Mapping[str, object], key: str) -> None:
        value = payload.get(key)
        if type(value) is not int or value < 0 or value > (2**63 - 1):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "output contains an invalid usage field")

    def require_optional_i32(payload: Mapping[str, object], key: str) -> None:
        if key not in payload or payload[key] is None:
            return
        value = payload[key]
        if type(value) is not int or not -(2**31) <= value <= (2**31 - 1):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "output contains an invalid exit code")

    def validate_web_search_action(action: Mapping[str, object]) -> None:
        action_type = _require_text(action, "type", nonempty=True)
        if action_type not in _CODEX_WEB_SEARCH_ACTIONS:
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "Codex output has an unknown web action")
        if action_type == "search":
            require_optional_text(action, "query")
            if "queries" in action and action["queries"] is not None:
                require_string_array(action["queries"], "web search queries must be an array of text")
        elif action_type == "open_page":
            require_optional_text(action, "url")
        elif action_type == "find_in_page":
            require_optional_text(action, "pattern")
            require_optional_text(action, "url")

    def validate_collab_state(state: object) -> None:
        state_payload = require_object(state, "collab agent state must be an object")
        require_enum(state_payload, "status", _CODEX_COLLAB_AGENT_STATUSES)
        require_optional_text(state_payload, "message")

    def validate_item(item: Mapping[str, object]) -> None:
        _require_text(item, "id", nonempty=True)
        item_type = _require_text(item, "type", nonempty=True)
        if item_type not in _CODEX_ITEM_TYPES:
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "Codex output has an unknown item variant")
        if item_type in {"agent_message", "reasoning"}:
            _require_text(item, "text")
        elif item_type == "command_execution":
            _require_text(item, "command")
            _require_text(item, "aggregated_output")
            require_optional_i32(item, "exit_code")
            require_enum(item, "status", _CODEX_COMMAND_STATUSES)
        elif item_type == "file_change":
            changes = require_array_field(item, "changes", "file change must contain a changes array")
            for change in changes:
                change_payload = require_object(change, "file change entry must be an object")
                _require_text(change_payload, "path")
                require_enum(change_payload, "kind", _CODEX_PATCH_KINDS)
            require_enum(item, "status", _CODEX_PATCH_STATUSES)
        elif item_type == "mcp_tool_call":
            _require_text(item, "server")
            _require_text(item, "tool")
            if "result" in item and item["result"] is not None:
                result = require_object(item["result"], "MCP result must be an object")
                require_array_field(result, "content", "MCP result must contain a content array")
            if "error" in item and item["error"] is not None:
                error = require_object(item["error"], "MCP error must be an object")
                _require_text(error, "message")
            require_enum(item, "status", _CODEX_MCP_STATUSES)
        elif item_type == "collab_tool_call":
            require_enum(item, "tool", _CODEX_COLLAB_TOOLS)
            _require_text(item, "sender_thread_id")
            require_string_array(
                item.get("receiver_thread_ids"),
                "collab receiver thread ids must be an array of text",
            )
            require_optional_text(item, "prompt")
            states = require_object_field(item, "agents_states", "collab tool call must contain agent states")
            for state in states.values():
                validate_collab_state(state)
            require_enum(item, "status", _CODEX_COLLAB_STATUSES)
        elif item_type == "web_search":
            _require_text(item, "id")
            _require_text(item, "query")
            validate_web_search_action(require_object_field(item, "action", "web search must contain an action"))
        elif item_type == "todo_list":
            todo_items = require_array_field(item, "items", "todo list must contain an items array")
            for todo in todo_items:
                todo_payload = require_object(todo, "todo item must be an object")
                _require_text(todo_payload, "text")
                if type(todo_payload.get("completed")) is not bool:
                    raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "todo item completion must be boolean")
        elif item_type == "error":
            _require_text(item, "message")

    event_type = event.get("type")
    if not isinstance(event_type, str):
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "JSONL event type must be text")
    if event_type not in _CODEX_EVENT_TYPES:
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SEQUENCE, "Codex output has an unknown event type")
    if event_type == "thread.started":
        _require_text(event, "thread_id", nonempty=True)
    elif event_type in {"item.started", "item.updated", "item.completed"}:
        item = event.get("item")
        if not isinstance(item, dict):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "JSONL item event must contain an object item")
        validate_item(item)
    elif event_type == "turn.completed":
        usage = require_object_field(event, "usage", "turn.completed must contain a usage object")
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
            require_usage_count(usage, key)
        if "cache_write_input_tokens" in usage:
            require_usage_count(usage, "cache_write_input_tokens")
    elif event_type == "turn.failed":
        error = require_object_field(event, "error", "turn.failed must contain an error object")
        _require_text(error, "message")
    elif event_type == "error":
        _require_text(event, "message")
    return event_type


def _validate_codex_sequence(events: list[dict[str, object]]) -> None:
    if not events:
        raise OutputProtocolError(ProtocolErrorCode.EMPTY_OUTPUT, "JSONL output is empty")
    terminal_indexes: list[int] = []
    for index, event in enumerate(events):
        event_type = _validate_codex_event(event)
        if event_type in {"error", "turn.failed"}:
            raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "Codex reported a terminal failure")
        if index == 0:
            if event_type != "thread.started":
                raise OutputProtocolError(
                    ProtocolErrorCode.INVALID_SEQUENCE, "Codex output must start with thread.started"
                )
            continue
        if index == 1:
            if event_type != "turn.started":
                raise OutputProtocolError(ProtocolErrorCode.INVALID_SEQUENCE, "Codex output must contain turn.started")
            continue
        if event_type == "turn.completed":
            terminal_indexes.append(index)
            continue
        if event_type in {"item.started", "item.updated", "item.completed"}:
            continue
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SEQUENCE, "Codex output has an invalid event sequence")

    if terminal_indexes != [len(events) - 1]:
        if terminal_indexes:
            raise OutputProtocolError(
                ProtocolErrorCode.INVALID_SEQUENCE, "Codex output has a duplicate or non-final terminal event"
            )
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SEQUENCE, "Codex output is missing turn.completed")


def _normalize_final_message(value: str | bytes | bytearray | None) -> str:
    if value is None:
        raise OutputProtocolError(ProtocolErrorCode.MISSING_FINAL_MESSAGE, "Codex final message is missing")
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OutputProtocolError(
                ProtocolErrorCode.INVALID_FINAL_MESSAGE, "Codex final message is not valid UTF-8"
            ) from exc
    raise OutputProtocolError(ProtocolErrorCode.INVALID_FINAL_MESSAGE, "Codex final message must be text")


def parse_codex_output(
    stdout: str | bytes | bytearray,
    final_message: str | bytes | bytearray | None,
) -> ParsedOutput:
    """Parse Codex JSONL and the authoritative ``--output-last-message`` text."""

    _validate_codex_sequence(_parse_json_lines(stdout))
    return ParsedOutput(OutputProtocol.CODEX, _normalize_final_message(final_message))


def _parse_terminal_result(
    value: str | bytes | bytearray,
    protocol: OutputProtocol,
) -> ParsedOutput:
    payload = _parse_json_object(value, empty_code=ProtocolErrorCode.EMPTY_OUTPUT)
    payload_type = payload.get("type")
    if payload_type == "error":
        raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "executor reported a terminal error")
    if payload_type != "result":
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "terminal output type must be result")

    subtype = payload.get("subtype")
    if not isinstance(subtype, str):
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "terminal output subtype must be text")
    allowed_subtypes = _CLAUDE_SUBTYPES if protocol is OutputProtocol.CLAUDE else _CURSOR_SUBTYPES
    if subtype not in allowed_subtypes:
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "terminal output has an unknown subtype")
    if subtype != "success":
        raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "executor reported an unsuccessful result")
    _require_false(payload, "is_error")
    if protocol is OutputProtocol.CLAUDE and "error" in payload:
        raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "terminal output contains an error field")
    if protocol is OutputProtocol.CLAUDE:
        api_error_status = payload.get("api_error_status")
        if api_error_status is not None:
            if type(api_error_status) is not int:
                raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "api error status must be an integer")
            raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "executor reported an API terminal error")
        terminal_reason = payload.get("terminal_reason")
        if terminal_reason is not None:
            if not isinstance(terminal_reason, str):
                raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "terminal reason must be text")
            if terminal_reason not in _CLAUDE_TERMINAL_REASONS:
                raise OutputProtocolError(
                    ProtocolErrorCode.INVALID_SHAPE, "terminal output has an unknown terminal reason"
                )
            if terminal_reason in {"api_error", "max_turns", "aborted_streaming", "aborted_tools"}:
                raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "executor reported a terminal error")
        errors = payload.get("errors")
        if errors is not None:
            if not isinstance(errors, list):
                raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "terminal errors must be an array")
            if not all(isinstance(error, str) for error in errors):
                raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "terminal errors must contain text")
            if errors:
                raise OutputProtocolError(ProtocolErrorCode.TERMINAL_FAILURE, "executor reported terminal errors")

    result = _require_text(
        payload,
        "result",
        missing_code=ProtocolErrorCode.MISSING_FINAL_MESSAGE,
        invalid_code=ProtocolErrorCode.INVALID_FINAL_MESSAGE,
    )
    _require_text(payload, "session_id", nonempty=True)
    _require_nonnegative_int(payload, "duration_ms")
    _require_nonnegative_int(payload, "duration_api_ms")
    if protocol is OutputProtocol.CLAUDE:
        _require_nonnegative_int(payload, "num_turns")
    elif "request_id" in payload and not isinstance(payload["request_id"], str):
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "request_id must be text when supplied")
    return ParsedOutput(protocol, result)


def parse_claude_output(stdout: str | bytes | bytearray) -> ParsedOutput:
    """Parse Claude Code's single JSON terminal result."""

    return _parse_terminal_result(stdout, OutputProtocol.CLAUDE)


def parse_cursor_output(stdout: str | bytes | bytearray) -> ParsedOutput:
    """Parse Cursor Agent's single JSON terminal result."""

    return _parse_terminal_result(stdout, OutputProtocol.CURSOR)


@dataclass(frozen=True, slots=True)
class ParsedCursorStatus:
    """The small, positive authentication result accepted from Cursor."""

    status: str
    is_authenticated: bool


_CURSOR_STATUS_TYPES = frozenset({"authenticated", "partially-authenticated", "unauthenticated", "error"})


def _require_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "status contains an invalid boolean field")
    return value


def _validate_cursor_identity(user_info: object) -> None:
    if not isinstance(user_info, dict):
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "authenticated status requires user information")
    for key in ("email", "firstName", "lastName", "teamId", "createdAt"):
        if key not in user_info:
            continue
        value = user_info[key]
        if value is not None and not isinstance(value, str):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "status contains invalid user information")
    if "userId" in user_info:
        user_id_value = user_info["userId"]
        if user_id_value is not None and not isinstance(user_id_value, str) and type(user_id_value) is not int:
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "status contains invalid user information")
    email = user_info.get("email")
    user_id = user_info.get("userId")
    positive_email = isinstance(email, str) and bool(email.strip())
    positive_user_id = (isinstance(user_id, str) and bool(user_id.strip())) or (type(user_id) is int and user_id > 0)
    if not (positive_email or positive_user_id):
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "authenticated status lacks user identity")


def parse_cursor_status(stdout: str | bytes | bytearray) -> ParsedCursorStatus:
    """Parse Cursor's documented JSON authentication status response."""

    payload = _parse_json_object(stdout, empty_code=ProtocolErrorCode.EMPTY_OUTPUT)
    status = _require_text(payload, "status", nonempty=True)
    if status not in _CURSOR_STATUS_TYPES:
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "status has an unknown variant")
    if status == "error":
        _require_text(payload, "message", nonempty=True)
        return ParsedCursorStatus(status=status, is_authenticated=False)

    is_authenticated = _require_bool(payload, "isAuthenticated")
    has_access_token = _require_bool(payload, "hasAccessToken")
    has_refresh_token = _require_bool(payload, "hasRefreshToken")
    if status == "authenticated":
        if not (is_authenticated and has_access_token and has_refresh_token):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "authenticated status has inconsistent flags")
        _validate_cursor_identity(payload.get("userInfo"))
        return ParsedCursorStatus(status=status, is_authenticated=True)
    if status == "partially-authenticated":
        if (is_authenticated, has_access_token, has_refresh_token) != (False, True, False):
            raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "partial status has inconsistent flags")
    elif (is_authenticated, has_access_token, has_refresh_token) != (False, False, False):
        raise OutputProtocolError(ProtocolErrorCode.INVALID_SHAPE, "unauthenticated status has inconsistent flags")
    return ParsedCursorStatus(status=status, is_authenticated=False)


__all__ = [
    "OutputProtocol",
    "OutputProtocolError",
    "ParsedCursorStatus",
    "ParsedOutput",
    "ProtocolErrorCode",
    "parse_claude_output",
    "parse_codex_output",
    "parse_cursor_output",
    "parse_cursor_status",
    "parse_strict_json_object",
]
