# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Mapping, TypeAlias
from unittest.mock import patch

import eval_harness.executors.claude_code as claude_executor_module
import eval_harness.executors.codex as codex_executor_module
import eval_harness.executors.cursor as cursor_executor_module
import eval_harness.judges.claude_code as claude_judge_module
import eval_harness.judges.codex as codex_judge_module
from eval_harness.capabilities import ExecutorOutput
from eval_harness.executors.base import ExecutionRequest, ExecutionStatus, TaskSpec
from eval_harness.executors.claude_code import ClaudeCodeExecutor
from eval_harness.executors.codex import CodexExecutor
from eval_harness.executors.cursor import CursorExecutor
from eval_harness.failures import FailureImpact, FailureKind
from eval_harness.judges.base import JudgeRequest, Verdict
from eval_harness.judges.claude_code import ClaudeCodeJudgeExecutor
from eval_harness.judges.codex import (
    CodexJudgeExecutor,
    _normalize_utf8_log,
    _paths_overlap,
    _protected_read_roots,
    _resolved_command_read_paths,
    _resolved_path,
)
from eval_harness.judges.pairwise import (
    _assert_directory,
    _copy_tree,
    aggregate,
    build_judge_prompt,
    matched_tasks,
    normalize_verdict,
    parse_verdict,
    prepare_trial,
    tree_hash,
    validate_reference_equivalence,
    write_trial_metadata,
)


_FAKE_CLI = """
#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path


def emit(stream, value):
    stream.buffer.write(value)
    stream.flush()


args = sys.argv[1:]
if "--version" in args:
    sys.stdout.write(os.environ.get("GDPVAL_FAKE_VERSION", "fake-cli 1.2.3") + "\\n")
    sys.exit(0)
if args[:2] == ["login", "status"]:
    emit(sys.stdout, os.environ.get("GDPVAL_FAKE_CODEX_AUTH", "chatgpt subscription").encode() + b"\\n")
    sys.exit(int(os.environ.get("GDPVAL_FAKE_STATUS_EXIT", "0")))
if args[:2] == ["auth", "status"]:
    payload = os.environ.get(
        "GDPVAL_FAKE_AUTH_JSON",
        '{"loggedIn": true, "authMethod": "claude.ai", '
        '"apiProvider": "firstparty", "subscriptionType": "pro"}',
    )
    emit(sys.stdout, payload.encode() + b"\\n")
    sys.exit(int(os.environ.get("GDPVAL_FAKE_STATUS_EXIT", "0")))
if args[:1] == ["status"]:
    emit(sys.stdout, os.environ.get("GDPVAL_FAKE_CURSOR_STATUS", "logged in account").encode() + b"\\n")
    sys.exit(int(os.environ.get("GDPVAL_FAKE_STATUS_EXIT", "0")))
if "sandbox" in args:
    sys.exit(int(os.environ.get("GDPVAL_FAKE_SANDBOX_EXIT", "0")))

capture = os.environ.get("GDPVAL_CAPTURE_ENV")
if capture:
    Path(capture).write_text(json.dumps(dict(os.environ), sort_keys=True), encoding="utf-8")
mode = os.environ.get("GDPVAL_FAKE_MODE", "success")
if mode == "mutate":
    reference = Path.cwd().parent / "cursor-reference-files-readonly" / "input.txt"
    reference.write_text("mutated\\n", encoding="utf-8")
if mode == "timeout":
    emit(sys.stdout, b"partial stdout \\xff\\n")
    emit(sys.stderr, b"partial stderr \\xfe\\n")
    time.sleep(30)
if mode != "no-deliverable":
    deliverables = Path.cwd() / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    (deliverables / "answer.txt").write_text("answer\\n", encoding="utf-8")
final_index = "--output-last-message"
if final_index in args:
    final_path = Path(args[args.index(final_index) + 1])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("GDPVAL_FAKE_FINAL_INVALID") == "1":
        final_path.write_bytes(b"final response \\xff\\n")
    else:
        final_path.write_bytes(os.environ.get("GDPVAL_FAKE_FINAL", "final response\\n").encode())
verdict = os.environ.get("GDPVAL_FAKE_VERDICT", "BOXED[A]")
suffix = b"\\xff\\n" if os.environ.get("GDPVAL_FAKE_INVALID_STDOUT") == "1" else b"\\n"
emit(sys.stdout, verdict.encode() + suffix)
emit(sys.stderr, b"diagnostic \\xfe\\n")
sys.exit(int(os.environ.get("GDPVAL_FAKE_EXIT", "0")))
"""


ExecutorClass: TypeAlias = type[CodexExecutor] | type[ClaudeCodeExecutor] | type[CursorExecutor]
ExecutorInstance: TypeAlias = CodexExecutor | ClaudeCodeExecutor | CursorExecutor
JudgeClass: TypeAlias = type[CodexJudgeExecutor] | type[ClaudeCodeJudgeExecutor]


def _make_fake_cli(root: Path) -> Path:
    command = root / "fake-agent"
    command.write_text(textwrap.dedent(_FAKE_CLI).lstrip(), encoding="utf-8")
    command.chmod(0o755)
    return command


def _local_environment(**values: str) -> dict[str, str]:
    environment = {"PATH": os.environ.get("PATH", os.defpath), "KEEP_ME": "retained"}
    environment.update(values)
    return environment


def _execution_request(root: Path, environment: Mapping[str, str] | None = None) -> ExecutionRequest:
    workspace = root / "workspace"
    return ExecutionRequest(
        task=TaskSpec(task_id="task-1", prompt="complete this local task"),
        workspace=workspace,
        deliverables_dir=workspace / "deliverables",
        executor_dir=root / "executor",
        model="test-model",
        timeout_seconds=0.5,
        environment=dict(environment or {}),
    )


def _judge_request(root: Path, environment: Mapping[str, str] | None = None) -> JudgeRequest:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("reference_files", "submission_a", "submission_b"):
        (workspace / name).mkdir()
    return JudgeRequest(
        task_id="task-1",
        task_prompt="choose the better local submission",
        workspace=workspace,
        reference_dir=workspace / "reference_files",
        submission_a_dir=workspace / "submission_a",
        submission_b_dir=workspace / "submission_b",
        executor_dir=root / "judge-executor",
        trial_index=1,
        swapped=False,
        model="test-model",
        timeout_seconds=0.5,
        environment=dict(environment or {}),
    )


def _assert_logs(test: unittest.TestCase, executor_dir: Path, stdout: str, stderr: str) -> None:
    test.assertEqual((executor_dir / "stdout.log").read_text(encoding="utf-8"), stdout)
    test.assertEqual((executor_dir / "stderr.log").read_text(encoding="utf-8"), stderr)


class ExecutorReliabilityTests(unittest.TestCase):
    def test_private_decoders_and_environment_scrubbing_are_fail_closed(self) -> None:
        self.assertEqual(codex_executor_module._text(None), "")
        self.assertEqual(codex_executor_module._text(b"bad \xff"), "bad \ufffd")
        self.assertEqual(claude_executor_module._text(None), "")
        self.assertEqual(cursor_executor_module._text(b"bad \xff"), "bad \ufffd")
        self.assertEqual(cursor_executor_module._text(None), "")

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-output.txt"
            self.assertIsNone(codex_executor_module._read_output_text(missing))
            directory = Path(tmp) / "directory"
            directory.mkdir()
            self.assertIsNone(codex_executor_module._read_output_text(directory))
            with (
                patch.object(Path, "is_file", return_value=True),
                patch.object(Path, "read_text", side_effect=OSError("unreadable")),
            ):
                self.assertIsNone(codex_executor_module._read_output_text(directory / "output.txt"))
            self.assertEqual(
                cursor_executor_module._tree_digest(Path(tmp) / "absent"),
                cursor_executor_module._tree_digest(Path(tmp) / "also-absent"),
            )
            empty_tree = Path(tmp) / "empty-tree" / "nested"
            empty_tree.mkdir(parents=True)
            self.assertEqual(
                cursor_executor_module._tree_digest(empty_tree.parent),
                cursor_executor_module._tree_digest(Path(tmp) / "absent"),
            )

        codex_env = codex_executor_module.subscription_environment(
            {"OPENAI_API_KEY": "secret", "CODEX_ACCESS_TOKEN": "token", "KEEP_ME": "yes"}
        )
        claude_env = claude_executor_module.subscription_environment(
            {"ANTHROPIC_API_KEY": "secret", "CLAUDE_CODE_USE_BEDROCK": "1", "KEEP_ME": "yes"}
        )
        cursor_env = cursor_executor_module.subscription_environment(
            {"CURSOR_API_KEY": "secret", "CURSOR_AUTH_TOKEN": "token", "KEEP_ME": "yes"}
        )
        self.assertEqual(codex_env, {"KEEP_ME": "yes"})
        self.assertEqual(claude_env, {"KEEP_ME": "yes"})
        self.assertEqual(cursor_env, {"KEEP_ME": "yes"})

    def test_executor_preflight_status_failures_are_reported_without_fallback(self) -> None:
        cases: list[tuple[str, ExecutorClass]] = [
            ("codex", CodexExecutor),
            ("claude-code", ClaudeCodeExecutor),
            ("cursor", CursorExecutor),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            command = _make_fake_cli(Path(tmp))
            for name, executor_factory in cases:
                with self.subTest(executor=name):
                    version = subprocess.CompletedProcess([], 0, "fake-version", "")
                    for failure in (OSError("status unavailable"), subprocess.TimeoutExpired([], 15)):
                        executor = executor_factory(command=str(command))
                        with patch.object(subprocess, "run", side_effect=[version, failure]) as run:
                            result = executor.preflight()
                        self.assertEqual(run.call_count, 2)
                        self.assertFalse(result.ok)
                        self.assertIn("could not inspect", result.details[0])

    def test_versions_handle_output_fallback_empty_output_and_failures(self) -> None:
        cases: list[ExecutorInstance] = [
            CodexExecutor(command="codex"),
            ClaudeCodeExecutor(command="claude"),
            CursorExecutor(command="agent"),
        ]
        for executor in cases:
            with self.subTest(executor=executor.name):
                completed = subprocess.CompletedProcess([], 0, stdout="", stderr="fallback 1.0")
                with patch.object(subprocess, "run", return_value=completed):
                    self.assertEqual(executor.version(), "fallback 1.0")
                empty = subprocess.CompletedProcess([], 0, stdout="", stderr="")
                with patch.object(subprocess, "run", return_value=empty):
                    self.assertIsNone(executor.version())
                with patch.object(subprocess, "run", side_effect=OSError("not installed")):
                    self.assertIsNone(executor.version())
                with patch.object(
                    subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired([], 10),
                ):
                    self.assertIsNone(executor.version())

    def test_preflight_rejects_missing_and_non_subscription_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = _make_fake_cli(Path(tmp))

            missing = CodexExecutor(command=str(Path(tmp) / "missing"))
            self.assertFalse(missing.preflight().ok)
            missing_claude = ClaudeCodeExecutor(command=str(Path(tmp) / "missing-claude"))
            self.assertFalse(missing_claude.preflight().ok)
            missing_cursor = CursorExecutor(command=str(Path(tmp) / "missing-cursor"))
            self.assertFalse(missing_cursor.preflight().ok)

            with patch.dict(os.environ, {"GDPVAL_FAKE_CODEX_AUTH": "API key login"}, clear=False):
                result = CodexExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "api")

            with patch.dict(os.environ, {"GDPVAL_FAKE_CODEX_AUTH": "logged in as another provider"}, clear=False):
                result = CodexExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "unknown")

            with patch.dict(
                os.environ,
                {"GDPVAL_FAKE_CODEX_AUTH": "not logged in", "GDPVAL_FAKE_STATUS_EXIT": "1"},
                clear=False,
            ):
                result = CodexExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertIn("not logged in", result.details[0])

            with patch.dict(
                os.environ,
                {"GDPVAL_FAKE_AUTH_JSON": "not json", "GDPVAL_FAKE_STATUS_EXIT": "0"},
                clear=False,
            ):
                result = ClaudeCodeExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "unknown")

            with patch.dict(
                os.environ,
                {
                    "GDPVAL_FAKE_AUTH_JSON": json.dumps(
                        {
                            "loggedIn": True,
                            "authMethod": "claude.ai",
                            "apiProvider": "console",
                            "subscriptionType": "pro",
                        }
                    ),
                    "GDPVAL_FAKE_STATUS_EXIT": "0",
                },
                clear=False,
            ):
                result = ClaudeCodeExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "claude.ai")

            with patch.dict(
                os.environ,
                {"GDPVAL_FAKE_AUTH_JSON": "", "GDPVAL_FAKE_STATUS_EXIT": "1"},
                clear=False,
            ):
                result = ClaudeCodeExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertIn("not logged in", result.details[0])

            with patch.dict(os.environ, {"GDPVAL_FAKE_CURSOR_STATUS": "API key authentication"}, clear=False):
                result = CursorExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "api")

            with patch.dict(os.environ, {"GDPVAL_FAKE_CURSOR_STATUS": "present but unclear"}, clear=False):
                result = CursorExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "unknown")

            with patch.dict(os.environ, {"GDPVAL_FAKE_CURSOR_STATUS": "not authenticated"}, clear=False):
                result = CursorExecutor(command=str(command)).preflight()
            self.assertFalse(result.ok)
            self.assertIn("not authenticated", result.details[0])

    def test_preflight_accepts_positive_subscription_and_account_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = _make_fake_cli(Path(tmp))
            with patch.dict(
                os.environ,
                {
                    "GDPVAL_FAKE_CODEX_AUTH": "ChatGPT subscription",
                    "GDPVAL_FAKE_AUTH_JSON": json.dumps(
                        {
                            "loggedIn": True,
                            "authMethod": "claude.ai",
                            "apiProvider": "firstparty",
                            "subscriptionType": "max",
                        }
                    ),
                    "GDPVAL_FAKE_CURSOR_STATUS": "authenticated account",
                    "GDPVAL_FAKE_STATUS_EXIT": "0",
                },
                clear=False,
            ):
                codex = CodexExecutor(command=str(command)).preflight()
                claude = ClaudeCodeExecutor(command=str(command)).preflight()
                cursor = CursorExecutor(command=str(command)).preflight()
            self.assertTrue(codex.ok)
            self.assertEqual(codex.auth_mode, "chatgpt-subscription")
            self.assertTrue(claude.ok)
            self.assertEqual(claude.auth_mode, "claude-subscription:max")
            self.assertTrue(cursor.ok)
            self.assertEqual(cursor.auth_mode, "cursor-account")

    def test_real_local_cli_execution_persists_output_and_scrubs_environment(self) -> None:
        cases: list[tuple[str, ExecutorClass]] = [
            ("codex", CodexExecutor),
            ("claude-code", ClaudeCodeExecutor),
            ("cursor", CursorExecutor),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _make_fake_cli(root)
            for name, executor_factory in cases:
                with self.subTest(executor=name):
                    case_root = root / name
                    capture = case_root / "captured-env.json"
                    environment = _local_environment(
                        GDPVAL_FAKE_MODE="success",
                        GDPVAL_FAKE_INVALID_STDOUT="1",
                        GDPVAL_FAKE_FINAL_INVALID="1",
                        GDPVAL_CAPTURE_ENV=str(capture),
                        OPENAI_API_KEY="secret-openai",
                        ANTHROPIC_API_KEY="secret-anthropic",
                        CURSOR_API_KEY="secret-cursor",
                    )
                    request = _execution_request(case_root, environment)
                    if name == "cursor":
                        references = request.workspace / "reference_files"
                        references.mkdir(parents=True)
                        (references / "input.txt").write_text("reference\n", encoding="utf-8")
                    executor = executor_factory(command=str(command))
                    executor._version = "fake-version"
                    result = executor.execute(request)
                    self.assertEqual(result.status, ExecutionStatus.COMPLETED)
                    self.assertEqual(result.exit_code, 0)
                    self.assertEqual(result.task_id, "task-1")
                    expected_outputs = (
                        {ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}
                        if name == "codex"
                        else {ExecutorOutput.ARTIFACT_FILES}
                    )
                    self.assertEqual(result.available_outputs, frozenset(expected_outputs))
                    self.assertIsNone(result.failure)
                    self.assertTrue((request.executor_dir / "prompt.txt").is_file())
                    self.assertIn("\ufffd", (request.executor_dir / "stdout.log").read_text(encoding="utf-8"))
                    self.assertIn("\ufffd", (request.executor_dir / "stderr.log").read_text(encoding="utf-8"))
                    captured = json.loads(capture.read_text(encoding="utf-8"))
                    if name == "codex":
                        self.assertNotIn("OPENAI_API_KEY", captured)
                    elif name == "claude-code":
                        self.assertNotIn("ANTHROPIC_API_KEY", captured)
                    else:
                        self.assertNotIn("CURSOR_API_KEY", captured)
                    self.assertEqual(captured["KEEP_ME"], "retained")
                    if name == "codex":
                        self.assertEqual(result.output_text, "final response \ufffd\n")
                        self.assertEqual(result.metadata["forced_login_method"], "chatgpt")
                    if name == "cursor":
                        self.assertTrue(result.metadata["reference_integrity_verified"])
                        self.assertFalse((request.workspace / "reference_files").is_symlink())

    def test_executors_fail_closed_for_no_deliverable_and_nonzero_exit(self) -> None:
        cases: list[tuple[str, ExecutorClass]] = [
            ("codex", CodexExecutor),
            ("claude-code", ClaudeCodeExecutor),
            ("cursor", CursorExecutor),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _make_fake_cli(root)
            for name, executor_factory in cases:
                with self.subTest(executor=name):
                    no_output_root = root / f"{name}-empty"
                    executor = executor_factory(command=str(command))
                    executor._version = "fake-version"
                    result = executor.execute(
                        _execution_request(no_output_root, _local_environment(GDPVAL_FAKE_MODE="no-deliverable"))
                    )
                    expected_status = ExecutionStatus.COMPLETED if name == "codex" else ExecutionStatus.NO_DELIVERABLE
                    self.assertEqual(result.status, expected_status)
                    expected_outputs = (
                        {ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}
                        if name == "codex"
                        else {ExecutorOutput.ARTIFACT_FILES}
                    )
                    self.assertEqual(result.available_outputs, frozenset(expected_outputs))
                    self.assertIsNone(result.failure)
                    failed_root = root / f"{name}-failed"
                    executor = executor_factory(command=str(command))
                    executor._version = "fake-version"
                    result = executor.execute(
                        _execution_request(
                            failed_root,
                            _local_environment(GDPVAL_FAKE_MODE="success", GDPVAL_FAKE_EXIT="3"),
                        )
                    )
                    self.assertEqual(result.status, ExecutionStatus.FAILED)
                    self.assertEqual(result.exit_code, 3)
                    self.assertEqual(result.available_outputs, frozenset())
                    self.assertIsNotNone(result.failure)
                    assert result.failure is not None
                    self.assertEqual(result.failure.kind, FailureKind.PROCESS)
                    self.assertEqual(result.failure.impact, FailureImpact.RUN)
                    self.assertIsNone(result.output_text)

    def test_cursor_reference_mutation_fails_closed_and_restores_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _make_fake_cli(root)
            request = _execution_request(root / "mutation", _local_environment(GDPVAL_FAKE_MODE="mutate"))
            references = request.workspace / "reference_files"
            references.mkdir(parents=True)
            (references / "input.txt").write_text("reference\n", encoding="utf-8")
            executor = CursorExecutor(command=str(command))
            executor._version = "fake-version"
            result = executor.execute(request)
            self.assertEqual(result.status, ExecutionStatus.FAILED)
            self.assertEqual(result.available_outputs, frozenset())
            self.assertIsNotNone(result.failure)
            assert result.failure is not None
            self.assertEqual(result.failure.kind, FailureKind.INTEGRITY)
            self.assertEqual(result.failure.impact, FailureImpact.RUN)
            self.assertIsNone(result.output_text)
            self.assertFalse(result.metadata["reference_integrity_verified"])
            self.assertIn("mutation", (request.executor_dir / "stderr.log").read_text(encoding="utf-8"))
            self.assertEqual((references / "input.txt").read_text(encoding="utf-8"), "mutated\n")

    def test_cursor_isolation_cleanup_and_symlink_failure_restore_original_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            references = workspace / "reference_files"
            references.mkdir(parents=True)
            (references / "input.txt").write_text("original\n", encoding="utf-8")
            protected = workspace.parent / "cursor-reference-files-readonly"
            protected.write_text("stale\n", encoding="utf-8")
            executor = CursorExecutor(command="agent")
            moved, digest = executor._isolate_reference_files(workspace)
            self.assertIsNotNone(moved)
            self.assertIsNotNone(digest)
            executor._restore_reference_files(workspace, moved)
            self.assertEqual((references / "input.txt").read_text(encoding="utf-8"), "original\n")

            shutil_target = workspace.parent / "cursor-reference-files-readonly"
            shutil_target.mkdir()
            (shutil_target / "stale.txt").write_text("stale\n", encoding="utf-8")
            moved, _ = executor._isolate_reference_files(workspace)
            executor._restore_reference_files(workspace, moved)
            executor._restore_reference_files(workspace, None)

            protected.mkdir()
            (protected / "input.txt").write_text("restored\n", encoding="utf-8")
            executor._restore_reference_files(workspace, protected)
            self.assertEqual((references / "input.txt").read_text(encoding="utf-8"), "restored\n")

            with patch.object(Path, "symlink_to", side_effect=OSError("symlinks disabled")):
                with self.assertRaisesRegex(RuntimeError, "symlink support"):
                    executor._isolate_reference_files(workspace)
            self.assertTrue(references.is_dir())
            self.assertEqual((references / "input.txt").read_text(encoding="utf-8"), "restored\n")

    def test_executor_timeout_interrupt_and_oserror_persist_durable_logs(self) -> None:
        cases: list[tuple[str, ExecutorClass]] = [
            ("codex", CodexExecutor),
            ("claude-code", ClaudeCodeExecutor),
            ("cursor", CursorExecutor),
        ]
        for name, executor_factory in cases:
            with self.subTest(executor=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                executor = executor_factory(command="fake-agent")
                executor._version = "fake-version"
                request = _execution_request(root / "timeout")
                if name == "cursor":
                    references = request.workspace / "reference_files"
                    references.mkdir(parents=True)
                    (references / "input.txt").write_text("reference\n", encoding="utf-8")
                timeout = subprocess.TimeoutExpired(["fake-agent"], 1, output=b"out \xff\n", stderr=b"err \xfe\n")
                with patch.object(subprocess, "run", side_effect=timeout) as run:
                    result = executor.execute(request)
                run.assert_called_once()
                self.assertEqual(result.status, ExecutionStatus.TIMED_OUT)
                self.assertEqual(result.available_outputs, frozenset())
                self.assertIsNotNone(result.failure)
                assert result.failure is not None
                self.assertEqual(result.failure.kind, FailureKind.TIMEOUT)
                self.assertEqual(result.failure.impact, FailureImpact.RUN)
                self.assertIsNone(result.output_text)
                _assert_logs(self, request.executor_dir, "out \ufffd\n", "err \ufffd\n")

                interrupted_root = root / "interrupt"
                executor = executor_factory(command="fake-agent")
                executor._version = "fake-version"
                request = _execution_request(interrupted_root)
                with patch.object(subprocess, "run", side_effect=KeyboardInterrupt):
                    result = executor.execute(request)
                self.assertEqual(result.status, ExecutionStatus.INTERRUPTED)
                self.assertEqual(result.available_outputs, frozenset())
                self.assertIsNone(result.output_text)
                self.assertIsNotNone(result.failure)
                assert result.failure is not None
                self.assertEqual(result.failure.kind, FailureKind.INTERRUPTED)
                self.assertEqual(result.failure.impact, FailureImpact.RUN)
                _assert_logs(self, request.executor_dir, "", "")

                failed_root = root / "oserror"
                executor = executor_factory(command="fake-agent")
                executor._version = "fake-version"
                request = _execution_request(failed_root)
                with patch.object(subprocess, "run", side_effect=OSError("launch failed")):
                    result = executor.execute(request)
                self.assertEqual(result.status, ExecutionStatus.FAILED)
                self.assertEqual(result.available_outputs, frozenset())
                self.assertIsNotNone(result.failure)
                assert result.failure is not None
                self.assertEqual(result.failure.kind, FailureKind.PROCESS)
                self.assertEqual(result.failure.impact, FailureImpact.RUN)
                self.assertIsNone(result.output_text)
                self.assertIn("launch failed", (request.executor_dir / "stderr.log").read_text(encoding="utf-8"))

    def test_build_commands_preserve_reasoning_and_network_policy_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = _execution_request(Path(tmp))
            command = CodexExecutor(reasoning_effort="high", network_enabled=False).build_command(request)
            self.assertIn('model_reasoning_effort="high"', command)
            self.assertIn('web_search="disabled"', command)
            claude_command = ClaudeCodeExecutor(network_enabled=True).build_command(request)
            settings = json.loads(claude_command[claude_command.index("--settings") + 1])
            self.assertNotIn("network", settings["sandbox"])
            self.assertNotIn("--disallowedTools", claude_command)

            executor = CodexExecutor(command="codex", reasoning_effort="high")
            executor._version = "fake-version"
            with patch.object(
                subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ):
                result = executor.execute(request)
            self.assertEqual(result.metadata["reasoning_effort_requested"], "high")


class JudgeReliabilityTests(unittest.TestCase):
    def test_judge_versions_and_codex_preflight_cover_sandbox_probe_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _make_fake_cli(root)
            codex = CodexJudgeExecutor(command=str(command), reasoning_effort="low")
            claude = ClaudeCodeJudgeExecutor(command=str(command), max_turns=3)
            with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "v", "")):
                self.assertEqual(codex._version_with_environment({}), "v")
            with patch.object(
                subprocess,
                "run",
                side_effect=(OSError("missing"), subprocess.TimeoutExpired([], 1)),
            ):
                self.assertIsNone(codex._version_with_environment({}))
                self.assertIsNone(codex._version_with_environment({}))
            with patch.object(
                subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "claude-v", ""),
            ):
                self.assertEqual(claude._version_with_environment({}), "claude-v")

            environment = _local_environment(
                GDPVAL_FAKE_CODEX_AUTH="chatgpt subscription",
                GDPVAL_FAKE_SANDBOX_EXIT="0",
            )
            preflight = codex.preflight(environment)
            self.assertTrue(preflight.ok)
            self.assertIn("sandbox probe passed", preflight.details[1])

            environment["GDPVAL_FAKE_CODEX_AUTH"] = "API key login"
            preflight = codex.preflight(environment)
            self.assertFalse(preflight.ok)
            self.assertEqual(preflight.auth_mode, "api")

            environment["GDPVAL_FAKE_CODEX_AUTH"] = "other login"
            preflight = codex.preflight(environment)
            self.assertFalse(preflight.ok)
            self.assertEqual(preflight.auth_mode, "unknown")

            environment["GDPVAL_FAKE_CODEX_AUTH"] = "chatgpt subscription"
            environment["GDPVAL_FAKE_SANDBOX_EXIT"] = "1"
            preflight = codex.preflight(environment)
            self.assertFalse(preflight.ok)
            self.assertIn("refusing blind judging", preflight.details[1])

            missing = CodexJudgeExecutor(command=str(root / "missing"))
            self.assertFalse(missing.preflight(environment).ok)

    def test_codex_judge_policy_helpers_and_probe_errors_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _make_fake_cli(root)
            self.assertTrue(_paths_overlap(root, root / "child"))
            protected = _protected_read_roots({"HOME": str(root), "USERPROFILE": str(root)})
            self.assertEqual(protected.count(root), 1)

            with patch.object(shutil, "which", return_value=None):
                self.assertTrue(_resolved_command_read_paths(str(command), {}))
            with patch.object(Path, "resolve", side_effect=OSError("resolve unavailable")):
                self.assertEqual(_resolved_path(root / "path"), (root / "path").absolute())
            with patch.object(Path, "resolve", side_effect=[OSError("strict unavailable"), command.absolute()]):
                resolved_paths = _resolved_command_read_paths(str(command), {})
            self.assertIn(str(command.absolute()), resolved_paths)

            judge = CodexJudgeExecutor(command=str(command))
            with patch.object(os, "name", "nt"):
                ok, detail = judge._probe_read_confinement({})
            self.assertFalse(ok)
            self.assertIn("native Windows", detail)
            with patch.object(codex_judge_module, "_resolved_command_read_paths", return_value=()):
                ok, detail = judge._probe_read_confinement({})
            self.assertFalse(ok)
            self.assertIn("could not resolve", detail)
            with patch.object(subprocess, "run", side_effect=OSError("probe failed")):
                ok, detail = judge._probe_read_confinement({"PATH": os.environ.get("PATH", os.defpath)})
            self.assertFalse(ok)
            self.assertIn("could not verify", detail)

            invalid_log = root / "invalid.log"
            invalid_log.write_bytes(b"bad \xff")
            _normalize_utf8_log(invalid_log)
            self.assertIn("\ufffd", invalid_log.read_text(encoding="utf-8"))
            invalid_log.write_bytes(b"bad \xff")
            with patch.object(Path, "write_bytes", side_effect=OSError("read-only")):
                _normalize_utf8_log(invalid_log)

            environment = _local_environment(GDPVAL_FAKE_CODEX_AUTH="chatgpt", GDPVAL_FAKE_STATUS_EXIT="1")
            with patch.object(
                subprocess,
                "run",
                side_effect=[subprocess.CompletedProcess([], 0, "v", ""), subprocess.CompletedProcess([], 1, "", "")],
            ):
                result = judge.preflight(environment)
            self.assertFalse(result.ok)
            self.assertIn("not logged in", result.details[0])

            environment["GDPVAL_FAKE_STATUS_EXIT"] = "0"
            with patch.object(
                subprocess,
                "run",
                side_effect=[subprocess.CompletedProcess([], 0, "v", ""), OSError("status failed")],
            ):
                result = judge.preflight(environment)
            self.assertFalse(result.ok)
            self.assertIn("could not inspect", result.details[0])

            request = _judge_request(root / "reasoning", _local_environment())
            judge = CodexJudgeExecutor(command=str(command), reasoning_effort="low")
            command_args = judge.build_command(request)
            self.assertIn('model_reasoning_effort="low"', command_args)
            judge._version = "fake-version"
            judge_result = judge.judge(request)
            self.assertEqual(judge_result.metadata["reasoning_effort_requested"], "low")

    def test_claude_judge_preflight_rejects_unverified_or_invalid_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _make_fake_cli(root)
            missing = ClaudeCodeJudgeExecutor(command=str(root / "missing"))
            self.assertFalse(missing.preflight(_local_environment()).ok)
            judge = ClaudeCodeJudgeExecutor(command=str(command))
            with patch.object(os, "name", "nt"):
                result = judge.preflight(_local_environment())
            self.assertFalse(result.ok)
            self.assertIn("native Windows", result.details[0])

            environment = _local_environment(GDPVAL_FAKE_AUTH_JSON="invalid json")
            result = judge.preflight(environment)
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "unknown")
            environment["GDPVAL_FAKE_AUTH_JSON"] = json.dumps({"loggedIn": False})
            result = judge.preflight(environment)
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "unknown")
            environment["GDPVAL_FAKE_AUTH_JSON"] = json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "apiProvider": "firstparty",
                    "subscriptionType": "enterprise",
                }
            )
            result = judge.preflight(environment)
            self.assertFalse(result.ok)
            self.assertEqual(result.auth_mode, "claude-subscription:enterprise")
            self.assertIn("disabled", result.details[1])

            environment["GDPVAL_FAKE_STATUS_EXIT"] = "1"
            environment["GDPVAL_FAKE_AUTH_JSON"] = ""
            result = judge.preflight(environment)
            self.assertFalse(result.ok)
            self.assertIn("not logged in", result.details[0])

            with patch.object(
                subprocess,
                "run",
                side_effect=[subprocess.CompletedProcess([], 0, "v", ""), OSError("status failed")],
            ):
                result = judge.preflight(_local_environment())
            self.assertFalse(result.ok)
            self.assertIn("could not inspect", result.details[0])

            self.assertEqual(claude_judge_module._text(None), "")
            self.assertEqual(claude_judge_module._text(b"judge \xff"), "judge \ufffd")
            with patch.object(
                subprocess,
                "run",
                side_effect=(OSError("missing"), subprocess.TimeoutExpired([], 1)),
            ):
                self.assertIsNone(judge._version_with_environment({}))
                self.assertIsNone(judge._version_with_environment({}))

    def test_real_local_judges_parse_verdicts_and_persist_scrubbed_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _make_fake_cli(root)
            environment = _local_environment(
                GDPVAL_FAKE_FINAL="BOXED[A]",
                GDPVAL_FAKE_VERDICT="BOXED[B]",
                GDPVAL_CAPTURE_ENV=str(root / "codex-env.json"),
                GDPVAL_FAKE_INVALID_STDOUT="1",
                OPENAI_API_KEY="secret",
                ANTHROPIC_API_KEY="secret",
            )
            codex = CodexJudgeExecutor(command=str(command))
            codex._version = "fake-version"
            request = _judge_request(root / "codex", environment)
            result = codex.judge(request)
            self.assertEqual(result.verdict, Verdict.A)
            self.assertIsNone(result.metadata["parse_error"])
            self.assertIn("\ufffd", result.stdout_path.read_text(encoding="utf-8"))
            self.assertEqual((request.workspace / "JUDGE_TASK.md").exists(), False)
            captured = json.loads((root / "codex-env.json").read_text(encoding="utf-8"))
            self.assertNotIn("OPENAI_API_KEY", captured)
            self.assertEqual(captured["KEEP_ME"], "retained")

            claude_environment = _local_environment(
                GDPVAL_FAKE_VERDICT="BOXED[B]",
                GDPVAL_CAPTURE_ENV=str(root / "claude-env.json"),
                ANTHROPIC_API_KEY="secret",
            )
            claude = ClaudeCodeJudgeExecutor(command=str(command))
            claude._version = "fake-version"
            request = _judge_request(root / "claude", claude_environment)
            result = claude.judge(request)
            self.assertEqual(result.verdict, Verdict.B)
            self.assertIsNone(result.metadata["parse_error"])
            self.assertIn("\ufffd", result.stderr_path.read_text(encoding="utf-8"))
            captured = json.loads((root / "claude-env.json").read_text(encoding="utf-8"))
            self.assertNotIn("ANTHROPIC_API_KEY", captured)
            self.assertEqual(captured["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"], "1")

    def test_judges_fail_closed_on_parse_timeout_oserror_and_nonzero_exit(self) -> None:
        cases: list[tuple[str, JudgeClass]] = [
            ("codex", CodexJudgeExecutor),
            ("claude", ClaudeCodeJudgeExecutor),
        ]
        for name, judge_factory in cases:
            with self.subTest(judge=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                judge = judge_factory(command="fake-agent")
                judge._version = "fake-version"
                invalid_root = root / "invalid"
                request = _judge_request(invalid_root, _local_environment())
                with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "invalid", "")):
                    result = judge.judge(request)
                self.assertIsNone(result.verdict)
                self.assertIsNotNone(result.metadata["parse_error"])

                timeout_root = root / "timeout"
                judge = judge_factory(command="fake-agent")
                judge._version = "fake-version"
                request = _judge_request(timeout_root, _local_environment())
                timeout = subprocess.TimeoutExpired(["fake-agent"], 1, output=b"judge \xff", stderr=b"err \xfe")
                with patch.object(subprocess, "run", side_effect=timeout):
                    result = judge.judge(request)
                self.assertIsNone(result.verdict)
                self.assertEqual(result.metadata["parse_error"], "judge timed out")
                if name == "claude":
                    self.assertIn("\ufffd", result.stdout_path.read_text(encoding="utf-8"))
                else:
                    self.assertEqual(result.stdout_path.read_text(encoding="utf-8"), "")

                error_root = root / "error"
                judge = judge_factory(command="fake-agent")
                judge._version = "fake-version"
                request = _judge_request(error_root, _local_environment())
                with patch.object(subprocess, "run", side_effect=OSError("judge launch failed")):
                    result = judge.judge(request)
                self.assertIsNone(result.verdict)
                parse_error = result.metadata["parse_error"]
                if not isinstance(parse_error, str):
                    self.fail("judge parse error is not text")
                self.assertIn("judge launch failed", parse_error)
                self.assertIn("judge launch failed", result.stderr_path.read_text(encoding="utf-8"))

                nonzero_root = root / "nonzero"
                judge = judge_factory(command="fake-agent")
                judge._version = "fake-version"
                request = _judge_request(nonzero_root, _local_environment())
                completed = subprocess.CompletedProcess([], 7, stdout="BOXED[A]", stderr="rejected")
                with patch.object(subprocess, "run", return_value=completed):
                    result = judge.judge(request)
                self.assertIsNone(result.verdict)
                self.assertIsNone(result.metadata["parse_error"])

                if name == "claude":
                    interrupt_root = root / "interrupt"
                    judge = judge_factory(command="fake-agent")
                    judge._version = "fake-version"
                    request = _judge_request(interrupt_root, _local_environment())
                    with patch.object(subprocess, "run", side_effect=KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            judge.judge(request)
                    self.assertTrue(request.executor_dir.joinpath("stdout.log").is_file())
                    self.assertTrue(request.executor_dir.joinpath("stderr.log").is_file())

    def test_pairwise_reliability_rejects_unsafe_inputs_and_persists_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing"
            with self.assertRaisesRegex(ValueError, "directory not found"):
                matched_tasks(missing, missing)

            candidate = root / "candidate"
            repeat = candidate / "task_x" / "repeat_0"
            repeat.mkdir(parents=True)
            (repeat / "artifact.txt").write_text("x", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "escapes candidate root"):
                _assert_directory(outside, within=candidate)
            with self.assertRaisesRegex(ValueError, "no task"):
                matched_tasks(candidate / "task_x" / "repeat_0", candidate / "task_x" / "repeat_0")

            symlink_target = root / "symlink-target"
            symlink_target.mkdir()
            (candidate / "task_link").symlink_to(symlink_target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked task"):
                matched_tasks(candidate, candidate)
            with self.assertRaisesRegex(ValueError, "symlink not allowed in judged artifact path"):
                _assert_directory(candidate / "task_link", within=candidate)

            candidate_a = root / "a"
            candidate_b = root / "b"
            for candidate_root in (candidate_a, candidate_b):
                task = candidate_root / "task_x" / "repeat_0"
                task.mkdir(parents=True)
                (task / "artifact.txt").write_text("x", encoding="utf-8")
            (candidate_a / "task_x" / "repeat_0" / "reference_files").mkdir()
            with self.assertRaisesRegex(ValueError, "presence differs"):
                validate_reference_equivalence(
                    candidate_a / "task_x" / "repeat_0", candidate_b / "task_x" / "repeat_0"
                )

            file_link = candidate_a / "task_x" / "repeat_0" / "linked.txt"
            file_link.symlink_to(candidate_b / "task_x" / "repeat_0" / "artifact.txt")
            with self.assertRaisesRegex(ValueError, "symlink not allowed"):
                tree_hash(candidate_a / "task_x" / "repeat_0")
            with self.assertRaisesRegex(ValueError, "symlink not allowed"):
                _copy_tree(candidate_a / "task_x" / "repeat_0", root / "staged", submission=True)

            nested_a = root / "nested-a" / "task_x" / "repeat_0"
            nested_b = root / "nested-b" / "task_x" / "repeat_0"
            (nested_a / "reference_files").mkdir(parents=True)
            (nested_b / "reference_files").mkdir(parents=True)
            for task, text in ((nested_a, "a"), (nested_b, "b")):
                (task / "subdir").mkdir()
                (task / "subdir" / "artifact.txt").write_text(text, encoding="utf-8")
                (task / "log.txt").write_text("identity", encoding="utf-8")
                (task / "reference_files" / "same.txt").write_text("same", encoding="utf-8")
            trial = prepare_trial(root / "out", "task_x", nested_a, nested_b, trial_index=0, seed=4)
            self.assertTrue((trial.submission_a_dir / "subdir" / "artifact.txt").is_file())
            self.assertFalse((trial.submission_a_dir / "log.txt").exists())
            self.assertTrue((trial.reference_dir / "same.txt").is_file())
            self.assertIn("Task:\nwrite a report", build_judge_prompt("write a report"))
            self.assertEqual(normalize_verdict(Verdict.B, False), Verdict.B)
            self.assertEqual(aggregate([])["score_a"], 0.0)
            metadata_path = trial.executor_dir / "metadata.json"
            write_trial_metadata(metadata_path, {"task": trial.task_key, "swapped": trial.swapped})
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["task"], "task_x")
            with self.assertRaises(ValueError):
                parse_verdict("")


if __name__ == "__main__":
    unittest.main()
