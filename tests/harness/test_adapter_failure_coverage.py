# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO, NoReturn, cast
from unittest.mock import patch

from eval_harness.executors.base import ExecutionRequest, ExecutionStatus, TaskSpec
from eval_harness.executors.claude_code import ClaudeCodeExecutor
from eval_harness.executors.codex import CodexExecutor
from eval_harness.executors.cursor import CursorExecutor
from eval_harness.executors.registry import create_executor, get_executor_descriptor, list_executors, main
from eval_harness.judges.base import JudgeRequest
from eval_harness.judges.claude_code import ClaudeCodeJudgeExecutor
from eval_harness.judges.codex import CodexJudgeExecutor
from eval_harness.judges.pairwise import parse_verdict


def _write_command(root: Path, name: str, body: str) -> Path:
    command = root / name
    command.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    command.chmod(0o755)
    return command


def _execution_request(root: Path, environment: dict[str, str] | None = None) -> ExecutionRequest:
    workspace = root / "workspace"
    return ExecutionRequest(
        task=TaskSpec(task_id="task-1", prompt="do the fake task"),
        workspace=workspace,
        deliverables_dir=workspace / "deliverables",
        executor_dir=root / "executor",
        model="test-model",
        timeout_seconds=1.0,
        environment=environment or {},
    )


def _judge_request(root: Path, environment: dict[str, str] | None = None) -> JudgeRequest:
    workspace = root / "judge-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return JudgeRequest(
        task_id="task-1",
        task_prompt="choose a submission",
        workspace=workspace,
        reference_dir=workspace / "reference_files",
        submission_a_dir=workspace / "submission_a",
        submission_b_dir=workspace / "submission_b",
        executor_dir=root / "judge-executor",
        trial_index=0,
        swapped=False,
        model="judge-model",
        timeout_seconds=1.0,
        environment=environment or {},
    )


def _raise_timeout(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise subprocess.TimeoutExpired(cmd=["fake"], timeout=1, output=b"partial \xff", stderr=b"error \xfe")


def _raise_interrupt(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise KeyboardInterrupt


def _raise_os_error(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise OSError("fake command unavailable")


class ExecutorAdapterFailureTests(unittest.TestCase):
    def test_executor_registry_lists_metadata_and_legacy_runtime_is_explicit(self) -> None:
        descriptors = list_executors()
        self.assertEqual(
            [descriptor.name for descriptor in descriptors], ["claude-code", "codex", "cursor", "stirrup"]
        )
        self.assertEqual(get_executor_descriptor("codex").usage_mode, "ChatGPT subscription login only")
        with self.assertRaisesRegex(ValueError, "unknown executor"):
            get_executor_descriptor("unknown")
        with patch("builtins.print") as printer:
            main()
        self.assertEqual(printer.call_count, 4)

    def test_executor_version_probes_use_replacement_decoding_and_fail_closed(self) -> None:
        for module, executor in (
            ("eval_harness.executors.codex", CodexExecutor(command="codex")),
            ("eval_harness.executors.claude_code", ClaudeCodeExecutor(command="claude")),
            ("eval_harness.executors.cursor", CursorExecutor(command="agent")),
        ):
            with patch(
                f"{module}.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="fake 1.0"),
            ) as run:
                self.assertEqual(executor.version(), "fake 1.0")
            self.assertEqual(run.call_args.kwargs["errors"], "replace")
            with patch(f"{module}.subprocess.run", side_effect=subprocess.TimeoutExpired([], 1)):
                self.assertIsNone(executor.version())
            with patch(f"{module}.subprocess.run", side_effect=OSError("probe unavailable")):
                self.assertIsNone(executor.version())

    def test_codex_commands_record_network_reasoning_and_subscription_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _execution_request(root, {})
            request = ExecutionRequest(
                task=request.task,
                workspace=request.workspace,
                deliverables_dir=request.deliverables_dir,
                executor_dir=request.executor_dir,
                model="reasoning-model",
                timeout_seconds=request.timeout_seconds,
                environment={"PATH": "/bin", "OPENAI_API_KEY": "secret", "KEEP": "yes"},
            )
            executor = CodexExecutor(network_enabled=False, reasoning_effort="high", command="codex")
            command = executor.build_command(request)
            self.assertIn('model_reasoning_effort="high"', command)
            self.assertIn("--model", command)
            self.assertIn('web_search="disabled"', command)

            captured: dict[str, object] = {}

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.update(kwargs)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            executor._version = "fake"
            with patch("eval_harness.executors.codex.subprocess.run", side_effect=fake_run):
                result = executor.execute(request)
            self.assertEqual(result.status, ExecutionStatus.NO_DELIVERABLE)
            self.assertEqual(captured["errors"], "replace")
            environment = captured["env"]
            assert isinstance(environment, dict)
            self.assertNotIn("OPENAI_API_KEY", environment)
            self.assertEqual(environment["KEEP"], "yes")
            self.assertEqual(result.metadata["reasoning_effort_requested"], "high")
            self.assertFalse(result.metadata["cloud_execution"])

    def test_claude_preflight_status_probe_errors_have_no_implicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _write_command(root, "claude", "exit 0\n")
            executor = ClaudeCodeExecutor(command=str(command))
            with patch(
                "eval_harness.executors.claude_code.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="claude fake", stderr=""),
                    subprocess.TimeoutExpired(["claude", "auth", "status"], 15),
                ],
            ):
                timed_out = executor.preflight()
            self.assertFalse(timed_out.ok)
            self.assertIn("could not inspect", timed_out.details[0])

            with patch(
                "eval_harness.executors.claude_code.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="claude fake", stderr=""),
                    subprocess.CompletedProcess([], 1, stdout="", stderr=""),
                ],
            ):
                logged_out = executor.preflight()
            self.assertFalse(logged_out.ok)
            self.assertEqual(logged_out.details, ("Claude Code is not logged in",))

    def test_cursor_reference_isolation_restores_on_symlink_failure_and_replaces_stale_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            references = workspace / "reference_files"
            references.mkdir(parents=True)
            (references / "input.txt").write_text("source\n", encoding="utf-8")
            protected = workspace.parent / "cursor-reference-files-readonly"
            protected.write_text("stale", encoding="utf-8")
            executor = CursorExecutor()
            with patch.object(Path, "symlink_to", side_effect=OSError("symlinks disabled")):
                with self.assertRaisesRegex(RuntimeError, "symlink support"):
                    executor._isolate_reference_files(workspace)
            self.assertTrue(references.is_dir())
            self.assertEqual((references / "input.txt").read_text(encoding="utf-8"), "source\n")
            self.assertFalse(protected.exists())

    def test_cursor_execution_handles_no_deliverable_interrupt_and_replacement_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = CursorExecutor(command="agent")
            executor._version = "agent fake"
            request = _execution_request(root)

            with patch(
                "eval_harness.executors.cursor.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout="out", stderr="err"),
            ) as run:
                no_deliverable = executor.execute(request)
            self.assertEqual(no_deliverable.status, ExecutionStatus.NO_DELIVERABLE)
            self.assertEqual(run.call_args.kwargs["errors"], "replace")

            references = request.workspace / "reference_files"
            references.mkdir(parents=True)
            (references / "source.txt").write_text("source", encoding="utf-8")
            with patch("eval_harness.executors.cursor.subprocess.run", side_effect=_raise_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    executor.execute(request)
            self.assertTrue(references.is_dir())
            self.assertEqual((references / "source.txt").read_text(encoding="utf-8"), "source")

    def test_executor_registries_construct_only_supported_local_adapters(self) -> None:
        codex = create_executor("codex", reasoning_effort="high")
        claude = create_executor("claude-code", claude_max_turns=3)
        cursor = create_executor("cursor", network_enabled=True)
        assert isinstance(codex, CodexExecutor)
        self.assertEqual(codex.reasoning_effort, "high")
        assert isinstance(claude, ClaudeCodeExecutor)
        self.assertEqual(claude.max_turns, 3)
        assert isinstance(cursor, CursorExecutor)
        self.assertTrue(cursor.network_enabled)
        with self.assertRaisesRegex(ValueError, "not available"):
            create_executor("stirrup")
        with self.assertRaisesRegex(ValueError, "unknown executor"):
            create_executor("missing")

    def test_codex_preflight_rejects_missing_api_unknown_and_nonzero_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = CodexExecutor(command=str(root / "missing"))
            result = missing.preflight()
            self.assertFalse(result.ok)
            self.assertIn("not found", result.details[0])

            command = _write_command(
                root,
                "codex",
                """
                if [ "$1" = "--version" ]; then echo 'codex 9'; exit 0; fi
                if [ "$1" = "login" ]; then
                    printf '%s\n' "${FAKE_AUTH:-chatgpt}"
                    exit "${FAKE_EXIT:-0}"
                fi
                exit 0
                """,
            )
            executor = CodexExecutor(command=str(command))
            with patch.dict(os.environ, {"FAKE_AUTH": "api key login", "FAKE_EXIT": "0"}, clear=False):
                api = executor.preflight()
            self.assertFalse(api.ok)
            self.assertEqual(api.auth_mode, "api")

            with patch.dict(os.environ, {"FAKE_AUTH": "local account", "FAKE_EXIT": "0"}, clear=False):
                unknown = executor.preflight()
            self.assertFalse(unknown.ok)
            self.assertEqual(unknown.auth_mode, "unknown")

            with patch.dict(os.environ, {"FAKE_AUTH": "logout", "FAKE_EXIT": "9"}, clear=False):
                failed = executor.preflight()
            self.assertFalse(failed.ok)
            self.assertEqual(failed.details, ("logout",))

    def test_codex_version_and_execution_persist_failure_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = CodexExecutor(command=str(root / "missing"))
            self.assertIsNone(executor.version())
            request = _execution_request(root)
            failed = executor.execute(request)
            self.assertEqual(failed.status, ExecutionStatus.FAILED)
            self.assertIn("No such file or directory", (request.executor_dir / "stderr.log").read_text())

            command = _write_command(
                root,
                "codex-run",
                """
                if [ "$1" = "--version" ]; then echo 'codex fake'; exit 0; fi
                if [ "$1" = "exec" ]; then
                    output=''
                    while [ "$#" -gt 0 ]; do
                        if [ "$1" = "--output-last-message" ]; then output="$2"; shift; fi
                        shift
                    done
                    case "${FAKE_MODE:-success}" in
                        success)
                            mkdir -p deliverables
                            printf 'artifact\n' > deliverables/result.txt
                            printf 'final \377\n' > "$output"
                            ;;
                        no-deliverable) printf 'no artifact\n' > "$output" ;;
                        nonzero) printf 'failure\n' >&2; exit 7 ;;
                    esac
                fi
                """,
            )
            executor = CodexExecutor(command=str(command))
            executor._version = "codex fake"
            completed = executor.execute(_execution_request(root, {"FAKE_MODE": "success"}))
            self.assertEqual(completed.status, ExecutionStatus.COMPLETED)
            self.assertEqual(completed.output_text, "final ÿ\n")
            self.assertEqual(completed.exit_code, 0)
            self.assertTrue(completed.metadata["cloud_execution"] is False)

            no_deliverable = executor.execute(
                _execution_request(root / "no-deliverable", {"FAKE_MODE": "no-deliverable"})
            )
            self.assertEqual(no_deliverable.status, ExecutionStatus.NO_DELIVERABLE)
            nonzero = executor.execute(_execution_request(root / "nonzero", {"FAKE_MODE": "nonzero"}))
            self.assertEqual(nonzero.status, ExecutionStatus.FAILED)
            self.assertEqual(nonzero.exit_code, 7)

    def test_codex_execution_timeout_and_interrupt_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _execution_request(root)
            executor = CodexExecutor(command="codex")
            executor._version = "test"
            with patch("eval_harness.executors.codex.subprocess.run", side_effect=_raise_timeout):
                timed_out = executor.execute(request)
            self.assertEqual(timed_out.status, ExecutionStatus.TIMED_OUT)
            self.assertEqual((request.executor_dir / "stdout.log").read_text(), "partial �")
            self.assertEqual((request.executor_dir / "stderr.log").read_text(), "error �")

            with patch("eval_harness.executors.codex.subprocess.run", side_effect=_raise_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    executor.execute(request)
            self.assertTrue((request.executor_dir / "stdout.log").is_file())

    def test_claude_preflight_accepts_subscription_and_rejects_malformed_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = ClaudeCodeExecutor(command=str(root / "missing"))
            self.assertFalse(missing.preflight().ok)
            command = _write_command(
                root,
                "claude",
                """
                if [ "$1" = "--version" ]; then echo 'claude fake'; exit 0; fi
                if [ "$1" = "auth" ]; then
                    if [ "${FAKE_AUTH:-valid}" = "invalid" ]; then echo 'not json'; exit 0; fi
                    if [ "${FAKE_AUTH:-valid}" = "api" ]; then echo '{\"loggedIn\":true,\"authMethod\":\"api\",\"apiProvider\":\"console\",\"subscriptionType\":\"\"}'; exit 0; fi
                    if [ "${FAKE_AUTH:-valid}" = "failed" ]; then echo 'logged out' >&2; exit 4; fi
                    echo '{\"loggedIn\":true,\"authMethod\":\"claude.ai\",\"apiProvider\":\"firstparty\",\"subscriptionType\":\"max\"}'
                fi
                """,
            )
            executor = ClaudeCodeExecutor(command=str(command))
            with patch.dict(os.environ, {"FAKE_AUTH": "valid"}, clear=False):
                valid = executor.preflight()
            self.assertTrue(valid.ok)
            self.assertEqual(valid.auth_mode, "claude-subscription:max")
            with patch.dict(os.environ, {"FAKE_AUTH": "invalid"}, clear=False):
                invalid = executor.preflight()
            self.assertFalse(invalid.ok)
            self.assertEqual(invalid.auth_mode, "unknown")
            with patch.dict(os.environ, {"FAKE_AUTH": "api"}, clear=False):
                api = executor.preflight()
            self.assertFalse(api.ok)
            self.assertEqual(api.auth_mode, "api")
            with patch.dict(os.environ, {"FAKE_AUTH": "failed"}, clear=False):
                failed = executor.preflight()
            self.assertFalse(failed.ok)
            self.assertEqual(failed.details, ("logged out",))

    def test_claude_execution_covers_no_deliverable_nonzero_timeout_and_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _write_command(
                root,
                "claude-run",
                """
                if [ "$1" = "--version" ]; then echo 'claude fake'; exit 0; fi
                if [ "$1" = "-p" ]; then
                    case "${FAKE_MODE:-no-deliverable}" in
                        success) mkdir -p deliverables; printf 'artifact\n' > deliverables/result.txt ;;
                        no-deliverable) printf 'stdout\n' ;;
                        nonzero) printf 'bad\n' >&2; exit 8 ;;
                    esac
                fi
                """,
            )
            executor = ClaudeCodeExecutor(command=str(command))
            executor._version = "claude fake"
            no_deliverable = executor.execute(_execution_request(root, {"FAKE_MODE": "no-deliverable"}))
            self.assertEqual(no_deliverable.status, ExecutionStatus.NO_DELIVERABLE)
            completed = executor.execute(_execution_request(root, {"FAKE_MODE": "success"}))
            self.assertEqual(completed.status, ExecutionStatus.COMPLETED)
            failed = executor.execute(_execution_request(root, {"FAKE_MODE": "nonzero"}))
            self.assertEqual(failed.status, ExecutionStatus.FAILED)
            self.assertEqual(failed.exit_code, 8)

            request = _execution_request(root)
            with patch("eval_harness.executors.claude_code.subprocess.run", side_effect=_raise_timeout):
                timed_out = executor.execute(request)
            self.assertEqual(timed_out.status, ExecutionStatus.TIMED_OUT)
            self.assertEqual((request.executor_dir / "stdout.log").read_text(), "partial �")

            with patch("eval_harness.executors.claude_code.subprocess.run", side_effect=_raise_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    executor.execute(request)
            failed_executor = ClaudeCodeExecutor(command=str(root / "missing"))
            result = failed_executor.execute(request)
            self.assertEqual(result.status, ExecutionStatus.FAILED)
            self.assertIn("No such file or directory", (request.executor_dir / "stderr.log").read_text())

    def test_cursor_preflight_rejects_auth_modes_and_accepts_account(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = CursorExecutor(command=str(root / "missing"))
            self.assertFalse(missing.preflight().ok)
            command = _write_command(
                root,
                "agent",
                """
                if [ "$1" = "--version" ]; then echo 'agent fake'; exit 0; fi
                if [ "$1" = "status" ]; then
                    case "${FAKE_AUTH:-account}" in
                        account) echo 'Authenticated account';;
                        api) echo 'API key configured';;
                        negative) echo 'Not authenticated'; exit 1;;
                        empty) exit 0;;
                    esac
                fi
                """,
            )
            executor = CursorExecutor(command=str(command))
            for mode, expected_mode in (("api", "api"), ("empty", "unknown")):
                with patch.dict(os.environ, {"FAKE_AUTH": mode}, clear=False):
                    result = executor.preflight()
                self.assertFalse(result.ok)
                self.assertEqual(result.auth_mode, expected_mode)
            with patch.dict(os.environ, {"FAKE_AUTH": "negative"}, clear=False):
                negative = executor.preflight()
            self.assertFalse(negative.ok)
            with patch.dict(os.environ, {"FAKE_AUTH": "account"}, clear=False):
                account = executor.preflight()
            self.assertTrue(account.ok)
            self.assertEqual(account.auth_mode, "cursor-account")

    def test_cursor_execution_restores_references_and_fails_on_mutation_or_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _write_command(
                root,
                "agent-run",
                """
                if [ "$1" = "--version" ]; then echo 'agent fake'; exit 0; fi
                if [ "$1" = "-p" ]; then
                    case "${FAKE_MODE:-success}" in
                        success) mkdir -p deliverables; printf 'artifact\n' > deliverables/result.txt ;;
                        mutate) printf 'tampered\n' > reference_files/input.txt ;;
                        nonzero) printf 'failed\n' >&2; exit 9 ;;
                    esac
                fi
                """,
            )
            executor = CursorExecutor(command=str(command))
            executor._version = "agent fake"
            request = _execution_request(root, {"FAKE_MODE": "success"})
            (request.workspace / "reference_files").mkdir(parents=True)
            (request.workspace / "reference_files" / "input.txt").write_text("source\n", encoding="utf-8")
            success = executor.execute(request)
            self.assertEqual(success.status, ExecutionStatus.COMPLETED)
            self.assertTrue(success.metadata["reference_integrity_verified"])
            self.assertEqual((request.workspace / "reference_files" / "input.txt").read_text(), "source\n")

            mutated = executor.execute(_execution_request(root, {"FAKE_MODE": "mutate"}))
            self.assertEqual(mutated.status, ExecutionStatus.FAILED)
            self.assertFalse(mutated.metadata["reference_integrity_verified"])
            self.assertIn("mutation", (root / "executor" / "stderr.log").read_text())

            no_reference = request.workspace / "no-reference"
            no_reference.mkdir()
            self.assertEqual(executor._isolate_reference_files(no_reference), (None, None))
            self.assertEqual(executor._restore_reference_files(no_reference, None), None)

            timeout_request = _execution_request(root)
            with patch("eval_harness.executors.cursor.subprocess.run", side_effect=_raise_timeout):
                timed_out = executor.execute(timeout_request)
            self.assertEqual(timed_out.status, ExecutionStatus.TIMED_OUT)
            failed_executor = CursorExecutor(command=str(root / "missing"))
            failed = failed_executor.execute(timeout_request)
            self.assertEqual(failed.status, ExecutionStatus.FAILED)


class JudgeAdapterFailureTests(unittest.TestCase):
    def test_codex_judge_policy_helpers_cover_runtime_boundaries_without_model_calls(self) -> None:
        from eval_harness.judges import codex as codex_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_bin = root / "runtime" / "bin"
            runtime_bin.mkdir(parents=True)
            command = runtime_bin / "codex"
            command.write_text("#!/bin/sh\n", encoding="utf-8")
            command.chmod(0o755)
            environment = {"PATH": str(runtime_bin), "HOME": str(root / "home")}
            paths = codex_module._resolved_command_read_paths("codex", environment)
            self.assertIn(str(command), paths)
            self.assertIn(str(runtime_bin.parent), paths)
            protected = codex_module._protected_read_roots(environment)
            self.assertIn((root / "home").resolve(), protected)
            protected_environment = {"PATH": str(runtime_bin), "HOME": str(runtime_bin.parent)}
            protected_paths = codex_module._resolved_command_read_paths("codex", protected_environment)
            self.assertNotIn(str(runtime_bin.parent), protected_paths)
            self.assertEqual(codex_module._resolved_command_read_paths("missing", environment), ())

            judge = CodexJudgeExecutor(command=str(command), reasoning_effort="low")
            with patch.object(os, "name", "nt"):
                ok, detail = judge._probe_read_confinement(environment)
            self.assertFalse(ok)
            self.assertIn("Windows", detail)
            with patch(
                "eval_harness.judges.codex.subprocess.run",
                side_effect=OSError("probe unavailable"),
            ):
                ok, detail = judge._probe_read_confinement(environment)
            self.assertFalse(ok)
            self.assertIn("could not verify", detail)
            with patch(
                "eval_harness.judges.codex.subprocess.run",
                return_value=subprocess.CompletedProcess([], 3, stdout="", stderr="denied"),
            ):
                ok, detail = judge._probe_read_confinement(environment)
            self.assertFalse(ok)
            self.assertIn("probe failed", detail)

            request = _judge_request(root)
            request = JudgeRequest(
                task_id=request.task_id,
                task_prompt=request.task_prompt,
                workspace=request.workspace,
                reference_dir=request.reference_dir,
                submission_a_dir=request.submission_a_dir,
                submission_b_dir=request.submission_b_dir,
                executor_dir=request.executor_dir,
                trial_index=request.trial_index,
                swapped=request.swapped,
                model=None,
                environment=environment,
            )
            command_without_model = judge.build_command(request)
            self.assertNotIn("--model", command_without_model)
            model_request = JudgeRequest(
                task_id=request.task_id,
                task_prompt=request.task_prompt,
                workspace=request.workspace,
                reference_dir=request.reference_dir,
                submission_a_dir=request.submission_a_dir,
                submission_b_dir=request.submission_b_dir,
                executor_dir=request.executor_dir,
                trial_index=request.trial_index,
                swapped=request.swapped,
                model="judge-model",
                environment=environment,
            )
            command_with_model = judge.build_command(model_request)
            self.assertIn('model_reasoning_effort="low"', command_with_model)
            self.assertIn("--model", command_with_model)

            log = root / "log"
            codex_module._normalize_utf8_log(log)
            self.assertFalse(log.exists())
            log.write_bytes(b"valid\n")
            codex_module._normalize_utf8_log(log)
            self.assertEqual(log.read_bytes(), b"valid\n")
            log.write_bytes(b"bad \xff\n")
            codex_module._normalize_utf8_log(log)
            self.assertEqual(log.read_text(encoding="utf-8"), "bad �\n")

            with patch.object(Path, "write_bytes", side_effect=OSError("read-only")):
                codex_module._normalize_utf8_log(log)

    def test_codex_judge_keyboard_interrupt_keeps_durable_logs_and_version_probe_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            judge = CodexJudgeExecutor(command="codex")
            request = _judge_request(root)
            with patch("eval_harness.judges.codex.subprocess.run", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    judge.judge(request)
            self.assertTrue(request.executor_dir.joinpath("stdout.log").is_file())
            with patch(
                "eval_harness.judges.codex.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="version"),
            ) as run:
                self.assertEqual(judge._version_with_environment({"PATH": "/bin"}), "version")
            self.assertEqual(run.call_args.kwargs["errors"], "replace")
            with patch("eval_harness.judges.codex.subprocess.run", side_effect=subprocess.TimeoutExpired([], 1)):
                self.assertIsNone(judge._version_with_environment({"PATH": "/bin"}))

    def test_codex_judge_normalizes_invalid_utf8_logs_and_keeps_nonzero_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            judge = CodexJudgeExecutor(command="codex")
            judge._version = "fake"
            request = _judge_request(root)

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
                stdout = cast(BinaryIO, kwargs["stdout"])
                stderr = cast(BinaryIO, kwargs["stderr"])
                stdout.write(b"partial \xff stdout\n")
                stderr.write(b"partial \xfe stderr\n")
                stdout.flush()
                stderr.flush()
                return subprocess.CompletedProcess(command, 9)

            with patch("eval_harness.judges.codex.subprocess.run", side_effect=fake_run):
                result = judge.judge(request)
            self.assertIsNone(result.verdict)
            self.assertEqual(result.exit_code, 9)
            self.assertEqual(request.executor_dir.joinpath("stdout.log").read_text(), "partial � stdout\n")
            self.assertEqual(request.executor_dir.joinpath("stderr.log").read_text(), "partial � stderr\n")
            self.assertIsNone(result.metadata["parse_error"])

    def test_codex_judge_preflight_status_probe_errors_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _write_command(root, "codex", "exit 0\n")
            judge = CodexJudgeExecutor(command=str(command))
            with patch(
                "eval_harness.judges.codex.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="fake", stderr=""),
                    subprocess.TimeoutExpired(["codex", "login", "status"], 15),
                ],
            ):
                timed_out = judge.preflight({"PATH": str(root)})
            self.assertFalse(timed_out.ok)
            self.assertIn("could not inspect", timed_out.details[0])

            with patch(
                "eval_harness.judges.codex.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="fake", stderr=""),
                    subprocess.CompletedProcess([], 1, stdout="", stderr=""),
                ],
            ):
                logged_out = judge.preflight({"PATH": str(root)})
            self.assertFalse(logged_out.ok)
            self.assertEqual(logged_out.details, ("Codex is not logged in",))

    def test_codex_judge_preflight_uses_fake_sandbox_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _write_command(
                root,
                "codex",
                """
                if [ "$1" = "--version" ]; then echo 'codex judge'; exit 0; fi
                if [ "$1" = "login" ]; then printf '%s\n' "${FAKE_AUTH:-ChatGPT subscription}"; exit "${FAKE_EXIT:-0}"; fi
                for argument in "$@"; do
                    if [ "$argument" = "sandbox" ]; then exit "${FAKE_SANDBOX:-0}"; fi
                done
                exit 0
                """,
            )
            judge = CodexJudgeExecutor(command=str(command))
            api = judge.preflight({"PATH": str(root), "FAKE_AUTH": "api key", "FAKE_EXIT": "0"})
            self.assertFalse(api.ok)
            self.assertEqual(api.auth_mode, "api")
            unknown = judge.preflight({"PATH": str(root), "FAKE_AUTH": "unknown", "FAKE_EXIT": "0"})
            self.assertFalse(unknown.ok)
            self.assertEqual(unknown.auth_mode, "unknown")
            sandbox_failed = judge.preflight(
                {"PATH": str(root), "FAKE_AUTH": "ChatGPT", "FAKE_SANDBOX": "9", "FAKE_EXIT": "0"}
            )
            self.assertFalse(sandbox_failed.ok)
            self.assertIn("sandbox probe failed", sandbox_failed.details[1])
            ready = judge.preflight({"PATH": str(root), "FAKE_AUTH": "ChatGPT", "FAKE_SANDBOX": "0", "FAKE_EXIT": "0"})
            self.assertTrue(ready.ok)
            self.assertEqual(ready.auth_mode, "chatgpt-subscription")

    def test_codex_judge_persists_verdict_parse_failure_timeout_and_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _write_command(
                root,
                "codex",
                """
                if [ "$1" = "exec" ]; then
                    output=''
                    while [ "$#" -gt 0 ]; do
                        if [ "$1" = "--output-last-message" ]; then output="$2"; shift; fi
                        shift
                    done
                    printf '%s\n' "${FAKE_VERDICT:-BOXED[A]}"
                    printf '%s\n' "${FAKE_VERDICT:-BOXED[A]}" > "$output"
                    exit "${FAKE_EXIT:-0}"
                fi
                if [ "$1" = "--version" ]; then echo fake; fi
                """,
            )
            judge = CodexJudgeExecutor(command=str(command))
            judge._version = "fake"
            valid = judge.judge(_judge_request(root, {"FAKE_VERDICT": "BOXED[A]"}))
            assert valid.verdict is not None
            self.assertEqual(valid.verdict.value, "A")
            self.assertEqual(valid.exit_code, 0)

            invalid_request = _judge_request(root, {"FAKE_VERDICT": "not a verdict"})
            invalid = judge.judge(invalid_request)
            self.assertIsNone(invalid.verdict)
            self.assertIn("standalone BOXED", str(invalid.metadata["parse_error"]))

            failed_request = _judge_request(root, {"FAKE_VERDICT": "BOXED[B]", "FAKE_EXIT": "3"})
            failed = judge.judge(failed_request)
            self.assertEqual(failed.exit_code, 3)
            self.assertIsNone(failed.verdict)

            timeout_request = _judge_request(root)
            with patch("eval_harness.judges.codex.subprocess.run", side_effect=_raise_timeout):
                timed_out = judge.judge(timeout_request)
            self.assertIsNone(timed_out.verdict)
            self.assertEqual(timed_out.metadata["parse_error"], "judge timed out")

            with patch("eval_harness.judges.codex.subprocess.run", side_effect=_raise_os_error):
                os_error = judge.judge(_judge_request(root))
            self.assertIn("fake command unavailable", str(os_error.metadata["parse_error"]))
            self.assertIn("fake command unavailable", os_error.stderr_path.read_text())

    def test_claude_judge_preflight_and_judge_failure_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _write_command(
                root,
                "claude",
                """
                if [ "$1" = "--version" ]; then echo 'claude judge'; exit 0; fi
                if [ "$1" = "auth" ]; then
                    if [ "${FAKE_AUTH:-valid}" = "invalid" ]; then echo bad; exit 0; fi
                    echo '{\"loggedIn\":true,\"authMethod\":\"claude.ai\",\"apiProvider\":\"firstparty\",\"subscriptionType\":\"pro\"}'
                    exit "${FAKE_EXIT:-0}"
                fi
                if [ "$1" = "-p" ]; then printf '%s\n' "${FAKE_VERDICT:-BOXED[TIE]}"; fi
                """,
            )
            judge = ClaudeCodeJudgeExecutor(command=str(command))
            invalid = judge.preflight({"PATH": str(root), "FAKE_AUTH": "invalid"})
            self.assertFalse(invalid.ok)
            self.assertEqual(invalid.auth_mode, "unknown")
            ready = judge.preflight({"PATH": str(root), "FAKE_AUTH": "valid"})
            self.assertFalse(ready.ok)
            self.assertIn("disabled", ready.details[1])

            judge._version = "fake"
            valid = judge.judge(_judge_request(root, {"FAKE_VERDICT": "BOXED[TIE]"}))
            assert valid.verdict is not None
            self.assertEqual(valid.verdict.value, "TIE")
            invalid_request = _judge_request(root, {"FAKE_VERDICT": "garbage"})
            invalid_result = judge.judge(invalid_request)
            self.assertIsNone(invalid_result.verdict)
            self.assertIn("standalone BOXED", str(invalid_result.metadata["parse_error"]))
            timeout_request = _judge_request(root)
            with patch("eval_harness.judges.claude_code.subprocess.run", side_effect=_raise_timeout):
                timeout = judge.judge(timeout_request)
            self.assertEqual(timeout.metadata["parse_error"], "judge timed out")
            with patch("eval_harness.judges.claude_code.subprocess.run", side_effect=_raise_os_error):
                os_error = judge.judge(_judge_request(root))
            self.assertIn("fake command unavailable", str(os_error.metadata["parse_error"]))

    def test_claude_judge_timeout_persists_replacement_decoded_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            judge = ClaudeCodeJudgeExecutor(command="claude")
            judge._version = "fake"
            request = _judge_request(root)
            with patch("eval_harness.judges.claude_code.subprocess.run", side_effect=_raise_timeout) as run:
                result = judge.judge(request)
            self.assertEqual(run.call_args.kwargs["errors"], "replace")
            self.assertIsNone(result.verdict)
            self.assertEqual(result.metadata["parse_error"], "judge timed out")
            self.assertEqual(request.executor_dir.joinpath("stdout.log").read_text(), "partial �")
            self.assertEqual(request.executor_dir.joinpath("stderr.log").read_text(), "error �")

    def test_claude_judge_preflight_platform_and_probe_failures_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = ClaudeCodeJudgeExecutor(command=str(root / "missing"))
            self.assertFalse(missing.preflight({"PATH": str(root)}).ok)
            command = _write_command(root, "claude", "exit 0\n")
            judge = ClaudeCodeJudgeExecutor(command=str(command))
            with patch.object(os, "name", "nt"):
                windows = judge.preflight({"PATH": str(root)})
            self.assertFalse(windows.ok)
            self.assertIn("Windows", windows.details[0])
            with patch(
                "eval_harness.judges.claude_code.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="version", stderr=""),
                    subprocess.TimeoutExpired(["claude", "auth", "status"], 15),
                ],
            ):
                timed_out = judge.preflight({"PATH": str(root)})
            self.assertFalse(timed_out.ok)
            self.assertIn("could not inspect", timed_out.details[0])
            with patch(
                "eval_harness.judges.claude_code.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="version", stderr=""),
                    subprocess.CompletedProcess([], 1, stdout="", stderr=""),
                ],
            ):
                logged_out = judge.preflight({"PATH": str(root)})
            self.assertFalse(logged_out.ok)
            self.assertEqual(logged_out.details, ("Claude Code is not logged in",))

            request = _judge_request(root)
            request = JudgeRequest(
                task_id=request.task_id,
                task_prompt=request.task_prompt,
                workspace=request.workspace,
                reference_dir=request.reference_dir,
                submission_a_dir=request.submission_a_dir,
                submission_b_dir=request.submission_b_dir,
                executor_dir=request.executor_dir,
                trial_index=request.trial_index,
                swapped=request.swapped,
                model=None,
            )
            self.assertNotIn("--model", judge.build_command(request))
            judge._version = "fake"
            with patch("eval_harness.judges.claude_code.subprocess.run", side_effect=_raise_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    judge.judge(request)
            self.assertTrue(request.executor_dir.joinpath("stdout.log").is_file())

    def test_judge_output_parser_rejects_empty_and_duplicate_verdicts(self) -> None:
        for text in ("", "reasoning", "BOXED[A]\nBOXED[B]", "BOXED[A]\nreasoning"):
            with self.assertRaises(ValueError):
                parse_verdict(text)
        self.assertEqual(parse_verdict("reasoning\nboxed[tie]").value, "TIE")


if __name__ == "__main__":
    unittest.main()
