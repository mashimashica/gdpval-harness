# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import patch

from eval_harness.judges.base import JudgeRequest
from eval_harness.judges.claude_code import ClaudeCodeJudgeExecutor
from eval_harness.judges.codex import CodexJudgeExecutor


class _BinaryStream(Protocol):
    def write(self, data: bytes) -> int: ...

    def flush(self) -> None: ...


class LocalJudgeExecutorTests(unittest.TestCase):
    def request(self, root: str, *, environment: dict[str, str] | None = None) -> JudgeRequest:
        base = Path(root)
        workspace = base / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return JudgeRequest(
            task_id="x",
            task_prompt="judge task",
            workspace=workspace,
            reference_dir=workspace / "reference_files",
            submission_a_dir=workspace / "submission_a",
            submission_b_dir=workspace / "submission_b",
            executor_dir=base / "executor",
            trial_index=0,
            swapped=False,
            model="example-model",
            environment=environment or {},
        )

    def test_codex_judge_uses_root_deny_workspace_read_profile(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            runtime_root = base / "runtime"
            runtime_bin = runtime_root / "bin"
            runtime_bin.mkdir(parents=True)
            command_path = runtime_bin / "codex"
            command_path.write_text("#!/bin/sh\n", encoding="utf-8")
            command_path.chmod(0o755)
            request = self.request(root, environment={"PATH": str(runtime_bin)})
            command = CodexJudgeExecutor(command=str(command_path)).build_command(request)
            self.assertEqual(command[:2], [str(command_path), "exec"])
            self.assertNotIn("--sandbox", command)
            self.assertIn("--ephemeral", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn('approval_policy="never"', command)
            self.assertIn('default_permissions="gdpval-harness-blind-judge"', command)
            profile = next(item for item in command if item.startswith("permissions.gdpval-harness-blind-judge="))
            self.assertIn('":root"="deny"', profile)
            self.assertIn('":minimal"="read"', profile)
            self.assertIn(json.dumps(str(request.workspace.resolve())), profile)
            self.assertIn(json.dumps(str(command_path.resolve())), profile)
            self.assertIn(json.dumps(str(runtime_root.resolve())), profile)
            self.assertIn("network={enabled=false}", profile)
            self.assertNotIn("cloud", command)
            self.assertEqual(command[-1], "-")

    def test_codex_runtime_root_does_not_reopen_auth_home(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            auth_home = base / "home"
            runtime_bin = auth_home / "bin"
            runtime_bin.mkdir(parents=True)
            command_path = runtime_bin / "codex"
            command_path.write_text("#!/bin/sh\n", encoding="utf-8")
            command_path.chmod(0o755)
            request = self.request(root, environment={"PATH": str(runtime_bin), "HOME": str(auth_home)})
            command = CodexJudgeExecutor(command=str(command_path)).build_command(request)
            profile = next(item for item in command if item.startswith("permissions.gdpval-harness-blind-judge="))
            self.assertIn(json.dumps(str(command_path.resolve())), profile)
            self.assertNotIn(f'{json.dumps(str(auth_home.resolve()))}="read"', profile)

    def test_codex_runtime_root_does_not_reopen_candidate_or_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            runtime_root = base / "shared"
            runtime_bin = runtime_root / "bin"
            runtime_bin.mkdir(parents=True)
            command_path = runtime_bin / "codex"
            command_path.write_text("#!/bin/sh\n", encoding="utf-8")
            command_path.chmod(0o755)
            candidate = runtime_root / "candidate"
            out = runtime_root / "out"
            home = base / "home"
            candidate.mkdir()
            out.mkdir()
            home.mkdir()
            request = self.request(root, environment={"PATH": str(runtime_bin), "HOME": str(home)})
            with patch.dict(
                os.environ,
                {
                    "GDPVAL_RUN_A": str(candidate),
                    "GDPVAL_RUN_B": str(base / "other-candidate"),
                    "OUT": str(out),
                },
                clear=False,
            ):
                command = CodexJudgeExecutor(command=str(command_path)).build_command(request)
            profile = next(item for item in command if item.startswith("permissions.gdpval-harness-blind-judge="))
            self.assertIn(json.dumps(str(command_path.resolve())), profile)
            self.assertNotIn(f'{json.dumps(str(runtime_root.resolve()))}="read"', profile)

    def test_codex_judge_shell_environment_excludes_parent_and_proxy_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            command_path = base / "codex"
            command_path.write_text("#!/bin/sh\n", encoding="utf-8")
            command_path.chmod(0o755)
            runtime_tmp = base / "runtime-tmp"
            runtime_tmp.mkdir()
            request = self.request(
                root,
                environment={
                    "PATH": "/candidate/identity/bin",
                    "HOME": "/real/home",
                    "HTTPS_PROXY": "http://alice:sekrit@proxy.example:8080",  # pragma: allowlist secret
                    "GDPVAL_LABEL_A": "baseline-secret",
                    "TMPDIR": str(runtime_tmp),
                    "TMP": str(runtime_tmp),
                    "TEMP": str(runtime_tmp),
                },
            )
            command = CodexJudgeExecutor(command=str(command_path)).build_command(request)
            self.assertIn("allow_login_shell=false", command)
            policy = next(item for item in command if item.startswith("shell_environment_policy="))
            self.assertIn('inherit="none"', policy)
            self.assertIn(json.dumps(str(request.workspace.resolve())), policy)
            self.assertIn(json.dumps(str(runtime_tmp)), policy)
            self.assertNotIn("proxy", policy.lower())
            self.assertNotIn("sekrit", policy)
            self.assertNotIn("/real/home", policy)
            self.assertNotIn("/candidate/identity/bin", policy)
            self.assertNotIn("baseline-secret", policy)

    def test_codex_judge_preserves_partial_subprocess_output_on_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            command_path = base / "codex"
            command_path.write_text("#!/bin/sh\n", encoding="utf-8")
            command_path.chmod(0o755)
            request = self.request(root, environment={"PATH": str(base)})
            judge = CodexJudgeExecutor(command=str(command_path))

            def interrupt_after_output(*args: object, **kwargs: object) -> None:
                del args
                stdout = cast(_BinaryStream, kwargs["stdout"])
                stderr = cast(_BinaryStream, kwargs["stderr"])
                stdout.write(b"partial \xff stdout\n")
                stderr.write(b"partial \xff stderr\n")
                stdout.flush()
                stderr.flush()
                raise KeyboardInterrupt

            with (
                patch("eval_harness.judges.codex.subprocess.run", side_effect=interrupt_after_output),
                self.assertRaises(KeyboardInterrupt),
            ):
                judge.judge(request)

            self.assertEqual(
                (request.executor_dir / "stdout.log").read_text(encoding="utf-8"),
                "partial \ufffd stdout\n",
            )
            self.assertEqual(
                (request.executor_dir / "stderr.log").read_text(encoding="utf-8"),
                "partial \ufffd stderr\n",
            )

    def test_claude_judge_restricts_reads_tools_network_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            request = self.request(root)
            command = ClaudeCodeJudgeExecutor(max_turns=10).build_command(request)
            self.assertEqual(command[:2], ["claude", "-p"])
            self.assertIn("--safe-mode", command)
            self.assertIn("--tools", command)
            self.assertEqual(command[command.index("--tools") + 1], "Bash")
            self.assertIn("--max-turns", command)
            self.assertEqual(command[command.index("--max-turns") + 1], "10")
            settings = json.loads(command[command.index("--settings") + 1])
            sandbox = settings["sandbox"]
            self.assertTrue(sandbox["failIfUnavailable"])
            self.assertFalse(sandbox["allowUnsandboxedCommands"])
            self.assertEqual(sandbox["network"]["allowedDomains"], [])
            self.assertEqual(sandbox["filesystem"]["denyRead"], ["/"])
            self.assertEqual(sandbox["filesystem"]["allowRead"], [str(request.workspace.resolve())])
            self.assertEqual(sandbox["filesystem"]["denyWrite"], ["/"])
            disallowed_index = command.index("--disallowedTools")
            self.assertIn("Read", command[disallowed_index + 1 :])
            self.assertNotIn("--cloud", command)


if __name__ == "__main__":
    unittest.main()
