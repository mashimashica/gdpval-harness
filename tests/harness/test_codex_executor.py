# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_harness.executors.base import ExecutionRequest, TaskSpec
from eval_harness.executors.codex import CodexExecutor, subscription_environment


class CodexExecutorTests(unittest.TestCase):
    def test_command_is_local_ephemeral_and_workspace_sandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            request = ExecutionRequest(
                task=TaskSpec(task_id="t", prompt="work"),
                workspace=base / "workspace",
                deliverables_dir=base / "workspace" / "deliverables",
                executor_dir=base / "executor",
                model="example-model",
            )
            command = CodexExecutor(network_enabled=False).build_command(request)
            self.assertEqual(command[:2], ["codex", "exec"])
            self.assertIn("--ephemeral", command)
            self.assertIn("--json", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn("--skip-git-repo-check", command)
            self.assertIn("--color", command)
            self.assertIn("workspace-write", command)
            self.assertIn('approval_policy="never"', command)
            self.assertIn("sandbox_workspace_write.network_access=false", command)
            self.assertIn("shell_environment_policy.ignore_default_excludes=false", command)
            self.assertNotIn("cloud", command)
            self.assertNotIn("--yolo", command)
            self.assertNotIn("work", command)
            self.assertEqual(command[-1], "-")

    def test_network_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            request = ExecutionRequest(
                task=TaskSpec(task_id="t", prompt="work"),
                workspace=base,
                deliverables_dir=base / "deliverables",
                executor_dir=base / "executor",
            )
            command = CodexExecutor(network_enabled=True).build_command(request)
            self.assertIn("sandbox_workspace_write.network_access=true", command)

    def test_subscription_environment_removes_api_credentials(self) -> None:
        env = subscription_environment(
            {
                "PATH": "/bin",
                "OPENAI_API_KEY": "secret",
                "CODEX_ACCESS_TOKEN": "secret-token",
                "KEEP_ME": "yes",
            }
        )
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_ACCESS_TOKEN", env)
        self.assertEqual(env["KEEP_ME"], "yes")

    def test_execute_collects_final_message_as_output_text_with_replacement_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            request = ExecutionRequest(
                task=TaskSpec(task_id="t", prompt="solve"),
                workspace=base / "workspace",
                deliverables_dir=base / "workspace" / "deliverables",
                executor_dir=base / "executor",
            )
            executor = CodexExecutor(network_enabled=False)
            executor._version = "codex-test"

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                del kwargs
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"final \\boxed{42} \xff\n")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("eval_harness.executors.codex.subprocess.run", side_effect=fake_run):
                result = executor.execute(request)

            self.assertEqual(result.output_text, "final \\boxed{42} \ufffd\n")
            self.assertEqual(result.metadata["output_text_source"], "executor/final-message.txt")


if __name__ == "__main__":
    unittest.main()
