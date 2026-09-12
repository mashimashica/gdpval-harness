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

    def test_execute_collects_authoritative_final_message_after_valid_jsonl(self) -> None:
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
                output_path.write_text("final \\boxed{42}\n", encoding="utf-8")
                stdout = (
                    '{"type":"thread.started","thread_id":"thread"}\n'
                    '{"type":"turn.started"}\n'
                    '{"type":"item.completed","item":{"id":"item","type":"agent_message",'
                    '"text":"intermediate"}}\n'
                    '{"type":"turn.completed","usage":{"input_tokens":0,"cached_input_tokens":0,'
                    '"output_tokens":0,"reasoning_output_tokens":0}}\n'
                )
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            with patch("eval_harness.executors.codex.subprocess.run", side_effect=fake_run):
                result = executor.execute(request)

            self.assertEqual(result.output_text, "final \\boxed{42}\n")
            self.assertEqual(result.metadata["output_text_source"], "executor/final-message.txt")
            self.assertEqual(result.runtime, "host-subprocess")
            self.assertIsNone(result.model_id)
            self.assertIsNone(result.effective_reasoning_effort)
            self.assertFalse(result.effective_reasoning_effort_available)


if __name__ == "__main__":
    unittest.main()
