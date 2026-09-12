# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from gdpval_harness.executors.base import ExecutionRequest, TaskSpec
from gdpval_harness.executors.cursor import CursorExecutor, subscription_environment


class CursorExecutorTests(unittest.TestCase):
    def request(self, root: str) -> ExecutionRequest:
        base = Path(root)
        workspace = base / "workspace"
        deliverables = workspace / "deliverables"
        executor_dir = base / "executor"
        workspace.mkdir(parents=True, exist_ok=True)
        deliverables.mkdir(parents=True, exist_ok=True)
        executor_dir.mkdir(parents=True, exist_ok=True)
        return ExecutionRequest(
            task=TaskSpec(task_id="t", prompt="real task text"),
            workspace=workspace,
            deliverables_dir=deliverables,
            executor_dir=executor_dir,
            model="example-model",
        )

    def test_command_is_headless_local_workspace_and_sandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            request = self.request(root)
            command = CursorExecutor(network_enabled=False).build_command(request)
            self.assertEqual(command[:2], ["agent", "-p"])
            self.assertIn("--trust", command)
            self.assertNotIn("--force", command)
            self.assertIn("--workspace", command)
            self.assertEqual(command[command.index("--workspace") + 1], str(request.workspace))
            self.assertIn("--output-format", command)
            self.assertEqual(command[command.index("--output-format") + 1], "json")
            self.assertIn("--sandbox", command)
            self.assertEqual(command[command.index("--sandbox") + 1], "enabled")
            self.assertIn("--model", command)
            self.assertNotIn("real task text", command)
            self.assertNotIn("--api-key", command)
            self.assertNotIn("--auth-token", command)
            self.assertNotIn("--worktree", command)

    def test_workspace_policy_denies_network_mcp_and_adds_readonly_reference_path(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root) / "workspace"
            readonly = Path(root) / "readonly-refs"
            workspace.mkdir()
            readonly.mkdir()
            CursorExecutor(network_enabled=False)._write_workspace_policy(workspace, readonly)
            sandbox = json.loads((workspace / ".cursor" / "sandbox.json").read_text())
            config = json.loads((workspace / ".cursor" / "cli.json").read_text())
            self.assertEqual(sandbox["type"], "workspace_readwrite")
            self.assertEqual(sandbox["networkPolicy"]["default"], "deny")
            self.assertTrue(sandbox["disableTmpWrite"])
            self.assertIn(str(readonly.resolve()), sandbox["additionalReadonlyPaths"])
            self.assertIn("Mcp(*:*)", config["permissions"]["deny"])
            self.assertIn("WebFetch(*)", config["permissions"]["deny"])
            self.assertIn("Write(reference_files/**)", config["permissions"]["deny"])
            self.assertIn("Shell(*)", config["permissions"]["allow"])

    def test_reference_tree_is_moved_outside_workspace_then_restored(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root) / "task" / "workspace"
            refs = workspace / "reference_files"
            refs.mkdir(parents=True)
            (refs / "input.txt").write_text("original\n", encoding="utf-8")
            executor = CursorExecutor()
            protected, digest = executor._isolate_reference_files(workspace)
            self.assertIsNotNone(protected)
            self.assertIsNotNone(digest)
            self.assertTrue(refs.is_symlink())
            self.assertFalse(str(protected).startswith(str(workspace) + os.sep))
            executor._restore_reference_files(workspace, protected)
            self.assertFalse(refs.is_symlink())
            self.assertEqual((refs / "input.txt").read_text(encoding="utf-8"), "original\n")

    def test_subscription_environment_removes_api_auth(self) -> None:
        env = subscription_environment(
            {
                "PATH": "/bin",
                "CURSOR_API_KEY": "secret",
                "CURSOR_AUTH_TOKEN": "token",
                "KEEP_ME": "yes",
            }
        )
        self.assertNotIn("CURSOR_API_KEY", env)
        self.assertNotIn("CURSOR_AUTH_TOKEN", env)
        self.assertEqual(env["KEEP_ME"], "yes")


if __name__ == "__main__":
    unittest.main()
