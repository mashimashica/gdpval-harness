# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gdpval_harness.executors.base import ExecutionRequest, TaskSpec
from gdpval_harness.executors.codex import CodexExecutor


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
            self.assertIn("workspace-write", command)
            self.assertIn("sandbox_workspace_write.network_access=false", command)
            self.assertNotIn("--cloud", command)
            self.assertNotIn("--yolo", command)
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


if __name__ == "__main__":
    unittest.main()
