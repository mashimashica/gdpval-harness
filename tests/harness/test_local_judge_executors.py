# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gdpval_harness.judges.base import JudgeRequest
from gdpval_harness.judges.claude_code import ClaudeCodeJudgeExecutor
from gdpval_harness.judges.codex import CodexJudgeExecutor


class LocalJudgeExecutorTests(unittest.TestCase):
    def request(self, root: str) -> JudgeRequest:
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
        )

    def test_codex_judge_uses_root_deny_workspace_read_profile(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            request = self.request(root)
            command = CodexJudgeExecutor().build_command(request)
            self.assertEqual(command[:2], ["codex", "exec"])
            self.assertNotIn("--sandbox", command)
            self.assertIn("--ephemeral", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn('approval_policy="never"', command)
            self.assertIn('default_permissions="gdpval-harness-blind-judge"', command)
            profile = next(
                item for item in command if item.startswith("permissions.gdpval-harness-blind-judge=")
            )
            self.assertIn('":root"="deny"', profile)
            self.assertIn('":minimal"="read"', profile)
            self.assertIn(json.dumps(str(request.workspace.resolve())), profile)
            self.assertIn('network={enabled=false}', profile)
            self.assertNotIn("cloud", command)
            self.assertEqual(command[-1], "-")

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
