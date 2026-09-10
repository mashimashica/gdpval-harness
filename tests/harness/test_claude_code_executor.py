# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gdpval_harness.executors.base import ExecutionRequest, TaskSpec
from gdpval_harness.executors.claude_code import ClaudeCodeExecutor, subscription_environment


class ClaudeCodeExecutorTests(unittest.TestCase):
    def request(self, root: str) -> ExecutionRequest:
        base = Path(root)
        return ExecutionRequest(
            task=TaskSpec(task_id="t", prompt="work"),
            workspace=base / "workspace",
            deliverables_dir=base / "workspace" / "deliverables",
            executor_dir=base / "executor",
            model="sonnet",
        )

    def test_command_uses_print_mode_and_fail_closed_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            command = ClaudeCodeExecutor(network_enabled=False, max_turns=7).build_command(self.request(root))
            self.assertEqual(command[:2], ["claude", "-p"])
            self.assertIn("--safe-mode", command)
            self.assertIn("--no-session-persistence", command)
            self.assertIn("--output-format", command)
            self.assertIn("json", command)
            self.assertIn("--permission-mode", command)
            self.assertIn("acceptEdits", command)
            self.assertIn("--tools", command)
            self.assertIn("Bash,Read,Edit,Write", command)
            self.assertIn("--strict-mcp-config", command)
            self.assertIn("--max-turns", command)
            self.assertNotIn("--cloud", command)
            self.assertNotIn("--environment", command)
            self.assertNotIn("--dangerously-skip-permissions", command)
            settings = json.loads(command[command.index("--settings") + 1])
            sandbox = settings["sandbox"]
            self.assertTrue(sandbox["enabled"])
            self.assertTrue(sandbox["failIfUnavailable"])
            self.assertFalse(sandbox["allowUnsandboxedCommands"])
            self.assertTrue(sandbox["autoAllowBashIfSandboxed"])
            self.assertTrue(sandbox["network"]["strictAllowlist"])
            self.assertEqual(sandbox["network"]["allowedDomains"], [])

    def test_network_enabled_keeps_sandbox_but_drops_empty_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            command = ClaudeCodeExecutor(network_enabled=True).build_command(self.request(root))
            settings = json.loads(command[command.index("--settings") + 1])
            self.assertTrue(settings["sandbox"]["enabled"])
            self.assertNotIn("network", settings["sandbox"])

    def test_subscription_environment_removes_api_and_cloud_routing(self) -> None:
        env = subscription_environment(
            {
                "PATH": "/bin",
                "ANTHROPIC_API_KEY": "secret",
                "ANTHROPIC_AUTH_TOKEN": "token",
                "CLAUDE_CODE_OAUTH_TOKEN": "oauth",
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "KEEP_ME": "yes",
            }
        )
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)
        self.assertNotIn("CLAUDE_CODE_USE_BEDROCK", env)
        self.assertEqual(env["KEEP_ME"], "yes")


if __name__ == "__main__":
    unittest.main()
