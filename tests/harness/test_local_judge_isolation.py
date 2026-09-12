# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_harness.judges.claude_code import ClaudeCodeJudgeExecutor
from eval_harness.judges.codex import CodexJudgeExecutor
from eval_harness.local_judge_runner import _judge_environment, _paths_overlap, _safe_temp_parent


class LocalJudgeIsolationTests(unittest.TestCase):
    def _fake_cli(self, root: Path, name: str) -> Path:
        path = root / name
        path.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
for name in GDPVAL_RUN_A GDPVAL_RUN_B GDPVAL_LABEL_A GDPVAL_LABEL_B FORBIDDEN_PARENT_SECRET; do
  if [[ -n \"${!name:-}\" ]]; then
    echo \"provenance leaked: $name\" >&2
    exit 91
  fi
done
while [[ \"${1-}\" == \"-c\" ]]; do
  shift 2
done
case \"${1-}\" in
  --version) echo \"fake-version\" ;;
  login) echo \"Logged in using ChatGPT\" ;;
  auth) echo '{\"loggedIn\":true,\"authMethod\":\"claude.ai\",\"apiProvider\":\"firstParty\",\"subscriptionType\":\"max\"}' ;;
  sandbox) exit 0 ;;
  *) exit 92 ;;
esac
""",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def test_codex_preflight_uses_provenance_free_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = self._fake_cli(root, "codex")
            with patch.dict(
                os.environ,
                {
                    "GDPVAL_RUN_A": "/secret/a",
                    "GDPVAL_RUN_B": "/secret/b",
                    "GDPVAL_LABEL_A": "baseline",
                    "GDPVAL_LABEL_B": "intervention",
                    "FORBIDDEN_PARENT_SECRET": "secret",
                },
                clear=False,
            ):
                result = CodexJudgeExecutor(command=str(command)).preflight(_judge_environment())
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.auth_mode, "chatgpt-subscription")

    def test_claude_preflight_fails_closed_without_non_model_read_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = self._fake_cli(root, "claude")
            with patch.dict(
                os.environ,
                {
                    "GDPVAL_RUN_A": "/secret/a",
                    "GDPVAL_RUN_B": "/secret/b",
                    "GDPVAL_LABEL_A": "baseline",
                    "GDPVAL_LABEL_B": "intervention",
                    "FORBIDDEN_PARENT_SECRET": "secret",
                },
                clear=False,
            ):
                result = ClaudeCodeJudgeExecutor(command=str(command)).preflight(_judge_environment())
            self.assertFalse(result.ok)
            self.assertTrue((result.auth_mode or "").startswith("claude-subscription:"))
            self.assertTrue(any("read confinement" in detail for detail in result.details))

    @unittest.skipIf(os.name == "nt", "POSIX system-temp candidate ordering test")
    def test_safe_temp_parent_rejects_caller_temp_inside_candidate_tree(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            candidate_a = root / "candidate-a"
            candidate_b = root / "candidate-b"
            out_dir = root / "out"
            for path in (candidate_a, candidate_b, out_dir):
                path.mkdir()
            with patch.dict(
                os.environ, {"TMPDIR": str(candidate_a), "TMP": str(candidate_a), "TEMP": str(candidate_a)}
            ):
                parent = _safe_temp_parent(candidate_a, candidate_b, out_dir)
            self.assertFalse(_paths_overlap(parent, candidate_a))
            self.assertFalse(_paths_overlap(parent, candidate_b))
            self.assertFalse(_paths_overlap(parent, out_dir))
            self.assertNotEqual(parent, candidate_a.resolve())

    def test_runtime_temp_overrides_do_not_copy_parent_temp_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_tmp = Path(tmp) / "runtime"
            runtime_tmp.mkdir()
            with patch.dict(
                os.environ,
                {"TMPDIR": "/candidate/tmp", "TMP": "/candidate/tmp", "TEMP": "/candidate/tmp"},
                clear=False,
            ):
                env = _judge_environment(runtime_tmp)
            self.assertEqual(env["TMPDIR"], str(runtime_tmp))
            self.assertEqual(env["TMP"], str(runtime_tmp))
            self.assertEqual(env["TEMP"], str(runtime_tmp))


if __name__ == "__main__":
    unittest.main()
