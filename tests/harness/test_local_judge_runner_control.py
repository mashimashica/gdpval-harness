# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gdpval_harness.judges.base import JudgeResult
import gdpval_harness.local_judge_runner as runner


class _InterruptJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, request):
        self.calls += 1
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        (request.executor_dir / "stdout.log").write_text("partial stdout\n", encoding="utf-8")
        (request.executor_dir / "stderr.log").write_text("partial stderr\n", encoding="utf-8")
        raise KeyboardInterrupt


class _FailingJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, request):
        self.calls += 1
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        stdout = request.executor_dir / "stdout.log"
        stderr = request.executor_dir / "stderr.log"
        stdout.write_text("transport stdout\n", encoding="utf-8")
        stderr.write_text("transport failure\n", encoding="utf-8")
        return JudgeResult(
            task_id=request.task_id,
            trial_index=request.trial_index,
            judge_executor="codex",
            verdict=None,
            executor_version="fake",
            invocation_mode="fake",
            auth_mode="chatgpt-subscription",
            started_at="start",
            finished_at="finish",
            exit_code=73,
            stdout_path=stdout,
            stderr_path=stderr,
            metadata={"parse_error": None},
        )


class LocalJudgeRunnerControlTests(unittest.TestCase):
    def _candidate(self, root: Path, name: str) -> Path:
        candidate = root / name
        repeat = candidate / "task_one" / "repeat_0"
        repeat.mkdir(parents=True)
        (repeat / "artifact.txt").write_text(name, encoding="utf-8")
        prompt_dir = candidate / "tasks" / "one" / "executor"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "prompt.txt").write_text(
            "wrapper\n\nTask:\nDo the task.\n",
            encoding="utf-8",
        )
        return candidate

    def _environment(self, root: Path, out: Path, a: Path, b: Path) -> dict[str, str]:
        return {
            "GDPVAL_RUN_A": str(a),
            "GDPVAL_RUN_B": str(b),
            "GDPVAL_JUDGE_EXECUTOR": "codex",
            "LIMIT": "1",
            "GDPVAL_JUDGE_TRIALS": "3",
            "OUT": str(out),
            "GDPVAL_WRITE_METADATA": "0",
        }

    def _preflight(self, root: Path) -> tuple[bool, dict[str, object]]:
        return True, {
            "judge_executor": "codex",
            "ok": True,
            "version": "fake",
            "auth_mode": "chatgpt-subscription",
            "temp_parent": str(root),
            "details": [],
        }

    def test_interrupt_persists_partial_logs_and_returns_130(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            a = self._candidate(root, "a")
            b = self._candidate(root, "b")
            judge = _InterruptJudge()
            with (
                patch.dict(os.environ, self._environment(root, out, a, b), clear=False),
                patch.object(runner, "_preflight", return_value=self._preflight(root)),
                patch.object(runner, "_ensure_dataset"),
                patch.object(runner, "_task_prompts", return_value={"task_one": "Do the task."}),
                patch.object(runner, "_judge_executor", return_value=judge),
            ):
                status = runner.run()

            self.assertEqual(status, 130)
            self.assertEqual(judge.calls, 1)
            persisted = out / "judge" / "tasks" / "task_one" / "trial_0" / "executor"
            self.assertEqual((persisted / "stdout.log").read_text(), "partial stdout\n")
            row = json.loads((out / "local-judge-results.jsonl").read_text().splitlines()[0])
            self.assertTrue(row["metadata"]["interrupted"])
            self.assertEqual(row["exit_code"], 130)

    def test_nonzero_exit_stops_after_first_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            a = self._candidate(root, "a")
            b = self._candidate(root, "b")
            judge = _FailingJudge()
            with (
                patch.dict(os.environ, self._environment(root, out, a, b), clear=False),
                patch.object(runner, "_preflight", return_value=self._preflight(root)),
                patch.object(runner, "_ensure_dataset"),
                patch.object(runner, "_task_prompts", return_value={"task_one": "Do the task."}),
                patch.object(runner, "_judge_executor", return_value=judge),
            ):
                status = runner.run()

            self.assertEqual(status, 1)
            self.assertEqual(judge.calls, 1)
            summary = json.loads((out / "local-judge-summary.json").read_text())
            self.assertIn("exit_code=73", summary["systemic_failure"])
            self.assertEqual(len((out / "local-judge-results.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
