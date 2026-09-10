# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gdpval_harness.local_judge_runner import _candidate_task_prompt, _judge_environment


class LocalJudgeRunnerTests(unittest.TestCase):
    def test_candidate_prompt_preserves_task_owned_carriage_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deliverables = root / "deliverables"
            deliverables.mkdir()
            executor = root / "tasks" / "x" / "executor"
            executor.mkdir(parents=True)
            task_prompt = b"first\r\nsecond\rthird"
            (executor / "prompt.txt").write_bytes(b"wrapper\nTask:\n" + task_prompt + b"\n")

            self.assertEqual(
                _candidate_task_prompt(deliverables, "task_x"),
                task_prompt.decode("utf-8"),
            )

    def test_judge_environment_excludes_candidate_provenance(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HOME": "/tmp/home",
                "PATH": "/bin",
                "GDPVAL_RUN_A": "/secret/a",
                "GDPVAL_RUN_B": "/secret/b",
                "GDPVAL_LABEL_A": "baseline",
                "GDPVAL_LABEL_B": "intervention",
                "SECRET_SHOULD_NOT_REACH_JUDGE": "secret",
            },
            clear=True,
        ):
            env = _judge_environment()

        self.assertEqual(env["HOME"], "/tmp/home")
        self.assertEqual(env["PATH"], "/bin")
        self.assertNotIn("GDPVAL_RUN_A", env)
        self.assertNotIn("GDPVAL_RUN_B", env)
        self.assertNotIn("GDPVAL_LABEL_A", env)
        self.assertNotIn("GDPVAL_LABEL_B", env)
        self.assertNotIn("SECRET_SHOULD_NOT_REACH_JUDGE", env)


if __name__ == "__main__":
    unittest.main()
