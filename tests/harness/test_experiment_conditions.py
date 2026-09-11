# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gdpval_harness.executors.base import ExecutionRequest, TaskSpec
from gdpval_harness.executors.codex import CodexExecutor
from gdpval_harness.judges.base import JudgeRequest
from gdpval_harness.judges.codex import CodexJudgeExecutor
from gdpval_harness.local_judge_runner import _candidate_task_prompt
from gdpval_harness.local_runner import (
    _condition_instructions,
    _validate_resume_condition,
    build_task_prompt,
)


class ExperimentConditionTests(unittest.TestCase):
    def _execution_request(self, root: Path) -> ExecutionRequest:
        workspace = root / "workspace"
        executor_dir = root / "executor"
        return ExecutionRequest(
            task=TaskSpec(task_id="task", prompt="Base GDPval task"),
            workspace=workspace,
            deliverables_dir=workspace / "deliverables",
            executor_dir=executor_dir,
        )

    def _judge_request(self, root: Path) -> JudgeRequest:
        workspace = root / "judge-workspace"
        return JudgeRequest(
            task_id="task",
            task_prompt="Judge this task",
            workspace=workspace,
            reference_dir=workspace / "reference_files",
            submission_a_dir=workspace / "submission_a",
            submission_b_dir=workspace / "submission_b",
            executor_dir=root / "judge-executor",
            trial_index=0,
            swapped=False,
        )

    def test_condition_file_is_external_and_label_is_not_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition.md"
            condition.write_text("Use the supplied work-design method.\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"GDPVAL_CONDITION_FILE": str(condition), "GDPVAL_CONDITION": "secret-label"},
                clear=True,
            ):
                instructions = _condition_instructions()
            prompt = build_task_prompt(
                TaskSpec(task_id="task", prompt="Base GDPval task"),
                root,
                network_policy="disabled",
                condition_instructions=instructions,
            )
            self.assertIn("Use the supplied work-design method.", prompt)
            self.assertNotIn("secret-label", prompt)
            self.assertEqual(prompt.rsplit("\nTask:\n", 1)[1], "Base GDPval task\n")

    def test_condition_may_contain_task_marker_when_canonical_prompt_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            executor_dir = run_root / "tasks" / "one" / "executor"
            executor_dir.mkdir(parents=True)
            (executor_dir / "task-prompt.txt").write_text("Base GDPval task", encoding="utf-8")
            (executor_dir / "prompt.txt").write_text(
                "Wrapper\n<condition>\nTemplate\nTask:\nplaceholder\n</condition>\n\nTask:\nBase GDPval task\n",
                encoding="utf-8",
            )
            deliverables = run_root / "deliverables"
            deliverables.mkdir()
            self.assertEqual(_candidate_task_prompt(deliverables, "task_one"), "Base GDPval task")

    def test_condition_file_must_be_nonempty_utf8_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.md"
            empty.write_text("   \n", encoding="utf-8")
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(empty)}, clear=True):
                with self.assertRaisesRegex(ValueError, "empty"):
                    _condition_instructions()

            invalid = root / "invalid.bin"
            invalid.write_bytes(b"\xff")
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(invalid)}, clear=True):
                with self.assertRaisesRegex(ValueError, "UTF-8"):
                    _condition_instructions()

            large = root / "large.md"
            large.write_bytes(b"x" * (1024 * 1024 + 1))
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(large)}, clear=True):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    _condition_instructions()

    def test_resume_requires_identical_condition_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition.md"
            condition.write_text("same intervention\n", encoding="utf-8")
            digest = hashlib.sha256(condition.read_bytes()).hexdigest()
            (root / "run-metadata.json").write_text(
                json.dumps(
                    {
                        "configuration": {
                            "condition": "treatment",
                            "condition_file_sha256": digest,
                            "condition_applied_to_prompt": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            env = {
                "RESUME": "1",
                "GDPVAL_CONDITION": "treatment",
                "GDPVAL_CONDITION_FILE": str(condition),
            }
            with patch.dict(os.environ, env, clear=True):
                _validate_resume_condition(root)

            condition.write_text("changed intervention\n", encoding="utf-8")
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    _validate_resume_condition(root)

    def test_conditioned_resume_without_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition.md"
            condition.write_text("intervention\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"RESUME": "1", "GDPVAL_CONDITION": "treatment", "GDPVAL_CONDITION_FILE": str(condition)},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "requires the existing run-metadata"):
                    _validate_resume_condition(root)

    def test_codex_policy_forces_chatgpt_and_disables_web_search_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = CodexExecutor(network_enabled=False).build_command(self._execution_request(Path(tmp)))
        joined = " ".join(command)
        self.assertIn('forced_login_method="chatgpt"', joined)
        self.assertIn('web_search="disabled"', joined)
        self.assertIn('approval_policy="never"', joined)

    def test_codex_policy_does_not_force_web_search_off_when_network_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = CodexExecutor(network_enabled=True).build_command(self._execution_request(Path(tmp)))
        self.assertNotIn('web_search="disabled"', " ".join(command))
        self.assertIn('forced_login_method="chatgpt"', " ".join(command))

    def test_codex_judge_forces_chatgpt_and_disables_web_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = CodexJudgeExecutor().build_command(self._judge_request(Path(tmp)))
        joined = " ".join(command)
        self.assertIn('forced_login_method="chatgpt"', joined)
        self.assertIn('web_search="disabled"', joined)
        self.assertIn('default_permissions="gdpval-harness-blind-judge"', joined)
        self.assertIn('network={enabled=false}', joined)
        self.assertNotIn("--sandbox read-only", joined)


if __name__ == "__main__":
    unittest.main()
