# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from typing import cast

from gdpval_harness.builders.prompt import build_skill_task
from gdpval_harness.executors.base import TaskSpec


class BuilderPromptTests(unittest.TestCase):
    def test_preserves_task_id_and_includes_only_target_prompt_content(self) -> None:
        task = TaskSpec(task_id="task-id-sentinel", prompt="Create a reusable calendar skill.")
        result = build_skill_task(
            task,
            (
                "reference_files/builder-inputs/input-001",
                "reference_files/builder-inputs/input-007",
            ),
        )

        self.assertIsInstance(result, TaskSpec)
        self.assertEqual(result.task_id, task.task_id)
        self.assertIn(task.prompt, result.prompt)
        self.assertNotIn(task.task_id, result.prompt)
        self.assertEqual(set(TaskSpec.__dataclass_fields__), {"task_id", "prompt"})

    def test_includes_neutral_targets_and_static_boundaries(self) -> None:
        targets = (
            "reference_files/builder-inputs/input-001",
            "reference_files/builder-inputs/input-002",
        )
        prompt = build_skill_task(TaskSpec("task", "target prompt"), targets).prompt

        for target in targets:
            self.assertIn(target, prompt)
        self.assertIn("one reusable Agent Skill", prompt)
        self.assertIn("Do not solve or submit the target task", prompt)
        self.assertIn("exactly one immediate directory", prompt)
        self.assertIn("./deliverables/<skill-name>/SKILL.md", prompt)
        self.assertIn("YAML `name`", prompt)
        self.assertIn("builder transcript", prompt)
        self.assertIn("generation provenance", prompt)
        self.assertIn("scores", prompt)
        self.assertIn("rubrics", prompt)
        self.assertIn("reference answers", prompt)

    def test_rejects_noncanonical_targets(self) -> None:
        task = TaskSpec("task", "prompt")
        invalid_targets = (
            "/reference_files/builder-inputs/input-001",
            "reference_files\\builder-inputs\\input-001",
            "reference_files/builder-inputs/../input-001",
            "reference_files/builder-inputs/input-01",
            "reference_files/builder-inputs/input-000",
            "reference_files/builder-inputs/input-001/",
            "other/input-001",
        )

        for target in invalid_targets:
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    build_skill_task(task, (target,))

    def test_rejects_duplicate_or_unsorted_targets(self) -> None:
        task = TaskSpec("task", "prompt")
        first = "reference_files/builder-inputs/input-001"
        second = "reference_files/builder-inputs/input-002"

        with self.assertRaises(ValueError):
            build_skill_task(task, (first, first))
        with self.assertRaises(ValueError):
            build_skill_task(task, (second, first))

    def test_rejects_invalid_task_and_target_types(self) -> None:
        with self.assertRaises(TypeError):
            # Preserve the invalid runtime task object at the prompt boundary.
            build_skill_task(cast(TaskSpec, object()), ())
        with self.assertRaises(TypeError):
            build_skill_task(TaskSpec("task", "prompt"), "reference_files/builder-inputs/input-001")
        with self.assertRaises(TypeError):
            # Preserve the invalid runtime target at the prompt boundary.
            build_skill_task(TaskSpec("task", "prompt"), cast(tuple[str, ...], (None,)))


if __name__ == "__main__":
    unittest.main()
