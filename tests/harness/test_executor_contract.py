# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gdpval_harness.executors.base import ExecutionStatus, TaskSpec
from gdpval_harness.executors.registry import get_executor_descriptor, list_executors
from gdpval_harness.layout import safe_task_id, task_layout


class ExecutorContractTests(unittest.TestCase):
    def test_registry_exposes_supported_executors(self) -> None:
        self.assertEqual([item.name for item in list_executors()], ["claude-code", "codex", "cursor", "stirrup"])
        self.assertEqual(get_executor_descriptor("stirrup").usage_mode, "provider-backed model API")
        self.assertEqual(get_executor_descriptor("claude-code").usage_mode, "Claude subscription login only")
        self.assertIn("Cursor account", get_executor_descriptor("cursor").usage_mode)

    def test_unknown_executor_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            get_executor_descriptor("unknown")

    def test_task_layout_keeps_executor_files_out_of_judge_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            layout = task_layout(Path(root), "abc/123")
            self.assertEqual(layout.workspace.name, "workspace")
            self.assertEqual(layout.executor_dir.name, "executor")
            self.assertEqual(layout.workspace_deliverables, layout.workspace / "deliverables")
            self.assertIn("task_abc_123", str(layout.judge_deliverables))
            self.assertNotEqual(layout.executor_dir.parent, layout.judge_deliverables.parent)

    def test_nonzero_repeats_use_isolated_workspace_and_executor_paths(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            first = task_layout(base, "task", repeat=0)
            second = task_layout(base, "task", repeat=1)
            self.assertNotEqual(first.workspace, second.workspace)
            self.assertNotEqual(first.executor_dir, second.executor_dir)
            self.assertEqual(second.workspace.parent.name, "repeat_1")
            self.assertEqual(second.judge_deliverables.name, "repeat_1")

    def test_negative_repeat_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                task_layout(Path(root), "task", repeat=-1)

    def test_task_contract_has_no_provider_fields(self) -> None:
        task = TaskSpec(task_id="x", prompt="do work")
        self.assertEqual(task.task_id, "x")
        self.assertEqual(ExecutionStatus.COMPLETED.value, "completed")

    def test_safe_task_id_rejects_empty_value(self) -> None:
        with self.assertRaises(ValueError):
            safe_task_id("///")


if __name__ == "__main__":
    unittest.main()
