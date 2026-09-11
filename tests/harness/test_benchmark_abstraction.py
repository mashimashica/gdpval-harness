# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gdpval_harness.benchmarks.base import BenchmarkTask, EvaluatorType
from gdpval_harness.benchmarks.gdpval import GDPvalBenchmark
from gdpval_harness.executors.base import TaskSpec


class BenchmarkAbstractionTests(unittest.TestCase):
    def test_executor_task_spec_contains_no_benchmark_specific_fields(self) -> None:
        self.assertEqual(set(TaskSpec.__dataclass_fields__), {"task_id", "prompt"})

    def test_gdpval_keeps_materialization_and_evaluation_data_outside_task_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "gdpval.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "prompt": "Create the deliverable.",
                        "reference_files": ["reference_files/input.csv"],
                        "reference_file_urls": ["https://example.invalid/input.csv"],
                        "sector": "finance",
                        "occupation": "analyst",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark = GDPvalBenchmark(
                root=root,
                dataset_path=dataset,
                prepare_script=root / "missing-prepare.py",
            )

            tasks = benchmark.load_tasks(1)
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertIsInstance(task, BenchmarkTask)
            self.assertEqual(task.execution, TaskSpec(task_id="task-1", prompt="Create the deliverable."))
            self.assertEqual(task.materialization["reference_files"], ("reference_files/input.csv",))
            self.assertEqual(task.materialization["reference_file_urls"], ("https://example.invalid/input.csv",))
            self.assertEqual(task.evaluation["sector"], "finance")
            self.assertEqual(task.evaluation["occupation"], "analyst")
            self.assertEqual(benchmark.evaluator_type, EvaluatorType.LLM_RUBRIC)

    def test_gdpval_prepare_is_noop_when_dataset_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "gdpval.jsonl"
            dataset.write_text('{"task_id":"x","prompt":"p"}\n', encoding="utf-8")
            benchmark = GDPvalBenchmark(
                root=root,
                dataset_path=dataset,
                prepare_script=root / "missing-prepare.py",
            )
            benchmark.prepare()
            self.assertTrue(benchmark.is_prepared())

    def test_gdpval_rejects_nonpositive_task_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "gdpval.jsonl"
            dataset.write_text('{"task_id":"x","prompt":"p"}\n', encoding="utf-8")
            benchmark = GDPvalBenchmark(root=root, dataset_path=dataset, prepare_script=root / "prepare.py")
            with self.assertRaises(ValueError):
                benchmark.load_tasks(0)


if __name__ == "__main__":
    unittest.main()
