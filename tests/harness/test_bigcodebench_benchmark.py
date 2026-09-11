# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gdpval_harness.benchmarks.base import EvaluatorType
from gdpval_harness.benchmarks.bigcodebench import BigCodeBenchBenchmark, _native_bigcodebench_evaluate
from gdpval_harness.executors.base import ExecutionResult, ExecutionStatus


class BigCodeBenchBenchmarkTests(unittest.TestCase):
    def benchmark(self, root: Path, *, resource_dir: Path | None = None) -> BigCodeBenchBenchmark:
        return BigCodeBenchBenchmark(
            root=root,
            dataset_path=root / "bigcodebench.jsonl",
            prepare_script=root / "prepare.py",
            resource_dir=resource_dir or root / "grader",
        )

    def write_task(self, benchmark: BigCodeBenchBenchmark) -> None:
        benchmark.dataset_path.write_text(
            json.dumps(
                {
                    "question": "Complete this function.",
                    "verifier_metadata": {
                        "task_id": "BigCodeBench/1",
                        "test": "def test_solution(): pass",
                        "entry_point": "solve",
                        "code_prompt": "def solve():",
                        "split": "hard",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def result(
        self,
        root: Path,
        *,
        output_text: str | None,
        status: ExecutionStatus = ExecutionStatus.NO_DELIVERABLE,
    ) -> ExecutionResult:
        return ExecutionResult(
            task_id="BigCodeBench/1",
            executor="codex",
            executor_version="test",
            invocation_mode="codex exec",
            auth_mode="chatgpt-subscription",
            workspace=root / "executor-workspace",
            deliverables_dir=root / "executor-workspace" / "deliverables",
            status=status,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=0 if status is ExecutionStatus.NO_DELIVERABLE else 1,
            output_text=output_text,
        )

    def test_load_tasks_preserves_codegen_prompt_and_verifier_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = self.benchmark(root)
            self.write_task(benchmark)

            task = benchmark.load_tasks(1)[0]

            self.assertEqual(task.execution.task_id, "BigCodeBench/1")
            self.assertEqual(
                task.execution.prompt,
                "Generate an executable Python function generated from the given prompt.\n\nComplete this function.",
            )
            self.assertEqual(task.evaluation["entry_point"], "solve")
            self.assertEqual(benchmark.revision, "v0.1.4")
            self.assertEqual(benchmark.evaluator_type, EvaluatorType.EXECUTABLE_TESTS)

    def test_evaluate_uses_native_grader_outside_executor_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grader = root / "grader"
            grader.mkdir()
            benchmark = self.benchmark(root, resource_dir=grader)
            self.write_task(benchmark)
            task = benchmark.load_tasks(1)[0]
            result = self.result(root, output_text="```python\nreturn 42\n```")

            with patch(
                "gdpval_harness.benchmarks.bigcodebench._native_bigcodebench_evaluate",
                return_value={
                    "reward": 1.0,
                    "status": "pass",
                    "extracted_model_code": "return 42",
                    "details": {},
                },
            ) as evaluator:
                evaluation = benchmark.evaluate(task, result)

            evaluator.assert_called_once_with(result.output_text, task.evaluation, resource_dir=grader.resolve())
            self.assertEqual(evaluation.metrics, {"pass_rate": 1.0})
            self.assertEqual(evaluation.details["status"], "pass")
            self.assertNotEqual(evaluation.details["grader_root"], evaluation.details["executor_workspace"])

    def test_failed_execution_scores_zero_without_invoking_grader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = self.benchmark(root)
            self.write_task(benchmark)
            task = benchmark.load_tasks(1)[0]
            result = self.result(root, output_text="```python\nreturn 42\n```", status=ExecutionStatus.FAILED)

            with patch("gdpval_harness.benchmarks.bigcodebench._native_bigcodebench_evaluate") as evaluator:
                evaluation = benchmark.evaluate(task, result)

            evaluator.assert_not_called()
            self.assertEqual(evaluation.metrics, {"pass_rate": 0.0})
            self.assertFalse(evaluation.details["grader_invoked"])

    def test_grader_directory_must_not_be_inside_executor_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "executor-workspace"
            grader = workspace / "grader"
            grader.mkdir(parents=True)
            benchmark = self.benchmark(root, resource_dir=grader)
            self.write_task(benchmark)
            task = benchmark.load_tasks(1)[0]
            result = self.result(root, output_text="```python\nreturn 42\n```")

            with self.assertRaisesRegex(RuntimeError, "grader directory must be separate"):
                benchmark.evaluate(task, result)

    def test_native_helper_uses_existing_extractor_venv_and_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resource_dir = Path(tmp)
            runner = resource_dir / "bcb_runner.py"
            runner.write_text("# fake runner\n", encoding="utf-8")
            venv_python = resource_dir / ".bcb_venv" / "bin" / "python"
            metadata = {
                "task_id": "BigCodeBench/1",
                "test": "test-code",
                "entry_point": "solve",
                "code_prompt": "def solve():",
            }

            def fake_run(command, **kwargs):
                self.assertEqual(command, [str(venv_python), str(runner)])
                self.assertEqual(kwargs["cwd"], resource_dir)
                payload = json.loads(kwargs["input"])
                self.assertEqual(payload["code"], "def solve():\n    pass\nreturn 42")
                self.assertEqual(payload["test_code"], "test-code")
                self.assertEqual(payload["entry_point"], "solve")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"status": "pass", "details": {"tests": 1}}),
                    stderr="",
                )

            with (
                patch(
                    "resources_servers.bigcodebench.code_extraction.preprocess_code_completion",
                    return_value="return 42",
                ),
                patch(
                    "resources_servers.bigcodebench.setup_bcb_venv.ensure_bcb_venv",
                    return_value=venv_python,
                ) as ensure_venv,
                patch("gdpval_harness.benchmarks.bigcodebench.subprocess.run", side_effect=fake_run),
            ):
                evaluation = _native_bigcodebench_evaluate(
                    "```python\nreturn 42\n```",
                    metadata,
                    resource_dir=resource_dir,
                )

            ensure_venv.assert_called_once_with(resource_dir / ".bcb_venv", "3.10")
            self.assertEqual(evaluation["reward"], 1.0)
            self.assertEqual(evaluation["status"], "pass")
            self.assertEqual(evaluation["extracted_model_code"], "return 42")


if __name__ == "__main__":
    unittest.main()
