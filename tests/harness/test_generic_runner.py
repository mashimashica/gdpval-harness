# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkEvaluation, BenchmarkTask, EvaluatorType
from gdpval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from gdpval_harness.runner import run_benchmark


class FakeBenchmark(Benchmark):
    name = "fake"
    revision = "test-revision"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def __init__(self, task_count: int = 3) -> None:
        self.task_count = task_count

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        return None

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        return [
            BenchmarkTask(execution=TaskSpec(task_id=f"task-{index}", prompt=f"prompt-{index}"))
            for index in range(min(limit, self.task_count))
        ]

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        (workspace / "input.txt").write_text(task.execution.prompt, encoding="utf-8")
        return ["input.txt"]

    def evaluate(self, task: BenchmarkTask, result: ExecutionResult) -> BenchmarkEvaluation:
        del result
        score = 1.0 if task.execution.task_id.endswith("0") else 0.0
        return BenchmarkEvaluation(task_id=task.execution.task_id, metrics={"accuracy": score})


class FakeExecutor(Executor):
    name = "fake-executor"
    invocation_mode = "fake"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def preflight(self) -> PreflightResult:
        return PreflightResult(
            executor=self.name,
            ok=True,
            version="fake-1",
            auth_mode="fake-local",
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls += 1
        failed = self.fail_on_call == self.calls
        status = ExecutionStatus.FAILED if failed else ExecutionStatus.NO_DELIVERABLE
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        (request.executor_dir / "stdout.log").write_text("fake\n", encoding="utf-8")
        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="fake-1",
            invocation_mode=self.invocation_mode,
            auth_mode="fake-local",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=status,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=1 if failed else 0,
            output_text="answer",
            metadata={"fake": True},
        )


class GenericRunnerTests(unittest.TestCase):
    def test_run_persists_each_task_and_aggregates_benchmark_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()

            summary = run_benchmark(FakeBenchmark(), executor, out_dir=out, limit=2)

            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.task_count, 2)
            self.assertEqual(summary.metrics, {"accuracy": 0.5})
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["materialized"], ["input.txt"])
            self.assertEqual(rows[0]["evaluation"]["metrics"], {"accuracy": 1.0})
            self.assertTrue((out / "tasks" / "task-0" / "executor" / "result.json").is_file())
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_tasks"], 2)

    def test_executor_failure_is_durable_and_aborts_remaining_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor(fail_on_call=1)

            summary = run_benchmark(FakeBenchmark(), executor, out_dir=out, limit=3)

            self.assertEqual(summary.status, "failed")
            self.assertEqual(summary.task_count, 1)
            self.assertEqual(executor.calls, 1)
            rows = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["execution"]["status"], "failed")
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["completed_tasks"], 1)

    def test_existing_run_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            executor = FakeExecutor()

            with self.assertRaises(FileExistsError):
                run_benchmark(FakeBenchmark(), executor, out_dir=out, limit=1)

            self.assertEqual(executor.calls, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
