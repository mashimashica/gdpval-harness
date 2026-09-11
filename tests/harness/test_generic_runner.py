# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkTask
from gdpval_harness.evaluators.base import (
    EvaluationPlan,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
    EvaluatorType,
)
from gdpval_harness.evaluators.gdpval import GDPvalExternalEvaluator
from gdpval_harness.evaluators.pairwise import PairwiseJudgeEvaluator
from gdpval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult
from gdpval_harness.runner import run_benchmark
from gdpval_harness.judges.pairwise import discover_tasks
from gdpval_harness.local_judge_runner import _candidate_task_prompt


class FakeBenchmark(Benchmark):
    name = "fake"
    revision = "test-revision"

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


class FakeEvaluator(Evaluator):
    name = "fake-evaluator"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def __init__(
        self,
        *,
        fail: bool = False,
        statuses: tuple[EvaluationStatus, ...] | None = None,
        invalid_plan_task_id: str | None = None,
    ) -> None:
        self.fail = fail
        self.statuses = statuses or (EvaluationStatus.COMPLETED,)
        self.invalid_plan_task_id = invalid_plan_task_id
        self.calls = 0
        self.preflight_calls = 0

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        self.preflight_calls += 1
        return EvaluatorPreflightResult(self.name, self.evaluator_type, True, version="fake-eval-1")

    def validate_plan(self, plan: EvaluationPlan) -> None:
        if plan.task_id == self.invalid_plan_task_id:
            raise ValueError("invalid evaluation plan")

    def evaluate(self, request):
        self.calls += 1
        if self.fail:
            raise RuntimeError("evaluator test failure")
        status = self.statuses[min(self.calls - 1, len(self.statuses) - 1)]
        return EvaluationResult(
            task_id=request.task_id,
            status=status,
            metrics={"accuracy": 1.0},
            details={"canonical_prompt": request.task_prompt},
        )


class FakeExecutor(Executor):
    name = "fake-executor"
    invocation_mode = "fake"

    def __init__(self, *, fail_on_call: int | None = None, outside_deliverables: bool = False) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.outside_deliverables = outside_deliverables

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
        deliverables_dir = request.deliverables_dir
        if self.outside_deliverables:
            deliverables_dir = request.workspace.parent.parent / "outside-deliverables"
            deliverables_dir.mkdir(parents=True, exist_ok=True)
        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="fake-1",
            invocation_mode=self.invocation_mode,
            auth_mode="fake-local",
            workspace=request.workspace,
            deliverables_dir=deliverables_dir,
            status=status,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=1 if failed else 0,
            output_text="answer",
            metadata={"fake": True},
        )


class PlanOnlyJudge(JudgeExecutor):
    name = "plan-only-judge"
    invocation_mode = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def preflight(self) -> JudgePreflightResult:
        return JudgePreflightResult(judge_executor=self.name, ok=True, version="fake-judge-1")

    def judge(self, request: JudgeRequest) -> JudgeResult:
        del request
        self.calls += 1
        raise AssertionError("pairwise plan validation must happen before judge calls")


class GenericRunnerTests(unittest.TestCase):
    def test_run_persists_each_task_and_aggregates_completed_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            evaluator = FakeEvaluator()

            summary = run_benchmark(FakeBenchmark(), evaluator, executor, out_dir=out, limit=2)

            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.task_count, 2)
            self.assertEqual(summary.metrics, {"accuracy": 1.0})
            self.assertEqual(summary.evaluation_status_counts, {"completed": 2})
            self.assertEqual(evaluator.preflight_calls, 1)
            self.assertEqual(evaluator.calls, 2)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["materialized"], ["input.txt"])
            self.assertEqual(rows[0]["evaluation"]["metrics"], {"accuracy": 1.0})
            self.assertEqual(rows[0]["evaluation"]["status"], "completed")
            self.assertEqual(
                (out / "tasks" / "task-0" / "executor" / "task-prompt.txt").read_text(encoding="utf-8"),
                "prompt-0",
            )
            self.assertTrue((out / "tasks" / "task-0" / "result.json").is_file())
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_tasks"], 2)
            self.assertEqual(metadata["evaluation_status_counts"], {"completed": 2})
            self.assertEqual(metadata["evaluator_id"], "fake-evaluator")

    def test_external_status_is_visible_and_not_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            summary = run_benchmark(
                FakeBenchmark(task_count=1),
                FakeEvaluator(statuses=(EvaluationStatus.EXTERNAL,)),
                FakeExecutor(),
                out_dir=out,
                limit=1,
            )
            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.metrics, {})
            self.assertEqual(summary.evaluation_status_counts, {"external": 1})
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["evaluation"]["status"], "external")

    def test_gdpval_handoff_matches_existing_consumers_and_task_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            benchmark = FakeBenchmark(task_count=1)
            benchmark.name = "gdpval"
            summary = run_benchmark(
                benchmark,
                GDPvalExternalEvaluator(),
                FakeExecutor(),
                out_dir=out,
                limit=1,
            )
            self.assertEqual(summary.evaluation_status_counts, {"external": 1})
            deliverables = out / "deliverables"
            discovered = discover_tasks(deliverables)
            self.assertEqual(set(discovered), {"task_task-0"})
            self.assertEqual(discovered["task_task-0"].name, "repeat_0")
            self.assertEqual(_candidate_task_prompt(deliverables, "task_task-0"), "prompt-0")

    def test_evaluator_preflight_precedes_executor_and_benchmark(self) -> None:
        events: list[str] = []

        class OrderedBenchmark(FakeBenchmark):
            def prepare(self) -> None:
                events.append("benchmark")

        class OrderedEvaluator(FakeEvaluator):
            def preflight(self, run_dir=None):
                events.append("evaluator")
                return super().preflight(run_dir)

        class OrderedExecutor(FakeExecutor):
            def preflight(self):
                events.append("executor")
                return super().preflight()

        with tempfile.TemporaryDirectory() as tmp:
            run_benchmark(
                OrderedBenchmark(task_count=1),
                OrderedEvaluator(),
                OrderedExecutor(),
                out_dir=Path(tmp) / "run",
                limit=1,
            )
        self.assertEqual(events[:3], ["evaluator", "executor", "benchmark"])

    def test_evaluator_preflight_failure_stops_before_executor_or_benchmark(self) -> None:
        class NotReadyEvaluator(FakeEvaluator):
            def preflight(self, run_dir=None):
                self.preflight_calls += 1
                return EvaluatorPreflightResult(
                    self.name,
                    self.evaluator_type,
                    False,
                    details=("math-verify==0.8.0 is unavailable",),
                )

        class RecordingBenchmark(FakeBenchmark):
            def __init__(self):
                super().__init__(task_count=1)
                self.prepared = False

            def prepare(self):
                self.prepared = True

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            benchmark = RecordingBenchmark()
            executor = FakeExecutor()
            with self.assertRaisesRegex(RuntimeError, "math-verify==0.8.0"):
                run_benchmark(benchmark, NotReadyEvaluator(), executor, out_dir=out, limit=1)
            self.assertFalse(benchmark.prepared)
            self.assertEqual(executor.calls, 0)
            self.assertFalse(out.exists())

    def test_all_evaluation_plans_validate_before_any_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            evaluator = FakeEvaluator(invalid_plan_task_id="task-1")
            with self.assertRaisesRegex(ValueError, "invalid evaluation plan"):
                run_benchmark(
                    FakeBenchmark(task_count=2),
                    evaluator,
                    executor,
                    out_dir=out,
                    limit=2,
                )
            self.assertEqual(executor.calls, 0)
            self.assertFalse(out.exists())

    def test_pairwise_injection_rejects_generic_single_candidate_plan_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            judge = PlanOnlyJudge()
            evaluator = PairwiseJudgeEvaluator(judge)
            with self.assertRaisesRegex(ValueError, "exactly two candidates"):
                run_benchmark(FakeBenchmark(task_count=2), evaluator, executor, out_dir=out, limit=2)
            self.assertEqual(executor.calls, 0)
            self.assertEqual(judge.calls, 0)
            self.assertFalse(out.exists())

    def test_evaluator_exception_persists_completed_execution_and_fails_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "evaluator test failure"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(fail=True),
                    FakeExecutor(),
                    out_dir=out,
                    limit=1,
                )

            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["execution"]["status"], "no_deliverable")
            self.assertEqual(row["evaluation"]["status"], "failed")
            self.assertEqual(row["evaluation"]["metrics"], {})
            self.assertEqual(row["evaluation"]["outcomes"], {})
            persisted = json.loads((out / "tasks" / "task-0" / "result.json").read_text())
            self.assertEqual(persisted["evaluation"]["status"], "failed")
            metadata = json.loads((out / "run-metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["evaluation_status_counts"], {"failed": 1})
            self.assertNotIn("evaluator test failure", json.dumps(row))

    def test_executor_failure_is_evaluated_then_aborts_remaining_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor(fail_on_call=1)
            evaluator = FakeEvaluator()

            summary = run_benchmark(FakeBenchmark(), evaluator, executor, out_dir=out, limit=3)

            self.assertEqual(summary.status, "failed")
            self.assertEqual(summary.task_count, 1)
            self.assertEqual(executor.calls, 1)
            self.assertEqual(evaluator.calls, 1)
            rows = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["execution"]["status"], "failed")
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")

    def test_mismatched_deliverables_are_rejected_before_evaluator_and_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            evaluator = FakeEvaluator()
            with self.assertRaisesRegex(ValueError, "deliverables"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    evaluator,
                    FakeExecutor(outside_deliverables=True),
                    out_dir=out,
                    limit=1,
                )
            self.assertEqual(evaluator.calls, 0)
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["evaluation"]["status"], "failed")
            self.assertFalse((out / "deliverables").exists())

    def test_existing_run_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_benchmark(FakeBenchmark(), FakeEvaluator(), FakeExecutor(), out_dir=out, limit=1)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
