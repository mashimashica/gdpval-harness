# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from gdpval_harness.interventions.base import (
    ApplicationMapping,
    Intervention,
    InterventionApplication,
    InterventionBundle,
    InterventionFile,
    InterventionManifest,
    InterventionPreflightResult,
    InterventionType,
    compute_bundle_sha256,
)
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult
from gdpval_harness.judges.pairwise import discover_tasks
from gdpval_harness.local_judge_runner import _candidate_task_prompt
from gdpval_harness.runner import run_benchmark


class FakeBenchmark(Benchmark):
    name = "fake"
    revision = "test-revision"

    def __init__(
        self,
        task_count: int = 3,
        *,
        events: list[str] | None = None,
        evaluation_metadata: dict[str, object] | None = None,
    ) -> None:
        self.task_count = task_count
        self.events = events if events is not None else []
        self.evaluation_metadata = evaluation_metadata or {}

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        self.events.append("benchmark-prepare")
        return None

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        return [
            BenchmarkTask(
                execution=TaskSpec(task_id=f"task-{index}", prompt=f"prompt-{index}"),
                evaluation=dict(self.evaluation_metadata),
            )
            for index in range(min(limit, self.task_count))
        ]

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        self.events.append(f"materialize:{task.execution.task_id}")
        (workspace / "input.txt").write_text(task.execution.prompt, encoding="utf-8")
        return ["input.txt"]

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        del workspace, network_policy
        self.events.append(f"execution-task:{task.execution.task_id}")
        return task.execution


class FakeIntervention(Intervention):
    name = "fake-intervention"
    intervention_type = InterventionType.PROMPT_OVERLAY

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        fail_task_id: str | None = None,
        invalid_task_id: str | None = None,
        mismatch_application: str | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.fail_task_id = fail_task_id
        self.invalid_task_id = invalid_task_id
        self.mismatch_application = mismatch_application
        self.preflight_calls = 0
        self.apply_calls = 0
        self.tasks: list[TaskSpec] = []
        self._bundle: InterventionBundle | None = None

    def preflight(self) -> InterventionPreflightResult:
        self.events.append("intervention-preflight")
        self.preflight_calls += 1
        content = b"secret intervention source"
        entry = InterventionFile(path="overlay.txt", size=len(content), sha256="".join(["a"] * 64))
        manifest = InterventionManifest(
            intervention_id="fake-intervention-id",
            intervention_type=self.intervention_type,
            source_revision="source-revision",
            revision_status="available",
            files=(entry,),
            bundle_sha256=compute_bundle_sha256(((entry.path, content),)),
            application=ApplicationMapping(method="prompt-overlay", target="task.prompt"),
        )
        self._bundle = InterventionBundle(root=Path("/external/secret/intervention-source"), manifest=manifest)
        return InterventionPreflightResult(
            name=self.name,
            intervention_type=self.intervention_type,
            ok=True,
            bundle=self._bundle,
            details=("fake intervention ready", "/external/secret/intervention-source"),
        )

    def validate_task(self, task: TaskSpec) -> None:
        self.events.append(f"validate:{task.task_id}")
        if task.task_id == self.invalid_task_id:
            raise ValueError("invalid intervention task")

    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        self.events.append(f"apply:{task.task_id}")
        self.apply_calls += 1
        self.tasks.append(task)
        if task.task_id == self.fail_task_id:
            raise RuntimeError("intervention application secret failure")
        assert self._bundle is not None
        bundle_sha256 = self._bundle.manifest.bundle_sha256
        manifest_sha256 = self._bundle.manifest.manifest_sha256 or ""
        application_mapping = self._bundle.manifest.application
        if self.mismatch_application == "bundle":
            bundle_sha256 = "0" * 64
        elif self.mismatch_application == "manifest":
            manifest_sha256 = "1" * 64
        elif self.mismatch_application == "mapping":
            application_mapping = ApplicationMapping(method="files", target="workspace")
        return InterventionApplication(
            application_run_id=application_run_id,
            task=TaskSpec(task.task_id, f"[OVERLAY FOR CONDITION SECRET]\n{task.prompt}"),
                materialized_files=(
                InterventionFile(
                    path="injected.txt",
                    size=7,
                    sha256="".join(["b"] * 64),
                ),
            ),
            bundle_sha256=bundle_sha256,
            manifest_sha256=manifest_sha256,
            application=application_mapping,
        )


class FakeEvaluator(Evaluator):
    name = "fake-evaluator"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def __init__(
        self,
        *,
        fail: bool = False,
        statuses: tuple[EvaluationStatus, ...] | None = None,
        invalid_plan_task_id: str | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.fail = fail
        self.statuses = statuses or (EvaluationStatus.COMPLETED,)
        self.invalid_plan_task_id = invalid_plan_task_id
        self.events = events if events is not None else []
        self.calls = 0
        self.preflight_calls = 0

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        self.events.append("evaluator-preflight")
        self.preflight_calls += 1
        return EvaluatorPreflightResult(self.name, self.evaluator_type, True, version="fake-eval-1")

    def validate_plan(self, plan: EvaluationPlan) -> None:
        if plan.task_id == self.invalid_plan_task_id:
            raise ValueError("invalid evaluation plan")

    def evaluate(self, request):
        self.calls += 1
        self.events.append(f"evaluate:{request.task_id}")
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

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        outside_deliverables: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.outside_deliverables = outside_deliverables
        self.events = events if events is not None else []
        self.tasks: list[TaskSpec] = []
        self.environments: list[dict[str, str]] = []

    def preflight(self) -> PreflightResult:
        self.events.append("executor-preflight")
        return PreflightResult(
            executor=self.name,
            ok=True,
            version="fake-1",
            auth_mode="fake-local",
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls += 1
        self.events.append(f"executor:{request.task.task_id}")
        self.tasks.append(request.task)
        self.environments.append(dict(request.environment))
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
    def test_explicit_intervention_order_evidence_and_prompt_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []
            benchmark = FakeBenchmark(task_count=1, events=events, evaluation_metadata={"secret": "sentinel"})
            evaluator = FakeEvaluator(events=events)
            intervention = FakeIntervention(events=events)
            executor = FakeExecutor(events=events)

            with patch.dict(
                os.environ,
                {
                    "GDPVAL_CONDITION": "condition-label-sentinel",
                    "GDPVAL_CONDITION_FILE": "/external/secret/condition-source",
                    "GDPVAL_CONDITION_APPLIED": "true",
                },
            ):
                run_benchmark(
                    benchmark,
                    evaluator,
                    executor,
                    out_dir=root / "run",
                    limit=1,
                    intervention=intervention,
                )

            self.assertEqual(
                events[:3],
                ["evaluator-preflight", "intervention-preflight", "executor-preflight"],
            )
            self.assertLess(events.index("materialize:task-0"), events.index("execution-task:task-0"))
            self.assertLess(events.index("execution-task:task-0"), events.index("apply:task-0"))
            self.assertLess(events.index("apply:task-0"), events.index("executor:task-0"))
            self.assertLess(events.index("executor:task-0"), events.index("evaluate:task-0"))
            self.assertEqual(executor.tasks[0].task_id, "task-0")
            self.assertIn("OVERLAY FOR CONDITION SECRET", executor.tasks[0].prompt)
            executor_environment = executor.environments[0]
            for key in ("GDPVAL_CONDITION", "GDPVAL_CONDITION_FILE", "GDPVAL_CONDITION_APPLIED"):
                self.assertNotIn(key, executor_environment)
            serialized_environment = json.dumps(executor_environment)
            self.assertNotIn("condition-label-sentinel", serialized_environment)
            self.assertNotIn("/external/secret/condition-source", serialized_environment)
            self.assertNotIn("/external/secret/intervention-source", serialized_environment)
            self.assertNotIn("OVERLAY FOR CONDITION SECRET", serialized_environment)

            run_dir = root / "run"
            canonical = run_dir / "tasks" / "task-0" / "executor" / "task-prompt.txt"
            self.assertEqual(canonical.read_text(encoding="utf-8"), "prompt-0")
            metadata = json.loads((run_dir / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["intervention"]["id"], "fake-intervention-id")
            self.assertEqual(metadata["intervention"]["type"], "prompt-overlay")
            self.assertEqual(
                metadata["intervention"]["files"],
                [{"path": "overlay.txt", "size": len(b"secret intervention source"), "sha256": "a" * 64}],
            )
            self.assertEqual(
                metadata["intervention"]["application"],
                {"method": "prompt-overlay", "target": "task.prompt"},
            )
            self.assertNotIn("/external/secret/intervention-source", json.dumps(metadata))
            self.assertNotIn("sentinel", json.dumps(metadata))

            row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
            evidence = row["intervention"]
            self.assertEqual(evidence["status"], "applied")
            self.assertEqual(evidence["materialized_files"][0]["path"], "injected.txt")
            self.assertEqual(evidence["materialized_files"][0]["size"], 7)
            self.assertNotIn("/external/secret/intervention-source", json.dumps(row))
            self.assertNotIn("sentinel", json.dumps(row))
            self.assertNotIn("task-0", evidence["application_run_id"])

    def test_none_intervention_is_the_default_identity_application(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            run_benchmark(FakeBenchmark(task_count=1), FakeEvaluator(), executor, out_dir=out, limit=1)

            self.assertEqual(executor.tasks[0].prompt, "prompt-0")
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["intervention"]["id"], "none")
            self.assertEqual(metadata["intervention"]["type"], "none")
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["intervention"]["status"], "applied")
            self.assertEqual(row["intervention"]["materialized_files"], [])

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

    def test_intervention_preflight_failure_stops_before_executor_or_benchmark(self) -> None:
        class NotReadyIntervention(FakeIntervention):
            def preflight(self) -> InterventionPreflightResult:
                self.events.append("intervention-preflight")
                return InterventionPreflightResult(
                    name=self.name,
                    intervention_type=self.intervention_type,
                    ok=False,
                    details=("intervention source unavailable",),
                )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            events: list[str] = []
            benchmark = FakeBenchmark(events=events)
            executor = FakeExecutor(events=events)
            with self.assertRaisesRegex(RuntimeError, "intervention source unavailable"):
                run_benchmark(
                    benchmark,
                    FakeEvaluator(events=events),
                    executor,
                    out_dir=out,
                    limit=1,
                    intervention=NotReadyIntervention(events=events),
                )
            self.assertEqual(executor.calls, 0)
            self.assertEqual(events[:2], ["evaluator-preflight", "intervention-preflight"])
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

    def test_later_intervention_task_validation_stops_before_any_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            with self.assertRaisesRegex(ValueError, "invalid intervention task"):
                run_benchmark(
                    FakeBenchmark(task_count=2),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=2,
                    intervention=FakeIntervention(invalid_task_id="task-1"),
                )
            self.assertEqual(executor.calls, 0)
            self.assertFalse(out.exists())

    def test_intervention_failure_persists_evidence_and_stops_later_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            intervention = FakeIntervention(fail_task_id="task-1")
            with self.assertRaisesRegex(RuntimeError, "intervention application secret failure"):
                run_benchmark(
                    FakeBenchmark(task_count=3),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=3,
                    intervention=intervention,
                )

            self.assertEqual(executor.calls, 1)
            self.assertEqual([task.task_id for task in executor.tasks], ["task-0"])
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["intervention"]["status"], "applied")
            self.assertEqual(rows[1]["intervention"]["status"], "failed")
            self.assertEqual(rows[1]["execution"], None)
            self.assertNotIn("secret failure", json.dumps(rows[1]))
            self.assertTrue((out / "tasks" / "task-0" / "executor" / "stdout.log").is_file())
            self.assertTrue((out / "tasks" / "task-0" / "result.json").is_file())
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["failure"]["phase"], "intervention_apply")

    def test_intervention_application_mismatch_fails_before_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            with self.assertRaisesRegex(ValueError, "mismatched bundle_sha256"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=1,
                    intervention=FakeIntervention(mismatch_application="bundle"),
                )

            self.assertEqual(executor.calls, 0)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["intervention"]["status"], "failed")
            self.assertIsNone(rows[0]["execution"])

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
