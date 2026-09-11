# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkTask
from gdpval_harness.builders.base import (
    BuildRequest,
    BuildResult,
    BuildStatus,
    Builder,
    BuilderPreflightResult,
)
from gdpval_harness.builders.inputs import load_builder_input_bundle
from gdpval_harness.evaluators.base import (
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
    EvaluatorType,
)
from gdpval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from gdpval_harness.experiments.base import (
    ExperimentArm,
    ExperimentInputSpec,
    ExperimentProfile,
    ExperimentRunConfig,
    LoadedExperimentProfile,
)
from gdpval_harness.experiments.runner import _make_schedule, _validate_build_result, run_builder_experiment
from gdpval_harness.interventions.agent_skill import load_agent_skill_bundle
from gdpval_harness.interventions.base import InterventionBundle


class _Benchmark(Benchmark):
    name = "generic-benchmark"
    revision = "revision-1"

    def __init__(self, task_count: int = 2) -> None:
        self.tasks = tuple(
            BenchmarkTask(TaskSpec(f"task-{index}", f"prompt-{index}"), evaluation={"private": "rubric"})
            for index in range(task_count)
        )
        self.prepare_calls = 0
        self.load_calls = 0
        self.materialize_calls = 0
        self.execution_task_calls = 0

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        self.prepare_calls += 1

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        self.load_calls += 1
        return list(self.tasks[:limit])

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        self.materialize_calls += 1
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "benchmark.txt").write_text(task.execution.task_id, encoding="utf-8")
        return ["benchmark.txt"]

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        del workspace, network_policy
        self.execution_task_calls += 1
        return task.execution


class _Evaluator(Evaluator):
    name = "generic-evaluator"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def __init__(self, *, fail_plan_at: int | None = None) -> None:
        self.fail_plan_at = fail_plan_at
        self.plan_calls: list[EvaluationPlan] = []
        self.evaluate_calls = 0

    def validate_plan(self, plan: EvaluationPlan) -> None:
        self.plan_calls.append(plan)
        if self.fail_plan_at is not None and len(self.plan_calls) == self.fail_plan_at:
            raise ValueError("private rubric sentinel")

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        return EvaluatorPreflightResult(self.name, self.evaluator_type, True)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.evaluate_calls += 1
        return EvaluationResult(request.task_id, EvaluationStatus.COMPLETED, {"accuracy": 1.0})


class _ApplicationExecutor(Executor):
    name = "generic-application"
    invocation_mode = "fake"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.preflight_calls = 0
        self.requests: list[ExecutionRequest] = []

    def preflight(self) -> PreflightResult:
        self.preflight_calls += 1
        return PreflightResult(self.name, True, version="application-1", auth_mode="test")

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        request.workspace.mkdir(parents=True, exist_ok=True)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        request.deliverables_dir.mkdir(parents=True, exist_ok=True)
        failed = self.fail_on_call == len(self.requests)
        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="application-1",
            invocation_mode=self.invocation_mode,
            auth_mode="test",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=ExecutionStatus.FAILED if failed else ExecutionStatus.COMPLETED,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=1 if failed else 0,
            output_text="private output sentinel",
            metadata={"private": "executor metadata sentinel"},
        )


class _Builder(Builder):
    name = "generic-builder"

    def __init__(self, *, fail_on_call: int | None = None, interrupt_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.interrupt_on_call = interrupt_on_call
        self.preflight_calls = 0
        self.requests: list[BuildRequest] = []

    def preflight(self) -> BuilderPreflightResult:
        self.preflight_calls += 1
        return BuilderPreflightResult(self.name, True, builder_executor="generic-builder-executor")

    def build(self, request: BuildRequest) -> BuildResult:
        self.requests.append(request)
        if self.interrupt_on_call == len(self.requests):
            raise KeyboardInterrupt
        request.runtime_root.mkdir(parents=True, exist_ok=False)
        request.artifact_root.mkdir(parents=True, exist_ok=False)
        if self.fail_on_call == len(self.requests):
            return BuildResult(
                build_run_id=request.build_run_id,
                task_id=request.task.task_id,
                builder=self.name,
                status=BuildStatus.FAILED,
                inputs=tuple(item.manifest for item in request.inputs),
                failure_phase="execution",
            )

        skill = request.artifact_root / "generic-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: generic-skill\ndescription: a generic test skill\n---\n\nUse the skill.\n",
            encoding="utf-8",
        )
        bundle = load_agent_skill_bundle(skill)
        workspace = request.runtime_root / "workspace"
        executor_dir = request.runtime_root / "executor"
        deliverables = workspace / "deliverables"
        workspace.mkdir()
        executor_dir.mkdir()
        deliverables.mkdir()
        execution = ExecutionResult(
            task_id=request.task.task_id,
            executor="generic-builder-executor",
            executor_version="builder-1",
            invocation_mode="fake",
            auth_mode="test",
            workspace=workspace,
            deliverables_dir=deliverables,
            status=ExecutionStatus.COMPLETED,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=0,
            output_text="private builder output sentinel",
            metadata={"private": "builder metadata sentinel"},
        )
        return BuildResult(
            build_run_id=request.build_run_id,
            task_id=request.task.task_id,
            builder=self.name,
            status=BuildStatus.COMPLETED,
            inputs=tuple(item.manifest for item in request.inputs),
            execution=execution,
            bundle=bundle,
        )


class BuilderExperimentRunnerTests(unittest.TestCase):
    def _fixture(self, root: Path, *, task_count: int = 2, fail_plan_at: int | None = None):
        source = root / "input-source"
        source.mkdir()
        (source / "guide.txt").write_text("allowlisted input", encoding="utf-8")
        source_bundle = load_builder_input_bundle(
            source,
            input_id="input-guide",
            input_type="reference",
            allowed_files=("guide.txt",),
        )
        profile_source = root / "generic-profile.json"
        profile_source.write_text("{}\n", encoding="utf-8")
        experiment_profile = ExperimentProfile(
            1,
            "generic-profile",
            _Benchmark.name,
            (
                ExperimentInputSpec(
                    "input-guide",
                    "reference",
                    None,
                    "unavailable",
                    ("guide.txt",),
                    source_bundle.manifest.bundle_sha256,
                ),
            ),
            (
                ExperimentArm("arm-first", ("input-guide",)),
                ExperimentArm("arm-second", ("input-guide",)),
            ),
        )
        profile = LoadedExperimentProfile(experiment_profile, profile_source, "a" * 64)
        config = ExperimentRunConfig(
            builder_executor="generic-builder-executor",
            application_executor="generic-application",
            evaluator="generic-evaluator",
            builder_model="builder-model",
            application_model="application-model",
            builder_timeout_seconds=1.0,
            application_timeout_seconds=2.0,
            builder_network_enabled=False,
            application_network_enabled=False,
            limit=max(task_count, 1),
            order_seed=19,
        )
        benchmark = _Benchmark(task_count)
        evaluator = _Evaluator(fail_plan_at=fail_plan_at)
        builder = _Builder()
        application = _ApplicationExecutor()
        return profile, config, benchmark, evaluator, builder, application, source_bundle

    def _run(self, root: Path, **kwargs: object):
        profile, config, benchmark, evaluator, builder, application, source_bundle = self._fixture(root, **kwargs)
        with patch(
            "gdpval_harness.experiments.runner.load_experiment_inputs",
            return_value={"input-guide": source_bundle},
        ) as input_loader, patch(
            "gdpval_harness.experiments.runner.secrets.token_hex",
            side_effect=[f"{index:032x}" for index in range(1, 100)],
        ), patch(
            "gdpval_harness.runner.secrets.token_urlsafe",
            side_effect=[f"application-run-{index}" for index in range(1, 100)],
        ):
            summary = run_builder_experiment(
                profile,
                config,
                benchmark,
                evaluator,
                builder,
                application,
                source_roots={"input-guide": root / "input-source"},
                out_dir=root / "output",
                runtime_root=root / "runtime",
            )
            input_loader.assert_called_once_with(profile.profile, {"input-guide": root / "input-source"})
        return summary, benchmark, evaluator, builder, application, source_bundle

    def test_generic_profile_runs_fixed_tasks_with_ordered_inputs_and_sealed_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, benchmark, evaluator, builder, application, source_bundle = self._run(root)
            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.task_count, 2)
            self.assertEqual(summary.arm_count, 2)
            self.assertEqual(summary.completed_applications, 4)
            self.assertEqual(benchmark.prepare_calls, 1)
            self.assertEqual(benchmark.load_calls, 1)
            self.assertEqual(benchmark.materialize_calls, 4)
            self.assertEqual(benchmark.execution_task_calls, 4)
            self.assertEqual(len(builder.requests), 4)
            self.assertEqual(len(application.requests), 4)
            self.assertGreaterEqual(len(evaluator.plan_calls), 2)
            self.assertTrue(all(plan.candidate_count == 1 for plan in evaluator.plan_calls))
            self.assertTrue(
                all(
                    request.inputs[0].manifest.input_id == "input-guide"
                    and any(request.task is task.execution for task in benchmark.tasks)
                    for request in builder.requests
                )
            )
            self.assertTrue(all(request.inputs[0] is builder.requests[0].inputs[0] for request in builder.requests))
            self.assertIs(builder.requests[0].inputs[0], source_bundle)
            self.assertTrue(all("private rubric" not in request.task.prompt for request in application.requests))
            self.assertTrue(all("private" not in json.dumps(request.environment) for request in application.requests))

            metadata = json.loads((root / "output" / "experiment-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_applications"], 4)
            self.assertNotIn("private", json.dumps(metadata))
            self.assertEqual(len(metadata["schedule"]), 4)
            self.assertTrue(all(entry["application"]["status"] == "completed" for entry in metadata["entries"]))
            ids = [entry["schedule_id"] for entry in metadata["schedule"]]
            self.assertEqual(len(ids), len(set(ids)))
            for entry in metadata["schedule"]:
                self.assertIn("output/applications/", entry["output_root"])
                self.assertIn("runtime/", entry["application_root"])
                self.assertNotIn("arm-first", json.dumps(entry["output_root"]))
                self.assertNotIn("generic-profile", json.dumps(entry["application_root"]))
                nested_metadata = json.loads(
                    (Path(entry["output_root"]) / "run-metadata.json").read_text(encoding="utf-8")
                )
                nested_row = json.loads(
                    (Path(entry["output_root"]) / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
                )
                self.assertNotEqual(entry["schedule_id"], nested_row["intervention"]["application_run_id"])
                self.assertEqual(nested_row["evaluation"]["metrics"]["accuracy"], 1.0)
                self.assertEqual(nested_metadata["status"], "completed")

            for request in application.requests:
                serialized_prompt = request.task.prompt
                serialized_environment = json.dumps(request.environment)
                workspace_contents = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in request.workspace.rglob("*")
                    if path.is_file()
                )
                for sentinel in (
                    "private builder output sentinel",
                    "builder metadata sentinel",
                    "private rubric sentinel",
                    "session",
                    "resume",
                ):
                    self.assertNotIn(sentinel, serialized_prompt)
                    self.assertNotIn(sentinel, serialized_environment)
                    self.assertNotIn(sentinel, workspace_contents)

    def test_plan_failure_happens_before_builder_or_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, builder, application, source_bundle = self._fixture(
                root, fail_plan_at=2
            )
            with patch(
                "gdpval_harness.experiments.runner.load_experiment_inputs",
                return_value={"input-guide": source_bundle},
            ):
                with self.assertRaisesRegex(ValueError, "private rubric sentinel"):
                    run_builder_experiment(
                        profile,
                        config,
                        benchmark,
                        evaluator,
                        builder,
                        application,
                        source_roots={"input-guide": root / "input-source"},
                        out_dir=root / "output",
                        runtime_root=root / "runtime",
                    )
            self.assertEqual(builder.requests, [])
            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "runtime").exists())

    def test_empty_selection_fails_before_builder_or_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, builder, application, source_bundle = self._fixture(
                root, task_count=0
            )
            with patch(
                "gdpval_harness.experiments.runner.load_experiment_inputs",
                return_value={"input-guide": source_bundle},
            ):
                with self.assertRaisesRegex(ValueError, "no selected tasks"):
                    run_builder_experiment(
                        profile,
                        config,
                        benchmark,
                        evaluator,
                        builder,
                        application,
                        source_roots={"input-guide": root / "input-source"},
                        out_dir=root / "output",
                        runtime_root=root / "runtime",
                    )
            self.assertEqual(builder.requests, [])
            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "runtime").exists())

    def test_task_path_sanitization_is_shared_with_application_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                profile,
                config,
                benchmark,
                evaluator,
                builder,
                application,
                source_bundle,
            ) = self._fixture(root, task_count=1)
            benchmark.tasks = (BenchmarkTask(TaskSpec("unsafe/task", "prompt"), evaluation={}),)
            with patch(
                "gdpval_harness.experiments.runner.load_experiment_inputs",
                return_value={"input-guide": source_bundle},
            ), patch(
                "gdpval_harness.experiments.runner.secrets.token_hex",
                side_effect=[f"{index:032x}" for index in range(1, 100)],
            ), patch(
                "gdpval_harness.runner.secrets.token_urlsafe",
                side_effect=[f"application-run-{index}" for index in range(1, 100)],
            ):
                summary = run_builder_experiment(
                    profile,
                    config,
                    benchmark,
                    evaluator,
                    builder,
                    application,
                    source_roots={"input-guide": root / "input-source"},
                    out_dir=root / "output",
                    runtime_root=root / "runtime",
                )
            self.assertEqual(summary.status, "completed")
            self.assertEqual([request.task.task_id for request in builder.requests], ["unsafe/task", "unsafe/task"])
            self.assertTrue(all("unsafe_task" in str(request.workspace) for request in application.requests))

    def test_seed_reproduces_pair_order_independently_of_schedule_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, _, _, _, _ = self._fixture(root)
            with patch(
                "gdpval_harness.experiments.runner.secrets.token_hex",
                side_effect=[f"first-{index:028x}" for index in range(4)],
            ):
                first = _make_schedule(
                    benchmark.tasks,
                    profile.profile.arms,
                    config.order_seed,
                    root / "first-output",
                    root / "first-runtime",
                )
            with patch(
                "gdpval_harness.experiments.runner.secrets.token_hex",
                side_effect=[f"second-{index:027x}" for index in range(4)],
            ):
                second = _make_schedule(
                    benchmark.tasks,
                    profile.profile.arms,
                    config.order_seed,
                    root / "second-output",
                    root / "second-runtime",
                )
            self.assertEqual(
                [(item.task.execution.task_id, item.arm.arm_id) for item in first],
                [(item.task.execution.task_id, item.arm.arm_id) for item in second],
            )
            self.assertNotEqual(
                [item.schedule_id for item in first],
                [item.schedule_id for item in second],
            )

    def test_build_failure_stops_schedule_and_durably_records_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, _, application, source_bundle = self._fixture(root)
            builder = _Builder(fail_on_call=2)
            with patch(
                "gdpval_harness.experiments.runner.load_experiment_inputs",
                return_value={"input-guide": source_bundle},
            ), patch(
                "gdpval_harness.experiments.runner.secrets.token_hex",
                side_effect=[f"{index:032x}" for index in range(1, 100)],
            ), patch(
                "gdpval_harness.runner.secrets.token_urlsafe",
                side_effect=[f"application-run-{index}" for index in range(1, 100)],
            ):
                summary = run_builder_experiment(
                    profile,
                    config,
                    benchmark,
                    evaluator,
                    builder,
                    application,
                    source_roots={"input-guide": root / "input-source"},
                    out_dir=root / "output",
                    runtime_root=root / "runtime",
                )
            self.assertEqual(summary.status, "failed")
            self.assertEqual(summary.completed_applications, 1)
            self.assertEqual(len(builder.requests), 2)
            self.assertEqual(len(application.requests), 1)
            metadata = json.loads((root / "output" / "experiment-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["completed_applications"], 1)
            self.assertEqual(metadata["entries"][1]["build"]["status"], "failed")
            self.assertIsNone(metadata["entries"][2]["build"])

    def test_builder_bundle_must_be_one_skill_directory_under_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, benchmark, _, builder, _, source_bundle = self._fixture(root)
            request = BuildRequest(
                build_run_id="schedule-id",
                task=benchmark.tasks[0].execution,
                inputs=(source_bundle,),
                runtime_root=root / "build-runtime",
                artifact_root=root / "build-artifacts",
            )
            result = builder.build(request)
            self.assertIsNotNone(result.bundle)
            assert result.bundle is not None
            invalid = BuildResult(
                build_run_id=result.build_run_id,
                task_id=result.task_id,
                builder=result.builder,
                status=result.status,
                inputs=result.inputs,
                execution=result.execution,
                bundle=InterventionBundle(root=request.artifact_root, manifest=result.bundle.manifest),
            )
            with self.assertRaisesRegex(ValueError, "outside the assigned artifact root"):
                _validate_build_result(invalid, request, builder)

    def test_builder_interrupt_is_durable_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, _, application, source_bundle = self._fixture(root)
            builder = _Builder(interrupt_on_call=1)
            with patch(
                "gdpval_harness.experiments.runner.load_experiment_inputs",
                return_value={"input-guide": source_bundle},
            ), patch(
                "gdpval_harness.experiments.runner.secrets.token_hex",
                side_effect=[f"{index:032x}" for index in range(1, 100)],
            ), patch(
                "gdpval_harness.runner.secrets.token_urlsafe",
                side_effect=[f"application-run-{index}" for index in range(1, 100)],
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_builder_experiment(
                        profile,
                        config,
                        benchmark,
                        evaluator,
                        builder,
                        application,
                        source_roots={"input-guide": root / "input-source"},
                        out_dir=root / "output",
                        runtime_root=root / "runtime",
                    )
            metadata = json.loads((root / "output" / "experiment-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "interrupted")
            self.assertEqual(metadata["entries"][0]["build"]["status"], "interrupted")
            self.assertEqual(len(application.requests), 0)

    def test_application_failure_stops_remaining_schedule_and_persists_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, builder, _, source_bundle = self._fixture(root)
            application = _ApplicationExecutor(fail_on_call=2)
            with patch(
                "gdpval_harness.experiments.runner.load_experiment_inputs",
                return_value={"input-guide": source_bundle},
            ), patch(
                "gdpval_harness.experiments.runner.secrets.token_hex",
                side_effect=[f"{index:032x}" for index in range(1, 100)],
            ), patch(
                "gdpval_harness.runner.secrets.token_urlsafe",
                side_effect=[f"application-run-{index}" for index in range(1, 100)],
            ):
                summary = run_builder_experiment(
                    profile,
                    config,
                    benchmark,
                    evaluator,
                    builder,
                    application,
                    source_roots={"input-guide": root / "input-source"},
                    out_dir=root / "output",
                    runtime_root=root / "runtime",
                )
            self.assertEqual(summary.status, "failed")
            self.assertEqual(summary.completed_applications, 1)
            self.assertEqual(len(builder.requests), 2)
            self.assertEqual(len(application.requests), 2)
            metadata = json.loads((root / "output" / "experiment-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["completed_applications"], 1)
            self.assertEqual(metadata["entries"][0]["application"]["status"], "completed")
            self.assertEqual(metadata["entries"][1]["application"]["status"], "failed")
            self.assertIsNone(metadata["entries"][2]["build"])
            failed_output = Path(metadata["entries"][1]["application"]["output_root"])
            self.assertEqual(
                json.loads((failed_output / "run-metadata.json").read_text(encoding="utf-8"))["status"],
                "failed",
            )
            self.assertTrue((failed_output / "results.jsonl").is_file())

    def test_loader_and_namespace_failures_precede_builder_and_root_creation(self) -> None:
        cases = ("loader", "overlap", "existing-output")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                profile, config, benchmark, evaluator, builder, application, source_bundle = self._fixture(root)
                out_dir = root / "output"
                runtime = root / "runtime"
                if case == "overlap":
                    out_dir = root / "input-source" / "planned-output"
                elif case == "existing-output":
                    out_dir.mkdir()
                    (out_dir / "sentinel").write_text("keep", encoding="utf-8")
                loader = patch(
                    "gdpval_harness.experiments.runner.load_experiment_inputs",
                    side_effect=ValueError("input hash sentinel"),
                )
                with loader as mocked_loader:
                    with self.assertRaises((ValueError, FileExistsError)):
                        run_builder_experiment(
                            profile,
                            config,
                            benchmark,
                            evaluator,
                            builder,
                            application,
                            source_roots={"input-guide": root / "input-source"},
                            out_dir=out_dir,
                            runtime_root=runtime,
                        )
                self.assertEqual(builder.requests, [])
                if case == "loader":
                    mocked_loader.assert_called_once()
                    self.assertFalse(out_dir.exists())
                    self.assertFalse(runtime.exists())
                elif case == "overlap":
                    self.assertFalse((out_dir / "experiment-metadata.json").exists())
                    self.assertFalse(runtime.exists())
                else:
                    self.assertEqual((out_dir / "sentinel").read_text(encoding="utf-8"), "keep")
                    self.assertFalse(runtime.exists())


if __name__ == "__main__":
    unittest.main()
