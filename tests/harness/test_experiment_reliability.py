# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import eval_harness.experiments.base as base_module
import eval_harness.experiments.profile as profile_module
import eval_harness.experiments.runner as runner_module
from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.builders.base import (
    Builder,
    BuilderInputBundle,
    BuilderPreflightResult,
    BuildRequest,
    BuildResult,
    BuildStatus,
)
from eval_harness.builders.executor_skill import ExecutorSkillBuilder
from eval_harness.builders.inputs import load_builder_input_bundle
from eval_harness.capabilities import ExecutorOutput
from eval_harness.evaluators.base import (
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
    EvaluatorType,
)
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from eval_harness.experiments.base import (
    ExperimentArm,
    ExperimentInputSpec,
    ExperimentProfile,
    ExperimentRunConfig,
    ExperimentRunSummary,
    LoadedExperimentProfile,
)
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.interventions import load_agent_skill_bundle
from eval_harness.provenance import RepositoryProvenance, canonical_json_sha256
from eval_harness.runner import RunSummary


class ReliabilityBenchmark(Benchmark):
    name = "reliability-benchmark"
    revision = "revision"

    def __init__(self, tasks: tuple[BenchmarkTask, ...]) -> None:
        self.tasks = tasks
        self.prepare_calls = 0

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        self.prepare_calls += 1

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        return list(self.tasks[:limit])

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        del task, workspace
        return []

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        del workspace, network_policy
        return task.execution


class ReliabilityEvaluator(Evaluator):
    name = "reliability-evaluator"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        return EvaluatorPreflightResult(self.name, self.evaluator_type, True, version="1")

    def validate_plan(self, plan: EvaluationPlan) -> None:
        del plan

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult(request.task_id, EvaluationStatus.COMPLETED, metrics={"score": 1.0})


class ReliabilityApplicationExecutor(Executor):
    name = "reliability-application"
    invocation_mode = "deterministic"

    def __init__(
        self,
        *,
        status: ExecutionStatus = ExecutionStatus.NO_DELIVERABLE,
        raise_on_execute: BaseException | None = None,
    ) -> None:
        self.status = status
        self.raise_on_execute = raise_on_execute
        self.requests: list[ExecutionRequest] = []

    def preflight(self) -> PreflightResult:
        return PreflightResult(self.name, True, version="1", auth_mode="local")

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        request.deliverables_dir.mkdir(parents=True, exist_ok=True)
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        successful = self.status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="1",
            invocation_mode=self.invocation_mode,
            auth_mode="local",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=self.status,
            started_at="2026-09-12T00:00:00+00:00",
            finished_at="2026-09-12T00:00:01+00:00",
            exit_code=0 if successful else 1,
            available_outputs=frozenset(),
            failure=None if successful else Failure(FailureKind.PROCESS, "test_failure", FailureImpact.RUN),
        )


class ReliabilityBuilderExecutor(Executor):
    name = "reliability-builder-executor"
    invocation_mode = "deterministic"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    def preflight(self) -> PreflightResult:
        return PreflightResult(self.name, True, version="1", auth_mode="local")

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        request.deliverables_dir.mkdir(parents=True, exist_ok=True)
        skill = request.deliverables_dir / "reliability-built-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: reliability-built-skill\ndescription: reliable generated skill\n---\n\nUse it.\n",
            encoding="utf-8",
        )
        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="1",
            invocation_mode=self.invocation_mode,
            auth_mode="local",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=ExecutionStatus.COMPLETED,
            started_at="2026-09-12T00:00:00+00:00",
            finished_at="2026-09-12T00:00:01+00:00",
            exit_code=0,
            available_outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
            failure=None,
        )


class ReliabilityBuilder(Builder):
    name = "reliability-builder"

    def preflight(self) -> BuilderPreflightResult:
        return BuilderPreflightResult(
            name=self.name,
            ok=True,
            builder_executor="reliability-executor",
            builder_executor_version="1",
        )

    def build(self, request: BuildRequest) -> BuildResult:
        raise RuntimeError("builder boundary failure")


class ExperimentReliabilityTests(unittest.TestCase):
    def _profile(self, root: Path, *, revision_status: str = "unavailable") -> LoadedExperimentProfile:
        source = root / "profile.json"
        source.write_text("{}\n", encoding="utf-8")
        revision = None if revision_status == "unavailable" else "a" * 40
        spec = ExperimentInputSpec("input-one", "reference", revision, revision_status, ("guide.txt",))
        profile = ExperimentProfile(
            1,
            "reliability-profile",
            "reliability-benchmark",
            (spec,),
            (ExperimentArm("arm-a", ("input-one",)),),
        )
        return LoadedExperimentProfile(profile, source, "a" * 64)

    def _source_and_bundle(self, root: Path) -> tuple[Path, BuilderInputBundle]:
        source = root / "source"
        source.mkdir()
        (source / "guide.txt").write_text("guide", encoding="utf-8")
        bundle = load_builder_input_bundle(
            source,
            input_id="input-one",
            input_type="reference",
            allowed_files=("guide.txt",),
        )
        return source, bundle

    def _run_fixture(
        self,
        root: Path,
        *,
        application_status: ExecutionStatus = ExecutionStatus.COMPLETED,
        application_error: BaseException | None = None,
    ) -> tuple[
        LoadedExperimentProfile,
        ExperimentRunConfig,
        ReliabilityBenchmark,
        ReliabilityEvaluator,
        ExecutorSkillBuilder,
        ReliabilityBuilderExecutor,
        ReliabilityApplicationExecutor,
        Path,
    ]:
        root.mkdir(parents=True, exist_ok=True)
        source, _ = self._source_and_bundle(root)
        profile = self._profile(root)
        config = ExperimentRunConfig(
            "reliability-builder-executor",
            "reliability-application",
            "reliability-evaluator",
            "builder-model",
            "application-model",
            1.0,
            2.0,
            False,
            False,
            1,
            17,
        )
        benchmark = ReliabilityBenchmark(
            (BenchmarkTask(TaskSpec("task-one", "target prompt"), evaluation={"private": "rubric"}),)
        )
        evaluator = ReliabilityEvaluator()
        builder_executor = ReliabilityBuilderExecutor()
        builder = ExecutorSkillBuilder(builder_executor)
        application = ReliabilityApplicationExecutor(status=application_status, raise_on_execute=application_error)
        return profile, config, benchmark, evaluator, builder, builder_executor, application, source

    def test_contracts_validate_identifiers_paths_arms_and_run_limits(self) -> None:
        for value in ("", "Upper", "with space", ".", "a" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                base_module._require_identifier("id", value)
        for value in ("", "../escape", "/absolute", "nested\\file", "a//b", "a/./b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                base_module._validate_allowed_file_path(value)
        with self.assertRaises(ValueError):
            base_module._validate_allowed_file_path("\ud800")
        with self.assertRaises(TypeError):
            base_module._as_tuple("items", "not-a-sequence")
        with self.assertRaises(TypeError):
            base_module._as_tuple("items", object())
        with self.assertRaises(TypeError):
            base_module._normalize_string_sequence("names", ("ok", 1))
        with self.assertRaises(ValueError):
            base_module._normalize_allowed_files(("a.txt", "A.txt"))
        with self.assertRaises(ValueError):
            base_module._normalize_allowed_files(("a", "a/b"))
        with self.assertRaises(TypeError):
            base_module._normalize_path("path", object())
        timeout_values: tuple[object, ...] = (True, 0, math.inf, -1.0)
        for timeout_value in timeout_values:
            with self.subTest(value=timeout_value), self.assertRaises((TypeError, ValueError)):
                base_module._require_timeout("timeout", timeout_value)
        with self.assertRaises(TypeError):
            base_module._require_bool("flag", "true")
        with self.assertRaises(ValueError):
            ExperimentInputSpec("input-one", "reference", None, "available", ("guide.txt",))
        with self.assertRaises(ValueError):
            ExperimentInputSpec("input-one", "reference", "revision", "unavailable", ("guide.txt",))
        with self.assertRaises(ValueError):
            ExperimentArm("arm-a", ("input-one", "input-one"))
        spec = ExperimentInputSpec("input-one", "reference", None, "unavailable", ("guide.txt",))
        with self.assertRaises(ValueError):
            ExperimentProfile(1, "profile", "benchmark", (spec, spec), (ExperimentArm("arm-a", ("input-one",)),))
        with self.assertRaises(ValueError):
            ExperimentProfile(1, "profile", "benchmark", (spec,), (ExperimentArm("arm-a", ("missing",)),))
        with self.assertRaises(ValueError):
            ExperimentProfile(1, "profile", "benchmark", (spec,), (ExperimentArm("arm-a", ()),))
        with self.assertRaises(ValueError):
            ExperimentRunSummary("profile", "benchmark", Path("out"), Path("runtime"), "completed", 1, 1, 2)

    def test_profile_loader_rejects_duplicate_nonfinite_unicode_and_shape_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = {
                "schema_version": 1,
                "profile_id": "profile",
                "benchmark": "benchmark",
                "inputs": [
                    {
                        "input_id": "input-one",
                        "input_type": "reference",
                        "source_revision": None,
                        "revision_status": "unavailable",
                        "allowed_files": ["guide.txt"],
                    }
                ],
                "arms": [{"arm_id": "arm-a", "builder_inputs": ["input-one"]}],
            }
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(valid), encoding="utf-8")
            loaded = profile_module.load_experiment_profile(profile_path)
            self.assertEqual(loaded.profile.profile_id, "profile")
            self.assertEqual(loaded.sha256, hashlib.sha256(profile_path.read_bytes()).hexdigest())

            invalid_payloads = (
                b'{"schema_version":1,"schema_version":1}',
                b"NaN",
                b'{"schema_version":1,"profile_id":"p","benchmark":"b","inputs":[],"arms":[],"extra":true}',
                b'{"schema_version":1,"profile_id":"p","benchmark":"b","inputs":{},"arms":[]}',
                b"\xff\xfe",
            )
            for index, raw in enumerate(invalid_payloads):
                path = root / f"invalid-{index}.json"
                path.write_bytes(raw)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    profile_module.load_experiment_profile(path)
            with self.assertRaises(profile_module._NonFiniteNumber):
                profile_module._parse_finite_float("inf")
            with self.assertRaises(profile_module._NonFiniteNumber):
                profile_module._reject_non_finite_number("NaN")
            with self.assertRaises(ValueError):
                profile_module._decode_profile(json.dumps("\ud800").encode("utf-8"))

            missing = root / "missing.json"
            with self.assertRaises(ValueError):
                profile_module._canonical_profile_path(missing)
            directory = root / "directory"
            directory.mkdir()
            with self.assertRaises(ValueError):
                profile_module._canonical_profile_path(directory)
            alias = root / "profile-alias.json"
            alias.symlink_to(profile_path)
            with self.assertRaises(ValueError):
                profile_module._canonical_profile_path(alias)
            oversized = root / "oversized.json"
            oversized.write_bytes(b"x" * (1 * 1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                profile_module._read_profile_bytes(oversized)

    def test_profile_input_binding_and_pinned_git_validation_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle = self._source_and_bundle(root)
            profile = self._profile(root)
            with self.assertRaises(ValueError):
                profile_module._validate_source_bindings(profile.profile, {})
            with self.assertRaises(ValueError):
                profile_module._validate_source_bindings(profile.profile, {"wrong": source})
            bindings, specs = profile_module._validate_source_bindings(profile.profile, {"input-one": source})
            self.assertEqual(bindings, {"input-one": source})
            self.assertEqual(tuple(spec.input_id for spec in specs), ("input-one",))

            loaded = profile_module.load_experiment_inputs(profile.profile, {"input-one": source})
            self.assertEqual(tuple(loaded), ("input-one",))
            self.assertEqual(loaded["input-one"].manifest.bundle_sha256, bundle.manifest.bundle_sha256)
            with self.assertRaises(ValueError):
                profile_module.load_experiment_inputs(profile.profile, {})
            wrong_spec = ExperimentInputSpec("input-one", "wrong-type", None, "unavailable", ("guide.txt",))
            with self.assertRaises(ValueError):
                profile_module._validate_loaded_bundle(bundle, wrong_spec)

            revision = "a" * 40
            head = subprocess.CompletedProcess[bytes](args=["git"], returncode=0, stdout=(revision + "\n").encode())
            blob = subprocess.CompletedProcess[bytes](args=["git"], returncode=0, stdout=b"guide")
            with patch.object(subprocess, "run", side_effect=[head, blob, head, blob]):
                profile_module._git_head(source, revision)
                self.assertEqual(profile_module._git_commit_blob(source, revision, "guide.txt"), b"guide")
            bad_head = subprocess.CompletedProcess[bytes](args=["git"], returncode=1, stdout=b"")
            with patch.object(subprocess, "run", return_value=bad_head), self.assertRaises(ValueError):
                profile_module._git_head(source, revision)
            bad_blob = subprocess.CompletedProcess[bytes](args=["git"], returncode=1, stdout=b"")
            with patch.object(subprocess, "run", return_value=bad_blob), self.assertRaises(ValueError):
                profile_module._git_commit_blob(source, revision, "guide.txt")

            available = self._profile(root, revision_status="available")
            with (
                patch.object(profile_module, "_git_head"),
                patch.object(profile_module, "_git_commit_blob", return_value=b"guide"),
            ):
                loaded_available = profile_module.load_experiment_inputs(available.profile, {"input-one": source})
            self.assertEqual(loaded_available["input-one"].manifest.source_revision, revision)

    def test_runner_validation_helpers_preserve_root_and_identity_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = self._profile(root)
            source, bundle = self._source_and_bundle(root)
            benchmark_task = BenchmarkTask(TaskSpec("task-one", "prompt"))
            benchmark = ReliabilityBenchmark((benchmark_task,))
            arm = profile.profile.arms[0]
            config = ExperimentRunConfig(
                "builder",
                "application",
                "evaluator",
                None,
                None,
                1.0,
                1.0,
                False,
                False,
                1,
                5,
            )
            del arm, config
            for value in ("", ".", "..", "a/b", "a\\b", "a\x00b"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    runner_module._safe_schedule_id(value)
            self.assertEqual(runner_module._safe_schedule_id("schedule-1"), "schedule-1")
            with self.assertRaises(ValueError):
                runner_module._canonical_planned_root(Path("relative"), label="root")
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                runner_module._canonical_planned_root(existing, label="root")
            profile_source = root / "profile-source"
            profile_source.mkdir()
            with self.assertRaises(ValueError):
                runner_module._validate_path_namespace(root / "planned", root / "planned", profile_source, {})
            with self.assertRaises(ValueError):
                runner_module._validate_path_namespace(
                    source / "planned", root / "runtime", root / "profile.json", {"input-one": source}
                )

            schedule = runner_module._make_schedule(
                (benchmark_task,), profile.profile.arms, 5, root / "out", root / "runtime"
            )
            self.assertEqual(len(schedule), 1)
            item = schedule[0]
            summary = RunSummary(
                benchmark.name,
                "application",
                item.output_root,
                item.application_root,
                "completed",
                1,
                {},
                {},
                {benchmark_task.execution.task_id: "application-run"},
            )
            self.assertEqual(runner_module._application_run_id(summary, item), "application-run")
            for application_ids in ({}, {"other": "run"}, {"task-one": ""}, {"task-one": item.schedule_id}):
                invalid_summary = replace_run_summary(summary, application_ids)
                with self.subTest(application_ids=application_ids), self.assertRaises((TypeError, ValueError)):
                    runner_module._application_run_id(invalid_summary, item)

            self.assertEqual(runner_module._entry({"entries": [{"ok": True}]}, 0), {"ok": True})
            with self.assertRaises(RuntimeError):
                runner_module._entry({}, 0)
            with self.assertRaises(RuntimeError):
                runner_module._entry({"entries": ["bad"]}, 0)

            input_specs = {"input-one": profile.profile.inputs[0]}
            loaded_inputs, canonical = runner_module._validate_loaded_inputs(
                profile,
                {"input-one": source},
                {"input-one": bundle},
            )
            self.assertEqual(loaded_inputs["input-one"], bundle)
            self.assertEqual(canonical["input-one"], source.resolve())
            del input_specs
            loaded_values: tuple[object, ...] = ([], {"other": bundle})
            for loaded in loaded_values:
                with self.subTest(loaded=loaded), self.assertRaises((TypeError, ValueError)):
                    runner_module._validate_loaded_inputs(profile, {"input-one": source}, loaded)

    def test_runner_payload_and_build_result_validation_are_typed_and_path_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = self._profile(root)
            source, bundle = self._source_and_bundle(root)
            task = BenchmarkTask(TaskSpec("task-one", "prompt"))
            benchmark = ReliabilityBenchmark((task,))
            config = ExperimentRunConfig(
                "builder", "application", "evaluator", "model", None, 1.0, 2.0, False, True, 1, 1
            )
            builder_preflight = BuilderPreflightResult("reliability-builder", True, "builder", "1")
            application_preflight = PreflightResult("application", True, "1", "local")
            evaluator_preflight = EvaluatorPreflightResult(
                "evaluator", EvaluatorType.BENCHMARK_NATIVE, True, version="1"
            )
            builder = ReliabilityBuilder()
            application = ReliabilityApplicationExecutor()
            configuration = runner_module._configuration_payload(
                profile,
                config,
                benchmark,
                {"input-one": bundle},
                builder,
                builder_preflight,
                application_preflight,
                evaluator_preflight,
                application,
            )
            builder_descriptor = configuration["builder"]
            evaluator_descriptor = configuration["evaluator"]
            if not isinstance(builder_descriptor, dict) or not isinstance(evaluator_descriptor, dict):
                raise AssertionError("configuration descriptors must be mappings")
            self.assertEqual(builder_descriptor["id"], "reliability-builder")
            evaluator_judge = evaluator_descriptor["judge"]
            if not isinstance(evaluator_judge, dict):
                raise AssertionError("evaluator judge descriptor must be a mapping")
            self.assertEqual(evaluator_judge["applicable"], False)
            self.assertEqual(canonical_json_sha256(configuration), canonical_json_sha256(configuration))

            runtime = root / "runtime"
            runtime.mkdir()
            artifact = runtime / "artifact"
            artifact.mkdir()
            skill = artifact / "skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("---\nname: skill\ndescription: skill\n---\n\ncontent\n", encoding="utf-8")
            sealed = load_agent_skill_bundle(skill)
            execution = ExecutionResult(
                task_id="task-one",
                executor="builder",
                executor_version="1",
                invocation_mode="deterministic",
                auth_mode="local",
                workspace=runtime,
                deliverables_dir=skill,
                status=ExecutionStatus.COMPLETED,
                started_at="start",
                finished_at="finish",
                exit_code=0,
                available_outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
                failure=None,
            )
            request = BuildRequest(
                "schedule",
                task.execution,
                (bundle,),
                runtime,
                artifact,
            )
            result = BuildResult(
                "schedule",
                "task-one",
                "reliability-builder",
                BuildStatus.COMPLETED,
                (bundle.manifest,),
                execution,
                sealed,
            )
            self.assertIs(runner_module._validate_build_result(result, request, builder), result)
            for invalid in (
                replace(result, build_run_id="other"),
                replace(result, task_id="other", execution=replace(execution, task_id="other")),
                replace(result, builder="other"),
                replace(result, inputs=()),
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    runner_module._validate_build_result(invalid, request, builder)

            self.assertEqual(runner_module._build_payload(result)["status"], "completed")
            self.assertEqual(
                runner_module._application_payload(
                    runner_module._make_schedule((task,), profile.profile.arms, 1, root / "o", root / "r")[0], "failed"
                )["status"],
                "failed",
            )
            self.assertEqual(runner_module._nullable_text(" text "), " text ")
            self.assertIsNone(runner_module._nullable_text(""))
            self.assertIsNone(runner_module._enum_text(object()))

    def test_builder_experiment_persists_task_provenance_artifacts_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, builder, builder_executor, application, source = self._run_fixture(
                root
            )
            repository = RepositoryProvenance("a" * 40, "available", "clean")
            with (
                patch.object(runner_module.secrets, "token_hex", return_value="schedule-0000000000000001"),
                patch("eval_harness.runner.secrets.token_urlsafe", return_value="application-run-1"),
                patch.object(runner_module, "repository_provenance", return_value=repository),
            ):
                summary = runner_module.run_builder_experiment(
                    profile,
                    config,
                    benchmark,
                    evaluator,
                    builder,
                    application,
                    source_roots={"input-one": source},
                    out_dir=root / "output",
                    runtime_root=root / "runtime",
                )

            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.completed_applications, 1)
            self.assertEqual(len(builder_executor.requests), 1)
            self.assertEqual(len(application.requests), 1)
            self.assertEqual(builder_executor.requests[0].task.task_id, "task-one")
            self.assertEqual(application.requests[0].task.task_id, "task-one")
            self.assertIn("target prompt", builder_executor.requests[0].task.prompt)
            self.assertNotIn("input-one", builder_executor.requests[0].task.prompt)
            self.assertNotIn(str(source), builder_executor.requests[0].task.prompt)

            metadata_path = root / "output" / "experiment-metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_applications"], 1)
            self.assertEqual(
                metadata["repository"],
                {"commit": "a" * 40, "revision_status": "available", "worktree_status": "clean"},
            )
            self.assertEqual(
                metadata["run_fingerprint_sha256"],
                canonical_json_sha256(
                    {
                        "configuration_sha256": metadata["configuration_sha256"],
                        "repository": metadata["repository"],
                        "tasks": metadata["tasks"],
                    }
                ),
            )
            self.assertEqual(metadata["entries"][0]["build"]["status"], "completed")
            self.assertEqual(metadata["entries"][0]["application"]["status"], "completed")
            artifact_root = Path(metadata["schedule"][0]["artifact_root"])
            self.assertTrue((artifact_root / "reliability-built-skill" / "SKILL.md").is_file())
            application_output = Path(metadata["entries"][0]["application"]["output_root"])
            result_row = json.loads((application_output / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(result_row["task_id"], "task-one")
            self.assertEqual(result_row["evaluation"]["metrics"], {"score": 1.0})
            self.assertEqual(result_row["intervention"]["application_run_id"], "application-run-1")
            self.assertEqual(
                json.loads((application_output / "run-metadata.json").read_text(encoding="utf-8"))["metrics"],
                {"score": 1.0},
            )
            self.assertEqual((source / "guide.txt").read_text(encoding="utf-8"), "guide")

            with self.assertRaises(FileExistsError):
                runner_module.run_builder_experiment(
                    profile,
                    config,
                    benchmark,
                    evaluator,
                    builder,
                    application,
                    source_roots={"input-one": source},
                    out_dir=root / "output",
                    runtime_root=root / "runtime",
                )
            self.assertEqual(len(builder_executor.requests), 1)

    def test_builder_experiment_application_failure_persists_partial_state_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, builder, builder_executor, application, source = self._run_fixture(
                root,
                application_status=ExecutionStatus.FAILED,
            )
            with (
                patch.object(runner_module.secrets, "token_hex", return_value="schedule-0000000000000002"),
                patch("eval_harness.runner.secrets.token_urlsafe", return_value="application-run-failed"),
            ):
                summary = runner_module.run_builder_experiment(
                    profile,
                    config,
                    benchmark,
                    evaluator,
                    builder,
                    application,
                    source_roots={"input-one": source},
                    out_dir=root / "output",
                    runtime_root=root / "runtime",
                )

            self.assertEqual(summary.status, "failed")
            self.assertEqual(summary.completed_applications, 0)
            self.assertEqual(len(builder_executor.requests), 1)
            self.assertEqual(len(application.requests), 1)
            metadata = json.loads((root / "output" / "experiment-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["completed_applications"], 0)
            self.assertEqual(metadata["entries"][0]["build"]["status"], "completed")
            self.assertEqual(metadata["entries"][0]["application"]["status"], "failed")
            output = Path(metadata["entries"][0]["application"]["output_root"])
            row = json.loads((output / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["execution"]["status"], "failed")
            self.assertEqual(row["evaluation"]["metrics"], {"score": 1.0})

    def test_builder_experiment_builder_and_application_exceptions_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, config, benchmark, evaluator, _, _, application, source = self._run_fixture(root)
            failing_builder = ReliabilityBuilder()
            config = replace(config, builder_executor="reliability-executor")
            with patch.object(runner_module.secrets, "token_hex", return_value="schedule-0000000000000003"):
                with self.assertRaisesRegex(RuntimeError, "builder boundary failure"):
                    runner_module.run_builder_experiment(
                        profile,
                        config,
                        benchmark,
                        evaluator,
                        failing_builder,
                        application,
                        source_roots={"input-one": source},
                        out_dir=root / "builder-output",
                        runtime_root=root / "builder-runtime",
                    )
            builder_metadata = json.loads(
                (root / "builder-output" / "experiment-metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(builder_metadata["status"], "failed")
            self.assertEqual(builder_metadata["completed_applications"], 0)
            self.assertEqual(builder_metadata["entries"][0]["build"], {"status": "failed", "phase": "exception"})
            self.assertIsNone(builder_metadata["entries"][0]["application"])
            self.assertFalse(application.requests)

            profile, config, benchmark, evaluator, builder, _, _, source = self._run_fixture(
                root / "application", application_error=RuntimeError("application boundary")
            )
            failing_application = ReliabilityApplicationExecutor(raise_on_execute=RuntimeError("application boundary"))
            with (
                patch.object(runner_module.secrets, "token_hex", return_value="schedule-0000000000000004"),
                patch("eval_harness.runner.secrets.token_urlsafe", return_value="application-run-error"),
            ):
                with self.assertRaisesRegex(RuntimeError, "application boundary"):
                    runner_module.run_builder_experiment(
                        profile,
                        config,
                        benchmark,
                        evaluator,
                        builder,
                        failing_application,
                        source_roots={"input-one": source},
                        out_dir=root / "application-output",
                        runtime_root=root / "application-runtime",
                    )
            application_metadata = json.loads(
                (root / "application-output" / "experiment-metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(application_metadata["status"], "failed")
            self.assertEqual(application_metadata["entries"][0]["build"]["status"], "completed")
            self.assertEqual(application_metadata["entries"][0]["application"]["status"], "failed")


def replace_run_summary(summary: RunSummary, application_run_ids: object) -> RunSummary:
    if isinstance(application_run_ids, dict):
        normalized = {str(key): str(value) for key, value in application_run_ids.items()}
    else:
        normalized = {}
    return RunSummary(
        summary.benchmark,
        summary.executor,
        summary.out_dir,
        summary.runtime_root,
        summary.status,
        summary.task_count,
        summary.metrics,
        summary.evaluation_status_counts,
        normalized,
    )


if __name__ == "__main__":
    unittest.main()
