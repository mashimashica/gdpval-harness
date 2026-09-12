# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Mapping, cast
from unittest.mock import patch

import eval_harness.local_judge_runner as local_judge_runner
import eval_harness.local_runner as local_runner
import eval_harness.provenance as provenance
import eval_harness.runner as generic_runner
from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
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
from eval_harness.failures import Failure, FailureImpact, FailureKind, RunAbort
from eval_harness.interventions.base import (
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
from eval_harness.interventions.none import NoneIntervention
from eval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult, Verdict
from eval_harness.layout import task_layout


def detail_text(payload: Mapping[str, object]) -> list[str]:
    value = payload.get("details")
    if not isinstance(value, list):
        raise AssertionError("preflight details must be a list")
    return [str(item) for item in value]


class ReliabilityBenchmark(Benchmark):
    name = "reliability"
    revision = "reliability-revision"

    def __init__(self, tasks: tuple[BenchmarkTask, ...] = ()) -> None:
        self.tasks = tasks or (BenchmarkTask(TaskSpec("task-one", "prompt-one")),)
        self.prepared = False
        self.materialize_calls = 0
        self.materialize_error: BaseException | None = None
        self.execution_tasks: list[TaskSpec] = []

    def is_prepared(self) -> bool:
        return self.prepared

    def prepare(self) -> None:
        self.prepared = True

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        return list(self.tasks[:limit])

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        self.materialize_calls += 1
        if self.materialize_error is not None:
            raise self.materialize_error
        (workspace / "input.txt").write_text(task.execution.prompt, encoding="utf-8")
        return ["input.txt"]

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        del workspace, network_policy
        self.execution_tasks.append(task.execution)
        return task.execution


class ReliabilityEvaluator(Evaluator):
    name = "reliability-evaluator"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def __init__(self) -> None:
        self.preflight_ok = True
        self.preflight_details: tuple[str, ...] = ()
        self.plan_error: str | None = None
        self.evaluation_error: BaseException | None = None
        self.evaluation_result_task_id: str | None = None
        self.calls = 0

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        return EvaluatorPreflightResult(
            self.name,
            self.evaluator_type,
            self.preflight_ok,
            version="reliability-evaluator-v1",
            details=self.preflight_details,
        )

    def validate_plan(self, plan: EvaluationPlan) -> None:
        if self.plan_error is not None:
            raise ValueError(self.plan_error)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls += 1
        if self.evaluation_error is not None:
            raise self.evaluation_error
        return EvaluationResult(
            task_id=self.evaluation_result_task_id or request.task_id,
            status=EvaluationStatus.COMPLETED,
            metrics={"accuracy": 1.0},
            details={"task": request.task_id},
        )


class ReliabilityExecutor(Executor):
    name = "reliability-executor"
    invocation_mode = "reliability"

    def __init__(self) -> None:
        self.preflight_ok = True
        self.preflight_executor = self.name
        self.preflight_details: tuple[str, ...] = ()
        self.result_task_id: str | None = None
        self.result_executor: str | None = None
        self.result_invocation_mode: str | None = None
        self.result_version: str | None = "reliability-executor-v1"
        self.result_auth_mode: str | None = "local"
        self.result_workspace: Path | None = None
        self.result_deliverables: Path | None = None
        self.result_status = ExecutionStatus.COMPLETED
        self.execute_error: BaseException | None = None
        self.calls = 0
        self.requests: list[ExecutionRequest] = []

    def preflight(self) -> PreflightResult:
        return PreflightResult(
            executor=self.preflight_executor,
            ok=self.preflight_ok,
            version="reliability-executor-v1",
            auth_mode="local",
            details=self.preflight_details,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls += 1
        self.requests.append(request)
        if self.execute_error is not None:
            raise self.execute_error
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        (request.executor_dir / "stdout.log").write_text("executor output\n", encoding="utf-8")
        successful = self.result_status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
        output_text = "executor output" if successful else None
        return ExecutionResult(
            task_id=self.result_task_id or request.task.task_id,
            executor=self.result_executor or self.name,
            executor_version=self.result_version,
            invocation_mode=self.result_invocation_mode or self.invocation_mode,
            auth_mode=self.result_auth_mode or "local",
            workspace=self.result_workspace or request.workspace,
            deliverables_dir=self.result_deliverables or request.deliverables_dir,
            status=self.result_status,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=0 if successful else 1,
            available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}) if output_text is not None else frozenset(),
            failure=(
                None
                if successful
                else Failure(
                    FailureKind.INTERRUPTED if self.result_status is ExecutionStatus.INTERRUPTED else FailureKind.PROCESS,
                    "interrupted" if self.result_status is ExecutionStatus.INTERRUPTED else "test_failure",
                    FailureImpact.RUN,
                )
            ),
            output_text=output_text,
        )


class ReliabilityIntervention(Intervention):
    name = "reliability-intervention"
    intervention_type = InterventionType.PROMPT_OVERLAY

    def __init__(self) -> None:
        content = b"reviewed intervention"
        entry = InterventionFile(path="overlay.txt", size=len(content), sha256="a" * 64)
        manifest = InterventionManifest(
            intervention_id="reliability-intervention-id",
            intervention_type=self.intervention_type,
            source_revision="source-revision",
            revision_status="available",
            files=(entry,),
            bundle_sha256=compute_bundle_sha256(((entry.path, content),)),
            application=ApplicationMapping(method="prompt-overlay", target="task.prompt"),
        )
        self.bundle = InterventionBundle(root=None, manifest=manifest)
        self.apply_error: BaseException | None = None
        self.mismatch: str | None = None
        self.apply_calls = 0

    def preflight(self) -> InterventionPreflightResult:
        return InterventionPreflightResult(
            name=self.name,
            intervention_type=self.intervention_type,
            ok=True,
            bundle=self.bundle,
            details=("ready",),
        )

    def validate_task(self, task: TaskSpec) -> None:
        del task

    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        del workspace
        self.apply_calls += 1
        if self.apply_error is not None:
            raise self.apply_error
        manifest = self.bundle.manifest
        returned_run_id = "wrong-run-id" if self.mismatch == "run-id" else application_run_id
        returned_task = TaskSpec("wrong-task", task.prompt) if self.mismatch == "task-id" else task
        returned_manifest_sha256 = "c" * 64 if self.mismatch == "manifest" else manifest.manifest_sha256 or ""
        returned_application = (
            ApplicationMapping(method="files", target="workspace")
            if self.mismatch == "mapping"
            else manifest.application
        )
        return InterventionApplication(
            application_run_id=returned_run_id,
            task=returned_task,
            materialized_files=(InterventionFile(path="overlay.txt", size=3, sha256="b" * 64),),
            bundle_sha256=manifest.bundle_sha256,
            manifest_sha256=returned_manifest_sha256,
            application=returned_application,
        )


class ReliabilityJudge(JudgeExecutor):
    name = "reliability-judge"
    invocation_mode = "reliability-judge"

    def __init__(self) -> None:
        self.result_verdict: Verdict | None = Verdict.A
        self.result_exit_code: int | None = 0
        self.error: BaseException | None = None
        self.calls = 0

    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        del environment
        return JudgePreflightResult(
            judge_executor=self.name,
            ok=True,
            version="judge-v1",
            auth_mode="local",
        )

    def judge(self, request: JudgeRequest) -> JudgeResult:
        self.calls += 1
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        stdout = request.executor_dir / "stdout.log"
        stderr = request.executor_dir / "stderr.log"
        stdout.write_text("judge output\n", encoding="utf-8")
        stderr.write_text("judge diagnostics\n", encoding="utf-8")
        if self.error is not None:
            raise self.error
        return JudgeResult(
            task_id=request.task_id,
            trial_index=request.trial_index,
            judge_executor=self.name,
            verdict=self.result_verdict,
            executor_version="judge-v1",
            invocation_mode=self.invocation_mode,
            auth_mode="local",
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=self.result_exit_code,
            stdout_path=stdout,
            stderr_path=stderr,
            metadata={"deterministic": True},
            reasoning_effort_requested=local_judge_runner._parse_reasoning_effort(),
        )


class RunnerReliabilityTests(unittest.TestCase):
    def test_local_runner_parsers_and_condition_file_fail_closed(self) -> None:
        with patch.dict(os.environ, {"GDPVAL_EXECUTOR_MAX_TURNS": "bad"}):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                local_runner._parse_max_turns()
        with patch.dict(os.environ, {"GDPVAL_EXECUTOR_MAX_TURNS": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                local_runner._parse_max_turns()
        with patch.dict(os.environ, {"LIMIT": "bad"}):
            with self.assertRaisesRegex(ValueError, "--limit must be an integer"):
                local_runner._parse_limit()
        with patch.dict(os.environ, {"LIMIT": "0"}):
            with self.assertRaisesRegex(ValueError, "--limit must be positive"):
                local_runner._parse_limit()
        with patch.dict(os.environ, {"GDPVAL_EXECUTOR_TIMEOUT": "bad"}):
            with self.assertRaisesRegex(ValueError, "must be numeric"):
                local_runner._parse_timeout()
        with patch.dict(os.environ, {"GDPVAL_EXECUTOR_TIMEOUT": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                local_runner._parse_timeout()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.txt"
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(missing)}):
                with self.assertRaisesRegex(ValueError, "not a readable file"):
                    local_runner._condition_file()
            oversized = root / "oversized.txt"
            oversized.write_bytes(b"x" * (local_runner._MAX_CONDITION_FILE_BYTES + 1))
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(oversized)}):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    local_runner._condition_file()
            empty = root / "empty.txt"
            empty.write_text("\n", encoding="utf-8")
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(empty)}):
                with self.assertRaisesRegex(ValueError, "empty"):
                    local_runner._condition_instructions()
            invalid = root / "invalid.txt"
            invalid.write_bytes(b"\xff")
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(invalid)}):
                with self.assertRaisesRegex(ValueError, "UTF-8"):
                    local_runner._condition_instructions()
            valid = root / "valid.txt"
            valid.write_text("  instruction  \n", encoding="utf-8")
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(valid)}):
                self.assertEqual(local_runner._condition_instructions(), "  instruction")
                self.assertEqual(local_runner._sha256_file(valid), local_runner._sha256_file(valid))
            with (
                patch.object(Path, "stat", side_effect=OSError("stat denied")),
                patch.object(Path, "is_file", return_value=True),
            ):
                with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(valid)}):
                    with self.assertRaisesRegex(ValueError, "could not inspect"):
                        local_runner._condition_file()
            with patch.object(Path, "open", side_effect=OSError("hash denied")):
                with self.assertRaisesRegex(ValueError, "could not hash"):
                    local_runner._sha256_file(valid)

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(local_runner._condition_file())
            self.assertIsNone(local_runner._condition_instructions())
            self.assertIsNone(local_runner._sha256_file(None))

    def test_local_runner_executor_selection_and_resume_mismatch_refusal(self) -> None:
        with patch.dict(os.environ, {"GDPVAL_EXECUTOR_NETWORK": "enabled"}, clear=True):
            self.assertEqual(local_runner._executor("codex").name, "codex")
            self.assertEqual(local_runner._executor("claude-code").name, "claude-code")
            self.assertEqual(local_runner._executor("cursor").name, "cursor")
        with self.assertRaisesRegex(ValueError, "not implemented"):
            local_runner._executor("unknown")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition = root / "condition.txt"
            condition.write_text("condition", encoding="utf-8")
            metadata = root / "run-metadata.json"
            with patch.dict(
                os.environ,
                {"RESUME": "1", "GDPVAL_REASONING_EFFORT": "high", "GDPVAL_CONDITION_FILE": str(condition)},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "requires the existing"):
                    local_runner._validate_resume_condition(root)

            metadata.write_text("not-json", encoding="utf-8")
            with patch.dict(os.environ, {"RESUME": "1"}, clear=True):
                with self.assertRaisesRegex(ValueError, "cannot verify"):
                    local_runner._validate_resume_condition(root)
            metadata.write_text(json.dumps({"configuration": []}), encoding="utf-8")
            with patch.dict(os.environ, {"RESUME": "1"}, clear=True):
                with self.assertRaisesRegex(ValueError, "no configuration"):
                    local_runner._validate_resume_condition(root)
            metadata.write_text(json.dumps({"configuration": {"condition": "old"}}), encoding="utf-8")
            with patch.dict(os.environ, {"RESUME": "1", "GDPVAL_CONDITION": "new"}, clear=True):
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    local_runner._validate_resume_condition(root)

            metadata.unlink()
            with patch.dict(os.environ, {"RESUME": "1", "GDPVAL_REASONING_EFFORT": "high"}, clear=True):
                with self.assertRaisesRegex(ValueError, "reasoning-effort"):
                    local_runner._validate_resume_condition(root)
            with patch.dict(os.environ, {"RESUME": "0", "GDPVAL_REASONING_EFFORT": "high"}, clear=True):
                local_runner._validate_resume_condition(root)

    def test_local_runner_copy_and_completion_records_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            (source / "nested").mkdir(parents=True)
            (source / "nested" / "file.txt").write_text("content", encoding="utf-8")
            self.assertEqual(local_runner._copy_tree_files(source, target), ["nested/file.txt"])
            self.assertEqual(local_runner._copy_tree_files(root / "absent", root / "empty"), [])
            symlink = source / "symlink"
            symlink.symlink_to(source / "nested" / "file.txt")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                local_runner._copy_tree_files(source, root / "symlink-target")

            task = TaskSpec("task-copy", "prompt")
            layout = task_layout(root / "out", task.task_id)
            layout.workspace.mkdir(parents=True)
            layout.executor_dir.mkdir(parents=True)
            layout.workspace_deliverables.mkdir(parents=True)
            (layout.workspace_deliverables / "answer.txt").write_text("answer", encoding="utf-8")
            (layout.workspace / "reference_files").mkdir()
            (layout.workspace / "reference_files" / "ref.txt").write_text("reference", encoding="utf-8")
            result = ExecutionResult(
                task_id=task.task_id,
                executor="codex",
                executor_version="1",
                invocation_mode="exec",
                auth_mode="subscription",
                workspace=layout.workspace,
                deliverables_dir=layout.workspace_deliverables,
                status=ExecutionStatus.COMPLETED,
                started_at="start",
                finished_at="finish",
                exit_code=0,
                available_outputs=frozenset(),
                failure=None,
                reasoning_effort_requested="high",
            )
            copied = local_runner._copy_final_deliverables(layout, result)
            self.assertEqual(copied, ["answer.txt"])
            finish = json.loads((layout.judge_deliverables / "finish_params.json").read_text(encoding="utf-8"))
            self.assertEqual(finish["status"], "completed")
            self.assertTrue((layout.judge_deliverables / "reference_files" / "ref.txt").is_file())

            local_runner._write_executor_metadata(layout, result, copied)
            metadata = json.loads((layout.executor_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reasoning_effort_requested"], "high")
            self.assertTrue(local_runner._already_completed(layout))
            (layout.executor_dir / "metadata.json").write_text("invalid", encoding="utf-8")
            self.assertFalse(local_runner._already_completed(layout))
            (layout.executor_dir / "metadata.json").write_text(
                json.dumps({"execution_status": ExecutionStatus.FAILED.value}), encoding="utf-8"
            )
            self.assertFalse(local_runner._already_completed(layout))
            (layout.executor_dir / "metadata.json").unlink()
            self.assertFalse(local_runner._already_completed(layout))

    def test_local_runner_preflight_collects_reliability_failures(self) -> None:
        benchmark = ReliabilityBenchmark()
        executor = ReliabilityExecutor()
        with (
            patch.object(local_runner, "_benchmark", return_value=benchmark),
            patch.object(local_runner, "_executor", return_value=executor),
            patch.object(local_runner, "_build_intervention", return_value=NoneIntervention()),
            patch.object(local_runner, "PREPARE_SCRIPT", Path("/definitely/missing/prepare.py")),
            patch.dict(
                os.environ,
                {
                    "PARALLEL": "4",
                    "GDPVAL_EXECUTOR_TIMEOUT": "1",
                    "LIMIT": "bad",
                    "GDPVAL_CONDITION_FILE": "/missing/condition.txt",
                },
                clear=True,
            ),
        ):
            ok, payload = local_runner.preflight("codex", Path("/tmp/reliability-out"), for_run=True)
        self.assertFalse(ok)
        details = detail_text(payload)
        self.assertTrue(any("prepare script is missing" in detail for detail in details))
        self.assertTrue(any("parallel 1" in detail for detail in details))
        self.assertTrue(any("not a readable file" in detail for detail in details))
        self.assertTrue(any("--limit must be an integer" in detail for detail in details))

        class ExplodingIntervention(Intervention):
            name = "exploding"
            intervention_type = InterventionType.NONE

            def preflight(self) -> InterventionPreflightResult:
                raise RuntimeError("preflight secret")

            def validate_task(self, task: TaskSpec) -> None:
                del task

            def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
                del task, workspace, application_run_id
                raise AssertionError("unreachable")

        with patch.object(local_runner, "_executor", return_value=executor), patch.dict(os.environ, {}, clear=True):
            ok, payload = local_runner.preflight(
                "codex", Path(tempfile.gettempdir()), for_run=False, intervention=ExplodingIntervention()
            )
        self.assertFalse(ok)
        self.assertIn("preflight failed", detail_text(payload)[0])

    def test_local_runner_metadata_wrapper_and_main_fail_closed(self) -> None:
        executor = ReliabilityExecutor()
        payload: dict[str, object] = {"executor": executor.name, "version": "v", "auth_mode": "local"}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with (
                patch.object(local_runner, "_executor", return_value=executor),
                patch(
                    "eval_harness.local_runner.subprocess.run", return_value=subprocess.CompletedProcess([], 0)
                ) as run,
            ):
                local_runner._write_run_metadata(out, payload)
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["cwd"], local_runner.ROOT)
            self.assertEqual(kwargs["env"]["OUT"], str(out))
            self.assertEqual(kwargs["env"]["GDPVAL_EXECUTOR_INVOCATION_MODE"], executor.invocation_mode)

        with (
            patch.object(local_runner, "preflight", return_value=(False, {"details": ["denied"]})),
            patch.dict(os.environ, {"GDPVAL_EXECUTOR": "codex", "OUT": "/tmp/reliability-run"}, clear=True),
        ):
            self.assertEqual(local_runner.run(), 2)
        with patch("eval_harness.local_runner.sys.argv", ["local_runner", "unknown"]):
            with self.assertRaisesRegex(SystemExit, "unknown local-runner mode"):
                local_runner.main()

    def test_local_runner_run_persists_resume_skip_and_each_failure_phase(self) -> None:
        task = BenchmarkTask(TaskSpec("task-one", "prompt-one"))
        cases: tuple[tuple[str, BaseException], ...] = (
            ("materialize", RuntimeError("materialize secret")),
            ("intervention", RuntimeError("intervention secret")),
            ("execute", RuntimeError("execute secret")),
        )
        for phase, error in cases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                out = root / "out"
                benchmark = ReliabilityBenchmark((task,))
                if phase == "materialize":
                    benchmark.materialize_error = error
                executor = ReliabilityExecutor()
                intervention = ReliabilityIntervention()
                if phase == "intervention":
                    intervention.apply_error = error
                if phase == "execute":
                    executor.execute_error = error
                with (
                    patch.object(local_runner, "preflight", return_value=(True, {"details": [], "executor": "codex"})),
                    patch.object(local_runner, "_benchmark", return_value=benchmark),
                    patch.object(local_runner, "_executor", return_value=executor),
                    patch.object(local_runner, "_load_tasks", return_value=[task]),
                    patch.object(local_runner, "_ensure_dataset"),
                    patch.object(local_runner, "_parse_limit", return_value=1),
                    patch.object(local_runner, "_parse_timeout", return_value=1.0),
                    patch.object(local_runner, "_build_intervention", return_value=intervention),
                    patch.dict(
                        os.environ,
                        {
                            "OUT": str(out),
                            "GDPVAL_EXECUTOR": "codex",
                            "GDPVAL_WRITE_METADATA": "0",
                        },
                        clear=True,
                    ),
                ):
                    self.assertEqual(local_runner.run(), 1)
                metadata = json.loads((out / "tasks" / "task-one" / "executor" / "metadata.json").read_text())
                self.assertEqual(metadata["execution_status"], "failed")
                expected_error = (
                    "intervention application failed before executor" if phase == "intervention" else str(error)
                )
                self.assertEqual(metadata["harness_error"], expected_error)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "resume"
            layout = task_layout(out, "task-one")
            layout.executor_dir.mkdir(parents=True)
            (layout.executor_dir / "metadata.json").write_text(
                json.dumps({"execution_status": ExecutionStatus.COMPLETED.value}), encoding="utf-8"
            )
            benchmark = ReliabilityBenchmark((task,))
            executor = ReliabilityExecutor()
            with (
                patch.object(local_runner, "preflight", return_value=(True, {"details": [], "executor": "codex"})),
                patch.object(local_runner, "_benchmark", return_value=benchmark),
                patch.object(local_runner, "_executor", return_value=executor),
                patch.object(local_runner, "_load_tasks", return_value=[task]),
                patch.object(local_runner, "_ensure_dataset"),
                patch.object(local_runner, "_parse_limit", return_value=1),
                patch.object(local_runner, "_parse_timeout", return_value=1.0),
                patch.object(local_runner, "_build_intervention", return_value=ReliabilityIntervention()),
                patch.dict(
                    os.environ,
                    {"OUT": str(out), "GDPVAL_EXECUTOR": "codex", "RESUME": "1", "GDPVAL_WRITE_METADATA": "0"},
                    clear=True,
                ),
            ):
                self.assertEqual(local_runner.run(), 0)
            self.assertEqual(executor.calls, 0)

    def test_local_runner_interrupts_are_durable_before_return(self) -> None:
        task = BenchmarkTask(TaskSpec("task-interrupt", "prompt"))
        for phase in ("materialize", "intervention", "execute"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                out = root / "out"
                benchmark = ReliabilityBenchmark((task,))
                executor = ReliabilityExecutor()
                intervention = ReliabilityIntervention()
                if phase == "materialize":
                    benchmark.materialize_error = KeyboardInterrupt()
                elif phase == "intervention":
                    intervention.apply_error = KeyboardInterrupt()
                else:
                    executor.execute_error = KeyboardInterrupt()
                with (
                    patch.object(local_runner, "preflight", return_value=(True, {"details": [], "executor": "codex"})),
                    patch.object(local_runner, "_benchmark", return_value=benchmark),
                    patch.object(local_runner, "_executor", return_value=executor),
                    patch.object(local_runner, "_load_tasks", return_value=[task]),
                    patch.object(local_runner, "_ensure_dataset"),
                    patch.object(local_runner, "_parse_limit", return_value=1),
                    patch.object(local_runner, "_parse_timeout", return_value=1.0),
                    patch.object(local_runner, "_build_intervention", return_value=intervention),
                    patch.dict(
                        os.environ,
                        {"OUT": str(out), "GDPVAL_EXECUTOR": "codex", "GDPVAL_WRITE_METADATA": "0"},
                        clear=True,
                    ),
                ):
                    self.assertEqual(local_runner.run(), 130)
                persisted = json.loads((out / "tasks" / "task-interrupt" / "executor" / "metadata.json").read_text())
                self.assertEqual(persisted["execution_status"], "interrupted")


class LocalJudgeReliabilityTests(unittest.TestCase):
    def _candidate(self, root: Path, name: str, *, prompt: str = "prompt-one") -> Path:
        candidate = root / name
        (candidate / "task_one" / "repeat_0").mkdir(parents=True)
        (candidate / "task_one" / "repeat_0" / "answer.txt").write_text(name, encoding="utf-8")
        executor = candidate / "tasks" / "one" / "executor"
        executor.mkdir(parents=True)
        (executor / "task-prompt.txt").write_text(prompt, encoding="utf-8")
        return candidate

    def test_local_judge_parsers_provenance_and_selection_errors(self) -> None:
        with patch.dict(os.environ, {"GDPVAL_JUDGE_REASONING_EFFORT": "high"}):
            self.assertEqual(local_judge_runner._parse_reasoning_effort(), "high")
        with self.assertRaisesRegex(ValueError, "must be codex"):
            local_judge_runner._judge_executor("unsupported")
        with patch.dict(os.environ, {"VALUE": "bad"}):
            with self.assertRaisesRegex(ValueError, "VALUE must be an integer"):
                local_judge_runner._positive_int("VALUE")
        with patch.dict(os.environ, {"VALUE": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                local_judge_runner._positive_int("VALUE")
        with patch.dict(os.environ, {"VALUE": "bad"}):
            with self.assertRaisesRegex(ValueError, "VALUE must be numeric"):
                local_judge_runner._positive_float("VALUE", 1.0)
        with patch.dict(os.environ, {"VALUE": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                local_judge_runner._positive_float("VALUE", 1.0)
        with patch.dict(os.environ, {"VALUE": "bad"}):
            with self.assertRaisesRegex(ValueError, "VALUE must be an integer"):
                local_judge_runner._integer("VALUE", 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = self._candidate(root, "a")
            b = self._candidate(root, "b")
            prompts = {"task_one": "prompt-one"}
            pair = [("task_one", a / "task_one" / "repeat_0", b / "task_one" / "repeat_0")]
            self.assertEqual(
                local_judge_runner._validate_selected_pairs(a, b, pair, prompts),
                {"task_one": hashlib.sha256(b"prompt-one").hexdigest()},
            )

            with self.assertRaisesRegex(ValueError, "benchmark prompt not found"):
                local_judge_runner._validate_selected_pairs(a, b, pair, {})
            wrong_a = self._candidate(root, "wrong-a", prompt="wrong")
            wrong_pair = [("task_one", wrong_a / "task_one" / "repeat_0", b / "task_one" / "repeat_0")]
            with self.assertRaisesRegex(ValueError, "candidate A prompt"):
                local_judge_runner._validate_selected_pairs(wrong_a, b, wrong_pair, prompts)
            wrong_b = self._candidate(root, "wrong-b", prompt="wrong")
            wrong_pair = [("task_one", a / "task_one" / "repeat_0", wrong_b / "task_one" / "repeat_0")]
            with self.assertRaisesRegex(ValueError, "candidate B prompt"):
                local_judge_runner._validate_selected_pairs(a, wrong_b, wrong_pair, prompts)

            canonical = a / "tasks" / "one" / "executor" / "task-prompt.txt"
            canonical.unlink()
            canonical.symlink_to(root / "prompt-target.txt")
            (root / "prompt-target.txt").write_text("prompt-one", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "symlinked"):
                local_judge_runner._candidate_task_prompt(a / "deliverables", "task_one")
            canonical.unlink()
            canonical.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                local_judge_runner._candidate_task_prompt(a / "deliverables", "task_one")

            fallback = b / "tasks" / "one" / "executor" / "prompt.txt"
            (b / "tasks" / "one" / "executor" / "task-prompt.txt").unlink()
            fallback.write_bytes(b"wrapper\nTask:\nfirst\xff\n")
            self.assertEqual(
                local_judge_runner._candidate_task_prompt(b / "deliverables", "task_one"),
                "first\ufffd",
            )
            fallback.write_text("wrapper only", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not contain"):
                local_judge_runner._candidate_task_prompt(b / "deliverables", "task_one")
            fallback.unlink()
            with self.assertRaisesRegex(ValueError, "missing the recorded"):
                local_judge_runner._candidate_task_prompt(b / "deliverables", "task_one")

    def test_local_judge_dataset_metadata_and_temp_root_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "benchmark.jsonl"
            prepare = root / "prepare.py"
            with (
                patch.object(local_judge_runner, "BENCHMARK_JSONL", dataset),
                patch.object(local_judge_runner, "PREPARE_SCRIPT", prepare),
            ):
                with self.assertRaisesRegex(RuntimeError, "prepare script is missing"):
                    local_judge_runner._ensure_dataset()
                prepare.write_text("prepare", encoding="utf-8")

                def prepare_dataset(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
                    del args, kwargs
                    dataset.write_text('{"task_id":"one","prompt":"prompt-one"}\n', encoding="utf-8")
                    return subprocess.CompletedProcess([], 0)

                with patch("eval_harness.local_judge_runner.subprocess.run", side_effect=prepare_dataset):
                    local_judge_runner._ensure_dataset()
                dataset.unlink()
                with patch(
                    "eval_harness.local_judge_runner.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 2),
                ):
                    with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                        local_judge_runner._ensure_dataset()

                dataset.write_text('\n{"task_id":"one","prompt":"prompt-one"}\n', encoding="utf-8")
                self.assertEqual(local_judge_runner._task_prompts(), {"task_one": "prompt-one"})

            deliverables = root / "deliverables"
            deliverables.mkdir()
            self.assertIsNone(local_judge_runner._read_run_metadata(deliverables))
            metadata = root / "run-metadata.json"
            metadata.write_text("invalid", encoding="utf-8")
            self.assertIsNone(local_judge_runner._read_run_metadata(deliverables))
            metadata.write_text(json.dumps({"configuration": "invalid", "repository": "invalid"}), encoding="utf-8")
            self.assertEqual(
                local_judge_runner._generator_info(deliverables),
                {"executor": None, "model": None, "repository_commit": None},
            )
            metadata.write_text(
                json.dumps(
                    {"configuration": {"executor": "codex", "model": "model"}, "repository": {"commit": "abc"}}
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                local_judge_runner._generator_info(deliverables),
                {"executor": "codex", "model": "model", "repository_commit": "abc"},
            )

            self.assertTrue(local_judge_runner._paths_overlap(root, root / "nested"))
            self.assertFalse(local_judge_runner._paths_overlap(root / "a", root / "b"))
            with patch("eval_harness.local_judge_runner.tempfile.mkdtemp", side_effect=OSError("no temp")):
                with self.assertRaisesRegex(RuntimeError, "no writable system temporary"):
                    local_judge_runner._safe_temp_parent(root / "a", root / "b", root / "out")

    def test_local_judge_preflight_reports_selection_resume_and_output_failures(self) -> None:
        judge = ReliabilityJudge()
        with patch.dict(os.environ, {"GDPVAL_JUDGE_EXECUTOR": "unsupported"}, clear=True):
            ok, payload = local_judge_runner._preflight(for_run=False)
        self.assertFalse(ok)
        self.assertIn("codex or claude-code", detail_text(payload)[0])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "out"
            with (
                patch.object(local_judge_runner, "_judge_executor", return_value=judge),
                patch.object(local_judge_runner, "BENCHMARK_JSONL", root / "missing.jsonl"),
                patch.object(local_judge_runner, "PREPARE_SCRIPT", root / "missing-prepare.py"),
                patch.dict(
                    os.environ,
                    {
                        "GDPVAL_JUDGE_EXECUTOR": "reliability-judge",
                        "GDPVAL_RUN_A": str(root / "missing-a"),
                        "GDPVAL_RUN_B": str(root / "missing-b"),
                        "OUT": str(out),
                        "LIMIT": "bad",
                        "RESUME": "1",
                    },
                    clear=True,
                ),
            ):
                ok, payload = local_judge_runner._preflight(for_run=True)
            self.assertFalse(ok)
            details = detail_text(payload)
            self.assertTrue(any("must be an integer" in item for item in details))
            self.assertTrue(any("resume is not supported" in item for item in details))
            self.assertTrue(any("candidate A" in item for item in details))
            self.assertTrue(any("both unavailable" in item for item in details))

            a = self._candidate(root, "a")
            b = self._candidate(root, "b")
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "local-judge-summary.json").write_text("old", encoding="utf-8")
            with (
                patch.object(local_judge_runner, "_judge_executor", return_value=judge),
                patch.object(local_judge_runner, "BENCHMARK_JSONL", root / "benchmark.jsonl"),
                patch.object(local_judge_runner, "_safe_temp_parent", side_effect=RuntimeError("overlap")),
                patch.dict(
                    os.environ,
                    {
                        "GDPVAL_JUDGE_EXECUTOR": "reliability-judge",
                        "GDPVAL_RUN_A": str(a),
                        "GDPVAL_RUN_B": str(b),
                        "OUT": str(occupied),
                        "LIMIT": "1",
                    },
                    clear=True,
                ),
            ):
                ok, payload = local_judge_runner._preflight(for_run=True)
            self.assertFalse(ok)
            details = detail_text(payload)
            self.assertTrue(any("prior run data" in item for item in details))
            self.assertTrue(any("isolated judge temp root" in item for item in details))

            with patch("eval_harness.local_judge_runner.Path.write_text", side_effect=OSError("read-only")):
                with (
                    patch.dict(
                        os.environ,
                        {
                            "GDPVAL_JUDGE_EXECUTOR": "reliability-judge",
                            "GDPVAL_RUN_A": str(a),
                            "GDPVAL_RUN_B": str(b),
                            "OUT": str(root / "unwritable"),
                        },
                        clear=True,
                    ),
                    patch.object(local_judge_runner, "_judge_executor", return_value=judge),
                ):
                    ok, payload = local_judge_runner._preflight(for_run=False)
            self.assertFalse(ok)
            self.assertTrue(any("not writable" in item for item in detail_text(payload)))

    def test_local_judge_persists_logs_rows_and_no_fallback_termination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-executor"
            source.mkdir()
            (source / "stdout.log").write_text("stdout", encoding="utf-8")
            target = root / "out" / "judge" / "tasks" / "task_one" / "trial_0" / "executor"
            target.mkdir(parents=True)
            (target / "stale").write_text("stale", encoding="utf-8")
            local_judge_runner._persist_executor_logs(root / "out", "task_one", 0, source, {"row": True})
            self.assertFalse((target / "stale").exists())
            self.assertEqual(json.loads((target / "metadata.json").read_text())["row"], True)

            with patch.dict(os.environ, {"GDPVAL_JUDGE_REASONING_EFFORT": "high"}):
                row = local_judge_runner._interrupted_row(
                    "task_one",
                    0,
                    True,
                    {
                        "judge_executor": "codex",
                        "ok": True,
                        "version": "v",
                        "auth_mode": "local",
                        "details": [],
                        "temp_parent": None,
                    },
                )
            self.assertEqual(row["reasoning_effort_requested"], "high")

            candidates_a = self._candidate(root, "candidate-a")
            candidates_b = self._candidate(root, "candidate-b")
            out = root / "judge-out"
            out.mkdir()
            judge = ReliabilityJudge()
            judge.result_verdict = None
            with (
                patch.object(
                    local_judge_runner,
                    "_preflight",
                    return_value=(
                        True,
                        {
                            "judge_executor": "reliability-judge",
                            "ok": True,
                            "version": "judge-v1",
                            "auth_mode": "local",
                            "temp_parent": str(root),
                            "details": [],
                        },
                    ),
                ),
                patch.object(local_judge_runner, "_ensure_dataset"),
                patch.object(local_judge_runner, "_task_prompts", return_value={"task_one": "prompt-one"}),
                patch.object(local_judge_runner, "_judge_executor", return_value=judge),
                patch.dict(
                    os.environ,
                    {
                        "GDPVAL_RUN_A": str(candidates_a),
                        "GDPVAL_RUN_B": str(candidates_b),
                        "GDPVAL_JUDGE_EXECUTOR": "reliability-judge",
                        "GDPVAL_JUDGE_REASONING_EFFORT": "high",
                        "LIMIT": "1",
                        "GDPVAL_JUDGE_TRIALS": "1",
                        "OUT": str(out),
                        "GDPVAL_WRITE_METADATA": "0",
                    },
                    clear=True,
                ),
            ):
                self.assertEqual(local_judge_runner.run(), 1)
            row = json.loads((out / "local-judge-results.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["reasoning_effort_requested"], "high")
            summary = json.loads((out / "local-judge-summary.json").read_text())
            self.assertEqual(summary["invalid_trials"], 1)
            self.assertEqual(summary["reasoning_effort_requested"], "high")

            with (
                patch.object(
                    local_judge_runner,
                    "_preflight",
                    return_value=(True, {"judge_executor": "reliability-judge", "details": [], "temp_parent": None}),
                ),
                patch.dict(
                    os.environ,
                    {
                        "GDPVAL_RUN_A": str(candidates_a),
                        "GDPVAL_RUN_B": str(candidates_b),
                        "LIMIT": "1",
                        "OUT": str(root / "no-temp"),
                    },
                    clear=True,
                ),
            ):
                with (
                    patch.object(local_judge_runner, "_ensure_dataset"),
                    patch.object(local_judge_runner, "_task_prompts", return_value={"task_one": "prompt-one"}),
                    patch.object(local_judge_runner, "_judge_executor", return_value=judge),
                ):
                    self.assertEqual(local_judge_runner.run(), 2)

            with patch("eval_harness.local_judge_runner.sys.argv", ["local_judge_runner", "unknown"]):
                with self.assertRaisesRegex(SystemExit, "unknown local-judge mode"):
                    local_judge_runner.main()


class GenericRunnerReliabilityTests(unittest.TestCase):
    def _result(self, root: Path, task_id: str = "task-one") -> ExecutionResult:
        workspace = root / "workspace"
        deliverables = workspace / "deliverables"
        workspace.mkdir(parents=True, exist_ok=True)
        deliverables.mkdir(parents=True, exist_ok=True)
        return ExecutionResult(
            task_id=task_id,
            executor="reliability-executor",
            executor_version="reliability-executor-v1",
            invocation_mode="reliability",
            auth_mode="local",
            workspace=workspace,
            deliverables_dir=deliverables,
            status=ExecutionStatus.COMPLETED,
            started_at="start",
            finished_at="finish",
            exit_code=0,
            available_outputs=frozenset(),
            failure=None,
        )

    def test_runner_atomic_records_and_path_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "record.json"
            generic_runner._write_json(path, {"status": "ok"})
            self.assertEqual(json.loads(path.read_text()), {"status": "ok"})
            with patch("eval_harness.runner.os.replace", side_effect=OSError("replace denied")):
                with self.assertRaisesRegex(OSError, "replace denied"):
                    generic_runner._write_json(root / "replace.json", {"status": "failed"})
            self.assertEqual(list(root.glob(".replace.json.tmp-*")), [])
            with patch("eval_harness.runner.os.fsync", side_effect=OSError("fsync denied")):
                with self.assertRaisesRegex(OSError, "fsync denied"):
                    generic_runner._write_json(root / "fsync.json", {"status": "failed"})
            self.assertEqual(list(root.glob(".fsync.json.tmp-*")), [])
            with patch("eval_harness.runner.os.open", side_effect=OSError("directory unsupported")):
                generic_runner._fsync_directory(root)

            with patch.dict(
                os.environ,
                {"GDPVAL_CONDITION": "secret", "GDPVAL_CONDITION_FILE": "secret-file", "KEEP": "yes"},
                clear=True,
            ):
                environment = generic_runner._executor_environment()
            self.assertEqual(environment, {"KEEP": "yes"})
            self.assertNotIn("secret", json.dumps(environment))

            metrics = generic_runner._aggregate_metrics(
                [
                    {
                        "evaluation": {
                            "status": EvaluationStatus.COMPLETED.value,
                            "metrics": {"score": 1, "ignored": True},
                        }
                    },
                    {
                        "evaluation": {
                            "status": EvaluationStatus.COMPLETED.value,
                            "metrics": {"score": 3, "ignored": 2},
                        }
                    },
                    {"evaluation": {"status": "failed", "metrics": {"score": 99}}},
                    {"evaluation": None},
                ]
            )
            self.assertEqual(metrics, {"ignored": 2.0, "score": 2.0})
            self.assertEqual(
                generic_runner._evaluation_status_counts(
                    [{"evaluation": {"status": "failed"}}, {"evaluation": None}, {"evaluation": {}}]
                ),
                {"failed": 1, "unknown": 1},
            )
            self.assertEqual(str(generic_runner._preflight_failure("x", ())), "x preflight failed: preflight failed")
            self.assertEqual(str(generic_runner._preflight_failure("x", "details")), "x preflight failed: details")

            self.assertTrue(generic_runner._path_matches(root, root))
            symlink_target = root / "target"
            symlink_target.mkdir()
            link = root / "link"
            link.symlink_to(symlink_target, target_is_directory=True)
            self.assertFalse(generic_runner._path_matches(link, symlink_target))
            # Deliberately pass an invalid path object to verify the boundary guard.
            self.assertFalse(generic_runner._path_matches(cast(Path, object()), root))
            self.assertEqual(
                generic_runner._canonical_planned_root(root / "fresh", existing_message="occupied"),
                (root / "fresh").resolve(),
            )
            with self.assertRaises(FileExistsError):
                generic_runner._canonical_planned_root(root, existing_message="occupied")
            existing_target = root / "existing-target"
            existing_target.mkdir()
            existing_link = root / "existing-link"
            existing_link.symlink_to(existing_target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                generic_runner._canonical_planned_root(existing_link, existing_message="occupied")
            with patch.object(Path, "resolve", side_effect=OSError("resolve denied")):
                with self.assertRaisesRegex(ValueError, "could not resolve"):
                    generic_runner._canonical_planned_root(root / "resolve-failure", existing_message="occupied")
            with self.assertRaisesRegex(ValueError, "separate"):
                generic_runner._ensure_roots_are_separate(root, root / "child")

            result = self._result(root)
            self.assertEqual(generic_runner._execution_payload(result)["status"], "completed")
            self.assertEqual(
                generic_runner._evaluation_interrupt_payload(KeyboardInterrupt())["status"], "interrupted"
            )
            error = generic_runner._evaluation_error_payload(RuntimeError("credential"), phase="test")
            self.assertNotIn("credential", json.dumps(error))

    def test_runner_preflight_and_arguments_stop_before_external_work(self) -> None:
        benchmark = ReliabilityBenchmark()
        evaluator = ReliabilityEvaluator()
        executor = ReliabilityExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "--limit"):
                generic_runner.run_benchmark(benchmark, evaluator, executor, out_dir=root / "out", limit=0)
            with self.assertRaisesRegex(ValueError, "executor-timeout"):
                generic_runner.run_benchmark(
                    benchmark,
                    evaluator,
                    executor,
                    out_dir=root / "out-timeout",
                    limit=1,
                    timeout_seconds=0,
                )

            executor.preflight_ok = False
            executor.preflight_details = ("executor unavailable",)
            with self.assertRaisesRegex(RuntimeError, "executor unavailable"):
                generic_runner.run_benchmark(benchmark, evaluator, executor, out_dir=root / "preflight", limit=1)
            self.assertFalse((root / "preflight").exists())

    def test_runner_intervention_mismatch_and_interrupt_records_are_durable(self) -> None:
        for mismatch, message in (
            ("run-id", "application_run_id"),
            ("task-id", "task_id"),
            ("manifest", "manifest_sha256"),
            ("mapping", "application mapping"),
        ):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                intervention = ReliabilityIntervention()
                intervention.mismatch = mismatch
                with self.assertRaisesRegex(ValueError, message):
                    generic_runner.run_benchmark(
                        ReliabilityBenchmark(),
                        ReliabilityEvaluator(),
                        ReliabilityExecutor(),
                        out_dir=root / "out",
                        limit=1,
                        intervention=intervention,
                    )
                row = json.loads((root / "out" / "results.jsonl").read_text().splitlines()[0])
                self.assertEqual(row["intervention"]["status"], "failed")
                self.assertIsNone(row["execution"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intervention = ReliabilityIntervention()
            intervention.apply_error = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                generic_runner.run_benchmark(
                    ReliabilityBenchmark(),
                    ReliabilityEvaluator(),
                    ReliabilityExecutor(),
                    out_dir=root / "out",
                    limit=1,
                    intervention=intervention,
                )
            row = json.loads((root / "out" / "results.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["intervention"]["status"], "interrupted")
            metadata = json.loads((root / "out" / "run-metadata.json").read_text())
            self.assertEqual(metadata["status"], "interrupted")

    def test_runner_execution_and_evaluation_identity_failures_are_persisted(self) -> None:
        for mode, message in (("task", "mismatched task_id"), ("workspace", "outside the assigned task workspace")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                executor = ReliabilityExecutor()
                if mode == "task":
                    executor.result_task_id = "wrong-task"
                else:
                    executor.result_workspace = root / "outside"
                with self.assertRaisesRegex(ValueError, message):
                    generic_runner.run_benchmark(
                        ReliabilityBenchmark(),
                        ReliabilityEvaluator(),
                        executor,
                        out_dir=root / "out",
                        limit=1,
                    )
                row = json.loads((root / "out" / "results.jsonl").read_text().splitlines()[0])
                self.assertEqual(row["evaluation"]["status"], "failed")
                self.assertEqual(executor.calls, 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = ReliabilityEvaluator()
            evaluator.evaluation_result_task_id = "wrong-evaluation-task"
            with self.assertRaisesRegex(ValueError, "evaluator returned task id"):
                generic_runner.run_benchmark(
                    ReliabilityBenchmark(), evaluator, ReliabilityExecutor(), out_dir=root / "out", limit=1
                )
            row = json.loads((root / "out" / "results.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["evaluation"]["status"], "failed")

            evaluator = ReliabilityEvaluator()
            evaluator.evaluation_error = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                generic_runner.run_benchmark(
                    ReliabilityBenchmark(), evaluator, ReliabilityExecutor(), out_dir=root / "interrupt", limit=1
                )
            metadata = json.loads((root / "interrupt" / "run-metadata.json").read_text())
            self.assertEqual(metadata["status"], "interrupted")
            row = json.loads((root / "interrupt" / "results.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["evaluation"]["status"], "interrupted")

    def test_runner_stops_before_evaluation_for_systemic_executor_failures(self) -> None:
        for status in (ExecutionStatus.FAILED, ExecutionStatus.TIMED_OUT, ExecutionStatus.INTERRUPTED):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                evaluator = ReliabilityEvaluator()
                executor = ReliabilityExecutor()
                executor.result_status = status
                with self.assertRaises(RunAbort) as raised:
                    generic_runner.run_benchmark(
                        ReliabilityBenchmark(),
                        evaluator,
                        executor,
                        out_dir=root / "out",
                        limit=1,
                    )
                self.assertEqual(str(raised.exception), "interrupted" if status is ExecutionStatus.INTERRUPTED else "test_failure")
                self.assertEqual(evaluator.calls, 0)
                row = json.loads((root / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(row["evaluation"]["status"], "skipped")
                self.assertEqual(row["evaluation"]["metrics"], {})
                self.assertEqual(row["evaluation"]["outcomes"], {})
                self.assertEqual(row["evaluation"]["details"], {"reason": "executor failure prevented evaluation"})
                self.assertEqual(row["execution"]["available_outputs"], [])
                expected_kind = "interrupted" if status is ExecutionStatus.INTERRUPTED else "process"
                expected_code = "interrupted" if status is ExecutionStatus.INTERRUPTED else "test_failure"
                self.assertEqual(
                    row["execution"]["failure"],
                    {"kind": expected_kind, "code": expected_code, "impact": "run"},
                )
                metadata = json.loads((root / "out" / "run-metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "interrupted" if status is ExecutionStatus.INTERRUPTED else "failed")


class ProvenanceReliabilityTests(unittest.TestCase):
    def test_provenance_rejects_malformed_command_results_and_non_utf8(self) -> None:
        sha = "a" * 40
        head = subprocess.CompletedProcess([], 0, stdout=sha.encode() + b"\n", stderr=b"")
        status = subprocess.CompletedProcess([], 0, stdout=b" M file\xff.txt\n", stderr=b"")
        with patch("eval_harness.provenance.subprocess.run", side_effect=[head, status]):
            observed = provenance.repository_provenance(Path("/tmp/provenance"))
        self.assertEqual(observed.commit, sha)
        self.assertEqual(observed.worktree_status, "dirty")

        malformed_statuses = ("bad status", " M", " Mxfile", " X file", "??")
        for value in malformed_statuses:
            with self.subTest(value=value):
                with patch(
                    "eval_harness.provenance.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, stdout=f"{sha}\n", stderr=""),
                        subprocess.CompletedProcess([], 0, stdout=value, stderr=""),
                    ],
                ):
                    observed = provenance.repository_provenance(Path("/tmp/provenance"))
                self.assertEqual(observed.worktree_status, "unavailable")

        with patch("eval_harness.provenance.subprocess.run", side_effect=RuntimeError("git unavailable")):
            self.assertEqual(
                provenance.repository_provenance(Path("/tmp/provenance")).revision_status,
                "unavailable",
            )
        result_with_bad_attributes: object = object()
        with patch("eval_harness.provenance.subprocess.run", return_value=result_with_bad_attributes):
            self.assertEqual(
                provenance.repository_provenance(Path("/tmp/provenance")).revision_status,
                "unavailable",
            )

        with self.assertRaises(ValueError):
            provenance.RepositoryProvenance(None, "available", "clean")
        self.assertEqual(provenance.repository_provenance(cast(Path | str, object())).revision_status, "unavailable")
        self.assertEqual(provenance._parse_worktree_status(""), "clean")
        self.assertIsNone(provenance._parse_worktree_status("\n"))


if __name__ == "__main__":
    unittest.main()
