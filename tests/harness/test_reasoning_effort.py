# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict, cast
from unittest.mock import patch

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkTask
from gdpval_harness.builders.base import BuilderInputBundle
from gdpval_harness.builders.executor_skill import ExecutorSkillBuilder
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
from gdpval_harness.executors.codex import CodexExecutor
from gdpval_harness.executors.registry import create_executor
from gdpval_harness.experiments.base import (
    ExperimentArm,
    ExperimentInputSpec,
    ExperimentProfile,
    ExperimentRunConfig,
    LoadedExperimentProfile,
)
from gdpval_harness.experiments.runner import run_builder_experiment
from gdpval_harness.judges.base import JudgeRequest
from gdpval_harness.judges.codex import CodexJudgeExecutor
from gdpval_harness.local_runner import _validate_resume_condition
from gdpval_harness.provenance import canonical_json_sha256
from gdpval_harness.reasoning import (
    REASONING_EFFORT_VALUES,
    ReasoningEffortOption,
    validate_reasoning_effort,
)
from gdpval_harness.runner import RunSummary, run_benchmark


class _ExperimentRunConfigValues(TypedDict):
    builder_executor: str
    application_executor: str
    evaluator: str
    builder_model: str | None
    application_model: str | None
    builder_timeout_seconds: float
    application_timeout_seconds: float
    builder_network_enabled: bool
    application_network_enabled: bool
    limit: int
    order_seed: int
    builder_reasoning_effort: ReasoningEffortOption
    application_reasoning_effort: ReasoningEffortOption


JsonObject = dict[str, object]


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AssertionError(f"expected a JSON object, got {type(value).__name__}")
    return cast(JsonObject, value)


def _json_array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError(f"expected a JSON array, got {type(value).__name__}")
    return value


def _load_json_object(text: str) -> JsonObject:
    return _json_object(json.loads(text))


class _FakeBenchmark(Benchmark):
    name = "reasoning-benchmark"
    revision = "reasoning-revision"

    def __init__(self, task_count: int = 2) -> None:
        self.tasks = tuple(BenchmarkTask(TaskSpec(f"task-{index}", f"prompt-{index}")) for index in range(task_count))

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        return None

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        return list(self.tasks[:limit])

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        del task, workspace
        return []


class _FakeEvaluator(Evaluator):
    name = "reasoning-evaluator"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def validate_plan(self, plan: EvaluationPlan) -> None:
        del plan

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        return EvaluatorPreflightResult(self.name, self.evaluator_type, True, version="evaluator-1")

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult(request.task_id, EvaluationStatus.COMPLETED, {"accuracy": 1.0})


class _FakeExecutor(Executor):
    """Deterministic codex-named executor used for metadata and experiment tests."""

    name = "codex"
    invocation_mode = "fake-codex"

    def __init__(self, reasoning_effort: ReasoningEffortOption, *, skill: bool = False) -> None:
        self.reasoning_effort = reasoning_effort
        self.skill = skill
        self.preflight_calls = 0
        self.requests: list[ExecutionRequest] = []

    def preflight(self) -> PreflightResult:
        self.preflight_calls += 1
        return PreflightResult(self.name, True, version="fake-codex-1", auth_mode="fake")

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        request.workspace.mkdir(parents=True, exist_ok=True)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        request.deliverables_dir.mkdir(parents=True, exist_ok=True)
        if self.skill:
            skill = request.deliverables_dir / "generated-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                "---\nname: generated-skill\ndescription: generated test skill\n---\n\nUse it.\n",
                encoding="utf-8",
            )
        else:
            (request.deliverables_dir / "answer.txt").write_text("answer\n", encoding="utf-8")
        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="fake-codex-1",
            invocation_mode=self.invocation_mode,
            auth_mode="fake",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=ExecutionStatus.COMPLETED,
            started_at="started",
            finished_at="finished",
            exit_code=0,
            output_text="answer",
            reasoning_effort_requested=self.reasoning_effort,
        )


class _NonCodexExecutor(_FakeExecutor):
    name = "claude-code"


class _ExperimentEvaluator(_FakeEvaluator):
    name = "experiment-evaluator"


class ReasoningEffortContractTests(unittest.TestCase):
    def _execution_request(self, root: Path) -> ExecutionRequest:
        return ExecutionRequest(
            task=TaskSpec("task", "do work"),
            workspace=root / "workspace",
            deliverables_dir=root / "workspace" / "deliverables",
            executor_dir=root / "executor",
            model="model",
        )

    def _judge_request(self, root: Path) -> JudgeRequest:
        workspace = root / "judge-workspace"
        return JudgeRequest(
            task_id="task",
            task_prompt="judge work",
            workspace=workspace,
            reference_dir=workspace / "reference_files",
            submission_a_dir=workspace / "submission_a",
            submission_b_dir=workspace / "submission_b",
            executor_dir=root / "judge-executor",
            trial_index=0,
            swapped=False,
            model="model",
        )

    def _run_config(self, **updates: object) -> ExperimentRunConfig:
        values: dict[str, object] = {
            "builder_executor": "codex",
            "application_executor": "codex",
            "evaluator": "experiment-evaluator",
            "builder_model": None,
            "application_model": None,
            "builder_timeout_seconds": 1.0,
            "application_timeout_seconds": 1.0,
            "builder_network_enabled": False,
            "application_network_enabled": False,
            "limit": 1,
            "order_seed": 0,
        }
        values.update(updates)
        return ExperimentRunConfig(**cast(_ExperimentRunConfigValues, values))

    def test_allowlist_is_exact_and_non_codex_effort_is_rejected(self) -> None:
        self.assertEqual(
            REASONING_EFFORT_VALUES,
            ("minimal", "low", "medium", "high", "xhigh", "max"),
        )
        self.assertEqual(
            tuple(validate_reasoning_effort(value) for value in REASONING_EFFORT_VALUES),
            REASONING_EFFORT_VALUES,
        )
        self.assertIsNone(validate_reasoning_effort(None))
        for value in ("ultra", "x-high", "MAX", "", 1, object()):
            with self.assertRaises(ValueError):
                validate_reasoning_effort(value)
        with self.assertRaises(ValueError):
            create_executor("claude-code", reasoning_effort="low")
        with self.assertRaises(ValueError):
            self._run_config(builder_executor="claude-code", builder_reasoning_effort="low")

    def test_application_and_local_judge_argv_add_one_exact_effort_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution_request = self._execution_request(root)
            application_without_effort = CodexExecutor(command="codex").build_command(execution_request)
            application_with_effort = CodexExecutor(command="codex", reasoning_effort="max").build_command(
                execution_request
            )
            self._assert_one_effort_pair(application_with_effort, application_without_effort)
            self.assertNotIn("model_reasoning_effort=", application_without_effort)
            self.assertEqual(
                CodexExecutor(command="codex").build_command(execution_request), application_without_effort
            )

            judge_request = self._judge_request(root)
            judge_without_effort = CodexJudgeExecutor(command="codex").build_command(judge_request)
            judge_with_effort = CodexJudgeExecutor(command="codex", reasoning_effort="max").build_command(
                judge_request
            )
            self._assert_one_effort_pair(judge_with_effort, judge_without_effort)
            self.assertNotIn("model_reasoning_effort=", judge_without_effort)
            self.assertEqual(CodexJudgeExecutor(command="codex").build_command(judge_request), judge_without_effort)

    def test_generic_metadata_and_typed_execution_record_track_effort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._run_generic(root / "baseline", None)
            baseline_again = self._run_generic(root / "baseline-again", None)
            requested = self._run_generic(root / "requested", "high")
            baseline_metadata = self._read_json(root / "baseline" / "run-metadata.json")
            baseline_again_metadata = self._read_json(root / "baseline-again" / "run-metadata.json")
            requested_metadata = self._read_json(root / "requested" / "run-metadata.json")
            baseline_configuration = _json_object(baseline_metadata["configuration"])
            baseline_executor = _json_object(baseline_configuration["executor"])
            requested_configuration = _json_object(requested_metadata["configuration"])
            requested_executor = _json_object(requested_configuration["executor"])

            self.assertEqual(baseline.status, "completed")
            self.assertEqual(baseline_again.status, "completed")
            self.assertEqual(requested.status, "completed")
            self.assertEqual(
                set(baseline_configuration),
                {
                    "benchmark",
                    "executor",
                    "evaluator",
                    "intervention",
                    "model",
                    "network_policy",
                    "limit",
                    "timeout_seconds",
                    "runtime_layout",
                },
            )
            self.assertNotIn("reasoning_effort_requested", baseline_metadata)
            self.assertNotIn("reasoning_effort_requested", baseline_executor)
            self.assertEqual(
                baseline_metadata["configuration_sha256"],
                canonical_json_sha256(baseline_configuration),
            )
            self.assertEqual(
                baseline_metadata["configuration_sha256"], baseline_again_metadata["configuration_sha256"]
            )
            self.assertEqual(
                baseline_metadata["run_fingerprint_sha256"], baseline_again_metadata["run_fingerprint_sha256"]
            )

            self.assertEqual(requested_metadata["reasoning_effort_requested"], "high")
            self.assertEqual(requested_executor["reasoning_effort_requested"], "high")
            self.assertNotEqual(baseline_metadata["configuration_sha256"], requested_metadata["configuration_sha256"])
            self.assertNotEqual(
                baseline_metadata["run_fingerprint_sha256"], requested_metadata["run_fingerprint_sha256"]
            )
            requested_row = self._read_jsonl(root / "requested" / "results.jsonl")[0]
            baseline_row = self._read_jsonl(root / "baseline" / "results.jsonl")[0]
            requested_execution = _json_object(requested_row["execution"])
            baseline_execution = _json_object(baseline_row["execution"])
            self.assertEqual(requested_execution["reasoning_effort_requested"], "high")
            self.assertNotIn("reasoning_effort_requested", baseline_execution)

    def test_non_codex_effort_rejects_before_preflight_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = _NonCodexExecutor("low")
            out_dir = root / "run"
            with self.assertRaises(ValueError):
                run_benchmark(
                    _FakeBenchmark(task_count=1),
                    _FakeEvaluator(),
                    executor,
                    out_dir=out_dir,
                    limit=1,
                )
            self.assertEqual(executor.preflight_calls, 0)
            self.assertFalse(out_dir.exists())

    def test_experiment_builder_and_application_efforts_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_metadata = self._run_experiment(root / "baseline", None, None)
            both_low_metadata = self._run_experiment(root / "both-low", "low", "low")
            builder_high_metadata = self._run_experiment(root / "builder-high", "high", "low")
            application_high_metadata = self._run_experiment(root / "application-high", "high", "high")

            baseline_config = _json_object(baseline_metadata["configuration"])
            baseline_run_config = _json_object(baseline_config["run_config"])
            baseline_builder = _json_object(baseline_config["builder"])
            baseline_application = _json_object(baseline_config["application_executor"])
            self.assertNotIn("builder_reasoning_effort_requested", baseline_run_config)
            self.assertNotIn("application_reasoning_effort_requested", baseline_run_config)
            self.assertNotIn("reasoning_effort_requested", baseline_builder)
            self.assertNotIn("reasoning_effort_requested", baseline_application)

            both_low_config = _json_object(both_low_metadata["configuration"])
            both_low_run_config = _json_object(both_low_config["run_config"])
            both_low_builder = _json_object(both_low_config["builder"])
            both_low_application = _json_object(both_low_config["application_executor"])
            self.assertEqual(both_low_run_config["builder_reasoning_effort_requested"], "low")
            self.assertEqual(both_low_run_config["application_reasoning_effort_requested"], "low")
            self.assertEqual(both_low_builder["reasoning_effort_requested"], "low")
            self.assertEqual(both_low_application["reasoning_effort_requested"], "low")

            builder_high_config = _json_object(builder_high_metadata["configuration"])
            builder_high_builder = _json_object(builder_high_config["builder"])
            builder_high_application = _json_object(builder_high_config["application_executor"])
            self.assertEqual(builder_high_builder["reasoning_effort_requested"], "high")
            self.assertEqual(builder_high_application["reasoning_effort_requested"], "low")
            self.assertNotEqual(
                both_low_metadata["configuration_sha256"], builder_high_metadata["configuration_sha256"]
            )
            self.assertNotEqual(
                both_low_metadata["run_fingerprint_sha256"], builder_high_metadata["run_fingerprint_sha256"]
            )

            application_high_config = _json_object(application_high_metadata["configuration"])
            application_high_builder = _json_object(application_high_config["builder"])
            application_high_application = _json_object(application_high_config["application_executor"])
            self.assertEqual(application_high_builder["reasoning_effort_requested"], "high")
            self.assertEqual(application_high_application["reasoning_effort_requested"], "high")
            self.assertNotEqual(
                builder_high_metadata["configuration_sha256"], application_high_metadata["configuration_sha256"]
            )
            self.assertNotEqual(
                builder_high_metadata["run_fingerprint_sha256"], application_high_metadata["run_fingerprint_sha256"]
            )

    def test_legacy_resume_accepts_missing_as_none_and_rejects_mismatch_or_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            metadata_path = out_dir / "run-metadata.json"
            metadata_path.write_text(json.dumps({"configuration": {}}), encoding="utf-8")
            with patch.dict("os.environ", {"RESUME": "1"}, clear=True):
                _validate_resume_condition(out_dir)

            metadata_path.write_text(
                json.dumps({"configuration": {"reasoning_effort_requested": None}}), encoding="utf-8"
            )
            with patch.dict("os.environ", {"RESUME": "1"}, clear=True):
                _validate_resume_condition(out_dir)

            with patch.dict("os.environ", {"RESUME": "1", "GDPVAL_REASONING_EFFORT": "high"}, clear=True):
                with self.assertRaises(ValueError):
                    _validate_resume_condition(out_dir)

            metadata_path.write_text(
                json.dumps({"configuration": {"reasoning_effort_requested": "max"}}), encoding="utf-8"
            )
            with patch.dict("os.environ", {"RESUME": "1", "GDPVAL_REASONING_EFFORT": "high"}, clear=True):
                with self.assertRaises(ValueError):
                    _validate_resume_condition(out_dir)

            metadata_path.write_text(
                json.dumps(
                    {
                        "configuration": {"reasoning_effort_requested": "max"},
                        "resume_fingerprint_sha256": "corrupted",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"RESUME": "1", "GDPVAL_REASONING_EFFORT": "max"}, clear=True):
                with self.assertRaises(ValueError):
                    _validate_resume_condition(out_dir)

    def test_old_codex_max_failure_is_durable_and_stops_generic_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = CodexExecutor(command="old-codex", reasoning_effort="max")
            executor._version = "old-codex-1"
            calls: list[list[str]] = []

            def fake_run(
                command: list[str], *, input: str | None = None, **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                del input, kwargs
                if command and command[0] == "old-codex":
                    calls.append(command)
                    return subprocess.CompletedProcess(command, 2, stdout="", stderr="invalid enum value max")
                raise AssertionError(f"unexpected subprocess command: {command!r}")

            with (
                patch.object(
                    executor,
                    "preflight",
                    return_value=PreflightResult(
                        "codex", True, version="old-codex-1", auth_mode="chatgpt-subscription"
                    ),
                ),
                patch("gdpval_harness.executors.codex.subprocess.run", side_effect=fake_run),
            ):
                summary = run_benchmark(
                    _FakeBenchmark(task_count=2),
                    _FakeEvaluator(),
                    executor,
                    out_dir=root / "run",
                    limit=2,
                )

            self.assertEqual(summary.status, "failed")
            self.assertEqual(len(calls), 1)
            self._assert_one_effort_pair(calls[0], None)
            self.assertIn(["-c", 'model_reasoning_effort="max"'], _pairs(calls[0]))
            self.assertNotIn("xhigh", calls[0])
            metadata = self._read_json(root / "run" / "run-metadata.json")
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["reasoning_effort_requested"], "max")
            rows = self._read_jsonl(root / "run" / "results.jsonl")
            self.assertEqual(len(rows), 1)
            execution = _json_object(rows[0]["execution"])
            self.assertEqual(execution["reasoning_effort_requested"], "max")
            self.assertEqual(execution["exit_code"], 2)
            self.assertTrue((root / "run" / "tasks" / "task-0" / "result.json").is_file())

    def test_old_codex_max_failure_is_durable_and_stops_experiment_applications(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            application = CodexExecutor(command="old-codex", reasoning_effort="max")
            application._version = "old-codex-1"
            calls: list[list[str]] = []

            def fake_run(
                command: list[str], *, input: str | None = None, **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                del input, kwargs
                if command and command[0] == "old-codex":
                    calls.append(command)
                    return subprocess.CompletedProcess(command, 2, stdout="", stderr="invalid enum value max")
                raise AssertionError(f"unexpected subprocess command: {command!r}")

            profile, config, benchmark, evaluator, builder, source_roots = self._experiment_fixture(
                root, task_count=2, application_reasoning_effort="max"
            )
            with (
                patch(
                    "gdpval_harness.experiments.runner.load_experiment_inputs",
                    return_value={"input-guide": source_roots},
                ),
                patch(
                    "gdpval_harness.experiments.runner.secrets.token_hex",
                    side_effect=[f"{index:032x}" for index in range(1, 20)],
                ),
                patch(
                    "gdpval_harness.runner.secrets.token_urlsafe",
                    side_effect=[f"application-run-{index}" for index in range(1, 20)],
                ),
                patch.object(
                    application,
                    "preflight",
                    return_value=PreflightResult(
                        "codex", True, version="old-codex-1", auth_mode="chatgpt-subscription"
                    ),
                ),
                patch("gdpval_harness.executors.codex.subprocess.run", side_effect=fake_run),
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
            self.assertEqual(summary.completed_applications, 0)
            self.assertEqual(len(calls), 1)
            self.assertIn(["-c", 'model_reasoning_effort="max"'], _pairs(calls[0]))
            self.assertNotIn("xhigh", calls[0])
            experiment_metadata = self._read_json(root / "output" / "experiment-metadata.json")
            self.assertEqual(experiment_metadata["status"], "failed")
            self.assertEqual(experiment_metadata["completed_applications"], 0)
            run_config = _json_object(experiment_metadata["run_config"])
            experiment_configuration = _json_object(experiment_metadata["configuration"])
            experiment_application = _json_object(experiment_configuration["application_executor"])
            self.assertEqual(run_config["application_reasoning_effort_requested"], "max")
            self.assertEqual(
                experiment_application["reasoning_effort_requested"],
                "max",
            )
            entries = _json_array(experiment_metadata["entries"])
            if len(entries) < 2:
                raise AssertionError("expected at least two experiment entries")
            first_entry = _json_object(entries[0])
            first_application = _json_object(first_entry["application"])
            application_metadata_path = Path(cast(str, first_application["run_metadata_path"]))
            application_metadata = self._read_json(application_metadata_path)
            self.assertEqual(application_metadata["status"], "failed")
            self.assertEqual(application_metadata["reasoning_effort_requested"], "max")
            application_row = self._read_jsonl(application_metadata_path.with_name("results.jsonl"))[0]
            application_execution = _json_object(application_row["execution"])
            self.assertEqual(application_execution["reasoning_effort_requested"], "max")
            self.assertEqual(application_execution["exit_code"], 2)
            second_entry = _json_object(entries[1])
            self.assertIsNone(second_entry["application"])

    def _run_generic(self, out_dir: Path, effort: ReasoningEffortOption) -> RunSummary:
        return run_benchmark(
            _FakeBenchmark(task_count=1),
            _FakeEvaluator(),
            _FakeExecutor(effort),
            out_dir=out_dir,
            limit=1,
        )

    def _experiment_fixture(
        self,
        root: Path,
        *,
        task_count: int = 1,
        builder_reasoning_effort: ReasoningEffortOption = None,
        application_reasoning_effort: ReasoningEffortOption = None,
    ) -> tuple[
        LoadedExperimentProfile,
        ExperimentRunConfig,
        _FakeBenchmark,
        _ExperimentEvaluator,
        ExecutorSkillBuilder,
        BuilderInputBundle,
    ]:
        source = root / "input-source"
        source.mkdir(parents=True)
        (source / "guide.txt").write_text("allowlisted input", encoding="utf-8")
        source_bundle = load_builder_input_bundle(
            source,
            input_id="input-guide",
            input_type="reference",
            allowed_files=("guide.txt",),
        )
        profile_source = root / "profile.json"
        profile_source.write_text("{}\n", encoding="utf-8")
        profile = LoadedExperimentProfile(
            ExperimentProfile(
                1,
                "reasoning-profile",
                _FakeBenchmark.name,
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
                (ExperimentArm("arm", ("input-guide",)),),
            ),
            profile_source,
            "a" * 64,
        )
        config = self._run_config(
            limit=task_count,
            builder_reasoning_effort=builder_reasoning_effort,
            application_reasoning_effort=application_reasoning_effort,
        )
        benchmark = _FakeBenchmark(task_count)
        evaluator = _ExperimentEvaluator()
        builder = ExecutorSkillBuilder(_FakeExecutor(builder_reasoning_effort, skill=True))
        return profile, config, benchmark, evaluator, builder, source_bundle

    def _run_experiment(
        self,
        root: Path,
        builder_reasoning_effort: ReasoningEffortOption,
        application_reasoning_effort: ReasoningEffortOption,
    ) -> dict[str, object]:
        profile, config, benchmark, evaluator, builder, source_bundle = self._experiment_fixture(
            root,
            builder_reasoning_effort=builder_reasoning_effort,
            application_reasoning_effort=application_reasoning_effort,
        )
        application = _FakeExecutor(application_reasoning_effort)
        with (
            patch(
                "gdpval_harness.experiments.runner.load_experiment_inputs",
                return_value={"input-guide": source_bundle},
            ),
            patch(
                "gdpval_harness.experiments.runner.secrets.token_hex",
                side_effect=[f"{index:032x}" for index in range(1, 20)],
            ),
            patch(
                "gdpval_harness.runner.secrets.token_urlsafe",
                side_effect=[f"application-run-{index}" for index in range(1, 20)],
            ),
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
        return self._read_json(root / "output" / "experiment-metadata.json")

    @staticmethod
    def _assert_one_effort_pair(command: list[str], baseline: list[str] | None) -> None:
        effort_tokens = [token for token in command if token.startswith("model_reasoning_effort=")]
        if baseline is None:
            if len(effort_tokens) != 1:
                raise AssertionError(f"expected one reasoning effort token, got {effort_tokens!r}")
            effort_index = command.index(effort_tokens[0])
            if effort_index == 0 or command[effort_index - 1] != "-c":
                raise AssertionError("reasoning effort must be one -c key/value pair")
            return
        if effort_tokens != ['model_reasoning_effort="max"']:
            raise AssertionError(f"expected exact max token once, got {effort_tokens!r}")
        effort_index = command.index(effort_tokens[0])
        if effort_index == 0 or command[effort_index - 1] != "-c":
            raise AssertionError("reasoning effort must be one -c key/value pair")
        without_effort = command[: effort_index - 1] + command[effort_index + 1 :]
        if without_effort != baseline:
            raise AssertionError("setting reasoning effort changed the command outside its one -c pair")

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        return _load_json_object(path.read_text(encoding="utf-8"))

    @classmethod
    def _read_jsonl(cls, path: Path) -> list[dict[str, object]]:
        return [_load_json_object(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _pairs(command: list[str]) -> list[list[str]]:
    return [command[index : index + 2] for index, value in enumerate(command[:-1]) if value == "-c"]


if __name__ == "__main__":
    unittest.main()
