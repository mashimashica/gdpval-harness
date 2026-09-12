# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, cast
from unittest.mock import Mock, patch

import eval_harness.benchmarks.registry as benchmark_registry
import eval_harness.cli as cli
import eval_harness.evaluators.registry as evaluator_registry
import eval_harness.executors.registry as executor_registry
import eval_harness.interventions.registry as intervention_registry
from eval_harness.benchmarks.aime26 import AIME26Benchmark
from eval_harness.benchmarks.bigcodebench import BigCodeBenchBenchmark
from eval_harness.benchmarks.gdpval import GDPvalBenchmark
from eval_harness.benchmarks.registry import create_benchmark, get_benchmark_descriptor, list_benchmarks
from eval_harness.capabilities import ExecutorOutput
from eval_harness.evaluators.base import (
    EvaluationCandidate,
    EvaluationPlan,
    EvaluationRequest,
    EvaluatorPreflightResult,
    EvaluatorType,
    require_two_candidates,
)
from eval_harness.evaluators.exact import ExactMatchEvaluator
from eval_harness.evaluators.registry import (
    EvaluatorDescriptor,
    create_evaluator,
    create_pairwise_evaluator,
    get_evaluator_descriptor,
    list_evaluator_descriptors,
    list_evaluators,
)
from eval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec
from eval_harness.executors.registry import create_executor, get_executor_descriptor, list_executors
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.interventions import NoneIntervention, get_intervention
from eval_harness.interventions.registry import create_intervention
from eval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult


class RegistryJudge(JudgeExecutor):
    name = "registry-judge"
    invocation_mode = "registry"

    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        del environment
        return JudgePreflightResult(judge_executor=self.name, ok=True)

    def judge(self, request: JudgeRequest) -> JudgeResult:
        del request
        raise AssertionError("registry construction must not judge")


def execution_result(root: Path, *, status: ExecutionStatus = ExecutionStatus.COMPLETED) -> ExecutionResult:
    workspace = root / "workspace"
    successful = status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
    output_text = "answer" if successful else None
    return ExecutionResult(
        runtime="test",
        task_id="task",
        executor="fake",
        executor_version="fake-v1",
        invocation_mode="test",
        auth_mode="local",
        workspace=workspace,
        deliverables_dir=workspace / "deliverables",
        status=status,
        started_at="2026-09-12T00:00:00+00:00",
        finished_at="2026-09-12T00:00:01+00:00",
        exit_code=0 if successful else 1,
        available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}) if output_text is not None else frozenset(),
        failure=None if successful else Failure(FailureKind.PROCESS, "test_failure", FailureImpact.RUN),
        output_text=output_text,
    )


class CLIRegistryTests(unittest.TestCase):
    @staticmethod
    def run_args(**updates: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "benchmark": "gdpval",
            "executor": "codex",
            "limit": 1,
            "model": None,
            "reasoning_effort": None,
            "out": None,
            "executor_timeout": 12600.0,
            "network": False,
            "claude_max_turns": 250,
            "intervention": "none",
            "intervention_source": None,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    @staticmethod
    def experiment_args(**updates: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "profile": Path("profile.json"),
            "input_root": ["input=source"],
            "limit": 1,
            "order_seed": 0,
            "out": Path("out"),
            "runtime_root": Path("runtime"),
            "builder_executor": "codex",
            "executor": "codex",
            "builder_model": None,
            "model": None,
            "builder_reasoning_effort": None,
            "application_reasoning_effort": None,
            "builder_timeout": 12600.0,
            "executor_timeout": 12600.0,
            "builder_network": False,
            "network": False,
            "builder_claude_max_turns": 250,
            "claude_max_turns": 250,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def test_registry_listings_and_factories_are_deterministic(self) -> None:
        self.assertEqual([item.name for item in list_benchmarks()], ["aime26", "bigcodebench", "gdpval"])
        self.assertEqual([item.name for item in list_executors()], ["claude-code", "codex", "cursor", "stirrup"])
        self.assertEqual(
            [item.name for item in list_evaluator_descriptors()],
            ["aime26-native", "bigcodebench-tests", "gdpval-external"],
        )
        self.assertEqual(list_evaluators(), list_evaluator_descriptors())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsInstance(create_benchmark("gdpval", root=root), GDPvalBenchmark)
            self.assertIsInstance(create_benchmark("aime26", root=root), AIME26Benchmark)
            self.assertIsInstance(create_benchmark("bigcodebench", root=root), BigCodeBenchBenchmark)

        with patch.object(benchmark_registry, "get_benchmark_descriptor", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(AssertionError, "unreachable"):
                benchmark_registry.create_benchmark("unsupported")

        self.assertEqual(get_evaluator_descriptor("aime26-native"), get_evaluator_descriptor("aime26"))
        self.assertEqual(get_executor_descriptor("codex").name, "codex")
        with self.assertRaisesRegex(ValueError, "unknown benchmark.*available"):
            get_benchmark_descriptor("missing")
        with self.assertRaisesRegex(ValueError, "unknown evaluator or benchmark.*available"):
            get_evaluator_descriptor("missing")
        with self.assertRaisesRegex(ValueError, "unknown executor.*available"):
            get_executor_descriptor("missing")

    def test_executor_and_evaluator_construction_never_calls_external_services(self) -> None:
        self.assertEqual(create_executor("codex").name, "codex")
        self.assertEqual(create_executor("claude-code", claude_max_turns=3).name, "claude-code")
        self.assertEqual(create_executor("cursor").name, "cursor")
        with self.assertRaisesRegex(ValueError, "not available through"):
            create_executor("stirrup")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(create_evaluator("gdpval", root=root).name, "gdpval-external")
            self.assertEqual(create_evaluator("aime26", root=root).name, "aime26-native")
            self.assertEqual(create_evaluator("bigcodebench", root=root).name, "bigcodebench-tests")
        pairwise = create_pairwise_evaluator(RegistryJudge(), trials=1, seed=9, model="fake", timeout_seconds=2.0)
        self.assertEqual(pairwise.name, "pairwise-judge")

        unsupported = EvaluatorDescriptor(
            name="unsupported",
            benchmark="unsupported",
            evaluator_type=EvaluatorType.BENCHMARK_NATIVE,
            status="test",
        )
        with patch.object(
            evaluator_registry,
            "get_evaluator_descriptor",
            return_value=unsupported,
        ):
            with self.assertRaisesRegex(AssertionError, "no evaluator factory"):
                create_evaluator("unsupported")

    def test_intervention_compatibility_getter_and_fail_closed_source_rules(self) -> None:
        self.assertIsInstance(get_intervention("none"), NoneIntervention)
        with self.assertRaisesRegex(ValueError, "unknown intervention type"):
            create_intervention("invalid")
        with self.assertRaisesRegex(ValueError, "requires an explicit source"):
            create_intervention("files")

        class UnhandledInterventionType:
            NONE = object()
            PROMPT_OVERLAY = object()
            FILES = object()
            AGENT_SKILL = object()

            def __call__(self, value: object) -> SimpleNamespace:
                del value
                return SimpleNamespace(value="unsupported")

        with patch.object(intervention_registry, "InterventionType", UnhandledInterventionType()):
            with self.assertRaisesRegex(AssertionError, "unhandled intervention type"):
                intervention_registry.create_intervention("unsupported", source=Path("source"))

    def test_none_intervention_requires_preflight_and_exact_task_contract(self) -> None:
        intervention = NoneIntervention()
        task = TaskSpec(task_id="task", prompt="prompt")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                intervention.apply(task, Path(temporary), application_run_id="run")

            preflight = intervention.preflight()
            self.assertTrue(preflight.ok)
            bundle = preflight.bundle
            assert bundle is not None
            self.assertEqual(bundle.manifest.application.method, "none")
            application = intervention.apply(task, Path(temporary), application_run_id="run")
            self.assertEqual(application.task, task)
            self.assertEqual(application.application_run_id, "run")
            with self.assertRaisesRegex(TypeError, "TaskSpec"):
                intervention.validate_task(cast(TaskSpec, object()))

    def test_exact_evaluator_reports_preflight_and_rejects_missing_metadata(self) -> None:
        evaluator = ExactMatchEvaluator()
        preflight = evaluator.preflight(Path("run"))
        self.assertEqual(
            preflight,
            EvaluatorPreflightResult(
                "exact-match",
                EvaluatorType.DETERMINISTIC_EXACT,
                True,
                version="1",
                details=("trimmed exact text comparison is ready",),
            ),
        )
        self.assertEqual(evaluator.evaluator_id, "exact-match")
        self.assertEqual(evaluator.evaluator_name, "exact-match")
        self.assertEqual(evaluator.evaluator, "exact-match")

        with tempfile.TemporaryDirectory() as temporary:
            result = execution_result(Path(temporary))
            missing_metadata = EvaluationRequest(
                task_id="task",
                task_prompt="prompt",
                candidates=(EvaluationCandidate("policy", result),),
            )
            with self.assertRaisesRegex(ValueError, "expected_answer"):
                evaluator.evaluate(missing_metadata)

            failed = execution_result(Path(temporary), status=ExecutionStatus.FAILED)
            evaluation = evaluator.evaluate(
                EvaluationRequest(
                    task_id="task",
                    task_prompt="prompt",
                    metadata={"expected_answer": "answer"},
                    candidates=(EvaluationCandidate("policy", failed),),
                )
            )
            self.assertEqual(evaluation.metrics, {"exact_match": 0.0})
            self.assertFalse(evaluation.outcomes["matched"])

    def test_evaluation_request_normalizes_results_and_pairwise_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = execution_result(root)
            request = EvaluationRequest("task", "prompt", candidates=cast(tuple[EvaluationCandidate, ...], (result,)))
            self.assertIsInstance(request.candidates[0], EvaluationCandidate)
            self.assertEqual(request.candidates[0].candidate_id, "candidate_0")
            self.assertIs(request.candidates[0].result, result)
            self.assertEqual(request.canonical_task_prompt, "prompt")
            self.assertEqual(request.evaluator_metadata, {})
            self.assertIsNone(request.evaluator_artifact_destination)

            with self.assertRaisesRegex(TypeError, "EvaluationCandidate or ExecutionResult"):
                EvaluationRequest("task", "prompt", candidates=cast(tuple[EvaluationCandidate, ...], (object(),)))

            first = EvaluationCandidate("same", result)
            second = EvaluationCandidate("same", result)
            with self.assertRaisesRegex(ValueError, "distinct candidate ids"):
                require_two_candidates(EvaluationRequest("task", "prompt", candidates=(first, second)))

            with self.assertRaisesRegex(ValueError, "non-negative"):
                EvaluationPlan("task", "prompt", {}, -1)

        preflight_result = EvaluatorPreflightResult(
            "evaluator",
            EvaluatorType.BENCHMARK_NATIVE,
            True,
            details=cast(tuple[str, ...], ("ready", 4)),
        )
        self.assertEqual(preflight_result.details, ("ready", "4"))
        self.assertEqual(preflight_result.evaluator_id, "evaluator")
        self.assertEqual(preflight_result.evaluator_name, "evaluator")
        self.assertEqual(preflight_result.evaluator, "evaluator")

    def test_cli_helpers_validate_numeric_inputs_and_default_output(self) -> None:
        output = cli._default_out("gdpval", "codex")
        self.assertEqual(output.parts[:3], ("results", "eval", "gdpval"))
        self.assertTrue(output.name.endswith("-codex"))
        for value in ("bad", "0"):
            with self.assertRaises(argparse.ArgumentTypeError):
                cli._positive_int(value)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "integer"):
            cli._nonnegative_int("bad")
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "non-negative"):
            cli._nonnegative_int("-1")
        for value in ("bad", "nan", "inf", "0"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                cli._positive_float(value)

        parser = cli._parser()
        parsed = parser.parse_args(["run", "gdpval", "--limit", "1"])
        self.assertEqual(parsed.command, "run")
        self.assertEqual(parsed.benchmark, "gdpval")

    def test_cli_prints_stable_registry_rows(self) -> None:
        benchmarks = io.StringIO()
        with contextlib.redirect_stdout(benchmarks):
            cli._print_benchmarks()
        lines = benchmarks.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual([line.split("\t", 1)[0] for line in lines], ["aime26", "bigcodebench", "gdpval"])
        self.assertIn("evaluator=aime26-native", lines[0])
        self.assertIn("sandbox=per-task writable workspace", lines[0])

        executors = io.StringIO()
        with contextlib.redirect_stdout(executors):
            cli._print_executors()
        executor_lines = executors.getvalue().splitlines()
        self.assertEqual(
            [line.split("\t", 1)[0] for line in executor_lines], ["claude-code", "codex", "cursor", "stirrup"]
        )
        self.assertIn("provider-backed model API", executor_lines[-1])

    def test_run_validation_rejects_before_factories_or_output(self) -> None:
        descriptor = SimpleNamespace(supported_executors=("codex",))
        cases = (
            ({"executor": "cursor"}, "does not support executor"),
            ({"limit": 0}, "--limit"),
            ({"executor_timeout": 0}, "executor-timeout"),
            ({"claude_max_turns": 0}, "claude-max-turns"),
            ({"intervention": "none", "intervention_source": Path("source")}, "intervention-source"),
            ({"intervention": "files", "intervention_source": None}, "intervention-source"),
        )
        for updates, message in cases:
            with self.subTest(message=message):
                factory = Mock()
                with (
                    patch.object(cli, "get_benchmark_descriptor", return_value=descriptor),
                    patch.object(cli, "get_executor_descriptor", return_value=SimpleNamespace()),
                    patch.object(cli, "create_benchmark", factory),
                    patch.object(cli, "run_benchmark", factory),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        cli._run(self.run_args(**updates))
                factory.assert_not_called()

    def test_run_fake_success_and_noncompleted_status_print_safe_summary(self) -> None:
        descriptor = SimpleNamespace(supported_executors=("codex",))
        summary = SimpleNamespace(
            benchmark="gdpval",
            executor="codex",
            out_dir=Path("results/out"),
            status="completed",
            task_count=1,
            metrics={"accuracy": 1.0},
            evaluation_status_counts={"completed": 1},
        )
        executor = object()
        with (
            patch.object(cli, "get_benchmark_descriptor", return_value=descriptor),
            patch.object(cli, "get_executor_descriptor", return_value=SimpleNamespace()),
            patch.object(cli, "create_intervention", return_value=NoneIntervention()) as intervention,
            patch.object(cli, "create_benchmark", return_value=object()) as benchmark,
            patch.object(cli, "create_evaluator", return_value=object()) as evaluator,
            patch.object(cli, "create_executor", return_value=executor) as create_executor_mock,
            patch.object(cli, "run_benchmark", return_value=summary) as run_benchmark_mock,
        ):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = cli._run(
                    self.run_args(
                        model="model-id",
                        reasoning_effort="high",
                        network=True,
                        intervention="none",
                    )
                )
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "completed")
        intervention.assert_called_once_with("none", source=None)
        benchmark.assert_called_once_with("gdpval")
        evaluator.assert_called_once_with("gdpval")
        create_executor_mock.assert_called_once_with(
            "codex", network_enabled=True, claude_max_turns=250, reasoning_effort="high"
        )
        self.assertEqual(run_benchmark_mock.call_args.kwargs["model"], "model-id")
        self.assertTrue(str(run_benchmark_mock.call_args.kwargs["out_dir"]).startswith("results/eval/gdpval/"))

        failed = SimpleNamespace(**{**summary.__dict__, "status": "failed"})
        with (
            patch.object(cli, "get_benchmark_descriptor", return_value=descriptor),
            patch.object(cli, "get_executor_descriptor", return_value=SimpleNamespace()),
            patch.object(cli, "create_intervention", return_value=NoneIntervention()),
            patch.object(cli, "create_benchmark", return_value=object()),
            patch.object(cli, "create_evaluator", return_value=object()),
            patch.object(cli, "create_executor", return_value=executor),
            patch.object(cli, "run_benchmark", return_value=failed),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli._run(self.run_args(out=Path("explicit-out"))), 1)

    def test_experiment_validation_and_fake_reasoning_routes(self) -> None:
        profile = SimpleNamespace(profile=SimpleNamespace(profile_id="profile", benchmark="gdpval"))
        invalid_cases = (
            ({"limit": 0}, "--limit"),
            ({"order_seed": -1}, "order-seed"),
            ({"builder_timeout": math.nan}, "builder-timeout"),
            ({"executor_timeout": 0.0}, "executor-timeout"),
            ({"builder_claude_max_turns": 0}, "builder-claude"),
            ({"claude_max_turns": 0}, "claude-max-turns"),
        )
        for updates, message in invalid_cases:
            with self.subTest(message=message), patch.object(cli, "load_experiment_profile", return_value=profile):
                with self.assertRaisesRegex(ValueError, message):
                    cli._experiment(self.experiment_args(**updates))

        builder = SimpleNamespace(name="codex")
        application = SimpleNamespace(name="codex")
        summary = SimpleNamespace(
            status="completed",
            arm_count=2,
            completed_applications=3,
            task_count=1,
            out_dir=Path("/tmp/out"),
            runtime_root=Path("/tmp/runtime"),
        )
        with (
            patch.object(cli, "load_experiment_profile", return_value=profile),
            patch.object(
                cli, "get_benchmark_descriptor", return_value=SimpleNamespace(supported_executors=("codex",))
            ),
            patch.object(cli, "get_executor_descriptor"),
            patch.object(cli, "create_benchmark", return_value=object()),
            patch.object(cli, "create_evaluator", return_value=SimpleNamespace(name="evaluator")),
            patch.object(cli, "create_executor", side_effect=[builder, application]) as executor_factory,
            patch.object(cli, "ExecutorSkillBuilder", return_value="builder-wrapper"),
            patch.object(cli, "run_builder_experiment", return_value=summary),
        ):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = cli._experiment(
                    self.experiment_args(
                        builder_reasoning_effort="high",
                        application_reasoning_effort="low",
                    )
                )
        self.assertEqual(result, 0)
        self.assertEqual(executor_factory.call_args_list[0].kwargs["reasoning_effort"], "high")
        self.assertEqual(executor_factory.call_args_list[1].kwargs["reasoning_effort"], "low")
        self.assertEqual(json.loads(stdout.getvalue())["status"], "completed")

        unexpected = SimpleNamespace(**{**summary.__dict__, "status": "unknown"})
        with (
            patch.object(cli, "load_experiment_profile", return_value=profile),
            patch.object(
                cli, "get_benchmark_descriptor", return_value=SimpleNamespace(supported_executors=("codex",))
            ),
            patch.object(cli, "get_executor_descriptor"),
            patch.object(cli, "create_benchmark", return_value=object()),
            patch.object(cli, "create_evaluator", return_value=SimpleNamespace(name="evaluator")),
            patch.object(cli, "create_executor", side_effect=[builder, application]),
            patch.object(cli, "ExecutorSkillBuilder", return_value="builder-wrapper"),
            patch.object(cli, "run_builder_experiment", return_value=unexpected),
        ):
            with self.assertRaisesRegex(ValueError, "unexpected experiment status"):
                cli._experiment(self.experiment_args())

    def test_cli_path_and_dispatch_failures_are_explicit(self) -> None:
        with patch.object(Path, "absolute", side_effect=OSError("path denied")):
            with self.assertRaisesRegex(ValueError, "could not resolve --out"):
                cli._planned_absolute_path(Path("out"), label="--out")
        with patch.object(Path, "absolute", return_value=Path("relative")):
            with self.assertRaisesRegex(ValueError, "must resolve to an absolute"):
                cli._planned_absolute_path(Path("out"), label="--out")

        parser = cli._parser()
        with (
            patch.object(parser, "parse_args", return_value=argparse.Namespace(command="unknown")),
            patch.object(parser, "error", side_effect=SystemExit(2)) as error,
        ):
            with patch.object(cli, "_parser", return_value=parser):
                with self.assertRaises(SystemExit):
                    cli.main([])
        error.assert_called_once_with("unknown command: unknown")

        with (
            patch.object(parser, "parse_args", return_value=argparse.Namespace(command="unknown")),
            patch.object(parser, "error", return_value=None),
        ):
            with patch.object(cli, "_parser", return_value=parser):
                self.assertEqual(cli.main([]), 2)

    def test_cli_main_dispatches_registry_commands_and_maps_handler_errors(self) -> None:
        benchmarks = io.StringIO()
        with contextlib.redirect_stdout(benchmarks):
            self.assertEqual(cli.main(["benchmarks"]), 0)
        self.assertIn("gdpval", benchmarks.getvalue())

        executors = io.StringIO()
        with contextlib.redirect_stdout(executors):
            self.assertEqual(cli.main(["executors"]), 0)
        self.assertIn("codex", executors.getvalue())

        with patch.object(cli, "_run", return_value=1) as run:
            self.assertEqual(cli.main(["run", "gdpval", "--limit", "1"]), 1)
        run.assert_called_once()

        experiment_argv = [
            "experiment",
            "profile.json",
            "--input-root",
            "input=source",
            "--limit",
            "1",
            "--order-seed",
            "0",
            "--out",
            "out",
            "--runtime-root",
            "runtime",
        ]
        with patch.object(cli, "_experiment", return_value=1) as experiment:
            self.assertEqual(cli.main(experiment_argv), 1)
        experiment.assert_called_once()

        stderr = io.StringIO()
        with (
            patch.object(cli, "_run", side_effect=ValueError("handler rejected input")),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(cli.main(["run", "gdpval", "--limit", "1"]), 2)
        self.assertIn("error: handler rejected input", stderr.getvalue())

    def test_executor_registry_main_prints_all_descriptors(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            executor_registry.main()
        lines = stdout.getvalue().splitlines()
        self.assertEqual([line.split("\t", 1)[0] for line in lines], ["claude-code", "codex", "cursor", "stirrup"])


if __name__ == "__main__":
    unittest.main()
