# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import unittest
from argparse import _SubParsersAction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import gdpval_harness.cli as cli
from gdpval_harness.experiments.base import ExperimentRunConfig


class ExperimentCLIHelpers:
    profile = SimpleNamespace(
        profile_id="profile-a",
        benchmark="benchmark-a",
        inputs=(SimpleNamespace(input_id="input-a"), SimpleNamespace(input_id="input-b")),
        arms=(SimpleNamespace(arm_id="arm-a"), SimpleNamespace(arm_id="arm-b")),
    )

    def argv(self, *extra: str) -> list[str]:
        return [
            "experiment",
            "profile.json",
            "--input-root",
            "input-a=source-a=preserved",
            "--input-root",
            "input-b=source-b",
            "--limit",
            "3",
            "--order-seed",
            "7",
            "--out",
            "results/out",
            "--runtime-root",
            "results/runtime",
            *extra,
        ]

    def patch_runtime(self, *, status: str = "completed") -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        self.loader = Mock(return_value=SimpleNamespace(profile=self.profile))
        self.benchmark_descriptor = Mock(
            return_value=SimpleNamespace(supported_executors=("application-cli", "codex"))
        )
        self.executor_descriptor = Mock()
        self.benchmark_factory = Mock(return_value=SimpleNamespace(name="benchmark-a"))
        self.evaluator_factory = Mock(return_value=SimpleNamespace(name="evaluator-a"))
        self.builder_executor = SimpleNamespace(name="builder-a")
        self.application_executor = SimpleNamespace(name="application-a")
        self.executor_factory = Mock(side_effect=[self.builder_executor, self.application_executor])
        self.builder_factory = Mock(return_value="builder-wrapper")
        self.summary = SimpleNamespace(
            out_dir=Path("/abs/out"),
            runtime_root=Path("/abs/runtime"),
            status=status,
            task_count=3,
            arm_count=2,
            completed_applications=5,
        )
        self.runner = Mock(return_value=self.summary)

        stack.enter_context(patch.object(cli, "load_experiment_profile", self.loader))
        stack.enter_context(patch.object(cli, "get_benchmark_descriptor", self.benchmark_descriptor))
        stack.enter_context(patch.object(cli, "get_executor_descriptor", self.executor_descriptor))
        stack.enter_context(patch.object(cli, "create_benchmark", self.benchmark_factory))
        stack.enter_context(patch.object(cli, "create_evaluator", self.evaluator_factory))
        stack.enter_context(patch.object(cli, "create_executor", self.executor_factory))
        stack.enter_context(patch.object(cli, "ExecutorSkillBuilder", self.builder_factory))
        stack.enter_context(patch.object(cli, "run_builder_experiment", self.runner))
        return stack


class ExperimentCLIParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helpers = ExperimentCLIHelpers()

    def test_experiment_config_and_factories_are_exact(self) -> None:
        with self.helpers.patch_runtime():
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = cli.main(
                    self.helpers.argv(
                        "--builder-executor",
                        "builder-cli",
                        "--executor",
                        "application-cli",
                        "--builder-model",
                        "builder-model",
                        "--model",
                        "application-model",
                        "--builder-timeout",
                        "11.5",
                        "--executor-timeout",
                        "12.5",
                        "--builder-network",
                        "--network",
                        "--builder-claude-max-turns",
                        "17",
                        "--claude-max-turns",
                        "19",
                    )
                )

            self.assertEqual(result, 0)
            self.helpers.loader.assert_called_once_with(Path.cwd() / "profile.json")
            self.helpers.benchmark_descriptor.assert_called_once_with("benchmark-a")
            self.helpers.executor_descriptor.assert_has_calls([call("builder-cli"), call("application-cli")])
            self.assertEqual(self.helpers.executor_descriptor.call_count, 2)
            self.helpers.benchmark_factory.assert_called_once_with("benchmark-a")
            self.helpers.evaluator_factory.assert_called_once_with("benchmark-a")
            self.helpers.executor_factory.assert_has_calls(
                [
                    call("builder-cli", network_enabled=True, claude_max_turns=17),
                    call("application-cli", network_enabled=True, claude_max_turns=19),
                ]
            )
            self.assertEqual(self.helpers.executor_factory.call_count, 2)
            self.assertIsNot(self.helpers.builder_executor, self.helpers.application_executor)
            self.helpers.builder_factory.assert_called_once_with(self.helpers.builder_executor)

            runner_args, runner_kwargs = self.helpers.runner.call_args
            self.assertIs(runner_args[0], self.helpers.loader.return_value)
            config = runner_args[1]
            self.assertEqual(
                config,
                ExperimentRunConfig(
                    builder_executor="builder-a",
                    application_executor="application-a",
                    evaluator="evaluator-a",
                    builder_model="builder-model",
                    application_model="application-model",
                    builder_timeout_seconds=11.5,
                    application_timeout_seconds=12.5,
                    builder_network_enabled=True,
                    application_network_enabled=True,
                    limit=3,
                    order_seed=7,
                ),
            )
            self.assertFalse(any("arm" in field for field in config.__dataclass_fields__))
            self.assertIs(runner_args[2], self.helpers.benchmark_factory.return_value)
            self.assertIs(runner_args[3], self.helpers.evaluator_factory.return_value)
            self.assertEqual(runner_args[4], "builder-wrapper")
            self.assertIs(runner_args[5], self.helpers.application_executor)
            self.assertEqual(
                runner_kwargs,
                {
                    "source_roots": {
                        "input-a": Path("source-a=preserved"),
                        "input-b": Path("source-b"),
                    },
                    "out_dir": (Path.cwd() / "results/out").absolute(),
                    "runtime_root": (Path.cwd() / "results/runtime").absolute(),
                },
            )
            self.assertEqual(
                json.loads(stdout.getvalue()),
                {
                    "application_executor": "application-a",
                    "arm_count": 2,
                    "benchmark": "benchmark-a",
                    "builder_executor": "builder-a",
                    "completed_applications": 5,
                    "out": "/abs/out",
                    "profile": "profile-a",
                    "runtime_root": "/abs/runtime",
                    "status": "completed",
                    "task_count": 3,
                },
            )

    def test_defaults_do_not_add_arm_overrides(self) -> None:
        with self.helpers.patch_runtime():
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(self.helpers.argv()), 0)

            self.helpers.executor_factory.assert_has_calls(
                [
                    call("codex", network_enabled=False, claude_max_turns=250),
                    call("codex", network_enabled=False, claude_max_turns=250),
                ]
            )
            config = self.helpers.runner.call_args.args[1]
            self.assertIsNone(config.builder_model)
            self.assertIsNone(config.application_model)
            self.assertEqual(config.builder_timeout_seconds, 12600.0)
            self.assertEqual(config.application_timeout_seconds, 12600.0)
            self.assertFalse(config.builder_network_enabled)
            self.assertFalse(config.application_network_enabled)
            self.assertEqual(config.limit, 3)
            self.assertEqual(config.order_seed, 7)

    def test_input_binding_uses_first_equals_and_rejects_empty_or_duplicate_ids(self) -> None:
        self.assertEqual(
            cli._parse_input_roots(["input-a=source=with-equals"]),
            {"input-a": Path("source=with-equals")},
        )
        for binding in ("=source", "input-a=", "input-a"):
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                cli._parse_input_roots([binding])
        with self.assertRaises(ValueError):
            cli._parse_input_roots(["input-a=source", "input-a=other"])

    def test_duplicate_binding_is_rejected_before_factories(self) -> None:
        with self.helpers.patch_runtime():
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = cli.main(self.helpers.argv("--input-root", "input-a=second"))
            self.assertEqual(result, 2)
            self.assertIn("duplicate", stderr.getvalue())
            self.helpers.benchmark_descriptor.assert_not_called()
            self.helpers.benchmark_factory.assert_not_called()
            self.helpers.evaluator_factory.assert_not_called()
            self.helpers.executor_factory.assert_not_called()
            self.helpers.runner.assert_not_called()

    def test_unsupported_application_executor_stops_before_factories(self) -> None:
        with self.helpers.patch_runtime():
            self.helpers.benchmark_descriptor.return_value = SimpleNamespace(supported_executors=("other",))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = cli.main(self.helpers.argv("--executor", "application-a"))
            self.assertEqual(result, 2)
            self.assertIn("does not support executor", stderr.getvalue())
            self.helpers.benchmark_factory.assert_not_called()
            self.helpers.evaluator_factory.assert_not_called()
            self.helpers.executor_factory.assert_not_called()
            self.helpers.runner.assert_not_called()

    def test_invalid_numeric_argument_is_rejected_before_profile_loader(self) -> None:
        with (
            patch.object(cli, "load_experiment_profile", Mock()) as loader,
            patch.object(cli, "run_builder_experiment", Mock()) as runner,
        ):
            with self.assertRaises(SystemExit) as raised:
                cli.main(self.helpers.argv("--limit", "0"))
            self.assertEqual(raised.exception.code, 2)
            loader.assert_not_called()
            runner.assert_not_called()

    def test_malformed_loader_result_is_rejected_before_factories(self) -> None:
        with self.helpers.patch_runtime():
            self.helpers.loader.return_value = SimpleNamespace(benchmark="benchmark-a")
            args = cli._parser().parse_args(self.helpers.argv())
            with self.assertRaises(AttributeError):
                cli._experiment(args)
            self.helpers.benchmark_descriptor.assert_not_called()
            self.helpers.runner.assert_not_called()

    def test_generic_experiment_help_has_required_controls(self) -> None:
        parser = cli._parser()
        subparser_group = parser._subparsers
        assert subparser_group is not None
        subparser_actions = subparser_group._group_actions
        assert subparser_actions
        subparsers = subparser_actions[0]
        assert isinstance(subparsers, _SubParsersAction)
        experiment_parser = subparsers.choices["experiment"]
        rendered = experiment_parser.format_help()
        flags = ("--input-root", "--limit", "--order-seed", "--out", "--runtime-root", "--builder-executor")
        for flag in flags:
            with self.subTest(flag=flag):
                self.assertIn(flag, rendered)


class ExperimentCLIReturnCodeTests(unittest.TestCase):
    def test_failed_and_interrupted_statuses_return_one_and_are_printed(self) -> None:
        helpers = ExperimentCLIHelpers()
        for status in ("failed", "interrupted"):
            with self.subTest(status=status), helpers.patch_runtime(status=status):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    result = cli.main(helpers.argv())
                self.assertEqual(result, 1)
                self.assertEqual(json.loads(stdout.getvalue())["status"], status)


if __name__ == "__main__":
    unittest.main()
