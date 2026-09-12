# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from unittest.mock import patch

import eval_harness.benchmarks.gdpval as gdpval_benchmark
import eval_harness.evaluators.aime26 as aime26_evaluator
import eval_harness.evaluators.bigcodebench as bigcodebench_evaluator
import eval_harness.evaluators.gdpval as gdpval_evaluator
import eval_harness.evaluators.pairwise as pairwise_evaluator
from eval_harness.benchmarks.aime26 import AIME26Benchmark
from eval_harness.benchmarks.base import BenchmarkTask
from eval_harness.benchmarks.bigcodebench import BigCodeBenchBenchmark
from eval_harness.benchmarks.gdpval import GDPvalBenchmark
from eval_harness.evaluators.aime26 import AIME26Evaluator
from eval_harness.evaluators.base import (
    EvaluationCandidate,
    EvaluationPlan,
    EvaluationRequest,
    EvaluationStatus,
)
from eval_harness.evaluators.bigcodebench import BigCodeBenchEvaluator
from eval_harness.evaluators.exact import ExactMatchEvaluator
from eval_harness.evaluators.gdpval import GDPvalExternalEvaluator
from eval_harness.evaluators.pairwise import PairwiseJudgeEvaluator
from eval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec
from eval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult, Verdict


def _execution(
    root: Path,
    *,
    task_id: str = "task",
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    output_text: str | None = "answer",
    workspace: Path | None = None,
    deliverables: Path | None = None,
) -> ExecutionResult:
    execution_workspace = workspace or root / "workspace"
    execution_deliverables = deliverables or execution_workspace / "deliverables"
    execution_deliverables.mkdir(parents=True, exist_ok=True)
    return ExecutionResult(
        task_id=task_id,
        executor="fake",
        executor_version="fake-1",
        invocation_mode="fake",
        auth_mode="local",
        workspace=execution_workspace,
        deliverables_dir=execution_deliverables,
        status=status,
        started_at="2026-09-12T00:00:00+00:00",
        finished_at="2026-09-12T00:00:01+00:00",
        exit_code=0 if status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE} else 1,
        output_text=output_text,
    )


def _single_request(
    root: Path,
    result: ExecutionResult,
    *,
    task_id: str = "task",
    metadata: Mapping[str, object] | None = None,
    artifact_dir: Path | None = None,
) -> EvaluationRequest:
    return EvaluationRequest(
        task_id=task_id,
        task_prompt="task prompt",
        metadata={} if metadata is None else metadata,
        candidates=(EvaluationCandidate("policy", result, result.deliverables_dir),),
        artifact_dir=artifact_dir,
    )


class _MathHelper:
    def __init__(self, extracted: str | None, raw_result: object) -> None:
        self.extracted = extracted
        self.raw_result = raw_result
        self.metric_calls = 0
        self.verify_calls = 0

    def _extract_last_boxed_answer(self, text: str) -> str | None:
        del text
        return self.extracted

    def LatexExtractionConfig(self) -> object:
        return object()

    def ExprExtractionConfig(self) -> object:
        return object()

    def math_metric(self, **kwargs: object) -> object:
        del kwargs
        self.metric_calls += 1
        return object()

    def _run_math_verify(self, verifier: object, expected: str, generated: str) -> object:
        del verifier, expected, generated
        self.verify_calls += 1
        return self.raw_result


class _Judge(JudgeExecutor):
    name = "fake-judge"
    invocation_mode = "fake"

    def __init__(
        self,
        *,
        verdict: Verdict | None = Verdict.A,
        task_id: str = "task",
        exit_code: int | None = 0,
        interruption: bool = False,
    ) -> None:
        self.verdict = verdict
        self.task_id = task_id
        self.exit_code = exit_code
        self.interruption = interruption
        self.requests: list[JudgeRequest] = []

    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        del environment
        return JudgePreflightResult(
            judge_executor=self.name,
            ok=True,
            version="fake-1",
            auth_mode="local",
            details=("ready",),
        )

    def judge(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        (request.executor_dir / "stdout.log").write_text("judge output", encoding="utf-8")
        if self.interruption:
            raise KeyboardInterrupt("interrupt sentinel")
        return JudgeResult(
            task_id=self.task_id,
            trial_index=request.trial_index,
            judge_executor=self.name,
            verdict=self.verdict,
            executor_version="fake-1",
            invocation_mode=self.invocation_mode,
            auth_mode="local",
            started_at="s",
            finished_at="f",
            exit_code=self.exit_code,
            stdout_path=request.executor_dir / "stdout.log",
            stderr_path=request.executor_dir / "stderr.log",
        )


class _KeywordOnlyPreflight:
    name = "keyword"

    def preflight(self, *, environment: Mapping[str, str]) -> JudgePreflightResult:
        return JudgePreflightResult(self.name, True, details=(str(environment),))


class _NoArgumentPreflight:
    name = "no-argument"

    def preflight(self) -> JudgePreflightResult:
        return JudgePreflightResult(self.name, True)


class BenchmarkPreparationCoverageTests(unittest.TestCase):
    def test_aime_preparation_and_empty_dataset_failures_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = AIME26Benchmark(
                root=root, dataset_path=root / "aime.jsonl", prepare_script=root / "prepare.py"
            )
            self.assertFalse(benchmark.is_prepared())
            with self.assertRaisesRegex(RuntimeError, "prepare script not found"):
                benchmark.prepare()

            benchmark.prepare_script.touch()
            with patch(
                "eval_harness.benchmarks.aime26.subprocess.run",
                return_value=subprocess.CompletedProcess([], 3),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()
            with patch(
                "eval_harness.benchmarks.aime26.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()

            benchmark.dataset_path.write_text("\n", encoding="utf-8")
            self.assertTrue(benchmark.is_prepared())
            benchmark.prepare()
            with self.assertRaises(ValueError):
                benchmark.load_tasks(0)
            with self.assertRaisesRegex(RuntimeError, "no AIME26 tasks"):
                benchmark.load_tasks(1)

    def test_bigcodebench_preparation_and_row_validation_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = BigCodeBenchBenchmark(
                root=root,
                dataset_path=root / "bigcode.jsonl",
                prepare_script=root / "prepare.py",
            )
            self.assertFalse(benchmark.is_prepared())
            with self.assertRaisesRegex(RuntimeError, "prepare script not found"):
                benchmark.prepare()

            benchmark.prepare_script.touch()
            with patch(
                "eval_harness.benchmarks.bigcodebench.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()
            with patch(
                "eval_harness.benchmarks.bigcodebench.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()

            benchmark.dataset_path.unlink(missing_ok=True)

            def prepare_dataset(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
                del args, kwargs
                benchmark.dataset_path.write_text(
                    json.dumps(
                        {
                            "question": "q",
                            "verifier_metadata": {
                                "task_id": "id",
                                "test": "assert True",
                                "entry_point": "solve",
                                "code_prompt": "def solve():",
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess([], 0)

            with patch("eval_harness.benchmarks.bigcodebench.subprocess.run", side_effect=prepare_dataset):
                benchmark.prepare()
            self.assertTrue(benchmark.is_prepared())
            benchmark.prepare()
            loaded = benchmark.load_tasks(1)[0]
            self.assertEqual(benchmark.materialize(loaded, root / "workspace"), [])

            benchmark.dataset_path.write_text(
                "\n" + json.dumps({"question": "q", "verifier_metadata": []}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                benchmark.load_tasks(0)
            with self.assertRaisesRegex(RuntimeError, "missing verifier_metadata"):
                benchmark.load_tasks(1)

            benchmark.dataset_path.write_text(
                json.dumps({"question": "q", "verifier_metadata": {"task_id": "id"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing test"):
                benchmark.load_tasks(1)
            benchmark.dataset_path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no BigCodeBench tasks"):
                benchmark.load_tasks(1)

    def test_gdpval_parsing_preparation_and_prompt_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = GDPvalBenchmark(
                root=root, dataset_path=root / "gdpval.jsonl", prepare_script=root / "prepare.py"
            )
            self.assertFalse(benchmark.is_prepared())
            with self.assertRaisesRegex(RuntimeError, "prepare script not found"):
                benchmark.prepare()
            benchmark.prepare_script.touch()
            with patch(
                "eval_harness.benchmarks.gdpval.subprocess.run",
                return_value=subprocess.CompletedProcess([], 2),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()
            with patch(
                "eval_harness.benchmarks.gdpval.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()

            benchmark.dataset_path.write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark.load_tasks(0)
            with self.assertRaisesRegex(RuntimeError, "no GDPval tasks"):
                benchmark.load_tasks(1)

            self.assertEqual(gdpval_benchmark._parse_sequence(None), ())
            self.assertEqual(gdpval_benchmark._parse_sequence("not-json"), ("not-json",))
            self.assertEqual(gdpval_benchmark._parse_sequence('["a", 2]'), ("a", "2"))
            self.assertEqual(gdpval_benchmark._parse_sequence(7), ())
            self.assertTrue(gdpval_benchmark._is_inside(root, root / "child"))
            self.assertFalse(gdpval_benchmark._is_inside(root, root.parent / "outside"))

            empty_workspace = root / "empty-workspace"
            empty_workspace.mkdir()
            self.assertEqual(gdpval_benchmark._reference_listing(empty_workspace), "None")
            reference = empty_workspace / "reference_files"
            reference.mkdir()
            self.assertEqual(gdpval_benchmark._reference_listing(empty_workspace), "None")
            (reference / "nested").mkdir()
            (reference / "nested" / "input.txt").write_text("input", encoding="utf-8")
            self.assertEqual(
                gdpval_benchmark._reference_listing(empty_workspace), "- reference_files/nested/input.txt"
            )

            benchmark.dataset_path.write_text(
                json.dumps(
                    {
                        "task_id": "gdp/task",
                        "prompt": "Make a report",
                        "reference_files": '["reference_files/input.txt"]',
                        "reference_file_urls": '["file:///tmp/input.txt"]',
                        "sector": None,
                        "occupation": "analyst",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task = benchmark.load_tasks(1)[0]
            prompt_workspace = root / "prompt-workspace"
            prompt_workspace.mkdir()
            execution_task = benchmark.execution_task(task, prompt_workspace, network_policy="disabled")
            self.assertEqual(execution_task.task_id, "gdp/task")
            self.assertIn("Network policy for model-generated tools: disabled", execution_task.prompt)
            self.assertIn("Task:\nMake a report", execution_task.prompt)
            self.assertIn("./deliverables/", execution_task.prompt)

    def test_gdpval_materialization_checks_counts_paths_and_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = GDPvalBenchmark(
                root=root, dataset_path=root / "tasks.jsonl", prepare_script=root / "prepare.py"
            )
            no_refs = BenchmarkTask(TaskSpec("none", "prompt"), materialization={})
            self.assertEqual(benchmark.materialize(no_refs, root / "no-refs"), [])
            mismatch = BenchmarkTask(
                TaskSpec("mismatch", "prompt"),
                materialization={"reference_files": ("a.txt",), "reference_file_urls": ()},
            )
            with self.assertRaisesRegex(RuntimeError, "count mismatch"):
                benchmark.materialize(mismatch, root / "mismatch")

            task = BenchmarkTask(
                TaskSpec("materialize", "prompt"),
                materialization={"reference_files": ("reference_files/a.txt",), "reference_file_urls": ("url",)},
            )
            with patch(
                "responses_api_agents.stirrup_agent.tasks.gdpval._download_reference_files",
                return_value=[],
            ):
                with self.assertRaisesRegex(RuntimeError, "materialized 0/1"):
                    benchmark.materialize(task, root / "wrong-count")
            with patch(
                "responses_api_agents.stirrup_agent.tasks.gdpval._download_reference_files",
                return_value=["../escape.txt"],
            ):
                with self.assertRaisesRegex(RuntimeError, "unsafe or missing"):
                    benchmark.materialize(task, root / "escape")

            workspace = root / "success"

            def download(files: list[str], urls: list[str], destination: Path) -> list[str]:
                self.assertEqual(files, ["reference_files/a.txt"])
                self.assertEqual(urls, ["url"])
                target = destination / files[0]
                target.parent.mkdir(parents=True)
                target.write_text("reference", encoding="utf-8")
                return files

            with patch(
                "responses_api_agents.stirrup_agent.tasks.gdpval._download_reference_files",
                side_effect=download,
            ):
                self.assertEqual(benchmark.materialize(task, workspace), ["reference_files/a.txt"])
            self.assertEqual((workspace / "reference_files" / "a.txt").read_text(encoding="utf-8"), "reference")


class NativeEvaluatorCoverageTests(unittest.TestCase):
    def test_aime_native_verifier_handles_empty_and_malformed_native_results(self) -> None:
        empty_helper = _MathHelper("", (1.0, "ignored"))
        with patch("eval_harness.evaluators.aime26.importlib.import_module", return_value=empty_helper):
            self.assertEqual(aime26_evaluator._native_math_evaluate("42", "no answer"), (0.0, None))
        self.assertEqual(empty_helper.metric_calls, 0)

        for raw_result, expected_message in (
            ((1.0,), "invalid result"),
            ((True, "42"), "non-numeric score"),
            ((1.0, 42), "non-string extracted answer"),
        ):
            helper = _MathHelper("42", raw_result)
            with self.subTest(raw_result=raw_result):
                with patch("eval_harness.evaluators.aime26.importlib.import_module", return_value=helper):
                    with self.assertRaisesRegex(TypeError, expected_message):
                        aime26_evaluator._native_math_evaluate("42", "\\boxed{42}")

        helper = _MathHelper("42", (0.5, "42"))
        with patch("eval_harness.evaluators.aime26.importlib.import_module", return_value=helper):
            self.assertEqual(aime26_evaluator._native_math_evaluate("42", "\\boxed{42}"), (0.5, "42"))
        self.assertEqual(helper.metric_calls, 1)
        self.assertEqual(helper.verify_calls, 1)

    def test_aime_preflight_reports_dependency_and_helper_failures(self) -> None:
        with patch(
            "eval_harness.evaluators.aime26.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("math-verify"),
        ):
            missing, detail, version = aime26_evaluator._math_verify_preflight()
        self.assertFalse(missing)
        self.assertIsNone(version)
        self.assertIn("not installed", detail)

        with patch(
            "eval_harness.evaluators.aime26.importlib.metadata.version",
            side_effect=RuntimeError("metadata unavailable"),
        ):
            available, detail, version = aime26_evaluator._math_verify_preflight()
        self.assertFalse(available)
        self.assertIsNone(version)
        self.assertIn("metadata unavailable", detail)

        with patch("eval_harness.evaluators.aime26.importlib.metadata.version", return_value="0.7.0"):
            available, detail, version = aime26_evaluator._math_verify_preflight()
        self.assertFalse(available)
        self.assertEqual(version, "0.7.0")
        self.assertIn("found math-verify==0.7.0", detail)

        with (
            patch("eval_harness.evaluators.aime26.importlib.metadata.version", return_value="0.8.0"),
            patch(
                "eval_harness.evaluators.aime26.importlib.import_module",
                side_effect=ImportError("helper unavailable"),
            ),
        ):
            available, detail, version = aime26_evaluator._math_verify_preflight()
        self.assertFalse(available)
        self.assertEqual(version, "0.8.0")
        self.assertIn("helper cannot import", detail)

        with (
            patch("eval_harness.evaluators.aime26.importlib.metadata.version", return_value="0.8.0"),
            patch("eval_harness.evaluators.aime26.importlib.import_module", return_value=object()),
        ):
            available, detail, version = aime26_evaluator._math_verify_preflight()
        self.assertTrue(available)
        self.assertEqual(version, "0.8.0")
        self.assertIn("native math verifier helper", detail)

    def test_aime_evaluate_rejects_missing_metadata_and_scores_empty_output(self) -> None:
        evaluator = AIME26Evaluator()
        with patch.object(aime26_evaluator, "_math_verify_preflight", return_value=(True, "ready", "0.8.0")):
            self.assertTrue(evaluator.preflight().ok)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _execution(root, output_text="")
            with self.assertRaisesRegex(ValueError, "expected_answer"):
                evaluator.evaluate(_single_request(root, result))

            evaluation = evaluator.evaluate(_single_request(root, result, metadata={"expected_answer": "42"}))
            self.assertEqual(evaluation.metrics, {"accuracy": 0.0})
            self.assertEqual(evaluation.details["reason"], "empty executor output")

    def test_bigcodebench_native_runner_handles_no_code_and_process_failures(self) -> None:
        metadata = {"test": "assert True", "entry_point": "solve", "code_prompt": "def solve():"}
        with tempfile.TemporaryDirectory() as tmp:
            resource = Path(tmp)
            (resource / "bcb_runner.py").write_text("runner", encoding="utf-8")
            with patch(
                "resources_servers.bigcodebench.code_extraction.preprocess_code_completion",
                return_value="",
            ):
                no_code = bigcodebench_evaluator._native_bigcodebench_evaluate(
                    "plain text", metadata, resource_dir=resource, bcb_python=resource / "python"
                )
            self.assertEqual(no_code["status"], "no_code_block")
            self.assertEqual(no_code["reward"], 0.0)

            for process_error, status in (
                (subprocess.TimeoutExpired(["python"], 1), "timeout"),
                (OSError("runner unavailable"), "error"),
            ):
                with self.subTest(status=status):
                    with (
                        patch(
                            "resources_servers.bigcodebench.code_extraction.preprocess_code_completion",
                            return_value="return 1",
                        ),
                        patch("eval_harness.evaluators.bigcodebench.subprocess.run", side_effect=process_error),
                    ):
                        result = bigcodebench_evaluator._native_bigcodebench_evaluate(
                            "```python\nreturn 1\n```",
                            metadata,
                            resource_dir=resource,
                            bcb_python=resource / "python",
                        )
                    self.assertEqual(result["status"], status)
                    self.assertEqual(result["reward"], 0.0)

            with (
                patch(
                    "resources_servers.bigcodebench.code_extraction.preprocess_code_completion",
                    return_value="return 1",
                ),
                patch(
                    "eval_harness.evaluators.bigcodebench.subprocess.run",
                    return_value=subprocess.CompletedProcess(["python"], 4, stdout="not json", stderr="bad"),
                ),
            ):
                malformed = bigcodebench_evaluator._native_bigcodebench_evaluate(
                    "```python\nreturn 1\n```",
                    metadata,
                    resource_dir=resource,
                    bcb_python=resource / "python",
                )
            self.assertEqual(malformed["status"], "error")
            self.assertEqual(cast(dict[str, object], malformed["details"])["returncode"], 4)

    def test_bigcodebench_evaluator_rejects_boundary_states_and_bad_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grader = root / "grader"
            grader.mkdir()
            evaluator = BigCodeBenchEvaluator(resource_dir=grader)
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                evaluator.evaluate(
                    _single_request(
                        root,
                        _execution(root, output_text="code"),
                        metadata={"test": "t", "entry_point": "e", "code_prompt": "p"},
                    )
                )

            (grader / "bcb_runner.py").write_text("runner", encoding="utf-8")
            evaluator._bcb_python = grader / "python"
            empty = evaluator.evaluate(
                _single_request(
                    root,
                    _execution(root, output_text=""),
                    metadata={"test": "t", "entry_point": "e", "code_prompt": "p"},
                )
            )
            self.assertEqual(empty.metrics, {"pass_rate": 0.0})
            self.assertEqual(empty.details["status"], "empty_output")

            with self.assertRaisesRegex(ValueError, "requires test"):
                evaluator.evaluate(_single_request(root, _execution(root, output_text="code"), metadata={}))

            with patch.object(
                bigcodebench_evaluator,
                "_native_bigcodebench_evaluate",
                return_value={"reward": "unsafe", "status": "error"},
            ):
                with self.assertRaisesRegex(TypeError, "non-numeric reward"):
                    evaluator.evaluate(
                        _single_request(
                            root,
                            _execution(root, output_text="code"),
                            metadata={"test": "t", "entry_point": "e", "code_prompt": "p"},
                        )
                    )

    def test_bigcodebench_preflight_reports_path_and_venv_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grader = root / "grader"
            grader.mkdir()
            evaluator = BigCodeBenchEvaluator(resource_dir=grader)
            overlap = evaluator.preflight(grader / "run")
            self.assertFalse(overlap.ok)
            self.assertIn("separate", overlap.details[0])
            missing = evaluator.preflight(root / "run")
            self.assertFalse(missing.ok)
            self.assertIn("runner is missing", missing.details[0])

            (grader / "bcb_runner.py").write_text("runner", encoding="utf-8")
            with patch(
                "resources_servers.bigcodebench.setup_bcb_venv.ensure_bcb_venv",
                side_effect=RuntimeError("venv unavailable"),
            ):
                failed = evaluator.preflight(root / "run")
            self.assertFalse(failed.ok)
            self.assertIn("venv unavailable", failed.details[0])


class GDPvalEvaluatorSafetyCoverageTests(unittest.TestCase):
    def test_gdpval_path_helpers_and_source_tree_reject_unsafe_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(Path, "absolute", side_effect=OSError("absolute unavailable")):
                self.assertEqual(gdpval_evaluator._absolute(Path("relative")), Path(os.path.abspath("relative")))
            with patch.object(Path, "resolve", side_effect=OSError("resolve unavailable")):
                self.assertEqual(gdpval_evaluator._resolved(root / "path"), Path(os.path.abspath(root / "path")))
            with patch.object(Path, "is_symlink", side_effect=OSError("stat unavailable")):
                with self.assertRaisesRegex(ValueError, "could not inspect"):
                    gdpval_evaluator._assert_no_symlink_components(root / "path")

            missing = root / "missing"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                gdpval_evaluator._assert_no_symlink_components(missing, allow_missing_final=False)
            file_path = root / "file"
            file_path.write_text("file", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                gdpval_evaluator._assert_directory(file_path, label="source")
            link = root / "link"
            link.symlink_to(root)
            with self.assertRaisesRegex(ValueError, "symlink not allowed"):
                gdpval_evaluator._assert_no_symlink_components(link / "nested")

            source = root / "source"
            (source / "nested").mkdir(parents=True)
            (source / "nested" / "input.txt").write_text("input", encoding="utf-8")
            self.assertEqual(
                [relative.as_posix() for _, relative in gdpval_evaluator._iter_source_entries(source)],
                ["nested", "nested/input.txt"],
            )
            fifo = source / "pipe"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                list(gdpval_evaluator._iter_source_entries(source))

            original_resolve = Path.resolve

            def escape_resolve(path: Path, strict: bool = False) -> Path:
                if path == source / "nested":
                    return root / "outside"
                return original_resolve(path, strict=strict)

            with patch.object(Path, "resolve", new=escape_resolve):
                with self.assertRaisesRegex(ValueError, "escapes its root"):
                    list(gdpval_evaluator._iter_source_entries(source))

    def test_gdpval_copy_and_publish_fail_closed_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "answer.txt").write_text("answer", encoding="utf-8")
            destination = root / "destination"
            destination.mkdir()
            (destination / "marker").write_text("marker", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                gdpval_evaluator._copy_tree_bytes(source, destination, allow_existing_destination=True)

            target_link_destination = root / "link-target"
            original_mkdir = Path.mkdir

            def create_racing_link(
                path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
            ) -> None:
                original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
                if path == target_link_destination:
                    (path / "answer.txt").symlink_to(root / "outside")

            with patch.object(Path, "mkdir", new=create_racing_link):
                with self.assertRaisesRegex(ValueError, "symlink not allowed"):
                    gdpval_evaluator._copy_tree_bytes(source, target_link_destination)

            nested_source = root / "nested-source"
            (nested_source / "child").mkdir(parents=True)
            nested_destination = root / "nested-destination"
            original_nested_mkdir = Path.mkdir

            def create_existing_child(
                path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
            ) -> None:
                if path == nested_destination / "child":
                    os.mkdir(path)
                original_nested_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

            with patch.object(Path, "mkdir", new=create_existing_child):
                with self.assertRaises(FileExistsError):
                    gdpval_evaluator._copy_tree_bytes(nested_source, nested_destination)
            copied_nested = gdpval_evaluator._copy_tree_bytes(nested_source, root / "nested-success")
            self.assertEqual(copied_nested, [])

            reference_link = root / "reference-link"
            reference_link.mkdir()
            source_reference = root / "source-reference"
            source_reference.symlink_to(reference_link, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked GDPval reference"):
                gdpval_evaluator._copy_reference_tree(source_reference, root / "copied-reference")
            missing_reference_destination = root / "missing-reference"
            gdpval_evaluator._copy_reference_tree(root / "missing-file", missing_reference_destination)
            self.assertFalse(missing_reference_destination.exists())
            (root / "missing-file").write_text("file", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                gdpval_evaluator._copy_reference_tree(root / "missing-file", root / "copied-reference")

            deliverables = root / "deliverables"
            deliverables.mkdir()
            workspace = root / "workspace"
            workspace.mkdir()
            with self.assertRaisesRegex(ValueError, "separate"):
                gdpval_evaluator._publish(
                    source_deliverables=deliverables,
                    source_workspace=workspace,
                    destination=workspace / "handoff",
                    result_executor="fake",
                    result_status="completed",
                )

            symlinked_reference_workspace = root / "symlinked-reference-workspace"
            symlinked_reference_workspace.mkdir()
            (symlinked_reference_workspace / "reference_files").symlink_to(reference_link, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked GDPval reference"):
                gdpval_evaluator._publish(
                    source_deliverables=deliverables,
                    source_workspace=symlinked_reference_workspace,
                    destination=root / "handoff",
                    result_executor="fake",
                    result_status="completed",
                )

            with patch("eval_harness.evaluators.gdpval.os.open", side_effect=OSError("directory fsync unavailable")):
                gdpval_evaluator._fsync_directory(root)

    def test_gdpval_publish_cleans_staging_after_atomic_rename_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deliverables = root / "deliverables"
            deliverables.mkdir()
            (deliverables / "answer.txt").write_text("answer", encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            destination = root / "handoff"
            with patch("eval_harness.evaluators.gdpval.os.replace", side_effect=OSError("rename failed")):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    gdpval_evaluator._publish(
                        source_deliverables=deliverables,
                        source_workspace=workspace,
                        destination=destination,
                        result_executor="fake",
                        result_status="completed",
                    )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".handoff.staging-*")), [])

            retained_empty = root / "retained-empty"
            original_rmdir = Path.rmdir

            def refuse_cleanup(path: Path) -> None:
                if path == retained_empty:
                    raise OSError("rmdir race")
                original_rmdir(path)

            with (
                patch("eval_harness.evaluators.gdpval.os.replace", side_effect=OSError("rename failed")),
                patch.object(Path, "rmdir", new=refuse_cleanup),
            ):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    gdpval_evaluator._publish(
                        source_deliverables=deliverables,
                        source_workspace=workspace,
                        destination=retained_empty,
                        result_executor="fake",
                        result_status="completed",
                    )
            self.assertTrue(retained_empty.is_dir())
            self.assertEqual(list(retained_empty.iterdir()), [])

    def test_gdpval_evaluator_handles_missing_request_destination(self) -> None:
        evaluator = GDPvalExternalEvaluator()
        with self.assertRaisesRegex(ValueError, "request.artifact_dir"):
            evaluator.evaluate(
                _single_request(
                    Path(tempfile.gettempdir()),
                    _execution(Path(tempfile.gettempdir())),
                    metadata={},
                    artifact_dir=None,
                )
            )


class PairwiseAndExactCoverageTests(unittest.TestCase):
    def test_pairwise_helpers_cover_environment_signatures_and_path_guards(self) -> None:
        self.assertEqual(pairwise_evaluator.sanitize_environment(None), {})
        self.assertEqual(
            pairwise_evaluator.sanitize_environment({"PATH": "/bin", "secret": "redacted"}),
            {"PATH": "/bin"},
        )
        keyword = pairwise_evaluator._call_preflight(cast(JudgeExecutor, _KeywordOnlyPreflight()), {"PATH": "/bin"})
        self.assertTrue(keyword.ok)
        no_argument = pairwise_evaluator._call_preflight(cast(JudgeExecutor, _NoArgumentPreflight()), {})
        self.assertTrue(no_argument.ok)
        with patch("eval_harness.evaluators.pairwise.inspect.signature", side_effect=TypeError("opaque")):
            fallback = pairwise_evaluator._call_preflight(cast(JudgeExecutor, _Judge()), {})
        self.assertTrue(fallback.ok)
        self.assertIsNone(pairwise_evaluator._jsonable_verdict(None))
        self.assertEqual(pairwise_evaluator._jsonable_verdict(Verdict.TIE), "TIE")
        self.assertEqual(pairwise_evaluator._jsonable_verdict(cast(Verdict, "unexpected")), "unexpected")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "missing"):
                pairwise_evaluator._assert_candidate_dir(None, candidate_id="a")
            with self.assertRaisesRegex(ValueError, "not found"):
                pairwise_evaluator._assert_candidate_dir(root / "missing", candidate_id="a")
            file_path = root / "file"
            file_path.write_text("file", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not found"):
                pairwise_evaluator._assert_candidate_dir(file_path, candidate_id="a")
            link_target = root / "link-target"
            link_target.mkdir()
            candidate_link = root / "link"
            candidate_link.symlink_to(link_target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                pairwise_evaluator._assert_candidate_dir(candidate_link, candidate_id="a")

            output = root / "output"
            with self.assertRaisesRegex(ValueError, "requires request"):
                pairwise_evaluator._assert_output_root(None)
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                pairwise_evaluator._assert_output_root(output)
            output_link = root / "output-link"
            output_link.symlink_to(link_target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                pairwise_evaluator._assert_output_root(output_link)

            parent_race = root / "parent-race"
            parent_race.mkdir()
            parent_race_checks = 0
            original_parent_is_symlink = Path.is_symlink

            def parent_becomes_symlink(path: Path) -> bool:
                nonlocal parent_race_checks
                if path == parent_race:
                    parent_race_checks += 1
                    return parent_race_checks >= 2
                return original_parent_is_symlink(path)

            with patch.object(Path, "is_symlink", new=parent_becomes_symlink):
                with self.assertRaisesRegex(ValueError, "output parent"):
                    pairwise_evaluator._assert_output_root(parent_race / "new-output")

            with patch.object(Path, "absolute", side_effect=OSError("absolute unavailable")):
                with self.assertRaisesRegex(ValueError, "not found"):
                    pairwise_evaluator._assert_candidate_dir(root / "still-missing", candidate_id="a")
            with patch.object(Path, "absolute", side_effect=OSError("absolute unavailable")):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    pairwise_evaluator._assert_output_root(output)

            with self.assertRaisesRegex(ValueError, "positive"):
                PairwiseJudgeEvaluator(_Judge(), trials=0)
            with self.assertRaisesRegex(ValueError, "positive"):
                PairwiseJudgeEvaluator(_Judge(), timeout_seconds=0)
            evaluator = PairwiseJudgeEvaluator(_Judge())
            self.assertIs(evaluator.judge, evaluator.judge_executor)
            with self.assertRaisesRegex(ValueError, "exactly two"):
                evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 1, root / "out"))
            with self.assertRaisesRegex(ValueError, "artifact destination"):
                evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 2))

    def test_pairwise_rejects_overlap_and_persists_fail_closed_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "first"
            second_dir = root / "second"
            for directory, text in ((first_dir, "a"), (second_dir, "b")):
                (directory / "reference_files").mkdir(parents=True)
                (directory / "reference_files" / "ref.txt").write_text("same", encoding="utf-8")
                (directory / "answer.txt").write_text(text, encoding="utf-8")
            first = EvaluationCandidate(
                "first", _execution(root, workspace=first_dir, deliverables=first_dir), first_dir
            )
            second = EvaluationCandidate(
                "second", _execution(root, workspace=second_dir, deliverables=second_dir), second_dir
            )
            request = EvaluationRequest("task", "prompt", candidates=(first, second), artifact_dir=first_dir / "out")
            judge = _Judge()
            evaluator = PairwiseJudgeEvaluator(judge, trials=1)
            evaluator.preflight(root / "run")
            with self.assertRaisesRegex(ValueError, "separate"):
                evaluator.evaluate(request)

            race_parent = root / "race-parent"
            race_parent.mkdir()
            race_output = race_parent / "output"
            original_is_symlink = Path.is_symlink
            parent_checks = 0

            def report_parent_race(path: Path) -> bool:
                nonlocal parent_checks
                if path == race_parent:
                    parent_checks += 1
                    return parent_checks >= 3
                return original_is_symlink(path)

            race_evaluator = PairwiseJudgeEvaluator(_Judge(), trials=1)
            race_evaluator.preflight(root / "run")
            with patch.object(Path, "is_symlink", new=report_parent_race):
                with self.assertRaisesRegex(ValueError, "symlink not allowed"):
                    race_evaluator.evaluate(
                        EvaluationRequest("task", "prompt", candidates=(first, second), artifact_dir=race_output)
                    )

            for judge_case, message in (
                (_Judge(task_id="other"), "failed closed"),
                (_Judge(verdict=None), "failed closed"),
            ):
                output = root / ("out-" + str(len(list(root.iterdir()))))
                case_evaluator = PairwiseJudgeEvaluator(judge_case, trials=1)
                case_evaluator.preflight(root / "run")
                with self.assertRaisesRegex(RuntimeError, message):
                    case_evaluator.evaluate(
                        EvaluationRequest("task", "prompt", candidates=(first, second), artifact_dir=output)
                    )
                metadata = output / "judge" / "tasks" / "task" / "trial_0" / "executor" / "metadata.json"
                self.assertIn("error", json.loads(metadata.read_text(encoding="utf-8")))

            interrupted = _Judge(interruption=True)
            interrupted_evaluator = PairwiseJudgeEvaluator(interrupted, trials=1)
            interrupted_evaluator.preflight(root / "run")
            interrupted_output = root / "interrupted"
            with self.assertRaises(KeyboardInterrupt):
                interrupted_evaluator.evaluate(
                    EvaluationRequest("task", "prompt", candidates=(first, second), artifact_dir=interrupted_output)
                )
            interruption_metadata = (
                interrupted_output / "judge" / "tasks" / "task" / "trial_0" / "executor" / "metadata.json"
            )
            self.assertEqual(
                json.loads(interruption_metadata.read_text(encoding="utf-8"))["error_type"], "KeyboardInterrupt"
            )

    def test_pairwise_metadata_writer_preserves_cleanup_on_serializer_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with (
                patch(
                    "eval_harness.evaluators.pairwise.write_trial_metadata", side_effect=RuntimeError("write failed")
                ),
                patch.object(Path, "unlink", side_effect=OSError("unlink failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    pairwise_evaluator._write_trial_metadata_preserving(directory, {"task_id": "task"})

    def test_exact_evaluator_failures_remain_deterministic(self) -> None:
        evaluator = ExactMatchEvaluator()
        preflight = evaluator.preflight(Path("run"))
        self.assertTrue(preflight.ok)
        self.assertIn("trimmed", preflight.details[0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _execution(root, status=ExecutionStatus.FAILED, output_text="answer")
            evaluation = evaluator.evaluate(_single_request(root, failed, metadata={"expected_answer": "answer"}))
            self.assertEqual(evaluation.status, EvaluationStatus.COMPLETED)
            self.assertEqual(evaluation.metrics, {"exact_match": 0.0})
            self.assertFalse(evaluation.outcomes["matched"])
            with self.assertRaisesRegex(ValueError, "expected_answer"):
                evaluator.evaluate(_single_request(root, failed))


if __name__ == "__main__":
    unittest.main()
