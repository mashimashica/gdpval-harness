# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval_harness.benchmarks.aime26 import AIME26Benchmark
from eval_harness.capabilities import ExecutorOutput
from eval_harness.evaluators.aime26 import AIME26Evaluator
from eval_harness.evaluators.base import EvaluationCandidate, EvaluationRequest, EvaluationStatus, EvaluatorType
from eval_harness.executors.base import ExecutionResult, ExecutionStatus
from eval_harness.failures import Failure, FailureImpact, FailureKind


class AIME26BenchmarkTests(unittest.TestCase):
    def benchmark(self, root: Path) -> AIME26Benchmark:
        return AIME26Benchmark(
            root=root,
            dataset_path=root / "aime26.jsonl",
            prepare_script=root / "prepare.py",
        )

    def execution_result(
        self,
        root: Path,
        *,
        output_text: str | None,
        status: ExecutionStatus = ExecutionStatus.NO_DELIVERABLE,
    ) -> ExecutionResult:
        effective_output = (
            output_text if status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE} else None
        )
        return ExecutionResult(
            task_id="aime26-01",
            executor="codex",
            executor_version="test",
            invocation_mode="codex exec",
            auth_mode="chatgpt-subscription",
            workspace=root / "workspace",
            deliverables_dir=root / "workspace" / "deliverables",
            status=status,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=0 if status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE} else 1,
            available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}) if effective_output is not None else frozenset(),
            failure=None
            if effective_output is not None
            else Failure(FailureKind.PROCESS, "test_failure", FailureImpact.RUN),
            output_text=effective_output,
        )

    def request(self, root: Path, result: ExecutionResult, expected: str = "42") -> EvaluationRequest:
        return EvaluationRequest(
            task_id="aime26-01",
            task_prompt="Solve q",
            metadata={"expected_answer": expected},
            candidates=(EvaluationCandidate("policy", result),),
        )

    def ready_evaluator(self) -> AIME26Evaluator:
        evaluator = AIME26Evaluator()
        with patch(
            "eval_harness.evaluators.aime26._math_verify_preflight",
            return_value=(True, "native verifier ready", "0.8.0"),
        ):
            self.assertTrue(evaluator.preflight(Path("/tmp/eval-run")).ok)
        return evaluator

    def test_load_tasks_preserves_native_math_prompt_and_expected_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = self.benchmark(root)
            benchmark.dataset_path.write_text(
                json.dumps({"question": "What is 6*7?", "expected_answer": "42"}) + "\n",
                encoding="utf-8",
            )

            task = benchmark.load_tasks(1)[0]

            self.assertEqual(task.execution.task_id, "aime26-01")
            self.assertEqual(
                task.execution.prompt,
                "Solve the following math problem. Make sure to put the answer (and only answer) "
                "inside \\boxed{}.\n\nWhat is 6*7?",
            )
            self.assertEqual(task.evaluation["expected_answer"], "42")
            self.assertNotIn("evaluate", type(benchmark).__dict__)
            self.assertEqual(AIME26Evaluator.evaluator_type, EvaluatorType.BENCHMARK_NATIVE)

    def test_preflight_requires_pinned_native_dependency_and_helper(self) -> None:
        evaluator = AIME26Evaluator()
        with patch(
            "eval_harness.evaluators.aime26._math_verify_preflight",
            return_value=(False, "./eval requires math-verify==0.8.0; found math-verify==0.7.0", "0.7.0"),
        ):
            result = evaluator.preflight(Path("/tmp/eval-run"))
        self.assertFalse(result.ok)
        self.assertIn("math-verify==0.8.0", result.details[0])

    def test_evaluate_uses_native_library_verifier_without_llm_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.execution_result(root, output_text="Reasoning... \\boxed{42}")
            evaluator = self.ready_evaluator()

            with patch(
                "eval_harness.evaluators.aime26._native_math_evaluate",
                return_value=(1.0, "42"),
            ) as verifier:
                evaluation = evaluator.evaluate(self.request(root, result))

            verifier.assert_called_once_with("42", "Reasoning... \\boxed{42}")
            self.assertEqual(evaluation.status, EvaluationStatus.COMPLETED)
            self.assertEqual(evaluation.metrics, {"accuracy": 1.0})
            self.assertEqual(evaluation.details["extracted_answer"], "42")
            self.assertFalse(evaluation.details["llm_judge_used"])

    def test_evaluation_requires_successful_dependency_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = AIME26Evaluator()
            result = self.execution_result(root, output_text="\\boxed{42}")
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                evaluator.evaluate(self.request(root, result))

    def test_failed_execution_scores_zero_without_calling_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.execution_result(root, output_text="\\boxed{42}", status=ExecutionStatus.FAILED)
            evaluator = self.ready_evaluator()

            with patch("eval_harness.evaluators.aime26._native_math_evaluate") as verifier:
                evaluation = evaluator.evaluate(self.request(root, result))

            verifier.assert_not_called()
            self.assertEqual(evaluation.metrics, {"accuracy": 0.0})
            self.assertEqual(evaluation.details["execution_status"], "failed")

    def test_materialize_creates_workspace_without_benchmark_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = self.benchmark(root)
            workspace = root / "workspace"
            benchmark.dataset_path.write_text(
                json.dumps({"question": "q", "expected_answer": "1"}) + "\n",
                encoding="utf-8",
            )
            task = benchmark.load_tasks(1)[0]
            self.assertEqual(benchmark.materialize(task, workspace), [])
            self.assertTrue(workspace.is_dir())


if __name__ == "__main__":
    unittest.main()
