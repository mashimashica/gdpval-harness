# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

from eval_harness.capabilities import ExecutorOutput
from eval_harness.evaluators.aime26 import AIME26Evaluator
from eval_harness.evaluators.base import (
    EvaluationCandidate,
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorType,
    require_one_candidate,
    require_two_candidates,
)
from eval_harness.evaluators.bigcodebench import BigCodeBenchEvaluator
from eval_harness.evaluators.exact import ExactMatchEvaluator
from eval_harness.executors.base import ExecutionResult, ExecutionStatus


def _result(root: Path, task_id: str = "task") -> ExecutionResult:
    workspace = root / "workspace"
    return ExecutionResult(
        task_id=task_id,
        executor="fake",
        executor_version="1",
        invocation_mode="fake",
        auth_mode="local",
        workspace=workspace,
        deliverables_dir=workspace / "deliverables",
        status=ExecutionStatus.COMPLETED,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        exit_code=0,
        available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}),
        failure=None,
        output_text="  answer  ",
    )


class EvaluatorContractTests(unittest.TestCase):
    def test_evaluation_plan_is_immutable_with_shallow_frozen_metadata(self) -> None:
        nested = {"source": "fixture"}
        plan = EvaluationPlan(
            task_id="task",
            task_prompt="prompt",
            metadata={"expected_answer": nested},
            candidate_count=1,
        )
        with self.assertRaises(TypeError):
            # Preserve the invalid mutation at the immutable mapping boundary.
            cast(dict[str, object], plan.metadata)["new_key"] = "value"
        with self.assertRaises(FrozenInstanceError):
            setattr(plan, "task_id", "other")
        # The freeze is intentionally shallow: evaluator metadata values retain
        # their normal object semantics without being copied recursively.
        expected_answer = cast(dict[str, object], plan.metadata["expected_answer"])
        expected_answer["source"] = "updated"
        self.assertEqual(expected_answer, {"source": "updated"})

    def test_exact_and_aime_plans_require_one_candidate_and_expected_answer(self) -> None:
        valid = EvaluationPlan(
            task_id="task",
            task_prompt="prompt",
            metadata={"expected_answer": "42"},
            candidate_count=1,
        )
        ExactMatchEvaluator().validate_plan(valid)
        AIME26Evaluator().validate_plan(valid)

        for evaluator in (ExactMatchEvaluator(), AIME26Evaluator()):
            with self.subTest(evaluator=evaluator.name):
                with self.assertRaisesRegex(ValueError, "exactly one candidate"):
                    evaluator.validate_plan(EvaluationPlan("task", "prompt", {"expected_answer": "42"}, 2))
                with self.assertRaisesRegex(ValueError, "expected_answer"):
                    evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 1))

    def test_bigcodebench_plan_requires_one_candidate_and_grader_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = BigCodeBenchEvaluator(resource_dir=Path(tmp))
            evaluator.validate_plan(
                EvaluationPlan(
                    "task",
                    "prompt",
                    {"test": "assert True", "entry_point": "solve", "code_prompt": "def solve():"},
                    1,
                )
            )
            with self.assertRaisesRegex(ValueError, "exactly one candidate"):
                evaluator.validate_plan(
                    EvaluationPlan(
                        "task",
                        "prompt",
                        {"test": "assert True", "entry_point": "solve", "code_prompt": "def solve():"},
                        2,
                    )
                )
            with self.assertRaisesRegex(ValueError, "entry_point"):
                evaluator.validate_plan(
                    EvaluationPlan("task", "prompt", {"test": "assert True", "code_prompt": "def solve():"}, 1)
                )

    def test_exact_is_trimmed_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _result(root)
            request = EvaluationRequest(
                task_id="task",
                task_prompt="prompt",
                metadata={"expected_answer": "answer"},
                candidates=(EvaluationCandidate("policy", result),),
            )
            evaluation = ExactMatchEvaluator().evaluate(request)
            self.assertEqual(evaluation.status, EvaluationStatus.COMPLETED)
            self.assertEqual(evaluation.metrics, {"exact_match": 1.0})
            self.assertTrue(evaluation.outcomes["matched"])

    def test_cardinality_and_task_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _result(root)
            request = EvaluationRequest(
                task_id="task",
                task_prompt="prompt",
                candidates=(EvaluationCandidate("policy", result),),
            )
            with self.assertRaises(ValueError):
                require_two_candidates(request)
            wrong = EvaluationRequest(
                task_id="other",
                task_prompt="prompt",
                candidates=(EvaluationCandidate("policy", result),),
            )
            with self.assertRaises(ValueError):
                require_one_candidate(wrong)

    def test_outcome_only_pairwise_result_has_no_universal_metric(self) -> None:
        result = EvaluationResult(
            task_id="task",
            status=EvaluationStatus.COMPLETED,
            metrics={},
            outcomes={"candidate_a": {"wins": 1}, "candidate_b": {"wins": 0}},
        )
        self.assertEqual(result.metrics, {})
        self.assertIn("candidate_a", result.outcomes)
        self.assertEqual(EvaluatorType.PAIRWISE.value, "pairwise")


if __name__ == "__main__":
    unittest.main()
