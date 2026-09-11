# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic exact-match evaluator."""

from __future__ import annotations

from pathlib import Path

from gdpval_harness.evaluators.base import (
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
    EvaluatorType,
    require_one_candidate,
)


class ExactMatchEvaluator(Evaluator):
    """Compare trimmed candidate text with ``metadata['expected_answer']``.

    Exact matching strips leading and trailing whitespace from both strings and
    otherwise compares them byte-for-byte as Python text.  It does not parse
    markdown, normalize case, or call a model.  The metric is ``exact_match``
    and is either ``1.0`` or ``0.0``.  A failed execution is a completed
    zero-score comparison when an execution result is available, so no fallback
    evaluator can silently change the result.
    """

    name = "exact-match"
    evaluator_type = EvaluatorType.DETERMINISTIC_EXACT

    def validate_plan(self, plan: EvaluationPlan) -> None:
        if plan.candidate_count != 1:
            raise ValueError("exact-match evaluator requires exactly one candidate")
        if "expected_answer" not in plan.metadata:
            raise ValueError("exact-match evaluator requires metadata['expected_answer']")

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        return EvaluatorPreflightResult(
            name=self.name,
            evaluator_type=self.evaluator_type,
            ok=True,
            version="1",
            details=("trimmed exact text comparison is ready",),
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        candidate = require_one_candidate(request)
        if "expected_answer" not in request.metadata:
            raise ValueError("exact-match evaluator requires metadata['expected_answer']")
        expected = str(request.metadata["expected_answer"]).strip()
        actual = (candidate.execution.output_text or "").strip()
        matched = candidate.execution.status.value in {"completed", "no_deliverable"} and actual == expected
        return EvaluationResult(
            task_id=request.task_id,
            status=EvaluationStatus.COMPLETED,
            metrics={"exact_match": 1.0 if matched else 0.0},
            outcomes={
                "candidate_id": candidate.candidate_id,
                "matched": matched,
            },
            details={
                "semantics": "trimmed exact text comparison",
                "expected_answer": expected,
                "execution_status": candidate.execution.status.value,
            },
        )


__all__ = ["ExactMatchEvaluator"]
