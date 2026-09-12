# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AIME26 evaluation through the math-with-judge library verifier."""

from __future__ import annotations

import importlib
import importlib.metadata
from numbers import Real
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


_MATH_VERIFY_DISTRIBUTION = "math-verify"
_MATH_VERIFY_VERSION = "0.8.0"
_TERMINAL_SUCCESS = {"completed", "no_deliverable"}


def _native_math_evaluate(expected_answer: str, generated_answer: str) -> tuple[float, str | None]:
    """Run the existing native verifier without the resource server's LLM fallback."""

    helper = importlib.import_module("resources_servers.math_with_judge.app")
    boxed_answer = helper._extract_last_boxed_answer(generated_answer)
    if boxed_answer is None or not boxed_answer.strip():
        return 0.0, None

    verifier = helper.math_metric(
        gold_extraction_target=(helper.LatexExtractionConfig(),),
        pred_extraction_target=(helper.ExprExtractionConfig(), helper.LatexExtractionConfig()),
    )
    raw_result: object = helper._run_math_verify(verifier, expected_answer, generated_answer)
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise TypeError("native math verifier returned an invalid result")
    score, extracted_answer = raw_result
    if not isinstance(score, Real) or isinstance(score, bool):
        raise TypeError("native math verifier returned a non-numeric score")
    if extracted_answer is not None and not isinstance(extracted_answer, str):
        raise TypeError("native math verifier returned a non-string extracted answer")
    return float(score), extracted_answer


def _math_verify_preflight() -> tuple[bool, str, str | None]:
    """Require the pinned distribution and import the exact resource helper."""

    try:
        version = importlib.metadata.version(_MATH_VERIFY_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return False, "math-verify==0.8.0 is required by ./eval but is not installed in this interpreter", None
    except Exception as exc:  # pragma: no cover - defensive metadata backend failure
        return False, f"could not inspect math-verify in the ./eval interpreter: {exc}", None
    if version != _MATH_VERIFY_VERSION:
        return (
            False,
            f"./eval requires math-verify=={_MATH_VERIFY_VERSION}; found math-verify=={version}",
            version,
        )
    try:
        importlib.import_module("resources_servers.math_with_judge.app")
    except Exception as exc:
        return (
            False,
            "math-verify=="
            f"{_MATH_VERIFY_VERSION} is installed but the native math_with_judge helper cannot import: {exc}",
            version,
        )
    return True, f"math-verify=={_MATH_VERIFY_VERSION} and native math verifier helper are available", version


class AIME26Evaluator(Evaluator):
    name = "aime26-native"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE
    revision = None

    def __init__(self) -> None:
        self._ready = False

    def validate_plan(self, plan: EvaluationPlan) -> None:
        if plan.candidate_count != 1:
            raise ValueError("AIME26 evaluator requires exactly one candidate")
        if "expected_answer" not in plan.metadata:
            raise ValueError("AIME26 evaluator requires metadata['expected_answer']")

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        ok, detail, version = _math_verify_preflight()
        self._ready = ok
        return EvaluatorPreflightResult(
            name=self.name,
            evaluator_type=self.evaluator_type,
            ok=ok,
            version=version,
            revision=self.revision,
            details=(detail,),
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if not self._ready:
            raise RuntimeError("successful AIME26 evaluator preflight is required before evaluate")
        candidate = require_one_candidate(request)
        expected_answer = str(request.metadata.get("expected_answer", ""))
        if "expected_answer" not in request.metadata:
            raise ValueError("AIME26 evaluator requires metadata['expected_answer']")

        output_text = candidate.execution.output_text or ""
        details: dict[str, object] = {
            "expected_answer": expected_answer,
            "extracted_answer": None,
            "execution_status": candidate.execution.status.value,
            "verifier": "math_with_judge/library",
            "llm_judge_used": False,
        }
        if candidate.execution.status.value not in _TERMINAL_SUCCESS:
            return EvaluationResult(
                task_id=request.task_id,
                status=EvaluationStatus.COMPLETED,
                metrics={"accuracy": 0.0},
                details=details,
            )
        if not output_text.strip():
            details["reason"] = "empty executor output"
            return EvaluationResult(
                task_id=request.task_id,
                status=EvaluationStatus.COMPLETED,
                metrics={"accuracy": 0.0},
                details=details,
            )

        reward, extracted_answer = _native_math_evaluate(expected_answer, output_text)
        details["extracted_answer"] = extracted_answer
        return EvaluationResult(
            task_id=request.task_id,
            status=EvaluationStatus.COMPLETED,
            metrics={"accuracy": float(reward)},
            details=details,
        )


__all__ = ["AIME26Evaluator", "_native_math_evaluate"]
