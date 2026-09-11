# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BigCodeBench extraction and executable test evaluation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Mapping

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


_BCB_PYTHON_VERSION = "3.10"
_MAX_AS_LIMIT = 30 * 1024
_MAX_DATA_LIMIT = 30 * 1024
_MAX_STACK_LIMIT = 10
_MIN_TIME_LIMIT = 1.0
_GT_TIME_LIMIT = 20.0
_SUBPROCESS_TIMEOUT = 240.0
_TERMINAL_SUCCESS = {"completed", "no_deliverable"}


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _native_bigcodebench_evaluate(
    output_text: str,
    verifier_metadata: Mapping[str, object],
    *,
    resource_dir: Path,
    bcb_python: Path | None = None,
) -> dict[str, object]:
    """Run the existing extractor and bcb_runner in the dedicated Python 3.10 venv."""

    from resources_servers.bigcodebench.code_extraction import preprocess_code_completion
    from resources_servers.bigcodebench.setup_bcb_venv import ensure_bcb_venv

    extracted = preprocess_code_completion(output_text)
    if not extracted:
        return {
            "reward": 0.0,
            "status": "no_code_block",
            "extracted_model_code": None,
            "details": None,
        }

    code_prompt = str(verifier_metadata["code_prompt"])
    calibrated = code_prompt + "\n    pass\n" + extracted
    grader_python = bcb_python or ensure_bcb_venv(resource_dir / ".bcb_venv", _BCB_PYTHON_VERSION)
    runner_path = resource_dir / "bcb_runner.py"
    payload = json.dumps(
        {
            "code": calibrated,
            "test_code": str(verifier_metadata["test"]),
            "entry_point": str(verifier_metadata["entry_point"]),
            "max_as_limit": _MAX_AS_LIMIT,
            "max_data_limit": _MAX_DATA_LIMIT,
            "max_stack_limit": _MAX_STACK_LIMIT,
            "min_time_limit": _MIN_TIME_LIMIT,
            "gt_time_limit": _GT_TIME_LIMIT,
        }
    )

    try:
        completed = subprocess.run(
            [str(grader_python), str(runner_path)],
            input=payload,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_SUBPROCESS_TIMEOUT,
            cwd=resource_dir,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "reward": 0.0,
            "status": "timeout",
            "extracted_model_code": extracted,
            "details": {"reason": "outer_subprocess_timeout"},
        }
    except OSError as exc:
        return {
            "reward": 0.0,
            "status": "error",
            "extracted_model_code": extracted,
            "details": {"reason": str(exc)},
        }

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "reward": 0.0,
            "status": "error",
            "extracted_model_code": extracted,
            "details": {
                "returncode": completed.returncode,
                "stderr": completed.stderr[:2000],
                "stdout": completed.stdout[:2000],
            },
        }

    status = result.get("status")
    return {
        "reward": 1.0 if status == "pass" else 0.0,
        "status": status,
        "extracted_model_code": extracted,
        "details": result.get("details"),
    }


class BigCodeBenchEvaluator(Evaluator):
    name = "bigcodebench-tests"
    evaluator_type = EvaluatorType.EXECUTABLE_TESTS
    version = "1"
    revision = "v0.1.4"

    def __init__(self, *, resource_dir: Path) -> None:
        self.resource_dir = resource_dir
        self._bcb_python: Path | None = None

    def validate_plan(self, plan: EvaluationPlan) -> None:
        if plan.candidate_count != 1:
            raise ValueError("BigCodeBench evaluator requires exactly one candidate")
        required_metadata = ("test", "entry_point", "code_prompt")
        missing = [key for key in required_metadata if key not in plan.metadata]
        if missing:
            raise ValueError(
                "BigCodeBench evaluator requires metadata keys: " + ", ".join(missing)
            )

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        grader_root = self.resource_dir.resolve()
        details: list[str] = []
        if run_dir is not None and _paths_overlap(grader_root, run_dir):
            self._bcb_python = None
            return EvaluatorPreflightResult(
                name=self.name,
                evaluator_type=self.evaluator_type,
                ok=False,
                version=self.version,
                revision=self.revision,
                details=("BigCodeBench grader directory must be separate from the evaluator run directory",),
            )
        runner_path = grader_root / "bcb_runner.py"
        if not runner_path.is_file():
            self._bcb_python = None
            return EvaluatorPreflightResult(
                name=self.name,
                evaluator_type=self.evaluator_type,
                ok=False,
                version=self.version,
                revision=self.revision,
                details=(f"BigCodeBench grader runner is missing: {runner_path}",),
            )
        try:
            from resources_servers.bigcodebench.setup_bcb_venv import ensure_bcb_venv

            self._bcb_python = ensure_bcb_venv(grader_root / ".bcb_venv", _BCB_PYTHON_VERSION)
        except Exception as exc:
            self._bcb_python = None
            return EvaluatorPreflightResult(
                name=self.name,
                evaluator_type=self.evaluator_type,
                ok=False,
                version=self.version,
                revision=self.revision,
                details=(f"BigCodeBench Python {_BCB_PYTHON_VERSION} grader venv is not ready: {exc}",),
            )
        details.append(f"dedicated Python {_BCB_PYTHON_VERSION} grader ready at {self._bcb_python}")
        return EvaluatorPreflightResult(
            name=self.name,
            evaluator_type=self.evaluator_type,
            ok=True,
            version=self.version,
            revision=self.revision,
            details=tuple(details),
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        candidate = require_one_candidate(request)
        if self._bcb_python is None:
            raise RuntimeError("successful BigCodeBench evaluator preflight is required before evaluate")
        grader_root = self.resource_dir.resolve()
        workspace = candidate.execution.workspace.resolve()
        if _paths_overlap(workspace, grader_root):
            raise RuntimeError("BigCodeBench grader directory must be separate from the executor workspace")

        required_metadata = ("test", "entry_point", "code_prompt")
        if any(key not in request.metadata for key in required_metadata):
            raise ValueError("BigCodeBench evaluator requires test, entry_point, and code_prompt metadata")
        details: dict[str, object] = {
            "execution_status": candidate.execution.status.value,
            "grader": "resources_servers/bigcodebench/bcb_runner.py",
            "grader_invoked": False,
        }
        if candidate.execution.status.value not in _TERMINAL_SUCCESS:
            return EvaluationResult(
                task_id=request.task_id,
                status=EvaluationStatus.COMPLETED,
                metrics={"pass_rate": 0.0},
                details=details,
            )
        output_text = candidate.execution.output_text or ""
        if not output_text.strip():
            details["status"] = "empty_output"
            return EvaluationResult(
                task_id=request.task_id,
                status=EvaluationStatus.COMPLETED,
                metrics={"pass_rate": 0.0},
                details=details,
            )

        native = _native_bigcodebench_evaluate(
            output_text,
            request.metadata,
            resource_dir=grader_root,
            bcb_python=self._bcb_python,
        )
        details.update(
            {
                "status": native.get("status"),
                "extracted_model_code": native.get("extracted_model_code"),
                "grader_details": native.get("details"),
                "grader_invoked": True,
                "grader_root": str(grader_root),
                "executor_workspace": str(workspace),
                "grader_python": str(self._bcb_python) if self._bcb_python else None,
            }
        )
        return EvaluationResult(
            task_id=request.task_id,
            status=EvaluationStatus.COMPLETED,
            metrics={"pass_rate": float(native["reward"])},
            details=details,
        )


__all__ = ["BigCodeBenchEvaluator", "_native_bigcodebench_evaluate"]
