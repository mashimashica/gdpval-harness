# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit local pairwise adapter for the existing blind JudgeExecutor API."""

from __future__ import annotations

import inspect
import os
import tempfile
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
    require_two_candidates,
)
from gdpval_harness.executors.base import ExecutionStatus
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult, Verdict
from gdpval_harness.judges.pairwise import (
    aggregate,
    build_judge_prompt,
    normalize_verdict,
    prepare_trial,
    validate_reference_equivalence,
    write_trial_metadata,
)
from gdpval_harness.layout import safe_task_id


# This mirrors the local judge runner's minimal environment.  The generic
# adapter accepts an already selected environment, but strips candidate paths,
# labels, and arbitrary secrets before passing it to a JudgeExecutor.
_ENVIRONMENT_ALLOWLIST = {
    "ALL_PROXY",
    "APPDATA",
    "CODEX_CA_CERTIFICATE",
    "CODEX_HOME",
    "COMSPEC",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "LOGNAME",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SystemRoot",
    "TERM",
    "USER",
    "USERPROFILE",
    "WINDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "XDG_STATE_HOME",
    "CLAUDE_CONFIG_DIR",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


def sanitize_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    """Keep only runtime/auth plumbing accepted by the blind judge boundary."""

    if not environment:
        return {}
    return {
        str(name): str(value)
        for name, value in environment.items()
        if str(name) in _ENVIRONMENT_ALLOWLIST
    }


def _call_preflight(judge: JudgeExecutor, environment: Mapping[str, str]) -> JudgePreflightResult:
    """Call old and current JudgeExecutor preflight spellings safely."""

    method = judge.preflight
    try:
        parameters = tuple(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        # A callable implemented in an extension or a mock may not expose a
        # signature.  The current concrete contract accepts an optional
        # environment, so use that form without catching errors raised inside
        # the implementation.
        return method(environment)
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    accepts_varargs = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
    keyword_environment = any(
        parameter.name == "environment" and parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters
    )
    if keyword_environment:
        return method(environment=environment)
    return method(environment) if positional or accepts_varargs else method()


def _preflight_fields(judge: JudgeExecutor, result: JudgePreflightResult) -> tuple[str | None, str | None, str | None]:
    return (
        getattr(result, "judge_executor", None) or getattr(judge, "name", None),
        getattr(result, "version", None),
        getattr(result, "auth_mode", None),
    )


def _assert_candidate_dir(path: Path | None, *, candidate_id: str) -> Path:
    if path is None:
        raise ValueError(f"pairwise candidate {candidate_id!r} is missing its judge-facing artifact directory")
    try:
        absolute = path.absolute()
    except OSError:
        absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink not allowed in pairwise candidate artifact path: {current}")
        if not current.exists():
            break
    if not absolute.is_dir():
        raise ValueError(f"pairwise candidate artifact directory not found: {path}")
    return absolute


def _assert_output_root(path: Path | None) -> Path:
    if path is None:
        raise ValueError("pairwise evaluator requires request.artifact_dir as a new output root")
    try:
        absolute = path.absolute()
    except OSError:
        absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink not allowed in pairwise output root: {current}")
        if not current.exists():
            break
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing pairwise output root: {absolute}")
    parent = absolute.parent
    if parent.exists() and parent.is_symlink():
        raise ValueError(f"symlink not allowed in pairwise output parent: {parent}")
    return absolute


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _write_trial_metadata_preserving(path: Path, payload: dict[str, object]) -> Path:
    """Write harness metadata without replacing a judge-created metadata file."""

    target = path / "metadata.json"
    if target.exists() or target.is_symlink():
        target = path / "harness-metadata.json"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite pairwise trial metadata path: {target}")
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=path)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        # Reuse the existing canonical metadata serializer, then publish with
        # an exclusive hard link so a concurrent judge artifact can never be
        # replaced by evaluator bookkeeping.
        write_trial_metadata(temporary, payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except BaseException:
        raise
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(target.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def _jsonable_verdict(value: Verdict | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Verdict):
        return value.value
    return str(value)


class PairwiseJudgeEvaluator(Evaluator):
    """Run an explicitly injected local :class:`JudgeExecutor` pairwise."""

    name = "pairwise-judge"
    evaluator_type = EvaluatorType.PAIRWISE
    version = "1"
    revision = "judge-pairwise-v1"

    def __init__(
        self,
        judge_executor: JudgeExecutor,
        *,
        trials: int = 2,
        seed: int = 42,
        model: str | None = None,
        timeout_seconds: float = 3600.0,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if trials <= 0:
            raise ValueError("pairwise trials must be positive")
        if timeout_seconds <= 0:
            raise ValueError("pairwise timeout_seconds must be positive")
        self.judge_executor = judge_executor
        self.trials = int(trials)
        self.seed = int(seed)
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.environment = sanitize_environment(environment)
        self._ready = False

    @property
    def judge(self) -> JudgeExecutor:
        """Compatibility spelling for callers that expose an injected judge."""

        return self.judge_executor

    def validate_plan(self, plan: EvaluationPlan) -> None:
        """Require an explicit two-candidate pairwise plan before execution."""

        if plan.candidate_count != 2:
            raise ValueError("pairwise evaluator requires an evaluation plan with exactly two candidates")
        if plan.artifact_dir is None:
            raise ValueError("pairwise evaluator requires an artifact destination in the evaluation plan")

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        result = _call_preflight(self.judge_executor, self.environment)
        self._ready = bool(getattr(result, "ok", False))
        judge_name, judge_version, judge_auth_mode = _preflight_fields(self.judge_executor, result)
        details = tuple(str(detail) for detail in (getattr(result, "details", ()) or ()))
        ok = bool(getattr(result, "ok", False))
        return EvaluatorPreflightResult(
            name=self.name,
            evaluator_type=self.evaluator_type,
            ok=ok,
            version=self.version,
            revision=self.revision,
            details=details,
            judge_executor=judge_name,
            judge_executor_version=judge_version,
            judge_auth_mode=judge_auth_mode,
            judge_model=self.model,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if not self._ready:
            raise RuntimeError("pairwise judge evaluator requires a successful preflight before evaluation")
        first, second = require_two_candidates(request)
        terminal = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
        if first.execution.status not in terminal or second.execution.status not in terminal:
            raise ValueError("pairwise evaluator requires terminal candidate executions")
        candidate_a_dir = _assert_candidate_dir(first.artifacts_dir, candidate_id=first.candidate_id)
        candidate_b_dir = _assert_candidate_dir(second.artifacts_dir, candidate_id=second.candidate_id)
        validate_reference_equivalence(candidate_a_dir, candidate_b_dir)

        output_root = _assert_output_root(request.artifact_dir)
        if _paths_overlap(output_root, candidate_a_dir) or _paths_overlap(output_root, candidate_b_dir):
            raise ValueError("pairwise output root must be separate from both candidate artifact directories")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        # Parent creation can encounter a symlink race; recheck the lexical
        # path before making the output root visible.
        if output_root.parent.is_symlink():
            raise ValueError(f"symlink not allowed in pairwise output parent: {output_root.parent}")
        output_root.mkdir(exist_ok=False)

        task_key = safe_task_id(request.task_id)
        judge_prompt = build_judge_prompt(request.task_prompt)
        trial_verdicts: list[dict[str, object]] = []
        normalized_values: list[Verdict] = []

        for trial_index in range(self.trials):
            prepared = prepare_trial(
                output_root,
                task_key,
                candidate_a_dir,
                candidate_b_dir,
                trial_index=trial_index,
                seed=self.seed,
            )
            runtime_tmp = prepared.executor_dir.parent / "runtime-tmp"
            runtime_tmp.mkdir(parents=True, exist_ok=True)
            trial_environment = dict(self.environment)
            trial_environment.update({"TMPDIR": str(runtime_tmp), "TMP": str(runtime_tmp), "TEMP": str(runtime_tmp)})
            judge_request = JudgeRequest(
                task_id=request.task_id,
                task_prompt=judge_prompt,
                workspace=prepared.workspace,
                reference_dir=prepared.reference_dir,
                submission_a_dir=prepared.submission_a_dir,
                submission_b_dir=prepared.submission_b_dir,
                executor_dir=prepared.executor_dir,
                trial_index=trial_index,
                swapped=prepared.swapped,
                model=self.model,
                timeout_seconds=self.timeout_seconds,
                environment=trial_environment,
            )
            row: dict[str, object] = {
                "task_id": request.task_id,
                "trial_index": trial_index,
                "swapped": prepared.swapped,
                "judge_executor": getattr(self.judge_executor, "name", None),
                "judge_model": self.model,
            }
            try:
                result = self.judge_executor.judge(judge_request)
            except Exception as exc:
                row.update(
                    {
                        "error_type": type(exc).__name__,
                        "error_message": "pairwise judge failure persisted before re-raise",
                        "error_details": {"phase": "judge_call", "exception_type": type(exc).__name__},
                    }
                )
                _write_trial_metadata_preserving(prepared.executor_dir, row)
                raise RuntimeError(
                    f"pairwise judge failed for task {request.task_id!r} trial {trial_index}"
                ) from exc
            except BaseException as exc:
                row.update(
                    {
                        "error_type": type(exc).__name__,
                        "error_message": "pairwise judge interrupted before result persistence",
                        "error_details": {"phase": "judge_call", "exception_type": type(exc).__name__},
                    }
                )
                _write_trial_metadata_preserving(prepared.executor_dir, row)
                raise

            row.update(
                {
                    "judge_executor": getattr(result, "judge_executor", None),
                    "judge_executor_version": getattr(result, "executor_version", None),
                    "judge_auth_mode": getattr(result, "auth_mode", None),
                    "exit_code": getattr(result, "exit_code", None),
                    "started_at": getattr(result, "started_at", None),
                    "finished_at": getattr(result, "finished_at", None),
                    "blind_verdict": _jsonable_verdict(getattr(result, "verdict", None)),
                    "metadata": dict(getattr(result, "metadata", {}) or {}),
                }
            )

            result_task_id = getattr(result, "task_id", None)
            verdict = getattr(result, "verdict", None)
            exit_code = getattr(result, "exit_code", None)
            normalized: Verdict | None = None
            failure: str | None = None
            if result_task_id != request.task_id:
                failure = "judge result task id mismatch"
            elif exit_code != 0:
                failure = "judge executor returned a nonzero exit code"
            elif not isinstance(verdict, Verdict):
                failure = "judge result is missing a valid verdict"
            else:
                normalized = normalize_verdict(verdict, prepared.swapped)

            row["normalized_verdict"] = _jsonable_verdict(normalized)
            if failure is not None:
                row["error"] = failure
                _write_trial_metadata_preserving(prepared.executor_dir, row)
                raise RuntimeError(f"pairwise judge failed closed for task {request.task_id!r} trial {trial_index}")

            assert normalized is not None
            normalized_values.append(normalized)
            trial_verdicts.append(
                {
                    "trial_index": trial_index,
                    "swapped": prepared.swapped,
                    "blind_verdict": verdict.value,
                    "normalized_verdict": normalized.value,
                }
            )
            _write_trial_metadata_preserving(prepared.executor_dir, row)

        counts = aggregate(normalized_values)
        wins_a = int(counts["wins_a"])
        wins_b = int(counts["wins_b"])
        ties = int(counts["ties"])
        outcomes = {
            "candidate_a": {"candidate_id": first.candidate_id, "wins": wins_a},
            "candidate_b": {"candidate_id": second.candidate_id, "wins": wins_b},
            "candidate_outcomes": {
                first.candidate_id: {"wins": wins_a},
                second.candidate_id: {"wins": wins_b},
            },
            "ties": ties,
            "trial_verdicts": trial_verdicts,
            "counts": {"trials": len(normalized_values), "wins_a": wins_a, "wins_b": wins_b, "ties": ties},
        }
        return EvaluationResult(
            task_id=request.task_id,
            status=EvaluationStatus.COMPLETED,
            metrics={},
            outcomes=outcomes,
            details={
                "judge_executor": getattr(self.judge_executor, "name", None),
                "judge_model": self.model,
                "trials": self.trials,
                "seed": self.seed,
                "prompt": "existing pairwise blind judge prompt",
                "output_root": str(output_root),
            },
        )


__all__ = ["PairwiseJudgeEvaluator", "sanitize_environment"]
