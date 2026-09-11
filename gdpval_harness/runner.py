# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic benchmark × evaluator × executor orchestration."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Mapping

from gdpval_harness.benchmarks.base import Benchmark
from gdpval_harness.evaluators.base import (
    EvaluationCandidate,
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
)
from gdpval_harness.executors.base import ExecutionRequest, ExecutionResult, ExecutionStatus, Executor
from gdpval_harness.interventions.base import (
    Intervention,
    InterventionApplication,
    InterventionPreflightResult,
    ensure_source_output_separation,
)
from gdpval_harness.interventions.none import NoneIntervention
from gdpval_harness.layout import task_layout


_SUCCESS_STATUSES = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
_LEGACY_CONDITION_ENVIRONMENT_KEYS = frozenset(
    {"GDPVAL_CONDITION", "GDPVAL_CONDITION_FILE", "GDPVAL_CONDITION_APPLIED"}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write a JSON file and fsync both file and containing dir."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(handle, payload: Mapping[str, object]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _executor_environment() -> dict[str, str]:
    """Pass the generic executor a clean environment without legacy labels."""

    return {
        key: value
        for key, value in os.environ.items()
        if key not in _LEGACY_CONDITION_ENVIRONMENT_KEYS
    }


def _value(value: object) -> object:
    return value.value if hasattr(value, "value") else value


def _evaluation_payload(evaluation: EvaluationResult) -> dict[str, object]:
    return {
        "status": str(_value(evaluation.status)),
        "metrics": dict(evaluation.metrics),
        "outcomes": dict(evaluation.outcomes),
        "details": dict(evaluation.details),
    }


def _evaluation_error_payload(exc: BaseException, *, phase: str = "evaluation") -> dict[str, object]:
    # Keep the durable failure record useful without serializing tracebacks,
    # request objects, candidate metadata, or arbitrary exception attributes.
    return {
        "status": "failed",
        "metrics": {},
        "outcomes": {},
        "details": {
            "error_type": type(exc).__name__,
            "error_message": "evaluation failure persisted before re-raise",
            "error_details": {
                "phase": phase,
                "exception_type": type(exc).__name__,
            },
        },
    }


def _evaluation_interrupt_payload(exc: BaseException) -> dict[str, object]:
    return {
        "status": "interrupted",
        "metrics": {},
        "outcomes": {},
        "details": {
            "error_type": type(exc).__name__,
            "error_message": "evaluation interrupted after executor returned",
            "error_details": {"phase": "evaluation_interrupt", "exception_type": type(exc).__name__},
        },
    }


def _execution_payload(result: ExecutionResult) -> dict[str, object]:
    return {
        "status": str(_value(result.status)),
        "executor": result.executor,
        "executor_version": result.executor_version,
        "invocation_mode": result.invocation_mode,
        "auth_mode": result.auth_mode,
        "workspace": str(result.workspace),
        "deliverables_dir": str(result.deliverables_dir),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "exit_code": result.exit_code,
        "output_text_present": bool(result.output_text),
        "metadata": dict(result.metadata),
    }


def _aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("status") != EvaluationStatus.COMPLETED.value:
            continue
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                values.setdefault(str(name), []).append(float(value))
    return {name: sum(samples) / len(samples) for name, samples in sorted(values.items()) if samples}


def _evaluation_status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        status = str(evaluation.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _path_matches(actual: Path, expected: Path) -> bool:
    """Require an executor result path to be the runner-assigned path."""

    try:
        actual_absolute = actual.absolute()
        expected_absolute = expected.absolute()
    except (AttributeError, OSError, TypeError):
        return False
    current = Path(actual_absolute.anchor)
    parts = actual_absolute.parts[1:] if actual_absolute.anchor else actual_absolute.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            return False
        if not current.exists():
            break
    try:
        return actual_absolute.resolve(strict=False) == expected_absolute.resolve(strict=False)
    except OSError:
        return False


def _preflight_failure(prefix: str, details: object) -> RuntimeError:
    if isinstance(details, (list, tuple)):
        message = "; ".join(str(detail) for detail in details) or "preflight failed"
    else:
        message = str(details) or "preflight failed"
    return RuntimeError(f"{prefix} preflight failed: {message}")


def _evaluator_metadata(evaluator: Evaluator, preflight: EvaluatorPreflightResult) -> dict[str, object]:
    evaluator_id = getattr(preflight, "name", None) or getattr(evaluator, "name", type(evaluator).__name__)
    evaluator_type = getattr(preflight, "evaluator_type", None) or getattr(evaluator, "evaluator_type", None)
    evaluator_type_value = _value(evaluator_type)
    evaluator_version = getattr(preflight, "version", None)
    if evaluator_version is None:
        evaluator_version = getattr(evaluator, "version", None)
    evaluator_revision = getattr(preflight, "revision", None)
    if evaluator_revision is None:
        evaluator_revision = getattr(evaluator, "revision", None)
    payload: dict[str, object] = {
        "id": evaluator_id,
        "type": evaluator_type_value,
        "version": evaluator_version,
        "revision": evaluator_revision,
        "preflight_details": list(getattr(preflight, "details", ()) or ()),
    }
    judge = {
        "executor": getattr(preflight, "judge_executor", None),
        "version": getattr(preflight, "judge_executor_version", None),
        "auth_mode": getattr(preflight, "judge_auth_mode", None),
        "model": getattr(preflight, "judge_model", None),
    }
    if any(value is not None for value in judge.values()):
        payload["judge"] = judge
    return payload


def _intervention_metadata(
    intervention: Intervention,
    preflight: InterventionPreflightResult,
) -> dict[str, object]:
    """Persist only the reviewed intervention descriptor and hashes."""

    bundle = preflight.bundle
    manifest = bundle.manifest if bundle is not None else None
    application = manifest.application if manifest is not None else None
    intervention_id = (
        manifest.intervention_id
        if manifest is not None
        else getattr(intervention, "intervention_id", None) or preflight.name
    )
    intervention_type = _value(getattr(preflight, "intervention_type", None))
    source_revision = manifest.source_revision if manifest is not None else None
    revision_status = manifest.revision_status if manifest is not None else "unavailable"
    files = (
        [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in manifest.files
        ]
        if manifest is not None
        else []
    )
    descriptor: dict[str, object] = {
        "id": intervention_id,
        "type": intervention_type,
        "revision": source_revision,
        "source_revision": source_revision,
        "revision_status": revision_status,
        "status": "ready" if preflight.ok else "failed",
        "bundle_sha256": manifest.bundle_sha256 if manifest is not None else None,
        "manifest_sha256": manifest.manifest_sha256 if manifest is not None else None,
        "files": files,
        "application": (
            {"method": application.method, "target": application.target}
            if application is not None
            else {"method": None, "target": None}
        ),
    }
    source_reference = getattr(intervention, "source_reference", None)
    if isinstance(source_reference, str) and source_reference and "\x00" not in source_reference:
        descriptor["source_reference"] = source_reference
    return descriptor


def _intervention_application_payload(
    application: InterventionApplication,
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    """Convert application evidence into a safe durable JSON payload."""

    mapping = application.application
    files = [
        {"path": item.path, "size": item.size, "sha256": item.sha256}
        for item in application.materialized_files
    ]
    payload = {
        "id": descriptor.get("id"),
        "type": descriptor.get("type"),
        "revision": descriptor.get("revision"),
        "source_revision": descriptor.get("source_revision"),
        "revision_status": descriptor.get("revision_status"),
        "status": "applied",
        "application_run_id": application.application_run_id,
        "bundle_sha256": application.bundle_sha256,
        "manifest_sha256": application.manifest_sha256,
        "application": {"method": mapping.method, "target": mapping.target},
        "materialized_files": files,
    }
    return payload


def _intervention_error_payload(
    exc: BaseException,
    *,
    descriptor: Mapping[str, object],
    application_run_id: str,
    phase: str,
    interrupted: bool = False,
) -> dict[str, object]:
    """Return safe intervention failure evidence without arbitrary exception text."""

    return {
        "id": descriptor.get("id"),
        "type": descriptor.get("type"),
        "revision": descriptor.get("revision"),
        "source_revision": descriptor.get("source_revision"),
        "revision_status": descriptor.get("revision_status"),
        "status": "interrupted" if interrupted else "failed",
        "application_run_id": application_run_id,
        "bundle_sha256": descriptor.get("bundle_sha256"),
        "manifest_sha256": descriptor.get("manifest_sha256"),
        "application": descriptor.get("application"),
        "materialized_files": [],
        "error_type": type(exc).__name__,
        "error_message": (
            "intervention application interrupted before executor"
            if interrupted
            else "intervention application failed before executor"
        ),
        "error_details": {"phase": phase, "exception_type": type(exc).__name__},
    }


@dataclass(frozen=True)
class RunSummary:
    benchmark: str
    executor: str
    out_dir: Path
    status: str
    task_count: int
    metrics: Mapping[str, float]
    evaluation_status_counts: Mapping[str, int]


def run_benchmark(
    benchmark: Benchmark,
    evaluator: Evaluator,
    executor: Executor,
    *,
    out_dir: Path,
    limit: int,
    model: str | None = None,
    timeout_seconds: float = 12600.0,
    intervention: Intervention | None = None,
) -> RunSummary:
    """Run benchmark tasks using the injected evaluator and executor."""

    if intervention is None:
        intervention = NoneIntervention()
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if timeout_seconds <= 0:
        raise ValueError("--executor-timeout must be positive")
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing run directory: {out_dir}")

    # Evaluator readiness is checked first.  This prevents an executor from
    # consuming a model call when the requested evaluation path is unavailable.
    evaluator_preflight = evaluator.preflight(run_dir=out_dir)
    if not evaluator_preflight.ok:
        raise _preflight_failure(getattr(evaluator, "name", "evaluator"), evaluator_preflight.details)

    intervention_preflight = intervention.preflight()
    if not intervention_preflight.ok:
        raise _preflight_failure(getattr(intervention, "name", "intervention"), intervention_preflight.details)
    intervention_bundle = intervention_preflight.bundle
    if intervention_bundle is not None and isinstance(intervention_bundle.root, Path):
        ensure_source_output_separation(intervention_bundle, out_dir)

    executor_preflight = executor.preflight()
    if not executor_preflight.ok:
        raise _preflight_failure(executor.name, executor_preflight.details)

    benchmark.prepare()
    tasks = list(benchmark.load_tasks(limit))
    # Validate every task plan before creating the run directory or allowing
    # any executor to consume a model call.  An evaluator such as the explicit
    # pairwise adapter can reject the generic runner's one-candidate plan, and
    # a later invalid task must not leave an earlier task partially executed.
    plans: list[EvaluationPlan] = []
    for task in tasks:
        layout = task_layout(out_dir, task.execution.task_id)
        plan = EvaluationPlan(
            task_id=task.execution.task_id,
            task_prompt=task.execution.prompt,
            metadata=task.evaluation,
            candidate_count=1,
            artifact_dir=layout.judge_deliverables,
        )
        evaluator.validate_plan(plan)
        intervention.validate_task(task.execution)
        plans.append(plan)
    out_dir.mkdir(parents=True, exist_ok=False)
    started_at = _now()
    metadata_path = out_dir / "run-metadata.json"
    results_path = out_dir / "results.jsonl"
    network_policy = "enabled" if bool(getattr(executor, "network_enabled", False)) else "disabled"
    evaluator_info = _evaluator_metadata(evaluator, evaluator_preflight)
    intervention_info = _intervention_metadata(intervention, intervention_preflight)
    base_metadata: dict[str, object] = {
        "schema_version": 2,
        "benchmark": benchmark.name,
        "benchmark_revision": getattr(benchmark, "revision", None),
        "evaluator_id": evaluator_info["id"],
        "evaluator_type": evaluator_info["type"],
        "evaluator_version": evaluator_info["version"],
        "evaluator_revision": evaluator_info["revision"],
        "evaluator": evaluator_info,
        "intervention_id": intervention_info["id"],
        "intervention_type": intervention_info["type"],
        "intervention_revision": intervention_info["revision"],
        "intervention": intervention_info,
        "executor": executor.name,
        "executor_version": executor_preflight.version,
        "auth_mode": executor_preflight.auth_mode,
        "model": model,
        "network_policy": network_policy,
        "limit": limit,
        "started_at": started_at,
        "status": "running",
    }
    if "judge" in evaluator_info:
        base_metadata["judge"] = evaluator_info["judge"]
    _write_json(metadata_path, base_metadata)

    rows: list[dict[str, object]] = []
    run_status = "completed"
    try:
        with results_path.open("x", encoding="utf-8", newline="\n") as results_handle:
            for task, plan in zip(tasks, plans):
                layout = task_layout(out_dir, task.execution.task_id)
                layout.workspace.mkdir(parents=True, exist_ok=False)
                layout.executor_dir.mkdir(parents=True, exist_ok=False)
                layout.workspace_deliverables.mkdir(parents=True, exist_ok=True)
                materialized = list(benchmark.materialize(task, layout.workspace))
                execution_task = benchmark.execution_task(
                    task,
                    layout.workspace,
                    network_policy=network_policy,
                )
                # local_judge_runner._candidate_task_prompt consumes this exact
                # canonical raw task prompt.  The executor wrapper is kept in
                # TaskSpec for the model, but is never used as provenance.
                canonical_path = layout.executor_dir / "task-prompt.txt"
                canonical_path.write_text(plan.task_prompt, encoding="utf-8")
                with canonical_path.open("rb") as canonical_handle:
                    canonical_handle.flush()
                    os.fsync(canonical_handle.fileno())
                application_run_id = secrets.token_urlsafe(24)
                try:
                    application = intervention.apply(
                        execution_task,
                        layout.workspace,
                        application_run_id=application_run_id,
                    )
                    if application.application_run_id != application_run_id:
                        raise ValueError("intervention returned a mismatched application_run_id")
                    if application.task.task_id != task.execution.task_id:
                        raise ValueError("intervention returned a mismatched task_id")
                    if application.bundle_sha256 != intervention_info["bundle_sha256"]:
                        raise ValueError("intervention returned a mismatched bundle_sha256")
                    if application.manifest_sha256 != intervention_info["manifest_sha256"]:
                        raise ValueError("intervention returned a mismatched manifest_sha256")
                    application_mapping = {
                        "method": application.application.method,
                        "target": application.application.target,
                    }
                    if application_mapping != intervention_info["application"]:
                        raise ValueError("intervention returned a mismatched application mapping")
                    intervention_payload = _intervention_application_payload(application, intervention_info)
                except KeyboardInterrupt as exc:
                    intervention_payload = _intervention_error_payload(
                        exc,
                        descriptor=intervention_info,
                        application_run_id=application_run_id,
                        phase="intervention_apply",
                        interrupted=True,
                    )
                    row = {
                        "task_id": task.execution.task_id,
                        "materialized": materialized,
                        "intervention": intervention_payload,
                        "execution": None,
                        "evaluation": None,
                    }
                    _write_json(layout.executor_dir.parent / "result.json", row)
                    _append_jsonl(results_handle, row)
                    rows.append(row)
                    base_metadata["failure"] = {
                        "phase": "intervention_apply",
                        "exception_type": type(exc).__name__,
                    }
                    run_status = "interrupted"
                    _write_run_metadata(metadata_path, base_metadata, status=run_status, rows=rows)
                    raise
                except Exception as exc:
                    intervention_payload = _intervention_error_payload(
                        exc,
                        descriptor=intervention_info,
                        application_run_id=application_run_id,
                        phase="intervention_apply",
                    )
                    row = {
                        "task_id": task.execution.task_id,
                        "materialized": materialized,
                        "intervention": intervention_payload,
                        "execution": None,
                        "evaluation": None,
                    }
                    _write_json(layout.executor_dir.parent / "result.json", row)
                    _append_jsonl(results_handle, row)
                    rows.append(row)
                    base_metadata["failure"] = {
                        "phase": "intervention_apply",
                        "exception_type": type(exc).__name__,
                    }
                    run_status = "failed"
                    _write_run_metadata(metadata_path, base_metadata, status=run_status, rows=rows)
                    raise
                request = ExecutionRequest(
                    task=application.task,
                    workspace=layout.workspace,
                    deliverables_dir=layout.workspace_deliverables,
                    executor_dir=layout.executor_dir,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    environment=_executor_environment(),
                )
                result = executor.execute(request)

                def persist_row(evaluation_payload: dict[str, object]) -> dict[str, object]:
                    row = {
                        "task_id": task.execution.task_id,
                        "materialized": materialized,
                        "intervention": intervention_payload,
                        "execution": _execution_payload(result),
                        "evaluation": evaluation_payload,
                    }
                    _write_json(layout.executor_dir.parent / "result.json", row)
                    _append_jsonl(results_handle, row)
                    rows.append(row)
                    return row

                try:
                    if result.task_id != task.execution.task_id:
                        raise ValueError("executor returned a mismatched task_id")
                    if not _path_matches(result.workspace, layout.workspace):
                        raise ValueError("executor returned a workspace outside the assigned task workspace")
                    if not _path_matches(result.deliverables_dir, layout.workspace_deliverables):
                        raise ValueError("executor returned deliverables outside the assigned task directory")
                except Exception as exc:
                    persist_row(_evaluation_error_payload(exc, phase="executor_result_validation"))
                    run_status = "failed"
                    _write_run_metadata(metadata_path, base_metadata, status=run_status, rows=rows)
                    raise

                candidate = EvaluationCandidate(
                    candidate_id="candidate",
                    execution=result,
                    artifacts_dir=result.deliverables_dir,
                )
                evaluation_request = EvaluationRequest(
                    task_id=plan.task_id,
                    task_prompt=plan.task_prompt,
                    metadata=plan.metadata,
                    candidates=(candidate,),
                    artifact_dir=plan.artifact_dir,
                )
                try:
                    evaluation = evaluator.evaluate(evaluation_request)
                    if evaluation.task_id != task.execution.task_id:
                        raise ValueError(
                            f"evaluator returned task id {evaluation.task_id!r}; "
                            f"expected {task.execution.task_id!r}"
                        )
                    evaluation_payload = _evaluation_payload(evaluation)
                except KeyboardInterrupt as exc:
                    persist_row(_evaluation_interrupt_payload(exc))
                    run_status = "interrupted"
                    _write_run_metadata(metadata_path, base_metadata, status=run_status, rows=rows)
                    raise
                except Exception as exc:
                    evaluation_payload = _evaluation_error_payload(exc)
                    persist_row(evaluation_payload)
                    run_status = "failed"
                    _write_run_metadata(
                        metadata_path,
                        base_metadata,
                        status=run_status,
                        rows=rows,
                    )
                    raise

                persist_row(evaluation_payload)

                if result.status not in _SUCCESS_STATUSES:
                    run_status = "failed"
                    break
    except KeyboardInterrupt:
        run_status = "interrupted"
        _write_run_metadata(metadata_path, base_metadata, status=run_status, rows=rows)
        raise
    except Exception:
        run_status = "failed"
        _write_run_metadata(metadata_path, base_metadata, status=run_status, rows=rows)
        raise

    metrics = _aggregate_metrics(rows)
    _write_run_metadata(metadata_path, base_metadata, status=run_status, rows=rows, metrics=metrics)
    return RunSummary(
        benchmark=benchmark.name,
        executor=executor.name,
        out_dir=out_dir,
        status=run_status,
        task_count=len(rows),
        metrics=metrics,
        evaluation_status_counts=_evaluation_status_counts(rows),
    )


def _write_run_metadata(
    metadata_path: Path,
    base_metadata: Mapping[str, object],
    *,
    status: str,
    rows: list[dict[str, object]],
    metrics: Mapping[str, float] | None = None,
) -> None:
    final_metadata = dict(base_metadata)
    final_metadata.update(
        {
            "status": status,
            "finished_at": _now(),
            "completed_tasks": len(rows),
            "metrics": dict(metrics if metrics is not None else _aggregate_metrics(rows)),
            "evaluation_status_counts": _evaluation_status_counts(rows),
        }
    )
    _write_json(metadata_path, final_metadata)


__all__ = ["RunSummary", "run_benchmark"]
