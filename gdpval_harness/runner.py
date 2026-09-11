# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkEvaluation
from gdpval_harness.executors.base import ExecutionRequest, ExecutionResult, ExecutionStatus, Executor
from gdpval_harness.layout import task_layout


_SUCCESS_STATUSES = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(handle, payload: Mapping[str, object]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _evaluation_payload(evaluation: BenchmarkEvaluation | None, *, external: bool) -> dict[str, object]:
    if evaluation is not None:
        return {
            "status": "completed",
            "metrics": dict(evaluation.metrics),
            "details": dict(evaluation.details),
        }
    if external:
        return {
            "status": "external",
            "metrics": {},
            "details": {},
        }
    return {
        "status": "skipped",
        "metrics": {},
        "details": {},
    }


def _execution_payload(result: ExecutionResult) -> dict[str, object]:
    return {
        "status": result.status.value,
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
        if not isinstance(evaluation, dict):
            continue
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                values.setdefault(str(name), []).append(float(value))
    return {name: sum(samples) / len(samples) for name, samples in sorted(values.items()) if samples}


@dataclass(frozen=True)
class RunSummary:
    benchmark: str
    executor: str
    out_dir: Path
    status: str
    task_count: int
    metrics: Mapping[str, float]


def run_benchmark(
    benchmark: Benchmark,
    executor: Executor,
    *,
    out_dir: Path,
    limit: int,
    model: str | None = None,
    timeout_seconds: float = 12600.0,
) -> RunSummary:
    if limit <= 0:
        raise ValueError("--limit must be positive")
    if timeout_seconds <= 0:
        raise ValueError("--executor-timeout must be positive")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {out_dir}")

    preflight = executor.preflight()
    if not preflight.ok:
        detail = "; ".join(preflight.details) or "preflight failed"
        raise RuntimeError(f"{executor.name} preflight failed: {detail}")

    benchmark.prepare()
    tasks = list(benchmark.load_tasks(limit))
    out_dir.mkdir(parents=True, exist_ok=False)
    started_at = _now()
    metadata_path = out_dir / "run-metadata.json"
    results_path = out_dir / "results.jsonl"
    network_policy = "enabled" if bool(getattr(executor, "network_enabled", False)) else "disabled"
    base_metadata: dict[str, object] = {
        "schema_version": 1,
        "benchmark": benchmark.name,
        "benchmark_revision": benchmark.revision,
        "evaluator_type": benchmark.evaluator_type.value,
        "executor": executor.name,
        "executor_version": preflight.version,
        "auth_mode": preflight.auth_mode,
        "model": model,
        "network_policy": network_policy,
        "limit": limit,
        "started_at": started_at,
        "status": "running",
    }
    _write_json(metadata_path, base_metadata)

    rows: list[dict[str, object]] = []
    run_status = "completed"
    try:
        with results_path.open("x", encoding="utf-8") as results_handle:
            for task in tasks:
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
                (layout.executor_dir / "canonical-task.txt").write_text(
                    task.execution.prompt,
                    encoding="utf-8",
                )
                request = ExecutionRequest(
                    task=execution_task,
                    workspace=layout.workspace,
                    deliverables_dir=layout.workspace_deliverables,
                    executor_dir=layout.executor_dir,
                    model=model,
                    timeout_seconds=timeout_seconds,
                )
                result = executor.execute(request)

                evaluation: BenchmarkEvaluation | None = None
                evaluation_external = False
                if result.status in _SUCCESS_STATUSES:
                    try:
                        evaluation = benchmark.evaluate(task, result)
                    except NotImplementedError:
                        evaluation_external = True

                row: dict[str, object] = {
                    "task_id": task.execution.task_id,
                    "materialized": materialized,
                    "execution": _execution_payload(result),
                    "evaluation": _evaluation_payload(evaluation, external=evaluation_external),
                }
                _write_json(layout.executor_dir.parent / "result.json", row)
                _append_jsonl(results_handle, row)
                rows.append(row)

                if result.status not in _SUCCESS_STATUSES:
                    run_status = "failed"
                    break
    except KeyboardInterrupt:
        run_status = "interrupted"
        final_metadata = dict(base_metadata)
        final_metadata.update(
            {
                "status": run_status,
                "finished_at": _now(),
                "completed_tasks": len(rows),
                "metrics": _aggregate_metrics(rows),
            }
        )
        _write_json(metadata_path, final_metadata)
        raise
    except Exception:
        run_status = "failed"
        final_metadata = dict(base_metadata)
        final_metadata.update(
            {
                "status": run_status,
                "finished_at": _now(),
                "completed_tasks": len(rows),
                "metrics": _aggregate_metrics(rows),
            }
        )
        _write_json(metadata_path, final_metadata)
        raise

    metrics = _aggregate_metrics(rows)
    final_metadata = dict(base_metadata)
    final_metadata.update(
        {
            "status": run_status,
            "finished_at": _now(),
            "completed_tasks": len(rows),
            "metrics": metrics,
        }
    )
    _write_json(metadata_path, final_metadata)
    return RunSummary(
        benchmark=benchmark.name,
        executor=executor.name,
        out_dir=out_dir,
        status=run_status,
        task_count=len(rows),
        metrics=metrics,
    )
