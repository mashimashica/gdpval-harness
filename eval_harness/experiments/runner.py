# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic orchestration for benchmark-task Builder experiments.

This module deliberately keeps the Builder stage outside the normal benchmark
runner.  A Builder creates one sealed intervention bundle for one task/arm
pair; the existing runner then consumes that bundle through the normal
``AgentSkillIntervention`` path.
"""

from __future__ import annotations

import json
import os
import random
import secrets as _secrets_module
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence, TypeVar

from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.builders.base import (
    Builder,
    BuilderInputBundle,
    BuilderInputManifest,
    BuilderPreflightResult,
    BuildRequest,
    BuildResult,
    BuildStatus,
)
from eval_harness.evaluators.base import EvaluationPlan, Evaluator, EvaluatorPreflightResult
from eval_harness.executors.base import Executor, PreflightResult, TaskSpec
from eval_harness.experiments.base import (
    ExperimentArm,
    ExperimentRunConfig,
    ExperimentRunSummary,
    LoadedExperimentProfile,
)
from eval_harness.experiments.profile import load_experiment_inputs
from eval_harness.failures import FailureKind, RunAbort
from eval_harness.interventions.agent_skill import AgentSkillIntervention
from eval_harness.interventions.base import InterventionBundle
from eval_harness.layout import task_layout
from eval_harness.provenance import (
    canonical_json_sha256,
    execution_record,
    repository_provenance,
    task_sha256,
)
from eval_harness.reasoning import validate_executor_reasoning_effort
from eval_harness.runner import RunSummary, run_benchmark


class _SecretsFacade:
    """Keep experiment ID patching isolated from the core runner's temp IDs."""

    @staticmethod
    def token_hex(nbytes: int = 32) -> str:
        return _secrets_module.token_hex(nbytes)


secrets = _SecretsFacade()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    """Fsync a directory where the host platform permits directory handles."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write UTF-8 JSON and durably replace the destination."""

    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}-{id(payload)}")
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


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"could not inspect path: {path}") from exc
    return True


def _canonical_existing(
    path: Path,
    *,
    label: str,
    directory: bool | None = None,
    require_canonical: bool = True,
) -> Path:
    """Require an existing, canonical, non-symlink path."""

    try:
        metadata = path.lstat()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} must be an existing canonical non-symlink path") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if directory is True and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be an existing directory")
    if directory is False and not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be an existing regular file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} must be canonical") from exc
    if require_canonical and resolved != path:
        raise ValueError(f"{label} must be canonical")
    return resolved


def _canonical_planned_root(path: Path, *, label: str) -> Path:
    """Validate one fresh absolute root without creating it."""

    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if _path_exists_without_following(path):
        raise FileExistsError(f"{label} already exists: {path}")
    try:
        _canonical_existing(path.parent, label=f"{label} parent", directory=True)
        resolved = path.resolve(strict=False)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} must be an absolute canonical path") from exc
    if resolved != path:
        raise ValueError(f"{label} must be canonical")
    if _path_exists_without_following(resolved):
        raise FileExistsError(f"{label} already exists: {path}")
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_path_namespace(
    out_root: Path,
    runtime_root: Path,
    profile_source: Path,
    sources: Mapping[str, Path],
) -> None:
    if _paths_overlap(out_root, runtime_root):
        raise ValueError("out_dir and runtime_root must be separate, non-overlapping roots")
    if _paths_overlap(out_root, profile_source) or _paths_overlap(runtime_root, profile_source):
        raise ValueError("planned roots must be separate from the profile source")
    for source in sources.values():
        if _paths_overlap(out_root, source) or _paths_overlap(runtime_root, source):
            raise ValueError("planned roots must be separate from experiment input sources")


_T = TypeVar("_T")


def _require_instance(value: object, expected: type[object], label: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be a {expected.__name__}")


def _require_exact_result(value: object, expected: type[_T], label: str) -> _T:
    if type(value) is not expected or not isinstance(value, expected):
        raise TypeError(f"{label} must be exactly a {expected.__name__}")
    return value


def _validate_loaded_inputs(
    profile: LoadedExperimentProfile,
    source_roots: Mapping[str, Path | str],
    loaded: object,
) -> tuple[dict[str, BuilderInputBundle], dict[str, Path]]:
    if not isinstance(source_roots, Mapping):
        raise TypeError("source_roots must be a mapping")
    input_specs = {item.input_id: item for item in profile.profile.inputs}
    if set(source_roots) != set(input_specs):
        raise ValueError("source_roots must contain exactly the profile input IDs")

    canonical_sources: dict[str, Path] = {}
    for input_id, source in source_roots.items():
        if not isinstance(input_id, str):
            raise TypeError("source_roots keys must be strings")
        try:
            source_path = Path(source)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"source root for {input_id!r} must be path-like") from exc
        canonical_sources[input_id] = _canonical_existing(
            source_path,
            label=f"experiment input {input_id!r} source root",
            directory=True,
            require_canonical=False,
        )

    if not isinstance(loaded, Mapping):
        raise TypeError("load_experiment_inputs must return a mapping of input IDs to bundles")
    bundles: dict[str, BuilderInputBundle] = {}
    for key, bundle in loaded.items():
        if not isinstance(key, str) or not key:
            raise TypeError("loaded experiment input keys must be non-empty strings")
        if type(bundle) is not BuilderInputBundle:
            raise TypeError("loaded experiment inputs must contain exact BuilderInputBundle instances")
        if key in bundles:
            raise ValueError(f"duplicate loaded experiment input: {key!r}")
        bundles[key] = bundle
    if set(bundles) != set(input_specs):
        raise ValueError("load_experiment_inputs returned an input set different from the profile")

    for input_id, spec in input_specs.items():
        bundle = bundles[input_id]
        manifest = bundle.manifest
        if type(manifest) is not BuilderInputManifest:
            raise TypeError("loaded experiment input manifests must be exact BuilderInputManifest instances")
        root = _canonical_existing(bundle.root, label=f"experiment input {input_id!r} source root", directory=True)
        if root != canonical_sources[input_id]:
            raise ValueError(f"loaded input {input_id!r} is bound to the wrong source root")
        if manifest.input_id != spec.input_id:
            raise ValueError(f"loaded input {input_id!r} has a mismatched input ID")
        if manifest.input_type != spec.input_type:
            raise ValueError(f"loaded input {input_id!r} has a mismatched input type")
        if manifest.source_revision != spec.source_revision or manifest.revision_status != spec.revision_status:
            raise ValueError(f"loaded input {input_id!r} has a mismatched source revision")
        manifest_paths = tuple(item.path for item in manifest.files)
        if manifest_paths != tuple(spec.allowed_files):
            raise ValueError(f"loaded input {input_id!r} has a mismatched allowlist")
        if spec.expected_bundle_sha256 is not None and manifest.bundle_sha256 != spec.expected_bundle_sha256:
            raise ValueError(f"loaded input {input_id!r} has a mismatched bundle hash")

    return bundles, canonical_sources


def _validate_profile_and_components(
    profile: LoadedExperimentProfile,
    run_config: ExperimentRunConfig,
    benchmark: Benchmark,
    evaluator: Evaluator,
    builder: Builder,
    application_executor: Executor,
    out_root: Path,
) -> tuple[EvaluatorPreflightResult, PreflightResult, BuilderPreflightResult]:
    _require_instance(profile, LoadedExperimentProfile, "profile")
    _require_instance(run_config, ExperimentRunConfig, "run_config")
    _require_instance(benchmark, Benchmark, "benchmark")
    _require_instance(evaluator, Evaluator, "evaluator")
    _require_instance(builder, Builder, "builder")
    _require_instance(application_executor, Executor, "application_executor")

    if benchmark.name != profile.profile.benchmark:
        raise ValueError("benchmark name does not match experiment profile")
    if not isinstance(evaluator.name, str) or evaluator.name != run_config.evaluator:
        raise ValueError("evaluator identity does not match run configuration")
    if not isinstance(application_executor.name, str) or application_executor.name != run_config.application_executor:
        raise ValueError("application executor identity does not match run configuration")

    builder_runtime = getattr(builder, "executor", None)
    if (
        validate_executor_reasoning_effort(
            run_config.builder_executor,
            getattr(builder_runtime, "reasoning_effort", None),
        )
        != run_config.builder_reasoning_effort
    ):
        raise ValueError("Builder reasoning_effort does not match the configured Builder executor")
    if (
        validate_executor_reasoning_effort(
            application_executor,
            getattr(application_executor, "reasoning_effort", None),
        )
        != run_config.application_reasoning_effort
    ):
        raise ValueError("application reasoning_effort does not match the configured application executor")

    evaluator_preflight = _require_exact_result(
        evaluator.preflight(run_dir=out_root), EvaluatorPreflightResult, "evaluator preflight result"
    )
    if evaluator_preflight.ok is not True or evaluator_preflight.name != run_config.evaluator:
        raise RuntimeError("evaluator preflight failed or returned a mismatched identity")

    application_preflight = _require_exact_result(
        application_executor.preflight(), PreflightResult, "application executor preflight result"
    )
    if (
        application_preflight.ok is not True
        or application_preflight.executor != run_config.application_executor
        or application_preflight.executor != application_executor.name
    ):
        raise RuntimeError("application executor preflight failed or returned a mismatched identity")

    builder_preflight = _require_exact_result(builder.preflight(), BuilderPreflightResult, "builder preflight result")
    if builder_preflight.ok is not True or builder_preflight.builder_executor != run_config.builder_executor:
        raise RuntimeError("builder preflight failed or returned a mismatched builder executor identity")
    builder_name = getattr(builder, "name", None)
    if not isinstance(builder_name, str) or builder_preflight.name != builder_name:
        raise RuntimeError("builder preflight returned a mismatched builder identity")

    return evaluator_preflight, application_preflight, builder_preflight


def _validate_tasks(benchmark: Benchmark, limit: int) -> tuple[BenchmarkTask, ...]:
    benchmark.prepare()
    loaded = benchmark.load_tasks(limit)
    try:
        tasks = tuple(loaded)
    except TypeError as exc:
        raise TypeError("benchmark.load_tasks must return a sequence of BenchmarkTask instances") from exc
    if not tasks:
        raise ValueError("benchmark.load_tasks returned no selected tasks")
    if len(tasks) > limit:
        raise ValueError("benchmark.load_tasks returned more tasks than the configured limit")
    seen: set[str] = set()
    for task in tasks:
        if type(task) is not BenchmarkTask:
            raise TypeError("benchmark tasks must be exact BenchmarkTask instances")
        if type(task.execution) is not TaskSpec:
            raise TypeError("benchmark task execution must be an exact TaskSpec")
        task_id = task.execution.task_id
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("benchmark task IDs must be non-empty strings")
        if task_id in seen:
            raise ValueError(f"benchmark task IDs must be unique: {task_id!r}")
        seen.add(task_id)
    return tasks


def _safe_schedule_id(value: object) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("opaque schedule IDs must be non-empty path components")
    if "\x00" in value or "/" in value or "\\" in value or PurePosixPath(value).as_posix() != value:
        raise ValueError("opaque schedule IDs must be a single relative path component")
    return value


@dataclass(frozen=True)
class _ScheduleItem:
    index: int
    task: BenchmarkTask
    arm: ExperimentArm
    schedule_id: str
    output_root: Path
    build_root: Path
    artifact_root: Path
    application_root: Path
    judge_artifact_root: Path


class _SelectedTaskBenchmark(Benchmark):
    """A fixed-task view used for one application invocation."""

    def __init__(self, original: Benchmark, selected: BenchmarkTask) -> None:
        self._original = original
        self._selected = selected
        self.name = original.name
        self.revision = getattr(original, "revision", None)

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        return None

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        if limit != 1:
            raise ValueError("selected benchmark wrapper only supports limit=1")
        return [self._selected]

    def materialize(self, task: BenchmarkTask, workspace: Path) -> Sequence[str]:
        if task is not self._selected:
            raise ValueError("selected benchmark wrapper received a different task")
        return self._original.materialize(self._selected, workspace)

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        if task is not self._selected:
            raise ValueError("selected benchmark wrapper received a different task")
        return self._original.execution_task(self._selected, workspace, network_policy=network_policy)


def _manifest_payload(manifest: BuilderInputManifest) -> dict[str, object]:
    return {
        "input_id": manifest.input_id,
        "input_type": manifest.input_type,
        "source_revision": manifest.source_revision,
        "revision_status": manifest.revision_status,
        "bundle_sha256": manifest.bundle_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "files": [{"path": item.path, "size": item.size, "sha256": item.sha256} for item in manifest.files],
    }


def _config_payload(config: ExperimentRunConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "builder_executor": config.builder_executor,
        "application_executor": config.application_executor,
        "evaluator": config.evaluator,
        "builder_model": config.builder_model,
        "application_model": config.application_model,
        "builder_timeout_seconds": config.builder_timeout_seconds,
        "application_timeout_seconds": config.application_timeout_seconds,
        "builder_network_enabled": config.builder_network_enabled,
        "application_network_enabled": config.application_network_enabled,
        "limit": config.limit,
        "order_seed": config.order_seed,
    }
    if config.builder_reasoning_effort is not None:
        payload["builder_reasoning_effort_requested"] = config.builder_reasoning_effort
    if config.application_reasoning_effort is not None:
        payload["application_reasoning_effort_requested"] = config.application_reasoning_effort
    return payload


def _nullable_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _enum_text(value: object) -> str | None:
    try:
        value = getattr(value, "value", value)
    except Exception:
        return None
    return _nullable_text(value)


def _executor_invocation_mode(executor: Executor) -> str | None:
    try:
        value = getattr(executor, "invocation_mode", None)
    except Exception:
        return None
    return _nullable_text(value)


def _builder_descriptor(
    builder: Builder,
    preflight: BuilderPreflightResult,
    config: ExperimentRunConfig,
) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "id": _nullable_text(getattr(builder, "name", None)) or _nullable_text(preflight.name),
        "executor": _nullable_text(preflight.builder_executor),
        "executor_version": _nullable_text(preflight.builder_executor_version),
        "invocation_mode": _nullable_text(preflight.builder_executor_invocation_mode),
        "auth_mode": _nullable_text(preflight.builder_executor_auth_mode),
        "model": config.builder_model,
        "network_policy": "enabled" if config.builder_network_enabled else "disabled",
    }
    if config.builder_reasoning_effort is not None:
        descriptor["reasoning_effort_requested"] = config.builder_reasoning_effort
    return descriptor


def _application_executor_descriptor(
    executor: Executor,
    preflight: PreflightResult,
    config: ExperimentRunConfig,
) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "id": _nullable_text(preflight.executor),
        "executor": _nullable_text(preflight.executor),
        "executor_version": _nullable_text(preflight.version),
        "invocation_mode": _executor_invocation_mode(executor),
        "auth_mode": _nullable_text(preflight.auth_mode),
        "model": config.application_model,
        "network_policy": "enabled" if config.application_network_enabled else "disabled",
    }
    if config.application_reasoning_effort is not None:
        descriptor["reasoning_effort_requested"] = config.application_reasoning_effort
    return descriptor


def _evaluator_descriptor(preflight: EvaluatorPreflightResult) -> dict[str, object]:
    evaluator_type = _enum_text(preflight.evaluator_type)
    judge_applicable = evaluator_type in {"llm-rubric", "pairwise"}
    return {
        "id": _nullable_text(preflight.name),
        "type": evaluator_type,
        "version": _nullable_text(preflight.version),
        "revision": _nullable_text(preflight.revision),
        "judge": {
            "applicable": judge_applicable,
            "executor": _nullable_text(preflight.judge_executor) if judge_applicable else None,
            "version": _nullable_text(preflight.judge_executor_version) if judge_applicable else None,
            "auth_mode": _nullable_text(preflight.judge_auth_mode) if judge_applicable else None,
            "model": _nullable_text(preflight.judge_model) if judge_applicable else None,
        },
    }


def _benchmark_descriptor(benchmark: Benchmark) -> dict[str, object]:
    revision = _nullable_text(getattr(benchmark, "revision", None))
    return {
        "id": _nullable_text(benchmark.name),
        "revision": revision,
        "revision_status": "available" if revision is not None else "unavailable",
    }


def _configuration_payload(
    profile: LoadedExperimentProfile,
    config: ExperimentRunConfig,
    benchmark: Benchmark,
    source_bundles: Mapping[str, BuilderInputBundle],
    builder: Builder,
    builder_preflight: BuilderPreflightResult,
    application_preflight: PreflightResult,
    evaluator_preflight: EvaluatorPreflightResult,
    application_executor: Executor,
) -> dict[str, object]:
    return {
        "profile": {"id": profile.profile.profile_id, "sha256": profile.sha256},
        "benchmark": _benchmark_descriptor(benchmark),
        "run_config": _config_payload(config),
        "inputs": [
            {
                "input_id": spec.input_id,
                "input_type": spec.input_type,
                "manifest": _manifest_payload(source_bundles[spec.input_id].manifest),
            }
            for spec in profile.profile.inputs
        ],
        "arms": [{"arm_id": arm.arm_id, "builder_inputs": list(arm.builder_inputs)} for arm in profile.profile.arms],
        "builder": _builder_descriptor(builder, builder_preflight, config),
        "application_executor": _application_executor_descriptor(application_executor, application_preflight, config),
        "evaluator": _evaluator_descriptor(evaluator_preflight),
    }


def _plan_artifact_dir(path: Path, task: BenchmarkTask, runtime_root: Path) -> Path:
    try:
        return task_layout(path, task.execution.task_id, runtime_root=runtime_root).judge_deliverables
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"could not plan evaluator artifact path for task {task.execution.task_id!r}") from exc


def _planned_paths(items: Sequence[_ScheduleItem], out_root: Path, runtime_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = [
        out_root,
        runtime_root,
        out_root / "experiment-metadata.json",
        out_root / "applications",
        runtime_root / "builds",
        runtime_root / "artifacts",
        runtime_root / "applications",
    ]
    for item in items:
        layout = task_layout(
            item.output_root,
            item.task.execution.task_id,
            runtime_root=item.application_root,
        )
        paths.extend(
            (
                item.output_root,
                item.build_root,
                item.artifact_root,
                item.application_root,
                layout.workspace,
                layout.executor_dir,
                layout.workspace_deliverables,
                item.judge_artifact_root,
            )
        )
    return tuple(paths)


def _validate_planned_paths(paths: Sequence[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        try:
            canonical = path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"could not resolve planned path: {path}") from exc
        if canonical in seen:
            raise ValueError(f"planned output path is duplicated: {path}")
        seen.add(canonical)
        if _path_exists_without_following(path) or _path_exists_without_following(canonical):
            raise FileExistsError(f"planned output path already exists: {path}")


def _make_schedule(
    tasks: Sequence[BenchmarkTask],
    arms: Sequence[ExperimentArm],
    order_seed: int,
    out_root: Path,
    runtime_root: Path,
) -> tuple[_ScheduleItem, ...]:
    pairs = [(task, arm) for task in tasks for arm in arms]
    random.Random(order_seed).shuffle(pairs)
    items: list[_ScheduleItem] = []
    seen_ids: set[str] = set()
    for index, (task, arm) in enumerate(pairs):
        schedule_id = _safe_schedule_id(secrets.token_hex(16))
        if schedule_id in seen_ids:
            raise ValueError("secrets.token_hex returned a colliding schedule ID")
        seen_ids.add(schedule_id)
        output_root = out_root / "applications" / schedule_id
        application_root = runtime_root / "applications" / schedule_id
        items.append(
            _ScheduleItem(
                index=index,
                task=task,
                arm=arm,
                schedule_id=schedule_id,
                output_root=output_root,
                build_root=runtime_root / "builds" / schedule_id,
                artifact_root=runtime_root / "artifacts" / schedule_id,
                application_root=application_root,
                judge_artifact_root=_plan_artifact_dir(output_root, task, application_root),
            )
        )
    return tuple(items)


def _schedule_payload(items: Sequence[_ScheduleItem], task_hashes: Mapping[str, str]) -> list[dict[str, object]]:
    return [
        {
            "index": item.index,
            "task_id": item.task.execution.task_id,
            "task_sha256": task_hashes[item.task.execution.task_id],
            "arm_id": item.arm.arm_id,
            "schedule_id": item.schedule_id,
            "output_root": str(item.output_root),
            "build_root": str(item.build_root),
            "artifact_root": str(item.artifact_root),
            "application_root": str(item.application_root),
        }
        for item in items
    ]


def _initial_metadata(
    profile: LoadedExperimentProfile,
    config: ExperimentRunConfig,
    benchmark: Benchmark,
    source_bundles: Mapping[str, BuilderInputBundle],
    source_roots: Mapping[str, Path],
    tasks: Sequence[BenchmarkTask],
    task_records: Sequence[Mapping[str, object]],
    task_hashes: Mapping[str, str],
    items: Sequence[_ScheduleItem],
    out_root: Path,
    runtime_root: Path,
    started_at: str,
    configuration: Mapping[str, object],
    configuration_sha256: str,
    repository: Mapping[str, object],
    run_fingerprint_sha256: str,
) -> dict[str, object]:
    inputs = []
    for spec in profile.profile.inputs:
        bundle = source_bundles[spec.input_id]
        inputs.append(
            {
                "input_id": spec.input_id,
                "input_type": spec.input_type,
                "source_root": str(source_roots[spec.input_id]),
                "manifest": _manifest_payload(bundle.manifest),
            }
        )
    arms = [{"arm_id": arm.arm_id, "builder_inputs": list(arm.builder_inputs)} for arm in profile.profile.arms]
    entries = [
        {
            "index": item.index,
            "task_id": item.task.execution.task_id,
            "task_sha256": task_hashes[item.task.execution.task_id],
            "arm_id": item.arm.arm_id,
            "schedule_id": item.schedule_id,
            "build": None,
            "application": None,
        }
        for item in items
    ]
    return {
        "schema_version": 2,
        "profile": {
            "id": profile.profile.profile_id,
            "sha256": profile.sha256,
            "source": str(profile.source),
        },
        "benchmark": {
            "id": benchmark.name,
            "revision": getattr(benchmark, "revision", None),
        },
        "run_config": _config_payload(config),
        "inputs": inputs,
        "arms": arms,
        "selected_task_ids": [task.execution.task_id for task in tasks],
        "tasks": list(task_records),
        "schedule": _schedule_payload(items, task_hashes),
        "roots": {
            "output": str(out_root),
            "applications": str(out_root / "applications"),
            "runtime": str(runtime_root),
            "builds": str(runtime_root / "builds"),
            "artifacts": str(runtime_root / "artifacts"),
            "applications_runtime": str(runtime_root / "applications"),
        },
        "entries": entries,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "completed_applications": 0,
        "configuration": dict(configuration),
        "configuration_sha256": configuration_sha256,
        "repository": dict(repository),
        "run_fingerprint_sha256": run_fingerprint_sha256,
    }


def _persist_metadata(path: Path, metadata: dict[str, object], *, status: str, completed: int, finished: bool) -> None:
    metadata["status"] = status
    metadata["completed_applications"] = completed
    if finished:
        metadata["finished_at"] = _now()
    _write_json(path, metadata)


def _entry(metadata: dict[str, object], index: int) -> dict[str, object]:
    entries = metadata.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("experiment metadata entries were corrupted")
    row = entries[index]
    if not isinstance(row, dict):
        raise RuntimeError("experiment metadata entry was corrupted")
    return row


def _artifact_payload(bundle: InterventionBundle) -> dict[str, object]:
    manifest = bundle.manifest
    return {
        "id": _nullable_text(manifest.intervention_id),
        "type": _enum_text(manifest.intervention_type),
        "source_revision": _nullable_text(manifest.source_revision),
        "revision_status": _nullable_text(manifest.revision_status),
        "bundle_sha256": manifest.bundle_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "files": [{"path": item.path, "size": item.size, "sha256": item.sha256} for item in manifest.files],
        "application": {
            "method": _nullable_text(manifest.application.method),
            "target": _nullable_text(manifest.application.target),
        },
    }


def _build_payload(result: BuildResult) -> dict[str, object]:
    status = result.status.value if isinstance(result.status, BuildStatus) else str(result.status)
    phase = None if result.failure_phase is None else result.failure_phase.value
    return {
        "build_run_id": result.build_run_id,
        "task_id": result.task_id,
        "builder": result.builder,
        "status": status,
        "phase": phase,
        "executor_invoked": result.executor_invoked,
        "inputs": [_manifest_payload(manifest) for manifest in result.inputs],
        "execution": execution_record(result.execution) if result.execution is not None else None,
        "artifact": _artifact_payload(result.bundle) if result.bundle is not None else None,
    }


def _application_payload(
    item: _ScheduleItem,
    status: str,
    application_run_id: str | None = None,
) -> dict[str, object]:
    application_run_id = _nullable_text(application_run_id)
    return {
        "output_root": str(item.output_root),
        "runtime_root": str(item.application_root),
        "run_metadata_path": str(item.output_root / "run-metadata.json"),
        "results_path": str(item.output_root / "results.jsonl"),
        "application_run_id": application_run_id,
        "application_run_id_status": "available" if application_run_id is not None else "unavailable",
        "status": status,
    }


def _application_run_id(summary: RunSummary, item: _ScheduleItem) -> str:
    run_ids = getattr(summary, "application_run_ids", None)
    if not isinstance(run_ids, Mapping):
        raise TypeError("application run summary must contain application_run_ids")
    task_id = item.task.execution.task_id
    if set(run_ids) != {task_id}:
        raise ValueError("application run summary returned mismatched application_run_ids")
    value = run_ids.get(task_id)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("application run summary returned an empty application_run_id")
    if value == item.schedule_id:
        raise ValueError("application_run_id must differ from schedule_id")
    return value


def _validate_build_result(result: object, request: BuildRequest, builder: Builder) -> BuildResult:
    result = _require_exact_result(result, BuildResult, "builder result")
    if result.build_run_id != request.build_run_id:
        raise ValueError("builder returned a mismatched build_run_id")
    if result.task_id != request.task.task_id:
        raise ValueError("builder returned a mismatched task_id")
    if result.builder != builder.name:
        raise ValueError("builder returned a mismatched builder identity")
    if tuple(result.inputs) != tuple(item.manifest for item in request.inputs):
        raise ValueError("builder returned mismatched input manifests")
    if result.status is BuildStatus.COMPLETED:
        bundle = result.bundle
        if bundle is None or type(bundle) is not InterventionBundle:
            raise TypeError("completed builder result must contain an exact InterventionBundle")
        if result.execution is None:
            raise ValueError("completed builder result must contain execution evidence")
        artifact_root = _canonical_existing(
            request.artifact_root,
            label="builder artifact root",
            directory=True,
        )
        if bundle.root is None:
            raise ValueError("completed builder result bundle must contain a filesystem root")
        bundle_root = _canonical_existing(bundle.root, label="sealed builder bundle root", directory=True)
        if bundle_root.parent != artifact_root:
            raise ValueError("builder bundle root is outside the assigned artifact root")
    return result


def _create_run_roots(out_root: Path, runtime_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=False)
    runtime_root.mkdir(parents=True, exist_ok=False)
    (out_root / "applications").mkdir(exist_ok=False)
    for category in ("builds", "artifacts", "applications"):
        (runtime_root / category).mkdir(exist_ok=False)
    for path in (
        out_root,
        runtime_root,
        out_root / "applications",
        runtime_root / "builds",
        runtime_root / "artifacts",
        runtime_root / "applications",
    ):
        _canonical_existing(path, label="experiment output directory", directory=True)


def run_builder_experiment(
    profile: LoadedExperimentProfile,
    run_config: ExperimentRunConfig,
    benchmark: Benchmark,
    evaluator: Evaluator,
    builder: Builder,
    application_executor: Executor,
    *,
    source_roots: Mapping[str, Path | str],
    out_dir: Path,
    runtime_root: Path,
) -> ExperimentRunSummary:
    """Run each selected benchmark task with each declared Builder arm."""

    _require_instance(profile, LoadedExperimentProfile, "profile")
    _require_instance(run_config, ExperimentRunConfig, "run_config")
    _require_instance(benchmark, Benchmark, "benchmark")
    _require_instance(evaluator, Evaluator, "evaluator")
    _require_instance(builder, Builder, "builder")
    _require_instance(application_executor, Executor, "application_executor")
    out_root = _canonical_planned_root(out_dir, label="out_dir")
    runtime_root = _canonical_planned_root(runtime_root, label="runtime_root")

    profile_source = _canonical_existing(profile.source, label="profile source", directory=False)
    preliminary_sources: dict[str, Path] = {}
    if not isinstance(source_roots, Mapping):
        raise TypeError("source_roots must be a mapping")
    for input_id, source in source_roots.items():
        preliminary_sources[input_id] = _canonical_existing(
            Path(source),
            label=f"experiment input {input_id!r} source root",
            directory=True,
            require_canonical=False,
        )
    _validate_path_namespace(out_root, runtime_root, profile_source, preliminary_sources)

    evaluator_preflight, application_preflight, builder_preflight = _validate_profile_and_components(
        profile,
        run_config,
        benchmark,
        evaluator,
        builder,
        application_executor,
        out_root,
    )

    # This is the one and only input-loader call for the whole experiment.
    loaded_inputs = load_experiment_inputs(profile.profile, source_roots)
    input_bundles, canonical_sources = _validate_loaded_inputs(profile, source_roots, loaded_inputs)
    _validate_path_namespace(out_root, runtime_root, profile_source, canonical_sources)

    tasks = _validate_tasks(benchmark, run_config.limit)
    schedule = _make_schedule(tasks, profile.profile.arms, run_config.order_seed, out_root, runtime_root)
    _validate_planned_paths(_planned_paths(schedule, out_root, runtime_root))

    # Validate every evaluator plan before the first Builder call.  A concrete
    # artifact destination is supplied because external evaluators require one;
    # the destination remains absent until run_benchmark creates it later.
    first_plan_artifact: dict[str, Path] = {}
    for item in schedule:
        first_plan_artifact.setdefault(item.task.execution.task_id, item.judge_artifact_root)
    for task in tasks:
        evaluator.validate_plan(
            EvaluationPlan(
                task_id=task.execution.task_id,
                task_prompt=task.execution.prompt,
                metadata=task.evaluation,
                candidate_count=1,
                artifact_dir=first_plan_artifact.get(task.execution.task_id),
            )
        )

    task_records = [{"task_id": task.execution.task_id, "task_sha256": task_sha256(task.execution)} for task in tasks]
    task_hashes = {str(record["task_id"]): str(record["task_sha256"]) for record in task_records}
    configuration = _configuration_payload(
        profile,
        run_config,
        benchmark,
        input_bundles,
        builder,
        builder_preflight,
        application_preflight,
        evaluator_preflight,
        application_executor,
    )
    configuration_sha256 = canonical_json_sha256(configuration)
    observed_repository = repository_provenance(Path(__file__).resolve().parents[2])
    repository = {
        "commit": observed_repository.commit,
        "revision_status": observed_repository.revision_status,
        "worktree_status": observed_repository.worktree_status,
    }
    run_fingerprint_sha256 = canonical_json_sha256(
        {
            "configuration_sha256": configuration_sha256,
            "repository": repository,
            "tasks": task_records,
        }
    )

    metadata_path = out_root / "experiment-metadata.json"
    started_at = _now()
    metadata = _initial_metadata(
        profile,
        run_config,
        benchmark,
        input_bundles,
        canonical_sources,
        tasks,
        task_records,
        task_hashes,
        schedule,
        out_root,
        runtime_root,
        started_at,
        configuration,
        configuration_sha256,
        repository,
        run_fingerprint_sha256,
    )

    # No harness-created directory or metadata exists before this point.
    _create_run_roots(out_root, runtime_root)
    _persist_metadata(metadata_path, metadata, status="running", completed=0, finished=False)

    completed = 0
    final_status = "completed"
    try:
        for item in schedule:
            request = BuildRequest(
                build_run_id=item.schedule_id,
                task=item.task.execution,
                inputs=tuple(input_bundles[input_id] for input_id in item.arm.builder_inputs),
                runtime_root=item.build_root,
                artifact_root=item.artifact_root,
                model=run_config.builder_model,
                timeout_seconds=run_config.builder_timeout_seconds,
            )
            try:
                raw_result = builder.build(request)
                build_result = _validate_build_result(raw_result, request, builder)
            except KeyboardInterrupt:
                row = _entry(metadata, item.index)
                row["build"] = {"status": BuildStatus.INTERRUPTED.value, "phase": "interrupt"}
                _persist_metadata(metadata_path, metadata, status="interrupted", completed=completed, finished=True)
                raise
            except Exception:
                row = _entry(metadata, item.index)
                row["build"] = {"status": BuildStatus.FAILED.value, "phase": "exception"}
                _persist_metadata(metadata_path, metadata, status="failed", completed=completed, finished=True)
                raise

            row = _entry(metadata, item.index)
            row["build"] = _build_payload(build_result)
            _persist_metadata(metadata_path, metadata, status="running", completed=completed, finished=False)

            if build_result.status is not BuildStatus.COMPLETED:
                final_status = "interrupted" if build_result.status is BuildStatus.INTERRUPTED else "failed"
                _persist_metadata(metadata_path, metadata, status=final_status, completed=completed, finished=True)
                return ExperimentRunSummary(
                    profile.profile.profile_id,
                    benchmark.name,
                    out_root,
                    runtime_root,
                    final_status,
                    len(tasks),
                    len(profile.profile.arms),
                    completed,
                )

            assert build_result.bundle is not None
            selected_benchmark = _SelectedTaskBenchmark(benchmark, item.task)
            application_run_id: str | None = None
            try:
                intervention = AgentSkillIntervention(build_result.bundle)
                if intervention.source_reference is not None:
                    raise RuntimeError("builder bundle must be handed off without a source reference")
                application_summary = run_benchmark(
                    selected_benchmark,
                    evaluator,
                    application_executor,
                    out_dir=item.output_root,
                    runtime_root=item.application_root,
                    limit=1,
                    model=run_config.application_model,
                    timeout_seconds=run_config.application_timeout_seconds,
                    intervention=intervention,
                )
                application_summary = _require_exact_result(application_summary, RunSummary, "application run summary")
                if (
                    application_summary.out_dir != item.output_root
                    or application_summary.runtime_root != item.application_root
                    or application_summary.benchmark != benchmark.name
                    or application_summary.executor != application_executor.name
                    or application_summary.task_count != 1
                ):
                    raise ValueError("application runner returned mismatched run identity")
                application_run_id = _application_run_id(application_summary, item)
            except KeyboardInterrupt:
                row["application"] = _application_payload(item, "interrupted")
                _persist_metadata(metadata_path, metadata, status="interrupted", completed=completed, finished=True)
                raise
            except RunAbort as exc:
                application_status = "interrupted" if exc.failure.kind is FailureKind.INTERRUPTED else "failed"
                row["application"] = _application_payload(item, application_status)
                _persist_metadata(
                    metadata_path,
                    metadata,
                    status=application_status,
                    completed=completed,
                    finished=True,
                )
                raise
            except Exception:
                row["application"] = _application_payload(item, "failed")
                _persist_metadata(metadata_path, metadata, status="failed", completed=completed, finished=True)
                raise

            application_status = application_summary.status
            if application_status != "completed":
                if application_status not in {"failed", "interrupted"}:
                    row["application"] = _application_payload(item, "failed", application_run_id)
                    _persist_metadata(metadata_path, metadata, status="failed", completed=completed, finished=True)
                    raise ValueError("application runner returned an invalid run status")
                row["application"] = _application_payload(item, application_status, application_run_id)
                _persist_metadata(
                    metadata_path,
                    metadata,
                    status=application_status,
                    completed=completed,
                    finished=True,
                )
                return ExperimentRunSummary(
                    profile.profile.profile_id,
                    benchmark.name,
                    out_root,
                    runtime_root,
                    application_status,
                    len(tasks),
                    len(profile.profile.arms),
                    completed,
                )

            row["application"] = _application_payload(item, "completed", application_run_id)
            completed += 1
            _persist_metadata(metadata_path, metadata, status="running", completed=completed, finished=False)
    except RunAbort as exc:
        final_status = "interrupted" if exc.failure.kind is FailureKind.INTERRUPTED else "failed"
        _persist_metadata(metadata_path, metadata, status=final_status, completed=completed, finished=True)
        raise
    except KeyboardInterrupt:
        final_status = "interrupted"
        _persist_metadata(metadata_path, metadata, status=final_status, completed=completed, finished=True)
        raise
    except Exception:
        final_status = "failed"
        _persist_metadata(metadata_path, metadata, status=final_status, completed=completed, finished=True)
        raise

    _persist_metadata(metadata_path, metadata, status=final_status, completed=completed, finished=True)
    return ExperimentRunSummary(
        profile.profile.profile_id,
        benchmark.name,
        out_root,
        runtime_root,
        final_status,
        len(tasks),
        len(profile.profile.arms),
        completed,
    )


__all__ = ["run_builder_experiment"]
