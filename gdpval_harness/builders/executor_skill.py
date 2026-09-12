# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Agent Skill by running an injected executor in an isolated runtime."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Sequence

from gdpval_harness.builders.artifact import (
    ArtifactHandoffError,
    GeneratedSkillValidationError,
    seal_generated_skill,
)
from gdpval_harness.builders.base import (
    Builder,
    BuilderInputBundle,
    BuilderPreflightResult,
    BuildFailurePhase,
    BuildRequest,
    BuildResult,
    BuildStatus,
)
from gdpval_harness.builders.inputs import (
    stage_builder_inputs,
    verify_staged_builder_inputs,
)
from gdpval_harness.builders.prompt import build_skill_task
from gdpval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
)
from gdpval_harness.interventions.base import InterventionBundle
from gdpval_harness.reasoning import validate_executor_reasoning_effort


_LEGACY_ENVIRONMENT_KEYS = (
    "GDPVAL_CONDITION",
    "GDPVAL_CONDITION_FILE",
    "GDPVAL_CONDITION_APPLIED",
)


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    """Require an existing, canonical, non-symlink directory."""

    try:
        path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an existing canonical non-symlink directory") from exc
    try:
        metadata = path.lstat()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} must be an existing canonical non-symlink directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} must be an existing canonical non-symlink directory") from exc
    if resolved != path:
        raise ValueError(f"{label} must be canonical")
    return resolved


def _validate_planned_absent_root(path: Path, *, label: str) -> Path:
    """Validate one root before any builder directory is created."""

    try:
        path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an absolute canonical path") from exc
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute canonical path")
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} could not be inspected") from exc
    else:
        raise FileExistsError(f"{label} already exists: {path}")

    _canonical_existing_directory(path.parent, label=f"{label} parent")
    try:
        resolved = path.resolve(strict=False)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} must be canonical") from exc
    if resolved != path:
        raise ValueError(f"{label} must be canonical")
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_input_roots(
    inputs: Sequence[BuilderInputBundle], runtime_root: Path, artifact_root: Path
) -> tuple[Path, ...]:
    source_roots: list[Path] = []
    for index, bundle in enumerate(inputs, start=1):
        if not isinstance(bundle, BuilderInputBundle):
            raise TypeError("build request inputs must contain BuilderInputBundle instances")
        source = _canonical_existing_directory(bundle.root, label=f"builder input {index} source root")
        if _paths_overlap(source, runtime_root) or _paths_overlap(source, artifact_root):
            raise ValueError("builder input source roots must be separate from runtime and artifact roots")
        source_roots.append(source)
    return tuple(source_roots)


def _validate_roots(request: BuildRequest) -> tuple[Path, Path, tuple[Path, ...]]:
    runtime_root = _validate_planned_absent_root(request.runtime_root, label="runtime root")
    artifact_root = _validate_planned_absent_root(request.artifact_root, label="artifact root")
    if _paths_overlap(runtime_root, artifact_root):
        raise ValueError("runtime and artifact roots must be separate")
    source_roots = _validate_input_roots(request.inputs, runtime_root, artifact_root)
    return runtime_root, artifact_root, source_roots


def _mkdir_canonical(path: Path, *, label: str) -> Path:
    try:
        path.mkdir()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"could not create {label}") from exc
    return _canonical_existing_directory(path, label=label)


def _create_runtime(runtime_root: Path) -> tuple[Path, Path, Path]:
    """Create and retain the neutral workspace, executor directory, and deliverables."""

    _mkdir_canonical(runtime_root, label="runtime root")
    workspace = _mkdir_canonical(runtime_root / "workspace", label="builder workspace")
    executor_dir = _mkdir_canonical(runtime_root / "executor", label="builder executor directory")
    deliverables_dir = _mkdir_canonical(workspace / "deliverables", label="builder deliverables directory")
    return workspace, executor_dir, deliverables_dir


def _ambient_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in _LEGACY_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    return environment


def _assigned_path_is_canonical(path: object, expected: Path) -> bool:
    try:
        actual = Path(path)  # type: ignore[arg-type]
        metadata = actual.lstat()
        resolved = actual.resolve(strict=True)
    except (OSError, TypeError, ValueError, RuntimeError):
        return False
    return (
        actual == expected
        and resolved == expected
        and not stat.S_ISLNK(metadata.st_mode)
        and stat.S_ISDIR(metadata.st_mode)
    )


class ExecutorSkillBuilder(Builder):
    """Run one executor task that generates a validated, sealed Agent Skill."""

    name = "executor-skill"

    def __init__(self, executor: Executor) -> None:
        if not isinstance(executor, Executor):
            raise TypeError("executor must be an Executor")
        self.executor = executor
        self.reasoning_effort = validate_executor_reasoning_effort(
            executor,
            getattr(executor, "reasoning_effort", None),
        )
        self._preflight_result: BuilderPreflightResult | None = None

    def _executor_invocation_mode(self) -> str | None:
        try:
            value = getattr(self.executor, "invocation_mode", None)
        except Exception:
            return None
        if isinstance(value, str) and value.strip():
            return value
        return None

    def preflight(self) -> BuilderPreflightResult:
        try:
            result = self.executor.preflight()
        except Exception as exc:
            mapped = BuilderPreflightResult(
                name=self.name,
                ok=False,
                builder_executor=self.executor.name,
                builder_executor_version=None,
                builder_executor_auth_mode=None,
                details=(f"executor preflight raised {type(exc).__name__}",),
                builder_executor_invocation_mode=self._executor_invocation_mode(),
            )
            self._preflight_result = mapped
            return mapped

        if not isinstance(result, PreflightResult):
            mapped = BuilderPreflightResult(
                name=self.name,
                ok=False,
                builder_executor=self.executor.name,
                builder_executor_version=None,
                builder_executor_auth_mode=None,
                details=("executor preflight returned an invalid result",),
                builder_executor_invocation_mode=self._executor_invocation_mode(),
            )
            self._preflight_result = mapped
            return mapped

        mapped = BuilderPreflightResult(
            name=self.name,
            ok=result.ok is True and result.executor == self.executor.name,
            builder_executor=result.executor,
            builder_executor_version=result.version,
            builder_executor_auth_mode=result.auth_mode,
            details=result.details,
            builder_executor_invocation_mode=self._executor_invocation_mode(),
        )
        self._preflight_result = mapped
        return mapped

    def build(self, request: BuildRequest) -> BuildResult:
        if not isinstance(request, BuildRequest):
            raise TypeError("request must be a BuildRequest")
        if not isinstance(request.task.task_id, str) or not request.task.task_id.strip():
            raise ValueError("build request task_id must be a non-empty string")

        preflight = self._preflight_result
        if preflight is None:
            preflight = self.preflight()
        if not preflight.ok:
            return self._failure(request, BuildStatus.FAILED, BuildFailurePhase.PREFLIGHT)

        try:
            runtime_root, artifact_root, _ = _validate_roots(request)
            workspace, executor_dir, deliverables_dir = _create_runtime(runtime_root)
            staged = stage_builder_inputs(request.inputs, workspace)
            task = build_skill_task(request.task, [item.target for item in staged])
            execution_request = ExecutionRequest(
                task=task,
                workspace=workspace,
                deliverables_dir=deliverables_dir,
                executor_dir=executor_dir,
                model=request.model,
                timeout_seconds=request.timeout_seconds,
                environment=_ambient_environment(),
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            return self._failure(request, BuildStatus.FAILED, BuildFailurePhase.INPUT_VALIDATION)

        try:
            execution = self.executor.execute(execution_request)
        except KeyboardInterrupt:
            raise
        except Exception:
            return self._failure(request, BuildStatus.FAILED, BuildFailurePhase.EXECUTION)

        if not isinstance(execution, ExecutionResult):
            return self._failure(request, BuildStatus.FAILED, BuildFailurePhase.EXECUTION)
        if execution.task_id != request.task.task_id:
            return self._failure(request, BuildStatus.FAILED, BuildFailurePhase.EXECUTION)
        try:
            status = ExecutionStatus(execution.status)
        except (TypeError, ValueError):
            return self._failure(request, BuildStatus.FAILED, BuildFailurePhase.EXECUTION)
        if execution.executor != self.executor.name:
            return self._result(
                request,
                BuildStatus.FAILED,
                execution=execution,
                phase=BuildFailurePhase.EXECUTION,
            )
        if not _assigned_path_is_canonical(execution.workspace, workspace):
            return self._result(
                request,
                BuildStatus.FAILED,
                execution=execution,
                phase=BuildFailurePhase.EXECUTION,
            )
        if not _assigned_path_is_canonical(execution.deliverables_dir, deliverables_dir):
            return self._result(
                request,
                BuildStatus.FAILED,
                execution=execution,
                phase=BuildFailurePhase.EXECUTION,
            )

        if status is ExecutionStatus.FAILED:
            return self._result(request, BuildStatus.FAILED, execution=execution, phase=BuildFailurePhase.EXECUTION)
        if status is ExecutionStatus.TIMED_OUT:
            return self._result(request, BuildStatus.TIMED_OUT, execution=execution, phase=BuildFailurePhase.EXECUTION)
        if status is ExecutionStatus.INTERRUPTED:
            return self._result(
                request, BuildStatus.INTERRUPTED, execution=execution, phase=BuildFailurePhase.EXECUTION
            )
        if status is ExecutionStatus.NO_DELIVERABLE:
            return self._result(
                request, BuildStatus.NO_ARTIFACT, execution=execution, phase=BuildFailurePhase.ARTIFACT_VALIDATION
            )
        if status is not ExecutionStatus.COMPLETED:
            return self._failure(request, BuildStatus.FAILED, BuildFailurePhase.EXECUTION)

        try:
            verify_staged_builder_inputs(staged, workspace)
        except KeyboardInterrupt:
            raise
        except Exception:
            return self._result(
                request,
                BuildStatus.INVALID_ARTIFACT,
                execution=execution,
                phase=BuildFailurePhase.ARTIFACT_VALIDATION,
            )

        try:
            bundle = seal_generated_skill(deliverables_dir, artifact_root)
        except GeneratedSkillValidationError:
            return self._result(
                request,
                BuildStatus.INVALID_ARTIFACT,
                execution=execution,
                phase=BuildFailurePhase.ARTIFACT_VALIDATION,
            )
        except (ArtifactHandoffError, FileExistsError):
            return self._result(
                request,
                BuildStatus.FAILED,
                execution=execution,
                phase=BuildFailurePhase.ARTIFACT_HANDOFF,
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            return self._result(
                request,
                BuildStatus.FAILED,
                execution=execution,
                phase=BuildFailurePhase.ARTIFACT_HANDOFF,
            )

        if not isinstance(bundle, InterventionBundle):
            return self._result(
                request,
                BuildStatus.FAILED,
                execution=execution,
                phase=BuildFailurePhase.ARTIFACT_HANDOFF,
            )

        return self._result(request, BuildStatus.COMPLETED, execution=execution, bundle=bundle)

    def _result(
        self,
        request: BuildRequest,
        status: BuildStatus,
        *,
        execution: ExecutionResult | None = None,
        bundle: InterventionBundle | None = None,
        phase: BuildFailurePhase | None = None,
    ) -> BuildResult:
        return BuildResult(
            build_run_id=request.build_run_id,
            task_id=request.task.task_id,
            builder=self.name,
            status=status,
            inputs=tuple(item.manifest for item in request.inputs),
            execution=execution,
            bundle=bundle,
            failure_phase=phase,
        )

    def _failure(self, request: BuildRequest, status: BuildStatus, phase: BuildFailurePhase) -> BuildResult:
        return self._result(request, status, phase=phase)


__all__ = ["ExecutorSkillBuilder"]
