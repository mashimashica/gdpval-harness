# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for building intervention bundles from benchmark inputs."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

from gdpval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec
from gdpval_harness.interventions.base import InterventionBundle, InterventionFile


class BuildStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    NO_ARTIFACT = "no_artifact"
    INVALID_ARTIFACT = "invalid_artifact"


class BuildFailurePhase(StrEnum):
    PREFLIGHT = "preflight"
    INPUT_VALIDATION = "input-validation"
    EXECUTION = "execution"
    ARTIFACT_VALIDATION = "artifact-validation"
    ARTIFACT_HANDOFF = "artifact-handoff"


def _require_nonempty_text(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_sha256(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _validate_manifest_path(path: str) -> None:
    """Require a relative, normalized POSIX path with no platform escapes."""

    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"builder input manifest path must be strict UTF-8: {path!r}") from exc

    pure = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        not path
        or "\x00" in path
        or "\\" in path
        or pure.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != path
        or unicodedata.normalize("NFC", path) != path
    ):
        raise ValueError(f"builder input manifest path must be relative and normalized: {path!r}")


def _validate_revision(source_revision: object, revision_status: object) -> None:
    if not isinstance(revision_status, str) or revision_status not in {
        "available",
        "unavailable",
        "not-applicable",
    }:
        raise ValueError("revision_status must be available, unavailable, or not-applicable")
    if source_revision is not None and (not isinstance(source_revision, str) or not source_revision.strip()):
        raise ValueError("source_revision must be a non-empty string when supplied")
    if revision_status == "available":
        if source_revision is None:
            raise ValueError("available revision_status requires source_revision")
    elif source_revision is not None:
        raise ValueError(f"{revision_status} revision_status requires source_revision=None")


def _canonical_builder_input_manifest_payload(manifest: "BuilderInputManifest") -> dict[str, object]:
    return {
        "input_id": manifest.input_id,
        "input_type": manifest.input_type,
        "source_revision": manifest.source_revision,
        "revision_status": manifest.revision_status,
        "files": [{"path": item.path, "size": item.size, "sha256": item.sha256} for item in manifest.files],
        "bundle_sha256": manifest.bundle_sha256,
    }


def canonical_builder_input_manifest_bytes(manifest: "BuilderInputManifest") -> bytes:
    """Return canonical JSON bytes for a builder input manifest without its own hash."""

    if not isinstance(manifest, BuilderInputManifest):
        raise TypeError("manifest must be a BuilderInputManifest")
    return json.dumps(
        _canonical_builder_input_manifest_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class BuilderInputManifest:
    input_id: str
    input_type: str
    source_revision: str | None
    revision_status: str
    files: Sequence[InterventionFile]
    bundle_sha256: str
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text("input_id", self.input_id)
        _require_nonempty_text("input_type", self.input_type)
        _validate_revision(self.source_revision, self.revision_status)

        try:
            files = tuple(self.files)
        except TypeError as exc:
            raise TypeError("builder input manifest files must be a sequence of InterventionFile instances") from exc
        if not files:
            raise ValueError("builder input manifest files must be non-empty")
        if any(not isinstance(item, InterventionFile) for item in files):
            raise TypeError("builder input manifest files must be InterventionFile instances")

        previous_paths: dict[str, str] = {}
        previous_components: dict[tuple[tuple[str, ...], str], str] = {}
        canonical_paths: list[tuple[str, ...]] = []
        for item in files:
            _validate_manifest_path(item.path)
            path_parts = PurePosixPath(item.path).parts
            component_keys = tuple(unicodedata.normalize("NFC", part).casefold() for part in path_parts)
            for depth, (part, component_key) in enumerate(zip(path_parts, component_keys)):
                component_prefix = component_keys[:depth]
                component_collision_key = (component_prefix, component_key)
                previous_component = previous_components.get(component_collision_key)
                if previous_component is not None and previous_component != part:
                    raise ValueError(
                        "builder input manifest files contain a Unicode/casefold component collision: "
                        f"{previous_component!r} and {part!r}"
                    )
                previous_components[component_collision_key] = part

            collision_key = "/".join(component_keys)
            previous = previous_paths.get(collision_key)
            if previous is not None:
                raise ValueError(
                    "builder input manifest files contain duplicate or Unicode/casefold-colliding paths: "
                    f"{previous!r} and {item.path!r}"
                )
            previous_paths[collision_key] = item.path
            canonical_paths.append(component_keys)

        for index, path_parts in enumerate(canonical_paths):
            for other_index, other_parts in enumerate(canonical_paths):
                if (
                    index != other_index
                    and len(path_parts) < len(other_parts)
                    and other_parts[: len(path_parts)] == path_parts
                ):
                    raise ValueError(
                        "builder input manifest files cannot use a file path as a directory prefix: "
                        f"{files[index].path!r} and {files[other_index].path!r}"
                    )

        paths = tuple(item.path for item in files)
        if paths != tuple(sorted(paths)):
            raise ValueError("builder input manifest files must be sorted by path")
        object.__setattr__(self, "files", files)

        _validate_sha256("builder input bundle_sha256", self.bundle_sha256)
        expected = hashlib.sha256(canonical_builder_input_manifest_bytes(self)).hexdigest()
        if self.manifest_sha256 is None:
            object.__setattr__(self, "manifest_sha256", expected)
        else:
            _validate_sha256("builder input manifest_sha256", self.manifest_sha256)
            if self.manifest_sha256 != expected:
                raise ValueError("builder input manifest_sha256 does not match canonical manifest contents")


@dataclass(frozen=True)
class BuilderInputBundle:
    root: Path
    manifest: BuilderInputManifest

    def __post_init__(self) -> None:
        try:
            root = Path(self.root)
        except TypeError as exc:
            raise TypeError("builder input bundle root must be path-like") from exc
        if not isinstance(self.manifest, BuilderInputManifest):
            raise TypeError("builder input bundle manifest must be a BuilderInputManifest")
        object.__setattr__(self, "root", root)


@dataclass(frozen=True)
class BuildRequest:
    build_run_id: str
    task: TaskSpec
    inputs: Sequence[BuilderInputBundle]
    runtime_root: Path
    artifact_root: Path
    model: str | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text("build_run_id", self.build_run_id)
        if not isinstance(self.task, TaskSpec):
            raise TypeError("build request task must be a TaskSpec")
        try:
            inputs = tuple(self.inputs)
        except TypeError as exc:
            raise TypeError("build request inputs must be a sequence of BuilderInputBundle instances") from exc
        if any(not isinstance(item, BuilderInputBundle) for item in inputs):
            raise TypeError("build request inputs must be BuilderInputBundle instances")
        input_ids = tuple(item.manifest.input_id for item in inputs)
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("build request input IDs must be unique")

        try:
            runtime_root = Path(self.runtime_root)
            artifact_root = Path(self.artifact_root)
        except TypeError as exc:
            raise TypeError("build request roots must be path-like") from exc

        timeout = self.timeout_seconds
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, Real):
                raise TypeError("build request timeout_seconds must be a positive real number")
            if not math.isfinite(float(timeout)) or timeout <= 0:
                raise ValueError("build request timeout_seconds must be positive")

        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "runtime_root", runtime_root)
        object.__setattr__(self, "artifact_root", artifact_root)


@dataclass(frozen=True)
class BuilderPreflightResult:
    name: str
    ok: bool
    builder_executor: str | None = None
    builder_executor_version: str | None = None
    builder_executor_auth_mode: str | None = None
    details: Sequence[str] = ()
    builder_executor_invocation_mode: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text("builder preflight name", self.name)
        if self.builder_executor_invocation_mode is not None:
            _require_nonempty_text(
                "builder preflight builder_executor_invocation_mode",
                self.builder_executor_invocation_mode,
            )
        try:
            details = tuple(str(detail) for detail in self.details)
        except TypeError as exc:
            raise TypeError("builder preflight details must be a sequence of strings") from exc
        object.__setattr__(self, "details", details)


@dataclass(frozen=True)
class BuildResult:
    build_run_id: str
    task_id: str
    builder: str
    status: BuildStatus
    inputs: Sequence[BuilderInputManifest]
    execution: ExecutionResult | None = None
    bundle: InterventionBundle | None = None
    failure_phase: BuildFailurePhase | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text("build_run_id", self.build_run_id)
        _require_nonempty_text("task_id", self.task_id)
        _require_nonempty_text("builder", self.builder)
        try:
            status = BuildStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid build status: {self.status!r}") from exc
        try:
            inputs = tuple(self.inputs)
        except TypeError as exc:
            raise TypeError("build result inputs must be a sequence of BuilderInputManifest instances") from exc
        if any(not isinstance(item, BuilderInputManifest) for item in inputs):
            raise TypeError("build result inputs must be BuilderInputManifest instances")

        execution = self.execution
        if execution is not None and not isinstance(execution, ExecutionResult):
            raise TypeError("build result execution must be an ExecutionResult")
        if execution is not None and execution.task_id != self.task_id:
            raise ValueError("build result execution task_id must match task_id")

        bundle = self.bundle
        if bundle is not None and not isinstance(bundle, InterventionBundle):
            raise TypeError("build result bundle must be an InterventionBundle")

        failure_phase = self.failure_phase
        if failure_phase is not None:
            try:
                failure_phase = BuildFailurePhase(failure_phase)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid build failure phase: {self.failure_phase!r}") from exc

        if status is BuildStatus.COMPLETED:
            if execution is None or bundle is None:
                raise ValueError("completed build results require execution and bundle")
            if failure_phase is not None:
                raise ValueError("completed build results cannot have a failure phase")
            try:
                execution_status = ExecutionStatus(execution.status)
            except (TypeError, ValueError) as exc:
                raise ValueError("completed build results require a valid execution status") from exc
            if execution_status is not ExecutionStatus.COMPLETED:
                raise ValueError("completed build results require completed execution")
        else:
            if bundle is not None:
                raise ValueError("non-completed build results cannot have a bundle")
            if failure_phase is None:
                raise ValueError("non-completed build results require a failure phase")
            if status in {BuildStatus.NO_ARTIFACT, BuildStatus.INVALID_ARTIFACT} and execution is None:
                raise ValueError(f"{status.value} build results require execution")

            if failure_phase in {BuildFailurePhase.PREFLIGHT, BuildFailurePhase.INPUT_VALIDATION}:
                if status is not BuildStatus.FAILED or execution is not None:
                    raise ValueError("preflight and input-validation failures require failed status without execution")
            elif failure_phase is BuildFailurePhase.EXECUTION:
                if status not in {BuildStatus.FAILED, BuildStatus.TIMED_OUT, BuildStatus.INTERRUPTED}:
                    raise ValueError("execution phase requires failed, timed-out, or interrupted build status")
                try:
                    execution_status = None if execution is None else ExecutionStatus(execution.status)
                except (TypeError, ValueError) as exc:
                    raise ValueError("execution phase requires a valid execution status") from exc
                expected_execution_status = {
                    BuildStatus.TIMED_OUT: ExecutionStatus.TIMED_OUT,
                    BuildStatus.INTERRUPTED: ExecutionStatus.INTERRUPTED,
                }.get(status)
                if (
                    expected_execution_status is not None
                    and execution_status is not None
                    and execution_status is not expected_execution_status
                ):
                    raise ValueError("execution status does not match build status")
            elif failure_phase is BuildFailurePhase.ARTIFACT_VALIDATION:
                artifact_execution_statuses = {
                    BuildStatus.NO_ARTIFACT: ExecutionStatus.NO_DELIVERABLE,
                    BuildStatus.INVALID_ARTIFACT: ExecutionStatus.COMPLETED,
                }
                expected_execution_status = artifact_execution_statuses.get(status)
                if execution is None or expected_execution_status is None:
                    raise ValueError("artifact-validation requires a matching real execution")
                try:
                    execution_status = ExecutionStatus(execution.status)
                except (TypeError, ValueError) as exc:
                    raise ValueError("artifact-validation requires a valid execution status") from exc
                if execution_status is not expected_execution_status:
                    raise ValueError("artifact-validation execution status does not match build status")
            elif failure_phase is BuildFailurePhase.ARTIFACT_HANDOFF:
                if status is not BuildStatus.FAILED or execution is None:
                    raise ValueError("artifact-handoff requires failed status with real execution")
                try:
                    execution_status = ExecutionStatus(execution.status)
                except (TypeError, ValueError) as exc:
                    raise ValueError("artifact-handoff requires a valid execution status") from exc
                if execution_status is not ExecutionStatus.COMPLETED:
                    raise ValueError("artifact-handoff requires completed execution")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "failure_phase", failure_phase)

    @property
    def executor_invoked(self) -> bool:
        return not (
            self.execution is None
            and self.failure_phase in {BuildFailurePhase.PREFLIGHT, BuildFailurePhase.INPUT_VALIDATION}
        )


class Builder(ABC):
    name: str

    @abstractmethod
    def preflight(self) -> BuilderPreflightResult:
        """Check builder readiness without issuing a model request."""

    @abstractmethod
    def build(self, request: BuildRequest) -> BuildResult:
        """Execute a build request."""


__all__ = (
    "BuildFailurePhase",
    "BuildRequest",
    "BuildResult",
    "BuildStatus",
    "Builder",
    "BuilderInputBundle",
    "BuilderInputManifest",
    "BuilderPreflightResult",
    "canonical_builder_input_manifest_bytes",
)
