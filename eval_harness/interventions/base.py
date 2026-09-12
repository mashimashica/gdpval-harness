# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts and integrity helpers for execution interventions.

Interventions are applied to the executor-facing task and workspace before a
model is invoked.  Their source metadata is kept in this module's immutable
records; it is never added to :class:`~eval_harness.executors.base.TaskSpec`.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Sequence

from eval_harness.executors.base import TaskSpec


class InterventionType(StrEnum):
    NONE = "none"
    PROMPT_OVERLAY = "prompt-overlay"
    FILES = "files"
    AGENT_SKILL = "agent-skill"


def _validate_sha256(label: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")


@dataclass(frozen=True)
class InterventionFile:
    """A logical or materialized file and its exact byte-level evidence.

    Manifest entries use a relative POSIX ``path``.  Applications use the
    same record with ``path`` set to the exact destination path so the runner
    can persist materialization evidence without consulting the source tree.
    """

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("intervention file path must be a non-empty string")
        if self.size < 0:
            raise ValueError("intervention file size must be non-negative")
        _validate_sha256("intervention file sha256", self.sha256)


@dataclass(frozen=True)
class ApplicationMapping:
    """Stable description of where an intervention is applied."""

    method: str
    target: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("intervention application method must be non-empty")
        if self.target is not None and (not isinstance(self.target, str) or not self.target):
            raise ValueError("intervention application target must be non-empty when supplied")


def _manifest_payload(
    *,
    intervention_id: str,
    intervention_type: InterventionType,
    source_revision: str | None,
    revision_status: str,
    files: Sequence[InterventionFile],
    bundle_sha256: str,
    application: ApplicationMapping,
) -> dict[str, object]:
    return {
        "intervention_id": intervention_id,
        "intervention_type": intervention_type.value,
        "source_revision": source_revision,
        "revision_status": revision_status,
        "files": [{"path": item.path, "size": item.size, "sha256": item.sha256} for item in files],
        "bundle_sha256": bundle_sha256,
        "application": {"method": application.method, "target": application.target},
    }


def canonical_manifest_bytes(manifest: "InterventionManifest") -> bytes:
    """Return canonical JSON bytes for a manifest without its own hash."""

    return json.dumps(
        _manifest_payload(
            intervention_id=manifest.intervention_id,
            intervention_type=manifest.intervention_type,
            source_revision=manifest.source_revision,
            revision_status=manifest.revision_status,
            files=manifest.files,
            bundle_sha256=manifest.bundle_sha256,
            application=manifest.application,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class InterventionManifest:
    """Self-contained source snapshot metadata.

    ``manifest_sha256`` is computed over canonical JSON containing every
    field except ``manifest_sha256`` itself.  No source filesystem path is a
    manifest field.
    """

    intervention_id: str
    intervention_type: InterventionType
    source_revision: str | None
    revision_status: str
    files: Sequence[InterventionFile]
    bundle_sha256: str
    application: ApplicationMapping
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.intervention_id:
            raise ValueError("intervention_id must be non-empty")
        intervention_type = InterventionType(self.intervention_type)
        object.__setattr__(self, "intervention_type", intervention_type)
        files = tuple(self.files)
        if any(not isinstance(item, InterventionFile) for item in files):
            raise TypeError("intervention manifest files must be InterventionFile instances")
        object.__setattr__(self, "files", files)
        if not isinstance(self.application, ApplicationMapping):
            raise TypeError("intervention manifest application must be an ApplicationMapping")
        _validate_sha256("intervention bundle_sha256", self.bundle_sha256)
        for item in self.files:
            _validate_manifest_path(item.path)
        expected = hashlib.sha256(canonical_manifest_bytes(self)).hexdigest()
        if self.manifest_sha256 is None:
            object.__setattr__(self, "manifest_sha256", expected)
        elif self.manifest_sha256 != expected:
            raise ValueError("intervention manifest_sha256 does not match canonical manifest contents")


@dataclass(frozen=True)
class InterventionBundle:
    """A preflighted source snapshot and its integrity manifest."""

    root: Path | None
    manifest: InterventionManifest


@dataclass(frozen=True)
class InterventionPreflightResult:
    """Result of checking an intervention before task execution."""

    name: str
    intervention_type: InterventionType
    ok: bool
    bundle: InterventionBundle | None = None
    details: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervention_type", InterventionType(self.intervention_type))
        object.__setattr__(self, "details", tuple(str(detail) for detail in self.details))
        if self.ok and self.bundle is None:
            raise ValueError("successful intervention preflight must include a bundle")


@dataclass(frozen=True)
class InterventionApplication:
    """Evidence returned after applying an intervention to one task."""

    application_run_id: str
    task: TaskSpec
    materialized_files: Sequence[InterventionFile]
    bundle_sha256: str
    manifest_sha256: str
    application: ApplicationMapping

    def __post_init__(self) -> None:
        if not self.application_run_id:
            raise ValueError("application_run_id must be non-empty")
        if not isinstance(self.task, TaskSpec):
            raise TypeError("intervention application task must be a TaskSpec")
        object.__setattr__(self, "materialized_files", tuple(self.materialized_files))
        if any(not isinstance(item, InterventionFile) for item in self.materialized_files):
            raise TypeError("intervention application files must be InterventionFile instances")
        for item in self.materialized_files:
            _validate_manifest_path(item.path)
        if not isinstance(self.application, ApplicationMapping):
            raise TypeError("intervention application mapping must be an ApplicationMapping")
        _validate_sha256("intervention application bundle_sha256", self.bundle_sha256)
        _validate_sha256("intervention application manifest_sha256", self.manifest_sha256)


class Intervention(ABC):
    """The intervention axis applied before an executor consumes a task."""

    name: str
    intervention_type: InterventionType

    @abstractmethod
    def preflight(self) -> InterventionPreflightResult:
        """Validate and snapshot the intervention source without applying it."""

    @abstractmethod
    def validate_task(self, task: TaskSpec) -> None:
        """Validate one executor-facing task before execution."""

    @abstractmethod
    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        """Apply the preflighted intervention and return durable evidence."""


def compute_bundle_sha256(entries: Iterable[tuple[str, bytes]]) -> str:
    """Hash ordered logical paths and exact bytes without ambiguous framing."""

    digest = hashlib.sha256()
    for logical_path, content in entries:
        path_bytes = logical_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def file_evidence(logical_path: str, content: bytes) -> InterventionFile:
    return InterventionFile(path=logical_path, size=len(content), sha256=hashlib.sha256(content).hexdigest())


def _validate_manifest_path(path: str) -> None:
    pure = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        pure.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in path
        or not path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"intervention manifest path must be relative and normalized: {path!r}")


def _ensure_workspace(workspace: Path) -> Path:
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("intervention workspace must be an existing non-symlink directory")
    return workspace


def _ensure_no_symlink_or_non_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"intervention destination parent is a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"intervention destination parent is not a directory: {path}")


def ensure_destination_parents(workspace: Path, logical_paths: Sequence[str]) -> None:
    """Prevalidate all destination parents before any destination is created."""

    workspace = _ensure_workspace(workspace)
    parents: set[Path] = set()
    for logical_path in logical_paths:
        _validate_manifest_path(logical_path)
        destination = workspace / PurePosixPath(logical_path)
        if destination.parent == workspace:
            continue
        current = workspace
        for part in PurePosixPath(logical_path).parts[:-1]:
            current = current / part
            parents.add(current)
    for parent in sorted(parents, key=lambda path: len(path.parts)):
        _ensure_no_symlink_or_non_directory(parent)
    for logical_path in logical_paths:
        destination = workspace / PurePosixPath(logical_path)
        if destination.is_symlink() or destination.exists():
            raise FileExistsError(f"intervention destination already exists: {destination}")


def ensure_source_output_separation(bundle: InterventionBundle, output: Path) -> None:
    """Reject source/output paths that overlap after symlink-aware resolution.

    ``output`` may be a planned path that does not exist yet, so it is resolved
    with ``strict=False`` while existing source components are resolved
    strictly.  This check is intentionally independent of any intervention's
    destination layout and can be used before an output directory is created.
    """

    if not isinstance(bundle, InterventionBundle) or not isinstance(bundle.root, Path):
        raise ValueError("intervention bundle must have a filesystem source root")
    source = bundle.root
    if source.is_symlink():
        raise ValueError("intervention source root must not be a symlink")
    source_resolved = source.resolve(strict=True)
    output_resolved = Path(output).resolve(strict=False)
    if (
        source_resolved == output_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError("intervention source and output paths must be separate")


def fsync_directory(path: Path) -> None:
    """Fsync a directory where the host platform exposes directory handles."""

    flags = getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, os.O_RDONLY | flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def revision_fields(source_revision: str | None, *, applicable: bool) -> tuple[str | None, str]:
    if not applicable:
        return None, "not-applicable"
    if source_revision is None:
        return None, "unavailable"
    if not source_revision:
        raise ValueError("source_revision must be non-empty when supplied")
    return source_revision, "available"


__all__ = [
    "ApplicationMapping",
    "Intervention",
    "InterventionApplication",
    "InterventionBundle",
    "InterventionFile",
    "InterventionManifest",
    "InterventionPreflightResult",
    "InterventionType",
    "canonical_manifest_bytes",
    "compute_bundle_sha256",
    "ensure_destination_parents",
    "ensure_source_output_separation",
    "file_evidence",
    "fsync_directory",
    "revision_fields",
]
