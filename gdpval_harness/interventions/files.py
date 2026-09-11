# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot and safely materialize an intervention file bundle."""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from pathlib import Path, PurePosixPath

from gdpval_harness.executors.base import TaskSpec
from gdpval_harness.interventions.base import (
    ApplicationMapping,
    Intervention,
    InterventionApplication,
    InterventionBundle,
    InterventionFile,
    InterventionManifest,
    InterventionPreflightResult,
    InterventionType,
    compute_bundle_sha256,
    ensure_destination_parents,
    file_evidence,
    fsync_directory,
    revision_fields,
)


_RESERVED_TOP_LEVEL = frozenset({"deliverables", "reference_files", ".gdpval", ".cursor", ".claude", ".agents"})


def _canonical_collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _validate_logical_path(path: str) -> None:
    pure = PurePosixPath(path)
    if "\\" in path or pure.is_absolute() or not path or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"intervention source contains an unsafe relative path: {path!r}")
    if pure.parts[0].casefold() in _RESERVED_TOP_LEVEL:
        raise ValueError(f"intervention source uses reserved top-level path: {pure.parts[0]!r}")


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError(f"intervention source contains a symlink: {path.name!r}")
    source_stat = path.lstat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"intervention source contains a non-regular file: {path.name!r}")
    return path.read_bytes()


def _scan_source(root: Path) -> tuple[tuple[str, bytes], ...]:
    if root.is_symlink():
        raise ValueError("intervention file source root must not be a symlink")
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("intervention file source root must be a directory")

    entries: list[tuple[str, bytes]] = []
    collision_keys: dict[str, str] = {}

    def visit(directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise ValueError(f"could not read intervention source directory: {exc}") from exc
        for child in children:
            relative = child.relative_to(root).as_posix()
            _validate_logical_path(relative)
            collision_key = _canonical_collision_key(relative)
            previous = collision_keys.get(collision_key)
            if previous is not None and previous != relative:
                raise ValueError(f"intervention source has a Unicode/casefold path collision: {previous!r}")
            collision_keys[collision_key] = relative
            if child.is_symlink():
                raise ValueError(f"intervention source contains a symlink: {relative!r}")
            child_stat = child.lstat()
            if stat.S_ISDIR(child_stat.st_mode):
                visit(child)
            elif stat.S_ISREG(child_stat.st_mode):
                entries.append((relative, _read_regular_file(child)))
            else:
                raise ValueError(f"intervention source contains a special file: {relative!r}")

    visit(root)
    if not entries:
        raise ValueError("intervention file source must contain at least one regular file")
    return tuple(sorted(entries, key=lambda entry: entry[0]))


def _snapshot_evidence(entries: tuple[tuple[str, bytes], ...]) -> tuple[InterventionFile, ...]:
    return tuple(file_evidence(path, content) for path, content in entries)


def _same_snapshot(bundle: InterventionBundle, entries: tuple[tuple[str, bytes], ...]) -> bool:
    manifest = bundle.manifest
    if compute_bundle_sha256(entries) != manifest.bundle_sha256:
        return False
    return _snapshot_evidence(entries) == tuple(manifest.files)


def _validate_destination_collisions(workspace: Path, logical_paths: tuple[str, ...]) -> None:
    """Reject existing names that collide under Unicode normalization/casefold."""

    for logical_path in logical_paths:
        current = workspace
        for part in PurePosixPath(logical_path).parts:
            if not current.is_dir():
                break
            key = _canonical_collision_key(part)
            matches = [entry for entry in current.iterdir() if _canonical_collision_key(entry.name) == key]
            if len(matches) > 1 or (matches and matches[0].name != part):
                raise ValueError(f"intervention destination has a Unicode/casefold path collision: {part!r}")
            if not matches:
                break
            current = matches[0]


def _mkdir_missing_parents(workspace: Path, logical_paths: tuple[str, ...], created: list[Path]) -> None:
    parents: set[Path] = set()
    for logical_path in logical_paths:
        current = workspace
        for part in PurePosixPath(logical_path).parts[:-1]:
            current = current / part
            parents.add(current)
    for parent in sorted(parents, key=lambda path: len(path.parts)):
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError(f"intervention destination parent is not a directory: {parent}")
            continue
        try:
            parent.mkdir()
        except FileExistsError:
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError(f"intervention destination parent is not a directory: {parent}") from None
        else:
            created.append(parent)


def _copy_exclusive(destination: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor != -1:
            os.close(descriptor)


class FilesIntervention(Intervention):
    """Copy a validated source directory into the task workspace."""

    name = "files"
    intervention_type = InterventionType.FILES

    def __init__(
        self,
        source: Path,
        *,
        intervention_id: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        self.source = Path(source)
        self.intervention_id = intervention_id or self.name
        self.source_revision = source_revision
        self._bundle: InterventionBundle | None = None

    def preflight(self) -> InterventionPreflightResult:
        self._bundle = None
        try:
            entries = _scan_source(self.source)
            source_revision, revision_status = revision_fields(self.source_revision, applicable=True)
            application = ApplicationMapping(method="workspace-files", target=".")
            manifest_files = _snapshot_evidence(entries)
            manifest = InterventionManifest(
                intervention_id=self.intervention_id,
                intervention_type=self.intervention_type,
                source_revision=source_revision,
                revision_status=revision_status,
                files=manifest_files,
                bundle_sha256=compute_bundle_sha256(entries),
                application=application,
            )
            self._bundle = InterventionBundle(root=self.source, manifest=manifest)
        except (OSError, ValueError) as exc:
            return InterventionPreflightResult(
                name=self.name,
                intervention_type=self.intervention_type,
                ok=False,
                details=(f"file intervention preflight failed: {exc}",),
            )
        return InterventionPreflightResult(
            name=self.name,
            intervention_type=self.intervention_type,
            ok=True,
            bundle=self._bundle,
            details=("file intervention source is ready",),
        )

    def validate_task(self, task: TaskSpec) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError("intervention task must be a TaskSpec")

    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        self.validate_task(task)
        bundle = self._require_bundle()
        entries = _scan_source(self.source)
        if not _same_snapshot(bundle, entries):
            raise RuntimeError("intervention file source changed after preflight")
        workspace = Path(workspace)
        logical_paths = tuple(path for path, _ in entries)
        ensure_destination_parents(workspace, logical_paths)
        _validate_destination_collisions(workspace, logical_paths)

        created_files: list[Path] = []
        created_dirs: list[Path] = []
        try:
            _mkdir_missing_parents(workspace, logical_paths, created_dirs)
            for logical_path, content in entries:
                destination = workspace / PurePosixPath(logical_path)
                _copy_exclusive(destination, content)
                created_files.append(destination)
                if destination.is_symlink() or destination.read_bytes() != content:
                    raise RuntimeError(f"intervention destination verification failed: {logical_path!r}")
                if hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(content).hexdigest():
                    raise RuntimeError(f"intervention destination hash verification failed: {logical_path!r}")
                fsync_directory(destination.parent)
            fsync_directory(workspace)
        except Exception:
            for destination in reversed(created_files):
                try:
                    destination.unlink()
                except OSError:
                    pass
            for directory in sorted(created_dirs, key=lambda path: len(path.parts), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            fsync_directory(workspace)
            raise

        materialized = tuple(file_evidence(path, content) for path, content in entries)
        return InterventionApplication(
            application_run_id=application_run_id,
            task=task,
            materialized_files=materialized,
            bundle_sha256=bundle.manifest.bundle_sha256,
            manifest_sha256=bundle.manifest.manifest_sha256 or "",
            application=bundle.manifest.application,
        )

    def _require_bundle(self) -> InterventionBundle:
        if self._bundle is None:
            raise RuntimeError("successful intervention preflight is required before apply")
        return self._bundle


__all__ = ["FilesIntervention"]
