# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safely seal a generated Agent Skill into an artifact directory.

The builder writes its generated files into a deliverables directory.  This
module gives that directory one narrow handoff path: exactly one validated
Agent Skill is copied to a newly-created artifact root, using the skill
loader's manifest as the complete allowlist.
"""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Iterable

from gdpval_harness.interventions import (
    InterventionBundle,
    InterventionFile,
    ensure_source_output_separation,
    load_agent_skill_bundle,
)
from gdpval_harness.interventions.base import fsync_directory


class GeneratedSkillValidationError(ValueError):
    """The generated deliverables or planned artifact destination is invalid."""


class ArtifactHandoffError(RuntimeError):
    """A validated Skill could not be durably copied or reloaded."""


def _existing_canonical_directory(path: Path, *, label: str) -> Path:
    """Return an existing canonical real directory or fail closed."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} must be an existing canonical directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must be an existing canonical directory") from exc
    if resolved != path:
        raise ValueError(f"{label} must be canonical")
    return resolved


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"could not inspect artifact root: {path}") from exc
    return True


def _validate_absent_artifact_root(path: Path) -> None:
    """Validate the planned root without creating or following it."""

    if _path_exists_without_following(path):
        if path.is_symlink():
            raise FileExistsError(f"artifact root already exists: {path}")
        raise FileExistsError(f"artifact root already exists: {path}")
    _existing_canonical_directory(path.parent, label="artifact root parent")


def _validate_manifest_paths(files: Iterable[InterventionFile]) -> tuple[InterventionFile, ...]:
    """Apply the destination path checks before any output is created.

    The accepted PR3 loader already performs these checks.  Repeating the
    inexpensive structural check here protects the copy routine if a bundle
    object is ever supplied by a caller other than that loader.
    """

    result = tuple(files)
    if not result:
        raise ValueError("Agent Skill manifest must contain at least one file")
    seen: dict[str, str] = {}
    for item in result:
        if not isinstance(item, InterventionFile):
            raise TypeError("Agent Skill manifest files must be InterventionFile instances")
        path = item.path
        pure = PurePosixPath(path)
        if (
            not path
            or "\\" in path
            or pure.is_absolute()
            or pure.as_posix() != path
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(f"Agent Skill manifest contains an unsafe relative path: {path!r}")
        key = unicodedata.normalize("NFC", path).casefold()
        previous = seen.get(key)
        if previous is not None and previous != path:
            raise ValueError(f"Agent Skill manifest has a Unicode/casefold path collision: {previous!r}")
        if previous is not None:
            raise ValueError(f"Agent Skill manifest contains a duplicate path: {path!r}")
        seen[key] = path
    return result


def _source_file(root: Path, item: InterventionFile) -> bytes:
    """Read and verify one manifest file without following source links."""

    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise RuntimeError("Agent Skill source changed after validation") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("Agent Skill source changed after validation")

    current = root
    pure = PurePosixPath(item.path)
    try:
        for part in pure.parts[:-1]:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError("Agent Skill source changed after validation")

        source_file = current / pure.parts[-1]
        metadata = source_file.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Agent Skill source changed after validation")
        content = source_file.read_bytes()
        after = source_file.lstat()
    except (OSError, IndexError) as exc:
        raise RuntimeError("Agent Skill source changed after validation") from exc

    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or after.st_size != len(content)
        or len(content) != item.size
        or hashlib.sha256(content).hexdigest() != item.sha256
    ):
        raise RuntimeError(f"Agent Skill source changed after validation: {item.path!r}")
    return content


def _source_entries(bundle: InterventionBundle, files: tuple[InterventionFile, ...]) -> tuple[bytes, ...]:
    if not isinstance(bundle.root, Path):
        raise ValueError("Agent Skill bundle must have a filesystem root")
    return tuple(_source_file(bundle.root, item) for item in files)


def _copy_exclusive(destination: Path, content: bytes) -> None:
    """Write exact bytes to a new regular file and durably flush the file."""

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


def _mkdir_exclusive(path: Path, created: list[Path]) -> None:
    """Create one directory, tracking it before any subsequent validation."""

    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"artifact destination already exists: {path}") from None
    created.append(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"artifact destination could not be verified: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"artifact destination is not a regular directory: {path}")
    fsync_directory(path)
    fsync_directory(path.parent)


def _destination_directories(root: Path, files: tuple[InterventionFile, ...]) -> tuple[Path, ...]:
    directories: list[Path] = []
    seen: set[Path] = set()
    for item in files:
        current = root
        for part in PurePosixPath(item.path).parts[:-1]:
            current = current / part
            if current not in seen:
                seen.add(current)
                directories.append(current)
    return tuple(directories)


def _verify_destination(path: Path, item: InterventionFile, content: bytes) -> None:
    try:
        metadata = path.lstat()
        actual = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"artifact destination could not be verified: {item.path!r}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != len(content)
        or actual != content
        or len(actual) != item.size
        or hashlib.sha256(actual).hexdigest() != item.sha256
    ):
        raise RuntimeError(f"artifact destination verification failed: {item.path!r}")


def _cleanup(created_files: list[Path], created_directories: list[Path], artifact_root: Path) -> None:
    """Remove only paths successfully created by this invocation."""

    for path in reversed(created_files):
        try:
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                path.unlink()
        except OSError:
            pass
    for path in reversed(created_directories):
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        fsync_directory(artifact_root.parent)
    except OSError:
        pass


def _same_sealed_manifest(source: InterventionBundle, sealed: InterventionBundle) -> bool:
    source_manifest = source.manifest
    sealed_manifest = sealed.manifest
    return (
        sealed_manifest.files == source_manifest.files
        and sealed_manifest.bundle_sha256 == source_manifest.bundle_sha256
        and sealed_manifest.manifest_sha256 == source_manifest.manifest_sha256
        and sealed_manifest.intervention_type == source_manifest.intervention_type
        and sealed_manifest.intervention_id == source_manifest.intervention_id
        and sealed_manifest.source_revision == source_manifest.source_revision
        and sealed_manifest.revision_status == source_manifest.revision_status
        and sealed_manifest.application == source_manifest.application
    )


def _validate_sealed_bundle(
    source: InterventionBundle,
    sealed: InterventionBundle,
    destination_root: Path,
) -> None:
    if not isinstance(sealed, InterventionBundle) or not isinstance(sealed.root, Path):
        raise ValueError("sealed Agent Skill loader result is not a filesystem bundle")
    try:
        sealed_metadata = sealed.root.lstat()
        sealed_canonical = sealed.root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("sealed Agent Skill root is unavailable") from exc
    if (
        stat.S_ISLNK(sealed_metadata.st_mode)
        or not stat.S_ISDIR(sealed_metadata.st_mode)
        or sealed_canonical != sealed.root
        or sealed_canonical != destination_root
    ):
        raise ValueError("sealed Agent Skill root is not the canonical artifact directory")
    if not _same_sealed_manifest(source, sealed):
        raise ValueError("sealed Agent Skill manifest does not match source bundle")


def seal_generated_skill(deliverables_dir: Path, artifact_root: Path) -> InterventionBundle:
    """Validate and seal the one generated Agent Skill in ``deliverables_dir``.

    The destination root is created only after the source shape, source
    manifest, source bytes, artifact parent, and source/output separation have
    all passed validation.  Every destination path is then created exclusively
    and removed on failure if this call created it.
    """

    try:
        deliverables = _existing_canonical_directory(Path(deliverables_dir), label="deliverables directory")
        artifact = Path(artifact_root)

        try:
            immediate = tuple(sorted(deliverables.iterdir(), key=lambda path: path.name))
        except OSError as exc:
            raise ValueError("could not inspect deliverables directory") from exc
        if len(immediate) != 1:
            raise ValueError("deliverables directory must contain exactly one immediate entry")
        skill_candidate = immediate[0]
        try:
            skill_metadata = skill_candidate.lstat()
        except OSError as exc:
            raise ValueError("deliverables entry is unavailable") from exc
        if stat.S_ISLNK(skill_metadata.st_mode) or not stat.S_ISDIR(skill_metadata.st_mode):
            raise ValueError("deliverables entry must be a real directory")

        # Deliberately pass no intervention_id or source_revision.  The validated
        # loader therefore derives the skill name and the complete allowlist from
        # the generated directory itself.
        source_bundle = load_agent_skill_bundle(skill_candidate)
        if not isinstance(source_bundle, InterventionBundle) or not isinstance(source_bundle.root, Path):
            raise ValueError("Agent Skill loader did not return a filesystem bundle")
        source_root = source_bundle.root
        try:
            source_canonical = source_root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Agent Skill source root is unavailable") from exc
        if source_root != source_canonical or source_root != skill_candidate:
            raise ValueError("Agent Skill loader returned a non-canonical source root")
        skill_name = source_root.name
        expected_target = f".gdpval/interventions/{skill_name}/SKILL.md"
        if source_bundle.manifest.application.target != expected_target:
            raise ValueError("Agent Skill manifest target does not match its validated name")
        files = _validate_manifest_paths(source_bundle.manifest.files)

        _validate_absent_artifact_root(artifact)
        ensure_source_output_separation(source_bundle, artifact)

        # Snapshot every allowlisted source file before creating any artifact path.
        initial_entries = _source_entries(source_bundle, files)
    except FileExistsError:
        # Preserve the standard no-overwrite signal for a pre-existing root.
        raise
    except GeneratedSkillValidationError:
        raise
    except Exception as exc:
        raise GeneratedSkillValidationError(str(exc)) from exc

    destination_root = artifact / skill_name
    created_files: list[Path] = []
    created_directories: list[Path] = []
    try:
        _mkdir_exclusive(artifact, created_directories)
        _mkdir_exclusive(destination_root, created_directories)
        for directory in _destination_directories(destination_root, files):
            _mkdir_exclusive(directory, created_directories)

        for item, initial_content in zip(files, initial_entries, strict=True):
            # Re-read immediately before every write.  This also rechecks all
            # parent directory types, so a source directory replacement cannot
            # redirect the copy through a symlink.
            content = _source_file(source_root, item)
            if content != initial_content:
                raise RuntimeError(f"Agent Skill source changed after validation: {item.path!r}")
            destination = destination_root / PurePosixPath(item.path)
            _copy_exclusive(destination, content)
            created_files.append(destination)
            _verify_destination(destination, item, content)
            fsync_directory(destination.parent)

        for directory in reversed(created_directories):
            fsync_directory(directory)
        fsync_directory(artifact.parent)

        sealed_bundle = load_agent_skill_bundle(destination_root)
        _validate_sealed_bundle(source_bundle, sealed_bundle, destination_root)

        # Confirm that the source still describes the same bytes after the
        # sealed tree has been reloaded.  The returned object is always the
        # reloaded artifact bundle, never the source bundle.
        final_entries = _source_entries(source_bundle, files)
        if final_entries != initial_entries:
            raise RuntimeError("Agent Skill source changed after sealing")
        return sealed_bundle
    except BaseException as exc:
        _cleanup(created_files, created_directories, artifact)
        if isinstance(exc, Exception):
            raise ArtifactHandoffError(str(exc)) from exc
        raise


__all__ = ["ArtifactHandoffError", "GeneratedSkillValidationError", "seal_generated_skill"]
