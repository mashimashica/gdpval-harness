# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and stage explicit builder creation inputs.

Builder inputs are intentionally narrower than a general directory snapshot.  A caller must
name every file that is allowed to enter a build.  The resulting manifest contains only logical
relative paths and byte-level evidence; the source root remains an implementation detail of the
in-memory bundle and is never part of its hash.
"""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

from gdpval_harness.builders.base import (
    BuilderInputBundle,
    BuilderInputManifest,
    canonical_builder_input_manifest_bytes,
)
from gdpval_harness.interventions.base import (
    InterventionFile,
    compute_bundle_sha256,
    file_evidence,
    fsync_directory,
    revision_fields,
)


_TARGET_PREFIX = "reference_files/builder-inputs"
_TARGET_PATTERN_PREFIX = f"{_TARGET_PREFIX}/input-"
_MAX_TARGET_NUMBER = 999


@dataclass(frozen=True)
class StagedBuilderInput:
    """Evidence for one neutral builder-input directory in a build workspace."""

    input_id: str
    target: str
    materialized_files: Sequence[InterventionFile]

    def __post_init__(self) -> None:
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("staged builder input input_id must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("staged builder input target must be a non-empty string")
        try:
            materialized_files = tuple(self.materialized_files)
        except TypeError as exc:
            raise TypeError("staged builder input materialized_files must be a sequence") from exc
        if any(not isinstance(item, InterventionFile) for item in materialized_files):
            raise TypeError("staged builder input materialized_files must contain InterventionFile instances")
        object.__setattr__(self, "materialized_files", materialized_files)


def _validate_relative_path(path: object, *, label: str) -> str:
    """Validate one strict, normalized, relative POSIX path."""

    if not isinstance(path, str):
        raise TypeError(f"{label} must be a string")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be strict UTF-8: {path!r}") from exc

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
        raise ValueError(f"{label} must be a relative normalized POSIX path: {path!r}")
    return path


def _collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _validate_path_component_collisions(paths: Sequence[str], *, label: str) -> None:
    """Reject casefold/NFC collisions among every logical path prefix."""

    seen: dict[tuple[str, ...], tuple[str, ...]] = {}
    for path in paths:
        parts = PurePosixPath(path).parts
        for count in range(1, len(parts) + 1):
            actual = parts[:count]
            key = tuple(_collision_key(part) for part in actual)
            previous = seen.get(key)
            if previous is not None and previous != actual:
                raise ValueError(
                    f"{label} contains a Unicode/casefold-colliding path component: "
                    f"{'/'.join(previous)!r} and {'/'.join(actual)!r}"
                )
            seen[key] = actual


def _validate_allowlist(allowed_files: Sequence[str]) -> tuple[str, ...]:
    if isinstance(allowed_files, (str, bytes)):
        raise TypeError("allowed_files must be a non-empty sequence of paths")
    try:
        paths = tuple(allowed_files)
    except TypeError as exc:
        raise TypeError("allowed_files must be a non-empty sequence of paths") from exc
    if not paths:
        raise ValueError("allowed_files must be a non-empty explicit allowlist")

    seen: dict[str, str] = {}
    for path in paths:
        path = _validate_relative_path(path, label="builder input allowlist path")
        key = _collision_key(path)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "builder input allowlist contains duplicate or Unicode/casefold-colliding paths: "
                f"{previous!r} and {path!r}"
            )
        seen[key] = path
    _validate_path_component_collisions(paths, label="builder input allowlist")
    return tuple(sorted(paths))


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    """Return an existing directory with a canonical, non-symlink final component."""

    try:
        metadata = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} must be an existing canonical non-symlink directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
        resolved_metadata = resolved.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} must be an existing canonical non-symlink directory") from exc
    if stat.S_ISLNK(resolved_metadata.st_mode) or not stat.S_ISDIR(resolved_metadata.st_mode):
        raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    if resolved != path:
        raise ValueError(f"{label} must be canonical")
    return resolved


def _load_source_root(source: Path | str) -> Path:
    try:
        source_path = Path(source)
        metadata = source_path.lstat()
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("builder input source root must be an existing non-symlink directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("builder input source root must be an existing non-symlink directory")
    try:
        root = source_path.resolve(strict=True)
        root_metadata = root.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError("builder input source root must be an existing non-symlink directory") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("builder input source root must be an existing non-symlink directory")
    return root


def _regular_file_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return metadata


def _directory_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return metadata


def _read_regular_file(path: Path, *, label: str) -> bytes:
    """Read one regular file through an exclusive no-follow descriptor."""

    before = _regular_file_metadata(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        descriptor_metadata = os.fstat(descriptor)
        if stat.S_ISLNK(descriptor_metadata.st_mode) or not stat.S_ISREG(descriptor_metadata.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read()
            after_descriptor = os.fstat(handle.fileno())
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read {label}") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)

    try:
        after = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} changed while reading") from exc
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or after_descriptor.st_size != len(content)
        or after.st_size != len(content)
    ):
        raise ValueError(f"{label} changed while reading")
    return content


def _read_allowlisted_file(root: Path, logical_path: str) -> bytes:
    """Read an allowlisted path while checking each traversed component."""

    _directory_metadata(root, label="builder input source root")
    pure = PurePosixPath(logical_path)
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        _directory_metadata(current, label=f"builder input path component {logical_path!r}")
    final = current / pure.parts[-1]
    return _read_regular_file(final, label=f"builder input file {logical_path!r}")


def _validate_manifest_files(files: Sequence[InterventionFile]) -> tuple[InterventionFile, ...]:
    try:
        normalized = tuple(files)
    except TypeError as exc:
        raise TypeError("builder input manifest files must be a sequence") from exc
    if not normalized:
        raise ValueError("builder input manifest files must be non-empty")
    seen: dict[str, str] = {}
    paths: list[str] = []
    for item in normalized:
        if not isinstance(item, InterventionFile):
            raise TypeError("builder input manifest files must contain InterventionFile instances")
        path = _validate_relative_path(item.path, label="builder input manifest path")
        key = _collision_key(path)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(f"builder input manifest paths collide: {previous!r} and {path!r}")
        seen[key] = path
        paths.append(path)
    _validate_path_component_collisions(paths, label="builder input manifest")
    if tuple(paths) != tuple(sorted(paths)):
        raise ValueError("builder input manifest files must be sorted by path")

    path_set = set(paths)
    for path in paths:
        parts = PurePosixPath(path).parts
        if any("/".join(parts[:index]) in path_set for index in range(1, len(parts))):
            raise ValueError("builder input manifest contains a file path that is also a directory")
    return normalized


def _manifest_hash(manifest: BuilderInputManifest) -> str:
    try:
        canonical = canonical_builder_input_manifest_bytes(manifest)
    except (TypeError, ValueError) as exc:
        raise ValueError("builder input manifest is not canonical") from exc
    return hashlib.sha256(canonical).hexdigest()


def _validate_manifest_hash(manifest: BuilderInputManifest) -> None:
    if not isinstance(manifest.manifest_sha256, str) or manifest.manifest_sha256 != _manifest_hash(manifest):
        raise ValueError("builder input manifest_sha256 does not match canonical manifest contents")


def _snapshot_manifest_files(root: Path, manifest: BuilderInputManifest) -> tuple[tuple[str, bytes], ...]:
    files = _validate_manifest_files(manifest.files)
    entries: list[tuple[str, bytes]] = []
    for item in files:
        try:
            content = _read_allowlisted_file(root, item.path)
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError(f"builder input source changed after validation: {item.path!r}") from exc
        if (
            len(content) != item.size
            or hashlib.sha256(content).hexdigest() != item.sha256
        ):
            raise RuntimeError(f"builder input source changed after validation: {item.path!r}")
        entries.append((item.path, content))

    bundle_sha256 = compute_bundle_sha256(entries)
    if bundle_sha256 != manifest.bundle_sha256:
        raise RuntimeError("builder input source changed after validation: bundle hash mismatch")
    return tuple(entries)


def load_builder_input_bundle(
    source: Path | str,
    *,
    input_id: str,
    input_type: str,
    allowed_files: Sequence[str],
    source_revision: str | None = None,
) -> BuilderInputBundle:
    """Load an explicit allowlist of creation-time files into an immutable bundle."""

    paths = _validate_allowlist(allowed_files)
    root = _load_source_root(source)
    entries = tuple((path, _read_allowlisted_file(root, path)) for path in paths)
    manifest_files = tuple(file_evidence(path, content) for path, content in entries)
    revision, revision_status = revision_fields(source_revision, applicable=True)
    manifest = BuilderInputManifest(
        input_id=input_id,
        input_type=input_type,
        source_revision=revision,
        revision_status=revision_status,
        files=manifest_files,
        bundle_sha256=compute_bundle_sha256(entries),
    )
    return BuilderInputBundle(root=root, manifest=manifest)


def _target_for_number(number: int) -> str:
    if number < 1 or number > _MAX_TARGET_NUMBER:
        raise ValueError("builder input count must be between 1 and 999")
    return f"{_TARGET_PATTERN_PREFIX}{number:03d}"


def _validate_target(target: str, *, expected: str | None = None) -> None:
    if expected is not None and target != expected:
        raise ValueError("staged builder input targets must use contiguous neutral numbering")
    if not target.startswith(_TARGET_PATTERN_PREFIX):
        raise ValueError("staged builder input target must be a neutral builder-input directory")
    suffix = target[len(_TARGET_PATTERN_PREFIX) :]
    if len(suffix) != 3 or not suffix.isdecimal() or not suffix.isascii() or int(suffix) == 0:
        raise ValueError("staged builder input target must use a three-digit input number")
    _validate_relative_path(target, label="staged builder input target")


def _validate_bundle_for_staging(bundle: BuilderInputBundle) -> tuple[Path, BuilderInputManifest]:
    if not isinstance(bundle, BuilderInputBundle):
        raise TypeError("builder inputs must be BuilderInputBundle instances")
    if not isinstance(bundle.root, Path):
        raise TypeError("builder input bundle root must be a Path")
    root = _canonical_existing_directory(bundle.root, label="builder input source root")
    manifest = bundle.manifest
    if not isinstance(manifest, BuilderInputManifest):
        raise TypeError("builder input bundle manifest must be a BuilderInputManifest")
    _validate_manifest_files(manifest.files)
    _validate_manifest_hash(manifest)
    return root, manifest


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not inspect destination path: {path}") from exc
    return True


def _validate_destination_namespace(workspace: Path, source_roots: Sequence[Path]) -> None:
    workspace = _canonical_existing_directory(workspace, label="builder workspace")
    for source_root in source_roots:
        if _paths_overlap(source_root, workspace):
            raise ValueError("builder input source and workspace paths must be separate")

    namespace = workspace / "reference_files"
    if _path_exists_without_following(namespace):
        if namespace.is_symlink():
            raise ValueError("builder workspace namespace must not be a symlink")
        raise FileExistsError(f"builder workspace namespace already exists: {namespace}")
    # Reject a differently-cased sibling as well.  Such a sibling is ambiguous on a
    # case-insensitive filesystem even though it is distinct on this host.
    try:
        collisions = [
            child
            for child in workspace.iterdir()
            if _collision_key(child.name) == _collision_key("reference_files")
        ]
    except OSError as exc:
        raise ValueError("could not inspect builder workspace") from exc
    if collisions:
        raise FileExistsError("builder workspace namespace has a Unicode/casefold collision")


def _copy_exclusive(destination: Path, content: bytes) -> None:
    """Write exact bytes to a new regular file and flush the file."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                os.fchmod(handle.fileno(), 0o444)
            except (AttributeError, OSError):
                pass
            os.fsync(handle.fileno())
    except Exception:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def _mkdir_exclusive(path: Path) -> None:
    try:
        path.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise FileExistsError(f"builder destination already exists: {path}") from exc
    _directory_metadata(path, label="builder destination directory")
    fsync_directory(path)
    fsync_directory(path.parent)


def _make_directory_read_only(path: Path) -> None:
    try:
        os.chmod(path, 0o555, follow_symlinks=False)
    except (AttributeError, OSError):
        pass
    fsync_directory(path)


def _remove_tree_without_following(path: Path) -> None:
    """Remove a call-created subtree without ever traversing a symlink."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        try:
            path.unlink()
        except OSError:
            pass
        return
    try:
        os.chmod(path, 0o755, follow_symlinks=False)
    except (AttributeError, OSError):
        pass
    try:
        children = tuple(path.iterdir())
    except OSError:
        children = ()
    for child in children:
        _remove_tree_without_following(child)
    try:
        path.rmdir()
    except OSError:
        pass


def stage_builder_inputs(
    inputs: Sequence[BuilderInputBundle],
    workspace: Path,
) -> tuple[StagedBuilderInput, ...]:
    """Validate every source, then copy allowlisted bytes into neutral directories."""

    if isinstance(inputs, (str, bytes)):
        raise TypeError("inputs must be a sequence of BuilderInputBundle instances")
    try:
        bundles = tuple(inputs)
    except TypeError as exc:
        raise TypeError("inputs must be a sequence of BuilderInputBundle instances") from exc
    try:
        workspace_path = Path(workspace)
    except (TypeError, ValueError) as exc:
        raise TypeError("workspace must be path-like") from exc

    validated: list[tuple[Path, BuilderInputManifest]] = []
    input_ids: set[str] = set()
    for bundle in bundles:
        root, manifest = _validate_bundle_for_staging(bundle)
        if manifest.input_id in input_ids:
            raise ValueError("builder input IDs must be unique")
        input_ids.add(manifest.input_id)
        validated.append((root, manifest))
    if len(validated) > _MAX_TARGET_NUMBER:
        raise ValueError("builder input count must be between 0 and 999")

    # All source reads and all destination preflight checks happen before the first mkdir.
    _validate_destination_namespace(workspace_path, tuple(root for root, _ in validated))
    snapshots: list[tuple[Path, BuilderInputManifest, tuple[tuple[str, bytes], ...]]] = []
    for root, manifest in validated:
        snapshots.append((root, manifest, _snapshot_manifest_files(root, manifest)))

    if not snapshots:
        return ()

    namespace = workspace_path / "reference_files"
    staged: list[StagedBuilderInput] = []
    try:
        _mkdir_exclusive(namespace)
        builder_inputs = namespace / "builder-inputs"
        _mkdir_exclusive(builder_inputs)

        for index, (_, manifest, entries) in enumerate(snapshots, start=1):
            target = _target_for_number(index)
            target_path = workspace_path / PurePosixPath(target)
            _mkdir_exclusive(target_path)
            created_directories: set[Path] = {target_path}
            for logical_path, _ in entries:
                current = target_path
                parts = PurePosixPath(logical_path).parts
                for part in parts[:-1]:
                    current = current / part
                    if current not in created_directories:
                        _mkdir_exclusive(current)
                        created_directories.add(current)

            materialized: list[InterventionFile] = []
            for logical_path, content in entries:
                destination = target_path / PurePosixPath(logical_path)
                _copy_exclusive(destination, content)
                materialized.append(
                    InterventionFile(
                        path=f"{target}/{logical_path}",
                        size=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                    )
                )
            staged.append(
                StagedBuilderInput(
                    input_id=manifest.input_id,
                    target=target,
                    materialized_files=tuple(materialized),
                )
            )

        # Detect a source mutation that happened while bytes were being copied.  The initial
        # snapshot above prevents writes on an already-stale bundle; this second snapshot keeps
        # a concurrent source change from silently producing a stale workspace.
        for root, manifest, _ in snapshots:
            _snapshot_manifest_files(root, manifest)

        # Protect the generated namespace from ordinary writes after it is complete.
        for directory in sorted(namespace.rglob("*"), key=lambda path: len(path.parts), reverse=True):
            if directory.is_dir() and not directory.is_symlink():
                _make_directory_read_only(directory)
        _make_directory_read_only(namespace)

        result = tuple(staged)
        verify_staged_builder_inputs(result, workspace_path)
        return result
    except BaseException:
        _remove_tree_without_following(namespace)
        try:
            fsync_directory(workspace_path)
        except OSError:
            pass
        raise


def _walk_namespace(path: Path, relative: str, files: set[str], directories: set[str]) -> None:
    _directory_metadata(path, label=f"builder destination directory {relative!r}")
    directories.add(relative)
    try:
        children = sorted(path.iterdir(), key=lambda child: child.name)
    except OSError as exc:
        raise ValueError(f"could not scan builder destination directory {relative!r}") from exc
    for child in children:
        child_relative = f"{relative}/{child.name}"
        try:
            metadata = child.lstat()
        except OSError as exc:
            raise ValueError(f"builder destination changed while verifying: {child_relative!r}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"builder destination contains a symlink: {child_relative!r}")
        if stat.S_ISDIR(metadata.st_mode):
            _walk_namespace(child, child_relative, files, directories)
        elif stat.S_ISREG(metadata.st_mode):
            files.add(child_relative)
        else:
            raise ValueError(f"builder destination contains a special file: {child_relative!r}")


def _read_materialized_file(path: Path, expected: InterventionFile) -> None:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"builder destination file is missing: {expected.path!r}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"builder destination file is not regular: {expected.path!r}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        descriptor_metadata = os.fstat(descriptor)
        if stat.S_ISLNK(descriptor_metadata.st_mode) or not stat.S_ISREG(descriptor_metadata.st_mode):
            raise ValueError(f"builder destination file is not regular: {expected.path!r}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read()
            after_descriptor = os.fstat(handle.fileno())
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read builder destination file: {expected.path!r}") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ValueError(f"builder destination changed while verifying: {expected.path!r}") from exc
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or after_descriptor.st_size != len(content)
        or after.st_size != len(content)
        or len(content) != expected.size
        or hashlib.sha256(content).hexdigest() != expected.sha256
    ):
        raise ValueError(f"builder destination bytes changed: {expected.path!r}")


def _expected_directories(file_paths: Sequence[str]) -> set[str]:
    directories = {_TARGET_PREFIX, "reference_files", "reference_files/builder-inputs"}
    for file_path in file_paths:
        parts = PurePosixPath(file_path).parts
        for index in range(1, len(parts)):
            directories.add("/".join(parts[:index]))
    return directories


def verify_staged_builder_inputs(staged: Sequence[StagedBuilderInput], workspace: Path) -> None:
    """Verify neutral staged files and their byte-level evidence without source access."""

    if isinstance(staged, (str, bytes)):
        raise TypeError("staged must be a sequence of StagedBuilderInput instances")
    try:
        records = tuple(staged)
    except TypeError as exc:
        raise TypeError("staged must be a sequence of StagedBuilderInput instances") from exc
    try:
        workspace_path = Path(workspace)
    except (TypeError, ValueError) as exc:
        raise TypeError("workspace must be path-like") from exc
    _canonical_existing_directory(workspace_path, label="builder workspace")

    input_ids: set[str] = set()
    expected_files: dict[str, InterventionFile] = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, StagedBuilderInput):
            raise TypeError("staged must contain StagedBuilderInput instances")
        if record.input_id in input_ids:
            raise ValueError("staged builder input IDs must be unique")
        input_ids.add(record.input_id)
        target = _target_for_number(index)
        _validate_target(record.target, expected=target)

        paths = [item.path for item in record.materialized_files]
        if tuple(paths) != tuple(sorted(paths)):
            raise ValueError("staged builder input evidence must be sorted by path")
        source_paths: list[str] = []
        for item in record.materialized_files:
            if not isinstance(item, InterventionFile):
                raise TypeError("staged builder input evidence must contain InterventionFile instances")
            path = _validate_relative_path(item.path, label="materialized builder input path")
            prefix = f"{target}/"
            if not path.startswith(prefix) or len(path) == len(prefix):
                raise ValueError("materialized builder input evidence must be workspace-relative to its target")
            source_path = path[len(prefix) :]
            _validate_relative_path(source_path, label="materialized builder input source path")
            source_paths.append(source_path)
            if path in expected_files:
                raise ValueError("staged builder input evidence contains duplicate paths")
            expected_files[path] = item
        _validate_path_component_collisions(source_paths, label="materialized builder input evidence")

    if not records:
        namespace = workspace_path / "reference_files"
        if _path_exists_without_following(namespace):
            raise ValueError("builder workspace contains an unexpected reference_files namespace")
        return

    namespace = workspace_path / "reference_files"
    if not _path_exists_without_following(namespace):
        raise ValueError("builder workspace is missing the reference_files namespace")
    files: set[str] = set()
    directories: set[str] = set()
    _walk_namespace(namespace, "reference_files", files, directories)

    expected_file_paths = set(expected_files)
    if files != expected_file_paths:
        raise ValueError("builder workspace contains an unexpected or missing materialized file")
    expected_directory_paths = _expected_directories(tuple(expected_files))
    if directories != expected_directory_paths:
        raise ValueError("builder workspace contains an unexpected or missing materialized directory")

    for path in sorted(expected_files):
        _read_materialized_file(workspace_path / PurePosixPath(path), expected_files[path])


__all__ = [
    "StagedBuilderInput",
    "load_builder_input_bundle",
    "stage_builder_inputs",
    "verify_staged_builder_inputs",
]
