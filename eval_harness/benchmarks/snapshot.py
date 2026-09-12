# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Immutable, content-addressed benchmark snapshots.

The snapshot boundary is deliberately independent from the benchmark runner.  A
benchmark adapter parses its source and stages task inputs; this module then
owns canonicalization, content addressing, publication, verification, and the
small execution projection that can be materialized into a workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast


if TYPE_CHECKING:
    from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
    from eval_harness.executors.base import TaskSpec


SNAPSHOT_SCHEMA: Final = "benchmark-snapshot"
SNAPSHOT_SCHEMA_VERSION: Final = 1
VIEW_SCHEMA: Final = "benchmark-snapshot-view-v1"
TASK_SCHEMA: Final = "benchmark-snapshot-task-v1"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DRIVE_RE: Final = re.compile(r"^[A-Za-z]:")
_MANIFEST_NAME: Final = "benchmark-snapshot.json"


class Availability(StrEnum):
    """Whether optional source metadata is known for a snapshot."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class SnapshotError(ValueError):
    """Raised when a snapshot cannot be built, decoded, or verified."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_digest(label: str, value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SnapshotError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SnapshotError("snapshot JSON contains invalid UTF-8 text") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotError("snapshot JSON contains an unsupported or non-finite value") from exc
    return encoded


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError("JSON contains duplicate object keys")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SnapshotError("JSON contains a non-finite number")
    return number


def _decode_json_object(raw: str, *, label: str) -> dict[str, object]:
    """Decode one strict JSON object from a benchmark-owned text source."""

    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(SnapshotError("JSON contains a non-finite number")),
            parse_float=_finite_json_float,
        )
    except SnapshotError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise SnapshotError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise SnapshotError(f"{label} must be a JSON object")
    return value


def _collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _validate_logical_path(path: object, *, label: str = "snapshot file path") -> str:
    if type(path) is not str or not path or "\\" in path or "\x00" in path:
        raise SnapshotError(f"{label} is unsafe")
    if unicodedata.normalize("NFC", path) != path:
        raise SnapshotError(f"{label} is not NFC-normalized")
    if _WINDOWS_DRIVE_RE.match(path) is not None:
        raise SnapshotError(f"{label} is unsafe")
    pure = PurePosixPath(path)
    if pure.is_absolute() or pure.parts != tuple(path.split("/")):
        raise SnapshotError(f"{label} is unsafe")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise SnapshotError(f"{label} is unsafe")
    return path


def _validate_file_projection(
    files: Sequence["SnapshotFile"], *, label: str = "snapshot files"
) -> tuple["SnapshotFile", ...]:
    incoming = tuple(files)
    if any(type(item) is not SnapshotFile for item in incoming):
        raise SnapshotError(f"{label} contain an invalid entry")
    ordered = tuple(sorted(incoming, key=lambda item: item.path))

    exact: set[str] = set()
    folded: dict[str, str] = {}
    path_keys: dict[str, str] = {}
    component_keys: dict[tuple[str, str], str] = {}
    for item in ordered:
        _validate_logical_path(item.path, label=label[:-1] if label.endswith("s") else label)
        if item.path in exact:
            raise SnapshotError(f"{label} contain a duplicate path")
        exact.add(item.path)

        key = _collision_key(item.path)
        previous = folded.get(key)
        if previous is not None and previous != item.path:
            raise SnapshotError(f"{label} contain a Unicode or casefold collision")
        folded[key] = item.path

        # Compare every normalized directory prefix against complete logical
        # paths.  This catches both exact and casefolded file/directory clashes.
        parts = item.path.split("/")
        parent_key = ""
        for component in parts:
            component_key = _collision_key(component)
            component_identity = (parent_key, component_key)
            previous_component = component_keys.get(component_identity)
            if previous_component is not None and previous_component != component:
                raise SnapshotError(f"{label} contain a Unicode or casefold directory collision")
            component_keys[component_identity] = component
            parent_key = f"{parent_key}/{component_key}" if parent_key else component_key
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index])
            prefix_key = _collision_key(prefix)
            if prefix_key in path_keys:
                raise SnapshotError(f"{label} contain a file and directory collision")
        path_keys[key] = item.path

    return ordered


def _freeze_json(value: object, *, label: str = "snapshot JSON") -> object:
    """Deep-copy and freeze the restricted JSON value domain."""

    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SnapshotError(f"{label} contains a non-finite number")
        return value
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SnapshotError(f"{label} contains invalid UTF-8 text") from exc
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SnapshotError(f"{label} object keys must be strings")
            frozen[key] = _freeze_json(item, label=label)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, label=label) for item in value)
    raise SnapshotError(f"{label} contains an unsupported value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {cast(str, key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _view_data_bytes(data: object) -> bytes:
    return _canonical_bytes(_thaw_json(data))


def _file_payload(file: "SnapshotFile") -> dict[str, object]:
    return {"path": file.path, "size": file.size, "sha256": file.sha256}


def _view_payload(view: "SnapshotView", *, include_hashes: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": VIEW_SCHEMA,
        "data": _thaw_json(view.data),
        "files": [_file_payload(item) for item in view.files],
    }
    if include_hashes:
        payload["data_sha256"] = view.data_sha256
        payload["view_sha256"] = view.view_sha256
    return payload


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """Exact bytes identified by a normalized logical POSIX path."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_logical_path(self.path)
        if type(self.size) is not int or self.size < 0:
            raise SnapshotError("snapshot file size must be a non-negative integer")
        _validate_digest("snapshot file sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class SnapshotView:
    """Deep-frozen JSON plus the exact file manifest visible in one view."""

    data: object
    files: Sequence[SnapshotFile] = ()
    data_sha256: str | None = None
    view_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise SnapshotError("snapshot view data must be a JSON object")
        frozen_data = _freeze_json(self.data)
        object.__setattr__(self, "data", frozen_data)
        incoming_files = tuple(self.files)
        files = _validate_file_projection(incoming_files)
        object.__setattr__(self, "files", files)

        data_digest = _sha256(_view_data_bytes(frozen_data))
        if self.data_sha256 is not None and self.data_sha256 != data_digest:
            raise SnapshotError("snapshot view data_sha256 does not match canonical data")
        object.__setattr__(self, "data_sha256", data_digest)
        payload = {
            "schema": VIEW_SCHEMA,
            "data_sha256": data_digest,
            "data": _thaw_json(frozen_data),
            "files": [_file_payload(item) for item in files],
        }
        view_digest = _sha256(_canonical_bytes(payload))
        if self.view_sha256 is not None and self.view_sha256 != view_digest:
            raise SnapshotError("snapshot view_sha256 does not match canonical view")
        object.__setattr__(self, "view_sha256", view_digest)

    def projection(self) -> dict[str, object]:
        """Return a fresh mutable projection of this view."""

        return {
            "data": _thaw_json(self.data),
            "files": [_file_payload(item) for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class SnapshotTask:
    """One task with separate executor and evaluator views."""

    task_id: str
    canonical_task_prompt: str
    execution_view: SnapshotView
    evaluation_view: SnapshotView
    canonical_prompt_sha256: str | None = None
    task_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise SnapshotError("snapshot task_id must be a non-empty string")
        if type(self.canonical_task_prompt) is not str:
            raise SnapshotError("snapshot task prompt must be a string")
        try:
            prompt_bytes = self.canonical_task_prompt.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SnapshotError("snapshot task prompt must be valid UTF-8") from exc
        if type(self.execution_view) is not SnapshotView or type(self.evaluation_view) is not SnapshotView:
            raise SnapshotError("snapshot task views must be SnapshotView instances")
        if self.execution_view.data:
            raise SnapshotError("snapshot execution view data must be empty")
        combined_files: dict[str, SnapshotFile] = {item.path: item for item in self.execution_view.files}
        for item in self.evaluation_view.files:
            shared = combined_files.get(item.path)
            if shared is not None and shared != item:
                raise SnapshotError("snapshot views disagree on shared file bytes")
            combined_files[item.path] = item
        _validate_file_projection(tuple(combined_files.values()), label="snapshot task files")
        prompt_digest = _sha256(prompt_bytes)
        if self.canonical_prompt_sha256 is not None and self.canonical_prompt_sha256 != prompt_digest:
            raise SnapshotError("snapshot task prompt_sha256 does not match prompt")
        object.__setattr__(self, "canonical_prompt_sha256", prompt_digest)
        payload = {
            "schema": TASK_SCHEMA,
            "task_id": self.task_id,
            "canonical_prompt_sha256": prompt_digest,
            "execution_view_sha256": self.execution_view.view_sha256,
            "evaluation_view_sha256": self.evaluation_view.view_sha256,
        }
        task_digest = _sha256(_canonical_bytes(payload))
        if self.task_sha256 is not None and self.task_sha256 != task_digest:
            raise SnapshotError("snapshot task_sha256 does not match task components")
        object.__setattr__(self, "task_sha256", task_digest)

    def task_spec(self) -> "TaskSpec":
        """Build the exact executor-facing TaskSpec projection."""

        from eval_harness.executors.base import TaskSpec

        return TaskSpec(task_id=self.task_id, prompt=self.canonical_task_prompt)

    def execution_projection(self) -> dict[str, object]:
        projection = self.execution_view.projection()
        return {"task_id": self.task_id, "canonical_task_prompt": self.canonical_task_prompt, **projection}

    def evaluation_projection(self) -> dict[str, object]:
        projection = self.evaluation_view.projection()
        return {"task_id": self.task_id, "canonical_task_prompt": self.canonical_task_prompt, **projection}


@dataclass(frozen=True, slots=True)
class SnapshotTaskContent:
    """Adapter output used while an acquisition workspace is still private."""

    evaluation_data: Mapping[str, object] = field(default_factory=dict)
    files: Sequence[tuple[str, Path | bytes]] = ()
    evaluation_files: Sequence[tuple[str, Path | bytes]] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_data, Mapping):
            raise SnapshotError("snapshot adapter evaluation view must be a JSON object")
        files = tuple(self.files)
        evaluation_files = tuple(self.evaluation_files)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "evaluation_files", evaluation_files)


@dataclass(frozen=True, slots=True)
class BenchmarkSnapshot:
    """A movable sealed snapshot and its semantic manifest."""

    benchmark_id: str
    source: str | None
    source_availability: Availability
    revision: str | None
    revision_availability: Availability
    tasks: Sequence[SnapshotTask]
    root: Path = field(default=Path("."), compare=False, repr=False)
    snapshot_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.benchmark_id) is not str or not self.benchmark_id:
            raise SnapshotError("benchmark_id must be a non-empty string")
        try:
            source_status = Availability(self.source_availability)
            revision_status = Availability(self.revision_availability)
        except (TypeError, ValueError) as exc:
            raise SnapshotError("snapshot metadata availability is invalid") from exc
        _validate_optional_metadata(self.source, source_status, "source")
        _validate_optional_metadata(self.revision, revision_status, "revision")
        object.__setattr__(self, "source_availability", source_status)
        object.__setattr__(self, "revision_availability", revision_status)

        tasks = tuple(self.tasks)
        if not tasks or any(type(task) is not SnapshotTask for task in tasks):
            raise SnapshotError("snapshot tasks must be exact SnapshotTask instances")
        task_ids: set[str] = set()
        for task in tasks:
            if task.task_id in task_ids:
                raise SnapshotError("snapshot task IDs must be unique")
            task_ids.add(task.task_id)
        object.__setattr__(self, "tasks", tasks)

        try:
            root = Path(self.root)
        except (TypeError, ValueError) as exc:
            raise SnapshotError("snapshot root must be a filesystem path") from exc
        object.__setattr__(self, "root", root)

        expected = self.compute_snapshot_sha256()
        if self.snapshot_sha256 is not None and self.snapshot_sha256 != expected:
            raise SnapshotError("snapshot_sha256 does not match canonical manifest")
        object.__setattr__(self, "snapshot_sha256", expected)

    def manifest_payload(self, *, include_snapshot_sha256: bool = True) -> dict[str, object]:
        tasks: list[dict[str, object]] = []
        for task in self.tasks:
            tasks.append(
                {
                    "task_id": task.task_id,
                    "canonical_task_prompt": task.canonical_task_prompt,
                    "canonical_prompt_sha256": task.canonical_prompt_sha256,
                    "execution_view": _view_payload(task.execution_view),
                    "evaluation_view": _view_payload(task.evaluation_view),
                    "task_sha256": task.task_sha256,
                }
            )
        payload: dict[str, object] = {
            "schema": SNAPSHOT_SCHEMA,
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "source": self.source,
            "source_availability": self.source_availability.value,
            "revision": self.revision,
            "revision_availability": self.revision_availability.value,
            "tasks": tasks,
        }
        if include_snapshot_sha256:
            payload["snapshot_sha256"] = self.snapshot_sha256
        return payload

    def canonical_manifest_bytes(self, *, include_snapshot_sha256: bool = True) -> bytes:
        return _canonical_bytes(self.manifest_payload(include_snapshot_sha256=include_snapshot_sha256))

    def compute_snapshot_sha256(self) -> str:
        return _sha256(self.canonical_manifest_bytes(include_snapshot_sha256=False))

    def task(self, task_id: str) -> SnapshotTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError("snapshot task not found")


def _validate_optional_metadata(value: object, status: Availability, label: str) -> None:
    if status is Availability.AVAILABLE:
        if type(value) is not str or not value:
            raise SnapshotError(f"available {label} requires a non-empty string")
    elif value is not None:
        raise SnapshotError(f"unavailable {label} requires null")


def _ensure_no_symlink_ancestors(path: Path, *, label: str) -> None:
    try:
        absolute = Path(os.path.abspath(path))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"{label} path is unavailable") from exc
    current = Path(absolute.root)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                raise SnapshotError(f"{label} contains a symlink")
        except OSError as exc:
            raise SnapshotError(f"{label} path cannot be inspected") from exc


def _lstat_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        _ensure_no_symlink_ancestors(path, label=label)
        info = path.lstat()
    except (OSError, SnapshotError) as exc:
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError(f"{label} cannot be read") from exc
    if stat.S_ISLNK(info.st_mode):
        raise SnapshotError(f"{label} contains a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise SnapshotError(f"{label} is not a regular file")
    return info


def _same_file_stat(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
        and first.st_ctime_ns == second.st_ctime_ns
    )


def _stream_race_safe(
    path: Path,
    *,
    label: str,
    consume: Callable[[bytes], object] | None = None,
) -> tuple[int, str]:
    before = _lstat_regular(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not _same_file_stat(before, opened):
            raise SnapshotError(f"{label} changed during read")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if consume is not None:
                consume(chunk)
        after_open = os.fstat(descriptor)
        after_path = _lstat_regular(path, label=label)
        if (
            not _same_file_stat(opened, after_open)
            or not _same_file_stat(before, after_path)
            or total != after_open.st_size
        ):
            raise SnapshotError(f"{label} changed during read")
        return total, digest.hexdigest()
    except SnapshotError:
        raise
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"{label} cannot be read") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_race_safe(path: Path, *, label: str) -> bytes:
    chunks: list[bytes] = []
    _stream_race_safe(path, label=label, consume=chunks.append)
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError as exc:
        raise SnapshotError("snapshot directory cannot be synced") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_exclusive(path: Path, content: bytes, *, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise SnapshotError(f"{label} already exists") from exc
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"{label} cannot be written") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _copy_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    """Stream one verified file into a fresh destination."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            size, digest = _stream_race_safe(source, label=label, consume=handle.write)
            if size != expected_size or digest != expected_sha256:
                raise SnapshotError(f"{label} bytes do not match manifest")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _safe_mkdir(path: Path, *, label: str) -> None:
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise SnapshotError(f"{label} already exists") from exc
    except OSError as exc:
        raise SnapshotError(f"{label} cannot be created") from exc


def _normalize_adapter_files(
    entries: Sequence[tuple[str, Path | bytes]], *, label: str
) -> tuple[tuple[str, Path | bytes], ...]:
    normalized: list[tuple[str, Path | bytes]] = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise SnapshotError(f"{label} contain an invalid entry")
        logical, source = entry
        _validate_logical_path(logical, label=label)
        if not logical.startswith("task_inputs/"):
            raise SnapshotError(f"{label} must use task_inputs paths")
        if type(source) is not bytes and not isinstance(source, Path):
            raise SnapshotError(f"{label} contain an invalid source")
        normalized.append((logical, source))
    # SnapshotView performs the final collision check after bytes are read;
    # checking logical paths here ensures an adapter cannot overwrite a staged
    # file while acquisition is in progress.
    paths = tuple(SnapshotFile(path=logical, size=0, sha256="0" * 64) for logical, _ in normalized)
    _validate_file_projection(paths, label=label)
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _content_to_task(task: "BenchmarkTask", content: SnapshotTaskContent, cas_root: Path) -> SnapshotTask:
    if type(task.execution.task_id) is not str or not task.execution.task_id:
        raise SnapshotError("benchmark task ID must be a non-empty string")
    if type(task.execution.prompt) is not str:
        raise SnapshotError("benchmark task prompt must be a string")

    execution_entries = _normalize_adapter_files(content.files, label="execution snapshot files")
    evaluation_entries = _normalize_adapter_files(content.evaluation_files, label="evaluation snapshot files")
    source_cache: dict[Path, tuple[int, str, os.stat_result]] = {}

    def ingest_blob(source: Path | bytes, *, label: str) -> tuple[int, str]:
        if isinstance(source, Path):
            cached = source_cache.get(source)
            if cached is not None:
                return cached[0], cached[1]
            temporary_fd, temporary_name = tempfile.mkstemp(prefix=".blob-", dir=str(cas_root))
            temporary = Path(temporary_name)
            try:
                source_before = _lstat_regular(source, label=label)
                with os.fdopen(temporary_fd, "wb") as handle:
                    temporary_fd = -1
                    size, digest = _stream_race_safe(source, label=label, consume=handle.write)
                    handle.flush()
                    os.fsync(handle.fileno())
                target = cas_root / digest
                if target.exists() or target.is_symlink():
                    existing_size, existing_digest = _stream_race_safe(target, label="snapshot blob")
                    if existing_size != size or existing_digest != digest:
                        raise SnapshotError("snapshot content-addressed blob collision")
                    temporary.unlink(missing_ok=True)
                else:
                    os.rename(temporary, target)
                source_cache[source] = (size, digest, source_before)
                return size, digest
            except SnapshotError:
                temporary.unlink(missing_ok=True)
                raise
            except (OSError, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                raise SnapshotError(f"{label} cannot be staged") from exc
            finally:
                if temporary_fd != -1:
                    os.close(temporary_fd)

        payload = bytes(source)
        size = len(payload)
        digest = _sha256(payload)
        target = cas_root / digest
        if target.exists() or target.is_symlink():
            existing_size, existing_digest = _stream_race_safe(target, label="snapshot blob")
            if existing_size != size or existing_digest != digest:
                raise SnapshotError("snapshot content-addressed blob collision")
        else:
            _write_exclusive(target, payload, label="snapshot blob")
        return size, digest

    def ingest(entries: Sequence[tuple[str, Path | bytes]], label: str) -> tuple[SnapshotFile, ...]:
        manifests: list[SnapshotFile] = []
        for logical, source in entries:
            size, digest = ingest_blob(source, label=label)
            evidence = SnapshotFile(path=logical, size=size, sha256=digest)
            manifests.append(evidence)
        for source, (size, digest, source_before) in source_cache.items():
            source_after = _lstat_regular(source, label=label)
            if not _same_file_stat(source_before, source_after):
                raise SnapshotError(f"{label} changed during acquisition")
        return _validate_file_projection(tuple(manifests), label=label)

    execution_files = ingest(execution_entries, "execution source file")
    evaluation_files = ingest(evaluation_entries, "evaluation source file")
    execution_view = SnapshotView({}, execution_files)
    evaluation_view = SnapshotView(content.evaluation_data, evaluation_files)
    snapshot_task = SnapshotTask(
        task_id=task.execution.task_id,
        canonical_task_prompt=task.execution.prompt,
        execution_view=execution_view,
        evaluation_view=evaluation_view,
    )
    return snapshot_task


def _source_paths(benchmark: "Benchmark") -> tuple[Path, ...]:
    paths = benchmark.snapshot_source_paths()
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        raise SnapshotError("benchmark source paths have an invalid shape")
    result: list[Path] = []
    for path in paths:
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except (TypeError, ValueError) as exc:
                raise SnapshotError("benchmark source path is invalid") from exc
        result.append(path)
    return tuple(result)


def _source_fingerprint(paths: Sequence[Path]) -> tuple[str, ...]:
    # The bytes are deliberately not part of the semantic manifest.  They are
    # used only to prove that an adapter parsed a stable source window.
    fingerprints: list[str] = []
    for path in paths:
        before = _lstat_regular(path, label="benchmark source")
        size, digest = _stream_race_safe(path, label="benchmark source")
        after = _lstat_regular(path, label="benchmark source")
        if not _same_file_stat(before, after):
            raise SnapshotError("benchmark source changed during acquisition")
        fingerprints.append(
            ":".join(
                (
                    str(before.st_dev),
                    str(before.st_ino),
                    str(size),
                    str(before.st_mtime_ns),
                    str(before.st_ctime_ns),
                    digest,
                )
            )
        )
    return tuple(fingerprints)


def _source_stat_fingerprint(paths: Sequence[Path]) -> tuple[tuple[int, int, int, int, int], ...]:
    """Capture source identity without reading source bytes."""

    result: list[tuple[int, int, int, int, int]] = []
    for path in paths:
        info = _lstat_regular(path, label="benchmark source")
        result.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(result)


def _adapter_content(benchmark: "Benchmark", task: "BenchmarkTask", workspace: Path) -> SnapshotTaskContent:
    content = benchmark.snapshot_task(task, workspace)
    if not isinstance(content, SnapshotTaskContent):
        raise SnapshotError("benchmark snapshot acquisition returned an invalid result")
    return content


def _metadata_from_benchmark(
    benchmark: "Benchmark",
) -> tuple[str, str | None, Availability, str | None, Availability]:
    benchmark_id = benchmark.name
    source = benchmark.source
    source_status = benchmark.source_availability
    revision = benchmark.revision
    revision_status = benchmark.revision_availability
    if type(benchmark_id) is not str or not benchmark_id:
        raise SnapshotError("benchmark metadata is invalid")
    if source is not None and type(source) is not str:
        raise SnapshotError("benchmark metadata is invalid")
    if revision is not None and type(revision) is not str:
        raise SnapshotError("benchmark metadata is invalid")
    try:
        return benchmark_id, source, Availability(source_status), revision, Availability(revision_status)
    except (TypeError, ValueError) as exc:
        raise SnapshotError("benchmark metadata is invalid") from exc


def acquire_snapshot(benchmark: "Benchmark", limit: int, destination: Path) -> BenchmarkSnapshot:
    """Acquire and atomically publish a fresh benchmark snapshot.

    The adapter is called only while a private sibling staging directory exists.
    Any error removes that directory and leaves an existing destination alone.
    """

    if type(limit) is not int or limit <= 0:
        raise SnapshotError("benchmark task limit must be a positive integer")
    if not isinstance(destination, Path):
        try:
            destination = Path(destination)
        except (TypeError, ValueError) as exc:
            raise SnapshotError("snapshot destination is invalid") from exc
    _ensure_no_symlink_ancestors(destination.parent, label="snapshot destination")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SnapshotError("snapshot destination parent cannot be created") from exc
    _ensure_no_symlink_ancestors(destination.parent, label="snapshot destination")
    if destination.exists() or destination.is_symlink():
        raise SnapshotError("snapshot destination must be fresh")

    stage: Path | None = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(destination.parent)))
        benchmark_id, source, source_status, revision, revision_status = _metadata_from_benchmark(benchmark)
        source_paths = _source_paths(benchmark)
        before = _source_fingerprint(source_paths)
        tasks = tuple(benchmark.load_tasks(limit))
        if not tasks:
            raise SnapshotError("benchmark returned no tasks")
        from eval_harness.benchmarks.base import BenchmarkTask

        if any(type(task) is not BenchmarkTask for task in tasks):
            raise SnapshotError("benchmark returned an invalid task")
        after_parse = _source_fingerprint(source_paths)
        if before != after_parse:
            raise SnapshotError("benchmark source changed during acquisition")

        blobs_root = stage / "blobs"
        digest_root = blobs_root / "sha256"
        _safe_mkdir(blobs_root, label="snapshot blob directory")
        _safe_mkdir(digest_root, label="snapshot digest directory")
        acquisition_root = stage / "acquisition"
        _safe_mkdir(acquisition_root, label="snapshot acquisition directory")
        snapshot_tasks: list[SnapshotTask] = []
        for index, task in enumerate(tasks):
            task_root = acquisition_root / f"task-{index:08d}"
            _safe_mkdir(task_root, label="snapshot task staging directory")
            content = _adapter_content(benchmark, task, task_root)
            snapshot_task = _content_to_task(task, content, digest_root)
            if snapshot_task.task_id in {item.task_id for item in snapshot_tasks}:
                raise SnapshotError("benchmark returned duplicate task IDs")
            snapshot_tasks.append(snapshot_task)
        after_stage = _source_fingerprint(source_paths)
        if before != after_stage:
            raise SnapshotError("benchmark source changed during acquisition")

        # The adapter workspace is private acquisition state.  Keeping it under
        # the eventual root would create an unsealed, unexpected tree.
        shutil.rmtree(acquisition_root)

        snapshot = BenchmarkSnapshot(
            benchmark_id=benchmark_id,
            source=source,
            source_availability=source_status,
            revision=revision,
            revision_availability=revision_status,
            tasks=tuple(snapshot_tasks),
            root=stage,
        )
        _fsync_directory(digest_root)
        _fsync_directory(blobs_root)
        _publish_snapshot(snapshot, stage)
        # Verification happens while the directory is still private.  It
        # catches both writer mistakes and an adapter that modified a staged
        # blob after it was ingested.
        verify_snapshot(snapshot)
        if destination.exists() or destination.is_symlink():
            raise SnapshotError("snapshot destination must be fresh")
        try:
            os.rename(stage, destination)
        except OSError as exc:
            raise SnapshotError("snapshot destination cannot be published") from exc
        stage = None
        _fsync_directory(destination.parent)
        return BenchmarkSnapshot(
            benchmark_id=snapshot.benchmark_id,
            source=snapshot.source,
            source_availability=snapshot.source_availability,
            revision=snapshot.revision,
            revision_availability=snapshot.revision_availability,
            tasks=snapshot.tasks,
            root=destination,
            snapshot_sha256=snapshot.snapshot_sha256,
        )
    except SnapshotError:
        raise
    except Exception as exc:
        raise SnapshotError("benchmark snapshot acquisition failed") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _expected_blobs(snapshot: BenchmarkSnapshot) -> dict[str, int]:
    blobs: dict[str, int] = {}
    for task in snapshot.tasks:
        for view in (task.execution_view, task.evaluation_view):
            for item in view.files:
                previous_size = blobs.setdefault(item.sha256, item.size)
                if previous_size != item.size:
                    raise SnapshotError("snapshot manifest has inconsistent blob sizes")
    return blobs


def _publish_snapshot(snapshot: BenchmarkSnapshot, root: Path) -> None:
    blobs_root = root / "blobs"
    digest_root = blobs_root / "sha256"
    if blobs_root.is_symlink() or not blobs_root.is_dir() or digest_root.is_symlink() or not digest_root.is_dir():
        raise SnapshotError("snapshot blob directory is invalid")
    _fsync_directory(digest_root)
    _fsync_directory(blobs_root)
    _write_exclusive(root / _MANIFEST_NAME, snapshot.canonical_manifest_bytes(), label="snapshot manifest")
    _fsync_directory(root)


def _decode_manifest(raw: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SnapshotError("snapshot manifest contains non-finite JSON")
            ),
            parse_float=_finite_json_float,
        )
    except SnapshotError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SnapshotError("snapshot manifest is not valid JSON") from exc
    if type(decoded) is not dict:
        raise SnapshotError("snapshot manifest must be a JSON object")
    return decoded


def _require_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise SnapshotError(f"{label} has an invalid schema")


def _parse_file(value: object) -> SnapshotFile:
    if type(value) is not dict:
        raise SnapshotError("snapshot file entry has an invalid schema")
    _require_keys(value, {"path", "size", "sha256"}, label="snapshot file entry")
    return SnapshotFile(
        path=cast(str, value["path"]),
        size=cast(int, value["size"]),
        sha256=cast(str, value["sha256"]),
    )


def _parse_view(value: object) -> SnapshotView:
    if type(value) is not dict:
        raise SnapshotError("snapshot view has an invalid schema")
    _require_keys(value, {"schema", "data", "files", "data_sha256", "view_sha256"}, label="snapshot view")
    if value["schema"] != VIEW_SCHEMA or type(value["files"]) is not list:
        raise SnapshotError("snapshot view has an invalid schema")
    files = tuple(_parse_file(item) for item in cast(list[object], value["files"]))
    return SnapshotView(
        data=value["data"],
        files=files,
        data_sha256=cast(str, value["data_sha256"]),
        view_sha256=cast(str, value["view_sha256"]),
    )


def _parse_task(value: object) -> SnapshotTask:
    if type(value) is not dict:
        raise SnapshotError("snapshot task has an invalid schema")
    _require_keys(
        value,
        {
            "task_id",
            "canonical_task_prompt",
            "canonical_prompt_sha256",
            "execution_view",
            "evaluation_view",
            "task_sha256",
        },
        label="snapshot task",
    )
    execution = _parse_view(value["execution_view"])
    evaluation = _parse_view(value["evaluation_view"])
    return SnapshotTask(
        task_id=cast(str, value["task_id"]),
        canonical_task_prompt=cast(str, value["canonical_task_prompt"]),
        execution_view=execution,
        evaluation_view=evaluation,
        canonical_prompt_sha256=cast(str, value["canonical_prompt_sha256"]),
        task_sha256=cast(str, value["task_sha256"]),
    )


def _snapshot_from_manifest(payload: dict[str, object], root: Path) -> BenchmarkSnapshot:
    _require_keys(
        payload,
        {
            "schema",
            "schema_version",
            "benchmark_id",
            "source",
            "source_availability",
            "revision",
            "revision_availability",
            "tasks",
            "snapshot_sha256",
        },
        label="snapshot manifest",
    )
    if (
        type(payload["schema"]) is not str
        or payload["schema"] != SNAPSHOT_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION
    ):
        raise SnapshotError("snapshot manifest has an unsupported schema")
    if type(payload["tasks"]) is not list:
        raise SnapshotError("snapshot manifest tasks must be an array")
    tasks = tuple(_parse_task(item) for item in cast(list[object], payload["tasks"]))
    source_availability = payload["source_availability"]
    revision_availability = payload["revision_availability"]
    if type(source_availability) is not str or type(revision_availability) is not str:
        raise SnapshotError("snapshot manifest availability is invalid")
    try:
        source_status = Availability(source_availability)
        revision_status = Availability(revision_availability)
    except (TypeError, ValueError) as exc:
        raise SnapshotError("snapshot manifest availability is invalid") from exc
    snapshot = BenchmarkSnapshot(
        benchmark_id=cast(str, payload["benchmark_id"]),
        source=cast(str | None, payload["source"]),
        source_availability=source_status,
        revision=cast(str | None, payload["revision"]),
        revision_availability=revision_status,
        tasks=tasks,
        root=root,
        snapshot_sha256=cast(str, payload["snapshot_sha256"]),
    )
    return snapshot


def _scan_sealed_root(root: Path, expected_blobs: set[str]) -> None:
    try:
        children = tuple(root.iterdir())
        names = {child.name for child in children}
        if names != {_MANIFEST_NAME, "blobs"}:
            raise SnapshotError("snapshot root contains unexpected content")
        for child in children:
            if child.is_symlink():
                raise SnapshotError("snapshot root contains a symlink")
            if child.name == _MANIFEST_NAME:
                _lstat_regular(child, label="snapshot manifest")
                continue
            if not child.is_dir():
                raise SnapshotError("snapshot root contains a non-directory")
            blob_children = {entry.name for entry in child.iterdir()}
            if blob_children != {"sha256"}:
                raise SnapshotError("snapshot blob directory contains unexpected content")
            sha_root = child / "sha256"
            if sha_root.is_symlink() or not sha_root.is_dir():
                raise SnapshotError("snapshot blob directory is invalid")
            digest_files = tuple(sha_root.iterdir())
            digest_names = {item.name for item in digest_files}
            if digest_names != expected_blobs:
                raise SnapshotError("snapshot blob store contains unexpected content")
            for item in digest_files:
                if item.is_symlink() or not item.is_file() or _SHA256_RE.fullmatch(item.name) is None:
                    raise SnapshotError("snapshot blob store contains an invalid node")
    except OSError as exc:
        raise SnapshotError("snapshot root cannot be listed") from exc


def verify_snapshot(snapshot: BenchmarkSnapshot | Path) -> BenchmarkSnapshot:
    """Verify the complete manifest, CAS, and sealed-root projection."""

    if isinstance(snapshot, Path):
        root = snapshot
        expected_sha256 = None
    elif type(snapshot) is BenchmarkSnapshot:
        root = snapshot.root
        expected_sha256 = snapshot.snapshot_sha256
    else:
        raise SnapshotError("snapshot must be a BenchmarkSnapshot or path")
    _ensure_no_symlink_ancestors(root, label="snapshot root")
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise SnapshotError("snapshot root cannot be read") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise SnapshotError("snapshot root must be a directory")
    raw_manifest = _read_race_safe(root / _MANIFEST_NAME, label="snapshot manifest")
    payload = _decode_manifest(raw_manifest)
    parsed = _snapshot_from_manifest(payload, root)
    if raw_manifest != parsed.canonical_manifest_bytes():
        raise SnapshotError("snapshot manifest is not canonical")
    if expected_sha256 is not None and expected_sha256 != parsed.snapshot_sha256:
        raise SnapshotError("snapshot object does not match sealed manifest")
    expected_blobs = _expected_blobs(parsed)
    _scan_sealed_root(root, set(expected_blobs))
    for digest, size in expected_blobs.items():
        actual_size, actual_digest = _stream_race_safe(
            root / "blobs" / "sha256" / digest,
            label="snapshot blob",
        )
        if actual_size != size or actual_digest != digest:
            raise SnapshotError("snapshot blob bytes do not match manifest")
    return parsed


def load_snapshot(root: Path) -> BenchmarkSnapshot:
    """Load and fully verify a canonical sealed snapshot from disk."""

    return verify_snapshot(root)


def read_evaluation_file(
    snapshot: BenchmarkSnapshot | Path,
    task_id: str,
    logical_path: str,
) -> bytes:
    """Read one allowlisted evaluation file without exposing the snapshot CAS."""

    verified = verify_snapshot(snapshot)
    if type(task_id) is not str or not task_id:
        raise SnapshotError("task_id must be a non-empty string")
    logical_path = _validate_logical_path(logical_path, label="evaluation file path")
    task = verified.task(task_id)
    entry = next((item for item in task.evaluation_view.files if item.path == logical_path), None)
    if entry is None:
        raise SnapshotError("evaluation file is not present in the task view")
    content = _read_race_safe(
        verified.root / "blobs" / "sha256" / entry.sha256,
        label="snapshot blob",
    )
    if len(content) != entry.size or _sha256(content) != entry.sha256:
        raise SnapshotError("snapshot blob bytes do not match manifest")
    return content


def materialize_evaluation(
    snapshot: BenchmarkSnapshot | Path,
    task_id: str,
    destination: Path,
) -> tuple[str, ...]:
    """Materialize only one task's evaluation allowlist into a fresh root."""

    verified = verify_snapshot(snapshot)
    if type(task_id) is not str or not task_id:
        raise SnapshotError("task_id must be a non-empty string")
    task = verified.task(task_id)
    if not isinstance(destination, Path):
        raise SnapshotError("evaluation destination must be a filesystem path")
    _ensure_no_symlink_ancestors(destination.parent, label="evaluation destination")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SnapshotError("evaluation destination parent cannot be created") from exc
    _ensure_no_symlink_ancestors(destination.parent, label="evaluation destination")
    if destination.exists() or destination.is_symlink():
        raise SnapshotError("evaluation destination must be fresh")

    stage: Path | None = None
    materialized: list[str] = []
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(destination.parent)))
        for item in task.evaluation_view.files:
            target = stage.joinpath(*item.path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            _ensure_no_symlink_ancestors(target.parent, label="evaluation staging")
            _copy_verified_file(
                verified.root / "blobs" / "sha256" / item.sha256,
                target,
                expected_size=item.size,
                expected_sha256=item.sha256,
                label="snapshot blob",
            )
            materialized.append(item.path)
        _fsync_directory(stage)
        if destination.exists() or destination.is_symlink():
            raise SnapshotError("evaluation destination must be fresh")
        os.rename(stage, destination)
        stage = None
        _fsync_directory(destination.parent)
        return tuple(materialized)
    except SnapshotError:
        raise
    except OSError as exc:
        raise SnapshotError("evaluation files cannot be materialized") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _prepare_materialization_root(workspace: Path) -> None:
    _ensure_no_symlink_ancestors(workspace, label="execution workspace")
    if workspace.exists() or workspace.is_symlink():
        if workspace.is_symlink() or not workspace.is_dir():
            raise SnapshotError("execution workspace is not a directory")
    else:
        try:
            workspace.mkdir(parents=True)
        except OSError as exc:
            raise SnapshotError("execution workspace cannot be created") from exc
        _ensure_no_symlink_ancestors(workspace, label="execution workspace")


def _preflight_materialization_paths(workspace: Path, files: Sequence[SnapshotFile]) -> None:
    _validate_file_projection(files, label="execution files")
    if not files:
        return
    # The entire task_inputs tree is published with one rename.  An existing
    # tree would make the projection ambiguous and could expose stale files.
    task_inputs_root = workspace / "task_inputs"
    if task_inputs_root.is_symlink() or task_inputs_root.exists():
        raise SnapshotError("execution destination already contains task_inputs")
    for sibling in workspace.iterdir():
        if _collision_key(sibling.name) == _collision_key("task_inputs"):
            raise SnapshotError("execution destination contains a path collision")
    for item in files:
        if not item.path.startswith("task_inputs/"):
            raise SnapshotError("execution files must use task_inputs paths")


def materialize_execution(snapshot: BenchmarkSnapshot | Path, task_id: str, workspace: Path) -> tuple[str, ...]:
    """Verify a snapshot and copy only one task's execution files from CAS."""

    verified = verify_snapshot(snapshot)
    if type(task_id) is not str or not task_id:
        raise SnapshotError("task_id must be a non-empty string")
    if not isinstance(workspace, Path):
        try:
            workspace = Path(workspace)
        except (TypeError, ValueError) as exc:
            raise SnapshotError("execution workspace is invalid") from exc
    task = verified.task(task_id)
    files = tuple(task.execution_view.files)
    _prepare_materialization_root(workspace)
    _preflight_materialization_paths(workspace, files)
    if not files:
        return ()

    # Stage the complete projection beside its final name and publish it in a
    # single rename, so a failed blob read or write leaves no partial tree.
    task_inputs_root = workspace / "task_inputs"
    stage: Path | None = None
    materialized: list[str] = []
    try:
        stage = Path(tempfile.mkdtemp(prefix=".task_inputs.staging-", dir=str(workspace)))
        _ensure_no_symlink_ancestors(stage, label="execution staging")
        for item in files:
            source = verified.root / "blobs" / "sha256" / item.sha256
            destination = stage.joinpath(*item.path.split("/")[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ensure_no_symlink_ancestors(destination.parent, label="execution staging")
            _copy_verified_file(
                source,
                destination,
                expected_size=item.size,
                expected_sha256=item.sha256,
                label="snapshot blob",
            )
            materialized.append(item.path)
        _fsync_directory(stage)
        if task_inputs_root.exists() or task_inputs_root.is_symlink():
            raise SnapshotError("execution destination already contains task_inputs")
        os.rename(stage, task_inputs_root)
        stage = None
        _ensure_no_symlink_ancestors(task_inputs_root, label="execution workspace")
        _fsync_directory(workspace)
        return tuple(materialized)
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


__all__ = [
    "Availability",
    "BenchmarkSnapshot",
    "SNAPSHOT_SCHEMA",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotError",
    "SnapshotFile",
    "SnapshotTask",
    "SnapshotTaskContent",
    "SnapshotView",
    "acquire_snapshot",
    "load_snapshot",
    "materialize_evaluation",
    "materialize_execution",
    "read_evaluation_file",
    "verify_snapshot",
]
