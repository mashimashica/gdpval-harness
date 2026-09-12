# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, relocatable run manifests and candidate result indexes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Self, cast

from eval_harness.candidate_bundle import (
    CandidateBundle,
    CandidateBundleError,
    SnapshotReference,
    VerifiedSnapshotBinding,
    load_candidate_bundle,
)


RUN_MANIFEST_SCHEMA: Final = "run-manifest"
RUN_MANIFEST_SCHEMA_VERSION: Final = 1
RUN_RESULT_SCHEMA: Final = "run-result"
RUN_RESULT_SCHEMA_VERSION: Final = 1
RUN_MANIFEST_NAME: Final = "run-manifest.json"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DRIVE_RE: Final = re.compile(r"^[A-Za-z]:")


class RunManifestError(ValueError):
    """Raised when a run manifest or indexed candidate is invalid."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_digest(label: str, value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RunManifestError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_text(label: str, value: object) -> str:
    if type(value) is not str or not value:
        raise RunManifestError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RunManifestError(f"{label} must be valid UTF-8") from exc
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RunManifestError("run JSON contains invalid UTF-8 text") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunManifestError("run JSON contains an unsupported or non-finite value") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RunManifestError("run JSON contains duplicate object keys")
        value[key] = item
    return value


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RunManifestError("run JSON contains a non-finite number")
    return number


def _decode_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunManifestError(f"{label} is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                RunManifestError("run JSON contains a non-finite number")
            ),
            parse_float=_finite_float,
        )
    except RunManifestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise RunManifestError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise RunManifestError(f"{label} must be a JSON object")
    return value


def _freeze_json(value: object, *, label: str) -> object:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RunManifestError(f"{label} contains a non-finite number")
        return value
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RunManifestError(f"{label} contains invalid UTF-8 text") from exc
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RunManifestError(f"{label} object keys must be strings")
            frozen[key] = _freeze_json(item, label=label)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, label=label) for item in value)
    raise RunManifestError(f"{label} contains an unsupported value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {cast(str, key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RunManifestError(
            f"{label} keys do not match schema; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _validate_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise RunManifestError(f"{label} is unsafe")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RunManifestError(f"{label} must be valid UTF-8") from exc
    if unicodedata.normalize("NFC", value) != value or _WINDOWS_DRIVE_RE.match(value) is not None:
        raise RunManifestError(f"{label} is unsafe")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.parts != tuple(value.split("/"))
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "/".join(pure.parts) != value
    ):
        raise RunManifestError(f"{label} is unsafe")
    return value


def _path_key(value: str) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in value.split("/"))


def _validate_candidate_bundle_path(value: str) -> str:
    value = _validate_relative_path(value, label="run result bundle_path")
    parts = value.split("/")
    if len(parts) != 2 or parts[0] != "candidates":
        raise RunManifestError("run result bundle_path must be one direct child of candidates/")
    return value


def _path_keys_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    shared = min(len(left), len(right))
    return left[:shared] == right[:shared]


def _ensure_distinct_links(snapshot_path: str, results_path: str) -> None:
    snapshot_key = _path_key(snapshot_path)
    results_key = _path_key(results_path)
    if _path_keys_overlap(snapshot_key, results_key):
        raise RunManifestError("snapshot_path and results_path collide")
    candidates_key = _path_key("candidates")
    manifest_key = _path_key(RUN_MANIFEST_NAME)
    for label, link_key in (("snapshot_path", snapshot_key), ("results_path", results_key)):
        if _path_keys_overlap(link_key, candidates_key):
            raise RunManifestError(f"{label} collides with the candidate bundle namespace")
        if _path_keys_overlap(link_key, manifest_key):
            raise RunManifestError(f"{label} collides with the run manifest")


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    try:
        absolute = Path(os.path.abspath(path))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunManifestError(f"{label} path is unavailable") from exc
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise RunManifestError(f"{label} path cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RunManifestError(f"{label} contains a symlink")
    return absolute


def _existing_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_symlinks(path, label=label)
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise RunManifestError(f"{label} cannot be read") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise RunManifestError(f"{label} must be a non-symlink directory")
    return absolute


def _regular_stat(path: Path, *, label: str) -> os.stat_result:
    absolute = _absolute_without_symlinks(path, label=label)
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise RunManifestError(f"{label} cannot be read") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RunManifestError(f"{label} must be a non-symlink regular file")
    return info


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


def _stat_seal(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_regular(path: Path, *, label: str) -> bytes:
    before = _regular_stat(path, label=label)
    descriptor: int | None = None
    chunks: list[bytes] = []
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not _same_identity(before, opened) or opened.st_size != before.st_size:
            raise RunManifestError(f"{label} changed during read")
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        after_path = _regular_stat(path, label=label)
        if (
            not _same_identity(opened, after_open)
            or not _same_identity(before, after_path)
            or opened.st_size != after_open.st_size
            or before.st_mtime_ns != after_path.st_mtime_ns
            or before.st_ctime_ns != after_path.st_ctime_ns
        ):
            raise RunManifestError(f"{label} changed during read")
    except RunManifestError:
        raise
    except OSError as exc:
        raise RunManifestError(f"{label} cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise RunManifestError(f"{label} changed during read")
    return content


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError as exc:
        raise RunManifestError("run directory cannot be synced") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_exclusive(path: Path, content: bytes, *, label: str) -> os.stat_result:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        total = 0
        while total < len(content):
            total += os.write(descriptor, content[total:])
        os.fsync(descriptor)
        return os.fstat(descriptor)
    except FileExistsError as exc:
        raise RunManifestError(f"{label} already exists") from exc
    except OSError as exc:
        raise RunManifestError(f"{label} cannot be written") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reference_payload(reference: SnapshotReference) -> dict[str, object]:
    return {
        "snapshot_sha256": reference.snapshot_sha256,
        "task_id": reference.task_id,
        "task_sha256": reference.task_sha256,
    }


def _parse_reference(value: object) -> SnapshotReference:
    if type(value) is not dict:
        raise RunManifestError("snapshot reference must be an object")
    _require_exact_keys(value, {"snapshot_sha256", "task_id", "task_sha256"}, label="snapshot reference")
    try:
        return SnapshotReference(
            snapshot_sha256=cast(str, value["snapshot_sha256"]),
            task_id=cast(str, value["task_id"]),
            task_sha256=cast(str, value["task_sha256"]),
        )
    except (CandidateBundleError, TypeError, ValueError) as exc:
        raise RunManifestError("snapshot reference is invalid") from exc


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    snapshot_path: str
    snapshot_sha256: str
    results_path: str
    configuration: Mapping[str, object]
    configuration_sha256: str | None
    ordered_tasks: tuple[SnapshotReference, ...]
    run_fingerprint_sha256: str | None
    root: Path = field(default=Path("."), compare=False, repr=False)

    def __post_init__(self) -> None:
        _require_text("run_id", self.run_id)
        snapshot_path = _validate_relative_path(self.snapshot_path, label="snapshot_path")
        results_path = _validate_relative_path(self.results_path, label="results_path")
        _ensure_distinct_links(snapshot_path, results_path)
        snapshot_digest = _require_digest("snapshot_sha256", self.snapshot_sha256)
        if not isinstance(self.configuration, Mapping):
            raise TypeError("run configuration must be a mapping")
        frozen = _freeze_json(self.configuration, label="run configuration")
        if not isinstance(frozen, Mapping):
            raise AssertionError("mapping freeze returned a non-mapping")
        object.__setattr__(self, "configuration", frozen)
        configuration_digest = _sha256(_canonical_bytes(_thaw_json(frozen)))
        if self.configuration_sha256 is not None and self.configuration_sha256 != configuration_digest:
            raise RunManifestError("configuration_sha256 does not match configuration")
        object.__setattr__(self, "configuration_sha256", configuration_digest)

        tasks = tuple(self.ordered_tasks)
        if not tasks or any(type(reference) is not SnapshotReference for reference in tasks):
            raise RunManifestError("ordered_tasks must contain SnapshotReference values")
        task_ids: set[str] = set()
        for reference in tasks:
            if reference.snapshot_sha256 != snapshot_digest:
                raise RunManifestError("ordered task snapshot digest does not match run snapshot")
            if reference.task_id in task_ids:
                raise RunManifestError("ordered task IDs must be unique")
            task_ids.add(reference.task_id)
        object.__setattr__(self, "ordered_tasks", tasks)
        if not isinstance(self.root, Path):
            raise TypeError("run root must be a Path")

        fingerprint = self.compute_run_fingerprint_sha256()
        if self.run_fingerprint_sha256 is not None and self.run_fingerprint_sha256 != fingerprint:
            raise RunManifestError("run_fingerprint_sha256 does not match run semantics")
        object.__setattr__(self, "run_fingerprint_sha256", fingerprint)

    def compute_run_fingerprint_sha256(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "configuration_sha256": self.configuration_sha256,
                    "snapshot_sha256": self.snapshot_sha256,
                    "ordered_tasks": [_reference_payload(reference) for reference in self.ordered_tasks],
                }
            )
        )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema": RUN_MANIFEST_SCHEMA,
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "snapshot_path": self.snapshot_path,
            "snapshot_sha256": self.snapshot_sha256,
            "results_path": self.results_path,
            "configuration": _thaw_json(self.configuration),
            "configuration_sha256": self.configuration_sha256,
            "ordered_tasks": [_reference_payload(reference) for reference in self.ordered_tasks],
            "run_fingerprint_sha256": self.run_fingerprint_sha256,
        }

    def canonical_manifest_bytes(self) -> bytes:
        return _canonical_bytes(self.manifest_payload())


@dataclass(frozen=True, slots=True)
class RunResultRow:
    sequence: int
    candidate_id: str
    snapshot_reference: SnapshotReference
    bundle_path: str
    bundle_sha256: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise RunManifestError("run result sequence must be a non-negative integer")
        _require_text("run result candidate_id", self.candidate_id)
        if type(self.snapshot_reference) is not SnapshotReference:
            raise TypeError("run result snapshot_reference must be a SnapshotReference")
        _validate_candidate_bundle_path(self.bundle_path)
        _require_digest("run result bundle_sha256", self.bundle_sha256)

    def payload(self) -> dict[str, object]:
        return {
            "schema": RUN_RESULT_SCHEMA,
            "schema_version": RUN_RESULT_SCHEMA_VERSION,
            "sequence": self.sequence,
            "candidate_id": self.candidate_id,
            "snapshot_reference": _reference_payload(self.snapshot_reference),
            "bundle_path": self.bundle_path,
            "bundle_sha256": self.bundle_sha256,
        }

    def canonical_line(self) -> bytes:
        return _canonical_bytes(self.payload()) + b"\n"


def _manifest_from_payload(payload: dict[str, object], root: Path) -> RunManifest:
    expected = {
        "schema",
        "schema_version",
        "run_id",
        "snapshot_path",
        "snapshot_sha256",
        "results_path",
        "configuration",
        "configuration_sha256",
        "ordered_tasks",
        "run_fingerprint_sha256",
    }
    _require_exact_keys(payload, expected, label="run manifest")
    if (
        type(payload["schema"]) is not str
        or payload["schema"] != RUN_MANIFEST_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION
    ):
        raise RunManifestError("run manifest schema is unsupported")
    tasks = payload["ordered_tasks"]
    configuration = payload["configuration"]
    if type(tasks) is not list or type(configuration) is not dict:
        raise RunManifestError("run manifest tasks and configuration have invalid types")
    return RunManifest(
        run_id=cast(str, payload["run_id"]),
        snapshot_path=cast(str, payload["snapshot_path"]),
        snapshot_sha256=cast(str, payload["snapshot_sha256"]),
        results_path=cast(str, payload["results_path"]),
        configuration=configuration,
        configuration_sha256=cast(str, payload["configuration_sha256"]),
        ordered_tasks=tuple(_parse_reference(value) for value in tasks),
        run_fingerprint_sha256=cast(str, payload["run_fingerprint_sha256"]),
        root=root,
    )


def _result_from_payload(payload: dict[str, object]) -> RunResultRow:
    expected = {
        "schema",
        "schema_version",
        "sequence",
        "candidate_id",
        "snapshot_reference",
        "bundle_path",
        "bundle_sha256",
    }
    _require_exact_keys(payload, expected, label="run result")
    if (
        type(payload["schema"]) is not str
        or payload["schema"] != RUN_RESULT_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != RUN_RESULT_SCHEMA_VERSION
    ):
        raise RunManifestError("run result schema is unsupported")
    return RunResultRow(
        sequence=cast(int, payload["sequence"]),
        candidate_id=cast(str, payload["candidate_id"]),
        snapshot_reference=_parse_reference(payload["snapshot_reference"]),
        bundle_path=cast(str, payload["bundle_path"]),
        bundle_sha256=cast(str, payload["bundle_sha256"]),
    )


def write_run_manifest(destination: Path, manifest: RunManifest) -> RunManifest:
    """Exclusively write ``run-manifest.json`` beneath an existing run root."""

    if not isinstance(destination, Path):
        raise TypeError("run destination must be a Path")
    if type(manifest) is not RunManifest:
        raise TypeError("manifest must be a RunManifest")
    root = _existing_directory(destination, label="run root")
    rebound = RunManifest(
        run_id=manifest.run_id,
        snapshot_path=manifest.snapshot_path,
        snapshot_sha256=manifest.snapshot_sha256,
        results_path=manifest.results_path,
        configuration=manifest.configuration,
        configuration_sha256=manifest.configuration_sha256,
        ordered_tasks=manifest.ordered_tasks,
        run_fingerprint_sha256=manifest.run_fingerprint_sha256,
        root=root,
    )
    _write_exclusive(root / RUN_MANIFEST_NAME, rebound.canonical_manifest_bytes(), label="run manifest")
    _fsync_directory(root)
    return rebound


def load_run_manifest(root: Path) -> RunManifest:
    """Load one canonical run manifest from an explicit run root."""

    if not isinstance(root, Path):
        raise TypeError("run root must be a Path")
    root = _existing_directory(root, label="run root")
    raw = _read_regular(root / RUN_MANIFEST_NAME, label="run manifest")
    manifest = _manifest_from_payload(_decode_object(raw, label="run manifest"), root)
    if raw != manifest.canonical_manifest_bytes():
        raise RunManifestError("run manifest is not canonical")
    snapshot = _absolute_without_symlinks(
        root.joinpath(*manifest.snapshot_path.split("/")),
        label="run snapshot",
    )
    try:
        snapshot_info = snapshot.lstat()
    except OSError as exc:
        raise RunManifestError("run snapshot cannot be read") from exc
    if not stat.S_ISDIR(snapshot_info.st_mode):
        raise RunManifestError("run snapshot must be a non-symlink directory")
    return manifest


_RUN_RESULT_WRITER_TOKEN = object()


class RunResultWriter:
    """Exclusive append-only writer for the sole candidate index."""

    __slots__ = ("_candidate_ids", "_lock", "_next_sequence", "_path", "_paths", "_seal")

    def __init__(self, path: Path, seal: tuple[int, int, int, int, int], *, _token: object) -> None:
        if _token is not _RUN_RESULT_WRITER_TOKEN:
            raise TypeError("RunResultWriter must be created by create")
        self._path = path
        self._seal = seal
        self._next_sequence = 0
        self._candidate_ids: set[str] = set()
        self._paths: set[tuple[str, ...]] = set()
        self._lock = threading.Lock()

    @classmethod
    def create(cls, path: Path) -> Self:
        if not isinstance(path, Path):
            raise TypeError("run results path must be a Path")
        absolute = _absolute_without_symlinks(path, label="run results")
        parent = _existing_directory(absolute.parent, label="run results parent")
        info = _write_exclusive(absolute, b"", label="run results")
        _fsync_directory(parent)
        return cls(absolute, _stat_seal(info), _token=_RUN_RESULT_WRITER_TOKEN)

    def _matches_path(self, path: Path) -> bool:
        """Return whether this live writer owns exactly ``path`` and it remains sealed."""

        if not isinstance(path, Path):
            return False
        try:
            absolute = _absolute_without_symlinks(path, label="run results")
            current = _regular_stat(absolute, label="run results")
        except RunManifestError:
            return False
        return absolute == self._path and _stat_seal(current) == self._seal

    def append(self, row: RunResultRow) -> None:
        if type(row) is not RunResultRow:
            raise TypeError("run result row must be a RunResultRow")
        content = row.canonical_line()
        path_key = _path_key(row.bundle_path)
        with self._lock:
            if row.sequence != self._next_sequence:
                raise RunManifestError("run result sequence is not the next monotonic value")
            if row.candidate_id in self._candidate_ids:
                raise RunManifestError("run result candidate_id is duplicated")
            if any(_path_keys_overlap(path_key, existing) for existing in self._paths):
                raise RunManifestError("run result bundle_path is duplicated or colliding")
            before = _regular_stat(self._path, label="run results")
            if _stat_seal(before) != self._seal:
                raise RunManifestError("run results changed outside the writer")

            descriptor: int | None = None
            try:
                descriptor = os.open(
                    self._path,
                    os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                )
                opened = os.fstat(descriptor)
                if _stat_seal(opened) != self._seal:
                    raise RunManifestError("run results changed outside the writer")
                total = 0
                while total < len(content):
                    total += os.write(descriptor, content[total:])
                os.fsync(descriptor)
                after = os.fstat(descriptor)
                after_path = _regular_stat(self._path, label="run results")
                if (
                    not _same_identity(after, before)
                    or not _same_identity(after_path, before)
                    or after.st_size != before.st_size + len(content)
                    or after_path.st_size != after.st_size
                    or after.st_mtime_ns != after_path.st_mtime_ns
                    or after.st_ctime_ns != after_path.st_ctime_ns
                ):
                    raise RunManifestError("run results changed during append")
            except RunManifestError:
                raise
            except OSError as exc:
                raise RunManifestError("run result cannot be appended") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)

            self._seal = _stat_seal(after_path)
            self._next_sequence += 1
            self._candidate_ids.add(row.candidate_id)
            self._paths.add(path_key)


def _read_result_rows(path: Path) -> tuple[RunResultRow, ...]:
    raw = _read_regular(path, label="run results")
    if not raw:
        return ()
    lines = raw.splitlines(keepends=True)
    rows: list[RunResultRow] = []
    for expected_sequence, line in enumerate(lines):
        if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
            raise RunManifestError("run results are not canonical JSONL")
        content = line[:-1]
        row = _result_from_payload(_decode_object(content, label="run result"))
        if content != row.canonical_line()[:-1]:
            raise RunManifestError("run result is not canonical")
        if row.sequence != expected_sequence:
            raise RunManifestError("run result sequence is not contiguous")
        rows.append(row)
    return tuple(rows)


def _validate_snapshot_binding(manifest: RunManifest, binding: VerifiedSnapshotBinding) -> None:
    for reference in manifest.ordered_tasks:
        try:
            binding.resolve(reference)
        except (CandidateBundleError, KeyError, TypeError, ValueError) as exc:
            raise RunManifestError("snapshot binding does not match the run manifest") from exc


def load_run_results(
    manifest: RunManifest,
    *,
    snapshot_binding: VerifiedSnapshotBinding,
) -> tuple[tuple[RunResultRow, CandidateBundle], ...]:
    """Load indexed bundles in durable ledger order using explicit links only."""

    if type(manifest) is not RunManifest:
        raise TypeError("manifest must be a RunManifest")
    if type(snapshot_binding) is not VerifiedSnapshotBinding:
        raise TypeError("snapshot_binding must be a VerifiedSnapshotBinding")
    root = _existing_directory(manifest.root, label="run root")
    persisted = load_run_manifest(root)
    if persisted != manifest:
        raise RunManifestError("run manifest changed after it was loaded")
    manifest = persisted
    snapshot_root = root.joinpath(*manifest.snapshot_path.split("/"))
    try:
        matches_snapshot_root = snapshot_binding._matches_snapshot_root(snapshot_root)
    except CandidateBundleError as exc:
        raise RunManifestError("run snapshot binding is invalid") from exc
    if not matches_snapshot_root:
        raise RunManifestError("snapshot binding is not the run manifest's explicit snapshot path")
    _validate_snapshot_binding(manifest, snapshot_binding)
    allowed_references = set(manifest.ordered_tasks)
    results_path = root.joinpath(*manifest.results_path.split("/"))
    rows = _read_result_rows(results_path)

    loaded: list[tuple[RunResultRow, CandidateBundle]] = []
    candidate_ids: set[str] = set()
    bundle_paths: set[tuple[str, ...]] = set()
    for row in rows:
        if row.snapshot_reference not in allowed_references:
            raise RunManifestError("run result references an unplanned task")
        if row.candidate_id in candidate_ids:
            raise RunManifestError("run result candidate_id is duplicated")
        path_key = _path_key(row.bundle_path)
        if any(_path_keys_overlap(path_key, existing) for existing in bundle_paths):
            raise RunManifestError("run result bundle_path is duplicated or colliding")
        bundle_root = _absolute_without_symlinks(
            root.joinpath(*row.bundle_path.split("/")),
            label="indexed candidate bundle",
        )
        try:
            bundle = load_candidate_bundle(bundle_root, snapshot_binding=snapshot_binding)
        except (CandidateBundleError, OSError) as exc:
            raise RunManifestError("indexed candidate bundle is invalid") from exc
        if bundle.candidate_id != row.candidate_id:
            raise RunManifestError("run result candidate_id does not match candidate bundle")
        if bundle.snapshot_reference != row.snapshot_reference:
            raise RunManifestError("run result snapshot reference does not match candidate bundle")
        if bundle.bundle_sha256 != row.bundle_sha256:
            raise RunManifestError("run result digest does not match candidate bundle")
        candidate_ids.add(row.candidate_id)
        bundle_paths.add(path_key)
        loaded.append((row, bundle))
    return tuple(loaded)


__all__ = [
    "RUN_MANIFEST_NAME",
    "RUN_MANIFEST_SCHEMA",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RUN_RESULT_SCHEMA",
    "RUN_RESULT_SCHEMA_VERSION",
    "RunManifest",
    "RunManifestError",
    "RunResultRow",
    "RunResultWriter",
    "load_run_manifest",
    "load_run_results",
    "write_run_manifest",
]
