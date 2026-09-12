# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Movable, content-addressed candidate bundles.

Candidate bundles are the only execution output accepted by the evaluation
layer.  They contain normalized, secret-free execution evidence and a private
content-addressed copy of the candidate artifacts.  Benchmark grading inputs
remain in a separately verified benchmark snapshot.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeVar, cast

from eval_harness.benchmarks.snapshot import (
    BenchmarkSnapshot,
    SnapshotError,
    SnapshotFile,
    SnapshotTask,
    VerifiedSnapshotAccess,
    open_verified_snapshot,
)
from eval_harness.capabilities import ExecutorCapabilities, ExecutorInput, ExecutorOutput
from eval_harness.executors.base import ExecutionStatus, TaskSpec
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.interventions.base import (
    ApplicationMapping,
    InterventionApplication,
    InterventionFile,
    InterventionManifest,
    InterventionType,
    canonical_manifest_bytes,
)
from eval_harness.reasoning import ReasoningEffortOption, validate_reasoning_effort


CANDIDATE_BUNDLE_SCHEMA: Final = "candidate-bundle"
CANDIDATE_BUNDLE_SCHEMA_VERSION: Final = 1
_MANIFEST_NAME: Final = "candidate-bundle.json"
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DRIVE_RE: Final = re.compile(r"^[A-Za-z]:")
_SUCCESS_STATUSES: Final = frozenset({ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE})


class CandidateBundleError(ValueError):
    """Raised when candidate content cannot be sealed, decoded, or verified."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_digest(label: str, value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise CandidateBundleError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_text(label: str, value: object, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not value:
        raise CandidateBundleError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CandidateBundleError(f"{label} must be valid UTF-8") from exc
    return value


def _require_string_or_none(label: str, value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise CandidateBundleError(f"{label} must be a string or null")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CandidateBundleError(f"{label} must be valid UTF-8") from exc
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
        raise CandidateBundleError("candidate JSON contains invalid UTF-8 text") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateBundleError("candidate JSON contains an unsupported or non-finite value") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateBundleError("candidate JSON contains duplicate object keys")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise CandidateBundleError("candidate JSON contains a non-finite number")
    return number


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CandidateBundleError(f"{label} is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CandidateBundleError("candidate JSON contains a non-finite number")
            ),
            parse_float=_finite_json_float,
        )
    except CandidateBundleError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise CandidateBundleError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise CandidateBundleError(f"{label} must be a JSON object")
    return value


def _freeze_json(value: object, *, label: str) -> object:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CandidateBundleError(f"{label} contains invalid UTF-8 text") from exc
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CandidateBundleError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise CandidateBundleError(f"{label} object keys must be strings")
            frozen[key] = _freeze_json(item, label=label)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, label=label) for item in value)
    raise CandidateBundleError(f"{label} contains an unsupported value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {cast(str, key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CandidateBundleError(f"{label} keys do not match schema; missing={missing!r}, extra={extra!r}")


def _collision_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _validate_logical_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise CandidateBundleError(f"{label} is unsafe")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CandidateBundleError(f"{label} must be valid UTF-8") from exc
    if unicodedata.normalize("NFC", value) != value or _WINDOWS_DRIVE_RE.match(value) is not None:
        raise CandidateBundleError(f"{label} is unsafe")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.parts != tuple(value.split("/"))
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "/".join(pure.parts) != value
    ):
        raise CandidateBundleError(f"{label} is unsafe")
    return value


class _LogicalFile(Protocol):
    @property
    def path(self) -> str: ...


_LogicalFileT = TypeVar("_LogicalFileT", bound=_LogicalFile)


def _validate_file_entries(entries: Sequence[_LogicalFileT], *, label: str) -> tuple[_LogicalFileT, ...]:
    incoming = tuple(entries)
    ordered = tuple(sorted(incoming, key=lambda item: item.path))
    exact: set[str] = set()
    folded: dict[str, str] = {}
    path_keys: set[str] = set()
    component_names: dict[tuple[str, str], str] = {}
    for item in ordered:
        _validate_logical_path(item.path, label=f"{label} path")
        if item.path in exact:
            raise CandidateBundleError(f"{label} contains a duplicate path")
        exact.add(item.path)
        folded_path = _collision_key(item.path)
        previous = folded.get(folded_path)
        if previous is not None and previous != item.path:
            raise CandidateBundleError(f"{label} contains a Unicode or casefold collision")
        folded[folded_path] = item.path
        parts = item.path.split("/")
        parent_key = ""
        for index, part in enumerate(parts):
            part_key = _collision_key(part)
            identity = (parent_key, part_key)
            previous_part = component_names.get(identity)
            if previous_part is not None and previous_part != part:
                raise CandidateBundleError(f"{label} contains a Unicode or casefold collision")
            component_names[identity] = part
            parent_key = f"{parent_key}/{part_key}" if parent_key else part_key
            if index < len(parts) - 1 and parent_key in path_keys:
                raise CandidateBundleError(f"{label} contains a file/directory prefix collision")
        path_keys.add(parent_key)
    return ordered


def _ensure_no_symlink_ancestors(path: Path, *, label: str) -> Path:
    try:
        absolute = Path(os.path.abspath(path))
    except (OSError, RuntimeError, ValueError) as exc:
        raise CandidateBundleError(f"{label} path is unavailable") from exc
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current /= part
        try:
            if current.is_symlink():
                raise CandidateBundleError(f"{label} contains a symlink")
        except OSError as exc:
            raise CandidateBundleError(f"{label} path cannot be inspected") from exc
        if not current.exists():
            break
    return absolute


def _paths_overlap(left: Path, right: Path) -> bool:
    left = Path(os.path.abspath(left))
    right = Path(os.path.abspath(right))
    return left == right or left in right.parents or right in left.parents


def _lstat_regular(path: Path, *, label: str) -> os.stat_result:
    _ensure_no_symlink_ancestors(path, label=label)
    try:
        info = path.lstat()
    except OSError as exc:
        raise CandidateBundleError(f"{label} cannot be read") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CandidateBundleError(f"{label} is not a regular file")
    return info


def _same_file_stat(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
        and first.st_ctime_ns == second.st_ctime_ns
    )


def _read_race_safe(path: Path, *, label: str) -> bytes:
    before = _lstat_regular(path, label=label)
    descriptor: int | None = None
    chunks: list[bytes] = []
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not _same_file_stat(before, opened):
            raise CandidateBundleError(f"{label} changed during read")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
        after_path = _lstat_regular(path, label=label)
        if not _same_file_stat(opened, after_open) or not _same_file_stat(before, after_path):
            raise CandidateBundleError(f"{label} changed during read")
    except CandidateBundleError:
        raise
    except OSError as exc:
        raise CandidateBundleError(f"{label} cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise CandidateBundleError(f"{label} changed during read")
    return content


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError as exc:
        raise CandidateBundleError("candidate directory cannot be synced") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_exclusive(path: Path, content: bytes, *, label: str) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CandidateBundleError(f"{label} already exists") from exc
    except OSError as exc:
        raise CandidateBundleError(f"{label} cannot be written") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class SnapshotReference:
    snapshot_sha256: str
    task_id: str
    task_sha256: str

    def __post_init__(self) -> None:
        _require_digest("snapshot reference snapshot_sha256", self.snapshot_sha256)
        _require_text("snapshot reference task_id", self.task_id)
        _require_digest("snapshot reference task_sha256", self.task_sha256)


_BOUND_ASSETS_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class BoundEvaluationAssets:
    """A capability restricted to one explicit snapshot-view file subset."""

    entries: tuple[SnapshotFile, ...]
    _task_id: str = field(repr=False, compare=False)
    _access: VerifiedSnapshotAccess = field(repr=False, compare=False)

    def __init__(
        self,
        entries: tuple[SnapshotFile, ...],
        task_id: str,
        access: VerifiedSnapshotAccess,
        *,
        _token: object,
    ) -> None:
        if _token is not _BOUND_ASSETS_TOKEN:
            raise TypeError("BoundEvaluationAssets must be created by VerifiedSnapshotBinding")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_task_id", task_id)
        object.__setattr__(self, "_access", access)
        self.__post_init__()

    @classmethod
    def _bind(
        cls,
        entries: tuple[SnapshotFile, ...],
        task_id: str,
        access: VerifiedSnapshotAccess,
    ) -> BoundEvaluationAssets:
        return cls(entries, task_id, access, _token=_BOUND_ASSETS_TOKEN)

    def __post_init__(self) -> None:
        if any(type(item) is not SnapshotFile for item in self.entries):
            raise CandidateBundleError("bound evaluation assets contain an invalid entry")
        normalized = _validate_file_entries(self.entries, label="bound evaluation assets")
        object.__setattr__(self, "entries", normalized)
        _require_text("bound evaluation asset task_id", self._task_id)

    def read_bytes(self, logical_path: str) -> bytes:
        logical_path = _validate_logical_path(logical_path, label="evaluation asset path")
        if not any(item.path == logical_path for item in self.entries):
            raise CandidateBundleError("evaluation asset is not in the bound allowlist")
        return self._access.read_view_file(self._task_id, logical_path, view="evaluation")

    def materialize(self, destination: Path) -> tuple[str, ...]:
        if not isinstance(destination, Path):
            raise TypeError("evaluation asset destination must be a Path")
        return self._access.materialize_view_subset(
            self._task_id,
            destination,
            view="evaluation",
            files=self.entries,
        )


_BOUND_VIEW_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class BoundEvaluationView:
    """Evaluator-only task projection bound to one verified snapshot."""

    reference: SnapshotReference
    canonical_task_prompt: str
    canonical_prompt_sha256: str
    evaluation_data: Mapping[str, object]
    evaluation_data_sha256: str
    evaluation_view_sha256: str
    allowed_task_inputs: BoundEvaluationAssets
    grader_assets: BoundEvaluationAssets

    def __init__(
        self,
        reference: SnapshotReference,
        canonical_task_prompt: str,
        canonical_prompt_sha256: str,
        evaluation_data: Mapping[str, object],
        evaluation_data_sha256: str,
        evaluation_view_sha256: str,
        allowed_task_inputs: BoundEvaluationAssets,
        grader_assets: BoundEvaluationAssets,
        *,
        _token: object,
    ) -> None:
        if _token is not _BOUND_VIEW_TOKEN:
            raise TypeError("BoundEvaluationView must be created by VerifiedSnapshotBinding")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "canonical_task_prompt", canonical_task_prompt)
        object.__setattr__(self, "canonical_prompt_sha256", canonical_prompt_sha256)
        object.__setattr__(self, "evaluation_data", evaluation_data)
        object.__setattr__(self, "evaluation_data_sha256", evaluation_data_sha256)
        object.__setattr__(self, "evaluation_view_sha256", evaluation_view_sha256)
        object.__setattr__(self, "allowed_task_inputs", allowed_task_inputs)
        object.__setattr__(self, "grader_assets", grader_assets)
        self.__post_init__()

    @classmethod
    def _bind(
        cls,
        *,
        reference: SnapshotReference,
        canonical_task_prompt: str,
        canonical_prompt_sha256: str,
        evaluation_data: Mapping[str, object],
        evaluation_data_sha256: str,
        evaluation_view_sha256: str,
        allowed_task_inputs: BoundEvaluationAssets,
        grader_assets: BoundEvaluationAssets,
    ) -> BoundEvaluationView:
        return cls(
            reference,
            canonical_task_prompt,
            canonical_prompt_sha256,
            evaluation_data,
            evaluation_data_sha256,
            evaluation_view_sha256,
            allowed_task_inputs,
            grader_assets,
            _token=_BOUND_VIEW_TOKEN,
        )

    def __post_init__(self) -> None:
        if type(self.reference) is not SnapshotReference:
            raise TypeError("bound evaluation view reference must be a SnapshotReference")
        prompt = _require_string_or_none("bound evaluation canonical prompt", self.canonical_task_prompt)
        if prompt is None:
            raise CandidateBundleError("bound evaluation canonical prompt must be a string")
        _require_digest("bound evaluation canonical_prompt_sha256", self.canonical_prompt_sha256)
        if _sha256(prompt.encode("utf-8")) != self.canonical_prompt_sha256:
            raise CandidateBundleError("bound evaluation canonical prompt digest does not match prompt")
        if not isinstance(self.evaluation_data, Mapping):
            raise TypeError("bound evaluation data must be a mapping")
        frozen = _freeze_json(self.evaluation_data, label="bound evaluation data")
        if not isinstance(frozen, Mapping):
            raise AssertionError("mapping freeze returned a non-mapping")
        object.__setattr__(self, "evaluation_data", frozen)
        _require_digest("bound evaluation data_sha256", self.evaluation_data_sha256)
        _require_digest("bound evaluation view_sha256", self.evaluation_view_sha256)
        if _sha256(_canonical_bytes(_thaw_json(frozen))) != self.evaluation_data_sha256:
            raise CandidateBundleError("bound evaluation data digest does not match data")
        if type(self.allowed_task_inputs) is not BoundEvaluationAssets:
            raise TypeError("allowed task inputs must be BoundEvaluationAssets")
        if type(self.grader_assets) is not BoundEvaluationAssets:
            raise TypeError("grader assets must be BoundEvaluationAssets")
        if (
            self.allowed_task_inputs._access is not self.grader_assets._access
            or self.allowed_task_inputs._task_id != self.reference.task_id
            or self.grader_assets._task_id != self.reference.task_id
        ):
            raise CandidateBundleError("bound evaluation assets do not match the referenced task")
        combined = (*self.allowed_task_inputs.entries, *self.grader_assets.entries)
        _validate_file_entries(combined, label="bound evaluation view assets")


_SNAPSHOT_BINDING_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSnapshotBinding:
    """Opaque access to one snapshot that was fully verified exactly once."""

    _access: VerifiedSnapshotAccess = field(repr=False, compare=False)

    def __init__(self, access: VerifiedSnapshotAccess, *, _token: object) -> None:
        if _token is not _SNAPSHOT_BINDING_TOKEN:
            raise TypeError("VerifiedSnapshotBinding must be created by load")
        object.__setattr__(self, "_access", access)

    @classmethod
    def load(cls, root: Path) -> VerifiedSnapshotBinding:
        if not isinstance(root, Path):
            raise TypeError("snapshot binding root must be a Path")
        return cls(open_verified_snapshot(root), _token=_SNAPSHOT_BINDING_TOKEN)

    def reference(self, task_id: str) -> SnapshotReference:
        task = self._task(task_id)
        snapshot_digest = _require_digest("snapshot digest", self._snapshot().snapshot_sha256)
        task_digest = _require_digest("snapshot task digest", task.task_sha256)
        return SnapshotReference(snapshot_digest, task.task_id, task_digest)

    def _task(self, task_id: str) -> SnapshotTask:
        _require_text("snapshot task_id", task_id)
        try:
            return self._snapshot().task(task_id)
        except KeyError as exc:
            raise CandidateBundleError("snapshot reference task was not found") from exc

    def _snapshot(self) -> BenchmarkSnapshot:
        try:
            return self._access.snapshot
        except SnapshotError as exc:
            raise CandidateBundleError("snapshot binding seal is invalid") from exc

    def resolve(self, ref: SnapshotReference) -> SnapshotTask:
        if type(ref) is not SnapshotReference:
            raise TypeError("snapshot binding reference must be a SnapshotReference")
        snapshot = self._snapshot()
        if snapshot.snapshot_sha256 != ref.snapshot_sha256:
            raise CandidateBundleError("snapshot reference digest does not match binding")
        task = self._task(ref.task_id)
        if task.task_sha256 != ref.task_sha256:
            raise CandidateBundleError("snapshot task digest does not match reference")
        return task

    def bind_evaluation_view(self, ref: SnapshotReference) -> BoundEvaluationView:
        task = self.resolve(ref)
        execution_by_path = {item.path: item for item in task.execution_view.files}
        allowed = tuple(item for item in task.evaluation_view.files if execution_by_path.get(item.path) == item)
        grader = tuple(item for item in task.evaluation_view.files if execution_by_path.get(item.path) != item)
        data = task.evaluation_view.data
        if not isinstance(data, Mapping):
            raise CandidateBundleError("snapshot evaluation view data must be a mapping")
        prompt_digest = _require_digest("snapshot canonical prompt digest", task.canonical_prompt_sha256)
        data_digest = _require_digest("snapshot evaluation data digest", task.evaluation_view.data_sha256)
        view_digest = _require_digest("snapshot evaluation view digest", task.evaluation_view.view_sha256)
        return BoundEvaluationView._bind(
            reference=ref,
            canonical_task_prompt=task.canonical_task_prompt,
            canonical_prompt_sha256=prompt_digest,
            evaluation_data=data,
            evaluation_data_sha256=data_digest,
            evaluation_view_sha256=view_digest,
            allowed_task_inputs=BoundEvaluationAssets._bind(allowed, task.task_id, self._access),
            grader_assets=BoundEvaluationAssets._bind(grader, task.task_id, self._access),
        )

    def materialize_execution(self, ref: SnapshotReference, workspace: Path) -> tuple[str, ...]:
        """Materialize only the verified execution view for one referenced task."""

        task = self.resolve(ref)
        if not isinstance(workspace, Path):
            raise TypeError("execution workspace must be a Path")
        return self._access.materialize_view_subset(
            task.task_id,
            workspace,
            view="execution",
            files=task.execution_view.files,
        )

    def _reject_snapshot_destination(self, destination: Path) -> None:
        try:
            self._access._reject_contained_destination(destination)
        except SnapshotError as exc:
            raise CandidateBundleError("candidate destination conflicts with its snapshot") from exc

    def _matches_snapshot_root(self, root: Path) -> bool:
        try:
            return self._access._matches_root(root)
        except SnapshotError as exc:
            raise CandidateBundleError("snapshot binding seal is invalid") from exc


@dataclass(frozen=True, slots=True)
class BundleFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_logical_path(self.path, label="bundle file path")
        if type(self.size) is not int or self.size < 0:
            raise CandidateBundleError("bundle file size must be a non-negative integer")
        _require_digest("bundle file sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class ExecutorEvidence:
    executor_id: str
    executor_version: str | None
    runtime: str
    invocation_mode: str
    auth_mode: str | None
    requested_model: str | None
    model_id: str | None
    reasoning_effort_requested: ReasoningEffortOption
    effective_reasoning_effort: ReasoningEffortOption
    effective_reasoning_effort_available: bool
    declared_capabilities: ExecutorCapabilities
    started_at: str
    finished_at: str
    exit_code: int | None

    def __post_init__(self) -> None:
        _require_text("executor evidence executor_id", self.executor_id)
        _require_text("executor evidence executor_version", self.executor_version, optional=True)
        _require_text("executor evidence runtime", self.runtime)
        _require_text("executor evidence invocation_mode", self.invocation_mode)
        _require_text("executor evidence auth_mode", self.auth_mode, optional=True)
        _require_text("executor evidence requested_model", self.requested_model, optional=True)
        _require_text("executor evidence model_id", self.model_id, optional=True)
        object.__setattr__(
            self,
            "reasoning_effort_requested",
            validate_reasoning_effort(self.reasoning_effort_requested),
        )
        effective = validate_reasoning_effort(self.effective_reasoning_effort)
        if type(self.effective_reasoning_effort_available) is not bool:
            raise TypeError("executor evidence effective_reasoning_effort_available must be a bool")
        if not self.effective_reasoning_effort_available and effective is not None:
            raise CandidateBundleError("unavailable effective reasoning effort must be None")
        object.__setattr__(self, "effective_reasoning_effort", effective)
        if type(self.declared_capabilities) is not ExecutorCapabilities:
            raise TypeError("executor evidence declared_capabilities must be ExecutorCapabilities")
        _require_text("executor evidence started_at", self.started_at)
        _require_text("executor evidence finished_at", self.finished_at)
        if self.exit_code is not None and (type(self.exit_code) is not int):
            raise TypeError("executor evidence exit_code must be an integer or None")


@dataclass(frozen=True, slots=True)
class InterventionEvidence:
    """Secret-free source and per-candidate intervention application evidence."""

    manifest: InterventionManifest
    application: InterventionApplication

    def __post_init__(self) -> None:
        if type(self.manifest) is not InterventionManifest:
            raise TypeError("intervention evidence manifest must be an InterventionManifest")
        if type(self.application) is not InterventionApplication:
            raise TypeError("intervention evidence application must be an InterventionApplication")
        manifest = self.manifest
        application = self.application
        try:
            intervention_type = InterventionType(manifest.intervention_type)
        except (TypeError, ValueError) as exc:
            raise CandidateBundleError("intervention evidence type is unsupported") from exc
        _require_text("intervention evidence intervention_id", manifest.intervention_id)
        revision = _require_text(
            "intervention evidence source_revision",
            manifest.source_revision,
            optional=True,
        )
        revision_status = _require_text("intervention evidence revision_status", manifest.revision_status)
        if revision_status not in {"available", "unavailable", "not-applicable"}:
            raise CandidateBundleError("intervention evidence revision_status is unsupported")
        if (revision_status == "available") != (revision is not None):
            raise CandidateBundleError("intervention evidence revision does not match its availability")
        files = _validate_intervention_files(manifest.files, label="intervention source files")
        if files != tuple(manifest.files):
            raise CandidateBundleError("intervention source files must be in canonical order")
        _require_digest("intervention bundle_sha256", manifest.bundle_sha256)
        manifest_digest = _require_digest("intervention manifest_sha256", manifest.manifest_sha256)
        _validate_application_mapping(manifest.application, intervention_type=intervention_type)
        if _sha256(canonical_manifest_bytes(manifest)) != manifest_digest:
            raise CandidateBundleError("intervention manifest digest does not match its contents")
        _require_text("intervention application_run_id", application.application_run_id)
        if type(application.task) is not TaskSpec:
            raise TypeError("intervention application task must be a TaskSpec")
        materialized = _validate_intervention_files(
            application.materialized_files,
            label="intervention materialized files",
        )
        if materialized != tuple(application.materialized_files):
            raise CandidateBundleError("intervention materialized files must be in canonical order")
        if application.bundle_sha256 != manifest.bundle_sha256:
            raise CandidateBundleError("intervention application bundle digest does not match manifest")
        if application.manifest_sha256 != manifest_digest:
            raise CandidateBundleError("intervention application manifest digest does not match manifest")
        if application.application != manifest.application:
            raise CandidateBundleError("intervention application mapping does not match manifest")
        _validate_application_mapping(application.application, intervention_type=intervention_type)
        _validate_materialized_intervention_files(
            intervention_type,
            manifest.files,
            materialized,
            target=manifest.application.target,
        )
        if intervention_type is not manifest.intervention_type:
            raise CandidateBundleError("intervention manifest type is not normalized")


def _validate_intervention_files(
    values: Sequence[InterventionFile],
    *,
    label: str,
) -> tuple[InterventionFile, ...]:
    files = tuple(values)
    if any(type(item) is not InterventionFile for item in files):
        raise TypeError(f"{label} must contain InterventionFile values")
    for item in files:
        _validate_logical_path(item.path, label=f"{label} path")
        if type(item.size) is not int or item.size < 0:
            raise CandidateBundleError(f"{label} size must be a non-negative integer")
        _require_digest(f"{label} sha256", item.sha256)
    return cast(tuple[InterventionFile, ...], _validate_file_entries(files, label=label))


def _validate_application_mapping(
    value: ApplicationMapping,
    *,
    intervention_type: InterventionType | None = None,
) -> None:
    if type(value) is not ApplicationMapping:
        raise TypeError("intervention application mapping must be an ApplicationMapping")
    method = _require_text("intervention application method", value.method)
    target = _require_text("intervention application target", value.target, optional=True)
    if target is not None and target not in {".", "task.prompt"}:
        _validate_logical_path(target, label="intervention application target")
    if intervention_type is None:
        return
    exact_mappings: dict[InterventionType, tuple[str, str | None]] = {
        InterventionType.NONE: ("none", None),
        InterventionType.PROMPT_OVERLAY: ("prompt-overlay", "task.prompt"),
        InterventionType.FILES: ("workspace-files", "."),
    }
    expected = exact_mappings.get(intervention_type)
    if expected is not None and (method, target) != expected:
        raise CandidateBundleError("intervention type and application mapping do not match")
    if intervention_type is InterventionType.AGENT_SKILL and (
        method != "workspace-reference"
        or target in {None, ".", "task.prompt"}
        or PurePosixPath(cast(str, target)).name != "SKILL.md"
    ):
        raise CandidateBundleError("agent skill application mapping is invalid")


def _validate_materialized_intervention_files(
    intervention_type: InterventionType,
    source_files: Sequence[InterventionFile],
    materialized_files: tuple[InterventionFile, ...],
    *,
    target: str | None,
) -> None:
    if intervention_type in {InterventionType.NONE, InterventionType.PROMPT_OVERLAY}:
        expected: tuple[InterventionFile, ...] = ()
    elif intervention_type is InterventionType.FILES:
        expected = tuple(source_files)
    else:
        if target is None:
            raise CandidateBundleError("agent skill application target is missing")
        if not any(item.path == "SKILL.md" for item in source_files):
            raise CandidateBundleError("agent skill source manifest is missing SKILL.md")
        target_root = PurePosixPath(target).parent
        expected = tuple(
            InterventionFile(
                path=(target_root / PurePosixPath(item.path)).as_posix(),
                size=item.size,
                sha256=item.sha256,
            )
            for item in source_files
        )
    if materialized_files != expected:
        raise CandidateBundleError("intervention materialized files do not match the source application")


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    status: ExecutionStatus
    output_text: str | None
    available_outputs: frozenset[ExecutorOutput]
    artifacts: tuple[BundleFile, ...]
    failure: Failure | None

    def __post_init__(self) -> None:
        try:
            status = ExecutionStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise CandidateBundleError("candidate outcome status is unsupported") from exc
        if self.output_text is not None and type(self.output_text) is not str:
            raise TypeError("candidate output_text must be a string or None")
        if isinstance(self.output_text, str):
            try:
                self.output_text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CandidateBundleError("candidate output_text must be valid UTF-8") from exc
        try:
            outputs = frozenset(ExecutorOutput(value) for value in self.available_outputs)
        except (TypeError, ValueError) as exc:
            raise CandidateBundleError("candidate available_outputs contains an unsupported channel") from exc
        if any(type(item) is not BundleFile for item in self.artifacts):
            raise TypeError("candidate artifacts must contain BundleFile entries")
        artifacts = cast(tuple[BundleFile, ...], _validate_file_entries(self.artifacts, label="candidate artifacts"))
        if self.failure is not None and type(self.failure) is not Failure:
            raise TypeError("candidate failure must be a Failure or None")

        successful = status in _SUCCESS_STATUSES
        if successful and self.failure is not None:
            raise CandidateBundleError("successful candidate outcome cannot carry a failure")
        if not successful:
            if self.failure is None or self.failure.impact is not FailureImpact.RUN:
                raise CandidateBundleError("failed candidate outcome requires a run-impact failure")
            if self.output_text is not None or outputs or artifacts:
                raise CandidateBundleError("failed candidate outcome cannot expose output channels")
        if (ExecutorOutput.FINAL_TEXT in outputs) != (self.output_text is not None):
            raise CandidateBundleError("FINAL_TEXT availability must match output_text presence")
        if artifacts and ExecutorOutput.ARTIFACT_FILES not in outputs:
            raise CandidateBundleError("non-empty candidate artifacts require ARTIFACT_FILES availability")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "available_outputs", outputs)
        object.__setattr__(self, "artifacts", artifacts)


def _snapshot_reference_payload(reference: SnapshotReference) -> dict[str, object]:
    return {
        "snapshot_sha256": reference.snapshot_sha256,
        "task_id": reference.task_id,
        "task_sha256": reference.task_sha256,
    }


def _file_payload(item: BundleFile) -> dict[str, object]:
    return {"path": item.path, "size": item.size, "sha256": item.sha256}


def _capabilities_payload(capabilities: ExecutorCapabilities) -> dict[str, object]:
    return {
        "inputs": sorted(value.value for value in capabilities.inputs),
        "outputs": sorted(value.value for value in capabilities.outputs),
    }


def _failure_payload(failure: Failure | None) -> dict[str, object] | None:
    if failure is None:
        return None
    return {"kind": failure.kind.value, "code": failure.code, "impact": failure.impact.value}


def _executor_evidence_payload(evidence: ExecutorEvidence) -> dict[str, object]:
    return {
        "executor_id": evidence.executor_id,
        "executor_version": evidence.executor_version,
        "runtime": evidence.runtime,
        "invocation_mode": evidence.invocation_mode,
        "auth_mode": evidence.auth_mode,
        "requested_model": evidence.requested_model,
        "model_id": evidence.model_id,
        "reasoning_effort_requested": evidence.reasoning_effort_requested,
        "effective_reasoning_effort": evidence.effective_reasoning_effort,
        "effective_reasoning_effort_available": evidence.effective_reasoning_effort_available,
        "declared_capabilities": _capabilities_payload(evidence.declared_capabilities),
        "started_at": evidence.started_at,
        "finished_at": evidence.finished_at,
        "exit_code": evidence.exit_code,
    }


def _intervention_file_payload(item: InterventionFile) -> dict[str, object]:
    return {"path": item.path, "size": item.size, "sha256": item.sha256}


def _application_mapping_payload(mapping: ApplicationMapping) -> dict[str, object]:
    return {"method": mapping.method, "target": mapping.target}


def _intervention_evidence_payload(evidence: InterventionEvidence) -> dict[str, object]:
    manifest = evidence.manifest
    application = evidence.application
    return {
        "manifest": {
            "intervention_id": manifest.intervention_id,
            "intervention_type": manifest.intervention_type.value,
            "source_revision": manifest.source_revision,
            "revision_status": manifest.revision_status,
            "files": [_intervention_file_payload(item) for item in manifest.files],
            "bundle_sha256": manifest.bundle_sha256,
            "application": _application_mapping_payload(manifest.application),
            "manifest_sha256": manifest.manifest_sha256,
        },
        "application": {
            "application_run_id": application.application_run_id,
            "task_id": application.task.task_id,
            "materialized_files": [_intervention_file_payload(item) for item in application.materialized_files],
            "bundle_sha256": application.bundle_sha256,
            "manifest_sha256": application.manifest_sha256,
            "application": _application_mapping_payload(application.application),
        },
    }


def _outcome_payload(outcome: CandidateOutcome) -> dict[str, object]:
    return {
        "status": outcome.status.value,
        "output_text": outcome.output_text,
        "available_outputs": sorted(value.value for value in outcome.available_outputs),
        "artifacts": [_file_payload(item) for item in outcome.artifacts],
        "failure": _failure_payload(outcome.failure),
    }


_CANDIDATE_BUNDLE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class CandidateBundle:
    candidate_id: str
    snapshot_reference: SnapshotReference
    canonical_task_prompt: str
    canonical_prompt_sha256: str
    effective_executor_prompt: str
    effective_prompt_sha256: str
    executor_evidence: ExecutorEvidence
    intervention_evidence: InterventionEvidence
    outcome: CandidateOutcome
    bundle_sha256: str | None = None
    root: Path = field(default=Path("."), compare=False, repr=False)

    def __init__(
        self,
        candidate_id: str,
        snapshot_reference: SnapshotReference,
        canonical_task_prompt: str,
        canonical_prompt_sha256: str,
        effective_executor_prompt: str,
        effective_prompt_sha256: str,
        executor_evidence: ExecutorEvidence,
        intervention_evidence: InterventionEvidence,
        outcome: CandidateOutcome,
        bundle_sha256: str | None = None,
        root: Path = Path("."),
        *,
        _token: object,
    ) -> None:
        if _token is not _CANDIDATE_BUNDLE_TOKEN:
            raise TypeError("CandidateBundle must be created by seal_candidate_bundle or load_candidate_bundle")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "snapshot_reference", snapshot_reference)
        object.__setattr__(self, "canonical_task_prompt", canonical_task_prompt)
        object.__setattr__(self, "canonical_prompt_sha256", canonical_prompt_sha256)
        object.__setattr__(self, "effective_executor_prompt", effective_executor_prompt)
        object.__setattr__(self, "effective_prompt_sha256", effective_prompt_sha256)
        object.__setattr__(self, "executor_evidence", executor_evidence)
        object.__setattr__(self, "intervention_evidence", intervention_evidence)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "bundle_sha256", bundle_sha256)
        object.__setattr__(self, "root", root)
        self.__post_init__()

    @classmethod
    def _create(
        cls,
        *,
        candidate_id: str,
        snapshot_reference: SnapshotReference,
        canonical_task_prompt: str,
        canonical_prompt_sha256: str,
        effective_executor_prompt: str,
        effective_prompt_sha256: str,
        executor_evidence: ExecutorEvidence,
        intervention_evidence: InterventionEvidence,
        outcome: CandidateOutcome,
        bundle_sha256: str | None = None,
        root: Path = Path("."),
    ) -> CandidateBundle:
        return cls(
            candidate_id,
            snapshot_reference,
            canonical_task_prompt,
            canonical_prompt_sha256,
            effective_executor_prompt,
            effective_prompt_sha256,
            executor_evidence,
            intervention_evidence,
            outcome,
            bundle_sha256,
            root,
            _token=_CANDIDATE_BUNDLE_TOKEN,
        )

    def __post_init__(self) -> None:
        _require_text("candidate_id", self.candidate_id)
        if type(self.snapshot_reference) is not SnapshotReference:
            raise TypeError("candidate snapshot_reference must be a SnapshotReference")
        canonical = _require_string_or_none("candidate canonical task prompt", self.canonical_task_prompt)
        effective = _require_string_or_none("candidate effective executor prompt", self.effective_executor_prompt)
        if canonical is None or effective is None:
            raise CandidateBundleError("candidate prompts must be strings")
        _require_digest("candidate canonical_prompt_sha256", self.canonical_prompt_sha256)
        _require_digest("candidate effective_prompt_sha256", self.effective_prompt_sha256)
        if _sha256(canonical.encode("utf-8")) != self.canonical_prompt_sha256:
            raise CandidateBundleError("candidate canonical prompt digest does not match prompt")
        if _sha256(effective.encode("utf-8")) != self.effective_prompt_sha256:
            raise CandidateBundleError("candidate effective prompt digest does not match prompt")
        if type(self.executor_evidence) is not ExecutorEvidence:
            raise TypeError("candidate executor_evidence must be ExecutorEvidence")
        if type(self.intervention_evidence) is not InterventionEvidence:
            raise TypeError("candidate intervention_evidence must be InterventionEvidence")
        if self.intervention_evidence.application.task.task_id != self.snapshot_reference.task_id:
            raise CandidateBundleError("intervention application task does not match candidate task")
        if self.intervention_evidence.application.task.prompt != effective:
            raise CandidateBundleError("intervention application prompt does not match effective prompt")
        if type(self.outcome) is not CandidateOutcome:
            raise TypeError("candidate outcome must be CandidateOutcome")
        if not self.outcome.available_outputs <= self.executor_evidence.declared_capabilities.outputs:
            raise CandidateBundleError("candidate output channels exceed declared executor capabilities")
        if not isinstance(self.root, Path):
            raise TypeError("candidate root must be a Path")
        expected = self.compute_bundle_sha256()
        if self.bundle_sha256 is not None and self.bundle_sha256 != expected:
            raise CandidateBundleError("candidate bundle_sha256 does not match semantic manifest")
        object.__setattr__(self, "bundle_sha256", expected)

    def manifest_payload(self, *, include_bundle_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": CANDIDATE_BUNDLE_SCHEMA,
            "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "snapshot_reference": _snapshot_reference_payload(self.snapshot_reference),
            "canonical_task_prompt": self.canonical_task_prompt,
            "canonical_prompt_sha256": self.canonical_prompt_sha256,
            "effective_executor_prompt": self.effective_executor_prompt,
            "effective_prompt_sha256": self.effective_prompt_sha256,
            "executor_evidence": _executor_evidence_payload(self.executor_evidence),
            "intervention_evidence": _intervention_evidence_payload(self.intervention_evidence),
            "outcome": _outcome_payload(self.outcome),
        }
        if include_bundle_sha256:
            payload["bundle_sha256"] = self.bundle_sha256
        return payload

    def canonical_manifest_bytes(self, *, include_bundle_sha256: bool = True) -> bytes:
        return _canonical_bytes(self.manifest_payload(include_bundle_sha256=include_bundle_sha256))

    def compute_bundle_sha256(self) -> str:
        return _sha256(self.canonical_manifest_bytes(include_bundle_sha256=False))

    def read_artifact(self, logical_path: str) -> bytes:
        """Read and reverify one artifact from this bound bundle."""

        logical_path = _validate_logical_path(logical_path, label="candidate artifact path")
        entry = next((item for item in self.outcome.artifacts if item.path == logical_path), None)
        if entry is None:
            raise CandidateBundleError("candidate artifact is not present in the bundle")
        _verify_bundle_storage(self)
        content = _read_race_safe(self.root / "blobs" / "sha256" / entry.sha256, label="candidate blob")
        if len(content) != entry.size or _sha256(content) != entry.sha256:
            raise CandidateBundleError("candidate blob bytes do not match artifact manifest")
        _verify_bundle_storage(self)
        return content

    def materialize_artifacts(self, destination: Path) -> tuple[str, ...]:
        """Atomically materialize exactly this bundle's verified artifacts."""

        if not isinstance(destination, Path):
            raise TypeError("candidate artifact destination must be a Path")
        if ExecutorOutput.ARTIFACT_FILES not in self.outcome.available_outputs:
            raise CandidateBundleError("candidate ARTIFACT_FILES channel is absent")
        destination = _ensure_no_symlink_ancestors(destination, label="candidate artifact destination")
        if _paths_overlap(self.root, destination):
            raise CandidateBundleError("candidate artifact destination overlaps its bundle")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CandidateBundleError("candidate artifact destination parent cannot be created") from exc
        _ensure_no_symlink_ancestors(destination.parent, label="candidate artifact destination")
        if destination.exists() or destination.is_symlink():
            raise CandidateBundleError("candidate artifact destination must be fresh")
        _verify_bundle_storage(self)
        stage: Path | None = None
        try:
            stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
            for item in self.outcome.artifacts:
                target = stage.joinpath(*item.path.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                size, digest = _copy_source_file(self.root / "blobs" / "sha256" / item.sha256, target)
                if size != item.size or digest != item.sha256:
                    raise CandidateBundleError("candidate blob bytes do not match artifact manifest")
            _verify_bundle_storage(self)
            _fsync_directory(stage)
            if destination.exists() or destination.is_symlink():
                raise CandidateBundleError("candidate artifact destination must be fresh")
            os.rename(stage, destination)
            stage = None
            _fsync_directory(destination.parent)
            return tuple(item.path for item in self.outcome.artifacts)
        except CandidateBundleError:
            raise
        except OSError as exc:
            raise CandidateBundleError("candidate artifacts cannot be materialized") from exc
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class _TreeNode:
    path: str
    kind: Literal["directory", "file"]
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _scan_artifact_tree(root: Path) -> tuple[tuple[_TreeNode, ...], tuple[tuple[str, Path], ...]]:
    absolute = _ensure_no_symlink_ancestors(root, label="artifact source")
    try:
        root_info = absolute.lstat()
    except OSError as exc:
        raise CandidateBundleError("artifact source cannot be read") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise CandidateBundleError("artifact source must be a non-symlink directory")

    nodes: list[_TreeNode] = []
    files: list[tuple[str, Path]] = []
    logical_nodes: list[tuple[str, bool]] = []

    def visit(directory: Path, prefix: str) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise CandidateBundleError("artifact source cannot be traversed") from exc
        for entry in entries:
            logical = f"{prefix}/{entry.name}" if prefix else entry.name
            _validate_logical_path(logical, label="artifact path")
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CandidateBundleError("artifact source node cannot be inspected") from exc
            mode = info.st_mode
            if stat.S_ISLNK(mode):
                raise CandidateBundleError("artifact source cannot contain symlinks")
            if stat.S_ISDIR(mode):
                kind: Literal["directory", "file"] = "directory"
                logical_nodes.append((logical, False))
            elif stat.S_ISREG(mode):
                kind = "file"
                logical_nodes.append((logical, True))
                files.append((logical, Path(entry.path)))
            else:
                raise CandidateBundleError("artifact source can contain only directories and regular files")
            nodes.append(
                _TreeNode(
                    logical,
                    kind,
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            )
            if kind == "directory":
                visit(Path(entry.path), logical)

    visit(absolute, "")
    _validate_tree_paths(logical_nodes)
    root_node = _TreeNode(
        "",
        "directory",
        root_info.st_dev,
        root_info.st_ino,
        root_info.st_size,
        root_info.st_mtime_ns,
        root_info.st_ctime_ns,
    )
    return (root_node, *nodes), tuple(files)


def _validate_tree_paths(nodes: Sequence[tuple[str, bool]]) -> None:
    folded: dict[tuple[str, ...], tuple[tuple[str, ...], bool]] = {}
    for logical, is_file in nodes:
        parts = tuple(logical.split("/"))
        keys: list[str] = []
        for index, part in enumerate(parts):
            keys.append(_collision_key(part))
            key = tuple(keys)
            actual = parts[: index + 1]
            previous = folded.get(key)
            if previous is not None:
                previous_path, previous_is_file = previous
                if previous_path != actual:
                    raise CandidateBundleError("artifact tree contains a Unicode or casefold collision")
                if index < len(parts) - 1 and previous_is_file:
                    raise CandidateBundleError("artifact tree contains a file/directory prefix collision")
            if index == len(parts) - 1:
                if previous is not None:
                    raise CandidateBundleError("artifact tree contains a duplicate path")
                folded[key] = (actual, is_file)


def _copy_source_file(source: Path, temporary: Path) -> tuple[int, str]:
    before = _lstat_regular(source, label="artifact file")
    source_fd: int | None = None
    target_fd: int | None = None
    digest = hashlib.sha256()
    total = 0
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(source_fd)
        if not _same_file_stat(before, opened):
            raise CandidateBundleError("artifact file changed during seal")
        target_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(target_fd, chunk[offset:])
        os.fsync(target_fd)
        after_open = os.fstat(source_fd)
        after_path = _lstat_regular(source, label="artifact file")
        if (
            not _same_file_stat(opened, after_open)
            or not _same_file_stat(before, after_path)
            or total != opened.st_size
        ):
            raise CandidateBundleError("artifact file changed during seal")
        return total, digest.hexdigest()
    except CandidateBundleError:
        raise
    except OSError as exc:
        raise CandidateBundleError("artifact file cannot be sealed") from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)


def _stage_artifacts(
    artifacts_root: Path | None,
    available_outputs: frozenset[ExecutorOutput],
    digest_root: Path,
) -> tuple[BundleFile, ...]:
    has_artifact_channel = ExecutorOutput.ARTIFACT_FILES in available_outputs
    if not has_artifact_channel:
        if artifacts_root is not None:
            raise CandidateBundleError("artifacts_root requires ARTIFACT_FILES availability")
        return ()
    if artifacts_root is None:
        raise CandidateBundleError("ARTIFACT_FILES availability requires artifacts_root")
    before, files = _scan_artifact_tree(artifacts_root)
    manifests: list[BundleFile] = []
    for index, (logical, source) in enumerate(files):
        temporary = digest_root / f".incoming-{index:08d}"
        try:
            size, digest = _copy_source_file(source, temporary)
            target = digest_root / digest
            if target.exists() or target.is_symlink():
                existing = _read_race_safe(target, label="candidate blob")
                if len(existing) != size or _sha256(existing) != digest:
                    raise CandidateBundleError("candidate content-addressed blob collision")
                temporary.unlink()
            else:
                os.rename(temporary, target)
            manifests.append(BundleFile(logical, size, digest))
        finally:
            temporary.unlink(missing_ok=True)
    after, _ = _scan_artifact_tree(artifacts_root)
    if before != after:
        raise CandidateBundleError("artifact source changed during seal")
    _fsync_directory(digest_root)
    return cast(tuple[BundleFile, ...], _validate_file_entries(manifests, label="candidate artifacts"))


def _bundle_for_seal(
    *,
    root: Path,
    candidate_id: str,
    snapshot_binding: VerifiedSnapshotBinding,
    snapshot_reference: SnapshotReference,
    effective_executor_prompt: str,
    executor_evidence: ExecutorEvidence,
    intervention_evidence: InterventionEvidence,
    status: ExecutionStatus,
    output_text: str | None,
    available_outputs: frozenset[ExecutorOutput],
    failure: Failure | None,
    artifacts: tuple[BundleFile, ...],
) -> CandidateBundle:
    task = snapshot_binding.resolve(snapshot_reference)
    prompt_digest = _require_digest("snapshot canonical prompt digest", task.canonical_prompt_sha256)
    effective = _require_string_or_none("effective executor prompt", effective_executor_prompt)
    if effective is None:
        raise CandidateBundleError("effective executor prompt must be a string")
    return CandidateBundle._create(
        candidate_id=candidate_id,
        snapshot_reference=snapshot_reference,
        canonical_task_prompt=task.canonical_task_prompt,
        canonical_prompt_sha256=prompt_digest,
        effective_executor_prompt=effective,
        effective_prompt_sha256=_sha256(effective.encode("utf-8")),
        executor_evidence=executor_evidence,
        intervention_evidence=intervention_evidence,
        outcome=CandidateOutcome(status, output_text, available_outputs, artifacts, failure),
        root=root,
    )


def seal_candidate_bundle(
    *,
    destination: Path,
    candidate_id: str,
    snapshot_binding: VerifiedSnapshotBinding,
    snapshot_reference: SnapshotReference,
    effective_executor_prompt: str,
    executor_evidence: ExecutorEvidence,
    intervention_evidence: InterventionEvidence,
    status: ExecutionStatus,
    output_text: str | None,
    available_outputs: frozenset[ExecutorOutput],
    failure: Failure | None,
    artifacts_root: Path | None,
) -> CandidateBundle:
    """Seal one candidate into a fresh, movable content-addressed root."""

    if not isinstance(destination, Path):
        raise TypeError("candidate destination must be a Path")
    if type(snapshot_binding) is not VerifiedSnapshotBinding:
        raise TypeError("snapshot_binding must be a VerifiedSnapshotBinding")
    if type(snapshot_reference) is not SnapshotReference:
        raise TypeError("snapshot_reference must be a SnapshotReference")
    if type(executor_evidence) is not ExecutorEvidence:
        raise TypeError("executor_evidence must be ExecutorEvidence")
    if type(intervention_evidence) is not InterventionEvidence:
        raise TypeError("intervention_evidence must be InterventionEvidence")
    try:
        normalized_outputs = frozenset(ExecutorOutput(value) for value in available_outputs)
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError("available_outputs contains an unsupported channel") from exc
    if artifacts_root is not None and not isinstance(artifacts_root, Path):
        raise TypeError("artifacts_root must be a Path or None")

    destination = _ensure_no_symlink_ancestors(destination, label="candidate destination")
    snapshot_binding._reject_snapshot_destination(destination)
    if artifacts_root is not None:
        artifacts_root = _ensure_no_symlink_ancestors(artifacts_root, label="artifact source")
        if _paths_overlap(destination, artifacts_root):
            raise CandidateBundleError("candidate destination overlaps its artifact source")
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CandidateBundleError("candidate destination parent cannot be created") from exc
    _ensure_no_symlink_ancestors(parent, label="candidate destination")
    if destination.exists() or destination.is_symlink():
        raise CandidateBundleError("candidate destination must be fresh")

    stage: Path | None = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
        blobs_root = stage / "blobs"
        digest_root = blobs_root / "sha256"
        blobs_root.mkdir()
        digest_root.mkdir()
        artifacts = _stage_artifacts(artifacts_root, normalized_outputs, digest_root)
        bundle = _bundle_for_seal(
            root=stage,
            candidate_id=candidate_id,
            snapshot_binding=snapshot_binding,
            snapshot_reference=snapshot_reference,
            effective_executor_prompt=effective_executor_prompt,
            executor_evidence=executor_evidence,
            intervention_evidence=intervention_evidence,
            status=status,
            output_text=output_text,
            available_outputs=normalized_outputs,
            failure=failure,
            artifacts=artifacts,
        )
        _fsync_directory(blobs_root)
        _write_exclusive(stage / _MANIFEST_NAME, bundle.canonical_manifest_bytes(), label="candidate manifest")
        _fsync_directory(stage)
        if destination.exists() or destination.is_symlink():
            raise CandidateBundleError("candidate destination must be fresh")
        os.rename(stage, destination)
        stage = None
        _fsync_directory(parent)
        return CandidateBundle._create(
            candidate_id=bundle.candidate_id,
            snapshot_reference=bundle.snapshot_reference,
            canonical_task_prompt=bundle.canonical_task_prompt,
            canonical_prompt_sha256=bundle.canonical_prompt_sha256,
            effective_executor_prompt=bundle.effective_executor_prompt,
            effective_prompt_sha256=bundle.effective_prompt_sha256,
            executor_evidence=bundle.executor_evidence,
            intervention_evidence=bundle.intervention_evidence,
            outcome=bundle.outcome,
            bundle_sha256=bundle.bundle_sha256,
            root=destination,
        )
    except CandidateBundleError:
        raise
    except OSError as exc:
        raise CandidateBundleError("candidate bundle cannot be published") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _parse_snapshot_reference(value: object) -> SnapshotReference:
    if type(value) is not dict:
        raise CandidateBundleError("candidate snapshot_reference must be an object")
    _require_exact_keys(value, {"snapshot_sha256", "task_id", "task_sha256"}, label="snapshot_reference")
    return SnapshotReference(
        snapshot_sha256=cast(str, value["snapshot_sha256"]),
        task_id=cast(str, value["task_id"]),
        task_sha256=cast(str, value["task_sha256"]),
    )


def _parse_capabilities(value: object) -> ExecutorCapabilities:
    if type(value) is not dict:
        raise CandidateBundleError("declared_capabilities must be an object")
    _require_exact_keys(value, {"inputs", "outputs"}, label="declared_capabilities")
    inputs = value["inputs"]
    outputs = value["outputs"]
    if type(inputs) is not list or type(outputs) is not list:
        raise CandidateBundleError("declared capability channels must be arrays")
    try:
        return ExecutorCapabilities(
            inputs=frozenset(ExecutorInput(cast(str, item)) for item in inputs),
            outputs=frozenset(ExecutorOutput(cast(str, item)) for item in outputs),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError("declared capabilities contain an unsupported channel") from exc


def _parse_executor_evidence(value: object) -> ExecutorEvidence:
    if type(value) is not dict:
        raise CandidateBundleError("executor_evidence must be an object")
    expected = {
        "executor_id",
        "executor_version",
        "runtime",
        "invocation_mode",
        "auth_mode",
        "requested_model",
        "model_id",
        "reasoning_effort_requested",
        "effective_reasoning_effort",
        "effective_reasoning_effort_available",
        "declared_capabilities",
        "started_at",
        "finished_at",
        "exit_code",
    }
    _require_exact_keys(value, expected, label="executor_evidence")
    return ExecutorEvidence(
        executor_id=cast(str, value["executor_id"]),
        executor_version=cast(str | None, value["executor_version"]),
        runtime=cast(str, value["runtime"]),
        invocation_mode=cast(str, value["invocation_mode"]),
        auth_mode=cast(str | None, value["auth_mode"]),
        requested_model=cast(str | None, value["requested_model"]),
        model_id=cast(str | None, value["model_id"]),
        reasoning_effort_requested=validate_reasoning_effort(value["reasoning_effort_requested"]),
        effective_reasoning_effort=validate_reasoning_effort(value["effective_reasoning_effort"]),
        effective_reasoning_effort_available=cast(bool, value["effective_reasoning_effort_available"]),
        declared_capabilities=_parse_capabilities(value["declared_capabilities"]),
        started_at=cast(str, value["started_at"]),
        finished_at=cast(str, value["finished_at"]),
        exit_code=cast(int | None, value["exit_code"]),
    )


def _parse_intervention_file(value: object, *, label: str) -> InterventionFile:
    if type(value) is not dict:
        raise CandidateBundleError(f"{label} must be an object")
    _require_exact_keys(value, {"path", "size", "sha256"}, label=label)
    if type(value["size"]) is not int:
        raise CandidateBundleError(f"{label} size must be an integer")
    try:
        return InterventionFile(
            path=cast(str, value["path"]),
            size=value["size"],
            sha256=cast(str, value["sha256"]),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError(f"{label} is invalid") from exc


def _parse_application_mapping(value: object) -> ApplicationMapping:
    if type(value) is not dict:
        raise CandidateBundleError("intervention application mapping must be an object")
    _require_exact_keys(value, {"method", "target"}, label="intervention application mapping")
    try:
        return ApplicationMapping(
            method=cast(str, value["method"]),
            target=cast(str | None, value["target"]),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError("intervention application mapping is invalid") from exc


def _parse_intervention_evidence(value: object, *, effective_prompt: str) -> InterventionEvidence:
    if type(value) is not dict:
        raise CandidateBundleError("intervention_evidence must be an object")
    _require_exact_keys(value, {"manifest", "application"}, label="intervention_evidence")
    manifest_value = value["manifest"]
    application_value = value["application"]
    if type(manifest_value) is not dict or type(application_value) is not dict:
        raise CandidateBundleError("intervention evidence records must be objects")
    _require_exact_keys(
        manifest_value,
        {
            "intervention_id",
            "intervention_type",
            "source_revision",
            "revision_status",
            "files",
            "bundle_sha256",
            "application",
            "manifest_sha256",
        },
        label="intervention manifest",
    )
    _require_exact_keys(
        application_value,
        {
            "application_run_id",
            "task_id",
            "materialized_files",
            "bundle_sha256",
            "manifest_sha256",
            "application",
        },
        label="intervention application",
    )
    manifest_files = manifest_value["files"]
    materialized_files = application_value["materialized_files"]
    if type(manifest_files) is not list or type(materialized_files) is not list:
        raise CandidateBundleError("intervention evidence file manifests must be arrays")
    try:
        manifest = InterventionManifest(
            intervention_id=cast(str, manifest_value["intervention_id"]),
            intervention_type=InterventionType(cast(str, manifest_value["intervention_type"])),
            source_revision=cast(str | None, manifest_value["source_revision"]),
            revision_status=cast(str, manifest_value["revision_status"]),
            files=tuple(_parse_intervention_file(item, label="intervention source file") for item in manifest_files),
            bundle_sha256=cast(str, manifest_value["bundle_sha256"]),
            application=_parse_application_mapping(manifest_value["application"]),
            manifest_sha256=cast(str, manifest_value["manifest_sha256"]),
        )
        application = InterventionApplication(
            application_run_id=cast(str, application_value["application_run_id"]),
            task=TaskSpec(cast(str, application_value["task_id"]), effective_prompt),
            materialized_files=tuple(
                _parse_intervention_file(item, label="intervention materialized file") for item in materialized_files
            ),
            bundle_sha256=cast(str, application_value["bundle_sha256"]),
            manifest_sha256=cast(str, application_value["manifest_sha256"]),
            application=_parse_application_mapping(application_value["application"]),
        )
        return InterventionEvidence(manifest, application)
    except CandidateBundleError:
        raise
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError("intervention evidence is invalid") from exc


def _parse_failure(value: object) -> Failure | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise CandidateBundleError("candidate failure must be an object or null")
    _require_exact_keys(value, {"kind", "code", "impact"}, label="candidate failure")
    try:
        return Failure(
            kind=FailureKind(cast(str, value["kind"])),
            code=cast(str, value["code"]),
            impact=FailureImpact(cast(str, value["impact"])),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError("candidate failure is invalid") from exc


def _parse_bundle_file(value: object) -> BundleFile:
    if type(value) is not dict:
        raise CandidateBundleError("candidate artifact entry must be an object")
    _require_exact_keys(value, {"path", "size", "sha256"}, label="candidate artifact")
    return BundleFile(cast(str, value["path"]), cast(int, value["size"]), cast(str, value["sha256"]))


def _parse_outcome(value: object) -> CandidateOutcome:
    if type(value) is not dict:
        raise CandidateBundleError("candidate outcome must be an object")
    _require_exact_keys(
        value,
        {"status", "output_text", "available_outputs", "artifacts", "failure"},
        label="candidate outcome",
    )
    raw_outputs = value["available_outputs"]
    raw_artifacts = value["artifacts"]
    if type(raw_outputs) is not list or type(raw_artifacts) is not list:
        raise CandidateBundleError("candidate output channels and artifacts must be arrays")
    try:
        outputs = frozenset(ExecutorOutput(cast(str, item)) for item in raw_outputs)
        status = ExecutionStatus(cast(str, value["status"]))
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError("candidate outcome contains an unsupported enum value") from exc
    return CandidateOutcome(
        status=status,
        output_text=cast(str | None, value["output_text"]),
        available_outputs=outputs,
        artifacts=tuple(_parse_bundle_file(item) for item in raw_artifacts),
        failure=_parse_failure(value["failure"]),
    )


def _bundle_from_manifest(payload: dict[str, object], root: Path) -> CandidateBundle:
    expected = {
        "schema",
        "schema_version",
        "candidate_id",
        "snapshot_reference",
        "canonical_task_prompt",
        "canonical_prompt_sha256",
        "effective_executor_prompt",
        "effective_prompt_sha256",
        "executor_evidence",
        "intervention_evidence",
        "outcome",
        "bundle_sha256",
    }
    _require_exact_keys(payload, expected, label="candidate manifest")
    if (
        type(payload["schema"]) is not str
        or payload["schema"] != CANDIDATE_BUNDLE_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != CANDIDATE_BUNDLE_SCHEMA_VERSION
    ):
        raise CandidateBundleError("candidate manifest schema is unsupported")
    effective_prompt = cast(str, payload["effective_executor_prompt"])
    return CandidateBundle._create(
        candidate_id=cast(str, payload["candidate_id"]),
        snapshot_reference=_parse_snapshot_reference(payload["snapshot_reference"]),
        canonical_task_prompt=cast(str, payload["canonical_task_prompt"]),
        canonical_prompt_sha256=cast(str, payload["canonical_prompt_sha256"]),
        effective_executor_prompt=effective_prompt,
        effective_prompt_sha256=cast(str, payload["effective_prompt_sha256"]),
        executor_evidence=_parse_executor_evidence(payload["executor_evidence"]),
        intervention_evidence=_parse_intervention_evidence(
            payload["intervention_evidence"],
            effective_prompt=effective_prompt,
        ),
        outcome=_parse_outcome(payload["outcome"]),
        bundle_sha256=cast(str, payload["bundle_sha256"]),
        root=root,
    )


def _scan_bundle_root(root: Path, expected_blobs: set[str]) -> None:
    try:
        children = tuple(root.iterdir())
        if {item.name for item in children} != {_MANIFEST_NAME, "blobs"}:
            raise CandidateBundleError("candidate root contains unexpected content")
        manifest = root / _MANIFEST_NAME
        _lstat_regular(manifest, label="candidate manifest")
        blobs = root / "blobs"
        digest_root = blobs / "sha256"
        if blobs.is_symlink() or not blobs.is_dir() or digest_root.is_symlink() or not digest_root.is_dir():
            raise CandidateBundleError("candidate blob directory is invalid")
        if {item.name for item in blobs.iterdir()} != {"sha256"}:
            raise CandidateBundleError("candidate blob directory contains unexpected content")
        digest_files = tuple(digest_root.iterdir())
        if {item.name for item in digest_files} != expected_blobs:
            raise CandidateBundleError("candidate blob store does not match artifact manifest")
        for item in digest_files:
            if item.is_symlink() or not item.is_file() or _SHA256_RE.fullmatch(item.name) is None:
                raise CandidateBundleError("candidate blob store contains an invalid node")
    except CandidateBundleError:
        raise
    except OSError as exc:
        raise CandidateBundleError("candidate root cannot be inspected") from exc


def _verify_bundle_storage(bundle: CandidateBundle) -> None:
    raw = _read_race_safe(bundle.root / _MANIFEST_NAME, label="candidate manifest")
    if raw != bundle.canonical_manifest_bytes():
        raise CandidateBundleError("candidate manifest changed after verification")
    _scan_bundle_root(bundle.root, {item.sha256 for item in bundle.outcome.artifacts})


def load_candidate_bundle(root: Path, *, snapshot_binding: VerifiedSnapshotBinding) -> CandidateBundle:
    """Load and fully verify one bundle against an explicit snapshot binding."""

    if not isinstance(root, Path):
        raise TypeError("candidate root must be a Path")
    if type(snapshot_binding) is not VerifiedSnapshotBinding:
        raise TypeError("snapshot_binding must be a VerifiedSnapshotBinding")
    root = _ensure_no_symlink_ancestors(root, label="candidate root")
    try:
        info = root.lstat()
    except OSError as exc:
        raise CandidateBundleError("candidate root cannot be read") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CandidateBundleError("candidate root must be a non-symlink directory")
    raw = _read_race_safe(root / _MANIFEST_NAME, label="candidate manifest")
    bundle = _bundle_from_manifest(_decode_json_object(raw, label="candidate manifest"), root)
    if raw != bundle.canonical_manifest_bytes():
        raise CandidateBundleError("candidate manifest is not canonical")

    task = snapshot_binding.resolve(bundle.snapshot_reference)
    if task.canonical_task_prompt != bundle.canonical_task_prompt:
        raise CandidateBundleError("candidate canonical prompt does not match snapshot task")
    if task.canonical_prompt_sha256 != bundle.canonical_prompt_sha256:
        raise CandidateBundleError("candidate canonical prompt digest does not match snapshot task")

    expected: dict[str, int] = {}
    for item in bundle.outcome.artifacts:
        previous = expected.get(item.sha256)
        if previous is not None and previous != item.size:
            raise CandidateBundleError("candidate artifact digest has inconsistent sizes")
        expected[item.sha256] = item.size
    _scan_bundle_root(root, set(expected))
    for digest, size in expected.items():
        content = _read_race_safe(root / "blobs" / "sha256" / digest, label="candidate blob")
        if len(content) != size or _sha256(content) != digest:
            raise CandidateBundleError("candidate blob bytes do not match artifact manifest")
    return bundle


__all__ = [
    "BoundEvaluationAssets",
    "BoundEvaluationView",
    "BundleFile",
    "CANDIDATE_BUNDLE_SCHEMA",
    "CANDIDATE_BUNDLE_SCHEMA_VERSION",
    "CandidateBundle",
    "CandidateBundleError",
    "CandidateOutcome",
    "ExecutorEvidence",
    "InterventionEvidence",
    "SnapshotReference",
    "VerifiedSnapshotBinding",
    "load_candidate_bundle",
    "seal_candidate_bundle",
]
