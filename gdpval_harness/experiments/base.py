# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable contracts for generic experiment profiles and runs."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence


_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_STATUSES = {"completed", "failed", "interrupted"}


def _require_identifier(label: str, value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must match [a-z0-9][a-z0-9._-]{{0,63}}")
    return value


def _require_nonempty_text(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _as_tuple(label: str, value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence") from exc


def _path_collision_key(value: str) -> str:
    return unicodedata.normalize("NFC", value.casefold())


def _validate_allowed_file_path(path: object) -> tuple[str, ...]:
    if not isinstance(path, str):
        raise TypeError("experiment input allowed_files must contain strings")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"experiment input allowed file must be strict UTF-8: {path!r}") from exc

    pure = PurePosixPath(path)
    windows = PureWindowsPath(path)
    components = path.split("/")
    if (
        not path
        or "\x00" in path
        or "\\" in path
        or pure.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(component in {"", ".", ".."} for component in components)
        or pure.as_posix() != path
        or unicodedata.normalize("NFC", path) != path
    ):
        raise ValueError(f"experiment input allowed file must be a normalized relative POSIX path: {path!r}")
    return tuple(components)


def _normalize_allowed_files(value: object) -> tuple[str, ...]:
    paths = _as_tuple("experiment input allowed_files", value)
    if not paths:
        raise ValueError("experiment input allowed_files must be non-empty")

    previous_paths: dict[str, str] = {}
    previous_components: dict[tuple[tuple[str, ...], str], str] = {}
    canonical_paths: list[tuple[str, ...]] = []
    normalized_paths: list[str] = []

    for path in paths:
        components = _validate_allowed_file_path(path)
        normalized_path = path  # type: ignore[assignment]
        component_keys = tuple(_path_collision_key(component) for component in components)

        for depth, (component, component_key) in enumerate(zip(components, component_keys)):
            prefix = component_keys[:depth]
            collision_key = (prefix, component_key)
            previous_component = previous_components.get(collision_key)
            if previous_component is not None and previous_component != component:
                raise ValueError(
                    "experiment input allowed_files contain a Unicode/casefold component collision: "
                    f"{previous_component!r} and {component!r}"
                )
            previous_components[collision_key] = component

        full_key = "/".join(component_keys)
        previous_path = previous_paths.get(full_key)
        if previous_path is not None:
            raise ValueError(
                "experiment input allowed_files contain duplicate or Unicode/casefold-colliding paths: "
                f"{previous_path!r} and {path!r}"
            )
        previous_paths[full_key] = normalized_path
        normalized_paths.append(normalized_path)
        canonical_paths.append(component_keys)

    for index, path_parts in enumerate(canonical_paths):
        for other_index, other_parts in enumerate(canonical_paths):
            if (
                index != other_index
                and len(path_parts) < len(other_parts)
                and other_parts[: len(path_parts)] == path_parts
            ):
                raise ValueError(
                    "experiment input allowed_files cannot use a file path as a directory prefix: "
                    f"{normalized_paths[index]!r} and {normalized_paths[other_index]!r}"
                )

    return tuple(sorted(normalized_paths))


def _validate_revision(source_revision: object, revision_status: object) -> None:
    if not isinstance(revision_status, str) or revision_status not in {"available", "unavailable"}:
        raise ValueError("revision_status must be available or unavailable")
    if revision_status == "available":
        if not isinstance(source_revision, str) or not source_revision.strip():
            raise ValueError("available revision_status requires a non-empty source_revision")
    elif source_revision is not None:
        raise ValueError("unavailable revision_status requires source_revision=None")


def _normalize_path(label: str, value: object) -> Path:
    try:
        return Path(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be path-like") from exc


def _require_int(label: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_timeout(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a positive real number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return normalized


def _require_bool(label: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")
    return value


def _normalize_string_sequence(label: str, value: object) -> tuple[str, ...]:
    values = _as_tuple(label, value)
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{label} must contain strings")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class ExperimentInputSpec:
    input_id: str
    input_type: str
    source_revision: str | None
    revision_status: str
    allowed_files: Sequence[str]
    expected_bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("experiment input_id", self.input_id)
        _require_nonempty_text("experiment input_type", self.input_type)
        _validate_revision(self.source_revision, self.revision_status)
        object.__setattr__(self, "allowed_files", _normalize_allowed_files(self.allowed_files))
        if self.expected_bundle_sha256 is not None:
            _require_sha256("experiment input expected_bundle_sha256", self.expected_bundle_sha256)


@dataclass(frozen=True)
class ExperimentArm:
    arm_id: str
    builder_inputs: Sequence[str]

    def __post_init__(self) -> None:
        _require_identifier("experiment arm_id", self.arm_id)
        builder_inputs = _normalize_string_sequence("experiment arm builder_inputs", self.builder_inputs)
        for builder_input in builder_inputs:
            _require_identifier("experiment arm builder input", builder_input)
        if len(builder_inputs) != len(set(builder_inputs)):
            raise ValueError("experiment arm builder_inputs must not contain duplicates")
        object.__setattr__(self, "builder_inputs", builder_inputs)


@dataclass(frozen=True)
class ExperimentProfile:
    schema_version: int
    profile_id: str
    benchmark: str
    inputs: Sequence[ExperimentInputSpec]
    arms: Sequence[ExperimentArm]

    def __post_init__(self) -> None:
        if _require_int("experiment schema_version", self.schema_version) != 1:
            raise ValueError("experiment schema_version must equal 1")
        _require_identifier("experiment profile_id", self.profile_id)
        _require_identifier("experiment benchmark", self.benchmark)

        inputs = _as_tuple("experiment profile inputs", self.inputs)
        if not inputs:
            raise ValueError("experiment profile inputs must be non-empty")
        if any(not isinstance(item, ExperimentInputSpec) for item in inputs):
            raise TypeError("experiment profile inputs must contain ExperimentInputSpec instances")
        input_ids = tuple(item.input_id for item in inputs)
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("experiment profile input IDs must be unique")

        arms = _as_tuple("experiment profile arms", self.arms)
        if not arms:
            raise ValueError("experiment profile arms must be non-empty")
        if any(not isinstance(item, ExperimentArm) for item in arms):
            raise TypeError("experiment profile arms must contain ExperimentArm instances")
        arm_ids = tuple(item.arm_id for item in arms)
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("experiment profile arm IDs must be unique")

        defined_input_ids = set(input_ids)
        used_input_ids: set[str] = set()
        for arm in arms:
            if len(arm.builder_inputs) != len(set(arm.builder_inputs)):
                raise ValueError(f"experiment arm {arm.arm_id!r} references an input more than once")
            unknown = set(arm.builder_inputs) - defined_input_ids
            if unknown:
                raise ValueError(f"experiment arm {arm.arm_id!r} references undefined inputs: {sorted(unknown)!r}")
            used_input_ids.update(arm.builder_inputs)
        if used_input_ids != defined_input_ids:
            missing = sorted(defined_input_ids - used_input_ids)
            raise ValueError(f"experiment profile inputs are unused by every arm: {missing!r}")

        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "arms", arms)


@dataclass(frozen=True)
class LoadedExperimentProfile:
    profile: ExperimentProfile
    source: Path
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ExperimentProfile):
            raise TypeError("loaded experiment profile profile must be an ExperimentProfile")
        object.__setattr__(self, "source", _normalize_path("loaded experiment profile source", self.source))
        _require_sha256("loaded experiment profile sha256", self.sha256)


@dataclass(frozen=True)
class ExperimentRunSummary:
    profile_id: str
    benchmark: str
    out_dir: Path
    runtime_root: Path
    status: str
    task_count: int
    arm_count: int
    completed_applications: int

    def __post_init__(self) -> None:
        _require_identifier("experiment run profile_id", self.profile_id)
        _require_identifier("experiment run benchmark", self.benchmark)
        object.__setattr__(self, "out_dir", _normalize_path("experiment run out_dir", self.out_dir))
        object.__setattr__(self, "runtime_root", _normalize_path("experiment run runtime_root", self.runtime_root))
        if not isinstance(self.status, str) or self.status not in _RUN_STATUSES:
            raise ValueError("experiment run status must be completed, failed, or interrupted")
        task_count = _require_int("experiment run task_count", self.task_count, minimum=0)
        arm_count = _require_int("experiment run arm_count", self.arm_count, minimum=0)
        completed_applications = _require_int(
            "experiment run completed_applications", self.completed_applications, minimum=0
        )
        if completed_applications > task_count * arm_count:
            raise ValueError("completed_applications cannot exceed task_count * arm_count")


@dataclass(frozen=True)
class ExperimentRunConfig:
    builder_executor: str
    application_executor: str
    evaluator: str
    builder_model: str | None
    application_model: str | None
    builder_timeout_seconds: float
    application_timeout_seconds: float
    builder_network_enabled: bool
    application_network_enabled: bool
    limit: int
    order_seed: int

    def __post_init__(self) -> None:
        _require_identifier("experiment builder_executor", self.builder_executor)
        _require_identifier("experiment application_executor", self.application_executor)
        _require_identifier("experiment evaluator", self.evaluator)
        for label, value in (
            ("experiment builder_model", self.builder_model),
            ("experiment application_model", self.application_model),
        ):
            if value is not None:
                _require_nonempty_text(label, value)
        object.__setattr__(
            self,
            "builder_timeout_seconds",
            _require_timeout("experiment builder_timeout_seconds", self.builder_timeout_seconds),
        )
        object.__setattr__(
            self,
            "application_timeout_seconds",
            _require_timeout("experiment application_timeout_seconds", self.application_timeout_seconds),
        )
        object.__setattr__(
            self,
            "builder_network_enabled",
            _require_bool("experiment builder_network_enabled", self.builder_network_enabled),
        )
        object.__setattr__(
            self,
            "application_network_enabled",
            _require_bool("experiment application_network_enabled", self.application_network_enabled),
        )
        _require_int("experiment limit", self.limit, minimum=1)
        _require_int("experiment order_seed", self.order_seed, minimum=0)
        if self.order_seed > 2**63 - 1:
            raise ValueError("experiment order_seed must be at most 2**63 - 1")


__all__ = (
    "ExperimentInputSpec",
    "ExperimentArm",
    "ExperimentProfile",
    "LoadedExperimentProfile",
    "ExperimentRunSummary",
    "ExperimentRunConfig",
)
