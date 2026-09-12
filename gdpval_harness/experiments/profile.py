# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load generic experiment profiles and bind their explicit input sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from gdpval_harness.builders.base import BuilderInputBundle
from gdpval_harness.builders.inputs import load_builder_input_bundle
from gdpval_harness.experiments.base import (
    ExperimentArm,
    ExperimentInputSpec,
    ExperimentProfile,
    LoadedExperimentProfile,
)


_MAX_PROFILE_BYTES = 1 * 1024 * 1024
_GIT_HEAD_TIMEOUT_SECONDS = 10.0
_GIT_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

_PROFILE_KEYS = frozenset({"schema_version", "profile_id", "benchmark", "inputs", "arms"})
_INPUT_KEYS = frozenset({"input_id", "input_type", "source_revision", "revision_status", "allowed_files"})
_INPUT_OPTIONAL_KEYS = frozenset({"expected_bundle_sha256"})
_ARM_KEYS = frozenset({"arm_id", "builder_inputs"})


class _DuplicateObjectKey(ValueError):
    """Internal JSON parser error for duplicate object members."""


class _NonFiniteNumber(ValueError):
    """Internal JSON parser error for non-finite numeric constants."""


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateObjectKey("duplicate object member")
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> None:
    raise _NonFiniteNumber("non-finite JSON number")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteNumber("non-finite JSON number")
    return parsed


def _canonical_profile_path(source: Path | str) -> Path:
    try:
        path = Path(source)
    except (TypeError, ValueError) as exc:
        raise ValueError("experiment profile source must be a path") from exc

    try:
        metadata = path.lstat()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("experiment profile source must be an existing canonical file") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("experiment profile source must be an existing canonical non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("experiment profile source must be an existing canonical file") from exc
    if resolved != path:
        raise ValueError("experiment profile source must be canonical")
    return path


def _read_profile_bytes(source: Path | str) -> tuple[Path, bytes]:
    path = _canonical_profile_path(source)
    try:
        before = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError("experiment profile source changed while reading") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("experiment profile source must be an existing canonical non-symlink file")
    if before.st_size > _MAX_PROFILE_BYTES:
        raise ValueError("experiment profile source exceeds the 1 MiB limit")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        descriptor_before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(descriptor_before.st_mode)
            or not stat.S_ISREG(descriptor_before.st_mode)
            or descriptor_before.st_dev != before.st_dev
            or descriptor_before.st_ino != before.st_ino
            or descriptor_before.st_size != before.st_size
        ):
            raise ValueError("experiment profile source changed while opening")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read(_MAX_PROFILE_BYTES + 1)
            descriptor_after = os.fstat(handle.fileno())
    except (OSError, ValueError) as exc:
        raise ValueError("could not read experiment profile source") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)

    try:
        after = path.lstat()
    except (OSError, ValueError) as exc:
        raise ValueError("experiment profile source changed while reading") from exc
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or descriptor_after.st_size != len(content)
        or after.st_size != len(content)
        or len(content) > _MAX_PROFILE_BYTES
    ):
        raise ValueError("experiment profile source changed while reading")
    return path, content


def _require_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("experiment profile JSON object expected")
    return value


def _require_exact_keys(
    value: object, expected: frozenset[str], *, optional: frozenset[str] = frozenset()
) -> dict[str, Any]:
    object_value = _require_object(value)
    actual = frozenset(object_value)
    if not expected.issubset(actual) or not actual.issubset(expected | optional):
        raise ValueError("experiment profile object has unknown or missing members")
    return object_value


def _decode_profile(raw: bytes) -> ExperimentProfile:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("experiment profile must be strict UTF-8") from exc

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_non_finite_number,
            parse_float=_parse_finite_float,
        )
        _validate_json_strings(payload)
        profile_object = _require_exact_keys(payload, _PROFILE_KEYS)
        input_values = profile_object["inputs"]
        arm_values = profile_object["arms"]
        if not isinstance(input_values, list) or not isinstance(arm_values, list):
            raise ValueError("experiment profile inputs and arms must be arrays")

        input_specs = []
        for input_value in input_values:
            input_object = _require_exact_keys(input_value, _INPUT_KEYS, optional=_INPUT_OPTIONAL_KEYS)
            input_specs.append(
                ExperimentInputSpec(
                    input_id=input_object["input_id"],
                    input_type=input_object["input_type"],
                    source_revision=input_object["source_revision"],
                    revision_status=input_object["revision_status"],
                    allowed_files=input_object["allowed_files"],
                    expected_bundle_sha256=input_object.get("expected_bundle_sha256"),
                )
            )

        arms = []
        for arm_value in arm_values:
            arm_object = _require_exact_keys(arm_value, _ARM_KEYS)
            arms.append(
                ExperimentArm(
                    arm_id=arm_object["arm_id"],
                    builder_inputs=arm_object["builder_inputs"],
                )
            )

        return ExperimentProfile(
            schema_version=profile_object["schema_version"],
            profile_id=profile_object["profile_id"],
            benchmark=profile_object["benchmark"],
            inputs=tuple(input_specs),
            arms=tuple(arms),
        )
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid experiment profile JSON shape") from exc


def _validate_json_strings(value: object) -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("experiment profile strings must be strict UTF-8") from exc
    elif isinstance(value, list):
        for item in value:
            _validate_json_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_json_strings(key)
            _validate_json_strings(item)


def load_experiment_profile(source: Path | str) -> LoadedExperimentProfile:
    """Load and validate one immutable experiment profile file."""

    path, raw = _read_profile_bytes(source)
    profile = _decode_profile(raw)
    return LoadedExperimentProfile(profile=profile, source=path, sha256=hashlib.sha256(raw).hexdigest())


def _validate_source_bindings(
    profile: ExperimentProfile, source_roots: Mapping[str, Path | str]
) -> tuple[dict[str, Path | str], tuple[ExperimentInputSpec, ...]]:
    if not isinstance(profile, ExperimentProfile):
        raise TypeError("profile must be an ExperimentProfile")
    if not isinstance(source_roots, Mapping):
        raise TypeError("source_roots must be a mapping")

    specs = tuple(profile.inputs)
    expected_ids = tuple(spec.input_id for spec in specs)
    try:
        bound_ids = tuple(source_roots.keys())
        bound_id_set = set(bound_ids)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("source_roots must have a unique hashable key for every profile input") from exc
    if len(bound_ids) != len(bound_id_set):
        raise ValueError("source_roots contains duplicate input bindings")
    if bound_id_set != set(expected_ids):
        raise ValueError("source_roots keys must exactly match profile input IDs")

    bindings: dict[str, Path | str] = {}
    for input_id in expected_ids:
        try:
            root = source_roots[input_id]
            Path(root)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("source_roots contains an invalid input source") from exc
        bindings[input_id] = root
    return bindings, specs


def _git_head(source_root: Path | str, expected_revision: str) -> None:
    try:
        source_text = str(source_root)
    except Exception as exc:
        raise ValueError("could not verify experiment input revision") from exc

    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", source_text, "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_HEAD_TIMEOUT_SECONDS,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise ValueError("could not verify experiment input revision") from exc

    try:
        returncode = result.returncode
        output = result.stdout
    except AttributeError as exc:
        raise ValueError("could not verify experiment input revision") from exc
    if returncode != 0:
        raise ValueError("could not verify experiment input revision")
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if not isinstance(output, str):
        raise ValueError("could not verify experiment input revision")
    lines = output.splitlines()
    if len(lines) != 1 or _GIT_HEAD_PATTERN.fullmatch(lines[0]) is None or lines[0] != expected_revision:
        raise ValueError("experiment input source revision does not match local Git HEAD")


def _git_commit_blob(source_root: Path | str, source_revision: str, logical_path: str) -> bytes:
    try:
        source_text = str(source_root)
    except Exception as exc:
        raise ValueError("could not verify experiment input commit contents") from exc

    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    object_name = f"{source_revision}:./{logical_path}"
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", source_text, "cat-file", "blob", object_name],
            capture_output=True,
            timeout=_GIT_HEAD_TIMEOUT_SECONDS,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise ValueError("could not verify experiment input commit contents") from exc

    try:
        returncode = result.returncode
        output = result.stdout
    except AttributeError as exc:
        raise ValueError("could not verify experiment input commit contents") from exc
    if returncode != 0 or not isinstance(output, bytes):
        raise ValueError("could not verify experiment input commit contents")
    return output


def _validate_commit_blobs(bundle: BuilderInputBundle, source_root: Path | str, source_revision: str) -> None:
    for entry in bundle.manifest.files:
        content = _git_commit_blob(source_root, source_revision, entry.path)
        if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ValueError("experiment input source differs from its pinned Git revision")


def _validate_loaded_bundle(bundle: object, spec: ExperimentInputSpec) -> BuilderInputBundle:
    if not isinstance(bundle, BuilderInputBundle):
        raise ValueError("builder input loader returned an invalid bundle")
    manifest = bundle.manifest
    if manifest.input_id != spec.input_id or manifest.input_type != spec.input_type:
        raise ValueError("builder input bundle identity does not match the profile")
    if manifest.source_revision != spec.source_revision or manifest.revision_status != spec.revision_status:
        raise ValueError("builder input bundle revision does not match the profile")
    if spec.expected_bundle_sha256 is not None and manifest.bundle_sha256 != spec.expected_bundle_sha256:
        raise ValueError("builder input bundle hash does not match the profile")
    return bundle


def load_experiment_inputs(
    profile: ExperimentProfile, source_roots: Mapping[str, Path | str]
) -> Mapping[str, BuilderInputBundle]:
    """Resolve every profile input from its exact source binding."""

    bindings, specs = _validate_source_bindings(profile, source_roots)
    bundles: dict[str, BuilderInputBundle] = {}
    for spec in specs:
        source_root = bindings[spec.input_id]
        if spec.revision_status == "available":
            _git_head(source_root, spec.source_revision or "")
        bundle = load_builder_input_bundle(
            source_root,
            input_id=spec.input_id,
            input_type=spec.input_type,
            allowed_files=spec.allowed_files,
            source_revision=spec.source_revision,
        )
        if spec.revision_status == "available":
            _git_head(source_root, spec.source_revision or "")
        validated_bundle = _validate_loaded_bundle(bundle, spec)
        if spec.revision_status == "available":
            _validate_commit_blobs(validated_bundle, source_root, spec.source_revision or "")
        bundles[spec.input_id] = validated_bundle
    return MappingProxyType(bundles)


__all__ = ("load_experiment_inputs", "load_experiment_profile")
