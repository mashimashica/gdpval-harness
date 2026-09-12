# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-independent executor input and output capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class ExecutorInput(StrEnum):
    """Inputs an executor may require from the harness."""

    PROMPT_TEXT = "prompt_text"
    WORKSPACE_FILES = "workspace_files"


class ExecutorOutput(StrEnum):
    """Output channels an executor may provide to the harness."""

    FINAL_TEXT = "final_text"
    ARTIFACT_FILES = "artifact_files"


def _normalize_inputs(values: Iterable[ExecutorInput]) -> frozenset[ExecutorInput]:
    try:
        return frozenset(ExecutorInput(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("executor inputs must use the supported capability names") from exc


def _normalize_outputs(values: Iterable[ExecutorOutput]) -> frozenset[ExecutorOutput]:
    try:
        return frozenset(ExecutorOutput(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("executor outputs must use the supported capability names") from exc


@dataclass(frozen=True, slots=True)
class ExecutionRequirements:
    """Capabilities an execution request needs from its selected executor."""

    inputs: frozenset[ExecutorInput]
    outputs: frozenset[ExecutorOutput]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _normalize_inputs(self.inputs))
        object.__setattr__(self, "outputs", _normalize_outputs(self.outputs))


@dataclass(frozen=True, slots=True)
class ExecutorCapabilities:
    """Capabilities supplied by one concrete executor implementation."""

    inputs: frozenset[ExecutorInput]
    outputs: frozenset[ExecutorOutput]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _normalize_inputs(self.inputs))
        object.__setattr__(self, "outputs", _normalize_outputs(self.outputs))


@dataclass(frozen=True, slots=True)
class CapabilityPreflightResult:
    """Pure capability comparison result produced before model execution."""

    ok: bool
    missing_inputs: tuple[ExecutorInput, ...]
    missing_outputs: tuple[ExecutorOutput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("capability preflight ok must be a bool")
        try:
            missing_inputs = tuple(ExecutorInput(value) for value in self.missing_inputs)
            missing_outputs = tuple(ExecutorOutput(value) for value in self.missing_outputs)
        except (TypeError, ValueError) as exc:
            raise ValueError("capability preflight missing channels must be supported") from exc
        if self.ok != (not missing_inputs and not missing_outputs):
            raise ValueError("capability preflight ok must match missing channels")
        object.__setattr__(self, "missing_inputs", missing_inputs)
        object.__setattr__(self, "missing_outputs", missing_outputs)


def preflight_capabilities(
    requirements: ExecutionRequirements,
    capabilities: ExecutorCapabilities,
) -> CapabilityPreflightResult:
    """Return missing channels without consulting a benchmark or invoking an executor."""

    if not isinstance(requirements, ExecutionRequirements):
        raise TypeError("requirements must be an ExecutionRequirements")
    if not isinstance(capabilities, ExecutorCapabilities):
        raise TypeError("capabilities must be an ExecutorCapabilities")
    missing_inputs = tuple(sorted(requirements.inputs - capabilities.inputs, key=lambda value: value.value))
    missing_outputs = tuple(sorted(requirements.outputs - capabilities.outputs, key=lambda value: value.value))
    return CapabilityPreflightResult(
        ok=not missing_inputs and not missing_outputs,
        missing_inputs=missing_inputs,
        missing_outputs=missing_outputs,
    )


__all__ = [
    "CapabilityPreflightResult",
    "ExecutionRequirements",
    "ExecutorCapabilities",
    "ExecutorInput",
    "ExecutorOutput",
    "preflight_capabilities",
]
