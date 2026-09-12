# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, allowlisted reasoning-effort configuration shared by Codex paths."""

from __future__ import annotations

from typing import Literal, cast


ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ReasoningEffortOption = ReasoningEffort | None

REASONING_EFFORT_VALUES: tuple[ReasoningEffort, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
_REASONING_EFFORT_SET = frozenset(REASONING_EFFORT_VALUES)


def validate_reasoning_effort(value: object) -> ReasoningEffortOption:
    """Normalize one optional effort value without accepting arbitrary CLI config."""

    if value is None:
        return None
    if not isinstance(value, str) or value not in _REASONING_EFFORT_SET:
        allowed = ", ".join(REASONING_EFFORT_VALUES)
        raise ValueError(f"reasoning_effort must be one of {allowed}; got {value!r}")
    return cast(ReasoningEffort, value)


def validate_executor_reasoning_effort(executor: object, value: object) -> ReasoningEffortOption:
    """Reject an effort requested for an executor that cannot carry Codex config."""

    normalized = validate_reasoning_effort(value)
    name = getattr(executor, "name", executor)
    if normalized is not None and name != "codex":
        raise ValueError(f"reasoning_effort is only supported for the codex executor; got {name!r}")
    return normalized


__all__ = (
    "REASONING_EFFORT_VALUES",
    "ReasoningEffort",
    "ReasoningEffortOption",
    "validate_executor_reasoning_effort",
    "validate_reasoning_effort",
)
