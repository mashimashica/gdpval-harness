# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable, secret-free failure records for execution and evaluation runs.

The failure boundary intentionally contains a small code rather than an
exception message.  Callers can persist a :class:`Failure` without copying a
credential, prompt, or provider response into run metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class FailureKind(StrEnum):
    """The phase or category that caused a run to stop."""

    AUTH = "auth"
    QUOTA = "quota"
    PROTOCOL = "protocol"
    INTEGRITY = "integrity"
    PROCESS = "process"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    INTERRUPTED = "interrupted"
    INTERNAL = "internal"


class FailureImpact(StrEnum):
    """Whether a failure can be isolated to one task or stops the run."""

    TASK = "task"
    RUN = "run"


# A code is deliberately smaller than a diagnostic.  This shape policy is
# only validation, not redaction; callers must use static implementation-
# defined codes and never pass provider messages, paths, or credentials.
_MAX_STABLE_CODE_LENGTH = 128
_STABLE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_RUN_FAILURE_KINDS = frozenset(
    {
        FailureKind.AUTH,
        FailureKind.QUOTA,
        FailureKind.PROTOCOL,
        FailureKind.INTEGRITY,
    }
)


@dataclass(frozen=True, slots=True)
class Failure:
    """An immutable, serializable failure classification.

    Authentication, quota, protocol, and integrity failures are systemic:
    continuing another task could silently change the experiment.  The
    constructor therefore rejects an inconsistent ``TASK`` impact instead of
    silently repairing the caller's record.
    """

    kind: FailureKind
    code: str
    impact: FailureImpact

    def __post_init__(self) -> None:
        try:
            kind = FailureKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("failure kind must be a supported failure kind") from exc
        try:
            impact = FailureImpact(self.impact)
        except (TypeError, ValueError) as exc:
            raise ValueError("failure impact must be task or run") from exc
        if (
            not isinstance(self.code, str)
            or len(self.code) > _MAX_STABLE_CODE_LENGTH
            or _STABLE_CODE.fullmatch(self.code) is None
        ):
            raise ValueError("failure code must be a non-empty stable code without diagnostic text")
        if kind in _RUN_FAILURE_KINDS and impact is FailureImpact.TASK:
            raise ValueError(f"{kind.value} failures must have run impact")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "impact", impact)


class RunAbort(RuntimeError):
    """Signal a systemic failure to the common runner."""

    failure: Failure

    def __init__(self, failure: Failure) -> None:
        if not isinstance(failure, Failure):
            raise TypeError("run abort requires a Failure")
        if failure.impact is not FailureImpact.RUN:
            raise ValueError("run abort requires a run-impact failure")
        self.failure = failure
        # Keep the exception text as stable as the persisted failure code.
        super().__init__(failure.code)


__all__ = ["Failure", "FailureImpact", "FailureKind", "RunAbort"]
