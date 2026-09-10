# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    NO_DELIVERABLE = "no_deliverable"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    reference_files: tuple[str, ...] = ()
    reference_file_urls: tuple[str, ...] = ()
    sector: str = ""
    occupation: str = ""


@dataclass(frozen=True)
class ExecutionRequest:
    task: TaskSpec
    workspace: Path
    deliverables_dir: Path
    executor_dir: Path
    model: str | None = None
    timeout_seconds: float | None = None
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    executor: str
    executor_version: str | None
    invocation_mode: str
    auth_mode: str
    workspace: Path
    deliverables_dir: Path
    status: ExecutionStatus
    started_at: str
    finished_at: str
    exit_code: int | None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreflightResult:
    executor: str
    ok: bool
    version: str | None = None
    auth_mode: str | None = None
    details: Sequence[str] = ()


class Executor(ABC):
    """Agent runtime contract; policy providers and judges are separate concerns."""

    name: str
    invocation_mode: str

    @abstractmethod
    def preflight(self) -> PreflightResult:
        """Check local availability/auth without issuing a model request."""

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute exactly one task and return a normalized result."""
