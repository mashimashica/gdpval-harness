# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from eval_harness.reasoning import ReasoningEffortOption, validate_reasoning_effort


class Verdict(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"


@dataclass(frozen=True)
class JudgePreflightResult:
    judge_executor: str
    ok: bool
    version: str | None = None
    auth_mode: str | None = None
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgeRequest:
    task_id: str
    task_prompt: str
    workspace: Path
    reference_dir: Path
    submission_a_dir: Path
    submission_b_dir: Path
    executor_dir: Path
    trial_index: int
    swapped: bool
    model: str | None = None
    timeout_seconds: float = 3600.0
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResult:
    task_id: str
    trial_index: int
    judge_executor: str
    verdict: Verdict | None
    executor_version: str | None
    invocation_mode: str
    auth_mode: str
    started_at: str
    finished_at: str
    exit_code: int | None
    stdout_path: Path
    stderr_path: Path
    metadata: Mapping[str, object] = field(default_factory=dict)
    reasoning_effort_requested: ReasoningEffortOption = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reasoning_effort_requested",
            validate_reasoning_effort(self.reasoning_effort_requested),
        )


class JudgeExecutor(ABC):
    name: str
    invocation_mode: str

    @abstractmethod
    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        raise NotImplementedError

    @abstractmethod
    def judge(self, request: JudgeRequest) -> JudgeResult:
        raise NotImplementedError
