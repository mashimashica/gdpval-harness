# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable evaluator contracts for the generic local evaluation harness.

Benchmarks select and materialize tasks.  Evaluators consume the canonical task,
the execution result(s), and evaluator-only metadata after execution.  Keeping
these records separate prevents evaluator details from becoming executor input
or candidate deliverables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from gdpval_harness.executors.base import ExecutionResult


class EvaluatorType(StrEnum):
    DETERMINISTIC_EXACT = "deterministic-exact"
    BENCHMARK_NATIVE = "benchmark-native"
    EXECUTABLE_TESTS = "executable-tests"
    LLM_RUBRIC = "llm-rubric"
    PAIRWISE = "pairwise"

    # ``EXACT`` is a useful spelling for callers constructing an evaluator
    # directly.  It remains an alias of the stable descriptor value.
    EXACT = "deterministic-exact"


class EvaluationStatus(StrEnum):
    COMPLETED = "completed"
    EXTERNAL = "external"
    DEFERRED = "external"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class EvaluationCandidate:
    """One named candidate presented to an evaluator.

    ``artifacts_dir`` is the evaluator-facing artifact root.  It is optional
    because deterministic evaluators may only need ``ExecutionResult.output_text``;
    evaluators that stage files validate it before reading it.
    """

    candidate_id: str
    execution: ExecutionResult
    artifacts_dir: Path | None = None

    @property
    def result(self) -> ExecutionResult:
        """Compatibility spelling for integrations that call this a result."""
        return self.execution


@dataclass(frozen=True)
class EvaluationPlan:
    """Immutable evaluator plan validated before any candidate is executed.

    ``metadata`` is shallow-frozen at construction.  Nested values retain their
    original semantics, while the top-level mapping cannot be changed after the
    plan is handed to an evaluator.
    """

    task_id: str
    task_prompt: str
    metadata: Mapping[str, object]
    candidate_count: int
    artifact_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.candidate_count < 0:
            raise ValueError("evaluation plan candidate_count must be non-negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class EvaluationRequest:
    """Evaluator input independent of any concrete :class:`Benchmark`."""

    task_id: str
    task_prompt: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    candidates: Sequence[EvaluationCandidate] = field(default_factory=tuple)
    artifact_dir: Path | None = None

    def __post_init__(self) -> None:
        # Freeze the sequence boundary so a mutable caller list cannot change
        # the request while an evaluator is running.
        normalized: list[EvaluationCandidate] = []
        for index, candidate in enumerate(self.candidates):
            if isinstance(candidate, EvaluationCandidate):
                normalized.append(candidate)
                continue
            if isinstance(candidate, ExecutionResult):
                normalized.append(
                    EvaluationCandidate(
                        candidate_id=f"candidate_{index}",
                        execution=candidate,
                        artifacts_dir=candidate.deliverables_dir,
                    )
                )
                continue
            raise TypeError("evaluation candidates must be EvaluationCandidate or ExecutionResult instances")
        object.__setattr__(self, "candidates", tuple(normalized))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def canonical_task_prompt(self) -> str:
        return self.task_prompt

    @property
    def evaluator_metadata(self) -> Mapping[str, object]:
        return self.metadata

    @property
    def evaluator_artifact_destination(self) -> Path | None:
        return self.artifact_dir


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation output with independent metric and outcome namespaces.

    Pairwise and rubric evaluators can therefore return structured outcomes
    while leaving ``metrics`` empty; no universal score is implied by this
    envelope.
    """

    task_id: str
    status: EvaluationStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    outcomes: Mapping[str, object] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "outcomes", dict(self.outcomes))
        object.__setattr__(self, "details", dict(self.details))


@dataclass(frozen=True)
class EvaluatorPreflightResult:
    """Result of evaluator readiness checks performed before execution."""

    name: str
    evaluator_type: EvaluatorType
    ok: bool
    version: str | None = None
    revision: str | None = None
    details: Sequence[str] = ()
    judge_executor: str | None = None
    judge_executor_version: str | None = None
    judge_auth_mode: str | None = None
    judge_model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", tuple(str(detail) for detail in self.details))

    @property
    def evaluator_id(self) -> str:
        return self.name

    @property
    def evaluator_name(self) -> str:
        return self.name

    @property
    def evaluator(self) -> str:
        return self.name


class Evaluator(ABC):
    """Evaluator axis of the benchmark × executor lifecycle."""

    name: str
    evaluator_type: EvaluatorType
    version: str | None = None
    revision: str | None = None

    @property
    def evaluator_id(self) -> str:
        return self.name

    @property
    def evaluator_name(self) -> str:
        return self.name

    @property
    def evaluator(self) -> str:
        return self.name

    @abstractmethod
    def validate_plan(self, plan: EvaluationPlan) -> None:
        """Validate evaluator metadata and candidate cardinality before execution."""

    @abstractmethod
    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        """Check evaluator dependencies and output readiness without execution."""

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate the already-completed candidate execution(s)."""


def require_candidates(request: EvaluationRequest, count: int) -> tuple[EvaluationCandidate, ...]:
    """Require exactly ``count`` candidates and matching task ids."""

    candidates = tuple(request.candidates)
    if len(candidates) != count:
        raise ValueError(f"evaluator {count}-candidate contract requires exactly {count} candidates")
    for candidate in candidates:
        if candidate.execution.task_id != request.task_id:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} task id {candidate.execution.task_id!r} "
                f"does not match request task id {request.task_id!r}"
            )
    return candidates


def require_one_candidate(request: EvaluationRequest) -> EvaluationCandidate:
    return require_candidates(request, 1)[0]


def require_two_candidates(request: EvaluationRequest) -> tuple[EvaluationCandidate, EvaluationCandidate]:
    first, second = require_candidates(request, 2)
    if first.candidate_id == second.candidate_id:
        raise ValueError("pairwise candidates must have distinct candidate ids")
    return first, second


__all__ = [
    "EvaluationCandidate",
    "EvaluationPlan",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationStatus",
    "Evaluator",
    "EvaluatorPreflightResult",
    "EvaluatorType",
    "require_candidates",
    "require_one_candidate",
    "require_two_candidates",
]
