# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from gdpval_harness.executors.base import ExecutionResult, TaskSpec


class EvaluatorType(StrEnum):
    BENCHMARK_NATIVE = "benchmark-native"
    EXECUTABLE_TESTS = "executable-tests"
    LLM_RUBRIC = "llm-rubric"
    PAIRWISE = "pairwise"


@dataclass(frozen=True)
class BenchmarkTask:
    """Benchmark-owned task data split from the executor-facing TaskSpec."""

    execution: TaskSpec
    materialization: Mapping[str, object] = field(default_factory=dict)
    evaluation: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Generic result envelope; metric names and meanings remain benchmark-owned."""

    task_id: str
    metrics: Mapping[str, float]
    details: Mapping[str, object] = field(default_factory=dict)


class Benchmark(ABC):
    """Minimum lifecycle shared by the benchmark integrations inspected by this harness."""

    name: str
    revision: str | None
    evaluator_type: EvaluatorType

    @abstractmethod
    def is_prepared(self) -> bool:
        """Return whether the benchmark task source is ready locally."""

    @abstractmethod
    def prepare(self) -> None:
        """Prepare benchmark task data without running an executor or evaluator."""

    @abstractmethod
    def load_tasks(self, limit: int) -> Sequence[BenchmarkTask]:
        """Select benchmark tasks and keep evaluator data outside TaskSpec."""

    @abstractmethod
    def materialize(self, task: BenchmarkTask, workspace: Path) -> Sequence[str]:
        """Materialize benchmark-owned executor inputs into a task workspace."""

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        """Build the executor-facing task after benchmark inputs have been materialized."""
        del workspace, network_policy
        return task.execution

    def evaluate(self, task: BenchmarkTask, result: ExecutionResult) -> BenchmarkEvaluation:
        """Evaluate one execution when the benchmark supports direct local evaluation."""
        raise NotImplementedError(f"{self.name} does not expose direct local evaluation through this adapter")
