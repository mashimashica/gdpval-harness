# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from eval_harness.benchmarks.snapshot import Availability
from eval_harness.executors.base import TaskSpec


if TYPE_CHECKING:
    from eval_harness.benchmarks.snapshot import BenchmarkSnapshot, SnapshotTaskContent


@dataclass(frozen=True)
class BenchmarkTask:
    """Benchmark-owned task data split from the executor-facing TaskSpec."""

    execution: TaskSpec
    materialization: Mapping[str, object] = field(default_factory=dict)
    evaluation: Mapping[str, object] = field(default_factory=dict)


class Benchmark(ABC):
    """Minimum lifecycle shared by the benchmark integrations inspected by this harness."""

    name: str
    source: str | None = None
    source_availability: Availability = Availability.UNAVAILABLE
    revision: str | None = None
    revision_availability: Availability = Availability.UNAVAILABLE

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

    def snapshot_source_paths(self) -> Sequence[Path]:
        """Return local source files that must remain stable during acquisition."""

        return ()

    def snapshot_task(self, task: BenchmarkTask, workspace: Path) -> "SnapshotTaskContent":
        """Build sanitized snapshot views and stage any task input files.

        Concrete adapters override this hook when their source has additional
        acquisition semantics.  The default places materialized inputs under
        the neutral ``task_inputs`` namespace and does not publish them to the
        evaluator view implicitly.
        """

        from eval_harness.benchmarks.snapshot import SnapshotTaskContent

        materialized = tuple(str(item) for item in self.materialize(task, workspace))
        entries: list[tuple[str, Path]] = []
        for relative in materialized:
            logical = relative if relative.startswith("task_inputs/") else f"task_inputs/{relative}"
            entries.append((logical, workspace / relative))
        return SnapshotTaskContent(
            evaluation_data=dict(task.evaluation),
            files=entries,
        )

    def acquire_snapshot(self, limit: int, destination: Path) -> "BenchmarkSnapshot":
        """Acquire a verified, movable content-addressed benchmark snapshot."""

        from eval_harness.benchmarks.snapshot import acquire_snapshot

        return acquire_snapshot(self, limit, destination)

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        """Build the executor-facing task after benchmark inputs have been materialized."""
        del workspace, network_policy
        return task.execution
