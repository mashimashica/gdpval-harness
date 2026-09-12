# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.benchmarks.snapshot import Availability, SnapshotTaskContent, _decode_json_object
from eval_harness.executors.base import TaskSpec


_MATH_PROMPT = (
    "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{question}"
)


class AIME26Benchmark(Benchmark):
    name = "aime26"
    source = "huggingface:MathArena/aime_2026/train"
    source_availability = Availability.AVAILABLE
    # Upstream prepare.py currently loads MathArena/aime_2026 without a pinned dataset revision.
    revision = None
    revision_availability = Availability.UNAVAILABLE

    def __init__(self, *, root: Path, dataset_path: Path, prepare_script: Path) -> None:
        self.root = root
        self.dataset_path = dataset_path
        self.prepare_script = prepare_script

    def is_prepared(self) -> bool:
        return self.dataset_path.is_file()

    def snapshot_source_paths(self) -> tuple[Path, ...]:
        return (self.dataset_path,)

    def prepare(self) -> None:
        if self.is_prepared():
            return
        if not self.prepare_script.is_file():
            raise RuntimeError(f"AIME26 prepare script not found: {self.prepare_script}")
        result = subprocess.run([sys.executable, str(self.prepare_script)], cwd=self.root, check=False)
        if result.returncode != 0 or not self.is_prepared():
            raise RuntimeError("failed to prepare AIME26 benchmark data")

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        if limit <= 0:
            raise ValueError("benchmark task limit must be positive")
        tasks: list[BenchmarkTask] = []
        with self.dataset_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = _decode_json_object(line, label="AIME26 dataset row")
                question = row.get("question")
                expected_answer = row.get("expected_answer")
                if type(question) is not str or type(expected_answer) not in {str, int}:
                    raise RuntimeError("AIME26 task fields have invalid types")
                tasks.append(
                    BenchmarkTask(
                        execution=TaskSpec(
                            task_id=f"aime26-{index:02d}",
                            prompt=_MATH_PROMPT.format(question=question),
                        ),
                        evaluation={
                            "question": question,
                            "expected_answer": str(expected_answer),
                        },
                    )
                )
                if len(tasks) >= limit:
                    break
        if not tasks:
            raise RuntimeError(f"no AIME26 tasks found in {self.dataset_path}")
        return tasks

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        del task
        workspace.mkdir(parents=True, exist_ok=True)
        return []

    def snapshot_task(self, task: BenchmarkTask, workspace: Path) -> SnapshotTaskContent:
        del workspace
        if set(task.evaluation) != {"question", "expected_answer"}:
            raise ValueError("AIME26 evaluation metadata is invalid")
        return SnapshotTaskContent(
            evaluation_data=dict(task.evaluation),
        )
