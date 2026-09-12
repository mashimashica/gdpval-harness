# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.benchmarks.snapshot import Availability, SnapshotTaskContent, _decode_json_object
from eval_harness.executors.base import TaskSpec


_PROMPT = "Generate an executable Python function generated from the given prompt.\n\n{question}"


class BigCodeBenchBenchmark(Benchmark):
    name = "bigcodebench"
    source = "huggingface:bigcode/bigcodebench-hard"
    source_availability = Availability.AVAILABLE
    revision = "v0.1.4"
    revision_availability = Availability.AVAILABLE

    def __init__(
        self,
        *,
        root: Path,
        dataset_path: Path,
        prepare_script: Path,
    ) -> None:
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
            raise RuntimeError(f"BigCodeBench prepare script not found: {self.prepare_script}")
        result = subprocess.run([sys.executable, str(self.prepare_script)], cwd=self.root, check=False)
        if result.returncode != 0 or not self.is_prepared():
            raise RuntimeError("failed to prepare BigCodeBench benchmark data")

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        if limit <= 0:
            raise ValueError("benchmark task limit must be positive")
        tasks: list[BenchmarkTask] = []
        with self.dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _decode_json_object(line, label="BigCodeBench dataset row")
                verifier_metadata = row.get("verifier_metadata")
                if not isinstance(verifier_metadata, dict):
                    raise RuntimeError("BigCodeBench task is missing verifier_metadata")
                for key in ("task_id", "test", "entry_point", "code_prompt"):
                    if type(verifier_metadata.get(key)) is not str:
                        raise RuntimeError(f"BigCodeBench verifier_metadata is missing {key}")
                question = row.get("question")
                if type(question) is not str:
                    raise RuntimeError("BigCodeBench question must be a string")
                tasks.append(
                    BenchmarkTask(
                        execution=TaskSpec(
                            task_id=verifier_metadata["task_id"],
                            prompt=_PROMPT.format(question=question),
                        ),
                        evaluation=dict(verifier_metadata),
                    )
                )
                if len(tasks) >= limit:
                    break
        if not tasks:
            raise RuntimeError(f"no BigCodeBench tasks found in {self.dataset_path}")
        return tasks

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        del task
        workspace.mkdir(parents=True, exist_ok=True)
        return []

    def snapshot_task(self, task: BenchmarkTask, workspace: Path) -> SnapshotTaskContent:
        del workspace
        return SnapshotTaskContent(evaluation_data=dict(task.evaluation))
