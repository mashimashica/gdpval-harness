# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkTask
from gdpval_harness.executors.base import TaskSpec


def _parse_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def _is_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reference_listing(workspace: Path) -> str:
    ref_root = workspace / "reference_files"
    if not ref_root.is_dir():
        return "None"
    files = [str(path.relative_to(workspace)) for path in sorted(ref_root.rglob("*")) if path.is_file()]
    return "\n".join(f"- {item}" for item in files) if files else "None"


class GDPvalBenchmark(Benchmark):
    name = "gdpval"
    # The current upstream prepare script loads openai/gdpval without a pinned
    # dataset revision. Preserve that fact rather than inventing a revision.
    revision = None

    def __init__(self, *, root: Path, dataset_path: Path, prepare_script: Path) -> None:
        self.root = root
        self.dataset_path = dataset_path
        self.prepare_script = prepare_script

    def is_prepared(self) -> bool:
        return self.dataset_path.is_file()

    def prepare(self) -> None:
        if self.is_prepared():
            return
        if not self.prepare_script.is_file():
            raise RuntimeError(f"GDPval prepare script not found: {self.prepare_script}")
        result = subprocess.run([sys.executable, str(self.prepare_script)], cwd=self.root, check=False)
        if result.returncode != 0 or not self.is_prepared():
            raise RuntimeError("failed to prepare GDPval benchmark data")

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        if limit <= 0:
            raise ValueError("benchmark task limit must be positive")
        tasks: list[BenchmarkTask] = []
        with self.dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                tasks.append(
                    BenchmarkTask(
                        execution=TaskSpec(task_id=str(row["task_id"]), prompt=str(row["prompt"])),
                        materialization={
                            "reference_files": _parse_sequence(row.get("reference_files")),
                            "reference_file_urls": _parse_sequence(row.get("reference_file_urls")),
                        },
                        evaluation={
                            "sector": str(row.get("sector") or ""),
                            "occupation": str(row.get("occupation") or ""),
                        },
                    )
                )
                if len(tasks) >= limit:
                    break
        if not tasks:
            raise RuntimeError(f"no GDPval tasks found in {self.dataset_path}")
        return tasks

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        reference_files = tuple(str(item) for item in task.materialization.get("reference_files", ()))
        reference_urls = tuple(str(item) for item in task.materialization.get("reference_file_urls", ()))
        if not reference_files and not reference_urls:
            return []
        if len(reference_files) != len(reference_urls):
            raise RuntimeError(f"task {task.execution.task_id}: reference file/url count mismatch")

        from responses_api_agents.stirrup_agent.tasks.gdpval import _download_reference_files

        downloaded = _download_reference_files(list(reference_files), list(reference_urls), workspace)
        if len(downloaded) != len(reference_files):
            raise RuntimeError(
                f"task {task.execution.task_id}: materialized {len(downloaded)}/{len(reference_files)} reference files"
            )
        for relative in downloaded:
            target = workspace / relative
            if not _is_inside(workspace, target) or not target.is_file():
                raise RuntimeError(
                    f"task {task.execution.task_id}: unsafe or missing materialized reference path: {relative}"
                )
        return downloaded

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        prompt = f"""You are completing a GDPval professional-work task in an isolated local workspace.

Work only on this task. Do not create, hand off, or continue the task in any cloud/background agent.
Use only tools actually available in this local runtime; do not assume packages or system tools are installed.

Reference files, when provided, are under the current workspace:
{_reference_listing(workspace)}

Final deliverables contract:
- Put every file that should be submitted for evaluation under ./deliverables/.
- Create ./deliverables/ if needed.
- Nested files and directories under ./deliverables/ are allowed.
- Keep scratch files, logs, caches, helper scripts, and executor metadata out of ./deliverables/.
- Do not modify the reference_files directory.
- Network policy for model-generated tools: {network_policy}.

Task:
{task.execution.prompt}
"""
        return TaskSpec(task_id=task.execution.task_id, prompt=prompt)
