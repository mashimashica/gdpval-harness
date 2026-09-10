# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_SAFE_TASK_ID = re.compile(r"[^A-Za-z0-9._-]+")


def safe_task_id(task_id: str) -> str:
    value = _SAFE_TASK_ID.sub("_", task_id).strip("._")
    if not value:
        raise ValueError("task_id must contain at least one safe character")
    return value


@dataclass(frozen=True)
class TaskLayout:
    task_id: str
    workspace: Path
    executor_dir: Path
    workspace_deliverables: Path
    judge_deliverables: Path


def task_layout(run_root: Path, task_id: str, repeat: int = 0) -> TaskLayout:
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    safe_id = safe_task_id(task_id)
    task_root = run_root / "tasks" / safe_id
    # Preserve the established repeat_0 path while isolating any additional
    # repeats so their workspace and executor logs cannot overwrite each other.
    if repeat:
        task_root = task_root / f"repeat_{repeat}"
    workspace = task_root / "workspace"
    return TaskLayout(
        task_id=task_id,
        workspace=workspace,
        executor_dir=task_root / "executor",
        workspace_deliverables=workspace / "deliverables",
        judge_deliverables=run_root / "deliverables" / f"task_{safe_id}" / f"repeat_{repeat}",
    )
