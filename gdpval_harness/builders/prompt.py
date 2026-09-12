# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt construction for the reusable Agent Skill builder.

This module deliberately depends only on the executor-facing :class:`TaskSpec`.  Builder
execution and artifact contracts are separate concerns and are not needed to construct the
model prompt.
"""

from __future__ import annotations

import re
from typing import Sequence

from gdpval_harness.executors.base import TaskSpec


_INPUT_TARGET_PATTERN = re.compile(r"\Areference_files/builder-inputs/input-(?!000\Z)[0-9]{3}\Z")


def _normalize_input_targets(input_targets: Sequence[str]) -> tuple[str, ...]:
    """Validate and tuple-ize the neutral input directories used by a build prompt."""

    if isinstance(input_targets, (str, bytes)):
        raise TypeError("input_targets must be a sequence of target strings, not a string")

    try:
        targets = tuple(input_targets)
    except TypeError as exc:
        raise TypeError("input_targets must be an iterable of target strings") from exc

    if any(not isinstance(target, str) for target in targets):
        raise TypeError("every input target must be a string")
    if any(_INPUT_TARGET_PATTERN.fullmatch(target) is None for target in targets):
        raise ValueError(
            "each input target must be a normalized relative POSIX directory matching "
            "reference_files/builder-inputs/input-<3 digits>"
        )
    if len(set(targets)) != len(targets):
        raise ValueError("input targets must be unique")
    if targets != tuple(sorted(targets)):
        raise ValueError("input targets must be supplied in sorted order")
    return targets


def _validate_task(task: TaskSpec) -> None:
    """Validate the small executor-facing task contract before prompt construction."""

    if not isinstance(task, TaskSpec):
        raise TypeError("task must be a TaskSpec")
    if not isinstance(task.task_id, str):
        raise TypeError("TaskSpec.task_id must be a string")
    if not isinstance(task.prompt, str):
        raise TypeError("TaskSpec.prompt must be a string")


def build_skill_task(task: TaskSpec, input_targets: Sequence[str]) -> TaskSpec:
    """Build the model task that asks for one reusable Agent Skill.

    Only the target task prompt and validated neutral input directory names are copied into the
    resulting prompt.  In particular, the task identifier is preserved on the returned
    :class:`TaskSpec` but is never interpolated into prompt text.
    """

    _validate_task(task)
    targets = _normalize_input_targets(input_targets)

    if targets:
        target_lines = "\n".join(f"- `{target}`" for target in targets)
    else:
        target_lines = "- (no creation input directories were supplied)"

    prompt = f"""Create one reusable Agent Skill for a future agent.

Do not solve or submit the target task, and do not create the target task's deliverable now. Your
job is to write the reusable instructions and resources that would help a future agent handle the
target task.

The output layout is strict: create exactly one immediate directory at
`./deliverables/<skill-name>/`, with its required file at
`./deliverables/<skill-name>/SKILL.md`. The YAML `name` in `SKILL.md` must exactly equal the
`<skill-name>` parent directory. Keep all normal Agent Skill resources inside that same directory.

The only creation inputs are these neutral relative directories:
{target_lines}

Use those directories only as creation inputs. Do not put known condition labels or arm labels,
source paths or source revisions, input IDs, a builder transcript, generation provenance, scores,
rubrics, or reference answers into the Skill. Do not add any other task, run, benchmark, or
evaluation metadata.

The target content to use is only the following task prompt:
--- BEGIN TARGET TASK PROMPT ---
{task.prompt}
--- END TARGET TASK PROMPT ---

Derive reusable Agent Skill guidance from that prompt and the creation inputs while preserving the
strict output layout and the provenance exclusions above."""
    return TaskSpec(task_id=task.task_id, prompt=prompt)


__all__ = ["build_skill_task"]
