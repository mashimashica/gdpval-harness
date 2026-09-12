# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, secret-safe reproducibility records and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from eval_harness.executors.base import ExecutionResult, TaskSpec


_GIT_TIMEOUT_SECONDS = 10.0
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PORCELAIN_STATUS_CHARS = frozenset(" MTA DRCU?!")


@dataclass(frozen=True)
class RepositoryProvenance:
    """Observed Git revision and worktree state without source identity details."""

    commit: str | None
    revision_status: str
    worktree_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.revision_status, str) or self.revision_status not in {"available", "unavailable"}:
            raise ValueError("revision_status must be available or unavailable")
        if not isinstance(self.worktree_status, str) or self.worktree_status not in {
            "clean",
            "dirty",
            "unavailable",
        }:
            raise ValueError("worktree_status must be clean, dirty, or unavailable")

        if self.revision_status == "available":
            if not isinstance(self.commit, str) or _GIT_OBJECT_RE.fullmatch(self.commit) is None:
                raise ValueError("available revision_status requires a lowercase Git object ID")
        else:
            if self.commit is not None:
                raise ValueError("unavailable revision_status requires commit=None")
            if self.worktree_status != "unavailable":
                raise ValueError("unavailable revision_status requires worktree_status=unavailable")


def canonical_json_sha256(value: object) -> str:
    """Hash a value's stable, compact JSON representation."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except TypeError as exc:
        raise TypeError("value must be JSON serializable") from exc
    except ValueError as exc:
        raise ValueError("value must contain finite JSON values and valid UTF-8 text") from exc
    return hashlib.sha256(encoded).hexdigest()


def task_sha256(task: TaskSpec) -> str:
    """Hash exactly the UTF-8 bytes persisted in the canonical task prompt file."""

    if type(task) is not TaskSpec:
        raise TypeError("task must be an exact TaskSpec")
    if not isinstance(task.prompt, str):
        raise TypeError("task prompt must be a string")
    return hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()


def _unavailable_provenance() -> RepositoryProvenance:
    return RepositoryProvenance(commit=None, revision_status="unavailable", worktree_status="unavailable")


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _git_command(start_dir: str, *arguments: str) -> list[str]:
    return [
        "git",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        start_dir,
        *arguments,
    ]


def _run_git_text(start_dir: str, arguments: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            _git_command(start_dir, *arguments),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_environment(),
            check=False,
        )
    except Exception:
        return None

    try:
        returncode = result.returncode
        output = result.stdout
    except Exception:
        return None
    if returncode != 0:
        return None
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return output if isinstance(output, str) else None


def _parse_commit(output: str) -> str | None:
    lines = output.splitlines()
    if len(lines) != 1 or _GIT_OBJECT_RE.fullmatch(lines[0]) is None:
        return None
    return lines[0]


def _parse_worktree_status(output: str) -> str | None:
    if output == "":
        return "clean"

    lines = output.splitlines()
    if not lines:
        return None
    for line in lines:
        if len(line) < 4 or line[2] != " " or line[0] not in _PORCELAIN_STATUS_CHARS:
            return None
        if line[1] not in _PORCELAIN_STATUS_CHARS or not line[3:]:
            return None
    return "dirty"


def repository_provenance(start: Path | str) -> RepositoryProvenance:
    """Observe a local repository without changing its files or index."""

    try:
        start_dir = str(Path(start))
    except (TypeError, ValueError):
        return _unavailable_provenance()

    head_output = _run_git_text(start_dir, ("rev-parse", "--verify", "HEAD"))
    if head_output is None:
        return _unavailable_provenance()
    commit = _parse_commit(head_output)
    if commit is None:
        return _unavailable_provenance()

    status_output = _run_git_text(
        start_dir,
        ("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=all"),
    )
    if status_output is None:
        return RepositoryProvenance(commit=commit, revision_status="available", worktree_status="unavailable")
    worktree_status = _parse_worktree_status(status_output)
    if worktree_status is None:
        worktree_status = "unavailable"
    return RepositoryProvenance(commit=commit, revision_status="available", worktree_status=worktree_status)


def execution_record(result: ExecutionResult) -> dict[str, object]:
    """Return typed execution evidence without carrying arbitrary metadata."""

    if not isinstance(result, ExecutionResult):
        raise TypeError("result must be an ExecutionResult")
    status = result.status.value if hasattr(result.status, "value") else result.status
    record: dict[str, object] = {
        "status": str(status),
        "executor": result.executor,
        "executor_version": result.executor_version,
        "invocation_mode": result.invocation_mode,
        "auth_mode": result.auth_mode,
        "workspace": str(result.workspace),
        "deliverables_dir": str(result.deliverables_dir),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "exit_code": result.exit_code,
        "output_text_present": result.output_text is not None,
        "available_outputs": sorted(output.value for output in result.available_outputs),
        "failure": (
            None
            if result.failure is None
            else {
                "kind": result.failure.kind.value,
                "code": result.failure.code,
                "impact": result.failure.impact.value,
            }
        ),
        "metadata": {},
    }
    if result.reasoning_effort_requested is not None:
        record["reasoning_effort_requested"] = result.reasoning_effort_requested
    return record


__all__ = (
    "RepositoryProvenance",
    "canonical_json_sha256",
    "task_sha256",
    "repository_provenance",
    "execution_record",
)
