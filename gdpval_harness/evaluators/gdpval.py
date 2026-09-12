# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDPval's generic-runner evaluator.

The generic local runner deliberately does not implement the GDPval rubric or
call a judge.  It publishes the policy's submitted files and task references
in the layout consumed by the existing GDPval judge path and reports an
external handoff.  The implementation is intentionally conservative around
filesystem boundaries because this directory is later read by a separate
judge process.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from gdpval_harness.evaluators.base import (
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
    EvaluatorType,
    require_one_candidate,
)
from gdpval_harness.executors.base import ExecutionStatus


_TERMINAL_SUCCESS = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
_FINISH_NAME = "finish_params.json"


def _absolute(path: Path) -> Path:
    try:
        return path.absolute()
    except OSError:
        return Path(os.path.abspath(path))


def _assert_no_symlink_components(path: Path, *, allow_missing_final: bool = True) -> Path:
    """Return an absolute path after rejecting symlink path components.

    ``Path.resolve`` alone is not sufficient here: it follows a symlink and
    would make a destination escape look like an ordinary in-tree path.  Walk
    the lexical path instead, checking every component that exists.  Missing
    components are safe to create later and are checked again after creation.
    """

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            is_symlink = current.is_symlink()
            exists = current.exists() or current.is_symlink()
        except OSError as exc:
            raise ValueError(f"could not inspect GDPval artifact path: {current}: {exc}") from exc
        if is_symlink:
            raise ValueError(f"symlink not allowed in GDPval artifact path: {current}")
        if not exists and not allow_missing_final:
            raise ValueError(f"GDPval artifact path does not exist: {path}")
        # Once a component is missing, later components are necessarily
        # missing too; there cannot be a symlink below it yet.
        if not exists:
            break
    return absolute


def _assert_directory(path: Path, *, label: str) -> Path:
    absolute = _assert_no_symlink_components(path, allow_missing_final=False)
    if not absolute.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return absolute


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return _absolute(path)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = _resolved(left)
    right_resolved = _resolved(right)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _iter_source_entries(root: Path) -> Iterable[tuple[Path, Path]]:
    """Yield regular source entries, rejecting symlinks and escapes."""

    root = _assert_directory(root, label="GDPval source directory")
    root_resolved = _resolved(root)
    for source in sorted(root.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"symlink not allowed in GDPval submitted artifacts: {source}")
        try:
            source_resolved = source.resolve(strict=True)
            source_resolved.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise ValueError(f"GDPval source path escapes its root: {source}") from exc
        if source.is_dir():
            yield source, source.relative_to(root)
        elif source.is_file():
            yield source, source.relative_to(root)
        else:
            raise ValueError(f"unsupported GDPval submitted artifact type: {source}")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    # Directory fsync is available on POSIX.  Windows does not support opening
    # a directory this way; byte durability and atomicity remain best effort on
    # that platform rather than masking a successful handoff.
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_tree_bytes(
    source: Path,
    destination: Path,
    *,
    allow_existing_destination: bool = False,
) -> list[str]:
    """Copy a submitted tree without source metadata and return file names."""

    copied: list[str] = []
    destination.mkdir(parents=True, exist_ok=allow_existing_destination)
    if allow_existing_destination and any(destination.iterdir()):
        raise FileExistsError(f"refusing to use non-empty GDPval staging directory: {destination}")
    for source_entry, relative in _iter_source_entries(source):
        target = destination / relative
        if target.is_symlink():
            raise ValueError(f"symlink not allowed in GDPval destination path: {target}")
        if source_entry.is_dir():
            target.mkdir(parents=True, exist_ok=False)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # copyfile transfers bytes only; in particular, it does not carry
        # evaluator metadata, source mtimes, ownership, or xattrs.
        shutil.copyfile(source_entry, target)
        _fsync_file(target)
        copied.append(str(relative))
    _fsync_directory(destination)
    return copied


def _copy_reference_tree(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"symlinked GDPval reference directory is not allowed: {source}")
    if not source.exists():
        return
    if not source.is_dir():
        raise ValueError(f"GDPval reference_files is not a directory: {source}")
    _copy_tree_bytes(source, destination)


def _write_durable_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _publish(
    *,
    source_deliverables: Path,
    source_workspace: Path,
    destination: Path,
    result_executor: str,
    result_status: str,
) -> tuple[str, ...]:
    """Stage and atomically publish a GDPval judge-facing artifact tree."""

    source_deliverables = _assert_directory(source_deliverables, label="GDPval deliverables")
    source_workspace = _assert_directory(source_workspace, label="GDPval workspace")
    destination = _assert_no_symlink_components(destination)

    for reserved in (Path(_FINISH_NAME), Path("reference_files")):
        source_reserved = source_deliverables / reserved
        if source_reserved.exists() or source_reserved.is_symlink():
            raise ValueError(f"submitted GDPval deliverables contain reserved top-level path: {reserved}")

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing GDPval handoff destination: {destination}")
    source_reference = source_workspace / "reference_files"
    if source_reference.is_symlink():
        raise ValueError(f"symlinked GDPval reference directory is not allowed: {source_reference}")

    # A destination nested inside a source would let a recursive copy modify
    # its own input.  Treat that as an unsafe path instead of guessing intent.
    if _paths_overlap(destination, source_deliverables) or _paths_overlap(destination, source_workspace):
        raise ValueError("GDPval handoff destination must be separate from executor source paths")

    parent = destination.parent
    _assert_no_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(parent, allow_missing_final=False)

    # mkdtemp creates a unique directory atomically and never reuses an
    # existing staging path.  The random suffix is useful in tests and in
    # operator logs without carrying candidate identity.
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-{secrets.token_hex(6)}-",
                dir=parent,
            )
        )
        _assert_no_symlink_components(staging, allow_missing_final=False)
        copied = _copy_tree_bytes(source_deliverables, staging, allow_existing_destination=True)

        if source_reference.exists():
            _copy_reference_tree(source_reference, staging / "reference_files")

        # The existing GDPval deliverable reader uses this marker to identify a
        # completed task.  Keep it limited to executor state and submitted file
        # names; benchmark rubric/evaluation metadata never enters the handoff.
        finish: dict[str, object] = {
            "executor": result_executor,
            "status": result_status,
            "files": copied,
            "summary": (
                "generic runner exported judge-compatible GDPval artifacts; "
                "rubric/pairwise evaluation remains external"
            ),
        }
        _write_durable_json(staging / _FINISH_NAME, finish)
        _fsync_directory(staging)
        _fsync_directory(parent)

        # Reserve the final path with an exclusive empty directory immediately
        # before the rename.  A concurrent creator therefore fails at mkdir;
        # os.replace only replaces the empty reservation made by this call and
        # never an unknown prior handoff containing a marker or deliverables.
        destination.mkdir(exist_ok=False)
        os.replace(staging, destination)
        staging = None
        _fsync_directory(parent)
        return tuple(copied)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if destination.is_dir() and not destination.is_symlink():
            try:
                if not any(destination.iterdir()):
                    destination.rmdir()
            except OSError:
                pass
        raise


class GDPvalExternalEvaluator(Evaluator):
    """Export GDPval artifacts for the existing external rubric/pairwise path."""

    name = "gdpval-external"
    evaluator_type = EvaluatorType.LLM_RUBRIC
    version = "1"
    revision = "external-handoff-v1"

    def validate_plan(self, plan: EvaluationPlan) -> None:
        """Validate the one-candidate external handoff before execution."""

        if plan.candidate_count != 1:
            raise ValueError("GDPval external evaluator requires an evaluation plan with exactly one candidate")
        if plan.artifact_dir is None:
            raise ValueError("GDPval external evaluator requires an artifact destination in the evaluation plan")

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        # Deliberately do not inspect or instantiate a judge/model client.
        details: tuple[str, ...] = (
            "GDPval evaluator exports judge-compatible artifacts only",
            "rubric/pairwise scoring remains external to the generic runner",
        )
        if run_dir is not None:
            details += (f"handoff run root: {run_dir}",)
        return EvaluatorPreflightResult(
            name=self.name,
            evaluator_type=self.evaluator_type,
            ok=True,
            version=self.version,
            revision=self.revision,
            details=details,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        candidate = require_one_candidate(request)
        result = candidate.execution
        if result.status not in _TERMINAL_SUCCESS:
            return EvaluationResult(
                task_id=request.task_id,
                status=EvaluationStatus.SKIPPED,
                metrics={},
                outcomes={},
                details={
                    "reason": "executor did not reach a terminal success state",
                    "execution_status": result.status.value,
                    "handoff": "not published",
                },
            )

        destination = request.artifact_dir
        if destination is None:
            raise ValueError("GDPval external evaluator requires request.artifact_dir")
        source_deliverables = candidate.artifacts_dir or result.deliverables_dir
        copied = _publish(
            source_deliverables=source_deliverables,
            source_workspace=result.workspace,
            destination=destination,
            result_executor=result.executor,
            result_status=result.status.value,
        )
        return EvaluationResult(
            task_id=request.task_id,
            status=EvaluationStatus.EXTERNAL,
            metrics={},
            outcomes={},
            details={
                "handoff_path": str(destination),
                "handoff_method": "atomic staged publish",
                "submitted_files": list(copied),
                "reference_files": bool((result.workspace / "reference_files").is_dir()),
                "evaluation": "rubric/pairwise remains external; no judge/model/API was invoked",
            },
        )


# Keep both spellings available to callers while the generic CLI describes the
# adapter as the external GDPval evaluator.
GDPvalEvaluator = GDPvalExternalEvaluator
ExternalGDPvalEvaluator = GDPvalExternalEvaluator


__all__ = [
    "GDPvalEvaluator",
    "GDPvalExternalEvaluator",
    "ExternalGDPvalEvaluator",
]
