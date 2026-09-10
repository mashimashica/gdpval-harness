# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gdpval_harness.judges.base import Verdict


_IGNORED_SUBMISSION_NAMES = {
    "finish_params.json",
    "history.json",
    "history.pkl",
    "metadata.json",
    "inprogress_history.json",
    "log.txt",
    "reference_files",
}
_VERDICT_LINE_RE = re.compile(r"BOXED\[(A|B|TIE)\]", re.IGNORECASE)


@dataclass(frozen=True)
class PreparedTrial:
    task_key: str
    trial_index: int
    swapped: bool
    workspace: Path
    reference_dir: Path
    submission_a_dir: Path
    submission_b_dir: Path
    executor_dir: Path


def _assert_no_symlink_components(path: Path) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink not allowed in judged artifact path: {current}")
    return absolute


def _assert_directory(path: Path, *, within: Path | None = None) -> Path:
    absolute = _assert_no_symlink_components(path)
    if not absolute.is_dir():
        raise ValueError(f"judged artifact directory not found: {path}")
    if within is not None:
        within_absolute = _assert_no_symlink_components(within)
        try:
            absolute.resolve().relative_to(within_absolute.resolve())
        except ValueError as exc:
            raise ValueError(f"judged artifact path escapes candidate root: {path}") from exc
    return absolute


def _iter_files(root: Path) -> Iterable[Path]:
    _assert_directory(root)
    return (path for path in sorted(root.rglob("*")) if path.is_file() or path.is_symlink())


def tree_hash(root: Path) -> str:
    _assert_directory(root)
    digest = hashlib.sha256()
    for path in _iter_files(root):
        if path.is_symlink():
            raise ValueError(f"symlink not allowed in judged artifacts: {path}")
        relative = path.relative_to(root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def discover_tasks(deliverables_root: Path) -> dict[str, Path]:
    root = _assert_directory(deliverables_root)
    tasks: dict[str, Path] = {}
    for task_dir in sorted(root.iterdir()):
        if task_dir.is_symlink() or not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            if task_dir.is_symlink() and task_dir.name.startswith("task_"):
                raise ValueError(f"symlinked task directory is not allowed: {task_dir}")
            continue
        _assert_directory(task_dir, within=root)
        repeat = task_dir / "repeat_0"
        if repeat.is_symlink():
            raise ValueError(f"symlinked repeat directory is not allowed: {repeat}")
        if repeat.is_dir():
            _assert_directory(repeat, within=root)
            tasks[task_dir.name] = repeat
    if not tasks:
        raise ValueError(f"no task_<id>/repeat_0 directories found in {deliverables_root}")
    return tasks


def matched_tasks(candidate_a: Path, candidate_b: Path) -> list[tuple[str, Path, Path]]:
    tasks_a = discover_tasks(candidate_a)
    tasks_b = discover_tasks(candidate_b)
    if set(tasks_a) != set(tasks_b):
        missing_a = sorted(set(tasks_b) - set(tasks_a))
        missing_b = sorted(set(tasks_a) - set(tasks_b))
        raise ValueError(f"candidate task sets differ; missing from A={missing_a}, missing from B={missing_b}")
    return [(name, tasks_a[name], tasks_b[name]) for name in sorted(tasks_a)]


def _copy_tree(source: Path, target: Path, *, submission: bool = False) -> None:
    source = _assert_directory(source)
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if relative.parts and submission and relative.parts[0] in _IGNORED_SUBMISSION_NAMES:
            continue
        if path.is_symlink():
            raise ValueError(f"symlink not allowed in judged artifacts: {relative}")
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _reference_dir(candidate: Path) -> Path:
    candidate = _assert_directory(candidate)
    reference = candidate / "reference_files"
    if reference.is_symlink():
        raise ValueError(f"symlinked reference directory is not allowed: {reference}")
    return reference


def validate_reference_equivalence(candidate_a: Path, candidate_b: Path) -> None:
    ref_a = _reference_dir(candidate_a)
    ref_b = _reference_dir(candidate_b)
    if ref_a.is_dir() != ref_b.is_dir():
        raise ValueError("candidate reference-file presence differs")
    if ref_a.is_dir():
        _assert_directory(ref_a, within=candidate_a)
        _assert_directory(ref_b, within=candidate_b)
        if tree_hash(ref_a) != tree_hash(ref_b):
            raise ValueError("candidate reference files differ")


def initial_swap(task_key: str, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{task_key}".encode()).digest()
    return bool(digest[0] & 1)


def prepare_trial(
    out_dir: Path,
    task_key: str,
    candidate_a: Path,
    candidate_b: Path,
    *,
    trial_index: int,
    seed: int,
) -> PreparedTrial:
    _assert_directory(candidate_a)
    _assert_directory(candidate_b)
    validate_reference_equivalence(candidate_a, candidate_b)
    swapped = initial_swap(task_key, seed) ^ bool(trial_index % 2)
    trial_root = out_dir / "judge" / "tasks" / task_key / f"trial_{trial_index}"
    shutil.rmtree(trial_root, ignore_errors=True)
    workspace = trial_root / "workspace"
    executor_dir = trial_root / "executor"
    reference_dir = workspace / "reference_files"
    submission_a_dir = workspace / "submission_a"
    submission_b_dir = workspace / "submission_b"
    executor_dir.mkdir(parents=True, exist_ok=True)

    if _reference_dir(candidate_a).is_dir():
        _copy_tree(_reference_dir(candidate_a), reference_dir)
    else:
        reference_dir.mkdir(parents=True, exist_ok=True)

    source_a, source_b = (candidate_b, candidate_a) if swapped else (candidate_a, candidate_b)
    _copy_tree(source_a, submission_a_dir, submission=True)
    _copy_tree(source_b, submission_b_dir, submission=True)
    return PreparedTrial(
        task_key=task_key,
        trial_index=trial_index,
        swapped=swapped,
        workspace=workspace,
        reference_dir=reference_dir,
        submission_a_dir=submission_a_dir,
        submission_b_dir=submission_b_dir,
        executor_dir=executor_dir,
    )


def build_judge_prompt(task_prompt: str) -> str:
    return f"""Blindly compare two submissions for the same professional task.

The current workspace contains:
- reference_files/: source material supplied with the task, if any
- submission_a/: anonymous submission A
- submission_b/: anonymous submission B

Inspect all relevant files. Judge which submission better completes the task overall, considering correctness, completeness, faithfulness to the requested work, and professional usability. Do not infer or discuss who produced either submission. Do not modify the submissions. Do not use cloud/background agents.

Task:
{task_prompt}

After your reasoning, end your final response with exactly one of:
BOXED[A]
BOXED[B]
BOXED[TIE]
"""


def parse_verdict(text: str) -> Verdict:
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        raise ValueError("judge output is empty")
    matches = []
    for line in nonempty_lines:
        match = _VERDICT_LINE_RE.fullmatch(line)
        if match:
            matches.append(match.group(1).upper())
    final_match = _VERDICT_LINE_RE.fullmatch(nonempty_lines[-1])
    if final_match is None or len(matches) != 1:
        raise ValueError(
            "judge output must end with exactly one standalone BOXED[A], BOXED[B], or BOXED[TIE] verdict"
        )
    return Verdict(final_match.group(1).upper())


def normalize_verdict(verdict: Verdict, swapped: bool) -> Verdict:
    if not swapped or verdict is Verdict.TIE:
        return verdict
    return Verdict.B if verdict is Verdict.A else Verdict.A


def aggregate(verdicts: Iterable[Verdict]) -> dict[str, float | int]:
    values = list(verdicts)
    wins_a = sum(value is Verdict.A for value in values)
    wins_b = sum(value is Verdict.B for value in values)
    ties = sum(value is Verdict.TIE for value in values)
    total = len(values)
    score_a = (wins_a + 0.5 * ties) / total if total else 0.0
    return {"trials": total, "wins_a": wins_a, "wins_b": wins_b, "ties": ties, "score_a": score_a}


def write_trial_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
