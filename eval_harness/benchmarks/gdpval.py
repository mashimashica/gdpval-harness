# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import unquote, urlparse

from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.benchmarks.snapshot import (
    Availability,
    SnapshotError,
    SnapshotTaskContent,
    _decode_json_object,
    _source_stat_fingerprint,
    _validate_logical_path,
)
from eval_harness.executors.base import TaskSpec


def _parse_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_value = value
        try:
            parsed: object = json.loads(raw_value)
        except json.JSONDecodeError:
            return (raw_value,)
        value = parsed
    if isinstance(value, (list, tuple)):
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
    source = "huggingface:openai/gdpval/train"
    source_availability = Availability.AVAILABLE
    # The current upstream prepare script loads openai/gdpval without a pinned
    # dataset revision. Preserve that fact rather than inventing a revision.
    revision = None
    revision_availability = Availability.UNAVAILABLE

    def __init__(
        self,
        *,
        root: Path,
        dataset_path: Path,
        prepare_script: Path,
        reference_downloader: Callable[[list[str], list[str], Path], Sequence[str]] | None = None,
    ) -> None:
        self.root = root
        self.dataset_path = dataset_path
        self.prepare_script = prepare_script
        self.reference_downloader = reference_downloader

    def is_prepared(self) -> bool:
        return self.dataset_path.is_file()

    def snapshot_source_paths(self) -> tuple[Path, ...]:
        return (self.dataset_path,)

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
                row = _decode_json_object(line, label="GDPval dataset row")
                task_id = row.get("task_id")
                prompt = row.get("prompt")
                if type(task_id) is not str or not task_id or type(prompt) is not str:
                    raise RuntimeError("GDPval task_id and prompt must be strings")
                sector_value = row.get("sector")
                occupation_value = row.get("occupation")
                sector = "" if sector_value is None else sector_value
                occupation = "" if occupation_value is None else occupation_value
                if type(sector) is not str or type(occupation) is not str:
                    raise RuntimeError("GDPval sector and occupation must be strings")
                rubric_json = _parse_rubric_json(row.get("rubric_json"))
                rubric_pretty = row.get("rubric_pretty")
                if rubric_pretty is None:
                    rubric_pretty = json.dumps(rubric_json, ensure_ascii=False, sort_keys=True, indent=2)
                if not isinstance(rubric_pretty, str):
                    raise RuntimeError("GDPval rubric_pretty must be a string")
                tasks.append(
                    BenchmarkTask(
                        execution=TaskSpec(task_id=task_id, prompt=prompt),
                        materialization={
                            "reference_files": _parse_snapshot_sequence(
                                row.get("reference_files"), label="GDPval reference_files"
                            ),
                            "reference_file_urls": _parse_snapshot_sequence(
                                row.get("reference_file_urls"), label="GDPval reference_file_urls"
                            ),
                        },
                        evaluation={
                            "sector": sector,
                            "occupation": occupation,
                            "rubric_json": rubric_json,
                            "rubric_pretty": rubric_pretty,
                        },
                    )
                )
                if len(tasks) >= limit:
                    break
        if not tasks:
            raise RuntimeError(f"no GDPval tasks found in {self.dataset_path}")
        return tasks

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        reference_files = _strict_sequence(task.materialization.get("reference_files"))
        reference_urls = _strict_sequence(task.materialization.get("reference_file_urls"))
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

    def snapshot_task(self, task: BenchmarkTask, workspace: Path) -> SnapshotTaskContent:
        """Download task attachments once and expose them in both views."""

        reference_files = _strict_sequence(task.materialization.get("reference_files"))
        reference_urls = _strict_sequence(task.materialization.get("reference_file_urls"))
        if len(reference_files) != len(reference_urls):
            raise SnapshotError("GDPval reference file metadata has a count mismatch")
        if any(not url for url in reference_urls):
            raise SnapshotError("GDPval attachment URL must be a non-empty string")

        logical_names = tuple(_safe_attachment_name(item) for item in reference_files)
        local_sources: list[Path] = []
        for url in reference_urls:
            local_source = _local_url_path(url)
            if local_source is not None:
                local_sources.append(local_source)
        before = _source_stat_fingerprint(local_sources)
        downloader = self.reference_downloader or _download_snapshot_references
        downloaded_result = downloader(list(reference_files), list(reference_urls), workspace)
        if isinstance(downloaded_result, (str, bytes)):
            raise SnapshotError("GDPval attachment downloader returned an invalid result")
        downloaded = list(downloaded_result)
        if len(downloaded) != len(reference_files):
            raise SnapshotError("GDPval attachment downloader returned an unexpected count")
        after = _source_stat_fingerprint(local_sources)
        if before != after:
            raise SnapshotError("GDPval attachment changed during acquisition")

        entries: list[tuple[str, Path]] = []
        for logical_name, downloaded_path in zip(logical_names, downloaded, strict=True):
            if type(downloaded_path) is not str:
                raise SnapshotError("GDPval attachment downloader returned an invalid path")
            relative = downloaded_path
            _validate_logical_path(relative, label="GDPval downloaded path")
            source = workspace / relative
            # A downloaded path is checked by the central race-safe reader.  We
            # also ensure it is inside this private task staging directory.
            try:
                source.relative_to(workspace)
            except ValueError as exc:
                raise SnapshotError("GDPval downloader returned a path outside staging") from exc
            entries.append((f"task_inputs/{logical_name}", source))

        return SnapshotTaskContent(
            evaluation_data=dict(task.evaluation),
            files=entries,
            evaluation_files=entries,
        )


def _parse_rubric_json(value: object) -> object:
    """Parse a GDPval rubric from either an object or JSON text."""

    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("GDPval rubric_json is not valid JSON") from exc
    try:
        # A compact encode/decode gives callers a detached, canonical JSON
        # value while rejecting NaN, Infinity, bytes, and arbitrary objects.
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeError("GDPval rubric_json is not valid JSON") from exc


def _strict_sequence(value: object) -> tuple[str, ...]:
    """Read already-normalized attachment metadata without coercion."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)) or any(type(item) is not str for item in value):
        raise SnapshotError("GDPval attachment metadata must contain strings")
    return tuple(value)


def _parse_snapshot_sequence(value: object, *, label: str) -> tuple[str, ...]:
    """Decode attachment metadata without coercing non-string values."""

    if value is None:
        return ()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return (value,)
        value = decoded
    if not isinstance(value, (list, tuple)) or any(type(item) is not str for item in value):
        raise SnapshotError(f"{label} must contain strings")
    return tuple(value)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = item
    return result


def _safe_attachment_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotError("GDPval attachment name is unsafe")
    name = value[len("reference_files/") :] if value.startswith("reference_files/") else value
    _validate_logical_path(name, label="GDPval attachment name")
    if name == "task_inputs" or name.startswith("task_inputs/"):
        raise SnapshotError("GDPval attachment name is unsafe")
    return name


def _local_url_path(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        return Path(unquote(parsed.path))
    if parsed.scheme == "":
        candidate = Path(url)
        return candidate if candidate.is_absolute() else None
    return None


def _download_snapshot_references(
    reference_files: Sequence[str], reference_urls: Sequence[str], workspace: Path
) -> list[str]:
    """Call the established downloader once without importing it in fake tests."""

    from responses_api_agents.stirrup_agent.tasks.gdpval import _download_reference_files

    return [str(item) for item in _download_reference_files(list(reference_files), list(reference_urls), workspace)]
