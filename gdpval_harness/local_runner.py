# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from gdpval_harness.executors.base import ExecutionRequest, ExecutionResult, ExecutionStatus, TaskSpec
from gdpval_harness.executors.claude_code import ClaudeCodeExecutor
from gdpval_harness.executors.codex import CodexExecutor
from gdpval_harness.layout import TaskLayout, task_layout


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSONL = Path(
    os.getenv("GDPVAL_BENCHMARK_JSONL", ROOT / "benchmarks" / "gdpval" / "data" / "gdpval_benchmark.jsonl")
)
PREPARE_SCRIPT = Path(os.getenv("GDPVAL_PREPARE_SCRIPT", ROOT / "benchmarks" / "gdpval" / "prepare.py"))
_TERMINAL_SUCCESS = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}


def _truthy(name: str) -> bool:
    return os.getenv(name, "").lower() not in {"", "0", "false", "no"}


def _parse_max_turns() -> int:
    raw = os.getenv("GDPVAL_EXECUTOR_MAX_TURNS", "250")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("GDPVAL_EXECUTOR_MAX_TURNS must be an integer") from exc
    if value <= 0:
        raise ValueError("GDPVAL_EXECUTOR_MAX_TURNS must be positive")
    return value


def _executor(name: str):
    network_enabled = os.getenv("GDPVAL_EXECUTOR_NETWORK", "disabled") == "enabled"
    if name == "codex":
        return CodexExecutor(network_enabled=network_enabled)
    if name == "claude-code":
        return ClaudeCodeExecutor(network_enabled=network_enabled, max_turns=_parse_max_turns())
    raise ValueError(f"local executor {name!r} is not implemented")


def _ensure_dataset() -> None:
    if BENCHMARK_JSONL.is_file():
        return
    if not PREPARE_SCRIPT.is_file():
        raise RuntimeError(f"GDPval prepare script not found: {PREPARE_SCRIPT}")
    result = subprocess.run([sys.executable, str(PREPARE_SCRIPT)], cwd=ROOT, check=False)
    if result.returncode != 0 or not BENCHMARK_JSONL.is_file():
        raise RuntimeError("failed to prepare GDPval benchmark data")


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


def _load_tasks(limit: int) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with BENCHMARK_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                TaskSpec(
                    task_id=str(row["task_id"]),
                    prompt=str(row["prompt"]),
                    reference_files=_parse_sequence(row.get("reference_files")),
                    reference_file_urls=_parse_sequence(row.get("reference_file_urls")),
                    sector=str(row.get("sector") or ""),
                    occupation=str(row.get("occupation") or ""),
                )
            )
            if len(tasks) >= limit:
                break
    if not tasks:
        raise RuntimeError(f"no GDPval tasks found in {BENCHMARK_JSONL}")
    return tasks


def _is_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _materialize_reference_files(task: TaskSpec, workspace: Path) -> list[str]:
    if not task.reference_files and not task.reference_file_urls:
        return []
    if len(task.reference_files) != len(task.reference_file_urls):
        raise RuntimeError(f"task {task.task_id}: reference file/url count mismatch")

    from responses_api_agents.stirrup_agent.tasks.gdpval import _download_reference_files

    downloaded = _download_reference_files(
        list(task.reference_files),
        list(task.reference_file_urls),
        workspace,
    )
    if len(downloaded) != len(task.reference_files):
        raise RuntimeError(
            f"task {task.task_id}: materialized {len(downloaded)}/{len(task.reference_files)} reference files"
        )
    for relative in downloaded:
        target = workspace / relative
        if not _is_inside(workspace, target) or not target.is_file():
            raise RuntimeError(f"task {task.task_id}: unsafe or missing materialized reference path: {relative}")
    return downloaded


def _reference_listing(workspace: Path) -> str:
    ref_root = workspace / "reference_files"
    if not ref_root.is_dir():
        return "None"
    files = [str(path.relative_to(workspace)) for path in sorted(ref_root.rglob("*")) if path.is_file()]
    return "\n".join(f"- {item}" for item in files) if files else "None"


def build_task_prompt(task: TaskSpec, workspace: Path, *, network_policy: str) -> str:
    return f"""You are completing a GDPval professional-work task in an isolated local workspace.

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
{task.prompt}
"""


def _prepare_layout(out_dir: Path, task: TaskSpec) -> TaskLayout:
    layout = task_layout(out_dir, task.task_id)
    shutil.rmtree(layout.workspace, ignore_errors=True)
    layout.workspace.mkdir(parents=True, exist_ok=True)
    layout.executor_dir.mkdir(parents=True, exist_ok=True)
    layout.workspace_deliverables.mkdir(parents=True, exist_ok=True)
    return layout


def _copy_tree_files(source: Path, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if not source.is_dir():
        return copied
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise RuntimeError(f"deliverable symlink is not allowed: {relative}")
        if path.is_dir():
            (target / relative).mkdir(parents=True, exist_ok=True)
            continue
        if path.is_file():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(str(relative))
    return copied


def _copy_final_deliverables(layout: TaskLayout, result: ExecutionResult) -> list[str]:
    shutil.rmtree(layout.judge_deliverables, ignore_errors=True)
    layout.judge_deliverables.mkdir(parents=True, exist_ok=True)
    copied = _copy_tree_files(layout.workspace_deliverables, layout.judge_deliverables)

    ref_root = layout.workspace / "reference_files"
    if ref_root.is_dir():
        _copy_tree_files(ref_root, layout.judge_deliverables / "reference_files")

    if result.status in _TERMINAL_SUCCESS:
        finish = {
            "executor": result.executor,
            "status": result.status.value,
            "files": copied,
            "summary": "local executor reached a terminal state",
        }
        (layout.judge_deliverables / "finish_params.json").write_text(
            json.dumps(finish, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return copied


def _write_executor_metadata(layout: TaskLayout, result: ExecutionResult, files: Iterable[str]) -> None:
    payload = {
        "task_id": result.task_id,
        "workspace": str(result.workspace),
        "deliverables_directory": str(result.deliverables_dir),
        "execution_status": result.status.value,
        "executor": result.executor,
        "executor_version": result.executor_version,
        "invocation_mode": result.invocation_mode,
        "auth_mode": result.auth_mode,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "exit_code": result.exit_code,
        "submitted_files": list(files),
        "metadata": dict(result.metadata),
    }
    (layout.executor_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _already_completed(layout: TaskLayout) -> bool:
    metadata_path = layout.executor_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        status = json.loads(metadata_path.read_text(encoding="utf-8")).get("execution_status")
    except (OSError, json.JSONDecodeError):
        return False
    return status in {item.value for item in _TERMINAL_SUCCESS}


def _parse_limit() -> int:
    raw = os.getenv("LIMIT")
    if not raw:
        raise ValueError("subscription-backed executors require an explicit --limit")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("--limit must be an integer") from exc
    if value <= 0:
        raise ValueError("--limit must be positive")
    return value


def _parse_timeout() -> float:
    raw = os.getenv("GDPVAL_EXECUTOR_TIMEOUT")
    if not raw:
        return 12600.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("--executor-timeout must be numeric") from exc
    if value <= 0:
        raise ValueError("--executor-timeout must be positive")
    return value


def preflight(executor_name: str, out_dir: Path, *, for_run: bool) -> tuple[bool, dict[str, object]]:
    try:
        executor = _executor(executor_name)
    except ValueError as exc:
        return False, {"executor": executor_name, "ok": False, "version": None, "auth_mode": None, "details": [str(exc)]}

    result = executor.preflight()
    details = list(result.details)
    ok = result.ok

    if not BENCHMARK_JSONL.is_file() and not PREPARE_SCRIPT.is_file():
        details.append(f"GDPval benchmark data is absent and prepare script is missing: {PREPARE_SCRIPT}")
        ok = False

    parallel = os.getenv("PARALLEL", "1")
    if parallel not in {"", "1"}:
        details.append("local subscription executors currently require --parallel 1")
        ok = False

    try:
        _parse_timeout()
        if executor_name == "claude-code":
            _parse_max_turns()
    except ValueError as exc:
        details.append(str(exc))
        ok = False

    if for_run:
        try:
            _parse_limit()
        except ValueError as exc:
            details.append(str(exc))
            ok = False

    try:
        probe = out_dir / ".gdpval-write-probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        details.append(f"output directory is not writable: {exc}")
        ok = False

    details.append(
        "benchmark data present"
        if BENCHMARK_JSONL.is_file()
        else "benchmark JSONL is not prepared yet; run will invoke the existing GDPval prepare script"
    )
    return ok, {
        "executor": executor_name,
        "ok": ok,
        "version": result.version,
        "auth_mode": result.auth_mode,
        "details": details,
    }


def _write_run_metadata(out_dir: Path, preflight_payload: dict[str, object]) -> None:
    env = os.environ.copy()
    env["OUT"] = str(out_dir)
    env["PERSIST_DELIVERABLES_DIR"] = str(out_dir / "deliverables")
    env["GDPVAL_EXECUTOR_VERSION"] = str(preflight_payload.get("version") or "")
    env["GDPVAL_EXECUTOR_AUTH_MODE"] = str(preflight_payload.get("auth_mode") or "")
    executor = _executor(str(preflight_payload["executor"]))
    env["GDPVAL_EXECUTOR_INVOCATION_MODE"] = executor.invocation_mode
    env["GDPVAL_EXECUTOR_WORKSPACE_ISOLATION"] = "per-task-directory"
    env["GDPVAL_EXECUTOR_NETWORK"] = os.getenv("GDPVAL_EXECUTOR_NETWORK", "disabled")
    env["GDPVAL_EXECUTOR_TOOL_PERMISSION_MODE"] = getattr(executor, "tool_permission_mode", "")
    env["GDPVAL_EXECUTOR_MAX_TURNS"] = os.getenv("GDPVAL_EXECUTOR_MAX_TURNS", "")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gdpval_run_metadata.py")], cwd=ROOT, env=env, check=True)


def run() -> int:
    executor_name = os.getenv("GDPVAL_EXECUTOR", "codex")
    out_dir = Path(os.getenv("OUT", "./results/gdpval")).resolve()
    ok, preflight_payload = preflight(executor_name, out_dir, for_run=True)
    for detail in preflight_payload["details"]:
        print(f"gdpval[{executor_name}]: {detail}", file=sys.stderr)
    if not ok:
        print(f"gdpval[{executor_name}]: preflight failed", file=sys.stderr)
        return 2

    if os.getenv("GDPVAL_WRITE_METADATA", "1") != "0":
        _write_run_metadata(out_dir, preflight_payload)

    _ensure_dataset()
    limit = _parse_limit()
    timeout = _parse_timeout()
    resume = _truthy("RESUME")
    executor = _executor(executor_name)
    network_policy = os.getenv("GDPVAL_EXECUTOR_NETWORK", "disabled")

    failures = 0
    for task in _load_tasks(limit):
        layout = task_layout(out_dir, task.task_id)
        if resume and _already_completed(layout):
            print(f"gdpval[{executor_name}]: skip terminal task {task.task_id}", file=sys.stderr)
            continue

        layout = _prepare_layout(out_dir, task)
        try:
            _materialize_reference_files(task, layout.workspace)
            prompt_task = TaskSpec(
                task_id=task.task_id,
                prompt=build_task_prompt(task, layout.workspace, network_policy=network_policy),
                reference_files=task.reference_files,
                reference_file_urls=task.reference_file_urls,
                sector=task.sector,
                occupation=task.occupation,
            )
            request = ExecutionRequest(
                task=prompt_task,
                workspace=layout.workspace,
                deliverables_dir=layout.workspace_deliverables,
                executor_dir=layout.executor_dir,
                model=os.getenv("GDPVAL_MODEL"),
                timeout_seconds=timeout,
                environment=os.environ.copy(),
            )
            result = executor.execute(request)
            copied = _copy_final_deliverables(layout, result)
        except KeyboardInterrupt:
            interrupted = {
                "task_id": task.task_id,
                "execution_status": ExecutionStatus.INTERRUPTED.value,
                "executor": executor_name,
            }
            (layout.executor_dir / "metadata.json").write_text(
                json.dumps(interrupted, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return 130
        except Exception as exc:
            failures += 1
            error = {
                "task_id": task.task_id,
                "execution_status": ExecutionStatus.FAILED.value,
                "executor": executor_name,
                "harness_error": str(exc),
            }
            (layout.executor_dir / "metadata.json").write_text(
                json.dumps(error, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"gdpval[{executor_name}]: task {task.task_id}: harness failure: {exc}", file=sys.stderr)
            continue

        _write_executor_metadata(layout, result, copied)
        print(
            f"gdpval[{executor_name}]: task {task.task_id}: {result.status.value} ({len(copied)} deliverable(s))",
            file=sys.stderr,
        )
        if result.status not in _TERMINAL_SUCCESS:
            failures += 1

    summary = {"executor": executor_name, "failed_tasks": failures}
    (out_dir / "executor-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


def check() -> int:
    executor_name = os.getenv("GDPVAL_EXECUTOR", "codex")
    out_dir = Path(os.getenv("OUT", "./results/gdpval")).resolve()
    ok, payload = preflight(executor_name, out_dir, for_run=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "check":
        raise SystemExit(check())
    if mode == "run":
        raise SystemExit(run())
    raise SystemExit(f"unknown local-runner mode: {mode}")


if __name__ == "__main__":
    main()
