# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable

from gdpval_harness.benchmarks.base import BenchmarkTask
from gdpval_harness.benchmarks.gdpval import GDPvalBenchmark
from gdpval_harness.executors.base import ExecutionRequest, ExecutionResult, ExecutionStatus, TaskSpec
from gdpval_harness.executors.claude_code import ClaudeCodeExecutor
from gdpval_harness.executors.codex import CodexExecutor
from gdpval_harness.executors.cursor import CursorExecutor
from gdpval_harness.interventions import (
    Intervention,
    InterventionApplication,
    NoneIntervention,
    PromptOverlayIntervention,
    apply_prompt_overlay,
)
from gdpval_harness.layout import TaskLayout, task_layout
from gdpval_harness.reasoning import ReasoningEffortOption, validate_executor_reasoning_effort


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSONL = Path(
    os.getenv("GDPVAL_BENCHMARK_JSONL", ROOT / "benchmarks" / "gdpval" / "data" / "gdpval_benchmark.jsonl")
)
PREPARE_SCRIPT = Path(os.getenv("GDPVAL_PREPARE_SCRIPT", ROOT / "benchmarks" / "gdpval" / "prepare.py"))
_TERMINAL_SUCCESS = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
_MAX_CONDITION_FILE_BYTES = 1024 * 1024
_CONDITION_ENVIRONMENT_KEYS = frozenset({"GDPVAL_CONDITION", "GDPVAL_CONDITION_FILE", "GDPVAL_CONDITION_APPLIED"})


def _benchmark() -> GDPvalBenchmark:
    return GDPvalBenchmark(root=ROOT, dataset_path=BENCHMARK_JSONL, prepare_script=PREPARE_SCRIPT)


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


def _parse_reasoning_effort() -> ReasoningEffortOption:
    raw = os.getenv("GDPVAL_REASONING_EFFORT")
    if raw in {None, ""}:
        return None
    return validate_executor_reasoning_effort("codex", raw)


def _executor(name: str) -> CodexExecutor | ClaudeCodeExecutor | CursorExecutor:
    network_enabled = os.getenv("GDPVAL_EXECUTOR_NETWORK", "disabled") == "enabled"
    reasoning_effort = _parse_reasoning_effort()
    validate_executor_reasoning_effort(name, reasoning_effort)
    if name == "codex":
        return CodexExecutor(network_enabled=network_enabled, reasoning_effort=reasoning_effort)
    if name == "claude-code":
        return ClaudeCodeExecutor(network_enabled=network_enabled, max_turns=_parse_max_turns())
    if name == "cursor":
        return CursorExecutor(network_enabled=network_enabled)
    raise ValueError(f"local executor {name!r} is not implemented")


def _build_intervention() -> Intervention:
    raw = os.getenv("GDPVAL_CONDITION_FILE")
    if not raw:
        return NoneIntervention()
    # Keep the user-supplied path intact for the intervention's symlink and
    # regular-file checks.  The legacy condition provenance helpers continue
    # to use the resolved path for their existing hash/resume behavior.
    return PromptOverlayIntervention(Path(raw).expanduser())


def _executor_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in _CONDITION_ENVIRONMENT_KEYS:
        environment.pop(name, None)
    return environment


def _condition_file() -> Path | None:
    raw = os.getenv("GDPVAL_CONDITION_FILE")
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"--condition-file is not a readable file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"could not inspect --condition-file: {exc}") from exc
    if size > _MAX_CONDITION_FILE_BYTES:
        raise ValueError(
            f"--condition-file exceeds {_MAX_CONDITION_FILE_BYTES} bytes; "
            "keep experiment instructions small and reviewable"
        )
    return path


def _condition_instructions() -> str | None:
    path = _condition_file()
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read --condition-file as UTF-8: {exc}") from exc
    if not text.strip():
        raise ValueError("--condition-file is empty")
    return text.rstrip()


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"could not hash --condition-file: {exc}") from exc
    return digest.hexdigest()


def _current_condition_provenance() -> dict[str, object]:
    path = _condition_file()
    return {
        "condition": os.getenv("GDPVAL_CONDITION"),
        "condition_file_sha256": _sha256_file(path),
        "condition_applied_to_prompt": path is not None,
    }


def _resume_fingerprint(provenance: dict[str, object]) -> str:
    encoded = json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_resume_condition(out_dir: Path) -> None:
    if not _truthy("RESUME"):
        return

    current = _current_condition_provenance()
    current_reasoning_effort = _parse_reasoning_effort()
    metadata_path = out_dir / "run-metadata.json"
    if not metadata_path.is_file():
        if current["condition"] is not None or current["condition_file_sha256"] is not None:
            raise ValueError(
                "conditioned --resume requires the existing run-metadata.json so condition provenance can be verified"
            )
        if current_reasoning_effort is not None:
            raise ValueError(
                "reasoning-effort --resume requires the existing run-metadata.json so requested configuration "
                "can be verified"
            )
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot verify condition provenance from existing run metadata: {exc}") from exc
    config = metadata.get("configuration") if isinstance(metadata, dict) else None
    if not isinstance(config, dict):
        raise ValueError("existing run metadata has no configuration object; refusing resume")

    existing = {
        "condition": config.get("condition"),
        "condition_file_sha256": config.get("condition_file_sha256"),
        "condition_applied_to_prompt": bool(config.get("condition_applied_to_prompt")),
    }
    existing_reasoning_effort = config.get("reasoning_effort_requested")
    if existing_reasoning_effort is None or existing_reasoning_effort == "":
        existing_reasoning_effort = None
    else:
        existing_reasoning_effort = validate_executor_reasoning_effort("codex", existing_reasoning_effort)
    existing_resume = {
        **existing,
        "reasoning_effort_requested": existing_reasoning_effort,
    }
    current_resume = {
        **current,
        "reasoning_effort_requested": current_reasoning_effort,
    }
    stored_fingerprint = metadata.get("resume_fingerprint_sha256") if isinstance(metadata, dict) else None
    fingerprints_match = (
        stored_fingerprint == _resume_fingerprint(existing_resume)
        if isinstance(stored_fingerprint, str)
        else existing_resume == current_resume
    )
    if not fingerprints_match or existing_resume != current_resume:
        raise ValueError(
            "--resume requested provenance differs from the existing run; "
            "use the original condition/reasoning-effort configuration or a new output directory"
        )


def _ensure_dataset() -> None:
    _benchmark().prepare()


def _load_tasks(limit: int) -> list[BenchmarkTask]:
    return list(_benchmark().load_tasks(limit))


def _materialize_reference_files(task: BenchmarkTask, workspace: Path) -> list[str]:
    return list(_benchmark().materialize(task, workspace))


def _reference_listing(workspace: Path) -> str:
    ref_root = workspace / "reference_files"
    if not ref_root.is_dir():
        return "None"
    files = [str(path.relative_to(workspace)) for path in sorted(ref_root.rglob("*")) if path.is_file()]
    return "\n".join(f"- {item}" for item in files) if files else "None"


def build_task_prompt(
    task: TaskSpec,
    workspace: Path,
    *,
    network_policy: str,
    condition_instructions: str | None = None,
) -> str:
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
{task.prompt}
"""
    prompt_task = TaskSpec(task_id=task.task_id, prompt=prompt)
    if condition_instructions:
        prompt_task = apply_prompt_overlay(prompt_task, condition_instructions)
    return prompt_task.prompt


def _prepare_layout(out_dir: Path, task: TaskSpec) -> TaskLayout:
    layout = task_layout(out_dir, task.task_id)
    # A replacement run owns this task's published judge-facing output. Remove
    # it before staging/execution so a failed or interrupted replacement cannot
    # leave a stale prior finish marker that a later judge mistakes for the new run.
    shutil.rmtree(layout.judge_deliverables, ignore_errors=True)
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


def _application_evidence(application: InterventionApplication) -> dict[str, object]:
    return {
        "application_run_id": application.application_run_id,
        "bundle_sha256": application.bundle_sha256,
        "manifest_sha256": application.manifest_sha256,
        "application": {
            "method": application.application.method,
            "target": application.application.target,
        },
        "materialized_files": [
            {"path": item.path, "size": item.size, "sha256": item.sha256} for item in application.materialized_files
        ],
    }


def _write_executor_metadata(
    layout: TaskLayout,
    result: ExecutionResult,
    files: Iterable[str],
    *,
    intervention_application: InterventionApplication | None = None,
) -> None:
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
    if result.reasoning_effort_requested is not None:
        payload["reasoning_effort_requested"] = result.reasoning_effort_requested
    if intervention_application is not None:
        payload["intervention_application"] = _application_evidence(intervention_application)
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


def _intervention_payload(intervention: Intervention) -> tuple[bool, dict[str, object]]:
    try:
        result = intervention.preflight()
    except Exception as exc:
        return False, {
            "name": getattr(intervention, "name", type(intervention).__name__),
            "type": str(getattr(intervention, "intervention_type", "unknown")),
            "ok": False,
            "details": [f"intervention preflight failed: {exc}"],
        }
    return result.ok, {
        "name": result.name,
        "type": result.intervention_type.value,
        "ok": result.ok,
        "details": list(result.details),
    }


def _validated_details(payload: dict[str, object]) -> list[str]:
    raw_details = payload.get("details")
    if not isinstance(raw_details, list):
        raise TypeError("preflight details must be a list")
    details = [item for item in raw_details if isinstance(item, str)]
    if len(details) != len(raw_details):
        raise TypeError("preflight details must contain strings")
    return details


def preflight(
    executor_name: str,
    out_dir: Path,
    *,
    for_run: bool,
    intervention: Intervention | None = None,
) -> tuple[bool, dict[str, object]]:
    if intervention is None:
        try:
            intervention = _build_intervention()
        except ValueError as exc:
            return False, {
                "executor": executor_name,
                "ok": False,
                "version": None,
                "auth_mode": None,
                "details": [str(exc)],
                "intervention": None,
            }

    intervention_ok, intervention_details = _intervention_payload(intervention)
    if not intervention_ok:
        return False, {
            "executor": executor_name,
            "ok": False,
            "version": None,
            "auth_mode": None,
            "details": _validated_details(intervention_details),
            "intervention": intervention_details,
        }

    try:
        executor = _executor(executor_name)
    except ValueError as exc:
        return False, {
            "executor": executor_name,
            "ok": False,
            "version": None,
            "auth_mode": None,
            "details": [str(exc)],
            "intervention": intervention_details,
        }

    result = executor.preflight()
    details = list(result.details)
    ok = result.ok

    benchmark = _benchmark()
    if not benchmark.is_prepared() and not PREPARE_SCRIPT.is_file():
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
        _condition_file()
        if for_run and _truthy("RESUME"):
            _validate_resume_condition(out_dir)
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
        if benchmark.is_prepared()
        else "benchmark JSONL is not prepared yet; run will invoke the existing GDPval prepare script"
    )
    if os.getenv("GDPVAL_CONDITION"):
        details.append(f"experiment condition label: {os.environ['GDPVAL_CONDITION']}")
    if os.getenv("GDPVAL_CONDITION_FILE"):
        details.append("external condition instructions validated")
    return ok, {
        "executor": executor_name,
        "ok": ok,
        "version": result.version,
        "auth_mode": result.auth_mode,
        "details": details,
        "intervention": intervention_details,
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
    env["GDPVAL_CONDITION_APPLIED"] = "true" if os.getenv("GDPVAL_CONDITION_FILE") else "false"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gdpval_run_metadata.py")], cwd=ROOT, env=env, check=True)


def run() -> int:
    executor_name = os.getenv("GDPVAL_EXECUTOR", "codex")
    out_dir = Path(os.getenv("OUT", "./results/gdpval")).resolve()
    intervention = _build_intervention()
    ok, preflight_payload = preflight(executor_name, out_dir, for_run=True, intervention=intervention)
    for detail in _validated_details(preflight_payload):
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
    for benchmark_task in _load_tasks(limit):
        task = benchmark_task.execution
        layout = task_layout(out_dir, task.task_id)
        if resume and _already_completed(layout):
            print(f"gdpval[{executor_name}]: skip terminal task {task.task_id}", file=sys.stderr)
            continue

        layout = _prepare_layout(out_dir, task)
        # Record the canonical GDPval task separately from the executor wrapper
        # and any experiment-condition text. Blind judging uses this file when present.
        (layout.executor_dir / "task-prompt.txt").write_text(task.prompt, encoding="utf-8")
        try:
            _materialize_reference_files(benchmark_task, layout.workspace)
            prompt_task = TaskSpec(
                task_id=task.task_id,
                prompt=build_task_prompt(
                    task,
                    layout.workspace,
                    network_policy=network_policy,
                ),
            )
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

        try:
            intervention.validate_task(prompt_task)
            application = intervention.apply(
                prompt_task,
                layout.workspace,
                application_run_id=uuid.uuid4().hex,
            )
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
                "harness_error": "intervention application failed before executor",
                "intervention_error_type": type(exc).__name__,
            }
            (layout.executor_dir / "metadata.json").write_text(
                json.dumps(error, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"gdpval[{executor_name}]: task {task.task_id}: intervention failed before executor "
                f"({type(exc).__name__})",
                file=sys.stderr,
            )
            continue

        try:
            request = ExecutionRequest(
                task=application.task,
                workspace=layout.workspace,
                deliverables_dir=layout.workspace_deliverables,
                executor_dir=layout.executor_dir,
                model=os.getenv("GDPVAL_MODEL"),
                timeout_seconds=timeout,
                environment=_executor_environment(),
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

        _write_executor_metadata(
            layout,
            result,
            copied,
            intervention_application=application,
        )
        print(
            f"gdpval[{executor_name}]: task {task.task_id}: {result.status.value} ({len(copied)} deliverable(s))",
            file=sys.stderr,
        )
        if result.status not in _TERMINAL_SUCCESS:
            failures += 1

    summary = {
        "executor": executor_name,
        "condition": os.getenv("GDPVAL_CONDITION"),
        "failed_tasks": failures,
    }
    reasoning_effort = _parse_reasoning_effort()
    if reasoning_effort is not None:
        summary["reasoning_effort_requested"] = reasoning_effort
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
