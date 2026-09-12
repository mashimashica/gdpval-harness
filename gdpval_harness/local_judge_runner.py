# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from gdpval_harness.judges.base import JudgeExecutor, JudgeRequest, Verdict
from gdpval_harness.judges.claude_code import ClaudeCodeJudgeExecutor
from gdpval_harness.judges.codex import CodexJudgeExecutor
from gdpval_harness.judges.pairwise import (
    aggregate,
    build_judge_prompt,
    matched_tasks,
    normalize_verdict,
    prepare_trial,
    tree_hash,
    validate_reference_equivalence,
    write_trial_metadata,
)
from gdpval_harness.layout import safe_task_id
from gdpval_harness.reasoning import ReasoningEffortOption, validate_executor_reasoning_effort


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSONL = Path(
    os.getenv("GDPVAL_BENCHMARK_JSONL", ROOT / "benchmarks" / "gdpval" / "data" / "gdpval_benchmark.jsonl")
)
PREPARE_SCRIPT = Path(os.getenv("GDPVAL_PREPARE_SCRIPT", ROOT / "benchmarks" / "gdpval" / "prepare.py"))
_TASK_PROMPT_MARKER = b"\nTask:\n"
_JUDGE_ENV_ALLOWLIST = {
    "ALL_PROXY",
    "APPDATA",
    "CODEX_CA_CERTIFICATE",
    "CODEX_HOME",
    "COMSPEC",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "LOGNAME",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SystemRoot",
    "TERM",
    "USER",
    "USERPROFILE",
    "WINDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "XDG_STATE_HOME",
    "CLAUDE_CONFIG_DIR",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


class _PreflightRecord(TypedDict):
    judge_executor: str
    ok: bool
    version: str | None
    auth_mode: str | None
    details: list[str]
    temp_parent: str | None


def _truthy(name: str) -> bool:
    return os.getenv(name, "").lower() not in {"", "0", "false", "no"}


def _judge_environment(runtime_tmp: Path | None = None) -> dict[str, str]:
    """Return runtime/auth-store plumbing only; never candidate identity or unrelated secrets."""
    env = {name: value for name in _JUDGE_ENV_ALLOWLIST if (value := os.getenv(name)) is not None}
    if runtime_tmp is not None:
        value = str(runtime_tmp)
        env.update({"TMPDIR": value, "TMP": value, "TEMP": value})
    return env


def _parse_reasoning_effort() -> ReasoningEffortOption:
    raw = os.getenv("GDPVAL_JUDGE_REASONING_EFFORT")
    if raw in {None, ""}:
        return None
    return validate_executor_reasoning_effort("codex", raw)


def _judge_executor(name: str) -> JudgeExecutor:
    reasoning_effort = _parse_reasoning_effort()
    validate_executor_reasoning_effort(name, reasoning_effort)
    if name == "codex":
        return CodexJudgeExecutor(reasoning_effort=reasoning_effort)
    if name == "claude-code":
        return ClaudeCodeJudgeExecutor()
    raise ValueError("local judge executor must be codex or claude-code")


def _positive_int(name: str, default: int | None = None) -> int:
    raw = os.getenv(name)
    if raw in {None, ""}:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    assert raw is not None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in {None, ""}:
        return default
    assert raw is not None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _integer(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in {None, ""}:
        return default
    assert raw is not None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _ensure_dataset() -> None:
    if BENCHMARK_JSONL.is_file():
        return
    if not PREPARE_SCRIPT.is_file():
        raise RuntimeError(f"GDPval benchmark data is absent and prepare script is missing: {PREPARE_SCRIPT}")
    result = subprocess.run([sys.executable, str(PREPARE_SCRIPT)], cwd=ROOT, check=False)
    if result.returncode != 0 or not BENCHMARK_JSONL.is_file():
        raise RuntimeError("failed to prepare GDPval benchmark data")


def _task_prompts() -> dict[str, str]:
    prompts: dict[str, str] = {}
    with BENCHMARK_JSONL.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"task_{safe_task_id(str(row['task_id']))}"
            prompts[key] = str(row["prompt"])
    return prompts


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_root(deliverables: Path) -> Path:
    return deliverables.parent if deliverables.name == "deliverables" else deliverables


def _read_run_metadata(deliverables: Path) -> dict[str, object] | None:
    path = _run_root(deliverables) / "run-metadata.json"
    if not path.is_file():
        return None
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            return None
        metadata[key] = value
    return metadata


def _generator_info(deliverables: Path) -> dict[str, object]:
    metadata = _read_run_metadata(deliverables) or {}
    config = metadata.get("configuration") if isinstance(metadata, dict) else {}
    if not isinstance(config, dict):
        config = {}
    repository = metadata.get("repository")
    commit = repository.get("commit") if isinstance(repository, dict) else None
    return {
        "executor": config.get("executor"),
        "model": config.get("model"),
        "repository_commit": commit,
    }


def _candidate_task_prompt(deliverables: Path, task_key: str) -> str:
    task_id = task_key.removeprefix("task_")
    executor_dir = _run_root(deliverables) / "tasks" / task_id / "executor"
    canonical = executor_dir / "task-prompt.txt"
    if canonical.is_symlink():
        raise ValueError(f"candidate canonical task prompt is symlinked for {task_key}: {canonical}")
    if canonical.is_file():
        try:
            return canonical.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"candidate canonical task prompt is not readable UTF-8 for {task_key}: {exc}") from exc

    path = executor_dir / "prompt.txt"
    if not path.is_file():
        raise ValueError(
            f"candidate run is missing the recorded task provenance for {task_key}: {executor_dir}; "
            "local judging requires provenance from a subscription-backed harness run"
        )
    raw = path.read_bytes()
    if _TASK_PROMPT_MARKER not in raw:
        raise ValueError(f"candidate executor prompt for {task_key} does not contain the GDPval task marker")
    recorded = raw.split(_TASK_PROMPT_MARKER, 1)[1]
    if recorded.endswith(b"\n"):
        recorded = recorded[:-1]
    return recorded.decode("utf-8", errors="replace")


def _validate_selected_pairs(
    candidate_a: Path,
    candidate_b: Path,
    pairs: list[tuple[str, Path, Path]],
    prompts: dict[str, str],
) -> dict[str, str]:
    prompt_hashes: dict[str, str] = {}
    for task_key, task_a, task_b in pairs:
        benchmark_prompt = prompts.get(task_key)
        if benchmark_prompt is None:
            raise ValueError(f"benchmark prompt not found for {task_key}")

        tree_hash(task_a)
        tree_hash(task_b)
        validate_reference_equivalence(task_a, task_b)

        prompt_a = _candidate_task_prompt(candidate_a, task_key)
        prompt_b = _candidate_task_prompt(candidate_b, task_key)
        if prompt_a != benchmark_prompt:
            raise ValueError(f"candidate A prompt for {task_key} differs from the current GDPval benchmark prompt")
        if prompt_b != benchmark_prompt:
            raise ValueError(f"candidate B prompt for {task_key} differs from the current GDPval benchmark prompt")
        prompt_hashes[task_key] = _sha256_text(benchmark_prompt)
    return prompt_hashes


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _safe_temp_parent(candidate_a: Path, candidate_b: Path, out_dir: Path) -> Path:
    protected = [candidate_a.resolve(), candidate_b.resolve(), out_dir.resolve()]
    candidates: list[Path] = []
    if os.name != "nt":
        candidates.extend((Path("/tmp"), Path("/var/tmp"), Path("/usr/tmp")))
    system_root = os.getenv("SystemRoot") or os.getenv("WINDIR")
    if system_root:
        candidates.append(Path(system_root) / "Temp")
    candidates.append(Path(tempfile.gettempdir()))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_dir() or any(_paths_overlap(resolved, path) for path in protected):
            continue
        try:
            probe = tempfile.mkdtemp(prefix=".gdpval-judge-probe-", dir=resolved)
        except OSError:
            continue
        else:
            shutil.rmtree(probe, ignore_errors=True)
            return resolved
    raise RuntimeError("no writable system temporary directory is disjoint from both candidates and --out")


def _preflight(for_run: bool) -> tuple[bool, _PreflightRecord]:
    judge_name = os.getenv("GDPVAL_JUDGE_EXECUTOR", "")
    details: list[str] = []
    ok = True
    try:
        judge = _judge_executor(judge_name)
        auth = judge.preflight(_judge_environment())
        ok = auth.ok
        details.extend(auth.details)
        version = auth.version
        auth_mode = auth.auth_mode
    except ValueError as exc:
        return False, {
            "judge_executor": judge_name,
            "ok": False,
            "version": None,
            "auth_mode": None,
            "temp_parent": None,
            "details": [str(exc)],
        }

    try:
        _positive_int("GDPVAL_JUDGE_TRIALS", 2)
        _positive_float("GDPVAL_JUDGE_TIMEOUT", 3600.0)
        _integer("JUDGE_SAMPLING_SEED", 42)
        if for_run:
            _positive_int("LIMIT")
    except ValueError as exc:
        details.append(str(exc))
        ok = False

    if for_run and _truthy("RESUME"):
        details.append("--resume is not supported for local subscription judges; use a new output directory")
        ok = False

    candidate_a = Path(os.getenv("GDPVAL_RUN_A", ""))
    candidate_b = Path(os.getenv("GDPVAL_RUN_B", ""))
    if for_run:
        for label, path in (("A", candidate_a), ("B", candidate_b)):
            if not path.is_dir():
                details.append(f"candidate {label} deliverables not found: {path}")
                ok = False
        if not BENCHMARK_JSONL.is_file() and not PREPARE_SCRIPT.is_file():
            details.append("GDPval benchmark data and prepare script are both unavailable")
            ok = False

    out_dir = Path(os.getenv("OUT", "./results/compare-runs"))
    if for_run:
        occupied = [
            out_dir / "local-judge-results.jsonl",
            out_dir / "local-judge-summary.json",
            out_dir / "run-metadata.json",
            out_dir / "judge",
        ]
        existing = [str(path) for path in occupied if path.exists()]
        if existing:
            details.append(
                "local judge output already contains prior run data; use a new --out directory: " + ", ".join(existing)
            )
            ok = False
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".judge-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        details.append(f"judge output directory is not writable: {exc}")
        ok = False

    temp_parent: Path | None = None
    if for_run and candidate_a.is_dir() and candidate_b.is_dir():
        try:
            temp_parent = _safe_temp_parent(candidate_a, candidate_b, out_dir)
        except (OSError, RuntimeError) as exc:
            details.append(f"cannot establish an isolated judge temp root: {exc}")
            ok = False

    return ok, {
        "judge_executor": judge_name,
        "ok": ok,
        "version": version,
        "auth_mode": auth_mode,
        "temp_parent": str(temp_parent) if temp_parent else None,
        "details": details,
    }


def _write_run_metadata(out_dir: Path, preflight: _PreflightRecord) -> None:
    env = os.environ.copy()
    env["OUT"] = str(out_dir)
    env["GDPVAL_JUDGE_EXECUTOR_VERSION"] = str(preflight.get("version") or "")
    env["GDPVAL_JUDGE_EXECUTOR_AUTH_MODE"] = str(preflight.get("auth_mode") or "")
    env["GDPVAL_EVALUATION_MODE"] = "local-subscription-pairwise"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gdpval_run_metadata.py")], cwd=ROOT, env=env, check=True)


def _persist_executor_logs(
    out_dir: Path, task_key: str, trial_index: int, source: Path, row: dict[str, object]
) -> None:
    target = out_dir / "judge" / "tasks" / task_key / f"trial_{trial_index}" / "executor"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    write_trial_metadata(target / "metadata.json", row)


def _interrupted_row(task_key: str, trial_index: int, swapped: bool, preflight: _PreflightRecord) -> dict[str, object]:
    row = {
        "task_id": task_key.removeprefix("task_"),
        "trial_index": trial_index,
        "swapped": swapped,
        "blind_verdict": None,
        "normalized_verdict": None,
        "judge_executor": preflight["judge_executor"],
        "judge_executor_version": preflight.get("version"),
        "judge_auth_mode": preflight.get("auth_mode"),
        "judge_model": os.getenv("GDPVAL_JUDGE_MODEL"),
        "exit_code": 130,
        "started_at": None,
        "finished_at": None,
        "metadata": {"interrupted": True},
    }
    reasoning_effort = _parse_reasoning_effort()
    if reasoning_effort is not None:
        row["reasoning_effort_requested"] = reasoning_effort
    return row


def run() -> int:
    ok, preflight = _preflight(for_run=True)
    for detail in preflight["details"]:
        print(f"gdpval[judge:{preflight['judge_executor']}]: {detail}", file=sys.stderr)
    if not ok:
        print("gdpval: local judge preflight failed", file=sys.stderr)
        return 2

    _ensure_dataset()
    candidate_a = Path(os.environ["GDPVAL_RUN_A"]).resolve()
    candidate_b = Path(os.environ["GDPVAL_RUN_B"]).resolve()
    out_dir = Path(os.getenv("OUT", "./results/compare-runs")).resolve()
    judge = _judge_executor(str(preflight["judge_executor"]))
    limit = _positive_int("LIMIT")
    trials = _positive_int("GDPVAL_JUDGE_TRIALS", 2)
    timeout = _positive_float("GDPVAL_JUDGE_TIMEOUT", 3600.0)
    seed = _integer("JUDGE_SAMPLING_SEED", 42)
    prompts = _task_prompts()

    try:
        pairs = matched_tasks(candidate_a, candidate_b)[:limit]
        prompt_hashes = _validate_selected_pairs(candidate_a, candidate_b, pairs, prompts)
    except (OSError, ValueError) as exc:
        print(f"gdpval: local judge candidate validation failed: {exc}", file=sys.stderr)
        return 2

    temp_parent_raw = preflight.get("temp_parent")
    if not isinstance(temp_parent_raw, str) or not temp_parent_raw:
        print("gdpval: isolated judge temp parent was not established", file=sys.stderr)
        return 2
    temp_parent = Path(temp_parent_raw)

    results_path = out_dir / "local-judge-results.jsonl"
    normalized_verdicts: list[Verdict] = []
    invalid_trials = 0
    interrupted = False
    systemic_failure: str | None = None

    with results_path.open("x", encoding="utf-8", newline="\n") as results_handle:
        stop = False
        for task_key, task_a, task_b in pairs:
            judge_prompt = build_judge_prompt(prompts[task_key])
            for trial_index in range(trials):
                with tempfile.TemporaryDirectory(prefix="gdpval-judge-", dir=temp_parent) as temp_root:
                    temp_root_path = Path(temp_root)
                    runtime_tmp = temp_root_path / "runtime-tmp"
                    runtime_tmp.mkdir()
                    prepared = prepare_trial(
                        temp_root_path, task_key, task_a, task_b, trial_index=trial_index, seed=seed
                    )
                    request = JudgeRequest(
                        task_id=task_key.removeprefix("task_"),
                        task_prompt=judge_prompt,
                        workspace=prepared.workspace,
                        reference_dir=prepared.reference_dir,
                        submission_a_dir=prepared.submission_a_dir,
                        submission_b_dir=prepared.submission_b_dir,
                        executor_dir=prepared.executor_dir,
                        trial_index=trial_index,
                        swapped=prepared.swapped,
                        model=os.getenv("GDPVAL_JUDGE_MODEL"),
                        timeout_seconds=timeout,
                        environment=_judge_environment(runtime_tmp),
                    )
                    try:
                        result = judge.judge(request)
                    except KeyboardInterrupt:
                        row = _interrupted_row(task_key, trial_index, prepared.swapped, preflight)
                        results_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        results_handle.flush()
                        _persist_executor_logs(out_dir, task_key, trial_index, prepared.executor_dir, row)
                        interrupted = True
                        stop = True
                        break

                    normalized = normalize_verdict(result.verdict, prepared.swapped) if result.verdict else None
                    if normalized is None:
                        invalid_trials += 1
                    else:
                        normalized_verdicts.append(normalized)
                    row = {
                        "task_id": result.task_id,
                        "task_prompt_sha256": prompt_hashes[task_key],
                        "trial_index": trial_index,
                        "swapped": prepared.swapped,
                        "blind_verdict": result.verdict.value if result.verdict else None,
                        "normalized_verdict": normalized.value if normalized else None,
                        "judge_executor": result.judge_executor,
                        "judge_executor_version": result.executor_version,
                        "judge_auth_mode": result.auth_mode,
                        "judge_model": os.getenv("GDPVAL_JUDGE_MODEL"),
                        "exit_code": result.exit_code,
                        "started_at": result.started_at,
                        "finished_at": result.finished_at,
                        "metadata": dict(result.metadata),
                    }
                    if result.reasoning_effort_requested is not None:
                        row["reasoning_effort_requested"] = result.reasoning_effort_requested
                    results_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    results_handle.flush()
                    _persist_executor_logs(out_dir, task_key, trial_index, prepared.executor_dir, row)

                    if result.exit_code is None or result.exit_code != 0:
                        systemic_failure = (
                            f"judge executor failed for {task_key} trial {trial_index}: exit_code={result.exit_code}"
                        )
                        stop = True
                        break
            if stop:
                break

    if os.getenv("GDPVAL_WRITE_METADATA", "1") != "0":
        _write_run_metadata(out_dir, preflight)

    generator_a = _generator_info(candidate_a)
    generator_b = _generator_info(candidate_b)
    judge_model = os.getenv("GDPVAL_JUDGE_MODEL")

    def candidate_summary(label: str | None, generator: dict[str, object]) -> dict[str, object]:
        generator_executor = generator.get("executor")
        generator_model = generator.get("model")
        same_executor = None if not generator_executor else generator_executor == preflight["judge_executor"]
        same_model = None if not generator_model or not judge_model else generator_model == judge_model
        return {
            "label": label,
            "generator": generator,
            "same_executor_as_judge": same_executor,
            "same_model_as_judge": same_model,
        }

    summary = {
        "evaluation_mode": "local-subscription-pairwise",
        "official_gdpval_aa_v2": False,
        "judge_executor": preflight["judge_executor"],
        "judge_executor_version": preflight["version"],
        "judge_auth_mode": preflight["auth_mode"],
        "judge_model": judge_model,
        "judge_environment_policy": "minimal-runtime-auth-allowlist",
        "judge_workspace_isolation": "per-trial-system-temp-disjoint-from-candidates-and-output",
        "provenance_write_timing": "after-all-judge-model-calls",
        "local_judge_resume_supported": False,
        "task_prompt_binding": "canonical task-prompt.txt when present; legacy recorded wrapper otherwise",
        "tasks": len(pairs),
        "trials_per_task": trials,
        "invalid_trials": invalid_trials,
        "interrupted": interrupted,
        "systemic_failure": systemic_failure,
        "result": aggregate(normalized_verdicts),
        "candidate_a": candidate_summary(os.getenv("GDPVAL_LABEL_A"), generator_a),
        "candidate_b": candidate_summary(os.getenv("GDPVAL_LABEL_B"), generator_b),
    }
    reasoning_effort = _parse_reasoning_effort()
    if reasoning_effort is not None:
        summary["reasoning_effort_requested"] = reasoning_effort
    (out_dir / "local-judge-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if interrupted:
        print("gdpval: local judge interrupted; partial logs/results preserved", file=sys.stderr)
        return 130
    if systemic_failure:
        print(f"gdpval: {systemic_failure}; stopping further judge calls", file=sys.stderr)
        return 1
    if invalid_trials:
        print(f"gdpval: {invalid_trials} local judge trial(s) were invalid", file=sys.stderr)
        return 1
    return 0


def check() -> int:
    ok, payload = _preflight(for_run=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        raise SystemExit(run())
    if mode == "check":
        raise SystemExit(check())
    raise SystemExit(f"unknown local-judge mode: {mode}")


if __name__ == "__main__":
    main()
