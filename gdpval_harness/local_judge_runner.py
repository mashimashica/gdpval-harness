# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from gdpval_harness.judges.base import JudgeRequest, Verdict
from gdpval_harness.judges.claude_code import ClaudeCodeJudgeExecutor
from gdpval_harness.judges.codex import CodexJudgeExecutor
from gdpval_harness.judges.pairwise import (
    aggregate,
    build_judge_prompt,
    matched_tasks,
    normalize_verdict,
    prepare_trial,
    write_trial_metadata,
)
from gdpval_harness.layout import safe_task_id


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSONL = Path(
    os.getenv("GDPVAL_BENCHMARK_JSONL", ROOT / "benchmarks" / "gdpval" / "data" / "gdpval_benchmark.jsonl")
)
PREPARE_SCRIPT = Path(os.getenv("GDPVAL_PREPARE_SCRIPT", ROOT / "benchmarks" / "gdpval" / "prepare.py"))


def _judge_executor(name: str):
    if name == "codex":
        return CodexJudgeExecutor()
    if name == "claude-code":
        return ClaudeCodeJudgeExecutor()
    raise ValueError("local judge executor must be codex or claude-code")


def _positive_int(name: str, default: int | None = None) -> int:
    raw = os.getenv(name)
    if raw in {None, ""}:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
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
    with BENCHMARK_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"task_{safe_task_id(str(row['task_id']))}"
            prompts[key] = str(row["prompt"])
    return prompts


def _run_root(deliverables: Path) -> Path:
    return deliverables.parent if deliverables.name == "deliverables" else deliverables


def _read_run_metadata(deliverables: Path) -> dict[str, object] | None:
    path = _run_root(deliverables) / "run-metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _generator_info(deliverables: Path) -> dict[str, object]:
    metadata = _read_run_metadata(deliverables) or {}
    config = metadata.get("configuration") if isinstance(metadata, dict) else {}
    if not isinstance(config, dict):
        config = {}
    return {
        "executor": config.get("executor"),
        "model": config.get("model"),
        "repository_commit": (metadata.get("repository") or {}).get("commit")
        if isinstance(metadata.get("repository"), dict)
        else None,
    }


def _preflight(for_run: bool) -> tuple[bool, dict[str, object]]:
    judge_name = os.getenv("GDPVAL_JUDGE_EXECUTOR", "")
    details: list[str] = []
    ok = True
    try:
        judge = _judge_executor(judge_name)
        auth = judge.preflight()
        ok = auth.ok
        details.extend(auth.details)
        version = auth.version
        auth_mode = auth.auth_mode
    except ValueError as exc:
        return False, {"judge_executor": judge_name, "ok": False, "version": None, "auth_mode": None, "details": [str(exc)]}

    try:
        _positive_int("GDPVAL_JUDGE_TRIALS", 2)
        _positive_float("GDPVAL_JUDGE_TIMEOUT", 3600.0)
        _integer("JUDGE_SAMPLING_SEED", 42)
        if for_run:
            _positive_int("LIMIT")
    except ValueError as exc:
        details.append(str(exc))
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
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".judge-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        details.append(f"judge output directory is not writable: {exc}")
        ok = False

    return ok, {
        "judge_executor": judge_name,
        "ok": ok,
        "version": version,
        "auth_mode": auth_mode,
        "details": details,
    }


def _write_run_metadata(out_dir: Path, preflight: dict[str, object]) -> None:
    env = os.environ.copy()
    env["OUT"] = str(out_dir)
    env["GDPVAL_JUDGE_EXECUTOR_VERSION"] = str(preflight.get("version") or "")
    env["GDPVAL_JUDGE_EXECUTOR_AUTH_MODE"] = str(preflight.get("auth_mode") or "")
    env["GDPVAL_EVALUATION_MODE"] = "local-subscription-pairwise"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gdpval_run_metadata.py")], cwd=ROOT, env=env, check=True)


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
    pairs = matched_tasks(candidate_a, candidate_b)[:limit]

    if os.getenv("GDPVAL_WRITE_METADATA", "1") != "0":
        _write_run_metadata(out_dir, preflight)

    results_path = out_dir / "local-judge-results.jsonl"
    normalized_verdicts: list[Verdict] = []
    invalid_trials = 0
    rows: list[dict[str, object]] = []

    for task_key, task_a, task_b in pairs:
        if task_key not in prompts:
            print(f"gdpval: benchmark prompt not found for {task_key}", file=sys.stderr)
            return 2
        judge_prompt = build_judge_prompt(prompts[task_key])
        for trial_index in range(trials):
            prepared = prepare_trial(out_dir, task_key, task_a, task_b, trial_index=trial_index, seed=seed)
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
                environment=os.environ.copy(),
            )
            result = judge.judge(request)
            normalized = normalize_verdict(result.verdict, prepared.swapped) if result.verdict else None
            if normalized is None:
                invalid_trials += 1
            else:
                normalized_verdicts.append(normalized)
            row = {
                "task_id": result.task_id,
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
            rows.append(row)
            write_trial_metadata(prepared.executor_dir / "metadata.json", row)

    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    generator_a = _generator_info(candidate_a)
    generator_b = _generator_info(candidate_b)
    judge_model = os.getenv("GDPVAL_JUDGE_MODEL")

    def candidate_summary(label: str | None, generator: dict[str, object]) -> dict[str, object]:
        generator_model = generator.get("model")
        return {
            "label": label,
            "generator": generator,
            "same_executor_as_judge": generator.get("executor") == preflight["judge_executor"],
            "same_model_as_judge": bool(generator_model and judge_model and generator_model == judge_model),
        }

    summary = {
        "evaluation_mode": "local-subscription-pairwise",
        "official_gdpval_aa_v2": False,
        "judge_executor": preflight["judge_executor"],
        "judge_executor_version": preflight["version"],
        "judge_auth_mode": preflight["auth_mode"],
        "judge_model": judge_model,
        "tasks": len(pairs),
        "trials_per_task": trials,
        "invalid_trials": invalid_trials,
        "result": aggregate(normalized_verdicts),
        "candidate_a": candidate_summary(os.getenv("GDPVAL_LABEL_A"), generator_a),
        "candidate_b": candidate_summary(os.getenv("GDPVAL_LABEL_B"), generator_b),
    }
    (out_dir / "local-judge-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
