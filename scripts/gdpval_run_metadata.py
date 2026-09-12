#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_value(*args: str) -> str | None:
    try:
        return (
            subprocess.check_output(["git", *args], text=True, errors="replace", stderr=subprocess.DEVNULL).strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(name: str) -> bool:
    return os.getenv(name, "").lower() not in {"", "0", "false", "no"}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> None:
    root = Path.cwd()
    out_dir = Path(os.getenv("OUT", "./results/gdpval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "run-metadata.json"

    command = os.getenv("GDPVAL_COMMAND", "run")
    status = git_value("status", "--porcelain")
    reference_manifest = os.getenv("GDPVAL_REFERENCE_MANIFEST")
    condition_file = os.getenv("GDPVAL_CONDITION_FILE")
    executor = None if command == "compare-runs" else os.getenv("GDPVAL_EXECUTOR", "stirrup")
    judge_executor = os.getenv("GDPVAL_JUDGE_EXECUTOR")
    configuration = {
        "env_yaml_sha256": sha256(root / "env.yaml"),
        "command": command,
        "profile": os.getenv("GDPVAL_PROFILE"),
        "evaluation_mode": os.getenv("GDPVAL_EVALUATION_MODE"),
        "condition": os.getenv("GDPVAL_CONDITION"),
        "condition_file": condition_file,
        "condition_file_sha256": sha256(Path(condition_file).expanduser()) if condition_file else None,
        "condition_applied_to_prompt": truthy("GDPVAL_CONDITION_APPLIED"),
        "executor": executor,
        "executor_timeout_seconds": os.getenv("GDPVAL_EXECUTOR_TIMEOUT") if executor else None,
        "executor_max_turns": os.getenv("GDPVAL_EXECUTOR_MAX_TURNS") if executor else None,
        "executor_version": os.getenv("GDPVAL_EXECUTOR_VERSION") if executor else None,
        "executor_invocation_mode": os.getenv("GDPVAL_EXECUTOR_INVOCATION_MODE") if executor else None,
        "executor_auth_mode": os.getenv("GDPVAL_EXECUTOR_AUTH_MODE") if executor else None,
        "executor_workspace_isolation": os.getenv("GDPVAL_EXECUTOR_WORKSPACE_ISOLATION") if executor else None,
        "executor_network_policy": os.getenv("GDPVAL_EXECUTOR_NETWORK") if executor else None,
        "executor_tool_permission_mode": os.getenv("GDPVAL_EXECUTOR_TOOL_PERMISSION_MODE") if executor else None,
        "provider": os.getenv("GDPVAL_PROVIDER") if executor == "stirrup" else None,
        "model_type": os.getenv("GDPVAL_MODEL_TYPE", "vllm_model") if executor == "stirrup" else None,
        "model": os.getenv("GDPVAL_MODEL") if executor else None,
        "base_url": os.getenv("GDPVAL_BASE_URL") if executor == "stirrup" else None,
        "policy_api_key_override": bool(os.getenv("GDPVAL_API_KEY")) if executor == "stirrup" else False,
        "reward_mode": os.getenv("GDPVAL_REWARD_MODE", "rubric"),
        "references_dir": os.getenv("GDPVAL_REFS"),
        "reference_manifest": reference_manifest,
        "candidate_a": os.getenv("GDPVAL_RUN_A"),
        "candidate_b": os.getenv("GDPVAL_RUN_B"),
        "label_a": os.getenv("GDPVAL_LABEL_A"),
        "label_b": os.getenv("GDPVAL_LABEL_B"),
        "single_reference_dir": os.getenv("GDPVAL_SINGLE_REFERENCE_DIR"),
        "reference_manifest_sha256": sha256(Path(reference_manifest)) if reference_manifest else None,
        "judge_executor": judge_executor,
        "judge_executor_version": os.getenv("GDPVAL_JUDGE_EXECUTOR_VERSION"),
        "judge_executor_auth_mode": os.getenv("GDPVAL_JUDGE_EXECUTOR_AUTH_MODE"),
        "judge_trials": os.getenv("GDPVAL_JUDGE_TRIALS"),
        "judge_timeout_seconds": os.getenv("GDPVAL_JUDGE_TIMEOUT"),
        "judge_panel": None if judge_executor else os.getenv("GDPVAL_JUDGE_PANEL", "aa-v2"),
        "judge_model": os.getenv("GDPVAL_JUDGE_MODEL"),
        "judge_base_url": None if judge_executor else os.getenv("GDPVAL_JUDGE_BASE_URL"),
        "judge_api_key_override": False if judge_executor else bool(os.getenv("GDPVAL_JUDGE_API_KEY")),
        "judge_sampling_seed": os.getenv("JUDGE_SAMPLING_SEED", "42"),
        "judge_only": truthy("JUDGE_ONLY"),
        "limit": os.getenv("LIMIT"),
        "parallel": os.getenv("PARALLEL"),
        "resume": truthy("RESUME"),
        "pin_gym": truthy("PIN_GYM"),
    }
    reasoning_effort = os.getenv("GDPVAL_REASONING_EFFORT") or None
    if executor == "codex" and reasoning_effort is not None:
        configuration["reasoning_effort_requested"] = reasoning_effort
    judge_reasoning_effort = os.getenv("GDPVAL_JUDGE_REASONING_EFFORT") or None
    if judge_executor == "codex" and judge_reasoning_effort is not None:
        configuration["judge_reasoning_effort_requested"] = judge_reasoning_effort
    resume_configuration = {
        "condition": configuration["condition"],
        "condition_file_sha256": configuration["condition_file_sha256"],
        "condition_applied_to_prompt": configuration["condition_applied_to_prompt"],
        "reasoning_effort_requested": configuration.get("reasoning_effort_requested"),
    }
    configuration_sha256 = canonical_sha256(configuration)
    resume_fingerprint_sha256 = canonical_sha256(resume_configuration)
    run_fingerprint_sha256 = canonical_sha256(
        {
            "configuration_sha256": configuration_sha256,
            "repository": {
                "commit": git_value("rev-parse", "HEAD"),
                "dirty": bool(status),
            },
        }
    )
    payload = {
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(status),
        },
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "resume_fingerprint_sha256": resume_fingerprint_sha256,
        "run_fingerprint_sha256": run_fingerprint_sha256,
        "output": {
            "directory": str(out_dir),
            "deliverables_dir": os.getenv("PERSIST_DELIVERABLES_DIR"),
        },
    }

    temp_path = metadata_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(metadata_path)
    print(f"gdpval: wrote run metadata to {metadata_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
