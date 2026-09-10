#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip() or None
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


def main() -> None:
    root = Path.cwd()
    out_dir = Path(os.getenv("OUT", "./results/gdpval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "run-metadata.json"

    status = git_value("status", "--porcelain")
    reference_manifest = os.getenv("GDPVAL_REFERENCE_MANIFEST")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(status),
        },
        "configuration": {
            "env_yaml_sha256": sha256(root / "env.yaml"),
            "profile": os.getenv("GDPVAL_PROFILE"),
            "provider": os.getenv("GDPVAL_PROVIDER"),
            "model_type": os.getenv("GDPVAL_MODEL_TYPE", "vllm_model"),
            "model": os.getenv("GDPVAL_MODEL"),
            "base_url": os.getenv("GDPVAL_BASE_URL"),
            "policy_api_key_override": bool(os.getenv("GDPVAL_API_KEY")),
            "reward_mode": os.getenv("GDPVAL_REWARD_MODE", "rubric"),
            "references_dir": os.getenv("GDPVAL_REFS"),
            "reference_manifest": reference_manifest,
            "reference_manifest_sha256": sha256(Path(reference_manifest)) if reference_manifest else None,
            "judge_panel": os.getenv("GDPVAL_JUDGE_PANEL", "aa-v2"),
            "judge_model": os.getenv("GDPVAL_JUDGE_MODEL"),
            "judge_base_url": os.getenv("GDPVAL_JUDGE_BASE_URL"),
            "judge_api_key_override": bool(os.getenv("GDPVAL_JUDGE_API_KEY")),
            "judge_sampling_seed": os.getenv("JUDGE_SAMPLING_SEED", "42"),
            "judge_only": truthy("JUDGE_ONLY"),
            "limit": os.getenv("LIMIT"),
            "parallel": os.getenv("PARALLEL"),
            "resume": truthy("RESUME"),
            "pin_gym": truthy("PIN_GYM"),
        },
        "output": {
            "directory": str(out_dir),
            "deliverables_dir": os.getenv("PERSIST_DELIVERABLES_DIR"),
        },
    }

    temp_path = metadata_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(metadata_path)
    print(f"gdpval: wrote run metadata to {metadata_path}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
