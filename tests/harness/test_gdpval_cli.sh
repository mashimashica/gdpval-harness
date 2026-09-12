#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash -n gdpval
bash -n eval
bash -n scripts/gdpval_provider.sh
bash -n scripts/gdpval_preflight.sh
bash -n tests/harness/test_experiment_conditions.sh
bash -n tests/harness/test_eval_cli.sh
bash -n nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh

help="$(./gdpval --help)"
grep -q './gdpval aa-v2 --refs DIR' <<<"$help"
grep -q './gdpval compare-runs --a DIR --b DIR' <<<"$help"
grep -q './gdpval executors' <<<"$help"
grep -q -- '--executor NAME' <<<"$help"
grep -q -- '--judge-executor NAME' <<<"$help"
grep -q -- '--reasoning-effort VALUE' <<<"$help"
grep -q -- '--judge-reasoning-effort VALUE' <<<"$help"
grep -q -- '--condition LABEL' <<<"$help"
grep -q -- '--condition-file FILE' <<<"$help"
grep -q 'rejected by Stirrup' <<<"$help"

providers="$(./gdpval providers)"
grep -q '^openai ' <<<"$providers"
grep -q '^gemini ' <<<"$providers"
grep -q '^openrouter ' <<<"$providers"

executors="$(./gdpval executors)"
grep -q '^claude-code' <<<"$executors"
grep -q '^codex' <<<"$executors"
grep -q '^cursor' <<<"$executors"
grep -q '^stirrup' <<<"$executors"

overrides="$(OPENAI_API_KEY=test-policy-key bash -c '
  source scripts/gdpval_provider.sh
  gdpval_apply_provider openai
  printf "%s|%s|%s" "$GDPVAL_MODEL_TYPE" "$GDPVAL_BASE_URL" "$GDPVAL_API_KEY"
')"
[[ "$overrides" == 'openai_model|https://api.openai.com/v1|test-policy-key' ]]

if ./gdpval run --executor not-real --no-metadata >/dev/null 2>&1; then
  echo "unknown executor unexpectedly passed preflight" >&2
  exit 1
fi
if ./gdpval aa-v2 --refs /tmp --executor not-real --no-metadata >/dev/null 2>&1; then
  echo "aa-v2 unexpectedly accepted a non-Stirrup executor" >&2
  exit 1
fi
if ./gdpval aa-v2 --refs /tmp --judge-executor codex --no-metadata >/dev/null 2>&1; then
  echo "aa-v2 unexpectedly accepted a local judge executor" >&2
  exit 1
fi
if ./gdpval run --executor stirrup --executor-timeout 10 --no-metadata >/dev/null 2>&1; then
  echo "Stirrup unexpectedly accepted an unsupported executor timeout" >&2
  exit 1
fi
if ./gdpval run --executor stirrup --reasoning-effort high --no-metadata >/dev/null 2>&1; then
  echo "Stirrup unexpectedly accepted a Codex reasoning effort" >&2
  exit 1
fi
if ./gdpval compare-runs --a /tmp/missing-a --b /tmp/missing-b --judge-reasoning-effort high >/dev/null 2>&1; then
  echo "compare-runs unexpectedly accepted a judge effort without a Codex judge" >&2
  exit 1
fi

pycache="$(mktemp -d)"
metadata_tmp="$(mktemp -d)"
trap 'rm -rf "$pycache" "$metadata_tmp"' EXIT
env -i PATH="$PATH" PYTHONPATH="$ROOT" GDPVAL_COMMAND=run GDPVAL_EXECUTOR=codex \
  OUT="$metadata_tmp/unset" python3 scripts/gdpval_run_metadata.py >/dev/null
env -i PATH="$PATH" PYTHONPATH="$ROOT" GDPVAL_COMMAND=run GDPVAL_EXECUTOR=codex \
  GDPVAL_REASONING_EFFORT=max OUT="$metadata_tmp/max" python3 scripts/gdpval_run_metadata.py >/dev/null
env -i PATH="$PATH" PYTHONPATH="$ROOT" GDPVAL_COMMAND=compare-runs GDPVAL_JUDGE_EXECUTOR=codex \
  OUT="$metadata_tmp/judge-unset" python3 scripts/gdpval_run_metadata.py >/dev/null
env -i PATH="$PATH" PYTHONPATH="$ROOT" GDPVAL_COMMAND=compare-runs GDPVAL_JUDGE_EXECUTOR=codex \
  GDPVAL_JUDGE_REASONING_EFFORT=max OUT="$metadata_tmp/judge-max" python3 scripts/gdpval_run_metadata.py >/dev/null
python3 - "$metadata_tmp" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
unset = json.loads((root / "unset" / "run-metadata.json").read_text(encoding="utf-8"))
maximum = json.loads((root / "max" / "run-metadata.json").read_text(encoding="utf-8"))
judge_unset = json.loads((root / "judge-unset" / "run-metadata.json").read_text(encoding="utf-8"))
judge_maximum = json.loads((root / "judge-max" / "run-metadata.json").read_text(encoding="utf-8"))
assert "reasoning_effort_requested" not in unset["configuration"]
assert maximum["configuration"]["reasoning_effort_requested"] == "max"
assert unset["configuration_sha256"] != maximum["configuration_sha256"]
assert unset["resume_fingerprint_sha256"] != maximum["resume_fingerprint_sha256"]
assert "judge_reasoning_effort_requested" not in judge_unset["configuration"]
assert judge_maximum["configuration"]["judge_reasoning_effort_requested"] == "max"
assert judge_unset["configuration_sha256"] != judge_maximum["configuration_sha256"]
PY
PYTHONPYCACHEPREFIX="$pycache" python3 -m py_compile \
  scripts/gdpval_run_metadata.py \
  eval_harness/__init__.py \
  eval_harness/cli.py \
  eval_harness/reasoning.py \
  eval_harness/layout.py \
  eval_harness/local_runner.py \
  eval_harness/local_judge_runner.py \
  eval_harness/runner.py \
  eval_harness/benchmarks/__init__.py \
  eval_harness/benchmarks/base.py \
  eval_harness/benchmarks/aime26.py \
  eval_harness/benchmarks/bigcodebench.py \
  eval_harness/benchmarks/gdpval.py \
  eval_harness/benchmarks/registry.py \
  eval_harness/executors/__init__.py \
  eval_harness/executors/base.py \
  eval_harness/executors/claude_code.py \
  eval_harness/executors/codex.py \
  eval_harness/executors/cursor.py \
  eval_harness/executors/registry.py \
  eval_harness/judges/__init__.py \
  eval_harness/judges/base.py \
  eval_harness/judges/codex.py \
  eval_harness/judges/claude_code.py \
  eval_harness/judges/pairwise.py
python3 -m unittest discover -s tests/harness -p 'test_*.py'
bash tests/harness/test_codex_executor.sh
bash tests/harness/test_claude_code_executor.sh
bash tests/harness/test_cursor_executor.sh
bash tests/harness/test_local_judge_executor.sh
bash tests/harness/test_experiment_conditions.sh
bash tests/harness/test_eval_cli.sh

printf 'gdpval harness self-test passed\n'
