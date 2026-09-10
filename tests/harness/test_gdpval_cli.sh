#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash -n gdpval
bash -n scripts/gdpval_provider.sh
bash -n scripts/gdpval_preflight.sh
bash -n nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh

help="$(./gdpval --help)"
grep -q './gdpval aa-v2 --refs DIR' <<<"$help"
grep -q './gdpval compare-runs --a DIR --b DIR' <<<"$help"
grep -q './gdpval executors' <<<"$help"
grep -q -- '--executor NAME' <<<"$help"
grep -q 'rejected by Stirrup' <<<"$help"

providers="$(./gdpval providers)"
grep -q '^openai ' <<<"$providers"
grep -q '^gemini ' <<<"$providers"
grep -q '^openrouter ' <<<"$providers"

executors="$(./gdpval executors)"
grep -q '^claude-code' <<<"$executors"
grep -q '^codex' <<<"$executors"
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
if ./gdpval run --executor stirrup --executor-timeout 10 --no-metadata >/dev/null 2>&1; then
  echo "Stirrup unexpectedly accepted an unsupported executor timeout" >&2
  exit 1
fi

pycache="$(mktemp -d)"
trap 'rm -rf "$pycache"' EXIT
PYTHONPYCACHEPREFIX="$pycache" python3 -m py_compile \
  scripts/gdpval_run_metadata.py \
  gdpval_harness/__init__.py \
  gdpval_harness/layout.py \
  gdpval_harness/local_runner.py \
  gdpval_harness/executors/__init__.py \
  gdpval_harness/executors/base.py \
  gdpval_harness/executors/claude_code.py \
  gdpval_harness/executors/codex.py \
  gdpval_harness/executors/registry.py
python3 -m unittest discover -s tests/harness -p 'test_*.py'
bash tests/harness/test_codex_executor.sh
bash tests/harness/test_claude_code_executor.sh

printf 'gdpval harness self-test passed\n'
