#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fake_bin="$work/bin"
mkdir -p "$fake_bin"
real_python="$(command -v python3)"
cat >"$fake_bin/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == -m && "${2-}" == eval_harness.local_runner ]]; then
  printf '%s\n' "${GDPVAL_EXECUTOR:-}" "${GDPVAL_CONDITION:-}" "${GDPVAL_CONDITION_FILE:-}" "${LIMIT:-}" >"${FAKE_CONDITION_ENV_LOG:?}"
  exit 0
fi
exec "${REAL_PYTHON:?}" "$@"
EOF
chmod +x "$fake_bin/python3"

condition="$work/intervention.md"
printf 'Apply an external reusable work-design process.\n' >"$condition"
env_log="$work/env.log"

PATH="$fake_bin:$PATH" \
REAL_PYTHON="$real_python" \
FAKE_CONDITION_ENV_LOG="$env_log" \
./gdpval run \
  --executor codex \
  --condition treatment \
  --condition-file "$condition" \
  --limit 1 \
  --no-metadata

mapfile -t values <"$env_log"
[[ "${values[0]}" == codex ]]
[[ "${values[1]}" == treatment ]]
[[ "${values[2]}" == "$condition" ]]
[[ "${values[3]}" == 1 ]]

if ./gdpval aa-v2 --refs "$work/refs" --condition treatment --no-metadata >/dev/null 2>&1; then
  echo "aa-v2 unexpectedly accepted an experiment condition" >&2
  exit 1
fi

mkdir -p "$work/a" "$work/b"
if ./gdpval compare-runs --a "$work/a" --b "$work/b" --condition treatment --no-metadata >/dev/null 2>&1; then
  echo "compare-runs unexpectedly accepted --condition" >&2
  exit 1
fi

if ./gdpval run --executor stirrup --condition-file "$condition" --limit 1 --no-metadata >/dev/null 2>&1; then
  echo "stirrup unexpectedly accepted --condition-file" >&2
  exit 1
fi

printf 'experiment condition CLI self-test passed\n'
