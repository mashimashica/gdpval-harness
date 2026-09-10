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

cat >"$fake_bin/codex" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${OPENAI_API_KEY:-}" || -n "${CODEX_ACCESS_TOKEN:-}" ]]; then exit 97; fi
printf '%s\n' "$*" >>"${FAKE_JUDGE_ARGS_LOG:?}"
case "${1-}" in
  --version) echo "codex-cli judge-test"; exit 0 ;;
  login) echo "Logged in using ChatGPT"; exit 0 ;;
  exec)
    final=""
    shift
    while (( $# )); do
      case "$1" in
        --output-last-message) final="$2"; shift 2 ;;
        --cd|--sandbox|--color|--model|-c) shift 2 ;;
        --ephemeral|--json|--skip-git-repo-check|--ignore-user-config) shift ;;
        -) shift; break ;;
        *) shift ;;
      esac
    done
    cat >/dev/null
    printf '\377reason\nBOXED[A]\n' >"$final"
    printf '\377{"type":"turn.completed"}\n'
    exit 0
    ;;
esac
exit 98
EOF
chmod +x "$fake_bin/codex"

cat >"$fake_bin/claude" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${ANTHROPIC_AUTH_TOKEN:-}" || -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then exit 97; fi
printf '%s\n' "$*" >>"${FAKE_JUDGE_ARGS_LOG:?}"
case "${1-}" in
  --version) echo "claude judge-test"; exit 0 ;;
  auth) echo '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max"}'; exit 0 ;;
  -p) printf '\377reason\nBOXED[B]\n'; exit 0 ;;
esac
exit 98
EOF
chmod +x "$fake_bin/claude"

benchmark="$work/tasks.jsonl"
cat >"$benchmark" <<'EOF'
{"task_id":"one","prompt":"Create the best professional deliverable."}
EOF
for candidate in a b; do
  repeat="$work/$candidate/task_one/repeat_0"
  mkdir -p "$repeat/reference_files"
  printf 'same reference\n' >"$repeat/reference_files/ref.txt"
  printf '%s submission\n' "$candidate" >"$repeat/result.txt"
done

args="$work/args.log"
: >"$args"
PATH="$fake_bin:$PATH" \
OPENAI_API_KEY=must-not-leak \
CODEX_ACCESS_TOKEN=must-not-leak \
FAKE_JUDGE_ARGS_LOG="$args" \
GDPVAL_BENCHMARK_JSONL="$benchmark" \
./gdpval compare-runs \
  --a "$work/a" --b "$work/b" \
  --judge-executor codex --judge-trials 2 --limit 1 \
  --out "$work/codex-out" --no-metadata

test -f "$work/codex-out/local-judge-results.jsonl"
test -f "$work/codex-out/local-judge-summary.json"
grep -q '"official_gdpval_aa_v2": false' "$work/codex-out/local-judge-summary.json"
grep -q -- '--sandbox read-only' "$args"
python3 - "$work/codex-out/judge/tasks/task_one/trial_0/executor/stdout.log" <<'PY'
from pathlib import Path
import sys
assert "\ufffd" in Path(sys.argv[1]).read_text(encoding="utf-8")
PY
if grep -R -q 'must-not-leak' "$work/codex-out"; then
  echo "secret leaked into Codex judge output" >&2
  exit 1
fi

: >"$args"
PATH="$fake_bin:$PATH" \
ANTHROPIC_API_KEY=must-not-leak \
CLAUDE_CODE_OAUTH_TOKEN=must-not-leak \
FAKE_JUDGE_ARGS_LOG="$args" \
GDPVAL_BENCHMARK_JSONL="$benchmark" \
./gdpval compare-runs \
  --a "$work/a" --b "$work/b" \
  --judge-executor claude-code --judge-trials 1 --limit 1 \
  --out "$work/claude-out" --no-metadata

test -f "$work/claude-out/local-judge-summary.json"
grep -q -- '--tools Bash,Read' "$args"
grep -q 'denyWrite' "$args"
grep -q '"judge_executor": "claude-code"' "$work/claude-out/local-judge-summary.json"
python3 - "$work/claude-out/judge/tasks/task_one/trial_0/executor/stdout.log" <<'PY'
from pathlib import Path
import sys
assert "\ufffd" in Path(sys.argv[1]).read_text(encoding="utf-8")
PY
if grep -R -q 'must-not-leak' "$work/claude-out"; then
  echo "secret leaked into Claude judge output" >&2
  exit 1
fi

: >"$args"
if PATH="$fake_bin:$PATH" FAKE_JUDGE_ARGS_LOG="$args" GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval compare-runs --a "$work/a" --b "$work/b" --judge-executor codex --out "$work/no-limit" --no-metadata >/dev/null 2>&1; then
  echo "local judge without --limit unexpectedly succeeded" >&2
  exit 1
fi
if grep -q '^exec ' "$args"; then
  echo "local judge without --limit issued a model judgement" >&2
  exit 1
fi

: >"$args"
PATH="$fake_bin:$PATH" FAKE_JUDGE_ARGS_LOG="$args" \
  ./gdpval check --judge-executor codex --out "$work/check" >/dev/null
if grep -q '^exec ' "$args"; then
  echo "local judge preflight issued a model judgement" >&2
  exit 1
fi

printf 'local judge executor self-test passed\n'
