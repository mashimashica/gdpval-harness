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
cat >"$fake_bin/claude" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for var in ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY; do
  if [[ -n "${!var:-}" ]]; then
    echo "forbidden Claude credential/routing variable leaked: $var" >&2
    exit 97
  fi
done
printf '%s\n' "$*" >>"${FAKE_CLAUDE_ARGS_LOG:?}"
case "${1-}" in
  --version)
    echo "2.1.999"
    exit 0
    ;;
  auth)
    if [[ "${2-}" == status ]]; then
      case "${FAKE_CLAUDE_AUTH:-subscription}" in
        api)
          echo '{"loggedIn":true,"authMethod":"apiKey","apiProvider":"firstParty","subscriptionType":null}'
          ;;
        console)
          echo '{"loggedIn":true,"authMethod":"console","apiProvider":"firstParty","subscriptionType":null}'
          ;;
        *)
          echo '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max"}'
          ;;
      esac
      exit 0
    fi
    ;;
  -p)
    cat >"${FAKE_CLAUDE_PROMPT_LOG:?}"
    mkdir -p "$PWD/deliverables/nested"
    printf 'fake claude deliverable\n' >"$PWD/deliverables/nested/result.txt"
    printf '{"type":"result","subtype":"success","is_error":false}\n\377'
    exit 0
    ;;
esac
exit 98
EOF
chmod +x "$fake_bin/claude"

benchmark="$work/tasks.jsonl"
cat >"$benchmark" <<'EOF'
{"task_id":"task-claude","prompt":"Create a professional work product.","reference_files":[],"reference_file_urls":[]}
EOF
args_log="$work/args.log"
prompt_log="$work/prompt.log"
: >"$args_log"
out="$work/run"

PATH="$fake_bin:$PATH" \
ANTHROPIC_API_KEY="must-not-leak" \
ANTHROPIC_AUTH_TOKEN="also-must-not-leak" \
ANTHROPIC_BASE_URL="https://api.example.invalid" \
CLAUDE_CODE_OAUTH_TOKEN="token-must-not-leak" \
CLAUDE_CODE_USE_BEDROCK=1 \
FAKE_CLAUDE_ARGS_LOG="$args_log" \
FAKE_CLAUDE_PROMPT_LOG="$prompt_log" \
GDPVAL_BENCHMARK_JSONL="$benchmark" \
GDPVAL_EXECUTOR_MAX_TURNS=7 \
./gdpval run --executor claude-code --limit 1 --out "$out" --no-metadata

test -f "$out/deliverables/task_task-claude/repeat_0/nested/result.txt"
test -f "$out/deliverables/task_task-claude/repeat_0/finish_params.json"
test -f "$out/tasks/task-claude/executor/stdout.log"
test -f "$out/tasks/task-claude/executor/stderr.log"
test -f "$out/tasks/task-claude/executor/prompt.txt"
test -f "$out/tasks/task-claude/executor/metadata.json"
grep -q '^-p ' "$args_log"
grep -q -- '--safe-mode' "$args_log"
grep -q -- '--permission-mode acceptEdits' "$args_log"
grep -q -- '--tools Bash,Read,Edit,Write' "$args_log"
grep -q -- '--max-turns 7' "$args_log"
grep -q 'failIfUnavailable' "$args_log"
grep -q 'strictAllowlist' "$args_log"
grep -q '"auth_mode": "claude-subscription"' "$out/tasks/task-claude/executor/metadata.json"
grep -q './deliverables/' "$prompt_log"
python3 - "$out/tasks/task-claude/executor/stdout.log" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(encoding="utf-8")
assert "\ufffd" in text, text
PY
if grep -R -q 'must-not-leak\|token-must-not-leak\|api.example.invalid' "$out"; then
  echo "secret or API routing value leaked into Claude run output" >&2
  exit 1
fi

: >"$args_log"
if PATH="$fake_bin:$PATH" \
  FAKE_CLAUDE_ARGS_LOG="$args_log" \
  FAKE_CLAUDE_PROMPT_LOG="$prompt_log" \
  GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval run --executor claude-code --out "$work/no-limit" --no-metadata >/dev/null 2>&1; then
  echo "Claude run without --limit unexpectedly succeeded" >&2
  exit 1
fi
if grep -q '^-p ' "$args_log"; then
  echo "Claude run without --limit issued a model execution" >&2
  exit 1
fi

: >"$args_log"
if PATH="$fake_bin:$PATH" \
  FAKE_CLAUDE_AUTH=api \
  FAKE_CLAUDE_ARGS_LOG="$args_log" \
  FAKE_CLAUDE_PROMPT_LOG="$prompt_log" \
  ./gdpval check --executor claude-code --out "$work/check" >/dev/null 2>&1; then
  echo "Claude API auth unexpectedly passed subscription preflight" >&2
  exit 1
fi
if grep -q '^-p ' "$args_log"; then
  echo "Claude preflight issued a model execution" >&2
  exit 1
fi

printf 'claude-code executor self-test passed\n'
