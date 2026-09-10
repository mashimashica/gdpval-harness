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
if [[ -n "${OPENAI_API_KEY:-}" || -n "${CODEX_ACCESS_TOKEN:-}" ]]; then
  echo "API credential leaked into Codex subprocess" >&2
  exit 97
fi
printf '%s\n' "$*" >>"${FAKE_CODEX_ARGS_LOG:?}"
case "${1-}" in
  --version)
    echo "codex-cli 9.9.9"
    exit 0
    ;;
  login)
    if [[ "${2-}" == status ]]; then
      if [[ "${FAKE_CODEX_AUTH:-chatgpt}" == api ]]; then
        echo "Logged in using API key"
      else
        echo "Logged in using ChatGPT"
      fi
      exit 0
    fi
    ;;
  exec)
    workspace=""
    final_message=""
    shift
    while (( $# )); do
      case "$1" in
        --cd) workspace="$2"; shift 2 ;;
        --output-last-message) final_message="$2"; shift 2 ;;
        --model|-c|--sandbox) shift 2 ;;
        --ephemeral|--json|--skip-git-repo-check|--ignore-user-config) shift ;;
        -) shift; break ;;
        *) shift ;;
      esac
    done
    cat >"${FAKE_CODEX_PROMPT_LOG:?}"
    mkdir -p "$workspace/deliverables"
    printf 'fake deliverable\n' >"$workspace/deliverables/result.txt"
    printf '{"type":"turn.completed"}\n'
    printf 'done\n' >"$final_message"
    exit 0
    ;;
esac
echo "unexpected fake codex invocation" >&2
exit 98
EOF
chmod +x "$fake_bin/codex"

benchmark="$work/tasks.jsonl"
cat >"$benchmark" <<'EOF'
{"task_id":"task-one","sector":"test","occupation":"tester","prompt":"Create the requested work product.","reference_files":[],"reference_file_urls":[],"rubric_json":[],"rubric_pretty":""}
EOF

args_log="$work/codex-args.log"
prompt_log="$work/prompt.log"
: >"$args_log"
out="$work/run"

PATH="$fake_bin:$PATH" \
OPENAI_API_KEY="must-not-leak" \
FAKE_CODEX_ARGS_LOG="$args_log" \
FAKE_CODEX_PROMPT_LOG="$prompt_log" \
GDPVAL_BENCHMARK_JSONL="$benchmark" \
./gdpval run --executor codex --limit 1 --out "$out" --no-metadata

test -f "$out/deliverables/task_task-one/repeat_0/result.txt"
test -f "$out/deliverables/task_task-one/repeat_0/finish_params.json"
test -f "$out/tasks/task-one/executor/stdout.log"
test -f "$out/tasks/task-one/executor/stderr.log"
test -f "$out/tasks/task-one/executor/metadata.json"

grep -q '^exec ' "$args_log"
grep -q -- '--ephemeral' "$args_log"
grep -q -- '--sandbox workspace-write' "$args_log"
grep -q -- '--ignore-user-config' "$args_log"
grep -q 'sandbox_workspace_write.network_access=false' "$args_log"
grep -q './deliverables/' "$prompt_log"
grep -q '"auth_mode": "chatgpt-subscription"' "$out/tasks/task-one/executor/metadata.json"
if grep -R -q 'must-not-leak' "$out"; then
  echo "secret leaked into run output" >&2
  exit 1
fi

: >"$args_log"
if PATH="$fake_bin:$PATH" \
  FAKE_CODEX_AUTH=api \
  FAKE_CODEX_ARGS_LOG="$args_log" \
  FAKE_CODEX_PROMPT_LOG="$prompt_log" \
  ./gdpval check --executor codex --out "$work/check" >/dev/null 2>&1; then
  echo "Codex API-key authentication unexpectedly passed preflight" >&2
  exit 1
fi
if grep -q '^exec ' "$args_log"; then
  echo "preflight issued a model execution" >&2
  exit 1
fi

printf 'codex executor self-test passed\n'
