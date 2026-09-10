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
    printf 'codex-cli 9.9.9\377\n'
    exit 0
    ;;
  login)
    if [[ "${2-}" == status ]]; then
      if [[ "${FAKE_CODEX_AUTH:-chatgpt}" == api ]]; then
        echo "Logged in using API key"
      else
        printf '\377Logged in using ChatGPT\n'
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
        --model|-c|--sandbox|--color) shift 2 ;;
        --ephemeral|--json|--skip-git-repo-check|--ignore-user-config) shift ;;
        -) shift; break ;;
        *) shift ;;
      esac
    done
    cat >"${FAKE_CODEX_PROMPT_LOG:?}"
    mkdir -p "$workspace/deliverables/nested"
    printf 'fake deliverable\n' >"$workspace/deliverables/nested/result.txt"
    printf '\377{"type":"turn.completed"}\n'
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

test -f "$out/deliverables/task_task-one/repeat_0/nested/result.txt"
test -f "$out/deliverables/task_task-one/repeat_0/finish_params.json"
test -f "$out/tasks/task-one/executor/stdout.log"
test -f "$out/tasks/task-one/executor/stderr.log"
test -f "$out/tasks/task-one/executor/prompt.txt"
test -f "$out/tasks/task-one/executor/metadata.json"

grep -q '^exec ' "$args_log"
grep -q -- '--ephemeral' "$args_log"
grep -q -- '--sandbox workspace-write' "$args_log"
grep -q -- '--ignore-user-config' "$args_log"
grep -q 'forced_login_method="chatgpt"' "$args_log"
grep -q 'approval_policy="never"' "$args_log"
grep -q 'sandbox_workspace_write.network_access=false' "$args_log"
grep -q 'web_search="disabled"' "$args_log"
grep -q './deliverables/' "$prompt_log"
grep -q '"auth_mode": "chatgpt-subscription"' "$out/tasks/task-one/executor/metadata.json"
python3 - "$out/tasks/task-one/executor/stdout.log" <<'PY'
from pathlib import Path
import sys
assert "\ufffd" in Path(sys.argv[1]).read_text(encoding="utf-8")
PY
if grep -R -q 'must-not-leak' "$out"; then
  echo "secret leaked into run output" >&2
  exit 1
fi

condition_file="$work/intervention.md"
printf 'Use the externally supplied work-design method before producing the deliverable.\n' >"$condition_file"
: >"$args_log"
condition_out="$work/condition-run"
PATH="$fake_bin:$PATH" \
FAKE_CODEX_ARGS_LOG="$args_log" \
FAKE_CODEX_PROMPT_LOG="$prompt_log" \
GDPVAL_BENCHMARK_JSONL="$benchmark" \
./gdpval run \
  --executor codex \
  --condition intervention \
  --condition-file "$condition_file" \
  --limit 1 \
  --out "$condition_out"

grep -q 'Additional experiment-condition instructions' "$prompt_log"
grep -q 'externally supplied work-design method' "$prompt_log"
python3 - "$condition_out/run-metadata.json" "$condition_file" <<'PY'
import hashlib, json, pathlib, sys
metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
condition_path = pathlib.Path(sys.argv[2])
config = metadata["configuration"]
assert config["condition"] == "intervention"
assert config["condition_applied_to_prompt"] is True
assert config["condition_file"] == str(condition_path)
assert config["condition_file_sha256"] == hashlib.sha256(condition_path.read_bytes()).hexdigest()
assert "externally supplied work-design method" not in pathlib.Path(sys.argv[1]).read_text()
PY

: >"$args_log"
if PATH="$fake_bin:$PATH" \
  FAKE_CODEX_ARGS_LOG="$args_log" \
  FAKE_CODEX_PROMPT_LOG="$prompt_log" \
  GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval run --executor codex --out "$work/no-limit" --no-metadata >/dev/null 2>&1; then
  echo "Codex run without --limit unexpectedly succeeded" >&2
  exit 1
fi
if grep -q '^exec ' "$args_log"; then
  echo "run without --limit issued a model execution" >&2
  exit 1
fi

: >"$args_log"
if PATH="$fake_bin:$PATH" \
  FAKE_CODEX_ARGS_LOG="$args_log" \
  FAKE_CODEX_PROMPT_LOG="$prompt_log" \
  GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval run --executor codex --condition-file "$work/missing.md" --limit 1 --out "$work/missing-condition" --no-metadata >/dev/null 2>&1; then
  echo "missing condition file unexpectedly passed preflight" >&2
  exit 1
fi
if grep -q '^exec ' "$args_log"; then
  echo "invalid condition file issued a model execution" >&2
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
