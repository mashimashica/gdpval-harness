#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fake_bin="$work/bin"
home="$work/home"
mkdir -p "$fake_bin" "$home"
args="$home/fake-judge-args.log"
: >"$args"

cat >"$fake_bin/codex" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${OPENAI_API_KEY:-}" || -n "${CODEX_ACCESS_TOKEN:-}" ]]; then exit 97; fi
original="$*"
printf '%s\n' "$original" >>"${HOME:?}/fake-judge-args.log"
while [[ "${1-}" == "-c" ]]; do shift 2; done
case "${1-}" in
  --version) echo "codex-cli judge-test"; exit 0 ;;
  login) echo "Logged in using ChatGPT"; exit 0 ;;
  sandbox)
    [[ "$original" == *'":root"="deny"'* ]] || exit 94
    [[ "$original" == *'":minimal"="read"'* ]] || exit 93
    [[ "$original" == *'network={enabled=false}'* ]] || exit 92
    exit 0
    ;;
  exec)
    for name in GDPVAL_RUN_A GDPVAL_RUN_B GDPVAL_LABEL_A GDPVAL_LABEL_B SECRET_SHOULD_NOT_REACH_JUDGE; do
      if [[ -n "${!name:-}" ]]; then
        echo "candidate identity or unrelated secret leaked into Codex judge: $name" >&2
        exit 96
      fi
    done
    [[ "$original" == *'":root"="deny"'* ]] || exit 91
    [[ "$original" == *'":minimal"="read"'* ]] || exit 90
    [[ "$original" != *'--sandbox read-only'* ]] || exit 89
    final=""
    shift
    while (( $# )); do
      case "$1" in
        --output-last-message) final="$2"; shift 2 ;;
        --cd|--color|--model|-c) shift 2 ;;
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
original="$*"
printf '%s\n' "$original" >>"${HOME:?}/fake-judge-args.log"
case "${1-}" in
  --version) echo "claude judge-test"; exit 0 ;;
  auth) echo '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max"}'; exit 0 ;;
  -p)
    echo "Claude model execution must not run while read confinement is unverified" >&2
    exit 88
    ;;
esac
exit 98
EOF
chmod +x "$fake_bin/claude"

benchmark="$work/tasks.jsonl"
cat >"$benchmark" <<'EOF'
{"task_id":"one","prompt":"Create the best professional deliverable."}
{"task_id":"two","prompt":"Create the second professional deliverable."}
EOF
for candidate in a b; do
  for task in one two; do
    repeat="$work/$candidate/task_$task/repeat_0"
    mkdir -p "$repeat/reference_files"
    printf 'same reference\n' >"$repeat/reference_files/ref.txt"
    printf '%s submission for %s\n' "$candidate" "$task" >"$repeat/result.txt"
    prompt_dir="$work/$candidate/tasks/$task/executor"
    mkdir -p "$prompt_dir"
    if [[ "$task" == one ]]; then
      task_prompt='Create the best professional deliverable.'
    else
      task_prompt='Create the second professional deliverable.'
    fi
    cat >"$prompt_dir/prompt.txt" <<EOF
You are completing a GDPval professional-work task in an isolated local workspace.

Task:
$task_prompt
EOF
  done
done
printf 'different reference\n' >"$work/b/task_two/repeat_0/reference_files/ref.txt"

: >"$args"
HOME="$home" \
PATH="$fake_bin:$PATH" \
OPENAI_API_KEY=must-not-leak \
CODEX_ACCESS_TOKEN=must-not-leak \
SECRET_SHOULD_NOT_REACH_JUDGE=parent-secret \
GDPVAL_BENCHMARK_JSONL="$benchmark" \
./gdpval compare-runs \
  --a "$work/a" --b "$work/b" \
  --label-a baseline-secret-label --label-b intervention-secret-label \
  --judge-executor codex --judge-trials 2 --limit 1 \
  --out "$work/codex-out"

test -f "$work/codex-out/local-judge-results.jsonl"
test -f "$work/codex-out/local-judge-summary.json"
test -f "$work/codex-out/run-metadata.json"
grep -q 'baseline-secret-label' "$work/codex-out/run-metadata.json"
grep -q '"official_gdpval_aa_v2": false' "$work/codex-out/local-judge-summary.json"
grep -q '"same_model_as_judge": null' "$work/codex-out/local-judge-summary.json"
grep -q '"judge_environment_policy": "minimal-runtime-auth-allowlist"' "$work/codex-out/local-judge-summary.json"
grep -q '"judge_workspace_isolation": "per-trial-system-temp-disjoint-from-candidates-and-output"' "$work/codex-out/local-judge-summary.json"
grep -q '"provenance_write_timing": "after-all-judge-model-calls"' "$work/codex-out/local-judge-summary.json"
grep -q '"local_judge_resume_supported": false' "$work/codex-out/local-judge-summary.json"
grep -q '"task_prompt_sha256"' "$work/codex-out/local-judge-results.jsonl"
grep -q 'sandbox --permission-profile gdpval-harness-blind-judge' "$args"
grep -q '":root"="deny"' "$args"
python3 - "$work/codex-out/judge/tasks/task_one/trial_0/executor/stdout.log" <<'PY'
from pathlib import Path
import sys
assert "\ufffd" in Path(sys.argv[1]).read_text(encoding="utf-8")
PY
if grep -R -q 'must-not-leak\|parent-secret' "$work/codex-out/judge"; then
  echo "secret leaked into Codex judge output" >&2
  exit 1
fi

# Claude Code remains a policy executor, but blind judging fails closed until a
# documented non-model read-confinement probe can verify the runtime boundary.
: >"$args"
if HOME="$home" \
  PATH="$fake_bin:$PATH" \
  ANTHROPIC_API_KEY=must-not-leak \
  CLAUDE_CODE_OAUTH_TOKEN=must-not-leak \
  SECRET_SHOULD_NOT_REACH_JUDGE=parent-secret \
  GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval compare-runs \
    --a "$work/a" --b "$work/b" \
    --label-a baseline-secret-label --label-b intervention-secret-label \
    --judge-executor claude-code --judge-trials 1 --limit 1 \
    --out "$work/claude-out" --no-metadata >/dev/null 2>&1; then
  echo "Claude blind judge unexpectedly passed unverified read confinement" >&2
  exit 1
fi
if grep -q '^-p ' "$args"; then
  echo "Claude blind judge issued a subscription-backed model call before verified confinement" >&2
  exit 1
fi
grep -q '^--version' "$args"
grep -q '^auth status' "$args"

occupied="$work/occupied"
mkdir -p "$occupied"
printf 'durable-sentinel\n' >"$occupied/local-judge-results.jsonl"
: >"$args"
if HOME="$home" PATH="$fake_bin:$PATH" GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval compare-runs --a "$work/a" --b "$work/b" --judge-executor codex --limit 1 \
  --out "$occupied" --no-metadata >/dev/null 2>&1; then
  echo "existing local judge output unexpectedly allowed overwrite" >&2
  exit 1
fi
grep -q '^durable-sentinel$' "$occupied/local-judge-results.jsonl"
if grep -q '^exec ' "$args"; then
  echo "existing local judge output was rejected after a model call" >&2
  exit 1
fi

cp "$work/b/tasks/one/executor/prompt.txt" "$work/original-prompt.txt"
cat >"$work/b/tasks/one/executor/prompt.txt" <<'EOF'
Wrapper

Task:
A different task prompt.
EOF
: >"$args"
if HOME="$home" PATH="$fake_bin:$PATH" GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval compare-runs --a "$work/a" --b "$work/b" --judge-executor codex --limit 1 \
  --out "$work/prompt-drift" --no-metadata >/dev/null 2>&1; then
  echo "prompt drift unexpectedly passed local judge validation" >&2
  exit 1
fi
if grep -q '^exec ' "$args"; then
  echo "prompt drift consumed a judge model call" >&2
  exit 1
fi
mv "$work/original-prompt.txt" "$work/b/tasks/one/executor/prompt.txt"

: >"$args"
if HOME="$home" PATH="$fake_bin:$PATH" GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval compare-runs --a "$work/a" --b "$work/b" --judge-executor codex --limit 2 \
  --out "$work/prevalidate" --no-metadata >/dev/null 2>&1; then
  echo "mismatched later pair unexpectedly passed local judge validation" >&2
  exit 1
fi
if grep -q '^exec ' "$args"; then
  echo "later-pair validation happened after a judge model call" >&2
  exit 1
fi

: >"$args"
if HOME="$home" PATH="$fake_bin:$PATH" GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval compare-runs --a "$work/a" --b "$work/b" --judge-executor codex --out "$work/no-limit" --no-metadata >/dev/null 2>&1; then
  echo "local judge without --limit unexpectedly succeeded" >&2
  exit 1
fi
if grep -q '^exec ' "$args"; then
  echo "local judge without --limit issued a model judgement" >&2
  exit 1
fi

: >"$args"
if HOME="$home" PATH="$fake_bin:$PATH" GDPVAL_BENCHMARK_JSONL="$benchmark" \
  ./gdpval compare-runs --a "$work/a" --b "$work/b" --judge-executor codex --limit 1 --resume \
  --out "$work/resume" --no-metadata >/dev/null 2>&1; then
  echo "local judge --resume unexpectedly succeeded" >&2
  exit 1
fi
if grep -q '^exec ' "$args"; then
  echo "local judge --resume consumed a model judgement" >&2
  exit 1
fi

: >"$args"
HOME="$home" PATH="$fake_bin:$PATH" \
  ./gdpval check --judge-executor codex --out "$work/check" >/dev/null
if grep -q '^exec ' "$args"; then
  echo "local judge preflight issued a model judgement" >&2
  exit 1
fi

printf 'local judge executor self-test passed\n'
