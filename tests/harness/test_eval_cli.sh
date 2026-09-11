#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash -n eval

help="$(./eval --help)"
grep -q 'benchmarks' <<<"$help"
grep -q 'executors' <<<"$help"
grep -q 'run' <<<"$help"
grep -q 'experiment' <<<"$help"

experiment_help="$(./eval experiment --help)"
grep -q -- '--input-root' <<<"$experiment_help"
grep -q -- '--order-seed' <<<"$experiment_help"
grep -q -- '--runtime-root' <<<"$experiment_help"

run_help="$(./eval run --help)"
grep -q -- '--intervention' <<<"$run_help"
grep -q -- '--intervention-source' <<<"$run_help"
grep -q 'agent-skill' <<<"$run_help"
grep -q 'workspace-reference' <<<"$run_help"

benchmarks="$(./eval benchmarks)"
grep -q '^aime26' <<<"$benchmarks"
grep -q '^bigcodebench' <<<"$benchmarks"
grep -q '^gdpval' <<<"$benchmarks"
grep -q 'benchmark-native' <<<"$benchmarks"
grep -q 'executable-tests' <<<"$benchmarks"
grep -q 'llm-rubric' <<<"$benchmarks"
grep -q 'math-verify==0.8.0' <<<"$benchmarks"
grep -q 'rubric/pairwise evaluation remains external' <<<"$benchmarks"

executors="$(./eval executors)"
grep -q '^codex' <<<"$executors"
grep -q '^claude-code' <<<"$executors"
grep -q '^cursor' <<<"$executors"
grep -q '^stirrup' <<<"$executors"
grep -q 'legacy ./gdpval path only' <<<"$executors"

if ./eval run aime26 --executor claude-code --limit 1 >/dev/null 2>&1; then
  echo "AIME26 unexpectedly accepted an unsupported executor" >&2
  exit 1
fi
if ./eval run bigcodebench --executor stirrup --limit 1 >/dev/null 2>&1; then
  echo "BigCodeBench unexpectedly accepted the legacy/API executor" >&2
  exit 1
fi
if ./eval run aime26 --executor codex --limit 0 >/dev/null 2>&1; then
  echo "generic eval unexpectedly accepted a non-positive limit" >&2
  exit 1
fi
if ./eval run aime26 --executor codex --limit 1 --intervention prompt-overlay >/dev/null 2>&1; then
  echo "generic eval unexpectedly accepted a missing intervention source" >&2
  exit 1
fi
if ./eval run aime26 --executor codex --limit 1 \
  --intervention none --intervention-source /tmp/unused >/dev/null 2>&1; then
  echo "generic eval unexpectedly accepted a source for the none intervention" >&2
  exit 1
fi
if ./eval run aime26 --executor codex --limit 1 \
  --intervention agent-skill >/dev/null 2>&1; then
  echo "generic eval unexpectedly accepted a missing agent-skill source" >&2
  exit 1
fi

printf 'generic eval CLI self-test passed\n'
