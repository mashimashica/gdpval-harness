#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

errors=0

ok() { printf 'ok    %s\n' "$*" >&2; }
warn() { printf 'warn  %s\n' "$*" >&2; }
fail() {
  printf 'error %s\n' "$*" >&2
  errors=$((errors + 1))
}

need_command() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "command: $1"
  else
    fail "missing command: $1"
  fi
}

need_command gym
need_command realpath
need_command python3

if [[ "${PIN_GYM:-0}" != 0 ]]; then
  need_command git
fi

case "${GDPVAL_JUDGE_PANEL:-aa-v2}" in
  aa-v2|single) ;;
  *) fail "GDPVAL judge panel must be aa-v2 or single (got ${GDPVAL_JUDGE_PANEL})" ;;
esac
if [[ "${GDPVAL_JUDGE_PANEL:-aa-v2}" == aa-v2 && -n "${GDPVAL_JUDGE_MODEL:-}" ]]; then
  warn "--judge-model applies only to --judge-panel single; AA v2 panel model IDs come from JUDGE_GPT_MODEL/JUDGE_GEMINI_MODEL/JUDGE_CLAUDE_MODEL"
fi

if [[ -f env.yaml ]]; then
  ok "configuration: env.yaml"
else
  warn "env.yaml not found; policy and judge configuration must come from CLI/environment overrides"
  [[ -n "${GDPVAL_MODEL:-}" ]] || fail "GDPVAL model is unset; pass --model or provide env.yaml"
  [[ -n "${GDPVAL_API_KEY:-}" ]] || fail "GDPVAL_API_KEY is unset and env.yaml is absent"
  if [[ "${GDPVAL_MODEL_TYPE:-vllm_model}" != inference_provider/* ]]; then
    [[ -n "${GDPVAL_BASE_URL:-}" ]] || fail "policy base URL is unset; pass --base-url or provide env.yaml"
  fi
  [[ -n "${JUDGE_API_KEY:-}" ]] || fail "JUDGE_API_KEY is unset and env.yaml is absent"
fi

if [[ "${JUDGE_ONLY:-false}" != true ]]; then
  if [[ -z "${GDPVAL_CONTAINER_PATH:-}" ]]; then
    fail "GDPVAL_CONTAINER_PATH is unset"
  elif [[ ! -r "$GDPVAL_CONTAINER_PATH" ]]; then
    fail "GDPVAL_CONTAINER_PATH is not readable: $GDPVAL_CONTAINER_PATH"
  else
    ok "sandbox: $GDPVAL_CONTAINER_PATH"
  fi

  if [[ -n "${TAVILY_API_KEY:-}" ]]; then
    ok "search credential: TAVILY_API_KEY is set"
  else
    fail "TAVILY_API_KEY is unset"
  fi
else
  ok "judge-only mode: sandbox and search credential are not required"
fi

if [[ -n "${GDPVAL_REFERENCE_MANIFEST:-}" ]]; then
  if [[ -r "$GDPVAL_REFERENCE_MANIFEST" ]]; then
    ok "reference manifest: $GDPVAL_REFERENCE_MANIFEST"
  else
    fail "reference manifest is not readable: $GDPVAL_REFERENCE_MANIFEST"
  fi
fi

if [[ "${GDPVAL_PROFILE:-}" == aa-v2 && -z "${GDPVAL_REFERENCE_MANIFEST:-}" ]]; then
  fail "AA v2 profile requires a reference manifest"
fi

if [[ "${GDPVAL_REWARD_MODE:-rubric}" == comparison ]]; then
  if [[ -z "${GDPVAL_REFS:-}" ]]; then
    fail "comparison mode requires a reference directory"
  elif [[ ! -d "$GDPVAL_REFS" ]]; then
    fail "reference directory not found: $GDPVAL_REFS"
  else
    ok "references: $GDPVAL_REFS"
  fi
fi

if (( errors > 0 )); then
  printf 'preflight failed: %d problem(s)\n' "$errors" >&2
  exit 1
fi

printf 'preflight passed\n' >&2
