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
#
# GDPval (real-world work products, judged).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY,
# JUDGE_API_KEY and TAVILY_API_KEY). Also needs the Apptainer sandbox built by
# build-gdpval-sif.sh, pointed at by GDPVAL_CONTAINER_PATH (see
# instruct/README.md). Run from the Gym repo root — the benchmark's dataset and
# prepare script resolve relative to your working directory. Results land in
# ./results/gdpval.
#
#   nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh                         # full benchmark (220 tasks x 1)
#   LIMIT=3 nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh  # output dir, concurrency
#   GDPVAL_MODEL_TYPE=<type> .../gdpval.sh                                               # alternate Gym model type
#   GDPVAL_MODEL=<id> GDPVAL_BASE_URL=<url> .../gdpval.sh                               # policy target overrides
#
# Scores each deliverable against its rubric. Comparison mode instead scores against
# reference deliverables you generate yourself, one subdirectory per reference model;
# see instruct/README.md for the layout and how to produce them.
#
# Note: PARALLEL is applied twice on purpose — the agent caps its own concurrent runs
# at 32 regardless of --concurrency, so raising only one of them does nothing.

# JUDGE_ONLY re-scores existing deliverables without running the agent, so it
# needs neither the sandbox nor a search key.
if [ "${JUDGE_ONLY:-false}" = "true" ]; then
  TAVILY_API_KEY=""
else
  GDPVAL_CONTAINER_PATH="${GDPVAL_CONTAINER_PATH:?export GDPVAL_CONTAINER_PATH (gdpval.sif from build-gdpval-sif.sh)}"
  [ -r "$GDPVAL_CONTAINER_PATH" ] || { echo "gdpval.sif not readable at $GDPVAL_CONTAINER_PATH" >&2; exit 1; }
  TAVILY_API_KEY="${TAVILY_API_KEY:?export TAVILY_API_KEY (one key, or [k1,k2] for several)}"
fi

STIR=gdpval_stirrup_agent.responses_api_agents.stirrup_agent
GDR=gdpval_resources_server.resources_servers.gdpval
JUDGE=gdpval_judge_model.responses_api_models.openai_model
MODEL_TYPE="${GDPVAL_MODEL_TYPE:-vllm_model}"

MODEL_OVR=()
if [ -n "${GDPVAL_BASE_URL:-}" ]; then
  MODEL_OVR+=("++policy_base_url=$GDPVAL_BASE_URL")
fi
if [ -n "${GDPVAL_MODEL:-}" ]; then
  MODEL_OVR+=("++policy_model_name=$GDPVAL_MODEL")
fi
if [ -n "${GDPVAL_API_KEY:-}" ]; then
  MODEL_OVR+=('++policy_api_key=${oc.env:GDPVAL_API_KEY}')
fi

if [ "$MODEL_TYPE" = "vllm_model" ]; then
  POLICY=policy_model.responses_api_models.vllm_model
  MODEL_OVR+=("++$POLICY.chat_template_kwargs={enable_thinking: true}"
              "++$POLICY.extra_body={skip_special_tokens: false}"
              "++$POLICY.sequential_reasoning_allowed=false")
fi

DELIVERABLES="${PERSIST_DELIVERABLES_DIR:-$(realpath -m "${OUT:-./results/gdpval}/deliverables")}"
REFERENCE_MANIFEST="${GDPVAL_REFERENCE_MANIFEST:-config/gdpval-aa-v2-references.tsv}"

MODE="${GDPVAL_REWARD_MODE:-rubric}"
[ "$MODE" = rubric ] || [ "$MODE" = comparison ] ||
  { echo "GDPVAL_REWARD_MODE must be rubric or comparison (got '$MODE')" >&2; exit 1; }

JUDGE_OVR=()
case "${GDPVAL_JUDGE_PANEL:-aa-v2}" in
  aa-v2) ;;
  single)
    JUDGE_OVR+=("++$GDR.judge_panel=null")
    if [ -n "${GDPVAL_JUDGE_MODEL:-}" ]; then
      JUDGE_OVR+=("++$GDR.judge_responses_create_params_overrides.model=$GDPVAL_JUDGE_MODEL")
    fi
    ;;
  *)
    echo "GDPVAL_JUDGE_PANEL must be aa-v2 or single" >&2
    exit 2
    ;;
esac

MODE_OVR=()
if [ "$MODE" = comparison ] && [ -n "${GDPVAL_SINGLE_REFERENCE_DIR:-}" ]; then
  MODE_OVR=("++$GDR.reward_mode=comparison"
            "++$GDR.reference_deliverables_dir=$GDPVAL_SINGLE_REFERENCE_DIR"
            "++$GDR.reference_elo=${GDPVAL_SINGLE_REFERENCE_ELO:-1000}")
  echo "gdpval: comparison against one unrated experiment reference" >&2
elif [ "$MODE" = comparison ]; then
  GDPVAL_REFS="${GDPVAL_REFS:?export GDPVAL_REFS (dir of reference deliverables)}"
  [ -r "$REFERENCE_MANIFEST" ] || {
    echo "reference manifest not readable: $REFERENCE_MANIFEST" >&2
    exit 1
  }
  MODE_OVR=("++$GDR.reward_mode=comparison")
  found=0
  expected=()

  while IFS=$'\t ' read -r name elo _; do
    [ -z "${name:-}" ] && continue
    [[ "$name" == \#* ]] && continue
    if [ -z "${elo:-}" ]; then
      echo "invalid reference manifest row for '$name': missing Elo" >&2
      exit 1
    fi
    expected+=("$name")
    [ -d "$GDPVAL_REFS/$name" ] || continue
    MODE_OVR+=("++$GDR.reference_models.$name.deliverables_dir=$GDPVAL_REFS/$name"
               "++$GDR.reference_models.$name.elo=$elo")
    found=$((found + 1))
  done < "$REFERENCE_MANIFEST"

  if [ "$found" -eq 0 ]; then
    echo "no reference sets found in $GDPVAL_REFS. Expected subdirectories named:" >&2
    for name in "${expected[@]}"; do echo "  $name" >&2; done
    exit 1
  fi

  if [ "$found" -ge 2 ]; then
    MODE_OVR+=("++multistage.enabled=true"
               "++multistage.stages=[{num_tasks: 45}, {num_models: $((found < 4 ? found : 4))}]")
  fi

  echo "gdpval: comparison against $found rated reference(s)" >&2
fi

GYM_PIN="${GYM_PIN:-57c15a22f8b82d3d859b71468fe3329f4e2093b4}"
if [ "${PIN_GYM:-0}" != 0 ]; then
  git rev-parse --verify -q "$GYM_PIN^{commit}" >/dev/null 2>&1 || git fetch origin "$GYM_PIN"
  git restore --source="$GYM_PIN" -- . ':(exclude)nemotron_recipes' || exit 1
  echo "pinned Gym to $GYM_PIN (recipes untouched; set PIN_GYM=0 to skip; git restore . to undo)"
fi

gym eval prepare --benchmark gdpval

gym eval run \
  --benchmark gdpval \
  --model-type "$MODEL_TYPE" \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/gdpval}/evaluator_rollouts.jsonl" \
  "++$STIR.tavily_api_key=$TAVILY_API_KEY" \
  "++$STIR.persist_deliverables_dir=$DELIVERABLES" \
  "++$GDR.judge_sampling_seed=${JUDGE_SAMPLING_SEED:-42}" \
  "++$GDR.persist_raw_judge_responses=true" \
  "++$GDR.preconvert_max_concurrent=30" \
  "++$JUDGE.max_concurrent_requests=10" \
  ${PARALLEL:+"++$STIR.concurrency=$PARALLEL"} \
  ${MODE_OVR[@]+"${MODE_OVR[@]}"} \
  ${JUDGE_OVR[@]+"${JUDGE_OVR[@]}"} \
  "${MODEL_OVR[@]}" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
