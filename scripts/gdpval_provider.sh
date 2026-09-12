#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Provider presets only fill values that the caller has not already set.
# Secrets stay in environment variables and are never appended to argv.
gdpval_list_providers() {
  cat <<'EOF'
openai      OpenAI Responses API via openai_model
gemini      Google Gemini OpenAI-compatible endpoint
openrouter  OpenRouter via inference_provider
litellm     LiteLLM proxy via litellm_model
vllm        Local/self-hosted vLLM
generic     Any OpenAI-compatible chat-completions endpoint
EOF
}

gdpval_apply_provider() {
  local provider="$1"
  case "$provider" in
    openai)
      export GDPVAL_MODEL_TYPE="${GDPVAL_MODEL_TYPE:-openai_model}"
      export GDPVAL_BASE_URL="${GDPVAL_BASE_URL:-https://api.openai.com/v1}"
      if [[ -z "${GDPVAL_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
        export GDPVAL_API_KEY="$OPENAI_API_KEY"
      fi
      ;;
    gemini)
      export GDPVAL_MODEL_TYPE="${GDPVAL_MODEL_TYPE:-inference_provider/gemini}"
      if [[ -z "${GDPVAL_API_KEY:-}" && -n "${GEMINI_API_KEY:-}" ]]; then
        export GDPVAL_API_KEY="$GEMINI_API_KEY"
      fi
      ;;
    openrouter)
      export GDPVAL_MODEL_TYPE="${GDPVAL_MODEL_TYPE:-inference_provider/openrouter}"
      if [[ -z "${GDPVAL_API_KEY:-}" && -n "${OPENROUTER_API_KEY:-}" ]]; then
        export GDPVAL_API_KEY="$OPENROUTER_API_KEY"
      fi
      ;;
    litellm)
      export GDPVAL_MODEL_TYPE="${GDPVAL_MODEL_TYPE:-litellm_model}"
      if [[ -z "${GDPVAL_BASE_URL:-}" && -n "${LITELLM_BASE_URL:-}" ]]; then
        export GDPVAL_BASE_URL="$LITELLM_BASE_URL"
      fi
      if [[ -z "${GDPVAL_API_KEY:-}" && -n "${LITELLM_API_KEY:-}" ]]; then
        export GDPVAL_API_KEY="$LITELLM_API_KEY"
      fi
      ;;
    vllm)
      export GDPVAL_MODEL_TYPE="${GDPVAL_MODEL_TYPE:-vllm_model}"
      export GDPVAL_BASE_URL="${GDPVAL_BASE_URL:-http://localhost:8000/v1}"
      export GDPVAL_API_KEY="${GDPVAL_API_KEY:-dummy}"
      ;;
    generic)
      export GDPVAL_MODEL_TYPE="${GDPVAL_MODEL_TYPE:-inference_provider}"
      ;;
    *)
      echo "unknown GDPval provider preset: $provider" >&2
      echo "available providers: openai, gemini, openrouter, litellm, vllm, generic" >&2
      return 2
      ;;
  esac
}
