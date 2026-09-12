#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root
readonly pre_commit_version="4.3.0"
readonly uv_version="0.11.29"
readonly uv_install_url="https://astral.sh/uv/${uv_version}/install.sh"
readonly uv_bin_dir="${repo_root}/.cache/nemo-gym-ci/uv-${uv_version}"

# shellcheck source=scripts/ci/sanitize_env.sh
source "${ci_dir}/sanitize_env.sh"
gym_ci_sanitize_environment lint
unset -f gym_ci_sanitize_environment

cd "${repo_root}"
if [[ ! -x "${uv_bin_dir}/uv" || ! -x "${uv_bin_dir}/uvx" ]]; then
    curl -LsSf "${uv_install_url}" | env UV_UNMANAGED_INSTALL="${uv_bin_dir}" sh
fi
export PATH="${uv_bin_dir}:${PATH}"
test "$(uv --version | awk '{print $2}')" = "${uv_version}"
test "$(uvx --version | awk '{print $2}')" = "${uv_version}"

# Keep the tool environment in the repository cache unless a CI provider supplies a shared one.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.cache/nemo-gym-ci/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"
exec uvx --from "pre-commit==${pre_commit_version}" pre-commit run --all-files --show-diff-on-failure --color=always
