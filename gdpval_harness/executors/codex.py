# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Mapping

from gdpval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
)


_API_ENV_VARS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "CODEX_ACCESS_TOKEN",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def subscription_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    for name in _API_ENV_VARS:
        env.pop(name, None)
    return env


class CodexExecutor(Executor):
    name = "codex"
    invocation_mode = "codex exec"

    def __init__(self, *, network_enabled: bool = False) -> None:
        self.network_enabled = network_enabled

    @staticmethod
    def version() -> str | None:
        try:
            result = subprocess.run(
                ["codex", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        text = (result.stdout or result.stderr).strip()
        return text or None

    def preflight(self) -> PreflightResult:
        if shutil.which("codex") is None:
            return PreflightResult(executor=self.name, ok=False, details=("codex command not found",))

        version = self.version()
        try:
            status = subprocess.run(
                ["codex", "login", "status"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
                env=subscription_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                details=(f"could not inspect Codex login status: {exc}",),
            )

        auth_text = f"{status.stdout}\n{status.stderr}".strip()
        normalized = auth_text.lower()
        if status.returncode != 0:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                details=(auth_text or "Codex is not logged in",),
            )
        if "api key" in normalized or "api-key" in normalized or "apikey" in normalized:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="api-key",
                details=("Codex is authenticated with an API key; ChatGPT subscription login is required",),
            )
        if "chatgpt" not in normalized:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=(
                    "Codex login method was not positively identified as ChatGPT; "
                    "refusing subscription-mode execution",
                    auth_text,
                ),
            )
        return PreflightResult(
            executor=self.name,
            ok=True,
            version=version,
            auth_mode="chatgpt-subscription",
            details=("Codex ChatGPT login detected",),
        )

    def build_command(self, request: ExecutionRequest) -> list[str]:
        final_message = request.executor_dir / "final-message.txt"
        network = "true" if self.network_enabled else "false"
        command = [
            "codex",
            "exec",
            "--cd",
            str(request.workspace),
            "--ephemeral",
            "--json",
            "--output-last-message",
            str(final_message),
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "-c",
            f"sandbox_workspace_write.network_access={network}",
        ]
        if request.model:
            command.extend(["--model", request.model])
        command.append("-")
        return command

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        command = self.build_command(request)
        stdout_path = request.executor_dir / "stdout.log"
        stderr_path = request.executor_dir / "stderr.log"
        exit_code: int | None = None
        status = ExecutionStatus.FAILED

        try:
            completed = subprocess.run(
                command,
                input=request.task.prompt,
                capture_output=True,
                text=True,
                cwd=request.workspace,
                env=subscription_environment(request.environment),
                timeout=request.timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            if exit_code == 0:
                status = (
                    ExecutionStatus.COMPLETED
                    if any(path.is_file() for path in request.deliverables_dir.iterdir())
                    else ExecutionStatus.NO_DELIVERABLE
                )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
            stderr_path.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else "", encoding="utf-8")
            status = ExecutionStatus.TIMED_OUT
        except KeyboardInterrupt:
            status = ExecutionStatus.INTERRUPTED
            raise
        except OSError as exc:
            stderr_path.write_text(str(exc) + "\n", encoding="utf-8")
            status = ExecutionStatus.FAILED

        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version=self.version(),
            invocation_mode=self.invocation_mode,
            auth_mode="chatgpt-subscription",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=status,
            started_at=started_at,
            finished_at=_utc_now(),
            exit_code=exit_code,
            metadata={
                "sandbox": "workspace-write",
                "network_policy": "enabled" if self.network_enabled else "disabled",
                "cloud_execution": False,
                "api_environment_removed": sorted(_API_ENV_VARS),
            },
        )
