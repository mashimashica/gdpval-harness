# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from gdpval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
)
from gdpval_harness.reasoning import ReasoningEffortOption, validate_reasoning_effort


_API_ENV_VARS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "CODEX_ACCESS_TOKEN",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _read_output_text(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def subscription_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for name in _API_ENV_VARS:
        env.pop(name, None)
    return env


class CodexExecutor(Executor):
    name = "codex"
    invocation_mode = "codex exec"
    tool_permission_mode = "workspace-write + approval_policy=never"

    def __init__(
        self,
        *,
        network_enabled: bool = False,
        command: str | None = None,
        reasoning_effort: ReasoningEffortOption = None,
    ) -> None:
        self.network_enabled = network_enabled
        self.command = command or os.getenv("GDPVAL_CODEX_COMMAND", "codex")
        self.reasoning_effort = validate_reasoning_effort(reasoning_effort)
        self._version: str | None = None

    def version(self) -> str | None:
        try:
            result = subprocess.run(
                [self.command, "--version"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
                env=subscription_environment(),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        text = (result.stdout or result.stderr).strip()
        self._version = text or None
        return self._version

    def preflight(self) -> PreflightResult:
        if shutil.which(self.command) is None and not os.path.isfile(self.command):
            return PreflightResult(executor=self.name, ok=False, details=(f"Codex command not found: {self.command}",))

        version = self.version()
        try:
            status = subprocess.run(
                [self.command, "login", "status"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
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
        if (
            "api key" in normalized
            or "api-key" in normalized
            or "apikey" in normalized
            or "access token" in normalized
        ):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="api",
                details=("Codex is using API/access-token authentication; ChatGPT subscription login is required",),
            )
        if "chatgpt" not in normalized:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=(
                    "Codex login method was not positively identified as ChatGPT; refusing subscription-mode execution",
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
        reasoning_effort = self.reasoning_effort
        final_message = request.executor_dir / "final-message.txt"
        network = "true" if self.network_enabled else "false"
        command = [
            self.command,
            "exec",
            "--cd",
            str(request.workspace),
            "--ephemeral",
            "--json",
            "--color",
            "never",
            "--output-last-message",
            str(final_message),
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "-c",
            'forced_login_method="chatgpt"',
            "-c",
            'approval_policy="never"',
            "-c",
            f"sandbox_workspace_write.network_access={network}",
            "-c",
            "shell_environment_policy.ignore_default_excludes=false",
        ]
        if not self.network_enabled:
            command.extend(["-c", 'web_search="disabled"'])
        if reasoning_effort is not None:
            command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if request.model:
            command.extend(["--model", request.model])
        command.append("-")
        return command

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        reasoning_effort = self.reasoning_effort
        request.workspace.mkdir(parents=True, exist_ok=True)
        request.deliverables_dir.mkdir(parents=True, exist_ok=True)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        command = self.build_command(request)
        stdout_path = request.executor_dir / "stdout.log"
        stderr_path = request.executor_dir / "stderr.log"
        final_message_path = request.executor_dir / "final-message.txt"
        prompt_path = request.executor_dir / "prompt.txt"
        prompt_path.write_text(request.task.prompt, encoding="utf-8")
        exit_code: int | None = None
        status = ExecutionStatus.FAILED
        stdout = ""
        stderr = ""

        try:
            completed = subprocess.run(
                command,
                input=request.task.prompt,
                capture_output=True,
                text=True,
                errors="replace",
                cwd=request.workspace,
                env=subscription_environment(request.environment),
                timeout=request.timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            if exit_code == 0:
                has_deliverable = request.deliverables_dir.is_dir() and any(
                    path.is_file() for path in request.deliverables_dir.rglob("*")
                )
                status = ExecutionStatus.COMPLETED if has_deliverable else ExecutionStatus.NO_DELIVERABLE
        except subprocess.TimeoutExpired as exc:
            stdout = _text(exc.stdout)
            stderr = _text(exc.stderr)
            status = ExecutionStatus.TIMED_OUT
        except KeyboardInterrupt:
            status = ExecutionStatus.INTERRUPTED
            raise
        except OSError as exc:
            stderr = str(exc) + "\n"
            status = ExecutionStatus.FAILED
        finally:
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")

        output_text = _read_output_text(final_message_path)
        metadata = {
            "sandbox": "workspace-write",
            "tool_permission_mode": self.tool_permission_mode,
            "network_policy": "enabled" if self.network_enabled else "disabled",
            "web_search": "default" if self.network_enabled else "disabled",
            "cloud_execution": False,
            "structured_output": "jsonl",
            "session_persistence": "ephemeral",
            "forced_login_method": "chatgpt",
            "output_text_source": "executor/final-message.txt",
            "api_environment_removed": sorted(_API_ENV_VARS),
            "command": command,
        }
        if reasoning_effort is not None:
            metadata["reasoning_effort_requested"] = reasoning_effort
        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version=self._version or self.version(),
            invocation_mode=self.invocation_mode,
            auth_mode="chatgpt-subscription",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=status,
            started_at=started_at,
            finished_at=_utc_now(),
            exit_code=exit_code,
            output_text=output_text,
            metadata=metadata,
            reasoning_effort_requested=reasoning_effort,
        )
