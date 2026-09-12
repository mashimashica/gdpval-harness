# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Mapping

from eval_harness.capabilities import ExecutorCapabilities, ExecutorInput, ExecutorOutput
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
)
from eval_harness.failures import Failure, FailureImpact, FailureKind


_API_AND_CLOUD_ENV_VARS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
}
_SUBSCRIPTION_TYPES = {"pro", "max", "team", "enterprise"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def subscription_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for name in _API_AND_CLOUD_ENV_VARS:
        env.pop(name, None)
    return env


class ClaudeCodeExecutor(Executor):
    name = "claude-code"
    invocation_mode = "claude -p"
    tool_permission_mode = "acceptEdits + restricted built-in tools + fail-closed Bash sandbox"
    capabilities = ExecutorCapabilities(
        inputs=frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES}),
        outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
    )

    def __init__(
        self,
        *,
        network_enabled: bool = False,
        max_turns: int = 250,
        command: str | None = None,
    ) -> None:
        self.network_enabled = network_enabled
        self.max_turns = max_turns
        resolved_command = command or os.getenv("GDPVAL_CLAUDE_COMMAND", "claude")
        if resolved_command is None:
            raise RuntimeError("Claude Code command could not be resolved")
        self.command = resolved_command
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
            return PreflightResult(
                executor=self.name, ok=False, details=(f"Claude command not found: {self.command}",)
            )

        version = self.version()
        try:
            status = subprocess.run(
                [self.command, "auth", "status"],
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
                details=(f"could not inspect Claude Code auth status: {exc}",),
            )

        raw = (status.stdout or status.stderr).strip()
        if status.returncode != 0:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                details=(raw or "Claude Code is not logged in",),
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=("claude auth status did not return JSON; refusing subscription-mode execution",),
            )

        logged_in = payload.get("loggedIn") is True
        auth_method = str(payload.get("authMethod") or "").lower()
        api_provider = str(payload.get("apiProvider") or "").lower()
        subscription_type = str(payload.get("subscriptionType") or "").lower()
        if (
            not logged_in
            or auth_method != "claude.ai"
            or api_provider != "firstparty"
            or subscription_type not in _SUBSCRIPTION_TYPES
        ):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode=auth_method or "unknown",
                details=(
                    "Claude Code must use a stored first-party claude.ai Pro/Max/Team/Enterprise subscription login; "
                    "Console/API, OAuth-token, gateway, Bedrock, Vertex, and Foundry modes are rejected",
                ),
            )
        return PreflightResult(
            executor=self.name,
            ok=True,
            version=version,
            auth_mode=f"claude-subscription:{subscription_type}",
            details=(f"Claude Code subscription login detected ({subscription_type})",),
        )

    def _sandbox_settings(self) -> str:
        sandbox: dict[str, object] = {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": True,
        }
        if not self.network_enabled:
            sandbox["network"] = {"strictAllowlist": True, "allowedDomains": []}
        return json.dumps({"sandbox": sandbox}, separators=(",", ":"))

    def build_command(self, request: ExecutionRequest) -> list[str]:
        command = [
            self.command,
            "-p",
            "Complete the GDPval task supplied on standard input. Follow its deliverables contract exactly.",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--safe-mode",
            "--permission-mode",
            "acceptEdits",
            "--tools",
            "Bash,Read,Edit,Write",
            "--strict-mcp-config",
            "--settings",
            self._sandbox_settings(),
            "--max-turns",
            str(self.max_turns),
        ]
        if not self.network_enabled:
            command.extend(["--disallowedTools", "WebFetch", "WebSearch", "mcp__*"])
        if request.model:
            command.extend(["--model", request.model])
        return command

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        request.workspace.mkdir(parents=True, exist_ok=True)
        request.deliverables_dir.mkdir(parents=True, exist_ok=True)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        stdout_path = request.executor_dir / "stdout.log"
        stderr_path = request.executor_dir / "stderr.log"
        prompt_path = request.executor_dir / "prompt.txt"
        prompt_path.write_text(request.task.prompt, encoding="utf-8")
        exit_code: int | None = None
        status = ExecutionStatus.FAILED
        stdout = ""
        stderr = ""
        has_deliverable = False
        failure: Failure | None = None

        try:
            completed = subprocess.run(
                self.build_command(request),
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
                has_deliverable = any(path.is_file() for path in request.deliverables_dir.rglob("*"))
                status = ExecutionStatus.COMPLETED if has_deliverable else ExecutionStatus.NO_DELIVERABLE
            else:
                failure = Failure(FailureKind.PROCESS, "process_exit", FailureImpact.RUN)
        except subprocess.TimeoutExpired as exc:
            stdout = _text(exc.stdout)
            stderr = _text(exc.stderr)
            status = ExecutionStatus.TIMED_OUT
            failure = Failure(FailureKind.TIMEOUT, "timeout", FailureImpact.RUN)
        except KeyboardInterrupt:
            status = ExecutionStatus.INTERRUPTED
            failure = Failure(FailureKind.INTERRUPTED, "interrupted", FailureImpact.RUN)
        except OSError as exc:
            stderr = str(exc) + "\n"
            status = ExecutionStatus.FAILED
            failure = Failure(FailureKind.PROCESS, "process_spawn", FailureImpact.RUN)
        finally:
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")

        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version=self._version or self.version(),
            invocation_mode=self.invocation_mode,
            auth_mode="claude-subscription",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=status,
            started_at=started_at,
            finished_at=_utc_now(),
            exit_code=exit_code,
            available_outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}) if failure is None else frozenset(),
            failure=failure,
            metadata={
                "safe_mode": True,
                "sandbox": "enabled-fail-closed",
                "tool_permission_mode": self.tool_permission_mode,
                "network_policy": "runtime-managed" if self.network_enabled else "strict-empty-allowlist",
                "cloud_execution": False,
                "max_turns": self.max_turns,
                "structured_output": "json",
                "session_persistence": "disabled",
                "api_environment_removed": sorted(_API_AND_CLOUD_ENV_VARS),
            },
        )
