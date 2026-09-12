# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
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
from eval_harness.executors.output_protocol import OutputProtocolError, parse_claude_output, parse_strict_json_object
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.reasoning import ReasoningEffortOption


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
_COMMAND_ENV = "EVAL_CLAUDE_COMMAND"
_SUPPORTED_VERSION = "2.1.259"
_VERSION_PATTERN = re.compile(r"(?<![A-Za-z0-9])2\.1\.259(?![A-Za-z0-9-])")


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


def _version_is_supported(version: str | None) -> bool:
    return version is not None and _VERSION_PATTERN.search(version) is not None


class ClaudeCodeExecutor(Executor):
    name = "claude-code"
    runtime = "host-subprocess"
    invocation_mode = "claude -p"
    tool_permission_mode = "acceptEdits + restricted built-in tools + fail-closed Bash sandbox"
    reasoning_effort: ReasoningEffortOption = None
    capabilities = ExecutorCapabilities(
        inputs=frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES}),
        outputs=frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
    )

    def __init__(
        self,
        *,
        network_enabled: bool = False,
        max_turns: int = 250,
        command: str | None = None,
    ) -> None:
        if type(network_enabled) is not bool:
            raise TypeError("network_enabled must be a bool")
        self.network_access_enabled = network_enabled
        self.max_turns = max_turns
        resolved_command = command or os.getenv(_COMMAND_ENV, "claude")
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
        if not _version_is_supported(version):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                details=(f"Claude Code version must be exactly {_SUPPORTED_VERSION}",),
            )
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
                details=("Claude Code is not logged in",),
            )
        try:
            payload = parse_strict_json_object(raw)
        except OutputProtocolError:
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=("claude auth status did not return JSON; refusing subscription-mode execution",),
            )

        logged_in_value = payload.get("loggedIn")
        auth_method_value = payload.get("authMethod")
        api_provider_value = payload.get("apiProvider")
        subscription_value = payload.get("subscriptionType")
        if (
            type(logged_in_value) is not bool
            or not isinstance(auth_method_value, str)
            or not isinstance(api_provider_value, str)
            or not isinstance(subscription_value, str)
        ):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=("claude auth status has an invalid shape; refusing subscription-mode execution",),
            )
        logged_in = logged_in_value
        auth_method = auth_method_value.lower()
        api_provider = api_provider_value.lower()
        subscription_type = subscription_value.lower()
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
        if not self.network_access_enabled:
            sandbox["network"] = {"strictAllowlist": True, "allowedDomains": []}
        return json.dumps({"sandbox": sandbox}, separators=(",", ":"))

    def build_command(self, request: ExecutionRequest) -> list[str]:
        command = [
            self.command,
            "-p",
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
        if not self.network_access_enabled:
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
        output_text: str | None = None
        protocol_error: str | None = None
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
            stdout = _text(completed.stdout)
            stderr = _text(completed.stderr)
            if exit_code == 0:
                try:
                    parsed = parse_claude_output(stdout)
                except OutputProtocolError as exc:
                    failure = Failure(FailureKind.PROTOCOL, "output_protocol", FailureImpact.RUN)
                    protocol_error = exc.code.value
                else:
                    output_text = parsed.output_text
                    has_deliverable = request.deliverables_dir.is_dir() and any(
                        path.is_file() for path in request.deliverables_dir.rglob("*")
                    )
                    status = (
                        ExecutionStatus.COMPLETED
                        if output_text is not None or has_deliverable
                        else ExecutionStatus.NO_DELIVERABLE
                    )
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

        metadata: dict[str, object] = {
            "safe_mode": True,
            "sandbox": "enabled-fail-closed",
            "tool_permission_mode": self.tool_permission_mode,
            "network_policy": "runtime-managed" if self.network_access_enabled else "strict-empty-allowlist",
            "cloud_execution": False,
            "max_turns": self.max_turns,
            "structured_output": "json",
            "session_persistence": "disabled",
            "api_environment_removed": sorted(_API_AND_CLOUD_ENV_VARS),
        }
        if protocol_error is not None:
            metadata["protocol_error"] = protocol_error
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
            available_outputs=(
                frozenset(
                    output
                    for output, present in (
                        (ExecutorOutput.FINAL_TEXT, output_text is not None),
                        (ExecutorOutput.ARTIFACT_FILES, True),
                    )
                    if present
                )
                if failure is None
                else frozenset()
            ),
            failure=failure,
            output_text=output_text,
            metadata=metadata,
            runtime="host-subprocess",
            model_id=request.model or None,
            effective_reasoning_effort=None,
            effective_reasoning_effort_available=False,
        )
