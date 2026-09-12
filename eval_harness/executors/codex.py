# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from eval_harness.capabilities import ExecutorCapabilities, ExecutorInput, ExecutorOutput
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
)
from eval_harness.executors.output_protocol import OutputProtocolError, parse_codex_output
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.reasoning import ReasoningEffortOption, validate_reasoning_effort


_API_ENV_VARS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "CODEX_ACCESS_TOKEN",
}
_COMMAND_ENV = "EVAL_CODEX_COMMAND"
_SUPPORTED_VERSION_MIN = (0, 154, 0)
_SUPPORTED_VERSION_MAX = (0, 155, 0)
_VERSION_PATTERN = re.compile(r"(?<![A-Za-z0-9])(\d+)\.(\d+)\.(\d+)(?![A-Za-z0-9-])")


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


def _read_output_bytes(path: Path) -> bytes | None:
    try:
        if not path.is_file():
            return None
        return path.read_bytes()
    except OSError:
        return None


def _version_is_supported(version: str | None) -> bool:
    if version is None:
        return False
    match = _VERSION_PATTERN.search(version)
    if match is None:
        return False
    parsed = tuple(int(part) for part in match.groups())
    return _SUPPORTED_VERSION_MIN <= parsed < _SUPPORTED_VERSION_MAX


def subscription_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for name in _API_ENV_VARS:
        env.pop(name, None)
    return env


class CodexExecutor(Executor):
    name = "codex"
    runtime = "host-subprocess"
    invocation_mode = "codex exec"
    tool_permission_mode = "workspace-write + approval_policy=never"
    reasoning_effort: ReasoningEffortOption
    capabilities = ExecutorCapabilities(
        inputs=frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES}),
        outputs=frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
    )

    def __init__(
        self,
        *,
        network_enabled: bool = False,
        command: str | None = None,
        reasoning_effort: ReasoningEffortOption = None,
    ) -> None:
        if type(network_enabled) is not bool:
            raise TypeError("network_enabled must be a bool")
        self.network_access_enabled = network_enabled
        resolved_command = command or os.getenv(_COMMAND_ENV, "codex")
        if resolved_command is None:
            raise RuntimeError("Codex command could not be resolved")
        self.command = resolved_command
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
        if not _version_is_supported(version):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                details=("Codex version is outside the supported 0.154.x range",),
            )
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
                details=("Codex is not logged in",),
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
        network = "true" if self.network_access_enabled else "false"
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
        if not self.network_access_enabled:
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
        final_message_cleanup_failed = False
        try:
            final_message_path.unlink(missing_ok=True)
        except OSError:
            final_message_cleanup_failed = True
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
            stdout = _text(completed.stdout)
            stderr = _text(completed.stderr)
            if exit_code == 0:
                final_message = None if final_message_cleanup_failed else _read_output_bytes(final_message_path)
                try:
                    parsed = parse_codex_output(stdout, final_message)
                except OutputProtocolError as exc:
                    failure = Failure(FailureKind.PROTOCOL, "output_protocol", FailureImpact.RUN)
                    protocol_error = exc.code.value
                else:
                    output_text = parsed.output_text
                    has_deliverable = request.deliverables_dir.is_dir() and any(
                        path.is_file() for path in request.deliverables_dir.rglob("*")
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

        if failure is None:
            status = (
                ExecutionStatus.COMPLETED
                if output_text is not None or has_deliverable
                else ExecutionStatus.NO_DELIVERABLE
            )
        available_outputs: frozenset[ExecutorOutput] = frozenset(
            output
            for output, present in (
                (ExecutorOutput.FINAL_TEXT, output_text is not None),
                (ExecutorOutput.ARTIFACT_FILES, failure is None),
            )
            if present
        )
        metadata: dict[str, object] = {
            "sandbox": "workspace-write",
            "tool_permission_mode": self.tool_permission_mode,
            "network_policy": "enabled" if self.network_access_enabled else "disabled",
            "web_search": "default" if self.network_access_enabled else "disabled",
            "cloud_execution": False,
            "structured_output": "jsonl",
            "session_persistence": "ephemeral",
            "forced_login_method": "chatgpt",
            "output_text_source": "executor/final-message.txt",
            "api_environment_removed": sorted(_API_ENV_VARS),
        }
        if protocol_error is not None:
            metadata["protocol_error"] = protocol_error
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
            available_outputs=available_outputs,
            failure=failure,
            output_text=output_text,
            metadata=metadata,
            reasoning_effort_requested=reasoning_effort,
            runtime="host-subprocess",
            model_id=request.model or None,
            effective_reasoning_effort=None,
            effective_reasoning_effort_available=False,
        )
