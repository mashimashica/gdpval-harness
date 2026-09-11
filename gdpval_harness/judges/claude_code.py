# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Mapping

from gdpval_harness.executors.claude_code import ClaudeCodeExecutor, subscription_environment
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult
from gdpval_harness.judges.pairwise import parse_verdict


_SUBSCRIPTION_TYPES = {"pro", "max", "team", "enterprise"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


class ClaudeCodeJudgeExecutor(JudgeExecutor):
    name = "claude-code"
    invocation_mode = "claude -p (local judge)"

    def __init__(self, command: str | None = None, max_turns: int | None = None) -> None:
        self.max_turns = max_turns or int(os.getenv("GDPVAL_JUDGE_MAX_TURNS", "80"))
        self.policy = ClaudeCodeExecutor(command=command, network_enabled=False, max_turns=self.max_turns)
        self._version: str | None = None

    def _version_with_environment(self, environment: Mapping[str, str] | None = None) -> str | None:
        try:
            result = subprocess.run(
                [self.policy.command, "--version"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
                env=subscription_environment(environment),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        text = (result.stdout or result.stderr).strip()
        self._version = text or None
        return self._version

    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        sanitized = subscription_environment(environment)
        if shutil.which(self.policy.command, path=sanitized.get("PATH")) is None and not os.path.isfile(self.policy.command):
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                details=(f"Claude command not found: {self.policy.command}",),
            )

        if os.name == "nt":
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                details=(
                    "native Windows Claude Code cannot provide the verified read confinement required for blind judging",
                ),
            )

        version = self._version_with_environment(sanitized)
        try:
            status = subprocess.run(
                [self.policy.command, "auth", "status"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
                env=sanitized,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                details=(f"could not inspect Claude Code auth status: {exc}",),
            )

        raw = (status.stdout or status.stderr).strip()
        if status.returncode != 0:
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                details=(raw or "Claude Code is not logged in",),
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=("claude auth status did not return JSON; refusing subscription-mode judging",),
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
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                auth_mode=auth_method or "unknown",
                details=(
                    "Claude Code must use a stored first-party claude.ai Pro/Max/Team/Enterprise subscription login; "
                    "Console/API, OAuth-token, gateway, Bedrock, Vertex, and Foundry modes are rejected",
                ),
            )

        # Blind judging requires an enforceable read boundary, not only a settings
        # declaration. Claude Code currently exposes no documented non-model CLI
        # probe that can prove denyRead/allowRead enforcement before the first
        # subscription-backed model call. Fail closed rather than making a blind
        # comparison claim on an unverified runtime. The policy executor remains
        # available; only Claude-as-judge is disabled by this guard.
        return JudgePreflightResult(
            judge_executor=self.name,
            ok=False,
            version=version,
            auth_mode=f"claude-subscription:{subscription_type}",
            details=(
                f"Claude Code subscription login detected ({subscription_type})",
                "Claude Code blind judging is disabled because filesystem read confinement cannot be verified "
                "with a documented non-model preflight probe; use --judge-executor codex or human validation",
            ),
        )

    @staticmethod
    def _sandbox_settings(request: JudgeRequest) -> str:
        settings = {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "allowUnsandboxedCommands": False,
                "autoAllowBashIfSandboxed": True,
                "filesystem": {
                    "denyRead": ["/"],
                    "allowRead": [str(request.workspace.resolve())],
                    "denyWrite": ["/"],
                },
                "network": {"strictAllowlist": True, "allowedDomains": []},
            }
        }
        return json.dumps(settings, separators=(",", ":"))

    def build_command(self, request: JudgeRequest) -> list[str]:
        command = [
            self.policy.command,
            "-p",
            "Read JUDGE_TASK.md using sandboxed Bash in the current local workspace, inspect the anonymous submissions, and return the requested BOXED verdict.",
            "--output-format",
            "text",
            "--no-session-persistence",
            "--safe-mode",
            "--permission-mode",
            "acceptEdits",
            "--tools",
            "Bash",
            "--strict-mcp-config",
            "--settings",
            self._sandbox_settings(request),
            "--max-turns",
            str(self.max_turns),
            "--disallowedTools",
            "Read",
            "Edit",
            "Write",
            "WebFetch",
            "WebSearch",
            "mcp__*",
        ]
        if request.model:
            command.extend(["--model", request.model])
        return command

    def judge(self, request: JudgeRequest) -> JudgeResult:
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = request.workspace / "JUDGE_TASK.md"
        stdout_path = request.executor_dir / "stdout.log"
        stderr_path = request.executor_dir / "stderr.log"
        prompt_path.write_text(request.task_prompt, encoding="utf-8")
        started_at = _now()
        exit_code = None
        stdout = stderr = ""
        verdict = None
        parse_error = None
        env = subscription_environment(request.environment)
        env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] = "1"
        try:
            completed = subprocess.run(
                self.build_command(request),
                text=True,
                errors="replace",
                capture_output=True,
                cwd=request.workspace,
                env=env,
                timeout=request.timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            if exit_code == 0:
                try:
                    verdict = parse_verdict(stdout)
                except ValueError as exc:
                    parse_error = str(exc)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _text(exc.stdout), _text(exc.stderr)
            parse_error = "judge timed out"
        except KeyboardInterrupt:
            raise
        except OSError as exc:
            stderr = str(exc)
            parse_error = str(exc)
        finally:
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")

        return JudgeResult(
            task_id=request.task_id,
            trial_index=request.trial_index,
            judge_executor=self.name,
            verdict=verdict,
            executor_version=self._version or self._version_with_environment(env),
            invocation_mode=self.invocation_mode,
            auth_mode="claude-subscription",
            started_at=started_at,
            finished_at=_now(),
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={
                "safe_mode": True,
                "sandbox": "configured-but-not-preflight-verifiable",
                "read_confinement": "not accepted for blind judging without a non-model runtime probe",
                "tool_permission_mode": "Bash only; Read/Edit/Write disabled",
                "network_policy": "strict-empty-allowlist",
                "cloud_execution": False,
                "max_turns": self.max_turns,
                "parse_error": parse_error,
            },
        )
