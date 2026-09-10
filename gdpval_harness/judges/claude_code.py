# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

from gdpval_harness.executors.claude_code import ClaudeCodeExecutor, subscription_environment
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult
from gdpval_harness.judges.pairwise import parse_verdict


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

    def preflight(self) -> JudgePreflightResult:
        result = self.policy.preflight()
        return JudgePreflightResult(
            judge_executor=self.name,
            ok=result.ok,
            version=result.version,
            auth_mode=result.auth_mode,
            details=result.details,
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
                    "denyWrite": [str(request.workspace.resolve())],
                },
                "network": {"strictAllowlist": True, "allowedDomains": []},
            }
        }
        return json.dumps(settings, separators=(",", ":"))

    def build_command(self, request: JudgeRequest) -> list[str]:
        command = [
            self.policy.command,
            "-p",
            "Read JUDGE_TASK.md in the current local workspace, inspect the anonymous submissions, and return the requested BOXED verdict.",
            "--output-format",
            "text",
            "--no-session-persistence",
            "--safe-mode",
            "--permission-mode",
            "acceptEdits",
            "--tools",
            "Bash,Read",
            "--strict-mcp-config",
            "--settings",
            self._sandbox_settings(request),
            "--max-turns",
            str(self.max_turns),
            "--disallowedTools",
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
            executor_version=self.policy.version(),
            invocation_mode=self.invocation_mode,
            auth_mode="claude-subscription",
            started_at=started_at,
            finished_at=_now(),
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={
                "safe_mode": True,
                "sandbox": "enabled-fail-closed/read-only-workspace",
                "tool_permission_mode": "Bash,Read only",
                "network_policy": "strict-empty-allowlist",
                "cloud_execution": False,
                "max_turns": self.max_turns,
                "parse_error": parse_error,
            },
        )
