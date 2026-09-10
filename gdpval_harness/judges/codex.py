# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from gdpval_harness.executors.codex import CodexExecutor, subscription_environment
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult
from gdpval_harness.judges.pairwise import parse_verdict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


class CodexJudgeExecutor(JudgeExecutor):
    name = "codex"
    invocation_mode = "codex exec (local judge)"

    def __init__(self, command: str | None = None) -> None:
        self.policy = CodexExecutor(command=command)

    def preflight(self) -> JudgePreflightResult:
        result = self.policy.preflight()
        return JudgePreflightResult(
            judge_executor=self.name,
            ok=result.ok,
            version=result.version,
            auth_mode=result.auth_mode,
            details=result.details,
        )

    def build_command(self, request: JudgeRequest) -> list[str]:
        command = [
            self.policy.command,
            "exec",
            "--cd",
            str(request.workspace),
            "--ephemeral",
            "--json",
            "--color",
            "never",
            "--output-last-message",
            str(request.executor_dir / "final-message.txt"),
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "-c",
            'forced_login_method="chatgpt"',
            "-c",
            'approval_policy="never"',
            "-c",
            "shell_environment_policy.ignore_default_excludes=false",
            "-c",
            'web_search="disabled"',
        ]
        if request.model:
            command.extend(["--model", request.model])
        command.append("-")
        return command

    def judge(self, request: JudgeRequest) -> JudgeResult:
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = request.executor_dir / "judge-prompt.txt"
        stdout_path = request.executor_dir / "stdout.log"
        stderr_path = request.executor_dir / "stderr.log"
        final_path = request.executor_dir / "final-message.txt"
        prompt_path.write_text(request.task_prompt, encoding="utf-8")
        started_at = _now()
        exit_code = None
        stdout = stderr = ""
        verdict = None
        parse_error = None
        try:
            completed = subprocess.run(
                self.build_command(request),
                input=request.task_prompt,
                text=True,
                errors="replace",
                capture_output=True,
                cwd=request.workspace,
                env=subscription_environment(request.environment),
                timeout=request.timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            if exit_code == 0:
                final_text = final_path.read_text(encoding="utf-8", errors="replace") if final_path.is_file() else stdout
                try:
                    verdict = parse_verdict(final_text)
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
            auth_mode="chatgpt-subscription",
            started_at=started_at,
            finished_at=_now(),
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={
                "sandbox": "read-only",
                "network_policy": "sandbox read-only + web_search disabled",
                "web_search": "disabled",
                "cloud_execution": False,
                "forced_login_method": "chatgpt",
                "parse_error": parse_error,
            },
        )
