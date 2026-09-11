# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from gdpval_harness.executors.codex import CodexExecutor, subscription_environment
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult
from gdpval_harness.judges.pairwise import parse_verdict


_PERMISSION_PROFILE = "gdpval-harness-blind-judge"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _toml_string(value: str) -> str:
    """JSON string quoting is also valid TOML basic-string quoting."""
    return json.dumps(value)


def _permission_profile(workspace: Path) -> str:
    workspace_key = _toml_string(str(workspace.resolve()))
    return (
        '{filesystem={":root"="deny",":minimal"="read",'
        f'{workspace_key}="read"}},network={{enabled=false}}}}'
    )


def _profile_overrides(workspace: Path) -> list[str]:
    return [
        "-c",
        f'default_permissions="{_PERMISSION_PROFILE}"',
        "-c",
        f"permissions.{_PERMISSION_PROFILE}={_permission_profile(workspace)}",
    ]


class CodexJudgeExecutor(JudgeExecutor):
    name = "codex"
    invocation_mode = "codex exec (local judge)"

    def __init__(self, command: str | None = None) -> None:
        self.policy = CodexExecutor(command=command)
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

    def _probe_read_confinement(self, environment: Mapping[str, str]) -> tuple[bool, str]:
        """Verify Codex can enforce root-deny/workspace-read without a model call."""
        if os.name == "nt":
            return False, (
                "native Windows Codex sandbox cannot currently guarantee the root-deny read profile used for blind judging; "
                "use macOS, Linux, or WSL2"
            )

        try:
            with tempfile.TemporaryDirectory(prefix="gdpval-codex-read-probe-") as tmp:
                root = Path(tmp)
                workspace = root / "workspace"
                workspace.mkdir()
                allowed = workspace / "allowed.txt"
                denied = root / "outside-secret.txt"
                allowed.write_text("allowed\n", encoding="utf-8")
                denied.write_text("must-not-be-readable\n", encoding="utf-8")

                command = [self.policy.command, "-c", 'forced_login_method="chatgpt"', "-c", 'web_search="disabled"']
                command.extend(_profile_overrides(workspace))
                command.extend(
                    [
                        "sandbox",
                        "--permission-profile",
                        _PERMISSION_PROFILE,
                        "--cd",
                        str(workspace),
                        "--",
                        "sh",
                        "-c",
                        'cat "$1" >/dev/null && ! cat "$2" >/dev/null 2>&1',
                        "sh",
                        str(allowed),
                        str(denied),
                    ]
                )
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=20,
                    env=subscription_environment(environment),
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"could not verify Codex read confinement: {exc}"

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            return False, (
                "Codex root-deny/workspace-read sandbox probe failed; refusing blind judging"
                + (f": {detail}" if detail else "")
            )
        return True, "Codex root-deny/workspace-read sandbox probe passed"

    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        sanitized = subscription_environment(environment)
        if shutil.which(self.policy.command, path=sanitized.get("PATH")) is None and not os.path.isfile(self.policy.command):
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                details=(f"Codex command not found: {self.policy.command}",),
            )

        version = self._version_with_environment(sanitized)
        try:
            status = subprocess.run(
                [self.policy.command, "login", "status"],
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
                details=(f"could not inspect Codex login status: {exc}",),
            )

        auth_text = f"{status.stdout}\n{status.stderr}".strip()
        normalized = auth_text.lower()
        if status.returncode != 0:
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                details=(auth_text or "Codex is not logged in",),
            )
        if any(marker in normalized for marker in ("api key", "api-key", "apikey", "access token")):
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                auth_mode="api",
                details=("Codex is using API/access-token authentication; ChatGPT subscription login is required",),
            )
        if "chatgpt" not in normalized:
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=(
                    "Codex login method was not positively identified as ChatGPT; refusing subscription-mode judging",
                    auth_text,
                ),
            )

        confinement_ok, confinement_detail = self._probe_read_confinement(sanitized)
        if not confinement_ok:
            return JudgePreflightResult(
                judge_executor=self.name,
                ok=False,
                version=version,
                auth_mode="chatgpt-subscription",
                details=("Codex ChatGPT login detected", confinement_detail),
            )
        return JudgePreflightResult(
            judge_executor=self.name,
            ok=True,
            version=version,
            auth_mode="chatgpt-subscription",
            details=("Codex ChatGPT login detected", confinement_detail),
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
            "--skip-git-repo-check",
            "--ignore-user-config",
            "-c",
            'forced_login_method="chatgpt"',
            "-c",
            'web_search="disabled"',
            "-c",
            'approval_policy="never"',
        ]
        command.extend(_profile_overrides(request.workspace))
        command.extend(
            [
                "-c",
                "shell_environment_policy.ignore_default_excludes=false",
            ]
        )
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
        sanitized = subscription_environment(request.environment)
        try:
            completed = subprocess.run(
                self.build_command(request),
                input=request.task_prompt,
                text=True,
                errors="replace",
                capture_output=True,
                cwd=request.workspace,
                env=sanitized,
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
            executor_version=self._version or self._version_with_environment(sanitized),
            invocation_mode=self.invocation_mode,
            auth_mode="chatgpt-subscription",
            started_at=started_at,
            finished_at=_now(),
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={
                "sandbox": "permission-profile/root-deny/workspace-read-only",
                "read_confinement": "root-deny + minimal-read + anonymous-workspace-read",
                "network_policy": "disabled by permission profile",
                "web_search": "disabled",
                "forced_login_method": "chatgpt",
                "cloud_execution": False,
                "parse_error": parse_error,
            },
        )
