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
_SYSTEM_RUNTIME_ROOTS = {Path("/"), Path("/bin"), Path("/sbin"), Path("/usr")}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _toml_string(value: str) -> str:
    """JSON string quoting is also valid TOML basic-string quoting."""
    return json.dumps(value)


def _resolved_command_read_paths(command: str, environment: Mapping[str, str]) -> tuple[str, ...]:
    located = shutil.which(command, path=environment.get("PATH"))
    if located is None and os.path.isfile(command):
        located = command
    if located is None:
        return ()

    direct = Path(located).absolute()
    try:
        resolved = direct.resolve(strict=True)
    except OSError:
        resolved = direct.resolve()

    candidates = [direct, resolved]
    if resolved.parent.name == "bin":
        runtime_root = resolved.parent.parent
        if runtime_root not in _SYSTEM_RUNTIME_ROOTS:
            candidates.append(runtime_root)

    paths: list[str] = []
    for path in candidates:
        value = str(path)
        if value not in paths:
            paths.append(value)
    return tuple(paths)


def _permission_profile(workspace: Path, runtime_read_paths: tuple[str, ...] = ()) -> str:
    entries = [
        '":root"="deny"',
        '":minimal"="read"',
        f"{_toml_string(str(workspace.resolve()))}=\"read\"",
    ]
    entries.extend(f"{_toml_string(path)}=\"read\"" for path in runtime_read_paths)
    return f'{{filesystem={{{",".join(entries)}}},network={{enabled=false}}}}'


def _profile_overrides(workspace: Path, runtime_read_paths: tuple[str, ...] = ()) -> list[str]:
    return [
        "-c",
        f'default_permissions="{_PERMISSION_PROFILE}"',
        "-c",
        f"permissions.{_PERMISSION_PROFILE}={_permission_profile(workspace, runtime_read_paths)}",
    ]


def _shell_environment_policy(request: JudgeRequest) -> str:
    runtime_tmp = str(request.environment.get("TMPDIR") or request.workspace.resolve())
    values = {
        "HOME": str(request.workspace.resolve()),
        "PATH": os.defpath,
        "TEMP": runtime_tmp,
        "TMP": runtime_tmp,
        "TMPDIR": runtime_tmp,
    }
    assignments = ",".join(f"{name}={_toml_string(value)}" for name, value in sorted(values.items()))
    return f'{{inherit="none",ignore_default_excludes=false,set={{{assignments}}}}}'


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

        runtime_read_paths = _resolved_command_read_paths(self.policy.command, environment)
        if not runtime_read_paths:
            return False, f"could not resolve Codex command for sandbox probe: {self.policy.command}"

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
                command.extend(_profile_overrides(workspace, runtime_read_paths))
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
        if not _resolved_command_read_paths(self.policy.command, sanitized):
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
        runtime_read_paths = _resolved_command_read_paths(self.policy.command, request.environment)
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
        command.extend(_profile_overrides(request.workspace, runtime_read_paths))
        command.extend(
            [
                "-c",
                "allow_login_shell=false",
                "-c",
                f"shell_environment_policy={_shell_environment_policy(request)}",
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
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        started_at = _now()
        exit_code = None
        verdict = None
        parse_error = None
        sanitized = subscription_environment(request.environment)
        try:
            with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
                completed = subprocess.run(
                    self.build_command(request),
                    input=request.task_prompt.encode("utf-8"),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    cwd=request.workspace,
                    env=sanitized,
                    timeout=request.timeout_seconds,
                    check=False,
                )
            exit_code = completed.returncode
            if exit_code == 0:
                stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
                final_text = final_path.read_text(encoding="utf-8", errors="replace") if final_path.is_file() else stdout
                try:
                    verdict = parse_verdict(final_text)
                except ValueError as exc:
                    parse_error = str(exc)
        except subprocess.TimeoutExpired:
            parse_error = "judge timed out"
        except KeyboardInterrupt:
            raise
        except OSError as exc:
            with stderr_path.open("ab") as stderr_handle:
                stderr_handle.write((str(exc) + "\n").encode("utf-8", errors="replace"))
            parse_error = str(exc)

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
                "shell_environment_policy": "anonymous-minimal-no-parent-inheritance",
                "web_search": "disabled",
                "forced_login_method": "chatgpt",
                "cloud_execution": False,
                "parse_error": parse_error,
            },
        )
