# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
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


_API_ENV_VARS = {"CURSOR_API_KEY", "CURSOR_AUTH_TOKEN"}
_NEGATIVE_STATUS_MARKERS = ("not authenticated", "not logged in", "unauthenticated")
_API_STATUS_MARKERS = ("api key", "api-key", "apikey", "auth token", "bearer token")
_ACCOUNT_STATUS_MARKERS = ("authenticated", "logged in", "account")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def subscription_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for name in _API_ENV_VARS:
        env.pop(name, None)
    return env


class CursorExecutor(Executor):
    name = "cursor"
    invocation_mode = "agent -p"
    tool_permission_mode = "project allowlist + Cursor sandbox"

    def __init__(self, *, network_enabled: bool = False, command: str | None = None) -> None:
        self.network_enabled = network_enabled
        resolved_command = command or os.getenv("GDPVAL_CURSOR_COMMAND", "agent")
        if resolved_command is None:
            raise RuntimeError("Cursor Agent command could not be resolved")
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
                executor=self.name, ok=False, details=(f"Cursor Agent command not found: {self.command}",)
            )

        version = self.version()
        try:
            status = subprocess.run(
                [self.command, "status"],
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
                details=(f"could not inspect Cursor Agent login status: {exc}",),
            )

        raw = f"{status.stdout}\n{status.stderr}".strip()
        normalized = raw.lower()
        if status.returncode != 0 or any(marker in normalized for marker in _NEGATIVE_STATUS_MARKERS):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                details=(raw or "Cursor Agent is not authenticated",),
            )
        if any(marker in normalized for marker in _API_STATUS_MARKERS):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="api",
                details=("Cursor API/token authentication is not permitted for account-backed execution",),
            )
        if not raw or not any(marker in normalized for marker in _ACCOUNT_STATUS_MARKERS):
            return PreflightResult(
                executor=self.name,
                ok=False,
                version=version,
                auth_mode="unknown",
                details=("Cursor status did not positively identify an authenticated account; refusing execution",),
            )

        return PreflightResult(
            executor=self.name,
            ok=True,
            version=version,
            auth_mode="cursor-account",
            details=("Cursor account login detected with API/token environment variables removed",),
        )

    def _write_workspace_policy(self, workspace: Path, readonly_reference_path: Path | None = None) -> None:
        cursor_dir = workspace / ".cursor"
        cursor_dir.mkdir(parents=True, exist_ok=True)
        network_default = "allow" if self.network_enabled else "deny"
        sandbox = {
            "type": "workspace_readwrite",
            "additionalReadwritePaths": [],
            "additionalReadonlyPaths": [str(readonly_reference_path.resolve())] if readonly_reference_path else [],
            "disableTmpWrite": True,
            "enableSharedBuildCache": False,
            "networkPolicy": {"default": network_default, "allow": [], "deny": []},
        }
        (cursor_dir / "sandbox.json").write_text(
            json.dumps(sandbox, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        deny = [
            "Mcp(*:*)",
            "Write(reference_files/**)",
            "Read(.env*)",
            "Write(.env*)",
        ]
        if not self.network_enabled:
            deny.append("WebFetch(*)")
        cli_config = {
            "version": 1,
            "editor": {"vimMode": False},
            "permissions": {
                "allow": ["Shell(*)", "Read(**)", "Write(**)"],
                "deny": deny,
            },
        }
        (cursor_dir / "cli.json").write_text(
            json.dumps(cli_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _isolate_reference_files(self, workspace: Path) -> tuple[Path | None, str | None]:
        visible = workspace / "reference_files"
        if not visible.is_dir():
            return None, None

        protected = workspace.parent / "cursor-reference-files-readonly"
        if protected.is_symlink() or protected.is_file():
            protected.unlink()
        elif protected.is_dir():
            shutil.rmtree(protected)
        visible.rename(protected)
        digest = _tree_digest(protected)
        try:
            visible.symlink_to(protected.resolve(), target_is_directory=True)
        except OSError:
            protected.rename(visible)
            raise RuntimeError(
                "Cursor reference isolation requires directory symlink support; refusing to run with writable references"
            )
        return protected, digest

    @staticmethod
    def _restore_reference_files(workspace: Path, protected: Path | None) -> None:
        if protected is None:
            return
        visible = workspace / "reference_files"
        if visible.is_symlink() or visible.is_file():
            visible.unlink()
        elif visible.is_dir():
            shutil.rmtree(visible)
        if protected.is_dir():
            protected.rename(visible)

    def build_command(self, request: ExecutionRequest) -> list[str]:
        command = [
            self.command,
            "-p",
            "--trust",
            "--workspace",
            str(request.workspace),
            "--output-format",
            "json",
            "--sandbox",
            "enabled",
        ]
        if request.model:
            command.extend(["--model", request.model])
        command.append(
            "Read GDPVAL_TASK.md in this local workspace and complete it. "
            "Write only final submitted artifacts under ./deliverables/. "
            "Do not hand off to a Cloud Agent and do not prefix any message with '&'."
        )
        return command

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        request.workspace.mkdir(parents=True, exist_ok=True)
        request.deliverables_dir.mkdir(parents=True, exist_ok=True)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        protected_references, reference_digest = self._isolate_reference_files(request.workspace)
        self._write_workspace_policy(request.workspace, protected_references)
        task_path = request.workspace / "GDPVAL_TASK.md"
        task_path.write_text(request.task.prompt, encoding="utf-8")
        (request.executor_dir / "prompt.txt").write_text(request.task.prompt, encoding="utf-8")

        started_at = _utc_now()
        stdout_path = request.executor_dir / "stdout.log"
        stderr_path = request.executor_dir / "stderr.log"
        exit_code: int | None = None
        status = ExecutionStatus.FAILED
        stdout = ""
        stderr = ""
        reference_integrity_ok = True

        try:
            completed = subprocess.run(
                self.build_command(request),
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
            if protected_references is not None and reference_digest is not None:
                reference_integrity_ok = _tree_digest(protected_references) == reference_digest
                if not reference_integrity_ok:
                    stderr += "\nCursor executor detected reference-file mutation; failing closed.\n"
            if exit_code == 0 and reference_integrity_ok:
                has_deliverable = any(path.is_file() for path in request.deliverables_dir.rglob("*"))
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
            self._restore_reference_files(request.workspace, protected_references)
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")

        return ExecutionResult(
            task_id=request.task.task_id,
            executor=self.name,
            executor_version=self._version or self.version(),
            invocation_mode=self.invocation_mode,
            auth_mode="cursor-account",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=status,
            started_at=started_at,
            finished_at=_utc_now(),
            exit_code=exit_code,
            metadata={
                "sandbox": "enabled/workspace_readwrite",
                "tool_permission_mode": self.tool_permission_mode,
                "network_policy": "allow" if self.network_enabled else "deny",
                "cloud_execution": False,
                "structured_output": "json",
                "usage_mode": "cursor-account-usage",
                "reference_files_isolation": "outside-workspace + additionalReadonlyPaths",
                "reference_integrity_verified": reference_integrity_ok,
                "api_environment_removed": sorted(_API_ENV_VARS),
                "command": self.build_command(request),
            },
        )
