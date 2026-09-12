# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-overlay intervention with strict source validation and re-hashing."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

from eval_harness.executors.base import TaskSpec
from eval_harness.interventions.base import (
    ApplicationMapping,
    Intervention,
    InterventionApplication,
    InterventionBundle,
    InterventionManifest,
    InterventionPreflightResult,
    InterventionType,
    compute_bundle_sha256,
    file_evidence,
    revision_fields,
)


MAX_PROMPT_OVERLAY_BYTES = 1 * 1024 * 1024
OVERLAY_BEGIN = "[BEGIN INTERVENTION PROMPT OVERLAY]"
OVERLAY_END = "[END INTERVENTION PROMPT OVERLAY]"


def apply_prompt_overlay(task: TaskSpec, text: str) -> TaskSpec:
    """Prefix ``task.prompt`` with a clearly delimited reviewed instruction.

    The original prompt is appended byte-for-byte as the final body.  This
    helper deliberately accepts no condition label or provenance value.
    """

    if not isinstance(task, TaskSpec):
        raise TypeError("intervention task must be a TaskSpec")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt overlay text must be non-whitespace text")
    prefix = f"{OVERLAY_BEGIN}\n{text}\n{OVERLAY_END}\n\n"
    return TaskSpec(task_id=task.task_id, prompt=prefix + task.prompt)


class PromptOverlayIntervention(Intervention):
    """Apply one reviewed UTF-8 text file as a prompt prefix."""

    name = "prompt-overlay"
    intervention_type = InterventionType.PROMPT_OVERLAY

    def __init__(
        self,
        source: Path,
        *,
        intervention_id: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        self.source = Path(source)
        self.intervention_id = intervention_id or self.name
        self.source_revision = source_revision
        self._bundle: InterventionBundle | None = None

    def preflight(self) -> InterventionPreflightResult:
        self._bundle = None
        try:
            content = self._read_source()
            text = content.decode("utf-8")
            if not text.strip():
                raise ValueError("prompt overlay source must contain non-whitespace text")
            source_revision, revision_status = revision_fields(self.source_revision, applicable=True)
            application = ApplicationMapping(method="prompt-overlay", target="task.prompt")
            entry = file_evidence("prompt_overlay.txt", content)
            manifest = InterventionManifest(
                intervention_id=self.intervention_id,
                intervention_type=self.intervention_type,
                source_revision=source_revision,
                revision_status=revision_status,
                files=(entry,),
                bundle_sha256=compute_bundle_sha256(((entry.path, content),)),
                application=application,
            )
            self._bundle = InterventionBundle(root=self.source, manifest=manifest)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return InterventionPreflightResult(
                name=self.name,
                intervention_type=self.intervention_type,
                ok=False,
                details=(f"prompt overlay preflight failed: {exc}",),
            )
        return InterventionPreflightResult(
            name=self.name,
            intervention_type=self.intervention_type,
            ok=True,
            bundle=self._bundle,
            details=("prompt overlay source is ready",),
        )

    def validate_task(self, task: TaskSpec) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError("intervention task must be a TaskSpec")

    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        del workspace
        self.validate_task(task)
        bundle = self._require_bundle()
        content = self._read_source()
        if hashlib.sha256(content).hexdigest() != bundle.manifest.files[0].sha256:
            raise RuntimeError("prompt overlay source changed after preflight")
        text = content.decode("utf-8")
        applied_task = apply_prompt_overlay(task, text)
        return InterventionApplication(
            application_run_id=application_run_id,
            task=applied_task,
            materialized_files=(),
            bundle_sha256=bundle.manifest.bundle_sha256,
            manifest_sha256=bundle.manifest.manifest_sha256 or "",
            application=bundle.manifest.application,
        )

    def _read_source(self) -> bytes:
        if self.source.is_symlink():
            raise ValueError("prompt overlay source must not be a symlink")
        source_stat = self.source.stat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("prompt overlay source must be a regular file")
        if source_stat.st_size > MAX_PROMPT_OVERLAY_BYTES:
            raise ValueError("prompt overlay source exceeds the 1 MiB limit")
        content = self.source.read_bytes()
        if len(content) > MAX_PROMPT_OVERLAY_BYTES:
            raise ValueError("prompt overlay source exceeds the 1 MiB limit")
        return content

    def _require_bundle(self) -> InterventionBundle:
        if self._bundle is None:
            raise RuntimeError("successful intervention preflight is required before apply")
        return self._bundle


__all__ = [
    "MAX_PROMPT_OVERLAY_BYTES",
    "OVERLAY_BEGIN",
    "OVERLAY_END",
    "PromptOverlayIntervention",
    "apply_prompt_overlay",
]
