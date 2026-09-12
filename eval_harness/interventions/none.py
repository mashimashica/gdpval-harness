# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The identity intervention."""

from __future__ import annotations

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
    revision_fields,
)


class NoneIntervention(Intervention):
    """Preserve the task and workspace exactly as supplied."""

    name = "none"
    intervention_type = InterventionType.NONE

    def __init__(self, *, intervention_id: str | None = None) -> None:
        self.intervention_id = intervention_id or self.name
        self._bundle: InterventionBundle | None = None

    def preflight(self) -> InterventionPreflightResult:
        source_revision, revision_status = revision_fields(None, applicable=False)
        application = ApplicationMapping(method="none", target=None)
        manifest = InterventionManifest(
            intervention_id=self.intervention_id,
            intervention_type=self.intervention_type,
            source_revision=source_revision,
            revision_status=revision_status,
            files=(),
            bundle_sha256=compute_bundle_sha256(()),
            application=application,
        )
        self._bundle = InterventionBundle(root=None, manifest=manifest)
        return InterventionPreflightResult(
            name=self.name,
            intervention_type=self.intervention_type,
            ok=True,
            bundle=self._bundle,
            details=("identity intervention is ready",),
        )

    def validate_task(self, task: TaskSpec) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError("intervention task must be a TaskSpec")

    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        del workspace
        self.validate_task(task)
        bundle = self._require_bundle()
        return InterventionApplication(
            application_run_id=application_run_id,
            task=task,
            materialized_files=(),
            bundle_sha256=bundle.manifest.bundle_sha256,
            manifest_sha256=bundle.manifest.manifest_sha256 or "",
            application=bundle.manifest.application,
        )

    def _require_bundle(self) -> InterventionBundle:
        if self._bundle is None:
            raise RuntimeError("successful intervention preflight is required before apply")
        return self._bundle


__all__ = ["NoneIntervention"]
