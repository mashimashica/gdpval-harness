# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit intervention factory."""

from __future__ import annotations

from pathlib import Path

from eval_harness.interventions.agent_skill import AgentSkillIntervention
from eval_harness.interventions.base import Intervention, InterventionType
from eval_harness.interventions.files import FilesIntervention
from eval_harness.interventions.none import NoneIntervention
from eval_harness.interventions.prompt_overlay import PromptOverlayIntervention


def create_intervention(
    intervention_type: InterventionType | str,
    *,
    source: Path | str | None = None,
    intervention_id: str | None = None,
    source_revision: str | None = None,
) -> Intervention:
    """Instantiate exactly the requested intervention and source shape."""

    try:
        kind = InterventionType(intervention_type)
    except ValueError as exc:
        available = ", ".join(item.value for item in InterventionType)
        raise ValueError(f"unknown intervention type {intervention_type!r}; available: {available}") from exc

    if kind is InterventionType.NONE:
        if source is not None:
            raise ValueError("none intervention does not accept a source")
        if source_revision is not None:
            raise ValueError("none intervention does not accept a source revision")
        return NoneIntervention(intervention_id=intervention_id)
    if source is None:
        raise ValueError(f"{kind.value} intervention requires an explicit source")
    if kind is InterventionType.PROMPT_OVERLAY:
        return PromptOverlayIntervention(
            Path(source), intervention_id=intervention_id, source_revision=source_revision
        )
    if kind is InterventionType.FILES:
        return FilesIntervention(Path(source), intervention_id=intervention_id, source_revision=source_revision)
    if kind is InterventionType.AGENT_SKILL:
        return AgentSkillIntervention.from_source(
            source,
            intervention_id=intervention_id,
            source_revision=source_revision,
        )
    raise AssertionError(f"unhandled intervention type: {kind}")


def get_intervention(
    intervention_type: InterventionType | str,
    *,
    source: Path | str | None = None,
    intervention_id: str | None = None,
    source_revision: str | None = None,
) -> Intervention:
    """Compatibility spelling for callers that use registry getters."""

    return create_intervention(
        intervention_type,
        source=source,
        intervention_id=intervention_id,
        source_revision=source_revision,
    )


__all__ = ["create_intervention", "get_intervention"]
