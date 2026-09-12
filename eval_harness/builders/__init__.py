# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder contracts for intervention input bundles."""

from eval_harness.builders.artifact import (
    ArtifactHandoffError,
    GeneratedSkillValidationError,
    seal_generated_skill,
)
from eval_harness.builders.base import (
    Builder,
    BuilderInputBundle,
    BuilderInputManifest,
    BuilderPreflightResult,
    BuildFailurePhase,
    BuildRequest,
    BuildResult,
    BuildStatus,
    canonical_builder_input_manifest_bytes,
)
from eval_harness.builders.executor_skill import ExecutorSkillBuilder
from eval_harness.builders.inputs import (
    StagedBuilderInput,
    load_builder_input_bundle,
    stage_builder_inputs,
    verify_staged_builder_inputs,
)
from eval_harness.builders.prompt import build_skill_task


__all__ = (
    "ArtifactHandoffError",
    "BuildFailurePhase",
    "BuildRequest",
    "BuildResult",
    "BuildStatus",
    "Builder",
    "BuilderInputBundle",
    "BuilderInputManifest",
    "BuilderPreflightResult",
    "ExecutorSkillBuilder",
    "GeneratedSkillValidationError",
    "StagedBuilderInput",
    "canonical_builder_input_manifest_bytes",
    "build_skill_task",
    "load_builder_input_bundle",
    "seal_generated_skill",
    "stage_builder_inputs",
    "verify_staged_builder_inputs",
)
