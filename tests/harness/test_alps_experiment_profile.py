# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from gdpval_harness.experiments.profile import load_experiment_profile


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_PATH = _REPOSITORY_ROOT / "config/experiments/alps-skill-creation.json"
_ALPS_HEAD = "cf31ca93a1b5379e2ddbd430f9ea416192fb6797"
_ALPS_BUNDLE_SHA256 = "1feb183fb6e01f7b48469c3669968df0741d93ddde428cfaad6e05350c2f28c4"
_SKILL_CREATOR_FILES = (
    "SKILL.md",
    "references/openai_yaml.md",
    "scripts/generate_openai_yaml.py",
    "scripts/init_skill.py",
    "scripts/quick_validate.py",
)
_ALPS_FILES = (
    "examples/README.md",
    "examples/assess-service-change/SKILL.md",
    "examples/assess-service-change/assets/baseline.csv",
    "examples/assess-service-change/assets/candidate.csv",
    "examples/assess-service-change/references/pilot-context.md",
    "examples/assess-service-change/references/tool-use.md",
    "examples/assess-service-change/scripts/compare_measurements.py",
    "skills/design-agent-work-system/SKILL.md",
    "skills/design-agent-work-system/references/agent-work-system-design.md",
    "skills/design-agent-work-system/references/examples.md",
    "skills/design-process-description/SKILL.md",
    "skills/design-process-description/references/SKILL-template.md",
    "skills/design-process-description/references/examples.md",
    "skills/design-process-description/references/process-framework.md",
)


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _walk_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    return []


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, list):
        return [key for child in value for key in _walk_keys(child)]
    if isinstance(value, dict):
        return [key for key, child in value.items() for key in (key, *_walk_keys(child))]
    return []


class AlpsExperimentProfileTests(unittest.TestCase):
    def test_profile_matches_pinned_alps_creation_reference(self) -> None:
        loaded = load_experiment_profile(_PROFILE_PATH)
        profile = loaded.profile

        self.assertEqual(profile.schema_version, 1)
        self.assertEqual(profile.profile_id, "alps-skill-creation-v1")
        self.assertEqual(profile.benchmark, "gdpval")
        self.assertEqual(tuple(spec.input_id for spec in profile.inputs), ("skill-creator", "alps-work-design"))

        skill_creator, alps = profile.inputs
        self.assertEqual(skill_creator.input_type, "agent-skill")
        self.assertIsNone(skill_creator.source_revision)
        self.assertEqual(skill_creator.revision_status, "unavailable")
        self.assertEqual(skill_creator.allowed_files, _SKILL_CREATOR_FILES)
        self.assertIsNone(skill_creator.expected_bundle_sha256)

        self.assertEqual(alps.input_type, "work-design-reference")
        self.assertEqual(alps.source_revision, _ALPS_HEAD)
        self.assertEqual(alps.revision_status, "available")
        self.assertEqual(alps.allowed_files, _ALPS_FILES)
        self.assertEqual(alps.expected_bundle_sha256, _ALPS_BUNDLE_SHA256)

        self.assertEqual(
            tuple((arm.arm_id, tuple(arm.builder_inputs)) for arm in profile.arms),
            (
                ("skill-creator-only", ("skill-creator",)),
                ("skill-creator-plus-alps", ("skill-creator", "alps-work-design")),
            ),
        )

    def test_raw_profile_has_no_runtime_or_source_binding_overrides(self) -> None:
        raw = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(raw), {"schema_version", "profile_id", "benchmark", "inputs", "arms"})
        self.assertEqual(tuple(raw["inputs"][0]["allowed_files"]), _SKILL_CREATOR_FILES)
        self.assertNotIn("expected_bundle_sha256", raw["inputs"][0])
        self.assertEqual(
            tuple(raw["inputs"][1]["allowed_files"]),
            _ALPS_FILES,
        )
        self.assertEqual(tuple(raw["arms"][0]["builder_inputs"]), ("skill-creator",))
        self.assertEqual(
            tuple(raw["arms"][1]["builder_inputs"]),
            ("skill-creator", "alps-work-design"),
        )

        forbidden_keys = {
            "application_executor",
            "application_model",
            "builder_executor",
            "builder_model",
            "executor",
            "model",
            "out",
            "runtime",
            "runtime_root",
            "source_root",
        }
        self.assertTrue(forbidden_keys.isdisjoint(_walk_keys(raw)))
        self.assertTrue(all(not value.startswith(("/", "~")) for value in _walk_strings(raw)))
        self.assertNotIn(str(_REPOSITORY_ROOT), _PROFILE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
