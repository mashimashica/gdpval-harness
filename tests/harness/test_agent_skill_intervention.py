# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import gdpval_harness.interventions.agent_skill as agent_skill_module
from gdpval_harness.executors.base import TaskSpec
from gdpval_harness.interventions import (
    AgentSkillIntervention,
    ApplicationMapping,
    InterventionBundle,
    InterventionType,
    canonical_manifest_bytes,
    create_intervention,
    ensure_source_output_separation,
    load_agent_skill_bundle,
)


class AgentSkillInterventionTests(unittest.TestCase):
    def _skill(self, root: Path, *, name: str = "demo-skill") -> Path:
        skill = root / name
        skill.mkdir(parents=True)
        (skill / "references").mkdir()
        (skill / "scripts").mkdir()
        (skill / "SKILL.md").write_text(
            "---\n"
            f"name: {name}\n"
            "description: A deterministic test skill\n"
            "license: Apache-2.0\n"
            "compatibility: local filesystem\n"
            "metadata:\n"
            '  version: "1"\n'
            "  owner: tests\n"
            "allowed-tools: Read Write\n"
            "---\n\n"
            "Follow the task-specific workflow.\n",
            encoding="utf-8",
        )
        (skill / "references" / "guide.md").write_bytes(b"reference bytes\x00\n")
        (skill / "scripts" / "run.sh").write_bytes(b"#!/bin/sh\nprintf test\n")
        return skill

    def test_loader_validates_frontmatter_and_includes_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            bundle = load_agent_skill_bundle(skill, intervention_id="custom-id")
            self.assertEqual(bundle.root, skill.resolve())
            self.assertEqual(bundle.manifest.intervention_type, InterventionType.AGENT_SKILL)
            self.assertEqual(bundle.manifest.intervention_id, "custom-id")
            self.assertEqual(bundle.manifest.revision_status, "unavailable")
            self.assertEqual(
                [item.path for item in bundle.manifest.files],
                ["SKILL.md", "references/guide.md", "scripts/run.sh"],
            )
            self.assertEqual(
                bundle.manifest.application,
                ApplicationMapping("workspace-reference", ".gdpval/interventions/demo-skill/SKILL.md"),
            )
            self.assertNotIn(str(skill), canonical_manifest_bytes(bundle.manifest).decode("utf-8"))

    def test_builder_seam_loads_bundle_then_constructs_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill = self._skill(Path(tmp))
            bundle = load_agent_skill_bundle(skill)
            intervention = AgentSkillIntervention(bundle)
            self.assertIsNone(intervention.source_reference)
            result = intervention.preflight()
            self.assertTrue(result.ok, result.details)
            from_source = AgentSkillIntervention.from_source(skill)
            self.assertEqual(from_source.source_reference, str(skill.resolve()))
            self.assertTrue(from_source.preflight().ok)
            for invalid_reference in (" ", "bad\x00reference", 3):
                with self.assertRaises(ValueError):
                    # Preserve the invalid runtime source reference at the constructor boundary.
                    AgentSkillIntervention(bundle, source_reference=cast(str, invalid_reference))

    def test_registry_uses_agent_skill_source_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill = self._skill(Path(tmp))
            self.assertIsInstance(create_intervention("agent-skill", source=skill), AgentSkillIntervention)
            with self.assertRaises(ValueError):
                create_intervention("agent-skill")

    def test_apply_materializes_all_files_and_derives_isolated_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            bundle = load_agent_skill_bundle(
                skill,
                intervention_id="SECRET-INTERVENTION-ID",
                source_revision="SECRET-REVISION",
            )
            intervention = AgentSkillIntervention(bundle, source_reference="SECRET-SOURCE-REFERENCE")
            self.assertTrue(intervention.preflight().ok)
            workspace = root / "workspace"
            workspace.mkdir()
            original = "Original task prompt\nwith exact spacing.  "
            application = intervention.apply(TaskSpec("task-1", original), workspace, application_run_id="run-1")

            expected_paths = [
                ".gdpval/interventions/demo-skill/SKILL.md",
                ".gdpval/interventions/demo-skill/references/guide.md",
                ".gdpval/interventions/demo-skill/scripts/run.sh",
            ]
            self.assertEqual([item.path for item in application.materialized_files], expected_paths)
            self.assertEqual(application.application_run_id, "run-1")
            self.assertEqual(application.application, bundle.manifest.application)
            self.assertEqual(application.task.task_id, "task-1")
            self.assertTrue(application.task.prompt.endswith(original))
            self.assertIn("`.gdpval/interventions/demo-skill/SKILL.md`", application.task.prompt)
            self.assertIn("`.gdpval/interventions/demo-skill/`", application.task.prompt)
            for sentinel in ("SECRET-SOURCE-REFERENCE", "SECRET-REVISION", "SECRET-INTERVENTION-ID", "rubric"):
                self.assertNotIn(sentinel, application.task.prompt)
            for item in application.materialized_files:
                destination = workspace / item.path
                self.assertTrue(destination.is_file())
                content = destination.read_bytes()
                self.assertEqual(item.size, len(content))
                self.assertEqual(item.sha256, hashlib.sha256(content).hexdigest())

    def test_apply_accepts_relative_workspace_after_chdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            intervention = AgentSkillIntervention.from_source(skill)
            self.assertTrue(intervention.preflight().ok)
            workspace = root / "workspace"
            workspace.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                application = intervention.apply(
                    TaskSpec("task-1", "prompt"),
                    Path("workspace"),
                    application_run_id="run-relative",
                )
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(application.application_run_id, "run-relative")
            self.assertTrue((workspace / ".gdpval/interventions/demo-skill/SKILL.md").is_file())

    def test_apply_requires_preflight_and_detects_source_tamper_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            intervention = AgentSkillIntervention.from_source(skill)
            workspace = root / "workspace"
            workspace.mkdir()
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertTrue(intervention.preflight().ok)
            (skill / "references" / "guide.md").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertEqual(list(workspace.iterdir()), [])

    def test_preflight_revalidates_manifest_type_target_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            bundle = load_agent_skill_bundle(skill)
            bad_file_manifest = replace(
                bundle.manifest,
                files=tuple(replace(item, sha256="0" * 64) for item in bundle.manifest.files),
                manifest_sha256=None,
            )
            bad_target_manifest = replace(
                bundle.manifest,
                application=ApplicationMapping("workspace-reference", ".gdpval/interventions/wrong/SKILL.md"),
                manifest_sha256=None,
            )
            bad_type_manifest = replace(
                bundle.manifest,
                intervention_type=InterventionType.FILES,
                manifest_sha256=None,
            )
            for name, manifest in (
                ("file hash", bad_file_manifest),
                ("target", bad_target_manifest),
                ("type", bad_type_manifest),
            ):
                with self.subTest(name=name):
                    result = AgentSkillIntervention(InterventionBundle(bundle.root, manifest)).preflight()
                    self.assertFalse(result.ok, result.details)

    def test_existing_destination_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            intervention = AgentSkillIntervention.from_source(skill)
            self.assertTrue(intervention.preflight().ok)
            workspace = root / "workspace"
            destination = workspace / ".gdpval" / "interventions" / "demo-skill"
            destination.mkdir(parents=True)
            (destination / "SKILL.md").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertEqual((destination / "SKILL.md").read_text(encoding="utf-8"), "existing")

    def test_copy_failure_cleans_only_files_and_directories_created_by_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            intervention = AgentSkillIntervention.from_source(skill)
            self.assertTrue(intervention.preflight().ok)
            workspace = root / "workspace"
            workspace.mkdir()
            real_copy = agent_skill_module._copy_exclusive
            calls = 0

            def fail_second(destination: Path, content: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("deterministic copy failure")
                real_copy(destination, content)

            with patch.object(agent_skill_module, "_copy_exclusive", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "deterministic copy failure"):
                    intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertEqual(list(workspace.iterdir()), [])

    def test_source_and_workspace_overlap_is_rejected_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            skill_inside = self._skill(workspace, name="inner-skill")
            first = AgentSkillIntervention.from_source(skill_inside)
            self.assertTrue(first.preflight().ok)
            with self.assertRaises(ValueError):
                first.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertFalse((workspace / ".gdpval").exists())

            skill = self._skill(root)
            source_workspace = skill / "workspace"
            source_workspace.mkdir()
            second = AgentSkillIntervention.from_source(skill)
            self.assertTrue(second.preflight().ok)
            with self.assertRaises(ValueError):
                second.apply(TaskSpec("task-1", "prompt"), source_workspace, application_run_id="run")
            self.assertFalse((source_workspace / ".gdpval").exists())

    def test_source_output_helper_handles_ancestors_and_symlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root)
            bundle = load_agent_skill_bundle(skill)
            with self.assertRaises(ValueError):
                ensure_source_output_separation(bundle, skill / "output")
            with self.assertRaises(ValueError):
                ensure_source_output_separation(bundle, skill)
            with self.assertRaises(ValueError):
                ensure_source_output_separation(bundle, root)
            alias = root / "alias"
            alias.symlink_to(skill, target_is_directory=True)
            with self.assertRaises(ValueError):
                ensure_source_output_separation(bundle, alias / "output")

    def test_invalid_skill_frontmatter_fails_closed(self) -> None:
        cases = {
            "missing opening": "name: demo-skill\ndescription: ok\n---\nbody\n",
            "missing closing": "---\nname: demo-skill\ndescription: ok\nbody\n",
            "missing name": "---\ndescription: ok\n---\nbody\n",
            "bad name": "---\nname: Demo Skill\ndescription: ok\n---\nbody\n",
            "missing description": "---\nname: demo-skill\n---\nbody\n",
            "blank license": "---\nname: demo-skill\ndescription: ok\nlicense: ' '\n---\nbody\n",
            "blank compatibility": "---\nname: demo-skill\ndescription: ok\ncompatibility: ' '\n---\nbody\n",
            "bad metadata": "---\nname: demo-skill\ndescription: ok\nmetadata: nope\n---\nbody\n",
            "bad allowed tools": "---\nname: demo-skill\ndescription: ok\nallowed-tools: 3\n---\nbody\n",
            "unsupported field": "---\nname: demo-skill\ndescription: ok\nunknown: value\n---\nbody\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, content in cases.items():
                with self.subTest(label=label):
                    skill = root / "demo-skill"
                    skill.mkdir(exist_ok=True)
                    (skill / "SKILL.md").write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "invalid Agent Skill"):
                        load_agent_skill_bundle(skill)
                    (skill / "SKILL.md").unlink()

    def test_frontmatter_bounds_and_parent_name_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            too_long_description = self._skill(root)
            (too_long_description / "SKILL.md").write_text(
                "---\nname: demo-skill\ndescription: " + "x" * 1025 + "\n---\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_agent_skill_bundle(too_long_description)

            too_long_compatibility = self._skill(root, name="compat-skill")
            (too_long_compatibility / "SKILL.md").write_text(
                "---\nname: compat-skill\ndescription: ok\ncompatibility: " + "x" * 501 + "\n---\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_agent_skill_bundle(too_long_compatibility)

            mismatch = self._skill(root, name="parent-name")
            (mismatch / "SKILL.md").write_text("---\nname: other-name\ndescription: ok\n---\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_agent_skill_bundle(mismatch)

    def test_invalid_encoding_symlink_special_path_and_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / "invalid-skill"
            invalid.mkdir()
            (invalid / "SKILL.md").write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                load_agent_skill_bundle(invalid)

            valid = self._skill(root, name="safe-skill")
            child = root / "outside.txt"
            child.write_text("outside", encoding="utf-8")
            (valid / "link.txt").symlink_to(child)
            with self.assertRaises(ValueError):
                load_agent_skill_bundle(valid)

            special = self._skill(root, name="special-skill")
            if os.name == "posix":
                os.mkfifo(special / "fifo", stat.S_IRUSR | stat.S_IWUSR)
                with self.assertRaises(ValueError):
                    load_agent_skill_bundle(special)

            backslash = self._skill(root, name="backslash-skill")
            (backslash / r"foo\bar").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_agent_skill_bundle(backslash)

            collision = self._skill(root, name="collision-skill")
            (collision / "A.txt").write_text("A", encoding="utf-8")
            (collision / "a.txt").write_text("a", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_agent_skill_bundle(collision)

            if os.name == "posix":
                with self.assertRaisesRegex(ValueError, "strict UTF-8"):
                    agent_skill_module._validate_relative_path("bad\udcff")

    def test_symlinked_skill_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = self._skill(root, name="real-skill")
            alias = root / "alias-skill"
            alias.symlink_to(skill, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "invalid Agent Skill"):
                load_agent_skill_bundle(alias)


if __name__ == "__main__":
    unittest.main()
