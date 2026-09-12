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
from unittest.mock import patch

import gdpval_harness.builders.artifact as artifact_module
from gdpval_harness.builders.artifact import ArtifactHandoffError, GeneratedSkillValidationError
from gdpval_harness.interventions import InterventionBundle, load_agent_skill_bundle


class GeneratedSkillArtifactTests(unittest.TestCase):
    def _skill(self, root: Path, *, name: str = "generated-skill") -> tuple[Path, Path]:
        deliverables = root / f"deliverables-{name}"
        deliverables.mkdir()
        skill = deliverables / name
        (skill / "references").mkdir(parents=True)
        (skill / "scripts").mkdir()
        (skill / "SKILL.md").write_bytes(
            (
                "---\n"
                f"name: {name}\n"
                "description: Generated skill for artifact tests\n"
                "license: Apache-2.0\n"
                "metadata:\n"
                '  version: "1"\n'
                "---\n\n"
                "Follow the generated workflow.\n"
            ).encode()
        )
        (skill / "references" / "guide.md").write_bytes(b"guide\x00bytes\n")
        (skill / "scripts" / "run.sh").write_bytes(b"#!/bin/sh\nprintf generated\n")
        return deliverables, skill

    def test_valid_nested_skill_seals_exact_manifest_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables, skill = self._skill(root)
            artifact_root = root / "artifacts"

            sealed = artifact_module.seal_generated_skill(deliverables, artifact_root)
            source = load_agent_skill_bundle(skill)
            destination = artifact_root / "generated-skill"
            assert sealed.root is not None

            self.assertEqual(sealed.root, destination.resolve())
            self.assertEqual(sealed.root, sealed.root.resolve(strict=True))
            self.assertEqual(sealed.root.parent, artifact_root.resolve())
            self.assertNotEqual(sealed.root, skill.resolve())
            self.assertEqual(sealed.manifest, source.manifest)
            self.assertEqual(sealed.manifest.intervention_id, "generated-skill")
            self.assertIsNone(sealed.manifest.source_revision)
            self.assertEqual(sealed.manifest.revision_status, "unavailable")
            self.assertEqual(
                sealed.manifest.application.method,
                "workspace-reference",
            )
            self.assertEqual(
                sealed.manifest.application.target,
                ".gdpval/interventions/generated-skill/SKILL.md",
            )

            expected_paths = {
                "SKILL.md",
                "references/guide.md",
                "scripts/run.sh",
            }
            self.assertEqual({item.path for item in sealed.manifest.files}, expected_paths)
            for item in sealed.manifest.files:
                content = (destination / item.path).read_bytes()
                self.assertEqual(item.size, len(content))
                self.assertEqual(item.sha256, hashlib.sha256(content).hexdigest())
            self.assertEqual(
                sealed.manifest.bundle_sha256,
                source.manifest.bundle_sha256,
            )
            self.assertEqual(
                sealed.manifest.manifest_sha256,
                source.manifest.manifest_sha256,
            )

    def test_extra_workspace_or_log_sibling_outside_deliverables_is_not_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables, _ = self._skill(root)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "transcript.txt").write_text("transcript", encoding="utf-8")
            (root / "builder.log").write_text("transcript", encoding="utf-8")
            artifact_root = root / "artifacts"

            sealed = artifact_module.seal_generated_skill(deliverables, artifact_root)
            assert sealed.root is not None
            copied = {path.relative_to(sealed.root).as_posix() for path in sealed.root.rglob("*") if path.is_file()}
            self.assertEqual(copied, {"SKILL.md", "references/guide.md", "scripts/run.sh"})
            self.assertFalse((sealed.root / "workspace").exists())
            self.assertFalse((sealed.root / "builder.log").exists())

    def test_deliverables_shape_must_have_one_real_directory(self) -> None:
        cases = ("zero", "multiple", "immediate-file")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                deliverables = root / "deliverables"
                deliverables.mkdir()
                if case == "multiple":
                    (deliverables / "one").mkdir()
                    (deliverables / "two").mkdir()
                elif case == "immediate-file":
                    (deliverables / "generated-skill").write_text("file", encoding="utf-8")
                artifact_root = root / "artifacts"

                with self.assertRaises(GeneratedSkillValidationError):
                    artifact_module.seal_generated_skill(deliverables, artifact_root)
                self.assertFalse(artifact_root.exists())

    def test_loader_rejects_name_frontmatter_symlink_special_and_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            deliverables = root / "bad-name"
            deliverables.mkdir()
            skill = deliverables / "Generated-Skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("---\nname: Generated-Skill\ndescription: bad\n---\n", encoding="utf-8")
            with self.assertRaises(GeneratedSkillValidationError):
                artifact_module.seal_generated_skill(deliverables, root / "bad-name-artifact")

            deliverables = root / "bad-frontmatter"
            deliverables.mkdir()
            skill = deliverables / "generated-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("name: generated-skill\n", encoding="utf-8")
            with self.assertRaises(GeneratedSkillValidationError):
                artifact_module.seal_generated_skill(deliverables, root / "bad-frontmatter-artifact")

            deliverables, skill = self._skill(root, name="symlink-skill")
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (skill / "link.txt").symlink_to(outside)
            with self.assertRaises(GeneratedSkillValidationError):
                artifact_module.seal_generated_skill(deliverables, root / "symlink-artifact")

            if os.name == "posix":
                deliverables, skill = self._skill(root, name="special-skill")
                os.mkfifo(skill / "fifo", stat.S_IRUSR | stat.S_IWUSR)
                with self.assertRaises(GeneratedSkillValidationError):
                    artifact_module.seal_generated_skill(deliverables, root / "special-artifact")

            deliverables, skill = self._skill(root, name="collision-skill")
            (skill / "A.txt").write_text("A", encoding="utf-8")
            (skill / "a.txt").write_text("a", encoding="utf-8")
            with self.assertRaises(GeneratedSkillValidationError):
                artifact_module.seal_generated_skill(deliverables, root / "collision-artifact")

    def test_existing_or_symlink_artifact_root_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables, _ = self._skill(root)

            existing = root / "existing-artifact"
            existing.mkdir()
            sentinel = existing / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                artifact_module.seal_generated_skill(deliverables, existing)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

            target = root / "symlink-target"
            target.mkdir()
            alias = root / "symlink-artifact"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                artifact_module.seal_generated_skill(deliverables, alias)
            self.assertTrue(alias.is_symlink())
            self.assertEqual(list(target.iterdir()), [])

    def test_source_and_artifact_overlap_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables, skill = self._skill(root)
            artifact_root = skill / "artifact"

            with self.assertRaises(GeneratedSkillValidationError):
                artifact_module.seal_generated_skill(deliverables, artifact_root)
            self.assertFalse(artifact_root.exists())
            self.assertEqual(
                sorted(path.relative_to(skill).as_posix() for path in skill.rglob("*")),
                ["SKILL.md", "references", "references/guide.md", "scripts", "scripts/run.sh"],
            )

    def test_second_copy_failure_removes_only_partial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables, _ = self._skill(root)
            artifact_root = root / "artifacts"
            real_copy = artifact_module._copy_exclusive
            calls = 0

            def fail_second(destination: Path, content: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("deterministic copy failure")
                real_copy(destination, content)

            with patch.object(artifact_module, "_copy_exclusive", side_effect=fail_second):
                with self.assertRaisesRegex(ArtifactHandoffError, "deterministic copy failure"):
                    artifact_module.seal_generated_skill(deliverables, artifact_root)
            self.assertEqual(calls, 2)
            self.assertFalse(artifact_root.exists())

    def test_keyboard_interrupt_cleans_partial_artifact_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables, skill = self._skill(root, name="interrupt-skill")
            artifact_root = root / "interrupt-artifact"
            original_skill = {
                path.relative_to(skill).as_posix(): path.read_bytes() for path in skill.rglob("*") if path.is_file()
            }
            real_copy = artifact_module._copy_exclusive
            calls = 0

            def interrupt_second(destination: Path, content: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                real_copy(destination, content)

            with patch.object(artifact_module, "_copy_exclusive", side_effect=interrupt_second):
                with self.assertRaises(KeyboardInterrupt):
                    artifact_module.seal_generated_skill(deliverables, artifact_root)
            self.assertEqual(calls, 2)
            self.assertFalse(artifact_root.exists())
            self.assertEqual(
                {path.relative_to(skill).as_posix(): path.read_bytes() for path in skill.rglob("*") if path.is_file()},
                original_skill,
            )

    def test_source_tamper_and_reload_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables, skill = self._skill(root, name="tamper-skill")
            artifact_root = root / "tamper-artifact"
            real_copy = artifact_module._copy_exclusive
            calls = 0

            def tamper_after_first(destination: Path, content: bytes) -> None:
                nonlocal calls
                calls += 1
                real_copy(destination, content)
                if calls == 1:
                    (skill / "references" / "guide.md").write_bytes(b"tampered")

            with patch.object(artifact_module, "_copy_exclusive", side_effect=tamper_after_first):
                with self.assertRaises(ArtifactHandoffError):
                    artifact_module.seal_generated_skill(deliverables, artifact_root)
            self.assertFalse(artifact_root.exists())

            deliverables, _ = self._skill(root, name="reload-skill")
            artifact_root = root / "reload-artifact"
            real_loader = load_agent_skill_bundle
            calls = 0

            def mismatching_loader(path: Path) -> InterventionBundle:
                nonlocal calls
                calls += 1
                bundle = real_loader(path)
                if calls == 2:
                    manifest = replace(bundle.manifest, intervention_id="wrong-id", manifest_sha256=None)
                    return InterventionBundle(bundle.root, manifest)
                return bundle

            with patch("gdpval_harness.builders.artifact.load_agent_skill_bundle", side_effect=mismatching_loader):
                with self.assertRaises(ArtifactHandoffError):
                    artifact_module.seal_generated_skill(deliverables, artifact_root)
            self.assertEqual(calls, 2)
            self.assertFalse(artifact_root.exists())


if __name__ == "__main__":
    unittest.main()
