# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from gdpval_harness.executors.base import TaskSpec
from gdpval_harness.interventions import (
    ApplicationMapping,
    FilesIntervention,
    InterventionFile,
    InterventionManifest,
    InterventionPreflightResult,
    InterventionType,
    NoneIntervention,
    PromptOverlayIntervention,
    apply_prompt_overlay,
    canonical_manifest_bytes,
    compute_bundle_sha256,
    create_intervention,
)


class InterventionContractTests(unittest.TestCase):
    def test_task_spec_remains_exactly_two_fields(self) -> None:
        self.assertEqual(set(TaskSpec.__dataclass_fields__), {"task_id", "prompt"})

    def test_records_are_frozen_and_hashes_are_canonical(self) -> None:
        content = b"hello"
        item = InterventionFile("nested/file.txt", len(content), hashlib.sha256(content).hexdigest())
        manifest = InterventionManifest(
            intervention_id="files",
            intervention_type=InterventionType.FILES,
            source_revision=None,
            revision_status="unavailable",
            files=(item,),
            bundle_sha256=compute_bundle_sha256(((item.path, content),)),
            application=ApplicationMapping("workspace-files", "."),
        )
        expected = hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
        self.assertEqual(manifest.manifest_sha256, expected)
        self.assertNotIn("manifest_sha256", canonical_manifest_bytes(manifest).decode())
        with self.assertRaises(FrozenInstanceError):
            setattr(item, "path", "changed")

    def test_successful_preflight_requires_a_bundle_and_manifest_paths_are_posix(self) -> None:
        with self.assertRaises(ValueError):
            InterventionPreflightResult(
                name="broken",
                intervention_type=InterventionType.NONE,
                ok=True,
            )
        content = b"x"
        item = InterventionFile("file.txt", 1, hashlib.sha256(content).hexdigest())
        with self.assertRaises(ValueError):
            InterventionManifest(
                intervention_id="files",
                intervention_type=InterventionType.FILES,
                source_revision=None,
                revision_status="unavailable",
                files=(InterventionFile("foo\\..\\bar", 1, item.sha256),),
                bundle_sha256=compute_bundle_sha256((("foo\\..\\bar", content),)),
                application=ApplicationMapping("workspace-files", "."),
            )

    def test_bundle_hash_depends_on_ordered_paths_and_exact_bytes(self) -> None:
        first = compute_bundle_sha256((("a.txt", b"a"), ("b.txt", b"b")))
        self.assertEqual(first, compute_bundle_sha256((("a.txt", b"a"), ("b.txt", b"b"))))
        self.assertNotEqual(first, compute_bundle_sha256((("b.txt", b"b"), ("a.txt", b"a"))))
        self.assertNotEqual(first, compute_bundle_sha256((("a.txt", b"A"), ("b.txt", b"b"))))

    def test_none_is_identity_and_marks_revision_not_applicable(self) -> None:
        task = TaskSpec("task-1", "keep this prompt exactly")
        intervention = NoneIntervention()
        result = intervention.preflight()
        self.assertTrue(result.ok)
        assert result.bundle is not None
        self.assertEqual(result.bundle.manifest.revision_status, "not-applicable")
        applied = intervention.apply(task, Path("/unused"), application_run_id="run-1")
        self.assertIs(applied.task, task)
        self.assertEqual(applied.materialized_files, ())
        self.assertEqual(applied.application, ApplicationMapping("none", None))

    def test_prompt_helper_preserves_original_prompt_as_final_body(self) -> None:
        task = TaskSpec("task-1", "  original\nexactly  ")
        applied = apply_prompt_overlay(task, "reviewed instructions")
        self.assertEqual(applied.task_id, task.task_id)
        self.assertTrue(applied.prompt.endswith(task.prompt))
        self.assertIn("[BEGIN INTERVENTION PROMPT OVERLAY]", applied.prompt)
        self.assertIn("[END INTERVENTION PROMPT OVERLAY]", applied.prompt)
        self.assertNotIn("condition", applied.prompt)
        with self.assertRaises(ValueError):
            apply_prompt_overlay(task, " \n\t")

    def test_prompt_overlay_rehashes_source_and_does_not_leak_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "overlay.txt"
            source.write_text("reviewed", encoding="utf-8")
            intervention = PromptOverlayIntervention(source)
            result = intervention.preflight()
            self.assertTrue(result.ok, result.details)
            assert result.bundle is not None
            manifest_bytes = canonical_manifest_bytes(result.bundle.manifest)
            self.assertNotIn(str(source), manifest_bytes.decode("utf-8"))

            source.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                intervention.apply(TaskSpec("task-1", "prompt"), root / "workspace", application_run_id="run")

    def test_prompt_overlay_source_rules_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whitespace = root / "whitespace.txt"
            whitespace.write_text(" \n", encoding="utf-8")
            self.assertFalse(PromptOverlayIntervention(whitespace).preflight().ok)

            invalid = root / "invalid.txt"
            invalid.write_bytes(b"\xff")
            self.assertFalse(PromptOverlayIntervention(invalid).preflight().ok)

            oversized = root / "oversized.txt"
            oversized.write_bytes(b"x" * (1024 * 1024 + 1))
            self.assertFalse(PromptOverlayIntervention(oversized).preflight().ok)

            target = root / "target.txt"
            target.write_text("valid", encoding="utf-8")
            link = root / "link.txt"
            link.symlink_to(target)
            self.assertFalse(PromptOverlayIntervention(link).preflight().ok)

    def test_registry_requires_exact_source_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "overlay.txt"
            source.write_text("text", encoding="utf-8")
            files_source = Path(tmp) / "files"
            files_source.mkdir()
            (files_source / "input.txt").write_text("input", encoding="utf-8")
            self.assertIsInstance(create_intervention("prompt-overlay", source=source), PromptOverlayIntervention)
            self.assertIsInstance(create_intervention("files", source=files_source), FilesIntervention)
            self.assertIsInstance(create_intervention("none"), NoneIntervention)
            with self.assertRaises(ValueError):
                create_intervention("prompt-overlay")
            with self.assertRaises(ValueError):
                create_intervention("none", source=source)
            with self.assertRaises(ValueError):
                create_intervention("none", source_revision="revision")
            with self.assertRaises(ValueError):
                create_intervention("unsupported")


if __name__ == "__main__":
    unittest.main()
