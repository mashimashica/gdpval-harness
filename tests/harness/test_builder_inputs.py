# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import gdpval_harness.builders.inputs as inputs_module
from gdpval_harness.builders.base import BuilderInputBundle, canonical_builder_input_manifest_bytes
from gdpval_harness.builders.inputs import (
    StagedBuilderInput,
    load_builder_input_bundle,
    stage_builder_inputs,
    verify_staged_builder_inputs,
)
from gdpval_harness.interventions.base import InterventionFile, compute_bundle_sha256


class BuilderInputTests(unittest.TestCase):
    def _source(self, root: Path, *, name: str = "source") -> Path:
        source = root / name
        (source / "nested").mkdir(parents=True)
        (source / "b.txt").write_bytes(b"bravo")
        (source / "a.txt").write_bytes(b"alpha")
        (source / "nested" / "c.bin").write_bytes(b"\x00charlie\xff")
        return source

    def _bundle(
        self, source: Path, *, input_id: str = "condition-secret", revision: str | None = None
    ) -> BuilderInputBundle:
        return load_builder_input_bundle(
            source,
            input_id=input_id,
            input_type="creation-files",
            allowed_files=("nested/c.bin", "b.txt", "a.txt"),
            source_revision=revision,
        )

    def _staged(self, root: Path) -> tuple[Path, Path, tuple[StagedBuilderInput, ...]]:
        source = self._source(root)
        bundle = self._bundle(source)
        workspace = root / "workspace"
        workspace.mkdir()
        return source, workspace, stage_builder_inputs((bundle,), workspace)

    def test_load_sorts_manifest_and_hashes_only_allowlisted_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            (source / "secret-token.txt").write_bytes(b"must never enter")
            bundle = self._bundle(source)

            self.assertEqual(tuple(item.path for item in bundle.manifest.files), ("a.txt", "b.txt", "nested/c.bin"))
            entries = (
                ("a.txt", b"alpha"),
                ("b.txt", b"bravo"),
                ("nested/c.bin", b"\x00charlie\xff"),
            )
            self.assertEqual(bundle.manifest.bundle_sha256, compute_bundle_sha256(entries))
            self.assertEqual(
                bundle.manifest.manifest_sha256,
                hashlib.sha256(canonical_builder_input_manifest_bytes(bundle.manifest)).hexdigest(),
            )
            self.assertNotIn(str(source), canonical_builder_input_manifest_bytes(bundle.manifest).decode())

            unavailable = bundle.manifest
            self.assertIsNone(unavailable.source_revision)
            self.assertEqual(unavailable.revision_status, "unavailable")
            available = self._bundle(source, revision="revision-7").manifest
            self.assertEqual(available.source_revision, "revision-7")
            self.assertEqual(available.revision_status, "available")

    def test_allowlist_is_required_safe_and_collision_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source(Path(temporary))
            unsafe = ("", "/absolute", "../escape", "./a.txt", "nested/../a.txt", r"nested\a.txt")
            for path in unsafe:
                with self.subTest(path=path), self.assertRaises((TypeError, ValueError)):
                    load_builder_input_bundle(
                        source,
                        input_id="id",
                        input_type="type",
                        allowed_files=(path,),
                    )
            for paths in (("a.txt", "a.txt"), ("a.txt", "A.txt"), ("é.txt", "e\u0301.txt")):
                with self.subTest(paths=paths), self.assertRaises(ValueError):
                    load_builder_input_bundle(
                        source,
                        input_id="id",
                        input_type="type",
                        allowed_files=paths,
                    )
            with self.assertRaises(ValueError):
                load_builder_input_bundle(source, input_id="id", input_type="type", allowed_files=())
            with self.assertRaises(TypeError):
                load_builder_input_bundle(
                    source,
                    input_id="id",
                    input_type="type",
                    # Preserve the invalid runtime string at the allowlist boundary.
                    allowed_files=cast(tuple[str, ...], "a.txt"),
                )

    def test_path_component_collisions_are_rejected_for_sources_and_forged_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "collision-source"
            (source / "A").mkdir(parents=True)
            (source / "a").mkdir()
            (source / "A" / "one.txt").write_bytes(b"one")
            (source / "a" / "two.txt").write_bytes(b"two")
            with self.assertRaises(ValueError):
                load_builder_input_bundle(
                    source,
                    input_id="id",
                    input_type="type",
                    allowed_files=("A/one.txt", "a/two.txt"),
                )

            source = root / "valid-source"
            source.mkdir()
            (source / "one.txt").write_bytes(b"one")
            (source / "two.txt").write_bytes(b"two")
            bundle = load_builder_input_bundle(
                source,
                input_id="id",
                input_type="type",
                allowed_files=("one.txt", "two.txt"),
            )
            workspace = root / "workspace"
            workspace.mkdir()
            staged = stage_builder_inputs((bundle,), workspace)
            target = staged[0].target
            forged = replace(
                staged[0],
                materialized_files=(
                    replace(staged[0].materialized_files[0], path=f"{target}/A/one.txt"),
                    replace(staged[0].materialized_files[1], path=f"{target}/a/two.txt"),
                ),
            )
            with self.assertRaises(ValueError):
                verify_staged_builder_inputs((forged,), workspace)

    def test_source_root_component_final_and_special_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            target = root / "outside.txt"
            target.write_bytes(b"outside")

            root_link = root / "root-link"
            root_link.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                self._bundle(root_link)

            component_target = root / "component-target"
            component_target.mkdir()
            (component_target / "value.txt").write_bytes(b"value")
            (source / "component-link").symlink_to(component_target, target_is_directory=True)
            with self.assertRaises(ValueError):
                load_builder_input_bundle(
                    source,
                    input_id="id",
                    input_type="type",
                    allowed_files=("component-link/value.txt",),
                )

            (source / "final-link").symlink_to(target)
            with self.assertRaises(ValueError):
                load_builder_input_bundle(
                    source,
                    input_id="id",
                    input_type="type",
                    allowed_files=("final-link",),
                )

            if hasattr(os, "mkfifo"):
                fifo = source / "fifo"
                os.mkfifo(fifo)
                with self.assertRaises(ValueError):
                    load_builder_input_bundle(
                        source,
                        input_id="id",
                        input_type="type",
                        allowed_files=("fifo",),
                    )

    def test_unlisted_secret_and_metadata_decoys_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            (source / ".env").write_bytes(b"secret")
            (source / "auth.json").write_bytes(b"credentials")
            (source / "history").mkdir()
            (source / "history" / "shell.log").write_bytes(b"history")
            (source / "profile").symlink_to(source / "a.txt")
            bundle = load_builder_input_bundle(
                source,
                input_id="id",
                input_type="type",
                allowed_files=("a.txt",),
            )
            self.assertEqual(tuple(item.path for item in bundle.manifest.files), ("a.txt",))

    def test_neutral_multi_input_staging_hides_ids_and_preserves_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._source(root, name="first")
            second = self._source(root, name="second")
            bundles = (
                self._bundle(first, input_id="known-condition-A"),
                self._bundle(second, input_id="known-condition-B"),
            )
            workspace = root / "workspace"
            workspace.mkdir()
            staged = stage_builder_inputs(bundles, workspace)

            self.assertEqual(tuple(record.input_id for record in staged), ("known-condition-A", "known-condition-B"))
            self.assertEqual(
                tuple(record.target for record in staged),
                (
                    "reference_files/builder-inputs/input-001",
                    "reference_files/builder-inputs/input-002",
                ),
            )
            for record in staged:
                self.assertNotIn(record.input_id, record.target)
                for item in record.materialized_files:
                    self.assertNotIn(record.input_id, item.path)
            verify_staged_builder_inputs(staged, workspace)

    def test_source_mutation_fails_before_destination_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            (source / "a.txt").write_bytes(b"changed")
            workspace = root / "workspace"
            workspace.mkdir()
            with self.assertRaises(RuntimeError):
                stage_builder_inputs((bundle,), workspace)
            self.assertFalse((workspace / "reference_files").exists())
            self.assertEqual(tuple(workspace.iterdir()), ())

    def test_workspace_namespace_and_source_overlap_never_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)

            workspace = root / "workspace-existing"
            workspace.mkdir()
            namespace = workspace / "reference_files"
            namespace.mkdir()
            sentinel = namespace / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                stage_builder_inputs((bundle,), workspace)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

            workspace = root / "workspace-symlink"
            workspace.mkdir()
            alias_target = root / "workspace-target"
            alias_target.mkdir()
            workspace.rmdir()
            workspace.symlink_to(alias_target, target_is_directory=True)
            with self.assertRaises(ValueError):
                stage_builder_inputs((bundle,), workspace)
            self.assertTrue(workspace.is_symlink())
            workspace.unlink()

            overlapping = source / "workspace-overlap"
            overlapping.mkdir()
            with self.assertRaises(ValueError):
                stage_builder_inputs((bundle,), overlapping)
            self.assertFalse((overlapping / "reference_files").exists())

    def test_copy_failure_rolls_back_only_the_new_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            workspace = root / "workspace"
            workspace.mkdir()
            sentinel = workspace / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            real_copy = inputs_module._copy_exclusive
            calls = 0

            def fail_second(destination: Path, content: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("deterministic copy failure")
                real_copy(destination, content)

            with patch.object(inputs_module, "_copy_exclusive", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "deterministic copy failure"):
                    stage_builder_inputs((bundle,), workspace)
            self.assertEqual(calls, 2)
            self.assertFalse((workspace / "reference_files").exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_source_tamper_during_copy_rolls_back_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            workspace = root / "workspace"
            workspace.mkdir()
            real_copy = inputs_module._copy_exclusive
            calls = 0

            def copy_then_tamper(destination: Path, content: bytes) -> None:
                nonlocal calls
                calls += 1
                real_copy(destination, content)
                if calls == 1:
                    (source / "a.txt").write_bytes(b"changed-after-preflight")

            with patch.object(inputs_module, "_copy_exclusive", side_effect=copy_then_tamper):
                with self.assertRaises(RuntimeError):
                    stage_builder_inputs((bundle,), workspace)
            self.assertFalse((workspace / "reference_files").exists())

    def test_verification_failure_after_read_only_modes_rolls_back_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            workspace = root / "workspace"
            workspace.mkdir()
            sentinel = workspace / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            def fail_after_read_only(staged: tuple[StagedBuilderInput, ...], destination: Path) -> None:
                materialized = destination / staged[0].materialized_files[0].path
                self.assertEqual(materialized.stat().st_mode & 0o222, 0)
                raise RuntimeError("deterministic verification failure")

            with patch.object(
                inputs_module,
                "verify_staged_builder_inputs",
                side_effect=fail_after_read_only,
            ):
                with self.assertRaisesRegex(RuntimeError, "deterministic verification failure"):
                    stage_builder_inputs((bundle,), workspace)
            self.assertFalse((workspace / "reference_files").exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_staged_record_is_frozen_and_evidence_is_a_tuple(self) -> None:
        content = b"value"
        item = InterventionFile(
            "reference_files/builder-inputs/input-001/value.txt",
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
        record = StagedBuilderInput("id", "reference_files/builder-inputs/input-001", [item])
        self.assertIsInstance(record.materialized_files, tuple)
        with self.assertRaises(FrozenInstanceError):
            setattr(record, "input_id", "changed")

    def test_verify_accepts_exact_tree_and_rejects_mutations(self) -> None:
        def mutate(kind: str) -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, workspace, staged = self._staged(root)
                destination = workspace / staged[0].materialized_files[0].path
                os.chmod(destination, 0o644)
                os.chmod(workspace / "reference_files", 0o755)
                os.chmod(workspace / "reference_files" / "builder-inputs", 0o755)
                os.chmod(workspace / staged[0].target, 0o755)
                if kind == "delete":
                    destination.unlink()
                elif kind == "mutate":
                    destination.write_bytes(b"tampered")
                elif kind == "extra":
                    (workspace / staged[0].target / "extra.txt").write_bytes(b"extra")
                elif kind == "symlink":
                    destination.unlink()
                    destination.symlink_to(source / "a.txt")
                elif kind == "special":
                    destination.unlink()
                    os.mkfifo(destination)
                with self.assertRaises(ValueError):
                    verify_staged_builder_inputs(staged, workspace)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workspace, staged = self._staged(root)
            verify_staged_builder_inputs(staged, workspace)
        for kind in ("delete", "mutate", "extra", "symlink"):
            mutate(kind)
        if hasattr(os, "mkfifo"):
            mutate("special")

    def test_empty_staging_is_valid_without_creating_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            self.assertEqual(stage_builder_inputs((), workspace), ())
            verify_staged_builder_inputs((), workspace)
            self.assertFalse((workspace / "reference_files").exists())


if __name__ == "__main__":
    unittest.main()
