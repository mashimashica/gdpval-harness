# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from gdpval_harness.executors.base import TaskSpec
from gdpval_harness.interventions import FilesIntervention


class FileInterventionTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "source"
        (source / "nested").mkdir(parents=True)
        (source / "a.txt").write_bytes(b"A\x00")
        (source / "nested" / "b.txt").write_bytes(b"B")
        return source

    def test_preflight_orders_files_and_apply_returns_exact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            workspace = root / "workspace"
            workspace.mkdir()
            intervention = FilesIntervention(source)
            preflight = intervention.preflight()
            self.assertTrue(preflight.ok, preflight.details)
            self.assertEqual(
                [item.path for item in preflight.bundle.manifest.files],  # type: ignore[union-attr]
                ["a.txt", "nested/b.txt"],
            )

            application = intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run-1")
            self.assertEqual([item.path for item in application.materialized_files], ["a.txt", "nested/b.txt"])
            self.assertEqual(application.application.method, "workspace-files")
            self.assertEqual(application.application.target, ".")
            for item in application.materialized_files:
                destination = workspace / item.path
                self.assertTrue(destination.is_file())
                self.assertEqual(destination.stat().st_size, item.size)
                self.assertEqual(destination.read_bytes(), (source / item.path).read_bytes())

    def test_source_tamper_is_detected_before_destination_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            workspace = root / "workspace"
            workspace.mkdir()
            intervention = FilesIntervention(source)
            self.assertTrue(intervention.preflight().ok)
            (source / "nested" / "b.txt").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertEqual(list(workspace.iterdir()), [])

    def test_existing_destination_prevents_any_partial_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "nested").mkdir()
            (workspace / "nested" / "b.txt").write_text("existing", encoding="utf-8")
            intervention = FilesIntervention(source)
            self.assertTrue(intervention.preflight().ok)
            with self.assertRaises(FileExistsError):
                intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertFalse((workspace / "a.txt").exists())
            self.assertEqual((workspace / "nested" / "b.txt").read_text(encoding="utf-8"), "existing")

    def test_preexisting_symlink_parent_is_rejected_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (workspace / "nested").symlink_to(outside, target_is_directory=True)
            intervention = FilesIntervention(source)
            self.assertTrue(intervention.preflight().ok)
            with self.assertRaises(ValueError):
                intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertEqual(list(outside.iterdir()), [])
            self.assertFalse((workspace / "a.txt").exists())

    def test_existing_casefold_parent_collision_is_rejected_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "Foo").mkdir(parents=True)
            (source / "Foo" / "x.txt").write_text("x", encoding="utf-8")
            workspace = root / "workspace"
            (workspace / "foo").mkdir(parents=True)
            intervention = FilesIntervention(source)
            self.assertTrue(intervention.preflight().ok)
            with self.assertRaisesRegex(ValueError, "casefold"):
                intervention.apply(TaskSpec("task-1", "prompt"), workspace, application_run_id="run")
            self.assertEqual(list((workspace / "foo").iterdir()), [])

    def test_source_requires_regular_files_and_rejects_symlinks_reserved_paths_and_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases: list[tuple[str, Callable[[], object]]] = []

            symlink_source = root / "symlink-source"
            symlink_source.mkdir()
            external = root / "external.txt"
            external.write_text("x", encoding="utf-8")
            (symlink_source / "link").symlink_to(external)
            cases.append(("symlink", lambda: FilesIntervention(symlink_source).preflight()))

            reserved_source = root / "reserved-source"
            (reserved_source / "deliverables").mkdir(parents=True)
            (reserved_source / "deliverables" / "x").write_text("x", encoding="utf-8")
            cases.append(("reserved", lambda: FilesIntervention(reserved_source).preflight()))

            if os.name == "posix":
                special_source = root / "special-source"
                special_source.mkdir()
                fifo = special_source / "fifo"
                os.mkfifo(fifo)
                cases.append(("special", lambda: FilesIntervention(special_source).preflight()))

            collision_source = root / "collision-source"
            collision_source.mkdir()
            (collision_source / "A.txt").write_text("A", encoding="utf-8")
            (collision_source / "a.txt").write_text("a", encoding="utf-8")
            cases.append(("collision", lambda: FilesIntervention(collision_source).preflight()))

            windows_path_source = root / "windows-path-source"
            windows_path_source.mkdir()
            (windows_path_source / r"foo\..\bar").write_text("x", encoding="utf-8")
            cases.append(("windows-path", lambda: FilesIntervention(windows_path_source).preflight()))

            for name, preflight in cases:
                with self.subTest(name=name):
                    result = preflight()
                    self.assertFalse(result.ok, result.details)

    def test_root_and_empty_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty"
            empty.mkdir()
            self.assertFalse(FilesIntervention(empty).preflight().ok)
            regular = root / "file"
            regular.write_text("x", encoding="utf-8")
            self.assertFalse(FilesIntervention(regular).preflight().ok)
            link = root / "link"
            link.symlink_to(empty, target_is_directory=True)
            self.assertFalse(FilesIntervention(link).preflight().ok)

    def test_destination_workspace_must_be_real_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            intervention = FilesIntervention(source)
            self.assertTrue(intervention.preflight().ok)
            with self.assertRaises(ValueError):
                intervention.apply(TaskSpec("task-1", "prompt"), root / "missing", application_run_id="run")

    @unittest.skipUnless(os.name == "posix", "FIFO test requires POSIX")
    def test_source_fifo_is_special_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            fifo = source / "fifo"
            os.mkfifo(fifo, stat.S_IRUSR | stat.S_IWUSR)
            result = FilesIntervention(source).preflight()
            self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
