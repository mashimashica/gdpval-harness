# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import eval_harness.run_manifest as run_module
from eval_harness.benchmarks.snapshot import Availability, BenchmarkSnapshot, acquire_snapshot
from eval_harness.candidate_bundle import SnapshotReference, VerifiedSnapshotBinding
from eval_harness.run_manifest import (
    RUN_MANIFEST_NAME,
    RunManifest,
    RunManifestError,
    RunResultRow,
    RunResultWriter,
    load_run_manifest,
    load_run_results,
    write_run_manifest,
)
from tests.harness.test_candidate_bundle import (
    _binding,
    _canonical,
    _seal,
    _snapshot_access,
    _SnapshotBenchmark,
)


_ZERO_DIGEST = "0" * 64
_ONE_DIGEST = "1" * 64


def _manifest(
    reference: SnapshotReference,
    *,
    run_id: str = "run-1",
    snapshot_path: str = "snapshot",
    results_path: str = "results.jsonl",
    configuration: dict[str, object] | None = None,
    ordered_tasks: tuple[SnapshotReference, ...] | None = None,
    root: Path = Path("."),
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        snapshot_path=snapshot_path,
        snapshot_sha256=reference.snapshot_sha256,
        results_path=results_path,
        configuration=configuration if configuration is not None else {"profile": "fixture", "seed": 7},
        configuration_sha256=None,
        ordered_tasks=(reference,) if ordered_tasks is None else ordered_tasks,
        run_fingerprint_sha256=None,
        root=root,
    )


def _prepare_run_root(root: Path) -> None:
    root.mkdir()
    (root / "snapshot").mkdir()


def _indexed_run(
    root: Path,
    binding: VerifiedSnapshotBinding,
    *,
    candidate_ids: tuple[str, ...] = ("candidate-1",),
) -> tuple[RunManifest, tuple[RunResultRow, ...]]:
    _prepare_run_root(root)
    reference = binding.reference("task-1")
    manifest = write_run_manifest(root, _manifest(reference))
    rows: list[RunResultRow] = []
    for sequence, candidate_id in enumerate(candidate_ids):
        bundle = _seal(root / "candidates" / candidate_id, binding, candidate_id=candidate_id)
        assert bundle.bundle_sha256 is not None
        rows.append(
            RunResultRow(
                sequence=sequence,
                candidate_id=candidate_id,
                snapshot_reference=reference,
                bundle_path=f"candidates/{candidate_id}",
                bundle_sha256=bundle.bundle_sha256,
            )
        )
    writer = RunResultWriter.create(root / manifest.results_path)
    for row in rows:
        writer.append(row)
    return manifest, tuple(rows)


def _replace_manifest(root: Path, payload: object) -> None:
    (root / RUN_MANIFEST_NAME).write_bytes(_canonical(payload))


class RunManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.access = _snapshot_access()
        self.binding = _binding(self.access)
        self.reference = self.binding.reference("task-1")

    def test_configuration_is_deep_frozen_and_fingerprint_has_exact_inputs(self) -> None:
        configuration: dict[str, object] = {"nested": {"values": [1, 2]}, "seed": 7}
        first = _manifest(self.reference, configuration=configuration, root=Path("first"))
        configuration["nested"] = {"values": [99]}
        self.assertEqual(first.configuration["nested"]["values"], (1, 2))  # type: ignore[index]
        with self.assertRaises(TypeError):
            first.configuration["new"] = True  # type: ignore[index]

        same = _manifest(
            self.reference,
            run_id="different-run-id",
            snapshot_path="other-snapshot-link",
            results_path="other-results.jsonl",
            configuration={"nested": {"values": [1, 2]}, "seed": 7},
            root=Path("second"),
        )
        self.assertEqual(first.run_fingerprint_sha256, same.run_fingerprint_sha256)
        self.assertEqual(first.configuration_sha256, same.configuration_sha256)
        changed_configuration = _manifest(self.reference, configuration={"seed": 8})
        self.assertNotEqual(first.run_fingerprint_sha256, changed_configuration.run_fingerprint_sha256)

        second_reference = SnapshotReference(self.reference.snapshot_sha256, "task-2", _ONE_DIGEST)
        forward = _manifest(self.reference, ordered_tasks=(self.reference, second_reference))
        reverse = _manifest(self.reference, ordered_tasks=(second_reference, self.reference))
        self.assertNotEqual(forward.run_fingerprint_sha256, reverse.run_fingerprint_sha256)

    def test_manifest_typed_invariants_fail_closed(self) -> None:
        invalid_paths = ("", ".", "..", "/tmp/x", "C:/tmp/x", "a\\b", "a//b", "a/../b", "e\u0301")
        for path in invalid_paths:
            with self.subTest(path=path), self.assertRaises(RunManifestError):
                _manifest(self.reference, snapshot_path=path)
            with self.subTest(results_path=path), self.assertRaises(RunManifestError):
                _manifest(self.reference, results_path=path)
        for snapshot_path, results_path in (("same", "same"), ("data", "data/results"), ("Data", "data")):
            with (
                self.subTest(snapshot_path=snapshot_path, results_path=results_path),
                self.assertRaises(RunManifestError),
            ):
                _manifest(self.reference, snapshot_path=snapshot_path, results_path=results_path)
        reserved_links = (
            ("candidates", "results.jsonl"),
            ("candidates/snapshot", "results.jsonl"),
            ("snapshot", "candidates"),
            ("snapshot", "Candidates/results.jsonl"),
            (RUN_MANIFEST_NAME, "results.jsonl"),
            (f"{RUN_MANIFEST_NAME}/snapshot", "results.jsonl"),
            ("snapshot", RUN_MANIFEST_NAME.upper()),
        )
        for snapshot_path, results_path in reserved_links:
            with (
                self.subTest(snapshot_path=snapshot_path, results_path=results_path),
                self.assertRaises(RunManifestError),
            ):
                _manifest(self.reference, snapshot_path=snapshot_path, results_path=results_path)

        invalid_configuration: tuple[object, ...] = (
            [],
            {"value": float("nan")},
            {"value": float("inf")},
            {1: "value"},
            {"value": b"bytes"},
            {"value": "\ud800"},
        )
        for configuration in invalid_configuration:
            with self.subTest(configuration=repr(configuration)), self.assertRaises((RunManifestError, TypeError)):
                _manifest(self.reference, configuration=cast(dict[str, object], configuration))

        valid_float = _manifest(self.reference, configuration={"ratio": 0.5})
        self.assertEqual(valid_float.configuration["ratio"], 0.5)
        with self.assertRaises(RunManifestError):
            _manifest(self.reference, run_id="\ud800")
        with self.assertRaises(RunManifestError):
            RunManifest(
                "run",
                "snapshot",
                "bad-digest",
                "results.jsonl",
                {},
                None,
                (self.reference,),
                None,
            )
        with self.assertRaises(TypeError):
            _manifest(self.reference, root=cast(Path, "run"))

        with self.assertRaises(RunManifestError):
            _manifest(self.reference, ordered_tasks=())
        with self.assertRaises(RunManifestError):
            _manifest(self.reference, ordered_tasks=(self.reference, self.reference))
        with self.assertRaises(RunManifestError):
            _manifest(
                self.reference,
                ordered_tasks=(SnapshotReference(_ZERO_DIGEST, self.reference.task_id, self.reference.task_sha256),),
            )
        with self.assertRaises(RunManifestError):
            RunManifest(
                "run",
                "snapshot",
                self.reference.snapshot_sha256,
                "results",
                {},
                _ZERO_DIGEST,
                (self.reference,),
                None,
            )
        with self.assertRaises(RunManifestError):
            RunManifest(
                "run",
                "snapshot",
                self.reference.snapshot_sha256,
                "results",
                {},
                None,
                (self.reference,),
                _ZERO_DIGEST,
            )

    def test_write_load_relocation_and_exclusive_manifest(self) -> None:
        run_root = self.root / "run"
        _prepare_run_root(run_root)
        original = _manifest(self.reference, root=Path("ignored"))
        written = write_run_manifest(run_root, original)
        self.assertEqual(written, original)
        self.assertEqual(written.root, run_root.absolute())
        raw = (run_root / RUN_MANIFEST_NAME).read_bytes()
        self.assertFalse(raw.endswith(b"\n"))
        loaded = load_run_manifest(run_root)
        self.assertEqual(loaded, written)
        with self.assertRaises(RunManifestError):
            write_run_manifest(run_root, original)

        relocated = self.root / "relocated"
        shutil.copytree(run_root, relocated)
        moved = load_run_manifest(relocated)
        self.assertEqual(moved.run_fingerprint_sha256, loaded.run_fingerprint_sha256)
        self.assertNotEqual(moved.root, loaded.root)

    def test_write_and_load_require_existing_nonsymlink_roots(self) -> None:
        missing = self.root / "missing"
        with self.assertRaises(RunManifestError):
            write_run_manifest(missing, _manifest(self.reference))
        regular = self.root / "regular"
        regular.write_text("x")
        with self.assertRaises(RunManifestError):
            write_run_manifest(regular, _manifest(self.reference))
        real = self.root / "real"
        _prepare_run_root(real)
        linked = self.root / "linked"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaises(RunManifestError):
            write_run_manifest(linked, _manifest(self.reference))
        with self.assertRaises(TypeError):
            write_run_manifest(cast(Path, "run"), _manifest(self.reference))
        with self.assertRaises(TypeError):
            write_run_manifest(real, cast(RunManifest, object()))
        with self.assertRaises(TypeError):
            load_run_manifest(cast(Path, "run"))

    def test_load_manifest_rejects_malformed_noncanonical_and_unknown_json(self) -> None:
        run_root = self.root / "run"
        _prepare_run_root(run_root)
        written = write_run_manifest(run_root, _manifest(self.reference))
        raw = written.canonical_manifest_bytes()
        invalid: tuple[tuple[str, bytes], ...] = (
            ("newline", raw + b"\n"),
            ("duplicate", b'{"schema":"run-manifest",' + raw[1:]),
            ("nonfinite", b'{"value":NaN}'),
            ("unknown", raw[:-1] + b',"unknown":true}'),
            ("array", b"[]"),
            ("invalid", b"{"),
            ("utf8", b"\xff"),
        )
        for name, content in invalid:
            root = self.root / name
            shutil.copytree(run_root, root)
            (root / RUN_MANIFEST_NAME).write_bytes(content)
            with self.subTest(name=name), self.assertRaises(RunManifestError):
                load_run_manifest(root)

        bool_version = self.root / "bool-version"
        shutil.copytree(run_root, bool_version)
        payload = cast(dict[str, object], json.loads(raw))
        payload["schema_version"] = True
        _replace_manifest(bool_version, payload)
        with self.assertRaises(RunManifestError):
            load_run_manifest(bool_version)

        malformed_shapes: tuple[dict[str, object], ...] = ({"ordered_tasks": {}}, {"configuration": []})
        for index, update in enumerate(malformed_shapes):
            malformed = self.root / f"malformed-shape-{index}"
            shutil.copytree(run_root, malformed)
            changed = cast(dict[str, object], json.loads(raw)) | update
            _replace_manifest(malformed, changed)
            with self.assertRaises(RunManifestError):
                load_run_manifest(malformed)

    def test_load_manifest_requires_explicit_existing_snapshot_link(self) -> None:
        run_root = self.root / "run"
        _prepare_run_root(run_root)
        write_run_manifest(run_root, _manifest(self.reference))
        shutil.rmtree(run_root / "snapshot")
        with self.assertRaises(RunManifestError):
            load_run_manifest(run_root)
        target = self.root / "snapshot-target"
        target.mkdir()
        (run_root / "snapshot").symlink_to(target, target_is_directory=True)
        with self.assertRaises(RunManifestError):
            load_run_manifest(run_root)

        regular_root = self.root / "regular-snapshot"
        _prepare_run_root(regular_root)
        write_run_manifest(regular_root, _manifest(self.reference))
        shutil.rmtree(regular_root / "snapshot")
        (regular_root / "snapshot").write_text("not a snapshot directory")
        with self.assertRaises(RunManifestError):
            load_run_manifest(regular_root)

    def test_scalar_and_filesystem_failures_are_normalized(self) -> None:
        with self.assertRaises(RunManifestError):
            run_module._canonical_bytes("\ud800")
        with self.assertRaises(RunManifestError):
            run_module._canonical_bytes(object())
        with self.assertRaises(RunManifestError):
            run_module._finite_float("1e999")
        with self.assertRaises(RunManifestError):
            run_module._validate_relative_path("\ud800", label="fixture")
        with patch.object(os.path, "abspath", side_effect=OSError("hidden")):
            with self.assertRaises(RunManifestError):
                run_module._absolute_without_symlinks(self.root, label="fixture")
        with patch.object(Path, "lstat", side_effect=OSError("hidden")):
            with self.assertRaises(RunManifestError):
                run_module._absolute_without_symlinks(self.root, label="fixture")
        with self.assertRaises(RunManifestError):
            run_module._regular_stat(self.root, label="fixture")

        regular = self.root / "regular-read"
        regular.write_bytes(b"content")
        with patch.object(os, "open", side_effect=OSError("hidden")):
            with self.assertRaises(RunManifestError):
                run_module._read_regular(regular, label="fixture")
        with patch.object(run_module, "_same_identity", return_value=False):
            with self.assertRaisesRegex(RunManifestError, "changed"):
                run_module._read_regular(regular, label="fixture")
        with patch.object(run_module, "_same_identity", side_effect=(True, False)):
            with self.assertRaisesRegex(RunManifestError, "changed"):
                run_module._read_regular(regular, label="fixture")
        with patch.object(os, "read", return_value=b""):
            with self.assertRaisesRegex(RunManifestError, "changed"):
                run_module._read_regular(regular, label="fixture")
        with patch.object(os, "fsync", side_effect=OSError("hidden")):
            with self.assertRaises(RunManifestError):
                run_module._fsync_directory(self.root)
        with patch.object(os, "open", side_effect=OSError("hidden")):
            with self.assertRaises(RunManifestError):
                run_module._write_exclusive(self.root / "write", b"content", label="fixture")

    def test_result_row_validates_schema_fields(self) -> None:
        row = RunResultRow(0, "candidate", self.reference, "candidates/one", _ZERO_DIGEST)
        self.assertTrue(row.canonical_line().endswith(b"\n"))

        invalid_rows: tuple[tuple[object, object, object, object, object], ...] = (
            (True, "candidate", self.reference, "candidate", _ZERO_DIGEST),
            (-1, "candidate", self.reference, "candidate", _ZERO_DIGEST),
            (0, "", self.reference, "candidate", _ZERO_DIGEST),
            (0, "candidate", object(), "candidate", _ZERO_DIGEST),
            (0, "candidate", self.reference, "../candidate", _ZERO_DIGEST),
            (0, "candidate", self.reference, "candidates/group/candidate", _ZERO_DIGEST),
            (0, "candidate", self.reference, "candidate", "bad"),
        )
        for values in invalid_rows:
            with self.subTest(values=values), self.assertRaises((RunManifestError, TypeError)):
                RunResultRow(*values)  # type: ignore[arg-type]

    def test_writer_is_exclusive_contiguous_and_durable(self) -> None:
        path = self.root / "results.jsonl"
        writer = RunResultWriter.create(path)
        self.assertEqual(path.read_bytes(), b"")
        with self.assertRaises(RunManifestError):
            RunResultWriter.create(path)
        with self.assertRaises(TypeError):
            RunResultWriter.create(cast(Path, "results"))
        with self.assertRaises(TypeError):
            writer.append(cast(RunResultRow, object()))

        with self.assertRaises(TypeError):
            RunResultWriter(path, (0, 0, 0, 0, 0), _token=object())
        first = RunResultRow(0, "one", self.reference, "candidates/one", _ZERO_DIGEST)
        second = RunResultRow(1, "two", self.reference, "candidates/two", _ONE_DIGEST)
        with patch("eval_harness.run_manifest.os.fsync", wraps=os.fsync) as fsync:
            writer.append(first)
            self.assertTrue(fsync.called)
        self.assertEqual(path.read_bytes(), first.canonical_line())
        with self.assertRaises(RunManifestError):
            writer.append(RunResultRow(3, "three", self.reference, "candidates/three", _ZERO_DIGEST))
        with self.assertRaises(RunManifestError):
            writer.append(RunResultRow(1, "one", self.reference, "candidates/other", _ZERO_DIGEST))
        with self.assertRaises(RunManifestError):
            writer.append(RunResultRow(1, "other", self.reference, "candidates/ONE", _ZERO_DIGEST))
        with self.assertRaises(RunManifestError):
            writer.append(RunResultRow(1, "other", self.reference, "candidates/one/child", _ZERO_DIGEST))
        writer.append(second)
        self.assertEqual(path.read_bytes(), first.canonical_line() + second.canonical_line())

    def test_writer_detects_external_file_changes(self) -> None:
        path = self.root / "results.jsonl"
        writer = RunResultWriter.create(path)
        path.write_bytes(b"external")
        with self.assertRaisesRegex(RunManifestError, "outside"):
            writer.append(RunResultRow(0, "one", self.reference, "candidates/one", _ZERO_DIGEST))

        other = self.root / "other.jsonl"
        other.write_bytes(b"")
        path.unlink()
        path.symlink_to(other)
        with self.assertRaises(RunManifestError):
            writer.append(RunResultRow(0, "one", self.reference, "candidates/one", _ZERO_DIGEST))

        same_size_path = self.root / "same-size-results.jsonl"
        same_size_writer = RunResultWriter.create(same_size_path)
        first = RunResultRow(0, "first", self.reference, "candidates/first", _ZERO_DIGEST)
        same_size_writer.append(first)
        before = same_size_path.stat()
        same_size_path.write_bytes(b"x" * before.st_size)
        changed = same_size_path.stat()
        if (changed.st_mtime_ns, changed.st_ctime_ns) == (before.st_mtime_ns, before.st_ctime_ns):
            os.utime(same_size_path, ns=(changed.st_atime_ns, changed.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(RunManifestError, "outside"):
            same_size_writer.append(RunResultRow(1, "second", self.reference, "candidates/second", _ONE_DIGEST))

    def test_writer_path_binding_and_append_races_fail_closed(self) -> None:
        path = self.root / "results.jsonl"
        writer = RunResultWriter.create(path)
        self.assertTrue(writer._matches_path(path))
        self.assertFalse(writer._matches_path(cast(Path, "results.jsonl")))
        self.assertFalse(writer._matches_path(self.root / "missing.jsonl"))

        opened_race = self.root / "opened-race.jsonl"
        opened_writer = RunResultWriter.create(opened_race)
        with patch.object(run_module, "_stat_seal", side_effect=(opened_writer._seal, (0, 0, 0, 0, 0))):
            with self.assertRaisesRegex(RunManifestError, "outside"):
                opened_writer.append(RunResultRow(0, "opened", self.reference, "candidates/opened", _ZERO_DIGEST))

        append_error = self.root / "append-error.jsonl"
        append_writer = RunResultWriter.create(append_error)
        with patch.object(os, "open", side_effect=OSError("hidden")):
            with self.assertRaises(RunManifestError):
                append_writer.append(RunResultRow(0, "error", self.reference, "candidates/error", _ZERO_DIGEST))

        append_race = self.root / "append-race.jsonl"
        race_writer = RunResultWriter.create(append_race)
        with patch.object(run_module, "_same_identity", return_value=False):
            with self.assertRaisesRegex(RunManifestError, "during append"):
                race_writer.append(RunResultRow(0, "race", self.reference, "candidates/race", _ZERO_DIGEST))

    def test_load_results_follows_index_order_and_survives_run_relocation(self) -> None:
        run_root = self.root / "run"
        _, rows = _indexed_run(run_root, self.binding, candidate_ids=("candidate-b", "candidate-a"))
        manifest = load_run_manifest(run_root)
        loaded = load_run_results(manifest, snapshot_binding=self.binding)
        self.assertEqual(tuple(row.candidate_id for row, _bundle in loaded), ("candidate-b", "candidate-a"))
        self.assertEqual(tuple(row for row, _bundle in loaded), rows)

        _seal(run_root / "candidates/unindexed", self.binding, candidate_id="unindexed")
        self.assertEqual(len(load_run_results(manifest, snapshot_binding=self.binding)), 2)

        relocated = self.root / "relocated"
        shutil.copytree(run_root, relocated)
        moved_manifest = load_run_manifest(relocated)
        moved = load_run_results(moved_manifest, snapshot_binding=self.binding)
        self.assertEqual(
            tuple(bundle.root for _row, bundle in moved),
            (
                relocated / "candidates/candidate-b",
                relocated / "candidates/candidate-a",
            ),
        )

        empty_root = self.root / "empty-run"
        _prepare_run_root(empty_root)
        empty_manifest = write_run_manifest(empty_root, _manifest(self.reference))
        RunResultWriter.create(empty_root / empty_manifest.results_path)
        self.assertEqual(load_run_results(empty_manifest, snapshot_binding=self.binding), ())

    def test_load_results_binds_the_explicit_physical_snapshot_and_relocation(self) -> None:
        run_root = self.root / "run-with-snapshot"
        run_root.mkdir()
        acquire_snapshot(_SnapshotBenchmark(), 1, run_root / "snapshot")
        binding = VerifiedSnapshotBinding.load(run_root / "snapshot")
        reference = binding.reference("task-1")
        manifest = write_run_manifest(run_root, _manifest(reference))
        bundle = _seal(run_root / "candidates/candidate", binding, candidate_id="candidate")
        assert bundle.bundle_sha256 is not None
        row = RunResultRow(0, "candidate", reference, "candidates/candidate", bundle.bundle_sha256)
        writer = RunResultWriter.create(run_root / manifest.results_path)
        writer.append(row)
        self.assertEqual(load_run_results(manifest, snapshot_binding=binding), ((row, bundle),))

        external_snapshot = self.root / "same-bytes-elsewhere"
        acquire_snapshot(_SnapshotBenchmark(), 1, external_snapshot)
        external_binding = VerifiedSnapshotBinding.load(external_snapshot)
        self.assertEqual(external_binding.reference("task-1"), reference)
        with self.assertRaisesRegex(RunManifestError, "explicit snapshot path"):
            load_run_results(manifest, snapshot_binding=external_binding)

        relocated = self.root / "relocated-with-snapshot"
        shutil.copytree(run_root, relocated)
        moved_manifest = load_run_manifest(relocated)
        moved_binding = VerifiedSnapshotBinding.load(relocated / "snapshot")
        moved = load_run_results(moved_manifest, snapshot_binding=moved_binding)
        self.assertEqual(moved[0][0], row)
        self.assertEqual(moved[0][1].bundle_sha256, bundle.bundle_sha256)
        self.assertEqual(moved[0][1].root, relocated / "candidates/candidate")
        with self.assertRaisesRegex(RunManifestError, "explicit snapshot path"):
            load_run_results(moved_manifest, snapshot_binding=binding)

    def test_load_results_rejects_row_bundle_and_binding_mismatches(self) -> None:
        cases = ("digest", "candidate-id", "reference", "missing")
        for case in cases:
            run_root = self.root / case
            _prepare_run_root(run_root)
            manifest = write_run_manifest(run_root, _manifest(self.reference))
            bundle = _seal(run_root / "candidates/real", self.binding, candidate_id="real")
            assert bundle.bundle_sha256 is not None
            reference = self.reference
            candidate_id = "real"
            bundle_path = "candidates/real"
            bundle_digest = bundle.bundle_sha256
            if case == "digest":
                bundle_digest = _ZERO_DIGEST
            elif case == "candidate-id":
                candidate_id = "other"
            elif case == "reference":
                reference = SnapshotReference(self.reference.snapshot_sha256, "other-task", _ONE_DIGEST)
            else:
                bundle_path = "candidates/missing"
            writer = RunResultWriter.create(run_root / manifest.results_path)
            writer.append(RunResultRow(0, candidate_id, reference, bundle_path, bundle_digest))
            with self.subTest(case=case), self.assertRaises(RunManifestError):
                load_run_results(load_run_manifest(run_root), snapshot_binding=self.binding)

        different_snapshot = BenchmarkSnapshot(
            benchmark_id="different",
            source=None,
            source_availability=Availability.UNAVAILABLE,
            revision=None,
            revision_availability=Availability.UNAVAILABLE,
            tasks=self.access.snapshot.tasks,
        )
        wrong_binding = _binding(type(self.access)(different_snapshot, self.access._blobs))
        manifest, _ = _indexed_run(self.root / "wrong-binding", self.binding)
        with self.assertRaises(RunManifestError):
            load_run_results(manifest, snapshot_binding=wrong_binding)

    def test_load_results_rejects_symlinks_tamper_and_manifest_change(self) -> None:
        run_root = self.root / "run"
        manifest, _ = _indexed_run(run_root, self.binding)
        external = self.root / "external"
        shutil.move(run_root / "candidates/candidate-1", external)
        (run_root / "candidates/candidate-1").symlink_to(external, target_is_directory=True)
        with self.assertRaises(RunManifestError):
            load_run_results(manifest, snapshot_binding=self.binding)

        changed = self.root / "changed"
        _, _ = _indexed_run(changed, self.binding)
        stale = load_run_manifest(changed)
        payload = stale.manifest_payload()
        payload["run_id"] = "changed-id"
        _replace_manifest(changed, payload)
        with self.assertRaisesRegex(RunManifestError, "changed"):
            load_run_results(stale, snapshot_binding=self.binding)

        sealed_root = self.root / "sealed-snapshot"
        sealed_root.mkdir()
        acquire_snapshot(_SnapshotBenchmark(), 1, sealed_root / "snapshot")
        sealed_binding = VerifiedSnapshotBinding.load(sealed_root / "snapshot")
        sealed_reference = sealed_binding.reference("task-1")
        sealed_manifest = write_run_manifest(sealed_root, _manifest(sealed_reference))
        RunResultWriter.create(sealed_root / sealed_manifest.results_path)
        snapshot_manifest = sealed_root / "snapshot/benchmark-snapshot.json"
        snapshot_manifest.write_bytes(snapshot_manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(RunManifestError, "binding is invalid"):
            load_run_results(sealed_manifest, snapshot_binding=sealed_binding)

    def test_load_results_rejects_malformed_noncanonical_and_duplicate_rows(self) -> None:
        original = self.root / "original"
        manifest, rows = _indexed_run(original, self.binding)
        raw_line = rows[0].canonical_line()
        unknown_payload = rows[0].payload()
        unknown_payload["unknown"] = True
        wrong_sequence = RunResultRow(
            2,
            rows[0].candidate_id,
            rows[0].snapshot_reference,
            rows[0].bundle_path,
            rows[0].bundle_sha256,
        ).canonical_line()
        invalid: tuple[tuple[str, bytes], ...] = (
            ("space", b" " + raw_line),
            ("truncated", raw_line[:-1]),
            ("blank", raw_line + b"\n"),
            ("duplicate-key", b'{"schema":"run-result",' + raw_line[1:]),
            ("nonfinite", b'{"value":NaN}\n'),
            ("unknown", _canonical(unknown_payload) + b"\n"),
            ("wrong-sequence", wrong_sequence),
            ("array", b"[]\n"),
            ("invalid-utf8", b"\xff\n"),
        )
        for name, content in invalid:
            root = self.root / name
            shutil.copytree(original, root)
            (root / manifest.results_path).write_bytes(content)
            with self.subTest(name=name), self.assertRaises(RunManifestError):
                load_run_results(load_run_manifest(root), snapshot_binding=self.binding)

        malformed_reference = rows[0].payload()
        malformed_reference["snapshot_reference"] = []
        bad_reference = self.root / "bad-reference"
        shutil.copytree(original, bad_reference)
        (bad_reference / manifest.results_path).write_bytes(_canonical(malformed_reference) + b"\n")
        with self.assertRaises(RunManifestError):
            load_run_results(load_run_manifest(bad_reference), snapshot_binding=self.binding)

        invalid_reference = rows[0].payload()
        invalid_reference["snapshot_reference"] = {
            "snapshot_sha256": "bad",
            "task_id": self.reference.task_id,
            "task_sha256": self.reference.task_sha256,
        }
        bad_reference_value = self.root / "bad-reference-value"
        shutil.copytree(original, bad_reference_value)
        (bad_reference_value / manifest.results_path).write_bytes(_canonical(invalid_reference) + b"\n")
        with self.assertRaises(RunManifestError):
            load_run_results(load_run_manifest(bad_reference_value), snapshot_binding=self.binding)

        wrong_schema = rows[0].payload()
        wrong_schema["schema_version"] = True
        bad_schema = self.root / "bad-result-schema"
        shutil.copytree(original, bad_schema)
        (bad_schema / manifest.results_path).write_bytes(_canonical(wrong_schema) + b"\n")
        with self.assertRaises(RunManifestError):
            load_run_results(load_run_manifest(bad_schema), snapshot_binding=self.binding)

        colliding = self.root / "colliding"
        shutil.copytree(original, colliding)
        second = RunResultRow(
            1,
            "other",
            self.reference,
            f"candidates/{rows[0].bundle_path.removeprefix('candidates/').upper()}",
            rows[0].bundle_sha256,
        )
        (colliding / manifest.results_path).write_bytes(raw_line + second.canonical_line())
        with self.assertRaises(RunManifestError):
            load_run_results(load_run_manifest(colliding), snapshot_binding=self.binding)

        duplicate_id = self.root / "duplicate-id"
        shutil.copytree(original, duplicate_id)
        duplicate_bundle = _seal(
            duplicate_id / "candidates/duplicate", self.binding, candidate_id=rows[0].candidate_id
        )
        assert duplicate_bundle.bundle_sha256 is not None
        duplicate = RunResultRow(
            1,
            rows[0].candidate_id,
            self.reference,
            "candidates/duplicate",
            duplicate_bundle.bundle_sha256,
        )
        (duplicate_id / manifest.results_path).write_bytes(raw_line + duplicate.canonical_line())
        with self.assertRaisesRegex(RunManifestError, "candidate_id is duplicated"):
            load_run_results(load_run_manifest(duplicate_id), snapshot_binding=self.binding)

    def test_load_results_requires_typed_arguments_and_explicit_results_path(self) -> None:
        run_root = self.root / "run"
        _prepare_run_root(run_root)
        manifest = write_run_manifest(run_root, _manifest(self.reference, results_path="index/results.jsonl"))
        (run_root / "results.jsonl").write_text("fallback must not be read")
        with self.assertRaises(RunManifestError):
            load_run_results(manifest, snapshot_binding=self.binding)
        with self.assertRaises(TypeError):
            load_run_results(cast(RunManifest, object()), snapshot_binding=self.binding)
        with self.assertRaises(TypeError):
            load_run_results(manifest, snapshot_binding=cast(VerifiedSnapshotBinding, object()))


if __name__ == "__main__":
    unittest.main()
