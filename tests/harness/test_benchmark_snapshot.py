# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from eval_harness.benchmarks.aime26 import AIME26Benchmark
from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.benchmarks.bigcodebench import BigCodeBenchBenchmark
from eval_harness.benchmarks.gdpval import GDPvalBenchmark
from eval_harness.benchmarks.snapshot import (
    Availability,
    BenchmarkSnapshot,
    SnapshotError,
    SnapshotFile,
    SnapshotTask,
    SnapshotTaskContent,
    SnapshotView,
    acquire_snapshot,
    load_snapshot,
    materialize_evaluation,
    materialize_execution,
    read_evaluation_file,
    verify_snapshot,
)
from eval_harness.executors.base import TaskSpec


_FIXTURE = Path(__file__).parent / "fixtures" / "benchmark-snapshot-v1.json"
_SHA256 = "0" * 64


class _ProjectionBenchmark(Benchmark):
    name = "projection-fixture"

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        raise AssertionError("snapshot acquisition must not prepare a benchmark")

    def load_tasks(self, limit: int) -> Sequence[BenchmarkTask]:
        del limit
        return (BenchmarkTask(TaskSpec("projection-task", "canonical prompt")),)

    def materialize(self, task: BenchmarkTask, workspace: Path) -> Sequence[str]:
        del task, workspace
        raise AssertionError("explicit snapshot_task must own acquisition")

    def snapshot_source_paths(self) -> Sequence[Path]:
        return ()

    def snapshot_task(self, task: BenchmarkTask, workspace: Path) -> SnapshotTaskContent:
        del task, workspace
        return SnapshotTaskContent(
            evaluation_data={"answer_sentinel": "evaluation-only"},
            files=(("task_inputs/allowed.txt", b"allowed"),),
            evaluation_files=(
                ("task_inputs/allowed.txt", b"allowed"),
                ("task_inputs/evaluation-only.txt", b"secret"),
            ),
        )


class BenchmarkSnapshotTests(unittest.TestCase):
    def test_records_are_frozen_and_deep_copy_json(self) -> None:
        data: dict[str, object] = {"nested": {"items": [1, 2]}}
        view = SnapshotView(data)
        task = SnapshotTask("task", "prompt", SnapshotView({}), view)
        data["nested"] = {"items": [99]}
        self.assertEqual(view.data["nested"]["items"], (1, 2))  # type: ignore[index]
        with self.assertRaises(TypeError):
            view.data["nested"] = {}  # type: ignore[index]
        with self.assertRaises(TypeError):
            task.evaluation_view.data["nested"]["items"] += (3,)  # type: ignore[index]

        projection = task.evaluation_projection()
        projection["data"]["nested"]["items"].append(3)  # type: ignore[index]
        self.assertEqual(view.data["nested"]["items"], (1, 2))  # type: ignore[index]
        self.assertEqual(set(task.task_spec().__dataclass_fields__), {"task_id", "prompt"})

    def test_file_paths_and_collisions_fail_closed(self) -> None:
        invalid = ("", ".", "..", "/tmp/x", "C:/tmp/x", "a\\b", "a//b", "a/./b", "a/../b", "e\u0301.txt")
        for path in invalid:
            with self.subTest(path=path), self.assertRaises(SnapshotError):
                SnapshotFile(path, 0, _SHA256)
        with self.assertRaises(SnapshotError):
            SnapshotFile("a", True, _SHA256)
        with self.assertRaises(SnapshotError):
            SnapshotView({}, (SnapshotFile("A/x", 1, _SHA256), SnapshotFile("a/y", 1, _SHA256)))
        with self.assertRaises(SnapshotError):
            SnapshotView({}, (SnapshotFile("foo", 1, _SHA256), SnapshotFile("foo/bar", 1, _SHA256)))

    def test_invalid_typed_records_fail_closed(self) -> None:
        empty = SnapshotView({})
        invalid_data: tuple[object, ...] = (
            [],
            {"value": float("nan")},
            {"value": float("inf")},
            {1: "non-string-key"},
            {"value": b"bytes"},
            {"value": "\ud800"},
        )
        for data in invalid_data:
            with self.subTest(data=repr(data)), self.assertRaises(SnapshotError):
                SnapshotView(data)
        with self.assertRaises(SnapshotError):
            SnapshotView({}, data_sha256=_SHA256)
        with self.assertRaises(SnapshotError):
            SnapshotView({}, view_sha256=_SHA256)
        with self.assertRaises(SnapshotError):
            SnapshotView({}, (SnapshotFile("same", 0, _SHA256), SnapshotFile("same", 0, _SHA256)))
        with self.assertRaises(SnapshotError):
            SnapshotView({}, (SnapshotFile("A", 0, _SHA256), SnapshotFile("a", 0, _SHA256)))

        invalid_tasks = (
            ("", "prompt", empty, empty, None, None),
            ("task", 7, empty, empty, None, None),
            ("task", "\ud800", empty, empty, None, None),
            ("task", "prompt", SnapshotView({"leak": True}), empty, None, None),
            ("task", "prompt", empty, empty, _SHA256, None),
            ("task", "prompt", empty, empty, None, _SHA256),
        )
        for arguments in invalid_tasks:
            with self.subTest(arguments=repr(arguments)), self.assertRaises(SnapshotError):
                SnapshotTask(*arguments)
        with self.assertRaises(SnapshotError):
            SnapshotTaskContent(evaluation_data=[])  # type: ignore[arg-type]

        task = SnapshotTask("task", "prompt", empty, empty)
        invalid_snapshots = (
            {"benchmark_id": ""},
            {"source": None, "source_availability": Availability.AVAILABLE},
            {"source": "unexpected", "source_availability": Availability.UNAVAILABLE},
            {"revision": None, "revision_availability": Availability.AVAILABLE},
            {"tasks": ()},
            {"tasks": (task, task)},
            {"root": object()},
            {"snapshot_sha256": _SHA256},
        )
        defaults: dict[str, object] = {
            "benchmark_id": "fixture",
            "source": None,
            "source_availability": Availability.UNAVAILABLE,
            "revision": None,
            "revision_availability": Availability.UNAVAILABLE,
            "tasks": (task,),
        }
        for changes in invalid_snapshots:
            with self.subTest(changes=changes), self.assertRaises((SnapshotError, TypeError)):
                BenchmarkSnapshot(**(defaults | changes))  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            BenchmarkSnapshot(**defaults).task("missing")  # type: ignore[arg-type]

    def test_fixture_has_known_canonical_hash_and_relocates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            root.mkdir()
            shutil.copy2(_FIXTURE, root / "benchmark-snapshot.json")
            (root / "blobs" / "sha256").mkdir(parents=True)
            snapshot = load_snapshot(root)
            self.assertEqual(
                snapshot.snapshot_sha256,
                "59fb580571a791e18ad539fc1d249d930d9e864d405cc72b3d9bd656d5148b3d",
            )
            relocated = Path(temporary) / "relocated"
            shutil.copytree(root, relocated)
            self.assertEqual(load_snapshot(relocated).snapshot_sha256, snapshot.snapshot_sha256)
            self.assertEqual(snapshot.tasks[0].task_spec(), load_snapshot(relocated).tasks[0].task_spec())

    def test_aime_answer_changes_evaluation_and_snapshot_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "aime.jsonl"
            dataset.write_text(json.dumps({"question": "q", "expected_answer": "1"}) + "\n", encoding="utf-8")
            benchmark = AIME26Benchmark(root=root, dataset_path=dataset, prepare_script=root / "prepare.py")
            first = benchmark.acquire_snapshot(1, root / "first")
            dataset.write_text(json.dumps({"question": "q", "expected_answer": "2"}) + "\n", encoding="utf-8")
            second = benchmark.acquire_snapshot(1, root / "second")
            self.assertEqual(first.tasks[0].canonical_task_prompt, second.tasks[0].canonical_task_prompt)
            self.assertNotEqual(
                first.tasks[0].evaluation_view.view_sha256,
                second.tasks[0].evaluation_view.view_sha256,
            )
            self.assertNotEqual(first.snapshot_sha256, second.snapshot_sha256)
            self.assertNotIn("expected_answer", first.tasks[0].execution_projection())

    def test_bigcode_test_and_code_prompt_are_evaluation_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "bigcode.jsonl"

            def write(test: str, code_prompt: str) -> None:
                dataset.write_text(
                    json.dumps(
                        {
                            "question": "q",
                            "verifier_metadata": {
                                "task_id": "task",
                                "test": test,
                                "entry_point": "solve",
                                "code_prompt": code_prompt,
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            write("assert 1", "def solve():")
            benchmark = BigCodeBenchBenchmark(root=root, dataset_path=dataset, prepare_script=root / "prepare.py")
            first = benchmark.acquire_snapshot(1, root / "first")
            write("assert 2", "def solve(x):")
            second = benchmark.acquire_snapshot(1, root / "second")
            self.assertEqual(first.tasks[0].canonical_task_prompt, second.tasks[0].canonical_task_prompt)
            self.assertNotEqual(
                first.tasks[0].evaluation_view.view_sha256,
                second.tasks[0].evaluation_view.view_sha256,
            )
            self.assertNotEqual(first.snapshot_sha256, second.snapshot_sha256)
            execution = first.tasks[0].execution_projection()
            serialized = json.dumps(execution)
            for forbidden in ("test", "code_prompt", "evaluation"):
                self.assertNotIn(forbidden, serialized)

    def test_gdpval_rubric_and_reference_are_sealed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "gdpval.jsonl"
            source = root / "source.txt"
            source.write_bytes(b"one")
            dataset.write_text(
                json.dumps(
                    {
                        "task_id": "gdp-task",
                        "prompt": "p",
                        "reference_files": ["reference_files/a/source.txt"],
                        "reference_file_urls": [source.as_uri()],
                        "rubric_json": '{"criterion": 1}',
                        "rubric_pretty": "criterion: one",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            calls = 0

            def downloader(files: list[str], urls: list[str], workspace: Path) -> list[str]:
                nonlocal calls
                calls += 1
                target = workspace / files[0]
                target.parent.mkdir(parents=True)
                target.write_bytes(source.read_bytes())
                return files

            benchmark = GDPvalBenchmark(
                root=root,
                dataset_path=dataset,
                prepare_script=root / "prepare.py",
                reference_downloader=downloader,
            )
            snapshot = benchmark.acquire_snapshot(1, root / "snapshot")
            self.assertEqual(calls, 1)
            task = snapshot.tasks[0]
            self.assertEqual(task.execution_view.files[0].path, "task_inputs/a/source.txt")
            self.assertEqual(task.evaluation_view.files, task.execution_view.files)
            self.assertNotIn("rubric_json", json.dumps(task.execution_projection()))
            source.write_bytes(b"changed after seal")
            workspace = root / "workspace"
            self.assertEqual(materialize_execution(snapshot, "gdp-task", workspace), ("task_inputs/a/source.txt",))
            self.assertEqual((workspace / "task_inputs/a/source.txt").read_bytes(), b"one")

    def test_manifest_and_blob_tamper_fail_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "aime.jsonl"
            dataset.write_text(json.dumps({"question": "q", "expected_answer": "1"}) + "\n", encoding="utf-8")
            snapshot = AIME26Benchmark(
                root=root,
                dataset_path=dataset,
                prepare_script=root / "prepare.py",
            ).acquire_snapshot(1, root / "snapshot")
            manifest = snapshot.root / "benchmark-snapshot.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            with self.assertRaises(SnapshotError):
                verify_snapshot(snapshot)
            manifest.write_bytes(snapshot.canonical_manifest_bytes())
            (snapshot.root / "unexpected").write_text("x", encoding="utf-8")
            with self.assertRaises(SnapshotError):
                materialize_execution(snapshot, "aime26-01", root / "workspace")

    def test_execution_materialization_excludes_evaluation_only_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = acquire_snapshot(_ProjectionBenchmark(), 1, root / "snapshot")
            task = snapshot.tasks[0]
            self.assertEqual(
                [item.path for item in task.execution_view.files],
                ["task_inputs/allowed.txt"],
            )
            self.assertEqual(
                [item.path for item in task.evaluation_view.files],
                ["task_inputs/allowed.txt", "task_inputs/evaluation-only.txt"],
            )
            self.assertNotIn("answer_sentinel", json.dumps(task.execution_projection()))

            workspace = root / "workspace"
            materialize_execution(snapshot, task.task_id, workspace)
            self.assertEqual((workspace / "task_inputs/allowed.txt").read_bytes(), b"allowed")
            self.assertFalse((workspace / "task_inputs/evaluation-only.txt").exists())
            self.assertEqual({path.name for path in workspace.iterdir()}, {"task_inputs"})

            self.assertEqual(
                read_evaluation_file(snapshot, task.task_id, "task_inputs/evaluation-only.txt"),
                b"secret",
            )
            evaluation_root = root / "evaluation"
            self.assertEqual(
                materialize_evaluation(snapshot, task.task_id, evaluation_root),
                ("task_inputs/allowed.txt", "task_inputs/evaluation-only.txt"),
            )
            self.assertEqual((evaluation_root / "task_inputs/evaluation-only.txt").read_bytes(), b"secret")
            self.assertFalse((evaluation_root / "benchmark-snapshot.json").exists())

            shared = SnapshotFile("task_inputs/shared", 1, _SHA256)
            conflicting = SnapshotFile("task_inputs/shared", 2, _SHA256)
            with self.assertRaises(SnapshotError):
                SnapshotTask(
                    "task",
                    "prompt",
                    SnapshotView({}, (shared,)),
                    SnapshotView({}, (conflicting,)),
                )

    def test_snapshot_identity_binds_metadata_order_selection_and_prompt(self) -> None:
        first = SnapshotTask("one", "prompt one", SnapshotView({}), SnapshotView({"answer": "1"}))
        second = SnapshotTask("two", "prompt two", SnapshotView({}), SnapshotView({"answer": "2"}))

        def build(
            *,
            source: str = "source-a",
            revision: str = "revision-a",
            tasks: Sequence[SnapshotTask] = (first, second),
            root: Path = Path("runtime-a"),
        ) -> BenchmarkSnapshot:
            return BenchmarkSnapshot(
                benchmark_id="fixture",
                source=source,
                source_availability=Availability.AVAILABLE,
                revision=revision,
                revision_availability=Availability.AVAILABLE,
                tasks=tasks,
                root=root,
            )

        baseline = build()
        self.assertEqual(baseline.snapshot_sha256, build(root=Path("runtime-b")).snapshot_sha256)
        self.assertNotEqual(baseline.snapshot_sha256, build(source="source-b").snapshot_sha256)
        self.assertNotEqual(baseline.snapshot_sha256, build(revision="revision-b").snapshot_sha256)
        self.assertNotEqual(baseline.snapshot_sha256, build(tasks=(second, first)).snapshot_sha256)
        self.assertNotEqual(baseline.snapshot_sha256, build(tasks=(first,)).snapshot_sha256)
        changed_prompt = SnapshotTask("one", "changed", SnapshotView({}), SnapshotView({"answer": "1"}))
        self.assertNotEqual(baseline.snapshot_sha256, build(tasks=(changed_prompt, second)).snapshot_sha256)
        self.assertNotIn("runtime-a", baseline.canonical_manifest_bytes().decode("utf-8"))

    def test_malformed_manifest_encodings_fail_closed(self) -> None:
        canonical = _FIXTURE.read_bytes()
        malformed = {
            "duplicate": b'{"schema":"benchmark-snapshot",' + canonical[1:],
            "nonfinite": canonical.replace(b'"schema_version":1', b'"schema_version":NaN'),
            "noncanonical": canonical + b"\n",
            "unknown-field": canonical[:-1] + b',"unknown":true}',
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, content in malformed.items():
                with self.subTest(name=name):
                    snapshot_root = root / name
                    (snapshot_root / "blobs" / "sha256").mkdir(parents=True)
                    (snapshot_root / "benchmark-snapshot.json").write_bytes(content)
                    with self.assertRaises(SnapshotError):
                        load_snapshot(snapshot_root)

    def test_malformed_manifest_schemas_fail_closed(self) -> None:
        fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        malformed: list[object] = [[], {"schema": "benchmark-snapshot"}]

        wrong_version = deepcopy(fixture)
        wrong_version["schema_version"] = True
        malformed.append(wrong_version)
        wrong_tasks = deepcopy(fixture)
        wrong_tasks["tasks"] = "task"
        malformed.append(wrong_tasks)
        wrong_availability_type = deepcopy(fixture)
        wrong_availability_type["source_availability"] = 1
        malformed.append(wrong_availability_type)
        wrong_availability_value = deepcopy(fixture)
        wrong_availability_value["source_availability"] = "sometimes"
        malformed.append(wrong_availability_value)
        wrong_task = deepcopy(fixture)
        wrong_task["tasks"] = [1]
        malformed.append(wrong_task)
        wrong_view = deepcopy(fixture)
        wrong_view["tasks"][0]["execution_view"] = 1
        malformed.append(wrong_view)
        wrong_files = deepcopy(fixture)
        wrong_files["tasks"][0]["execution_view"]["files"] = "files"
        malformed.append(wrong_files)
        wrong_file = deepcopy(fixture)
        wrong_file["tasks"][0]["execution_view"]["files"] = [1]
        malformed.append(wrong_file)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, payload in enumerate(malformed):
                with self.subTest(index=index):
                    snapshot_root = root / str(index)
                    (snapshot_root / "blobs" / "sha256").mkdir(parents=True)
                    (snapshot_root / "benchmark-snapshot.json").write_text(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    with self.assertRaises(SnapshotError):
                        load_snapshot(snapshot_root)

    def test_blob_change_missing_symlink_and_extra_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = acquire_snapshot(_ProjectionBenchmark(), 1, root / "sealed")
            digest = sealed.tasks[0].execution_view.files[0].sha256
            external = root / "external"
            external.write_bytes(b"allowed")

            for mutation in ("changed", "missing", "symlink", "extra"):
                with self.subTest(mutation=mutation):
                    candidate = root / mutation
                    shutil.copytree(sealed.root, candidate)
                    blob = candidate / "blobs" / "sha256" / digest
                    if mutation == "changed":
                        blob.write_bytes(b"changed")
                    elif mutation == "missing":
                        blob.unlink()
                    elif mutation == "symlink":
                        blob.unlink()
                        blob.symlink_to(external)
                    else:
                        (candidate / "blobs" / "sha256" / ("1" * 64)).write_bytes(b"extra")
                    with self.assertRaises(SnapshotError):
                        verify_snapshot(candidate)

            workspace = root / "workspace"
            with self.assertRaises(SnapshotError):
                materialize_execution(root / "changed", "projection-task", workspace)
            self.assertFalse(workspace.exists())

    def test_invalid_adapter_contracts_never_publish(self) -> None:
        benchmark = _ProjectionBenchmark()
        task = BenchmarkTask(TaskSpec("task", "prompt"))
        invalid_content = (
            SnapshotTaskContent(files=(("outside.txt", b"x"),)),
            SnapshotTaskContent(files=(("task_inputs/x", object()),)),  # type: ignore[arg-type]
            SnapshotTaskContent(files=(("task_inputs/x",),)),  # type: ignore[arg-type]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(SnapshotError):
                acquire_snapshot(benchmark, 0, root / "zero")
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "sentinel"
            sentinel.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(SnapshotError):
                acquire_snapshot(benchmark, 1, existing)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

            cases: list[tuple[str, str, object]] = [
                ("empty", "load_tasks", ()),
                ("invalid-task", "load_tasks", (object(),)),
                ("duplicate-task", "load_tasks", (task, task)),
                ("invalid-content", "snapshot_task", object()),
                ("bad-source-shape", "snapshot_source_paths", "source"),
                ("bad-source-entry", "snapshot_source_paths", (object(),)),
            ]
            cases.extend(
                (f"content-{index}", "snapshot_task", content) for index, content in enumerate(invalid_content)
            )
            for name, method, result in cases:
                with self.subTest(name=name):
                    destination = root / name
                    with patch.object(benchmark, method, return_value=result), self.assertRaises(SnapshotError):
                        acquire_snapshot(benchmark, 1, destination)
                    self.assertFalse(destination.exists())

            source_directory = root / "source-directory"
            source_directory.mkdir()
            with (
                patch.object(benchmark, "snapshot_source_paths", return_value=(source_directory,)),
                self.assertRaises(SnapshotError),
            ):
                acquire_snapshot(benchmark, 1, root / "directory-source")
            self.assertFalse((root / "directory-source").exists())

            with patch.object(benchmark, "load_tasks", side_effect=RuntimeError("adapter failed")):
                with self.assertRaises(SnapshotError):
                    acquire_snapshot(benchmark, 1, root / "adapter-error")
            self.assertFalse((root / "adapter-error").exists())

    def test_sealed_root_and_projection_destinations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = acquire_snapshot(_ProjectionBenchmark(), 1, root / "sealed")
            external_manifest = root / "external-manifest"
            external_manifest.write_bytes((sealed.root / "benchmark-snapshot.json").read_bytes())
            external_directory = root / "external-directory"
            external_directory.mkdir()

            mutations = ("manifest-symlink", "blobs-file", "blob-child-extra", "sha-symlink")
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    candidate = root / mutation
                    shutil.copytree(sealed.root, candidate)
                    if mutation == "manifest-symlink":
                        manifest = candidate / "benchmark-snapshot.json"
                        manifest.unlink()
                        manifest.symlink_to(external_manifest)
                    elif mutation == "blobs-file":
                        shutil.rmtree(candidate / "blobs")
                        (candidate / "blobs").write_text("invalid", encoding="utf-8")
                    elif mutation == "blob-child-extra":
                        (candidate / "blobs" / "extra").mkdir()
                    else:
                        shutil.rmtree(candidate / "blobs" / "sha256")
                        (candidate / "blobs" / "sha256").symlink_to(external_directory, target_is_directory=True)
                    with self.assertRaises(SnapshotError):
                        verify_snapshot(candidate)

            root_file = root / "root-file"
            root_file.write_text("invalid", encoding="utf-8")
            with self.assertRaises(SnapshotError):
                verify_snapshot(root_file)
            root_link = root / "root-link"
            root_link.symlink_to(sealed.root, target_is_directory=True)
            with self.assertRaises(SnapshotError):
                verify_snapshot(root_link)
            with self.assertRaises(SnapshotError):
                verify_snapshot("invalid")  # type: ignore[arg-type]

            other_task = SnapshotTask("other", "prompt", SnapshotView({}), SnapshotView({}))
            mismatched = BenchmarkSnapshot(
                "other",
                None,
                Availability.UNAVAILABLE,
                None,
                Availability.UNAVAILABLE,
                (other_task,),
                root=sealed.root,
            )
            with self.assertRaises(SnapshotError):
                verify_snapshot(mismatched)

            with self.assertRaises(SnapshotError):
                read_evaluation_file(sealed, "projection-task", "task_inputs/missing")
            with self.assertRaises(SnapshotError):
                read_evaluation_file(sealed, "", "task_inputs/allowed.txt")
            existing_evaluation = root / "existing-evaluation"
            existing_evaluation.mkdir()
            with self.assertRaises(SnapshotError):
                materialize_evaluation(sealed, "projection-task", existing_evaluation)

            workspace_file = root / "workspace-file"
            workspace_file.write_text("invalid", encoding="utf-8")
            with self.assertRaises(SnapshotError):
                materialize_execution(sealed, "projection-task", workspace_file)
            colliding_workspace = root / "colliding-workspace"
            colliding_workspace.mkdir()
            (colliding_workspace / "TASK_INPUTS").mkdir()
            with self.assertRaises(SnapshotError):
                materialize_execution(sealed, "projection-task", colliding_workspace)
            with self.assertRaises(SnapshotError):
                materialize_execution(sealed, "", root / "empty-task")

    def test_source_ctime_change_is_rejected_without_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "aime.jsonl"
            original = (json.dumps({"question": "q", "expected_answer": "1"}) + "\n").encode()
            dataset.write_bytes(original)

            class MutatingAIME(AIME26Benchmark):
                def load_tasks(self, limit: int) -> list[BenchmarkTask]:
                    result = super().load_tasks(limit)
                    mtime = dataset.stat().st_mtime_ns
                    dataset.write_bytes(b"changed\n")
                    dataset.write_bytes(original)
                    os.utime(dataset, ns=(mtime, mtime))
                    return result

            benchmark = MutatingAIME(root=root, dataset_path=dataset, prepare_script=root / "prepare.py")
            destination = root / "snapshot"
            with self.assertRaises(SnapshotError):
                acquire_snapshot(benchmark, 1, destination)
            self.assertFalse(destination.exists())

    def test_source_mutation_during_streaming_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "aime.jsonl"
            row = (json.dumps({"question": "q", "expected_answer": "1"}) + "\n").encode("utf-8")
            original = row + b" " * (1024 * 1024)
            dataset.write_bytes(original)
            original_mtime = dataset.stat().st_mtime_ns
            real_read = os.read
            mutated = False

            def mutating_read(descriptor: int, size: int) -> bytes:
                nonlocal mutated
                content = real_read(descriptor, size)
                if content and not mutated:
                    mutated = True
                    dataset.write_bytes(original)
                    os.utime(dataset, ns=(original_mtime, original_mtime))
                return content

            benchmark = AIME26Benchmark(root=root, dataset_path=dataset, prepare_script=root / "prepare.py")
            destination = root / "snapshot"
            with patch.object(os, "read", side_effect=mutating_read), self.assertRaises(SnapshotError):
                acquire_snapshot(benchmark, 1, destination)
            self.assertTrue(mutated)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
