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

import eval_harness.benchmarks.gdpval as gdpval_module
import eval_harness.benchmarks.snapshot as snapshot_module
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


class _DefaultSnapshotBenchmark(Benchmark):
    name = "default-snapshot-fixture"

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        raise AssertionError("snapshot acquisition must not prepare a benchmark")

    def load_tasks(self, limit: int) -> Sequence[BenchmarkTask]:
        del limit
        return (
            BenchmarkTask(
                TaskSpec("default-task", "canonical prompt"),
                evaluation={"answer": "evaluation-only"},
            ),
        )

    def materialize(self, task: BenchmarkTask, workspace: Path) -> Sequence[str]:
        del task
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "input.txt").write_bytes(b"input")
        return ("input.txt",)


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

    def test_canonical_json_and_manifest_primitives_fail_closed(self) -> None:
        with self.assertRaises(SnapshotError):
            SnapshotFile("file", 0, "not-a-digest")
        with self.assertRaises(SnapshotError):
            snapshot_module._canonical_bytes({"text": "\ud800"})
        with self.assertRaises(SnapshotError):
            snapshot_module._canonical_bytes({"unsupported": object()})
        with self.assertRaises(SnapshotError):
            snapshot_module._decode_json_object('{"number":1e999}', label="row")
        with self.assertRaises(SnapshotError):
            snapshot_module._decode_json_object("{", label="row")
        with self.assertRaises(SnapshotError):
            snapshot_module._decode_json_object("[]", label="row")
        self.assertEqual(SnapshotView({"finite": 1.5}).data["finite"], 1.5)  # type: ignore[index]

        with self.assertRaises(SnapshotError):
            snapshot_module._validate_file_projection((object(),))  # type: ignore[arg-type]
        with self.assertRaises(SnapshotError):
            SnapshotTask("task", "prompt", object(), SnapshotView({}))  # type: ignore[arg-type]
        with self.assertRaises(SnapshotError):
            BenchmarkSnapshot(
                "fixture",
                None,
                "unknown",  # type: ignore[arg-type]
                None,
                Availability.UNAVAILABLE,
                (SnapshotTask("task", "prompt", SnapshotView({}), SnapshotView({})),),
            )

    def test_default_snapshot_hook_publishes_only_execution_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = acquire_snapshot(_DefaultSnapshotBenchmark(), 1, root / "snapshot")
            task = snapshot.tasks[0]
            self.assertEqual(tuple(file.path for file in task.execution_view.files), ("task_inputs/input.txt",))
            self.assertEqual(task.evaluation_view.files, ())
            self.assertEqual(task.evaluation_projection()["data"], {"answer": "evaluation-only"})
            workspace = root / "workspace"
            self.assertEqual(materialize_execution(snapshot, task.task_id, workspace), ("task_inputs/input.txt",))
            self.assertEqual((workspace / "task_inputs" / "input.txt").read_bytes(), b"input")

    def test_fixture_has_known_canonical_hash_and_relocates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            root.mkdir()
            shutil.copy2(_FIXTURE, root / "benchmark-snapshot.json")
            (root / "blobs" / "sha256").mkdir(parents=True)
            snapshot = load_snapshot(root)
            self.assertEqual(
                snapshot.snapshot_sha256,
                # Deterministic public fixture digest, independently recomputed from canonical JSON.
                "59fb580571a791e18ad539fc1d249d930d9e864d405cc72b3d9bd656d5148b3d",  # pragma: allowlist secret
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

    def test_snapshot_adapters_reject_malformed_task_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aime_data = root / "aime.jsonl"
            aime_data.write_text(json.dumps({"question": "q", "expected_answer": []}) + "\n", encoding="utf-8")
            aime = AIME26Benchmark(root=root, dataset_path=aime_data, prepare_script=root / "prepare.py")
            with self.assertRaises(RuntimeError):
                aime.load_tasks(1)
            with self.assertRaises(ValueError):
                aime.snapshot_task(BenchmarkTask(TaskSpec("task", "prompt"), evaluation={"unexpected": 1}), root)

            bigcode_data = root / "bigcode.jsonl"
            bigcode_data.write_text(
                json.dumps(
                    {
                        "question": 1,
                        "verifier_metadata": {
                            "task_id": "task",
                            "test": "test",
                            "entry_point": "entry",
                            "code_prompt": "prompt",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            bigcode = BigCodeBenchBenchmark(root=root, dataset_path=bigcode_data, prepare_script=root / "prepare.py")
            with self.assertRaises(RuntimeError):
                bigcode.load_tasks(1)

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

    def test_gdpval_snapshot_rejects_malformed_rows_and_downloader_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "gdpval.jsonl"
            prepare_script = root / "prepare.py"
            malformed_rows = (
                {"task_id": 1, "prompt": "p"},
                {"task_id": "task", "prompt": "p", "sector": 1},
                {"task_id": "task", "prompt": "p", "rubric_pretty": 1},
            )
            for index, row in enumerate(malformed_rows):
                with self.subTest(row=index):
                    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    benchmark = GDPvalBenchmark(root=root, dataset_path=dataset, prepare_script=prepare_script)
                    with self.assertRaises(RuntimeError):
                        benchmark.load_tasks(1)

            task = BenchmarkTask(
                TaskSpec("task", "prompt"),
                materialization={
                    "reference_files": ("reference_files/input.txt",),
                    "reference_file_urls": ("https://example.invalid/input.txt",),
                },
            )
            workspace = root / "workspace"
            workspace.mkdir()

            def result_case(result: object) -> GDPvalBenchmark:
                def downloader(files: list[str], urls: list[str], destination: Path) -> Sequence[str]:
                    del files, urls, destination
                    return result  # type: ignore[return-value]

                return GDPvalBenchmark(
                    root=root,
                    dataset_path=dataset,
                    prepare_script=prepare_script,
                    reference_downloader=downloader,
                )

            malformed_tasks = (
                BenchmarkTask(
                    TaskSpec("task", "prompt"),
                    materialization={"reference_files": ("input.txt",), "reference_file_urls": ()},
                ),
                BenchmarkTask(
                    TaskSpec("task", "prompt"),
                    materialization={"reference_files": ("input.txt",), "reference_file_urls": ("",)},
                ),
            )
            for index, malformed_task in enumerate(malformed_tasks):
                with self.subTest(metadata=index), self.assertRaises(SnapshotError):
                    result_case(()).snapshot_task(malformed_task, workspace)
            for index, result in enumerate(("input.txt", (), (object(),), ("/outside",))):
                with self.subTest(result=index), self.assertRaises(SnapshotError):
                    result_case(result).snapshot_task(task, workspace)

            local_source = root / "source.txt"
            local_source.write_bytes(b"original")
            local_task = BenchmarkTask(
                TaskSpec("task", "prompt"),
                materialization={
                    "reference_files": ("reference_files/input.txt",),
                    "reference_file_urls": (local_source.as_uri(),),
                },
            )

            def mutate_source(files: list[str], urls: list[str], destination: Path) -> Sequence[str]:
                del urls
                local_source.write_bytes(b"changed")
                target = destination / files[0]
                target.parent.mkdir(parents=True)
                target.write_bytes(b"original")
                return files

            changing = GDPvalBenchmark(
                root=root,
                dataset_path=dataset,
                prepare_script=prepare_script,
                reference_downloader=mutate_source,
            )
            with self.assertRaises(SnapshotError):
                changing.snapshot_task(local_task, root / "mutation-workspace")

    def test_gdpval_strict_json_and_attachment_helpers(self) -> None:
        for rubric in ('{"key":1,"key":2}', "NaN", {"value": b"bytes"}):
            with self.subTest(rubric=repr(rubric)), self.assertRaises(RuntimeError):
                gdpval_module._parse_rubric_json(rubric)
        self.assertEqual(gdpval_module._strict_sequence("input.txt"), ("input.txt",))
        with self.assertRaises(SnapshotError):
            gdpval_module._strict_sequence((1,))
        self.assertEqual(gdpval_module._parse_snapshot_sequence(None, label="files"), ())
        self.assertEqual(gdpval_module._parse_snapshot_sequence("not-json", label="files"), ("not-json",))
        with self.assertRaises(SnapshotError):
            gdpval_module._parse_snapshot_sequence("1", label="files")
        with self.assertRaises(SnapshotError):
            gdpval_module._safe_attachment_name("")
        with self.assertRaises(SnapshotError):
            gdpval_module._safe_attachment_name("task_inputs/secret")
        self.assertEqual(gdpval_module._local_url_path("/tmp/input"), Path("/tmp/input"))
        self.assertIsNone(gdpval_module._local_url_path("relative/input"))
        self.assertIsNone(gdpval_module._local_url_path("https://example.invalid/input"))

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
            "invalid-utf8": b"\xff",
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

    def test_filesystem_failures_are_translated_and_partial_files_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.write_bytes(b"source")

            with patch.object(os.path, "abspath", side_effect=ValueError("invalid")):
                with self.assertRaises(SnapshotError):
                    snapshot_module._ensure_no_symlink_ancestors(source, label="source")
            with patch.object(Path, "is_symlink", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    snapshot_module._ensure_no_symlink_ancestors(source, label="source")
            with self.assertRaises(SnapshotError):
                snapshot_module._lstat_regular(root / "missing", label="source")
            with patch.object(os, "open", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    snapshot_module._stream_race_safe(source, label="source")

            real_close = os.close

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                raise OSError("close failed")

            with patch.object(os, "close", side_effect=close_then_fail):
                self.assertEqual(snapshot_module._stream_race_safe(source, label="source")[0], len(b"source"))
            with patch.object(os, "fsync", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    snapshot_module._fsync_directory(root)

            existing = root / "existing"
            existing.write_bytes(b"sentinel")
            with self.assertRaises(SnapshotError):
                snapshot_module._write_exclusive(existing, b"replacement", label="file")
            self.assertEqual(existing.read_bytes(), b"sentinel")
            with patch.object(os, "open", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    snapshot_module._write_exclusive(root / "denied", b"content", label="file")
            with (
                patch.object(os, "fdopen", side_effect=ValueError("invalid descriptor")),
                patch.object(os, "close", side_effect=close_then_fail),
                self.assertRaises(SnapshotError),
            ):
                snapshot_module._write_exclusive(root / "invalid-handle", b"content", label="file")

            partial = root / "partial"
            with self.assertRaises(SnapshotError):
                snapshot_module._copy_verified_file(
                    source,
                    partial,
                    expected_size=len(b"source"),
                    expected_sha256="1" * 64,
                    label="source",
                )
            self.assertFalse(partial.exists())
            with self.assertRaises(SnapshotError):
                snapshot_module._safe_mkdir(root / "existing", label="directory")
            with patch.object(Path, "mkdir", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    snapshot_module._safe_mkdir(root / "denied-directory", label="directory")

    def test_acquisition_metadata_and_publication_failures_never_publish_a_partial_snapshot(self) -> None:
        benchmark = _ProjectionBenchmark()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malformed_tasks = (
                BenchmarkTask(TaskSpec("", "prompt")),
                BenchmarkTask(TaskSpec("task", 1)),  # type: ignore[arg-type]
            )
            for index, task in enumerate(malformed_tasks):
                destination = root / f"task-{index}"
                with (
                    patch.object(benchmark, "load_tasks", return_value=(task,)),
                    self.assertRaises(SnapshotError),
                ):
                    acquire_snapshot(benchmark, 1, destination)
                self.assertFalse(destination.exists())

            metadata_cases: tuple[tuple[str, object], ...] = (
                ("name", ""),
                ("source", object()),
                ("revision", object()),
                ("source_availability", "sometimes"),
            )
            for attribute, value in metadata_cases:
                destination = root / f"metadata-{attribute}"
                with (
                    patch.object(benchmark, attribute, value),
                    self.assertRaises(SnapshotError),
                ):
                    acquire_snapshot(benchmark, 1, destination)
                self.assertFalse(destination.exists())

            with self.assertRaises(SnapshotError):
                acquire_snapshot(benchmark, 1, object())  # type: ignore[arg-type]
            with patch.object(Path, "mkdir", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    acquire_snapshot(benchmark, 1, root / "missing-parent" / "snapshot")

            after_stage = root / "changed-after-stage"
            with (
                patch.object(snapshot_module, "_source_fingerprint", side_effect=((), (), ("changed",))),
                self.assertRaises(SnapshotError),
            ):
                acquire_snapshot(benchmark, 1, after_stage)
            self.assertFalse(after_stage.exists())

            raced_destination = root / "raced"
            real_verify = snapshot_module.verify_snapshot

            def occupy_after_verification(snapshot: BenchmarkSnapshot | Path) -> BenchmarkSnapshot:
                verified = real_verify(snapshot)
                raced_destination.mkdir()
                (raced_destination / "sentinel").write_bytes(b"external")
                return verified

            with (
                patch.object(snapshot_module, "verify_snapshot", side_effect=occupy_after_verification),
                self.assertRaises(SnapshotError),
            ):
                acquire_snapshot(benchmark, 1, raced_destination)
            self.assertEqual((raced_destination / "sentinel").read_bytes(), b"external")

            rename_failure = root / "rename-failure"
            with patch.object(os, "rename", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    acquire_snapshot(benchmark, 1, rename_failure)
            self.assertFalse(rename_failure.exists())

    def test_sealed_root_and_projection_destinations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = acquire_snapshot(_ProjectionBenchmark(), 1, root / "sealed")
            external_manifest = root / "external-manifest"
            external_manifest.write_bytes((sealed.root / "benchmark-snapshot.json").read_bytes())
            external_directory = root / "external-directory"
            external_directory.mkdir()

            mutations = ("manifest-symlink", "blobs-file", "blobs-symlink", "blob-child-extra", "sha-symlink")
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
                    elif mutation == "blobs-symlink":
                        shutil.rmtree(candidate / "blobs")
                        (candidate / "blobs").symlink_to(external_directory, target_is_directory=True)
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
            with self.assertRaises(SnapshotError):
                verify_snapshot(root / "missing-root")
            with patch.object(Path, "iterdir", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    verify_snapshot(sealed)

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
            with self.assertRaises(SnapshotError):
                materialize_evaluation(sealed, "", root / "empty-evaluation-task")
            with self.assertRaises(SnapshotError):
                materialize_evaluation(sealed, "projection-task", object())  # type: ignore[arg-type]
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
            with self.assertRaises(SnapshotError):
                materialize_execution(sealed, "projection-task", object())  # type: ignore[arg-type]

            exact_collision = root / "exact-collision"
            (exact_collision / "task_inputs").mkdir(parents=True)
            with self.assertRaises(SnapshotError):
                materialize_execution(sealed, "projection-task", exact_collision)

    def test_projection_publication_races_leave_no_partial_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = acquire_snapshot(_ProjectionBenchmark(), 1, root / "sealed")

            evaluation_destination = root / "evaluation-race"
            real_fsync = snapshot_module._fsync_directory

            def race_evaluation(path: Path) -> None:
                real_fsync(path)
                if path.name.startswith(".evaluation-race.staging-"):
                    evaluation_destination.mkdir()

            with (
                patch.object(snapshot_module, "_fsync_directory", side_effect=race_evaluation),
                self.assertRaises(SnapshotError),
            ):
                materialize_evaluation(sealed, "projection-task", evaluation_destination)
            self.assertEqual(tuple(evaluation_destination.iterdir()), ())

            evaluation_rename = root / "evaluation-rename"
            with patch.object(os, "rename", side_effect=OSError("denied")):
                with self.assertRaises(SnapshotError):
                    materialize_evaluation(sealed, "projection-task", evaluation_rename)
            self.assertFalse(evaluation_rename.exists())

            workspace = root / "workspace-race"
            workspace.mkdir()
            task_inputs = workspace / "task_inputs"

            def race_execution(path: Path) -> None:
                real_fsync(path)
                if path.name.startswith(".task_inputs.staging-"):
                    task_inputs.mkdir()

            with (
                patch.object(snapshot_module, "_fsync_directory", side_effect=race_execution),
                self.assertRaises(SnapshotError),
            ):
                materialize_execution(sealed, "projection-task", workspace)
            self.assertEqual(tuple(task_inputs.iterdir()), ())

            dataset = root / "aime.jsonl"
            dataset.write_text(json.dumps({"question": "q", "expected_answer": "1"}) + "\n", encoding="utf-8")
            empty_snapshot = AIME26Benchmark(
                root=root,
                dataset_path=dataset,
                prepare_script=root / "prepare.py",
            ).acquire_snapshot(1, root / "empty-snapshot")
            self.assertEqual(materialize_execution(empty_snapshot, "aime26-01", root / "empty-workspace"), ())

    def test_evaluation_read_detects_tamper_after_initial_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = acquire_snapshot(_ProjectionBenchmark(), 1, root / "sealed")
            entry = sealed.tasks[0].evaluation_view.files[0]
            blob = sealed.root / "blobs" / "sha256" / entry.sha256
            real_verify = snapshot_module.verify_snapshot

            def tamper_after_verification(snapshot: BenchmarkSnapshot | Path) -> BenchmarkSnapshot:
                verified = real_verify(snapshot)
                blob.write_bytes(b"tampered")
                return verified

            with (
                patch.object(snapshot_module, "verify_snapshot", side_effect=tamper_after_verification),
                self.assertRaises(SnapshotError),
            ):
                read_evaluation_file(sealed, "projection-task", entry.path)

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
