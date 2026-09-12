# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import eval_harness.candidate_bundle as candidate_module
from eval_harness.benchmarks import snapshot as snapshot_module
from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.benchmarks.snapshot import (
    Availability,
    BenchmarkSnapshot,
    SnapshotError,
    SnapshotFile,
    SnapshotTask,
    SnapshotTaskContent,
    SnapshotView,
    VerifiedSnapshotAccess,
    acquire_snapshot,
    open_verified_snapshot,
)
from eval_harness.candidate_bundle import (
    BoundEvaluationAssets,
    BoundEvaluationView,
    BundleFile,
    CandidateBundle,
    CandidateBundleError,
    CandidateOutcome,
    ExecutorEvidence,
    InterventionEvidence,
    SnapshotReference,
    VerifiedSnapshotBinding,
    load_candidate_bundle,
    seal_candidate_bundle,
)
from eval_harness.capabilities import ExecutorCapabilities, ExecutorInput, ExecutorOutput
from eval_harness.executors.base import ExecutionStatus, TaskSpec
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.interventions.base import (
    ApplicationMapping,
    InterventionApplication,
    InterventionFile,
    InterventionManifest,
    InterventionType,
)
from eval_harness.interventions.none import NoneIntervention


_ZERO_DIGEST = "0" * 64


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class _FakeSnapshotAccess:
    def __init__(self, snapshot: BenchmarkSnapshot, blobs: dict[str, bytes]) -> None:
        self._snapshot = snapshot
        self._blobs = dict(blobs)
        self.reads: list[tuple[str, str, str]] = []
        self.materializations: list[tuple[str, tuple[str, ...]]] = []

    @property
    def snapshot(self) -> BenchmarkSnapshot:
        return self._snapshot

    def read_view_file(
        self,
        task_id: str,
        logical_path: str,
        *,
        view: Literal["execution", "evaluation"],
    ) -> bytes:
        task = self.snapshot.task(task_id)
        selected = task.execution_view if view == "execution" else task.evaluation_view
        entry = next(item for item in selected.files if item.path == logical_path)
        self.reads.append((task_id, logical_path, view))
        return self._blobs[entry.sha256]

    def materialize_view_subset(
        self,
        task_id: str,
        destination: Path,
        *,
        view: Literal["execution", "evaluation"],
        files: Sequence[SnapshotFile],
    ) -> tuple[str, ...]:
        del view
        if destination.exists():
            raise ValueError("destination must be fresh")
        destination.mkdir()
        paths: list[str] = []
        for entry in files:
            target = destination.joinpath(*entry.path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._blobs[entry.sha256])
            paths.append(entry.path)
        result = tuple(paths)
        self.materializations.append((task_id, result))
        return result

    def _reject_contained_destination(self, destination: Path) -> None:
        del destination

    def _matches_root(self, root: Path) -> bool:
        return root.is_dir() and not root.is_symlink()


class _SnapshotBenchmark(Benchmark):
    name = "candidate-fixture"

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        raise AssertionError("snapshot acquisition must not prepare")

    def load_tasks(self, limit: int) -> Sequence[BenchmarkTask]:
        del limit
        return (BenchmarkTask(TaskSpec("task-1", "canonical prompt")),)

    def materialize(self, task: BenchmarkTask, workspace: Path) -> Sequence[str]:
        del task, workspace
        raise AssertionError("snapshot_task owns fixture materialization")

    def snapshot_task(self, task: BenchmarkTask, workspace: Path) -> SnapshotTaskContent:
        del task, workspace
        return SnapshotTaskContent(
            evaluation_data={"rubric": {"answer": "secret", "weights": [1, 2]}},
            files=(("task_inputs/shared.txt", b"executor-visible"),),
            evaluation_files=(
                ("task_inputs/shared.txt", b"executor-visible"),
                ("task_inputs/grader-reference.txt", b"grader-secret"),
            ),
        )


class _EmptySnapshotBenchmark(_SnapshotBenchmark):
    name = "empty-candidate-fixture"

    def snapshot_task(self, task: BenchmarkTask, workspace: Path) -> SnapshotTaskContent:
        del task, workspace
        return SnapshotTaskContent(evaluation_data={})


def _snapshot_access() -> _FakeSnapshotAccess:
    shared = b"executor-visible"
    grader = b"grader-secret"
    shared_entry = SnapshotFile("task_inputs/shared.txt", len(shared), _digest(shared))
    grader_entry = SnapshotFile("grader/reference.txt", len(grader), _digest(grader))
    task = SnapshotTask(
        "task-1",
        "canonical prompt",
        SnapshotView({}, (shared_entry,)),
        SnapshotView(
            {"rubric": {"answer": "secret", "weights": [1, 2]}},
            (shared_entry, grader_entry),
        ),
    )
    snapshot = BenchmarkSnapshot(
        benchmark_id="fixture",
        source=None,
        source_availability=Availability.UNAVAILABLE,
        revision="fixture-v1",
        revision_availability=Availability.AVAILABLE,
        tasks=(task,),
    )
    return _FakeSnapshotAccess(snapshot, {shared_entry.sha256: shared, grader_entry.sha256: grader})


def _binding(access: _FakeSnapshotAccess) -> VerifiedSnapshotBinding:
    with patch.object(candidate_module, "open_verified_snapshot", return_value=access):
        return VerifiedSnapshotBinding.load(Path("ignored-snapshot-root"))


def _capabilities(*outputs: ExecutorOutput) -> ExecutorCapabilities:
    return ExecutorCapabilities(
        inputs=frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES}),
        outputs=frozenset(outputs),
    )


def _evidence(*outputs: ExecutorOutput) -> ExecutorEvidence:
    return ExecutorEvidence(
        executor_id="fixture-executor",
        executor_version="1.2.3",
        runtime="host-subprocess",
        invocation_mode="subprocess",
        auth_mode="local",
        requested_model="requested-model",
        model_id="resolved-model",
        reasoning_effort_requested="high",
        effective_reasoning_effort="medium",
        effective_reasoning_effort_available=True,
        declared_capabilities=_capabilities(*outputs),
        started_at="2026-09-12T00:00:00Z",
        finished_at="2026-09-12T00:00:01Z",
        exit_code=0,
    )


def _intervention_evidence(*, prompt: str = "effective prompt") -> InterventionEvidence:
    intervention = NoneIntervention(intervention_id="none-fixture")
    preflight = intervention.preflight()
    if preflight.bundle is None:
        raise AssertionError("fixture intervention preflight did not produce a bundle")
    application = intervention.apply(
        TaskSpec("task-1", prompt),
        Path("unused-workspace"),
        application_run_id="application-1",
    )
    return InterventionEvidence(preflight.bundle.manifest, application)


def _agent_skill_evidence(*, prompt: str = "effective prompt") -> InterventionEvidence:
    source = InterventionFile("SKILL.md", len(b"skill"), _digest(b"skill"))
    mapping = ApplicationMapping("workspace-reference", "interventions/demo/SKILL.md")
    manifest = InterventionManifest(
        intervention_id="demo-skill",
        intervention_type=InterventionType.AGENT_SKILL,
        source_revision="revision",
        revision_status="available",
        files=(source,),
        bundle_sha256=_digest(b"agent-skill-bundle"),
        application=mapping,
    )
    assert manifest.manifest_sha256 is not None
    application = InterventionApplication(
        application_run_id="application-skill",
        task=TaskSpec("task-1", prompt),
        materialized_files=(InterventionFile("interventions/demo/SKILL.md", source.size, source.sha256),),
        bundle_sha256=manifest.bundle_sha256,
        manifest_sha256=manifest.manifest_sha256,
        application=mapping,
    )
    return InterventionEvidence(manifest, application)


def _seal(
    destination: Path,
    binding: VerifiedSnapshotBinding,
    *,
    candidate_id: str = "candidate-1",
    output_text: str | None = "answer",
    outputs: frozenset[ExecutorOutput] = frozenset({ExecutorOutput.FINAL_TEXT}),
    artifacts_root: Path | None = None,
    evidence: ExecutorEvidence | None = None,
    intervention_evidence: InterventionEvidence | None = None,
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    failure: Failure | None = None,
    effective_executor_prompt: str = "effective prompt",
) -> CandidateBundle:
    return seal_candidate_bundle(
        destination=destination,
        candidate_id=candidate_id,
        snapshot_binding=binding,
        snapshot_reference=binding.reference("task-1"),
        effective_executor_prompt=effective_executor_prompt,
        executor_evidence=evidence or _evidence(*outputs),
        intervention_evidence=intervention_evidence or _intervention_evidence(prompt=effective_executor_prompt),
        status=status,
        output_text=output_text,
        available_outputs=outputs,
        failure=failure,
        artifacts_root=artifacts_root,
    )


def _rewrite_manifest(root: Path, update: dict[str, object]) -> None:
    path = root / "candidate-bundle.json"
    payload = cast(dict[str, object], json.loads(path.read_text()))
    payload.update(update)
    payload.pop("bundle_sha256", None)
    payload["bundle_sha256"] = _digest(_canonical(payload))
    path.write_bytes(_canonical(payload))


def _rewrite_intervention_target(root: Path, target: str) -> None:
    path = root / "candidate-bundle.json"
    payload = cast(dict[str, object], json.loads(path.read_text()))
    intervention = cast(dict[str, object], payload["intervention_evidence"])
    manifest = cast(dict[str, object], intervention["manifest"])
    application = cast(dict[str, object], intervention["application"])
    cast(dict[str, object], manifest["application"])["target"] = target
    cast(dict[str, object], application["application"])["target"] = target
    manifest_without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest_digest = _digest(_canonical(manifest_without_hash))
    manifest["manifest_sha256"] = manifest_digest
    application["manifest_sha256"] = manifest_digest
    payload.pop("bundle_sha256")
    payload["bundle_sha256"] = _digest(_canonical(payload))
    path.write_bytes(_canonical(payload))


class CandidateBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.access = _snapshot_access()
        self.binding = _binding(self.access)
        self.reference = self.binding.reference("task-1")

    def test_binding_builds_deep_frozen_scoped_evaluation_view(self) -> None:
        view = self.binding.bind_evaluation_view(self.reference)
        self.assertEqual(view.reference, self.reference)
        self.assertEqual(view.canonical_task_prompt, "canonical prompt")
        self.assertEqual(tuple(item.path for item in view.allowed_task_inputs.entries), ("task_inputs/shared.txt",))
        self.assertEqual(tuple(item.path for item in view.grader_assets.entries), ("grader/reference.txt",))
        self.assertEqual(view.allowed_task_inputs.read_bytes("task_inputs/shared.txt"), b"executor-visible")
        with self.assertRaises(CandidateBundleError):
            view.allowed_task_inputs.read_bytes("grader/reference.txt")
        self.assertEqual(view.grader_assets.read_bytes("grader/reference.txt"), b"grader-secret")

        allowed = self.root / "allowed"
        grader = self.root / "grader"
        self.assertEqual(view.allowed_task_inputs.materialize(allowed), ("task_inputs/shared.txt",))
        self.assertEqual(view.grader_assets.materialize(grader), ("grader/reference.txt",))
        self.assertTrue((allowed / "task_inputs/shared.txt").is_file())
        self.assertFalse((allowed / "grader/reference.txt").exists())
        self.assertTrue((grader / "grader/reference.txt").is_file())
        self.assertFalse((grader / "task_inputs/shared.txt").exists())
        self.assertFalse(hasattr(view.allowed_task_inputs, "root"))
        with self.assertRaises(TypeError):
            view.evaluation_data["new"] = True  # type: ignore[index]
        rubric = cast(dict[str, object], view.evaluation_data["rubric"])
        with self.assertRaises(TypeError):
            rubric["answer"] = "changed"
        execution = self.root / "execution"
        self.assertEqual(self.binding.materialize_execution(self.reference, execution), ("task_inputs/shared.txt",))
        self.assertTrue((execution / "task_inputs/shared.txt").is_file())
        with self.assertRaises(TypeError):
            self.binding.materialize_execution(self.reference, cast(Path, "workspace"))

    def test_binding_requires_exact_reference(self) -> None:
        with self.assertRaises(TypeError):
            VerifiedSnapshotBinding(cast(VerifiedSnapshotAccess, self.access), _token=object())
        with self.assertRaises(TypeError):
            BoundEvaluationAssets((), "task-1", cast(VerifiedSnapshotAccess, self.access), _token=object())
        with self.assertRaises(TypeError):
            VerifiedSnapshotBinding.load(cast(Path, "not-a-path"))
        with self.assertRaises(TypeError):
            self.binding.resolve(cast(SnapshotReference, object()))
        with self.assertRaises(CandidateBundleError):
            self.binding.reference("missing")
        with self.assertRaises(CandidateBundleError):
            self.binding.resolve(SnapshotReference(_ZERO_DIGEST, "task-1", self.reference.task_sha256))
        with self.assertRaises(CandidateBundleError):
            self.binding.resolve(SnapshotReference(self.reference.snapshot_sha256, "task-1", _ZERO_DIGEST))
        with self.assertRaises(CandidateBundleError):
            self.binding.resolve(SnapshotReference(self.reference.snapshot_sha256, "missing", _ZERO_DIGEST))

    def test_scalar_collection_and_bound_view_validation_fail_closed(self) -> None:
        with self.assertRaises(CandidateBundleError):
            SnapshotReference("bad", "task-1", _ZERO_DIGEST)
        with self.assertRaises(CandidateBundleError):
            SnapshotReference(_ZERO_DIGEST, "", _ZERO_DIGEST)
        canonical_values: tuple[object, ...] = ("\ud800", {"value": object()}, {"value": float("nan")})
        for value in canonical_values:
            with self.subTest(canonical=repr(value)), self.assertRaises(CandidateBundleError):
                candidate_module._canonical_bytes(value)
        with self.assertRaises(CandidateBundleError):
            candidate_module._finite_json_float("1e999")
        frozen_values: tuple[object, ...] = ("\ud800", float("nan"), {1: "value"}, object())
        for value in frozen_values:
            with self.subTest(frozen=repr(value)), self.assertRaises(CandidateBundleError):
                candidate_module._freeze_json(value, label="fixture")

        duplicate = BundleFile("same", 0, _digest(b""))
        for entries in (
            (duplicate, duplicate),
            (BundleFile("A/x", 0, _ZERO_DIGEST), BundleFile("a/x", 0, _ZERO_DIGEST)),
            (BundleFile("a", 0, _ZERO_DIGEST), BundleFile("a/b", 0, _ZERO_DIGEST)),
        ):
            with self.subTest(entries=entries), self.assertRaises(CandidateBundleError):
                CandidateOutcome(
                    ExecutionStatus.COMPLETED,
                    None,
                    frozenset({ExecutorOutput.ARTIFACT_FILES}),
                    entries,
                    None,
                )

        view = self.binding.bind_evaluation_view(self.reference)
        with self.assertRaises(CandidateBundleError):
            BoundEvaluationAssets._bind(
                (cast(SnapshotFile, object()),),
                "task-1",
                cast(VerifiedSnapshotAccess, self.access),
            )
        with self.assertRaises(TypeError):
            view.allowed_task_inputs.materialize(cast(Path, "destination"))
        invalid_views: tuple[dict[str, object], ...] = (
            {"reference": object()},
            {"canonical_task_prompt": cast(str, 1)},
            {"canonical_prompt_sha256": _ZERO_DIGEST},
            {"evaluation_data": cast(dict[str, object], [])},
            {"evaluation_data_sha256": _ZERO_DIGEST},
            {"allowed_task_inputs": object()},
            {"grader_assets": object()},
        )
        view_values: dict[str, object] = {
            "reference": view.reference,
            "canonical_task_prompt": view.canonical_task_prompt,
            "canonical_prompt_sha256": view.canonical_prompt_sha256,
            "evaluation_data": view.evaluation_data,
            "evaluation_data_sha256": view.evaluation_data_sha256,
            "evaluation_view_sha256": view.evaluation_view_sha256,
            "allowed_task_inputs": view.allowed_task_inputs,
            "grader_assets": view.grader_assets,
        }
        with self.assertRaises(TypeError):
            BoundEvaluationView(**view_values, _token=object())  # type: ignore[arg-type]
        for update in invalid_views:
            with self.subTest(update=update), self.assertRaises((CandidateBundleError, TypeError)):
                BoundEvaluationView._bind(**(view_values | update))  # type: ignore[arg-type]

        other_access = _snapshot_access()
        other_binding = _binding(other_access)
        other_assets = other_binding.bind_evaluation_view(other_binding.reference("task-1")).grader_assets
        with self.assertRaises(CandidateBundleError):
            BoundEvaluationView._bind(**(view_values | {"grader_assets": other_assets}))  # type: ignore[arg-type]

        malformed_access = _snapshot_access()
        malformed_task = malformed_access.snapshot.task("task-1")
        object.__setattr__(malformed_task.evaluation_view, "data", [])
        malformed_binding = _binding(malformed_access)
        with self.assertRaises(CandidateBundleError):
            malformed_binding.bind_evaluation_view(malformed_binding.reference("task-1"))

    def test_verified_snapshot_access_rechecks_only_selected_assets(self) -> None:
        snapshot_root = self.root / "snapshot"
        snapshot = acquire_snapshot(_SnapshotBenchmark(), 1, snapshot_root)
        with patch.object(snapshot_module, "verify_snapshot", wraps=snapshot_module.verify_snapshot) as verify:
            access = open_verified_snapshot(snapshot_root)
            self.assertEqual(verify.call_count, 1)
            self.assertEqual(access.snapshot.root, Path("."))
            self.assertFalse(hasattr(access, "root"))
            self.assertEqual(
                access.read_view_file("task-1", "task_inputs/shared.txt", view="execution"),
                b"executor-visible",
            )
            with self.assertRaises(SnapshotError):
                access.read_view_file("task-1", "task_inputs/grader-reference.txt", view="execution")
            grader_entry = snapshot.task("task-1").evaluation_view.files[0]
            if grader_entry.path != "task_inputs/grader-reference.txt":
                grader_entry = snapshot.task("task-1").evaluation_view.files[1]
            destination = self.root / "selected"
            self.assertEqual(
                access.materialize_view_subset(
                    "task-1",
                    destination,
                    view="evaluation",
                    files=(grader_entry,),
                ),
                ("task_inputs/grader-reference.txt",),
            )
            self.assertTrue((destination / "task_inputs/grader-reference.txt").is_file())
            self.assertFalse((destination / "task_inputs/shared.txt").exists())
            self.assertEqual(verify.call_count, 1)

            with self.assertRaises(SnapshotError):
                access.materialize_view_subset(
                    "task-1",
                    self.root / "wrong-entry",
                    view="evaluation",
                    files=(SnapshotFile(grader_entry.path, grader_entry.size, _ZERO_DIGEST),),
                )
            with self.assertRaises(SnapshotError):
                access.materialize_view_subset(
                    "task-1",
                    snapshot_root / "inside",
                    view="evaluation",
                    files=(),
                )

    def test_verified_snapshot_access_rejects_seal_and_target_tampering(self) -> None:
        snapshot_root = self.root / "snapshot"
        snapshot = acquire_snapshot(_SnapshotBenchmark(), 1, snapshot_root)
        access = open_verified_snapshot(snapshot_root)
        shared = next(
            item for item in snapshot.task("task-1").evaluation_view.files if item.path == "task_inputs/shared.txt"
        )
        (snapshot_root / "blobs/sha256" / shared.sha256).write_bytes(b"tampered")
        with self.assertRaises(SnapshotError):
            access.read_view_file("task-1", shared.path, view="evaluation")

        other_root = self.root / "other-snapshot"
        acquire_snapshot(_SnapshotBenchmark(), 1, other_root)
        other = open_verified_snapshot(other_root)
        (other_root / "unexpected").write_text("tamper")
        with self.assertRaisesRegex(SnapshotError, "seal"):
            other.materialize_view_subset("task-1", self.root / "unused", view="evaluation", files=())

        third_root = self.root / "third-snapshot"
        acquire_snapshot(_SnapshotBenchmark(), 1, third_root)
        third = open_verified_snapshot(third_root)
        with self.assertRaises(TypeError):
            VerifiedSnapshotAccess(third.snapshot, third_root, object(), _token=object())  # type: ignore[arg-type]
        with self.assertRaises(SnapshotError):
            third.read_view_file("missing", "x", view="evaluation")
        with self.assertRaises(SnapshotError):
            third.read_view_file("task-1", "x", view=cast(Literal["execution", "evaluation"], "invalid"))

        empty_root = self.root / "empty-snapshot"
        acquire_snapshot(_EmptySnapshotBenchmark(), 1, empty_root)
        empty_binding = VerifiedSnapshotBinding.load(empty_root)
        empty_reference = empty_binding.reference("task-1")
        (empty_root / "unexpected").write_text("tamper")
        with self.assertRaises(CandidateBundleError):
            empty_binding.resolve(empty_reference)
        with self.assertRaises(CandidateBundleError):
            empty_binding.bind_evaluation_view(empty_reference)

    def test_public_snapshot_binding_seals_and_loads_bundle(self) -> None:
        snapshot_root = self.root / "snapshot"
        acquire_snapshot(_SnapshotBenchmark(), 1, snapshot_root)
        binding = VerifiedSnapshotBinding.load(snapshot_root)
        bundle = _seal(self.root / "public-bundle", binding)
        self.assertEqual(load_candidate_bundle(bundle.root, snapshot_binding=binding), bundle)

    def test_output_presence_and_failure_invariants(self) -> None:
        absent = CandidateOutcome(ExecutionStatus.COMPLETED, None, frozenset(), (), None)
        empty_text = CandidateOutcome(
            ExecutionStatus.COMPLETED,
            "",
            frozenset({ExecutorOutput.FINAL_TEXT}),
            (),
            None,
        )
        empty_artifacts = CandidateOutcome(
            ExecutionStatus.NO_DELIVERABLE,
            None,
            frozenset({ExecutorOutput.ARTIFACT_FILES}),
            (),
            None,
        )
        self.assertIsNone(absent.output_text)
        self.assertEqual(empty_text.output_text, "")
        self.assertIn(ExecutorOutput.ARTIFACT_FILES, empty_artifacts.available_outputs)

        run_failure = Failure(FailureKind.PROCESS, "executor.failed", FailureImpact.RUN)
        task_failure = Failure(FailureKind.PROCESS, "executor.failed", FailureImpact.TASK)
        invalid: tuple[tuple[object, object, object, object, object], ...] = (
            (ExecutionStatus.COMPLETED, None, frozenset(), (), run_failure),
            (ExecutionStatus.FAILED, None, frozenset(), (), None),
            (ExecutionStatus.FAILED, None, frozenset(), (), task_failure),
            (ExecutionStatus.FAILED, "x", frozenset({ExecutorOutput.FINAL_TEXT}), (), run_failure),
            (ExecutionStatus.COMPLETED, None, frozenset({ExecutorOutput.FINAL_TEXT}), (), None),
            (ExecutionStatus.COMPLETED, "x", frozenset(), (), None),
            (ExecutionStatus.COMPLETED, None, frozenset(), (BundleFile("x", 0, _digest(b"")),), None),
            ("unknown", None, frozenset(), (), None),
            (ExecutionStatus.COMPLETED, 1, frozenset(), (), None),
            (ExecutionStatus.COMPLETED, None, frozenset({"unknown"}), (), None),
            (ExecutionStatus.COMPLETED, None, frozenset(), (object(),), None),
            (ExecutionStatus.COMPLETED, None, frozenset(), (), object()),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((CandidateBundleError, TypeError, ValueError)):
                CandidateOutcome(*args)  # type: ignore[arg-type]

        for status in (ExecutionStatus.FAILED, ExecutionStatus.TIMED_OUT, ExecutionStatus.INTERRUPTED):
            result = CandidateOutcome(status, None, frozenset(), (), run_failure)
            self.assertEqual(result.failure, run_failure)

    def test_record_validation_and_declared_channel_invariant(self) -> None:
        for path in ("", ".", "..", "/x", "C:/x", "a\\b", "a//b", "a/../b", "e\u0301.txt"):
            with self.subTest(path=path), self.assertRaises(CandidateBundleError):
                BundleFile(path, 0, _ZERO_DIGEST)
        with self.assertRaises(CandidateBundleError):
            BundleFile("x", True, _ZERO_DIGEST)
        with self.assertRaises(CandidateBundleError):
            BundleFile("x", 0, "invalid")

        evidence_values = _evidence(ExecutorOutput.FINAL_TEXT)
        invalid_evidence: tuple[dict[str, object], ...] = (
            {"executor_id": ""},
            {"executor_version": ""},
            {"runtime": ""},
            {"invocation_mode": ""},
            {"auth_mode": ""},
            {"requested_model": ""},
            {"model_id": ""},
            {"reasoning_effort_requested": "unsupported"},
            {"effective_reasoning_effort": "unsupported"},
            {"effective_reasoning_effort_available": 1},
            {"effective_reasoning_effort": "high", "effective_reasoning_effort_available": False},
            {"declared_capabilities": object()},
            {"started_at": ""},
            {"started_at": None},
            {"finished_at": ""},
            {"finished_at": None},
            {"exit_code": True},
        )
        base = {field: getattr(evidence_values, field) for field in evidence_values.__dataclass_fields__}
        for update in invalid_evidence:
            with self.subTest(update=update), self.assertRaises((CandidateBundleError, TypeError, ValueError)):
                ExecutorEvidence(**(base | update))

        unavailable = ExecutorEvidence(
            **(base | {"effective_reasoning_effort": None, "effective_reasoning_effort_available": False})
        )
        reported_default = ExecutorEvidence(
            **(base | {"effective_reasoning_effort": None, "effective_reasoning_effort_available": True})
        )
        self.assertFalse(unavailable.effective_reasoning_effort_available)
        self.assertTrue(reported_default.effective_reasoning_effort_available)

        outcome = CandidateOutcome(
            ExecutionStatus.COMPLETED,
            "answer",
            frozenset({ExecutorOutput.FINAL_TEXT}),
            (),
            None,
        )
        with self.assertRaises(CandidateBundleError):
            CandidateBundle._create(
                candidate_id="candidate",
                snapshot_reference=self.reference,
                canonical_task_prompt="canonical prompt",
                canonical_prompt_sha256=_digest(b"canonical prompt"),
                effective_executor_prompt="effective",
                effective_prompt_sha256=_digest(b"effective"),
                executor_evidence=_evidence(),
                intervention_evidence=_intervention_evidence(prompt="effective"),
                outcome=outcome,
            )

    def test_intervention_evidence_is_mandatory_bound_and_secret_free(self) -> None:
        evidence = _intervention_evidence()
        self.assertEqual(evidence.manifest.intervention_type.value, "none")
        self.assertEqual(evidence.manifest.files, ())
        self.assertEqual(evidence.application.materialized_files, ())
        self.assertEqual(evidence.application.bundle_sha256, evidence.manifest.bundle_sha256)
        bundle = _seal(self.root / "bundle", self.binding, intervention_evidence=evidence)
        raw = (bundle.root / "candidate-bundle.json").read_text()
        self.assertIn('"intervention_id":"none-fixture"', raw)
        self.assertIn('"application_run_id":"application-1"', raw)
        self.assertNotIn("unused-workspace", raw)
        self.assertNotIn("condition", raw)

        mismatched_hash = replace(evidence.application, bundle_sha256=_ZERO_DIGEST)
        with self.assertRaises(CandidateBundleError):
            InterventionEvidence(evidence.manifest, mismatched_hash)
        mismatched_mapping = replace(
            evidence.application,
            application=ApplicationMapping("workspace-files", "."),
        )
        with self.assertRaises(CandidateBundleError):
            InterventionEvidence(evidence.manifest, mismatched_mapping)
        with self.assertRaises(TypeError):
            InterventionEvidence(
                cast(InterventionManifest, object()),
                cast(InterventionApplication, object()),
            )

        payload = cast(dict[str, object], json.loads(raw))
        intervention_payload = cast(dict[str, object], payload["intervention_evidence"])
        application_payload = cast(dict[str, object], intervention_payload["application"])
        application_payload["task_id"] = "wrong-task"
        payload.pop("bundle_sha256")
        payload["bundle_sha256"] = _digest(_canonical(payload))
        (bundle.root / "candidate-bundle.json").write_bytes(_canonical(payload))
        with self.assertRaisesRegex(CandidateBundleError, "task"):
            load_candidate_bundle(bundle.root, snapshot_binding=self.binding)

    def test_intervention_evidence_rejects_every_unbound_or_inconsistent_field(self) -> None:
        def corrupted(record: object, **values: object) -> object:
            clone = copy.copy(record)
            for name, value in values.items():
                object.__setattr__(clone, name, value)
            return clone

        canonical_file = InterventionFile("skill/a.txt", 1, _digest(b"a"))
        later_file = InterventionFile("skill/z.txt", 1, _digest(b"z"))
        valid_files_manifest = InterventionManifest(
            intervention_id="files",
            intervention_type=InterventionType.FILES,
            source_revision="revision",
            revision_status="available",
            files=(canonical_file, later_file),
            bundle_sha256=_digest(b"bundle"),
            application=ApplicationMapping("workspace-files", "."),
        )
        assert valid_files_manifest.manifest_sha256 is not None
        valid_files_application = InterventionApplication(
            application_run_id="application",
            task=TaskSpec("task-1", "effective prompt"),
            materialized_files=(canonical_file, later_file),
            bundle_sha256=valid_files_manifest.bundle_sha256,
            manifest_sha256=valid_files_manifest.manifest_sha256,
            application=valid_files_manifest.application,
        )
        InterventionEvidence(valid_files_manifest, valid_files_application)

        bad_manifest_values: tuple[dict[str, object], ...] = (
            {"intervention_type": "unsupported"},
            {"intervention_id": ""},
            {"revision_status": "unsupported"},
            {"source_revision": "revision", "revision_status": "unavailable"},
            {"files": (later_file, canonical_file)},
            {"files": (object(),)},
            {"bundle_sha256": "bad"},
            {"manifest_sha256": _ZERO_DIGEST},
            {"application": object()},
        )
        for values in bad_manifest_values:
            manifest = cast(InterventionManifest, corrupted(valid_files_manifest, **values))
            with self.subTest(manifest=values), self.assertRaises((CandidateBundleError, TypeError)):
                InterventionEvidence(manifest, valid_files_application)

        bad_application_values: tuple[dict[str, object], ...] = (
            {"application_run_id": ""},
            {"task": object()},
            {"materialized_files": (later_file, canonical_file)},
            {"materialized_files": (object(),)},
            {"bundle_sha256": _ZERO_DIGEST},
            {"manifest_sha256": _ZERO_DIGEST},
            {"application": ApplicationMapping("prompt-overlay", None)},
            {"application": object()},
        )
        for values in bad_application_values:
            application = cast(InterventionApplication, corrupted(valid_files_application, **values))
            with self.subTest(application=values), self.assertRaises((CandidateBundleError, TypeError)):
                InterventionEvidence(valid_files_manifest, application)

        for files_materialized in (
            (canonical_file,),
            (canonical_file, InterventionFile(later_file.path, later_file.size, _ZERO_DIGEST)),
            (canonical_file, InterventionFile("skill/other.txt", later_file.size, later_file.sha256)),
        ):
            application = replace(valid_files_application, materialized_files=files_materialized)
            with self.subTest(materialized=files_materialized), self.assertRaises(CandidateBundleError):
                InterventionEvidence(valid_files_manifest, application)

        none_evidence = _intervention_evidence()
        with self.assertRaises(CandidateBundleError):
            InterventionEvidence(
                none_evidence.manifest,
                replace(none_evidence.application, materialized_files=(canonical_file,)),
            )

        overlay_mapping = ApplicationMapping("prompt-overlay", "task.prompt")
        overlay_manifest = InterventionManifest(
            intervention_id="overlay",
            intervention_type=InterventionType.PROMPT_OVERLAY,
            source_revision=None,
            revision_status="unavailable",
            files=(canonical_file,),
            bundle_sha256=_digest(b"overlay"),
            application=overlay_mapping,
        )
        assert overlay_manifest.manifest_sha256 is not None
        overlay_application = InterventionApplication(
            application_run_id="overlay-application",
            task=TaskSpec("task-1", "effective prompt"),
            materialized_files=(),
            bundle_sha256=overlay_manifest.bundle_sha256,
            manifest_sha256=overlay_manifest.manifest_sha256,
            application=overlay_mapping,
        )
        InterventionEvidence(overlay_manifest, overlay_application)
        with self.assertRaises(CandidateBundleError):
            InterventionEvidence(
                overlay_manifest,
                replace(overlay_application, materialized_files=(canonical_file,)),
            )

        skill_evidence = _agent_skill_evidence()
        self.assertEqual(
            tuple(item.path for item in skill_evidence.application.materialized_files),
            ("interventions/demo/SKILL.md",),
        )
        skill_file = skill_evidence.application.materialized_files[0]
        for skill_materialized in (
            (),
            (InterventionFile(skill_file.path, skill_file.size, _ZERO_DIGEST),),
            (InterventionFile("interventions/demo/OTHER.md", skill_file.size, skill_file.sha256),),
        ):
            with self.subTest(skill_materialized=skill_materialized), self.assertRaises(CandidateBundleError):
                InterventionEvidence(
                    skill_evidence.manifest,
                    replace(skill_evidence.application, materialized_files=skill_materialized),
                )
        readme = InterventionFile("README.md", 1, _digest(b"r"))
        no_skill_manifest = InterventionManifest(
            intervention_id="missing-skill-entry",
            intervention_type=InterventionType.AGENT_SKILL,
            source_revision="revision",
            revision_status="available",
            files=(readme,),
            bundle_sha256=_digest(b"missing-skill-entry"),
            application=skill_evidence.manifest.application,
        )
        assert no_skill_manifest.manifest_sha256 is not None
        no_skill_application = InterventionApplication(
            application_run_id="missing-skill-application",
            task=skill_evidence.application.task,
            materialized_files=(InterventionFile("interventions/demo/README.md", readme.size, readme.sha256),),
            bundle_sha256=no_skill_manifest.bundle_sha256,
            manifest_sha256=no_skill_manifest.manifest_sha256,
            application=no_skill_manifest.application,
        )
        with self.assertRaises(CandidateBundleError):
            InterventionEvidence(no_skill_manifest, no_skill_application)

        invalid_files: tuple[InterventionFile, ...] = (
            cast(InterventionFile, corrupted(canonical_file, path="../escape")),
            cast(InterventionFile, corrupted(canonical_file, size=-1)),
            cast(InterventionFile, corrupted(canonical_file, sha256="bad")),
        )
        for item in invalid_files:
            manifest = cast(InterventionManifest, corrupted(valid_files_manifest, files=(item,)))
            with self.subTest(item=item), self.assertRaises(CandidateBundleError):
                InterventionEvidence(manifest, valid_files_application)

    def test_persisted_intervention_target_must_be_safe_logical_path(self) -> None:
        original = _seal(
            self.root / "skill-bundle",
            self.binding,
            intervention_evidence=_agent_skill_evidence(),
        )
        self.assertEqual(load_candidate_bundle(original.root, snapshot_binding=self.binding), original)
        unsafe_targets = (
            "/runtime/skill/SKILL.md",
            "C:/runtime/skill/SKILL.md",
            "runtime\\skill\\SKILL.md",
            "runtime/../skill/SKILL.md",
            "runtime/skill\x00/SKILL.md",
            "interventions/e\u0301/SKILL.md",
            "interventions/demo/README.md",
        )
        for index, target in enumerate(unsafe_targets):
            root = self.root / f"unsafe-target-{index}"
            shutil.copytree(original.root, root)
            _rewrite_intervention_target(root, target)
            with self.subTest(target=target), self.assertRaises(CandidateBundleError):
                load_candidate_bundle(root, snapshot_binding=self.binding)

    def test_seal_load_relocate_and_preserve_empty_channels(self) -> None:
        artifacts = self.root / "empty-artifacts"
        artifacts.mkdir()
        bundle = _seal(
            self.root / "bundle",
            self.binding,
            output_text="",
            outputs=frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
            artifacts_root=artifacts,
        )
        self.assertEqual(bundle.outcome.output_text, "")
        self.assertEqual(bundle.outcome.artifacts, ())
        self.assertIn(ExecutorOutput.ARTIFACT_FILES, bundle.outcome.available_outputs)
        self.assertEqual(tuple((bundle.root / "blobs/sha256").iterdir()), ())
        present_empty = self.root / "present-empty-artifacts"
        self.assertEqual(bundle.materialize_artifacts(present_empty), ())
        self.assertTrue(present_empty.is_dir())
        loaded = load_candidate_bundle(bundle.root, snapshot_binding=self.binding)
        self.assertEqual(loaded, bundle)

        relocated = self.root / "relocated"
        shutil.copytree(bundle.root, relocated)
        moved = load_candidate_bundle(relocated, snapshot_binding=self.binding)
        self.assertEqual(moved, bundle)
        self.assertNotEqual(moved.root, bundle.root)
        self.assertEqual(moved.bundle_sha256, bundle.bundle_sha256)
        manifest = cast(dict[str, object], json.loads((relocated / "candidate-bundle.json").read_text()))
        self.assertNotIn("root", manifest)
        self.assertNotIn("snapshot_path", manifest)

    def test_artifacts_are_sorted_deduplicated_and_source_independent(self) -> None:
        source = self.root / "artifacts"
        (source / "nested").mkdir(parents=True)
        (source / "z.txt").write_bytes(b"same")
        (source / "nested/a.txt").write_bytes(b"same")
        (source / "empty.txt").write_bytes(b"")
        bundle = _seal(
            self.root / "bundle",
            self.binding,
            outputs=frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
            artifacts_root=source,
        )
        self.assertEqual(
            tuple(item.path for item in bundle.outcome.artifacts),
            ("empty.txt", "nested/a.txt", "z.txt"),
        )
        self.assertEqual(len(tuple((bundle.root / "blobs/sha256").iterdir())), 2)
        source.rename(self.root / "moved-source")
        loaded = load_candidate_bundle(bundle.root, snapshot_binding=self.binding)
        self.assertEqual(loaded.bundle_sha256, bundle.bundle_sha256)
        self.assertEqual(loaded.read_artifact("nested/a.txt"), b"same")
        with self.assertRaises(CandidateBundleError):
            loaded.read_artifact("missing")
        materialized = self.root / "materialized"
        self.assertEqual(
            loaded.materialize_artifacts(materialized),
            ("empty.txt", "nested/a.txt", "z.txt"),
        )
        self.assertEqual((materialized / "nested/a.txt").read_bytes(), b"same")
        with self.assertRaises(TypeError):
            loaded.materialize_artifacts(cast(Path, "destination"))
        with self.assertRaises(CandidateBundleError):
            loaded.materialize_artifacts(loaded.root / "inside")

        empty = _seal(
            self.root / "empty-bundle",
            self.binding,
            output_text=None,
            outputs=frozenset(),
        )
        absent = self.root / "absent-artifacts"
        with self.assertRaisesRegex(CandidateBundleError, "channel is absent"):
            empty.materialize_artifacts(absent)
        self.assertFalse(absent.exists())

    def test_artifact_source_rejects_unsafe_colliding_and_special_nodes(self) -> None:
        cases: list[Path] = []
        casefold = self.root / "casefold"
        (casefold / "A").mkdir(parents=True)
        (casefold / "a").mkdir()
        (casefold / "A/x").write_text("x")
        (casefold / "a/y").write_text("y")
        cases.append(casefold)
        prefix = self.root / "prefix"
        prefix.mkdir()
        (prefix / "Foo").write_text("file")
        (prefix / "foo").mkdir()
        (prefix / "foo/x").write_text("nested")
        cases.append(prefix)
        decomposed = self.root / "decomposed"
        decomposed.mkdir()
        (decomposed / "e\u0301.txt").write_text("x")
        cases.append(decomposed)
        backslash = self.root / "backslash"
        backslash.mkdir()
        (backslash / "a\\b").write_text("x")
        cases.append(backslash)
        linked = self.root / "linked"
        linked.mkdir()
        (linked / "target").write_text("x")
        (linked / "alias").symlink_to("target")
        cases.append(linked)
        fifo = self.root / "fifo"
        fifo.mkdir()
        os.mkfifo(fifo / "pipe")
        cases.append(fifo)

        for index, source in enumerate(cases):
            with self.subTest(source=source.name), self.assertRaises(CandidateBundleError):
                _seal(
                    self.root / f"rejected-{index}",
                    self.binding,
                    outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
                    output_text=None,
                    artifacts_root=source,
                )

    def test_seal_detects_mutation_and_never_replaces_destination(self) -> None:
        source = self.root / "source"
        source.mkdir()
        artifact = source / "answer.txt"
        artifact.write_bytes(b"before")
        destination = self.root / "bundle"
        original_copy = candidate_module._copy_source_file

        def mutating_copy(path: Path, temporary: Path) -> tuple[int, str]:
            result = original_copy(path, temporary)
            path.write_bytes(b"after")
            return result

        with patch.object(candidate_module, "_copy_source_file", side_effect=mutating_copy):
            with self.assertRaisesRegex(CandidateBundleError, "changed"):
                _seal(
                    destination,
                    self.binding,
                    outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
                    output_text=None,
                    artifacts_root=source,
                )
        self.assertFalse(destination.exists())

        destination.mkdir()
        marker = destination / "keep"
        marker.write_text("untouched")
        with self.assertRaises(CandidateBundleError):
            _seal(destination, self.binding)
        self.assertEqual(marker.read_text(), "untouched")

    def test_seal_rejects_artifact_and_snapshot_destination_overlap_before_staging(self) -> None:
        source = self.root / "source"
        source.mkdir()
        (source / "artifact").write_text("x")
        inside_source = source / "candidate"
        with self.assertRaisesRegex(CandidateBundleError, "overlaps"):
            _seal(
                inside_source,
                self.binding,
                output_text=None,
                outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
                artifacts_root=source,
            )
        self.assertFalse(inside_source.exists())
        self.assertFalse(any(path.name.startswith(".candidate.staging-") for path in source.iterdir()))

        snapshot_root = self.root / "snapshot"
        acquire_snapshot(_SnapshotBenchmark(), 1, snapshot_root)
        binding = VerifiedSnapshotBinding.load(snapshot_root)
        reference = binding.reference("task-1")
        inside_snapshot = snapshot_root / "candidate"
        with self.assertRaises(CandidateBundleError):
            _seal(inside_snapshot, binding)
        self.assertFalse(inside_snapshot.exists())
        self.assertEqual(binding.reference("task-1"), reference)

    def test_seal_channel_and_source_contracts_fail_closed(self) -> None:
        directory = self.root / "artifacts"
        directory.mkdir()
        regular = self.root / "regular"
        regular.write_text("x")
        linked = self.root / "linked-root"
        linked.symlink_to(directory, target_is_directory=True)
        cases: tuple[tuple[frozenset[ExecutorOutput], Path | None], ...] = (
            (frozenset(), directory),
            (frozenset({ExecutorOutput.ARTIFACT_FILES}), None),
            (frozenset({ExecutorOutput.ARTIFACT_FILES}), regular),
            (frozenset({ExecutorOutput.ARTIFACT_FILES}), linked),
        )
        for index, (outputs, source) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(CandidateBundleError):
                _seal(
                    self.root / f"bad-{index}",
                    self.binding,
                    output_text=None,
                    outputs=outputs,
                    artifacts_root=source,
                )
        with self.assertRaises(TypeError):
            seal_candidate_bundle(
                destination=cast(Path, "bad"),
                candidate_id="candidate",
                snapshot_binding=self.binding,
                snapshot_reference=self.reference,
                effective_executor_prompt="prompt",
                executor_evidence=_evidence(),
                intervention_evidence=_intervention_evidence(),
                status=ExecutionStatus.COMPLETED,
                output_text=None,
                available_outputs=frozenset(),
                failure=None,
                artifacts_root=None,
            )

    def test_filesystem_failures_are_normalized_and_partial_staging_is_removed(self) -> None:
        regular = self.root / "regular-file"
        regular.write_bytes(b"content")
        with patch.object(os.path, "abspath", side_effect=OSError("hidden")):
            with self.assertRaises(CandidateBundleError):
                candidate_module._ensure_no_symlink_ancestors(regular, label="fixture")
        with self.assertRaises(CandidateBundleError):
            candidate_module._lstat_regular(self.root / "missing", label="fixture")
        with self.assertRaises(CandidateBundleError):
            candidate_module._lstat_regular(self.root, label="fixture")
        with patch.object(os, "open", side_effect=OSError("hidden")):
            with self.assertRaises(CandidateBundleError):
                candidate_module._read_race_safe(regular, label="fixture")
        with patch.object(candidate_module, "_same_file_stat", return_value=False):
            with self.assertRaisesRegex(CandidateBundleError, "changed"):
                candidate_module._read_race_safe(regular, label="fixture")
        with patch.object(candidate_module, "_same_file_stat", side_effect=(True, False)):
            with self.assertRaisesRegex(CandidateBundleError, "changed"):
                candidate_module._read_race_safe(regular, label="fixture")
        with patch.object(os, "read", return_value=b""):
            with self.assertRaisesRegex(CandidateBundleError, "changed"):
                candidate_module._read_race_safe(regular, label="fixture")
        with patch.object(os, "fsync", side_effect=OSError("hidden")):
            with self.assertRaises(CandidateBundleError):
                candidate_module._fsync_directory(self.root)

        existing = self.root / "existing"
        existing.write_bytes(b"keep")
        with self.assertRaises(CandidateBundleError):
            candidate_module._write_exclusive(existing, b"new", label="fixture")
        target = self.root / "write-failure"
        with patch.object(os, "open", side_effect=OSError("hidden")):
            with self.assertRaises(CandidateBundleError):
                candidate_module._write_exclusive(target, b"new", label="fixture")

        missing_artifacts = self.root / "missing-artifacts"
        with self.assertRaises(CandidateBundleError):
            _seal(
                self.root / "missing-artifact-bundle",
                self.binding,
                output_text=None,
                outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
                artifacts_root=missing_artifacts,
            )
        source = self.root / "source-errors"
        source.mkdir()
        (source / "artifact").write_text("x")
        with patch.object(os, "scandir", side_effect=OSError("hidden")):
            with self.assertRaises(CandidateBundleError):
                _seal(
                    self.root / "scan-error",
                    self.binding,
                    output_text=None,
                    outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
                    artifacts_root=source,
                )
        with patch.object(os, "rename", side_effect=OSError("hidden")):
            with self.assertRaises(CandidateBundleError):
                _seal(self.root / "publish-error", self.binding)
        self.assertFalse(any(path.name.startswith(".publish-error.staging-") for path in self.root.iterdir()))

    def test_seal_rejects_argument_mismatch_publish_race_and_cas_collision(self) -> None:
        common: dict[str, object] = {
            "destination": self.root / "typed",
            "candidate_id": "candidate",
            "snapshot_binding": self.binding,
            "snapshot_reference": self.reference,
            "effective_executor_prompt": "effective prompt",
            "executor_evidence": _evidence(),
            "intervention_evidence": _intervention_evidence(),
            "status": ExecutionStatus.COMPLETED,
            "output_text": None,
            "available_outputs": frozenset(),
            "failure": None,
            "artifacts_root": None,
        }
        invalid: tuple[dict[str, object], ...] = (
            {"snapshot_binding": object()},
            {"snapshot_reference": object()},
            {"executor_evidence": object()},
            {"intervention_evidence": object()},
            {"available_outputs": frozenset({"unknown"})},
            {"artifacts_root": "path"},
            {"effective_executor_prompt": None},
        )
        for index, update in enumerate(invalid):
            arguments = common | update | {"destination": self.root / f"typed-{index}"}
            with self.subTest(update=update), self.assertRaises((CandidateBundleError, TypeError)):
                seal_candidate_bundle(**arguments)  # type: ignore[arg-type]

        destination = self.root / "raced"
        original_write = candidate_module._write_exclusive

        def race_after_manifest(path: Path, content: bytes, *, label: str) -> None:
            original_write(path, content, label=label)
            destination.mkdir()

        with patch.object(candidate_module, "_write_exclusive", side_effect=race_after_manifest):
            with self.assertRaises(CandidateBundleError):
                _seal(destination, self.binding)
        self.assertTrue(destination.is_dir())

        source = self.root / "collision-source"
        source.mkdir()
        (source / "a").write_bytes(b"a")
        (source / "b").write_bytes(b"b")

        def collide(path: Path, temporary: Path) -> tuple[int, str]:
            temporary.write_bytes(path.read_bytes())
            return 1, _ZERO_DIGEST

        with patch.object(candidate_module, "_copy_source_file", side_effect=collide):
            with self.assertRaisesRegex(CandidateBundleError, "collision"):
                _seal(
                    self.root / "collision-bundle",
                    self.binding,
                    output_text=None,
                    outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
                    artifacts_root=source,
                )

    def test_load_rejects_prompt_manifest_and_blob_tampering(self) -> None:
        source = self.root / "artifacts"
        source.mkdir()
        (source / "answer.txt").write_bytes(b"answer")
        original = _seal(
            self.root / "original",
            self.binding,
            outputs=frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
            artifacts_root=source,
        )
        digest = original.outcome.artifacts[0].sha256

        prompt = self.root / "prompt"
        shutil.copytree(original.root, prompt)
        _rewrite_manifest(
            prompt,
            {
                "canonical_task_prompt": "different prompt",
                "canonical_prompt_sha256": _digest(b"different prompt"),
            },
        )
        with self.assertRaisesRegex(CandidateBundleError, "snapshot task"):
            load_candidate_bundle(prompt, snapshot_binding=self.binding)

        transformations: tuple[tuple[str, Callable[[Path], object]], ...] = (
            ("missing", lambda root: (root / "blobs/sha256" / digest).unlink()),
            ("changed", lambda root: (root / "blobs/sha256" / digest).write_bytes(b"changed")),
            ("extra-root", lambda root: (root / "extra").write_text("x")),
            ("extra-blob", lambda root: (root / "blobs/sha256" / _ZERO_DIGEST).write_bytes(b"")),
        )
        for name, transformation in transformations:
            root = self.root / name
            shutil.copytree(original.root, root)
            transformation(root)
            with self.subTest(name=name), self.assertRaises(CandidateBundleError):
                load_candidate_bundle(root, snapshot_binding=self.binding)

        linked_blob = self.root / "linked-blob"
        shutil.copytree(original.root, linked_blob)
        blob = linked_blob / "blobs/sha256" / digest
        blob.unlink()
        blob.symlink_to(source / "answer.txt")
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(linked_blob, snapshot_binding=self.binding)

        access_tamper = self.root / "access-tamper"
        shutil.copytree(original.root, access_tamper)
        verified = load_candidate_bundle(access_tamper, snapshot_binding=self.binding)
        (access_tamper / "extra").write_text("late")
        with self.assertRaises(CandidateBundleError):
            verified.read_artifact("answer.txt")

        materialize_tamper = self.root / "materialize-tamper"
        shutil.copytree(original.root, materialize_tamper)
        verified = load_candidate_bundle(materialize_tamper, snapshot_binding=self.binding)
        (materialize_tamper / "blobs/sha256" / digest).write_bytes(b"late")
        with self.assertRaises(CandidateBundleError):
            verified.materialize_artifacts(self.root / "tampered-output")
        self.assertFalse((self.root / "tampered-output").exists())

    def test_load_rejects_noncanonical_duplicate_nonfinite_and_unknown_json(self) -> None:
        original = _seal(self.root / "original", self.binding)
        raw = (original.root / "candidate-bundle.json").read_bytes()
        invalid: tuple[tuple[str, bytes], ...] = (
            ("newline", raw + b"\n"),
            ("duplicate", b'{"schema":"candidate-bundle",' + raw[1:]),
            ("nonfinite", b'{"value":NaN}'),
            ("unknown", raw[:-1] + b',"unknown":true}'),
            ("array", b"[]"),
            ("invalid-json", b"{"),
            ("invalid-utf8", b"\xff"),
        )
        for name, content in invalid:
            root = self.root / name
            shutil.copytree(original.root, root)
            (root / "candidate-bundle.json").write_bytes(content)
            with self.subTest(name=name), self.assertRaises(CandidateBundleError):
                load_candidate_bundle(root, snapshot_binding=self.binding)

        wrong_version = self.root / "wrong-version"
        shutil.copytree(original.root, wrong_version)
        payload = cast(dict[str, object], json.loads(raw))
        payload["schema_version"] = True
        (wrong_version / "candidate-bundle.json").write_bytes(_canonical(payload))
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(wrong_version, snapshot_binding=self.binding)

    def test_load_rejects_malformed_nested_records_without_fallback(self) -> None:
        original = _seal(self.root / "original", self.binding)
        manifest_path = original.root / "candidate-bundle.json"
        original_payload = cast(dict[str, object], json.loads(manifest_path.read_bytes()))

        def rejected(path: tuple[str, ...], value: object) -> None:
            payload = copy.deepcopy(original_payload)
            cursor = payload
            for component in path[:-1]:
                cursor = cast(dict[str, object], cursor[component])
            cursor[path[-1]] = value
            payload.pop("bundle_sha256", None)
            payload["bundle_sha256"] = _digest(_canonical(payload))
            manifest_path.write_bytes(_canonical(payload))
            with self.assertRaises((CandidateBundleError, TypeError, ValueError)):
                load_candidate_bundle(original.root, snapshot_binding=self.binding)

        malformed: tuple[tuple[tuple[str, ...], object], ...] = (
            (("snapshot_reference",), []),
            (("executor_evidence",), []),
            (("executor_evidence", "declared_capabilities"), []),
            (("executor_evidence", "declared_capabilities", "inputs"), {}),
            (("executor_evidence", "declared_capabilities", "outputs"), ["unknown"]),
            (("executor_evidence", "started_at"), None),
            (("executor_evidence", "finished_at"), None),
            (("intervention_evidence",), []),
            (("intervention_evidence", "manifest"), []),
            (("intervention_evidence", "application"), []),
            (("intervention_evidence", "manifest", "files"), {}),
            (("intervention_evidence", "manifest", "files"), [None]),
            (
                ("intervention_evidence", "manifest", "files"),
                [{"path": "x", "size": True, "sha256": _ZERO_DIGEST}],
            ),
            (("intervention_evidence", "manifest", "application"), []),
            (
                ("intervention_evidence", "manifest", "application"),
                {"method": "", "target": None},
            ),
            (("intervention_evidence", "application", "application"), []),
            (("intervention_evidence", "manifest", "intervention_type"), "unknown"),
            (("outcome",), []),
            (("outcome", "available_outputs"), {}),
            (("outcome", "available_outputs"), ["unknown"]),
            (("outcome", "artifacts"), [None]),
            (("outcome", "failure"), []),
            (("outcome", "failure"), {"kind": "unknown", "code": "bad", "impact": "run"}),
        )
        for path, value in malformed:
            with self.subTest(path=path, value=value):
                rejected(path, value)

    def test_load_rejects_inconsistent_artifact_sizes_and_invalid_roots(self) -> None:
        source = self.root / "source"
        source.mkdir()
        (source / "a").write_bytes(b"a")
        (source / "b").write_bytes(b"bb")
        bundle = _seal(
            self.root / "bundle",
            self.binding,
            output_text=None,
            outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
            artifacts_root=source,
        )
        payload = cast(dict[str, object], json.loads((bundle.root / "candidate-bundle.json").read_bytes()))
        outcome = cast(dict[str, object], payload["outcome"])
        artifacts = cast(list[dict[str, object]], outcome["artifacts"])
        artifacts[1]["sha256"] = artifacts[0]["sha256"]
        payload.pop("bundle_sha256")
        payload["bundle_sha256"] = _digest(_canonical(payload))
        (bundle.root / "candidate-bundle.json").write_bytes(_canonical(payload))
        with self.assertRaisesRegex(CandidateBundleError, "inconsistent sizes"):
            load_candidate_bundle(bundle.root, snapshot_binding=self.binding)

        missing = self.root / "missing"
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(missing, snapshot_binding=self.binding)
        regular = self.root / "regular"
        regular.write_text("x")
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(regular, snapshot_binding=self.binding)

        original = _seal(self.root / "other", self.binding)
        external = self.root / "external-blobs"
        shutil.move(original.root / "blobs", external)
        (original.root / "blobs").symlink_to(external, target_is_directory=True)
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(original.root, snapshot_binding=self.binding)

        extra = _seal(self.root / "extra-under-blobs", self.binding)
        (extra.root / "blobs/unexpected").write_text("x")
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(extra.root, snapshot_binding=self.binding)

    def test_load_requires_explicit_matching_binding_and_nonsymlink_root(self) -> None:
        bundle = _seal(self.root / "bundle", self.binding)
        other = _snapshot_access()
        other_snapshot = BenchmarkSnapshot(
            benchmark_id="other",
            source=None,
            source_availability=Availability.UNAVAILABLE,
            revision=None,
            revision_availability=Availability.UNAVAILABLE,
            tasks=other.snapshot.tasks,
        )
        wrong_binding = _binding(_FakeSnapshotAccess(other_snapshot, other._blobs))
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(bundle.root, snapshot_binding=wrong_binding)
        with self.assertRaises(TypeError):
            load_candidate_bundle(bundle.root, snapshot_binding=cast(VerifiedSnapshotBinding, None))
        with self.assertRaises(TypeError):
            load_candidate_bundle(cast(Path, "bundle"), snapshot_binding=self.binding)

        linked = self.root / "linked"
        linked.symlink_to(bundle.root, target_is_directory=True)
        with self.assertRaises(CandidateBundleError):
            load_candidate_bundle(linked, snapshot_binding=self.binding)

    def test_bundle_identity_excludes_root_and_validates_prompt_hashes(self) -> None:
        outcome = CandidateOutcome(ExecutionStatus.COMPLETED, None, frozenset(), (), None)
        values = {
            "candidate_id": "candidate",
            "snapshot_reference": self.reference,
            "canonical_task_prompt": "canonical prompt",
            "canonical_prompt_sha256": _digest(b"canonical prompt"),
            "effective_executor_prompt": "",
            "effective_prompt_sha256": _digest(b""),
            "executor_evidence": _evidence(),
            "intervention_evidence": _intervention_evidence(prompt=""),
            "outcome": outcome,
        }
        with self.assertRaises(TypeError):
            CandidateBundle(**values, root=Path("blocked"))  # type: ignore[arg-type]
        first = CandidateBundle._create(**values, root=Path("one"))  # type: ignore[arg-type]
        second = CandidateBundle._create(**values, root=Path("two"))  # type: ignore[arg-type]
        self.assertEqual(first, second)
        self.assertEqual(first.bundle_sha256, second.bundle_sha256)
        with self.assertRaises(CandidateBundleError):
            CandidateBundle._create(**(values | {"canonical_prompt_sha256": _ZERO_DIGEST}))  # type: ignore[arg-type]
        with self.assertRaises(CandidateBundleError):
            CandidateBundle._create(**(values | {"effective_prompt_sha256": _ZERO_DIGEST}))  # type: ignore[arg-type]
        with self.assertRaises(CandidateBundleError):
            CandidateBundle._create(**values, bundle_sha256=_ZERO_DIGEST)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
