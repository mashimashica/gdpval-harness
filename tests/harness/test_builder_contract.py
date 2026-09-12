# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import inspect
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Protocol, Sequence, cast

from eval_harness.builders import (
    Builder,
    BuilderInputBundle,
    BuilderInputManifest,
    BuilderPreflightResult,
    BuildFailurePhase,
    BuildRequest,
    BuildResult,
    BuildStatus,
    canonical_builder_input_manifest_bytes,
)
from eval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.interventions.base import (
    ApplicationMapping,
    InterventionBundle,
    InterventionFile,
    InterventionManifest,
    InterventionType,
)


_DIGEST = "a" * 64


class _BuildRequestConstructor(Protocol):
    def __call__(
        self,
        build_run_id: str,
        task: TaskSpec,
        inputs: Sequence[BuilderInputBundle],
        runtime_root: Path,
        artifact_root: Path,
        **unexpected: object,
    ) -> BuildRequest:
        raise NotImplementedError


class BuilderContractTests(unittest.TestCase):
    def file(self, path: str = "input.txt", content: bytes = b"input") -> InterventionFile:
        return InterventionFile(path, len(content), hashlib.sha256(content).hexdigest())

    def manifest(
        self,
        input_id: str = "input-1",
        files: tuple[InterventionFile, ...] | None = None,
        *,
        source_revision: str | None = None,
        revision_status: str = "unavailable",
        manifest_sha256: str | None = None,
    ) -> BuilderInputManifest:
        return BuilderInputManifest(
            input_id=input_id,
            input_type="files",
            source_revision=source_revision,
            revision_status=revision_status,
            files=files or (self.file(),),
            bundle_sha256=_DIGEST,
            manifest_sha256=manifest_sha256,
        )

    def bundle(self, manifest: BuilderInputManifest | None = None) -> BuilderInputBundle:
        return BuilderInputBundle(Path("/source"), manifest or self.manifest())

    def execution(
        self,
        task_id: str = "task-1",
        status: ExecutionStatus = ExecutionStatus.COMPLETED,
    ) -> ExecutionResult:
        return ExecutionResult(
            task_id=task_id,
            executor="fake",
            executor_version=None,
            invocation_mode="subscription",
            auth_mode="login",
            workspace=Path("/workspace"),
            deliverables_dir=Path("/workspace/deliverables"),
            status=status,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            exit_code=0 if status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE} else 1,
            available_outputs=frozenset(),
            failure=None
            if status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
            else Failure(FailureKind.PROCESS, "test_failure", FailureImpact.RUN),
        )

    def intervention_bundle(self) -> InterventionBundle:
        manifest = InterventionManifest(
            intervention_id="intervention-1",
            intervention_type=InterventionType.FILES,
            source_revision=None,
            revision_status="unavailable",
            files=(self.file("source.txt"),),
            bundle_sha256=_DIGEST,
            application=ApplicationMapping("workspace-files", "."),
        )
        return InterventionBundle(Path("/source"), manifest)

    def test_public_dataclass_shapes_are_exact_and_records_are_frozen(self) -> None:
        self.assertEqual(
            tuple(BuilderInputManifest.__dataclass_fields__),
            (
                "input_id",
                "input_type",
                "source_revision",
                "revision_status",
                "files",
                "bundle_sha256",
                "manifest_sha256",
            ),
        )
        self.assertEqual(tuple(BuilderInputBundle.__dataclass_fields__), ("root", "manifest"))
        self.assertEqual(
            tuple(BuildRequest.__dataclass_fields__),
            ("build_run_id", "task", "inputs", "runtime_root", "artifact_root", "model", "timeout_seconds"),
        )
        self.assertEqual(
            tuple(BuilderPreflightResult.__dataclass_fields__),
            (
                "name",
                "ok",
                "builder_executor",
                "builder_executor_version",
                "builder_executor_auth_mode",
                "details",
                "builder_executor_invocation_mode",
            ),
        )
        for forbidden in ("execution", "output_text", "metadata", "credentials", "session", "resume"):
            self.assertNotIn(forbidden, BuilderPreflightResult.__dataclass_fields__)
        self.assertEqual(
            tuple(BuildResult.__dataclass_fields__),
            ("build_run_id", "task_id", "builder", "status", "inputs", "execution", "bundle", "failure_phase"),
        )
        manifest = self.manifest()
        self.assertIsInstance(manifest.files, tuple)
        with self.assertRaises(FrozenInstanceError):
            setattr(manifest, "input_id", "changed")

    def test_manifest_hash_is_canonical_and_excludes_source_root_and_self(self) -> None:
        manifest = self.manifest()
        canonical = canonical_builder_input_manifest_bytes(manifest)
        self.assertEqual(manifest.manifest_sha256, hashlib.sha256(canonical).hexdigest())
        text = canonical.decode("utf-8")
        self.assertNotIn("manifest_sha256", text)
        self.assertNotIn("/source", text)
        self.assertNotIn("source", text.split('"files"', 1)[0])

        supplied = self.manifest(manifest_sha256=manifest.manifest_sha256)
        self.assertEqual(supplied, manifest)
        with self.assertRaises(ValueError):
            self.manifest(manifest_sha256=_DIGEST)

    def test_manifest_rejects_invalid_hash_paths_order_collisions_and_revision_fields(self) -> None:
        for digest in ("A" * 64, "g" * 64, "a" * 63):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                BuilderInputManifest("input", "files", None, "unavailable", (self.file(),), digest)

        for path in ("../escape.txt", "/absolute.txt", r"folder\file.txt", "folder//file.txt", "./file.txt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.manifest(files=(self.file(path),))

        with self.assertRaises(ValueError):
            self.manifest(files=(self.file("z.txt"), self.file("a.txt")))
        with self.assertRaises(ValueError):
            self.manifest(files=(self.file("A.txt"), self.file("a.txt")))
        with self.assertRaises(ValueError):
            self.manifest(files=(self.file("A/x"), self.file("a/y")))
        with self.assertRaises(ValueError):
            self.manifest(files=(self.file("a"), self.file("a/b")))
        with self.assertRaises(ValueError):
            self.manifest(files=(self.file("e\u0301.txt"),))

        with self.assertRaises(ValueError):
            self.manifest(source_revision="rev", revision_status="unavailable")
        with self.assertRaises(ValueError):
            self.manifest(source_revision=None, revision_status="available")
        with self.assertRaises(ValueError):
            self.manifest(source_revision=None, revision_status="unknown")
        with self.assertRaises(ValueError):
            self.manifest(source_revision="", revision_status="available")

        with self.assertRaises(ValueError):
            BuilderInputManifest("input", "files", None, "unavailable", (), _DIGEST)
        with self.assertRaises(TypeError):
            # Preserve the invalid runtime file record at the manifest boundary.
            BuilderInputManifest(
                "input",
                "files",
                None,
                "unavailable",
                cast(tuple[InterventionFile, ...], (object(),)),
                _DIGEST,
            )

    def test_request_normalizes_paths_and_inputs_and_rejects_bad_shape(self) -> None:
        manifest = self.manifest()
        request = BuildRequest(
            "build-1",
            TaskSpec("task-1", "prompt"),
            [self.bundle(manifest)],
            Path("/runtime"),
            Path("/artifacts"),
            timeout_seconds=1.5,
        )
        self.assertIsInstance(request.inputs, tuple)
        self.assertIsInstance(request.runtime_root, Path)
        self.assertIsInstance(request.artifact_root, Path)
        self.assertIs(request.task, request.task)

        with self.assertRaises(ValueError):
            BuildRequest("", TaskSpec("task-1", "prompt"), (), Path("/runtime"), Path("/artifacts"))
        with self.assertRaises(TypeError):
            # Preserve the invalid runtime task object at the request boundary.
            BuildRequest(
                "build-1",
                cast(TaskSpec, object()),
                (),
                Path("/runtime"),
                Path("/artifacts"),
            )
        with self.assertRaises(TypeError):
            BuildRequest(
                "build-1",
                TaskSpec("task-1", "prompt"),
                cast(tuple[BuilderInputBundle, ...], (object(),)),
                Path("/runtime"),
                Path("/artifacts"),
            )
        with self.assertRaises(ValueError):
            BuildRequest(
                "build-1",
                TaskSpec("task-1", "prompt"),
                (self.bundle(manifest), self.bundle(manifest)),
                Path("/runtime"),
                Path("/artifacts"),
            )
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                BuildRequest(
                    "build-1",
                    TaskSpec("task-1", "prompt"),
                    (),
                    Path("/runtime"),
                    Path("/artifacts"),
                    timeout_seconds=timeout,
                )
        with self.assertRaises(TypeError):
            BuildRequest(
                "build-1",
                TaskSpec("task-1", "prompt"),
                (),
                Path("/runtime"),
                Path("/artifacts"),
                # Preserve the invalid runtime timeout at the request boundary.
                timeout_seconds=cast(float, "10"),
            )
        with self.assertRaises(TypeError):
            # Exercise rejection of an unexpected constructor keyword.
            constructor = cast(_BuildRequestConstructor, BuildRequest)
            constructor(
                "build-1",
                TaskSpec("task-1", "prompt"),
                (),
                Path("/runtime"),
                Path("/artifacts"),
                source_root=Path("/forbidden"),
            )

    def test_preflight_requires_name_and_freezes_string_details(self) -> None:
        result = BuilderPreflightResult(
            "builder",
            True,
            # Preserve the runtime coercion fixture for non-string detail values.
            details=cast(tuple[str, ...], ["ready", 3]),
            builder_executor_invocation_mode="subscription",
        )
        self.assertEqual(result.details, ("ready", "3"))
        self.assertEqual(result.builder_executor_invocation_mode, "subscription")
        with self.assertRaises(ValueError):
            BuilderPreflightResult("", False)
        with self.assertRaises(ValueError):
            BuilderPreflightResult(" ", False)
        for invocation_mode in ("", " ", 42):
            with self.subTest(invocation_mode=invocation_mode), self.assertRaises(ValueError):
                BuilderPreflightResult(
                    "builder",
                    True,
                    # Preserve the invalid runtime invocation mode at the contract boundary.
                    builder_executor_invocation_mode=cast(str, invocation_mode),
                )

    def test_build_result_success_and_failure_invariants(self) -> None:
        manifest = self.manifest()
        execution = self.execution()
        intervention = self.intervention_bundle()
        success = BuildResult(
            "build-1", "task-1", "builder", BuildStatus.COMPLETED, [manifest], execution, intervention
        )
        self.assertEqual(success.status, BuildStatus.COMPLETED)
        self.assertIsInstance(success.inputs, tuple)
        self.assertTrue(success.executor_invoked)

        with self.assertRaises(ValueError):
            BuildResult("build-1", "task-1", "builder", BuildStatus.COMPLETED, [manifest])
        for execution_status in (ExecutionStatus.FAILED, ExecutionStatus.NO_DELIVERABLE):
            with self.subTest(execution_status=execution_status), self.assertRaises(ValueError):
                BuildResult(
                    "build-1",
                    "task-1",
                    "builder",
                    BuildStatus.COMPLETED,
                    [manifest],
                    self.execution(status=execution_status),
                    intervention,
                )
        with self.assertRaises(ValueError):
            BuildResult(
                "build-1",
                "task-1",
                "builder",
                BuildStatus.COMPLETED,
                [manifest],
                execution,
                intervention,
                BuildFailurePhase.EXECUTION,
            )
        with self.assertRaises(ValueError):
            BuildResult(
                "build-1",
                "task-1",
                "builder",
                BuildStatus.FAILED,
                [manifest],
                None,
                intervention,
                BuildFailurePhase.EXECUTION,
            )
        with self.assertRaises(ValueError):
            BuildResult("build-1", "task-1", "builder", BuildStatus.FAILED, [manifest])
        with self.assertRaises(ValueError):
            BuildResult(
                "build-1",
                "task-1",
                "builder",
                BuildStatus.NO_ARTIFACT,
                [manifest],
                None,
                None,
                BuildFailurePhase.ARTIFACT_VALIDATION,
            )
        with self.assertRaises(ValueError):
            BuildResult(
                "build-1",
                "task-1",
                "builder",
                BuildStatus.FAILED,
                [manifest],
                self.execution("other"),
                None,
                BuildFailurePhase.EXECUTION,
            )
        with self.assertRaises(TypeError):
            BuildResult(
                "build-1",
                "task-1",
                "builder",
                BuildStatus.FAILED,
                cast(tuple[BuilderInputManifest, ...], (object(),)),
                None,
                None,
                BuildFailurePhase.PREFLIGHT,
            )

    def test_phase_status_and_execution_mappings_fail_closed(self) -> None:
        manifest = self.manifest()
        intervention = self.intervention_bundle()
        invalid_cases = (
            ("preflight with execution", BuildStatus.FAILED, BuildFailurePhase.PREFLIGHT, self.execution()),
            (
                "input validation with execution",
                BuildStatus.FAILED,
                BuildFailurePhase.INPUT_VALIDATION,
                self.execution(),
            ),
            ("preflight timed out", BuildStatus.TIMED_OUT, BuildFailurePhase.PREFLIGHT, None),
            ("input validation interrupted", BuildStatus.INTERRUPTED, BuildFailurePhase.INPUT_VALIDATION, None),
            ("completed failed execution", BuildStatus.COMPLETED, None, self.execution(status=ExecutionStatus.FAILED)),
            (
                "completed no deliverable execution",
                BuildStatus.COMPLETED,
                None,
                self.execution(status=ExecutionStatus.NO_DELIVERABLE),
            ),
            (
                "timed out with failed execution",
                BuildStatus.TIMED_OUT,
                BuildFailurePhase.EXECUTION,
                self.execution(status=ExecutionStatus.FAILED),
            ),
            (
                "interrupted with completed execution",
                BuildStatus.INTERRUPTED,
                BuildFailurePhase.EXECUTION,
                self.execution(),
            ),
            (
                "no artifact with completed execution",
                BuildStatus.NO_ARTIFACT,
                BuildFailurePhase.ARTIFACT_VALIDATION,
                self.execution(),
            ),
            (
                "invalid artifact with no deliverable execution",
                BuildStatus.INVALID_ARTIFACT,
                BuildFailurePhase.ARTIFACT_VALIDATION,
                self.execution(status=ExecutionStatus.NO_DELIVERABLE),
            ),
            ("no artifact without execution", BuildStatus.NO_ARTIFACT, BuildFailurePhase.ARTIFACT_VALIDATION, None),
            (
                "invalid artifact without execution",
                BuildStatus.INVALID_ARTIFACT,
                BuildFailurePhase.ARTIFACT_VALIDATION,
                None,
            ),
            ("handoff without execution", BuildStatus.FAILED, BuildFailurePhase.ARTIFACT_HANDOFF, None),
            (
                "handoff with no deliverable execution",
                BuildStatus.FAILED,
                BuildFailurePhase.ARTIFACT_HANDOFF,
                self.execution(status=ExecutionStatus.NO_DELIVERABLE),
            ),
            (
                "handoff with wrong build status",
                BuildStatus.TIMED_OUT,
                BuildFailurePhase.ARTIFACT_HANDOFF,
                self.execution(),
            ),
        )
        for label, status, phase, execution in invalid_cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                BuildResult(
                    "build-1",
                    "task-1",
                    "builder",
                    status,
                    [manifest],
                    execution,
                    intervention if status is BuildStatus.COMPLETED else None,
                    phase,
                )

    def test_valid_execution_and_artifact_phase_mappings(self) -> None:
        manifest = self.manifest()
        valid_cases = (
            (BuildStatus.FAILED, BuildFailurePhase.EXECUTION, None),
            (BuildStatus.FAILED, BuildFailurePhase.EXECUTION, ExecutionStatus.COMPLETED),
            (BuildStatus.FAILED, BuildFailurePhase.EXECUTION, ExecutionStatus.FAILED),
            (BuildStatus.TIMED_OUT, BuildFailurePhase.EXECUTION, ExecutionStatus.TIMED_OUT),
            (BuildStatus.INTERRUPTED, BuildFailurePhase.EXECUTION, ExecutionStatus.INTERRUPTED),
            (BuildStatus.NO_ARTIFACT, BuildFailurePhase.ARTIFACT_VALIDATION, ExecutionStatus.NO_DELIVERABLE),
            (BuildStatus.INVALID_ARTIFACT, BuildFailurePhase.ARTIFACT_VALIDATION, ExecutionStatus.COMPLETED),
            (BuildStatus.FAILED, BuildFailurePhase.ARTIFACT_HANDOFF, ExecutionStatus.COMPLETED),
        )
        for status, phase, execution_status in valid_cases:
            with self.subTest(status=status, phase=phase, execution_status=execution_status):
                execution = None if execution_status is None else self.execution(status=execution_status)
                result = BuildResult("build-1", "task-1", "builder", status, [manifest], execution, None, phase)
                self.assertTrue(result.executor_invoked)

    def test_executor_invoked_distinguishes_preflight_input_validation_and_later_phases(self) -> None:
        for phase in (BuildFailurePhase.PREFLIGHT, BuildFailurePhase.INPUT_VALIDATION):
            result = BuildResult("build-1", "task-1", "builder", BuildStatus.FAILED, (), None, None, phase)
            self.assertFalse(result.executor_invoked)
        result = BuildResult(
            "build-1", "task-1", "builder", BuildStatus.FAILED, (), None, None, BuildFailurePhase.EXECUTION
        )
        self.assertTrue(result.executor_invoked)
        result = BuildResult(
            "build-1",
            "task-1",
            "builder",
            BuildStatus.NO_ARTIFACT,
            (),
            self.execution(status=ExecutionStatus.NO_DELIVERABLE),
            None,
            BuildFailurePhase.ARTIFACT_VALIDATION,
        )
        self.assertTrue(result.executor_invoked)
        result = BuildResult(
            "build-1",
            "task-1",
            "builder",
            BuildStatus.FAILED,
            (),
            self.execution(),
            None,
            BuildFailurePhase.ARTIFACT_HANDOFF,
        )
        self.assertTrue(result.executor_invoked)

    def test_task_spec_and_builder_abstract_signatures_remain_stable(self) -> None:
        self.assertEqual(set(TaskSpec.__dataclass_fields__), {"task_id", "prompt"})
        self.assertEqual(tuple(inspect.signature(Builder.preflight).parameters), ("self",))
        self.assertEqual(Builder.__abstractmethods__, frozenset({"preflight", "build"}))
        self.assertEqual(Builder.__annotations__["name"], "str")
        self.assertEqual(
            set(__import__("eval_harness.builders", fromlist=["__all__"]).__all__),
            {
                "ArtifactHandoffError",
                "BuildFailurePhase",
                "BuildRequest",
                "BuildResult",
                "BuildStatus",
                "Builder",
                "BuilderInputBundle",
                "BuilderInputManifest",
                "BuilderPreflightResult",
                "ExecutorSkillBuilder",
                "GeneratedSkillValidationError",
                "StagedBuilderInput",
                "canonical_builder_input_manifest_bytes",
                "build_skill_task",
                "load_builder_input_bundle",
                "seal_generated_skill",
                "stage_builder_inputs",
                "verify_staged_builder_inputs",
            },
        )


if __name__ == "__main__":
    unittest.main()
