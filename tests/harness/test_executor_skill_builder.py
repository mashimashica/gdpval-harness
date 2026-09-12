# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

from eval_harness.builders import BuildFailurePhase, BuildRequest, BuildStatus
from eval_harness.builders.artifact import ArtifactHandoffError
from eval_harness.builders.executor_skill import ExecutorSkillBuilder
from eval_harness.builders.inputs import load_builder_input_bundle
from eval_harness.capabilities import ExecutorOutput
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from eval_harness.failures import Failure, FailureImpact, FailureKind
from eval_harness.interventions import AgentSkillIntervention
from eval_harness.reasoning import ReasoningEffortOption


class DeterministicExecutor(Executor):
    name = "deterministic"
    runtime = "test"
    invocation_mode = "deterministic-test"
    network_access_enabled: bool = False
    reasoning_effort: ReasoningEffortOption = None

    def __init__(
        self,
        *,
        preflight_ok: bool = True,
        preflight_executor: str | None = None,
        preflight_exception: Exception | None = None,
        status: ExecutionStatus = ExecutionStatus.COMPLETED,
        result_task_id: str | None = None,
        result_executor: str | None = None,
        result_workspace: Path | None = None,
        result_deliverables: Path | None = None,
        invalid_skill: bool = False,
        tamper_inputs: bool = False,
        raise_error: bool = False,
    ) -> None:
        self.preflight_ok = preflight_ok
        self.preflight_executor = preflight_executor
        self.preflight_exception = preflight_exception
        self.status = status
        self.result_task_id = result_task_id
        self.result_executor = result_executor
        self.result_workspace = result_workspace
        self.result_deliverables = result_deliverables
        self.invalid_skill = invalid_skill
        self.tamper_inputs = tamper_inputs
        self.raise_error = raise_error
        self.preflight_calls = 0
        self.execute_calls = 0
        self.last_request: ExecutionRequest | None = None

    def preflight(self) -> PreflightResult:
        self.preflight_calls += 1
        if self.preflight_exception is not None:
            raise self.preflight_exception
        return PreflightResult(
            executor=self.preflight_executor or self.name,
            ok=self.preflight_ok,
            version="deterministic-1",
            auth_mode="test-login",
            details=("deterministic preflight",),
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.execute_calls += 1
        self.last_request = request
        (request.executor_dir / "stdout.log").write_text("builder stdout", encoding="utf-8")
        (request.executor_dir / "transcript.txt").write_text("builder transcript", encoding="utf-8")
        if self.tamper_inputs:
            staged_file = request.workspace / "reference_files/builder-inputs/input-001/input.txt"
            staged_file.chmod(0o644)
            staged_file.write_text("tampered", encoding="utf-8")
        if self.status is ExecutionStatus.COMPLETED:
            skill = request.deliverables_dir / "calendar-skill"
            skill.mkdir()
            if self.invalid_skill:
                (skill / "SKILL.md").write_text("not a valid skill", encoding="utf-8")
            else:
                (skill / "SKILL.md").write_text(
                    "---\n"
                    "name: calendar-skill\n"
                    "description: A reusable calendar workflow\n"
                    "license: Apache-2.0\n"
                    "---\n\n"
                    "Use the calendar workflow.\n",
                    encoding="utf-8",
                )
                (skill / "references").mkdir()
                (skill / "references" / "guide.md").write_text("guide", encoding="utf-8")
        if self.raise_error:
            raise RuntimeError("deterministic executor failure")
        successful = self.status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
        output_text = "BUILDER-OUTPUT-SENTINEL" if successful else None
        return ExecutionResult(
            runtime="test",
            task_id=self.result_task_id or request.task.task_id,
            executor=self.result_executor or self.name,
            executor_version="deterministic-1",
            invocation_mode=self.invocation_mode,
            auth_mode="test-login",
            workspace=self.result_workspace or request.workspace,
            deliverables_dir=self.result_deliverables or request.deliverables_dir,
            status=self.status,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            exit_code=0 if successful else 1,
            available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}) if output_text is not None else frozenset(),
            failure=None if successful else Failure(FailureKind.PROCESS, "test_failure", FailureImpact.RUN),
            output_text=output_text,
            metadata={"builder_transcript": "BUILDER-METADATA-SENTINEL"},
        )


class ApplicationExecutor(Executor):
    name = "application-deterministic"
    runtime = "test"
    invocation_mode = "deterministic-application-test"
    network_access_enabled: bool = False
    reasoning_effort: ReasoningEffortOption = None

    def __init__(self) -> None:
        self.execute_calls = 0
        self.last_request: ExecutionRequest | None = None

    def preflight(self) -> PreflightResult:
        return PreflightResult(self.name, True, "application-1", "test-login", ("ready",))

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.execute_calls += 1
        self.last_request = request
        request.executor_dir.mkdir()
        request.deliverables_dir.mkdir()
        return ExecutionResult(
            runtime="test",
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="application-1",
            invocation_mode=self.invocation_mode,
            auth_mode="test-login",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=ExecutionStatus.NO_DELIVERABLE,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            exit_code=0,
            available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}),
            failure=None,
            output_text="APPLICATION-OUTPUT-SENTINEL",
            metadata={"application_metadata": "APPLICATION-METADATA-SENTINEL"},
        )


class ExecutorSkillBuilderTests(unittest.TestCase):
    def _request(self, root: Path, *, executor: DeterministicExecutor | None = None) -> BuildRequest:
        control = root / "control"
        control.mkdir()
        for name, value in {
            "AGENTS.md": "CONTROL-AGENTS-SENTINEL",
            "profile.json": "CONTROL-PROFILE-SENTINEL",
            "rubric.txt": "CONTROL-RUBRIC-SENTINEL",
            "reference-answer.txt": "CONTROL-REFERENCE-ANSWER-SENTINEL",
            "other-arm.txt": "CONTROL-OTHER-ARM-SENTINEL",
            "condition.txt": "CONTROL-CONDITION-SENTINEL",
        }.items():
            (control / name).write_text(value, encoding="utf-8")
        source = control / "source-SOURCE-PATH-SENTINEL"
        source.mkdir()
        (source / "input.txt").write_text("ALLOWLISTED-INPUT-SENTINEL", encoding="utf-8")
        bundle = load_builder_input_bundle(
            source,
            input_id="INPUT-ID-SENTINEL",
            input_type="reference-files",
            allowed_files=("input.txt",),
            source_revision="REVISION-SENTINEL",
        )
        return BuildRequest(
            build_run_id="build-run",
            task=TaskSpec("task-id", "canonical TARGET-PROMPT-SENTINEL"),
            inputs=(bundle,),
            runtime_root=control / "runtime",
            artifact_root=control / "artifact",
            model="model-sentinel",
            timeout_seconds=12.5,
        )

    def test_preflight_maps_identity_and_failed_preflight_never_executes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = DeterministicExecutor(preflight_ok=False)
            builder = ExecutorSkillBuilder(executor)
            preflight = builder.preflight()
            self.assertEqual(preflight.name, "executor-skill")
            self.assertFalse(preflight.ok)
            self.assertEqual(preflight.builder_executor, "deterministic")
            self.assertEqual(preflight.builder_executor_version, "deterministic-1")
            self.assertEqual(preflight.builder_executor_auth_mode, "test-login")
            self.assertEqual(preflight.builder_executor_invocation_mode, "deterministic-test")
            result = builder.build(self._request(root))
            self.assertEqual(result.status, BuildStatus.FAILED)
            self.assertEqual(result.failure_phase, BuildFailurePhase.PREFLIGHT)
            self.assertIsNone(result.execution)
            self.assertEqual(executor.preflight_calls, 1)
            self.assertEqual(executor.execute_calls, 0)

    def test_preflight_fail_closed_maps_type_only_and_rejects_truthy_or_wrong_identity(self) -> None:
        cases = (
            ("raised secret", {"preflight_exception": RuntimeError("PREFLIGHT-SECRET-SENTINEL")}, "RuntimeError"),
            # Preserve the invalid truthy value to exercise fail-closed handling.
            ("truthy non-bool", {"preflight_ok": cast(bool, "truthy")}, None),
            ("wrong identity", {"preflight_executor": "other-executor"}, None),
        )
        for label, options, exception_type in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                executor = DeterministicExecutor(**options)
                builder = ExecutorSkillBuilder(executor)
                preflight = builder.preflight()
                self.assertFalse(preflight.ok)
                self.assertEqual(executor.preflight_calls, 1)
                self.assertEqual(preflight.builder_executor_invocation_mode, "deterministic-test")
                if exception_type is not None:
                    self.assertEqual(preflight.details, (f"executor preflight raised {exception_type}",))
                    self.assertNotIn("PREFLIGHT-SECRET-SENTINEL", preflight.details[0])
                result = builder.build(self._request(root))
                self.assertEqual(result.status, BuildStatus.FAILED)
                self.assertEqual(result.failure_phase, BuildFailurePhase.PREFLIGHT)
                self.assertIsNone(result.execution)
                self.assertEqual(executor.preflight_calls, 1)
                self.assertEqual(executor.execute_calls, 0)

    def test_neutral_runtime_prompt_and_environment_exclude_provenance(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                "os.environ",
                {
                    "GDPVAL_CONDITION": "legacy-condition",
                    "GDPVAL_CONDITION_FILE": "legacy-condition-file",
                    "GDPVAL_CONDITION_APPLIED": "legacy-applied",
                    "BUILDER_AMBIENT_SENTINEL": "ambient-value",
                },
            ),
        ):
            root = Path(temporary)
            executor = DeterministicExecutor()
            result = ExecutorSkillBuilder(executor).build(self._request(root))
            self.assertEqual(result.status, BuildStatus.COMPLETED)
            assert executor.last_request is not None
            execution_request = executor.last_request
            self.assertEqual(execution_request.workspace, root / "control/runtime/workspace")
            self.assertEqual(execution_request.executor_dir, root / "control/runtime/executor")
            self.assertEqual(execution_request.deliverables_dir, root / "control/runtime/workspace/deliverables")
            self.assertEqual(execution_request.model, "model-sentinel")
            self.assertEqual(execution_request.timeout_seconds, 12.5)
            self.assertIn("canonical TARGET-PROMPT-SENTINEL", execution_request.task.prompt)
            self.assertIn("reference_files/builder-inputs/input-001", execution_request.task.prompt)
            serialized_environment = "\n".join(
                f"{key}={value}" for key, value in execution_request.environment.items()
            )
            for sentinel in (
                "INPUT-ID-SENTINEL",
                "SOURCE-PATH-SENTINEL",
                "REVISION-SENTINEL",
                "model-sentinel",
                "legacy-condition",
                "legacy-condition-file",
                "legacy-applied",
            ):
                self.assertNotIn(sentinel, execution_request.task.prompt)
                self.assertNotIn(sentinel, serialized_environment)
            for key in ("GDPVAL_CONDITION", "GDPVAL_CONDITION_FILE", "GDPVAL_CONDITION_APPLIED"):
                self.assertNotIn(key, execution_request.environment)
            self.assertEqual(execution_request.environment["BUILDER_AMBIENT_SENTINEL"], "ambient-value")

    def test_invalid_roots_fail_before_execution_without_overwriting_sentinels(self) -> None:
        cases = (
            "runtime equals artifact",
            "source is runtime ancestor",
            "source is artifact ancestor",
            "existing root",
            "symlink root",
        )
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request = self._request(root)
                source_root = request.inputs[0].root
                control = root / "control"
                if label == "runtime equals artifact":
                    request = replace(request, artifact_root=request.runtime_root)
                elif label == "source is runtime ancestor":
                    request = replace(request, runtime_root=source_root / "planned-runtime")
                elif label == "source is artifact ancestor":
                    request = replace(request, artifact_root=source_root / "planned-artifact")
                elif label == "existing root":
                    request.runtime_root.mkdir()
                    (request.runtime_root / "sentinel").write_text("keep", encoding="utf-8")
                else:
                    target = control / "symlink-target"
                    target.mkdir()
                    (target / "sentinel").write_text("keep", encoding="utf-8")
                    alias = control / "runtime-alias"
                    alias.symlink_to(target, target_is_directory=True)
                    request = replace(request, runtime_root=alias)

                executor = DeterministicExecutor()
                result = ExecutorSkillBuilder(executor).build(request)
                self.assertEqual(result.status, BuildStatus.FAILED)
                self.assertEqual(result.failure_phase, BuildFailurePhase.INPUT_VALIDATION)
                self.assertIsNone(result.execution)
                self.assertEqual(executor.execute_calls, 0)
                self.assertEqual((control / "AGENTS.md").read_text(encoding="utf-8"), "CONTROL-AGENTS-SENTINEL")
                self.assertEqual(
                    (control / "reference-answer.txt").read_text(encoding="utf-8"),
                    "CONTROL-REFERENCE-ANSWER-SENTINEL",
                )
                self.assertFalse(request.artifact_root.exists())
                if label == "existing root":
                    self.assertEqual((request.runtime_root / "sentinel").read_text(encoding="utf-8"), "keep")
                if label == "symlink root":
                    self.assertTrue(request.runtime_root.is_symlink())
                    self.assertEqual(
                        (control / "symlink-target" / "sentinel").read_text(encoding="utf-8"),
                        "keep",
                    )

    def test_runtime_logs_and_deliverables_persist_after_success_and_execution_failure(self) -> None:
        for status, expected in (
            (ExecutionStatus.COMPLETED, BuildStatus.COMPLETED),
            (ExecutionStatus.FAILED, BuildStatus.FAILED),
            (ExecutionStatus.TIMED_OUT, BuildStatus.TIMED_OUT),
            (ExecutionStatus.INTERRUPTED, BuildStatus.INTERRUPTED),
            (ExecutionStatus.NO_DELIVERABLE, BuildStatus.NO_ARTIFACT),
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                executor = DeterministicExecutor(status=status)
                result = ExecutorSkillBuilder(executor).build(self._request(root))
                self.assertEqual(result.status, expected)
                self.assertTrue((root / "control/runtime/executor/stdout.log").is_file())
                self.assertTrue((root / "control/runtime/executor/transcript.txt").is_file())
                self.assertTrue((root / "control/runtime/workspace/deliverables").is_dir())
                self.assertEqual((root / "control/artifact").exists(), expected is BuildStatus.COMPLETED)

    def test_normal_executor_exception_has_no_fabricated_execution_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = DeterministicExecutor(raise_error=True)
            result = ExecutorSkillBuilder(executor).build(self._request(root))
            self.assertEqual(result.status, BuildStatus.FAILED)
            self.assertEqual(result.failure_phase, BuildFailurePhase.EXECUTION)
            self.assertIsNone(result.execution)
            self.assertTrue((root / "control/runtime/executor/stdout.log").is_file())

    def test_mismatched_execution_identity_and_paths_fail_closed(self) -> None:
        cases = (
            {"result_task_id": "different-task"},
            {"result_executor": "wrong-executor"},
            {"result_workspace": Path("/tmp/wrong-workspace")},
            {"result_deliverables": Path("/tmp/wrong-deliverables")},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                executor = DeterministicExecutor(**overrides)
                result = ExecutorSkillBuilder(executor).build(self._request(root))
                self.assertEqual(result.status, BuildStatus.FAILED)
                self.assertEqual(result.failure_phase, BuildFailurePhase.EXECUTION)
                if "result_task_id" in overrides:
                    self.assertIsNone(result.execution)
                else:
                    self.assertIsNotNone(result.execution)
                self.assertFalse((root / "control/artifact").exists())

    def test_staged_input_tamper_is_invalid_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = DeterministicExecutor(tamper_inputs=True)
            result = ExecutorSkillBuilder(executor).build(self._request(root))
            self.assertEqual(result.status, BuildStatus.INVALID_ARTIFACT)
            self.assertEqual(result.failure_phase, BuildFailurePhase.ARTIFACT_VALIDATION)
            self.assertIsNotNone(result.execution)
            self.assertFalse((root / "control/artifact").exists())

    def test_invalid_skill_and_handoff_failure_are_mapped_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = ExecutorSkillBuilder(DeterministicExecutor(invalid_skill=True)).build(self._request(root))
            self.assertEqual(invalid.status, BuildStatus.INVALID_ARTIFACT)
            self.assertEqual(invalid.failure_phase, BuildFailurePhase.ARTIFACT_VALIDATION)
            self.assertFalse((root / "control/artifact").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = DeterministicExecutor()
            with patch(
                "eval_harness.builders.executor_skill.seal_generated_skill",
                side_effect=ArtifactHandoffError("handoff failed"),
            ):
                result = ExecutorSkillBuilder(executor).build(self._request(root))
            self.assertEqual(result.status, BuildStatus.FAILED)
            self.assertEqual(result.failure_phase, BuildFailurePhase.ARTIFACT_HANDOFF)
            self.assertIsNotNone(result.execution)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            executor = DeterministicExecutor()

            def race(deliverables: Path, artifact: Path) -> None:
                artifact.mkdir()
                (artifact / "racer-sentinel").write_text("keep", encoding="utf-8")
                raise FileExistsError("artifact root appeared after execution")

            with patch(
                "eval_harness.builders.executor_skill.seal_generated_skill",
                side_effect=race,
            ):
                result = ExecutorSkillBuilder(executor).build(request)
            self.assertEqual(result.status, BuildStatus.FAILED)
            self.assertEqual(result.failure_phase, BuildFailurePhase.ARTIFACT_HANDOFF)
            self.assertIsNotNone(result.execution)
            self.assertEqual((request.artifact_root / "racer-sentinel").read_text(encoding="utf-8"), "keep")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            request.artifact_root.mkdir()
            (request.artifact_root / "sentinel").write_text("keep", encoding="utf-8")
            executor = DeterministicExecutor()
            result = ExecutorSkillBuilder(executor).build(request)
            self.assertEqual(result.status, BuildStatus.FAILED)
            self.assertEqual(result.failure_phase, BuildFailurePhase.INPUT_VALIDATION)
            self.assertIsNone(result.execution)
            self.assertEqual((request.artifact_root / "sentinel").read_text(encoding="utf-8"), "keep")
            self.assertEqual(executor.execute_calls, 0)

    def test_success_seals_artifact_only_and_applies_from_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_executor = DeterministicExecutor()
            builder = ExecutorSkillBuilder(build_executor)
            preflight = builder.preflight()
            self.assertTrue(preflight.ok)
            self.assertEqual(preflight.builder_executor_invocation_mode, "deterministic-test")
            result = builder.build(self._request(root))
            self.assertEqual(result.status, BuildStatus.COMPLETED)
            assert result.bundle is not None
            self.assertIsNone(AgentSkillIntervention(result.bundle).source_reference)
            intervention = AgentSkillIntervention(result.bundle)
            self.assertTrue(intervention.preflight().ok)
            neutral = root / "neutral"
            neutral.mkdir()
            application_workspace = neutral / "application-workspace"
            application_workspace.mkdir()
            control = root / "control"
            self.assertNotIn(control.resolve(), application_workspace.resolve().parents)
            application_executor_dir = neutral / "application-executor"
            application_deliverables = neutral / "application-deliverables"
            # This verifies harness handoff only; it does not assert semantic content or OS read confinement.
            application = intervention.apply(
                TaskSpec("application-task", "canonical application task"),
                application_workspace,
                application_run_id="fresh-application-run",
            )
            self.assertEqual(application.task.task_id, "application-task")
            self.assertEqual(application.application_run_id, "fresh-application-run")
            self.assertIn("canonical application task", application.task.prompt)
            self.assertIn(".gdpval/interventions/calendar-skill/SKILL.md", application.task.prompt)
            application_file_paths = sorted(
                path.relative_to(application_workspace).as_posix()
                for path in application_workspace.rglob("*")
                if path.is_file()
            )
            self.assertEqual(
                application_file_paths,
                [
                    ".gdpval/interventions/calendar-skill/SKILL.md",
                    ".gdpval/interventions/calendar-skill/references/guide.md",
                ],
            )
            application_files = "\n".join(
                path.read_text(encoding="utf-8") for path in application_workspace.rglob("*") if path.is_file()
            )
            for sentinel in (
                "builder stdout",
                "builder transcript",
                "BUILDER-OUTPUT-SENTINEL",
                "BUILDER-METADATA-SENTINEL",
                "INPUT-ID-SENTINEL",
                "SOURCE-PATH-SENTINEL",
                "REVISION-SENTINEL",
                "TARGET-PROMPT-SENTINEL",
                "model-sentinel",
                "ALLOWLISTED-INPUT-SENTINEL",
                "CONTROL-AGENTS-SENTINEL",
                "CONTROL-PROFILE-SENTINEL",
                "CONTROL-RUBRIC-SENTINEL",
                "CONTROL-REFERENCE-ANSWER-SENTINEL",
                "CONTROL-OTHER-ARM-SENTINEL",
                "CONTROL-CONDITION-SENTINEL",
            ):
                self.assertNotIn(sentinel, application.task.prompt)
                self.assertNotIn(sentinel, application_files)
            application_executor = ApplicationExecutor()
            application_result = application_executor.execute(
                ExecutionRequest(
                    task=application.task,
                    workspace=application_workspace,
                    deliverables_dir=application_deliverables,
                    executor_dir=application_executor_dir,
                    environment={"PATH": "/usr/bin"},
                )
            )
            self.assertEqual(application_executor.execute_calls, 1)
            assert application_executor.last_request is not None
            for field in ("execution", "result", "output_text", "metadata", "session", "resume"):
                self.assertNotIn(field, ExecutionRequest.__dataclass_fields__)
            application_request_serialized = repr(
                (
                    application_executor.last_request.task.task_id,
                    application_executor.last_request.task.prompt,
                    application_executor.last_request.workspace.as_posix(),
                    application_executor.last_request.executor_dir.as_posix(),
                    application_executor.last_request.deliverables_dir.as_posix(),
                    tuple(f"{key}={value}" for key, value in application_executor.last_request.environment.items()),
                )
            )
            self.assertIsNot(result.execution, application_result)
            self.assertIsNot(result.execution, application_executor.last_request)
            self.assertEqual(application_executor.last_request.environment, {"PATH": "/usr/bin"})
            self.assertNotIn("APPLICATION-OUTPUT-SENTINEL", application_files)
            self.assertNotIn("APPLICATION-METADATA-SENTINEL", application_files)
            self.assertNotIn("BUILDER-OUTPUT-SENTINEL", application_request_serialized)
            self.assertNotIn("BUILDER-METADATA-SENTINEL", application_request_serialized)
            self.assertIsNot(build_executor, application_executor)
            assert build_executor.last_request is not None
            self.assertNotEqual(build_executor.last_request.workspace, application_workspace)
            self.assertNotEqual(build_executor.last_request.executor_dir, application_executor_dir)
            self.assertNotEqual(build_executor.last_request.deliverables_dir, application_deliverables)
            self.assertTrue((root / "control/artifact/calendar-skill/SKILL.md").is_file())
            self.assertTrue((root / "control/runtime/workspace/deliverables/calendar-skill/SKILL.md").is_file())
            self.assertTrue((application_workspace / ".gdpval/interventions/calendar-skill/SKILL.md").is_file())


if __name__ == "__main__":
    unittest.main()
