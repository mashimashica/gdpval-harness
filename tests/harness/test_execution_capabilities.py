# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from eval_harness.capabilities import (
    CapabilityPreflightResult,
    ExecutionRequirements,
    ExecutorCapabilities,
    ExecutorInput,
    ExecutorOutput,
    preflight_capabilities,
)
from eval_harness.executors.base import ExecutionResult, ExecutionStatus
from eval_harness.executors.claude_code import ClaudeCodeExecutor
from eval_harness.executors.codex import CodexExecutor
from eval_harness.executors.cursor import CursorExecutor
from eval_harness.failures import Failure, FailureImpact, FailureKind, RunAbort


class ExecutionCapabilitiesTests(unittest.TestCase):
    @staticmethod
    def _result(
        *,
        status: ExecutionStatus = ExecutionStatus.COMPLETED,
        available_outputs: frozenset[ExecutorOutput] = frozenset(),
        failure: Failure | None = None,
        output_text: str | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            task_id="task",
            executor="fake",
            executor_version="1",
            invocation_mode="test",
            auth_mode="local",
            workspace=Path("/tmp/workspace"),
            deliverables_dir=Path("/tmp/deliverables"),
            status=status,
            started_at="start",
            finished_at="finish",
            exit_code=0 if status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE} else 1,
            available_outputs=available_outputs,
            failure=failure,
            output_text=output_text,
        )

    def test_capability_types_normalize_to_immutable_sets(self) -> None:
        requirements = ExecutionRequirements(
            inputs=frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES}),
            outputs=frozenset({ExecutorOutput.FINAL_TEXT}),
        )
        capabilities = ExecutorCapabilities(
            inputs=frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES}),
            outputs=frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
        )
        self.assertEqual(requirements.inputs, frozenset(ExecutorInput))
        self.assertEqual(capabilities.outputs, frozenset(ExecutorOutput))
        with self.assertRaises(AttributeError):
            requirements.inputs.add(ExecutorInput.WORKSPACE_FILES)  # type: ignore[attr-defined]

    def test_preflight_is_benchmark_independent_and_reports_sorted_missing_channels(self) -> None:
        requirements = ExecutionRequirements(
            inputs=frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES}),
            outputs=frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
        )
        capabilities = ExecutorCapabilities(
            inputs=frozenset({ExecutorInput.PROMPT_TEXT}),
            outputs=frozenset({ExecutorOutput.FINAL_TEXT}),
        )
        result = preflight_capabilities(requirements, capabilities)
        self.assertEqual(
            result,
            CapabilityPreflightResult(
                ok=False,
                missing_inputs=(ExecutorInput.WORKSPACE_FILES,),
                missing_outputs=(ExecutorOutput.ARTIFACT_FILES,),
            ),
        )
        empty_requirements = ExecutionRequirements(inputs=frozenset(), outputs=frozenset())
        empty_capabilities = ExecutorCapabilities(inputs=frozenset(), outputs=frozenset())
        empty_result = preflight_capabilities(empty_requirements, empty_capabilities)
        self.assertFalse(empty_result.missing_inputs)
        self.assertTrue(empty_result.ok)

    def test_current_adapters_declare_explicit_capabilities(self) -> None:
        expected_inputs = frozenset({ExecutorInput.PROMPT_TEXT, ExecutorInput.WORKSPACE_FILES})
        self.assertEqual(
            CodexExecutor.capabilities.inputs,
            expected_inputs,
        )
        self.assertEqual(
            CodexExecutor.capabilities.outputs,
            frozenset({ExecutorOutput.FINAL_TEXT, ExecutorOutput.ARTIFACT_FILES}),
        )
        for executor in (ClaudeCodeExecutor, CursorExecutor):
            with self.subTest(executor=executor.__name__):
                self.assertEqual(executor.capabilities.inputs, expected_inputs)
                self.assertEqual(executor.capabilities.outputs, frozenset({ExecutorOutput.ARTIFACT_FILES}))

    def test_unknown_capability_and_wrong_contract_types_fail_closed(self) -> None:
        empty_requirements = ExecutionRequirements(inputs=frozenset(), outputs=frozenset())
        empty_capabilities = ExecutorCapabilities(inputs=frozenset(), outputs=frozenset())
        with self.assertRaises(ValueError):
            ExecutionRequirements(inputs={"benchmark_name"}, outputs=frozenset())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ExecutorCapabilities(inputs=frozenset(), outputs={"artifacts"})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            preflight_capabilities(object(), empty_capabilities)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            preflight_capabilities(empty_requirements, object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CapabilityPreflightResult(ok=1, missing_inputs=(), missing_outputs=())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CapabilityPreflightResult(ok=False, missing_inputs=("unknown",), missing_outputs=())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CapabilityPreflightResult(ok=False, missing_inputs=None, missing_outputs=())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CapabilityPreflightResult(ok=True, missing_inputs=(ExecutorInput.PROMPT_TEXT,), missing_outputs=())

    def test_execution_result_preserves_actual_empty_and_artifact_channels(self) -> None:
        empty_final = self._result(
            available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}),
            output_text="",
        )
        self.assertEqual(empty_final.output_text, "")
        self.assertEqual(empty_final.available_outputs, frozenset({ExecutorOutput.FINAL_TEXT}))

        empty_artifacts = self._result(
            available_outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
        )
        self.assertIsNone(empty_artifacts.output_text)
        self.assertEqual(empty_artifacts.available_outputs, frozenset({ExecutorOutput.ARTIFACT_FILES}))

        no_deliverable = self._result(
            status=ExecutionStatus.NO_DELIVERABLE,
            available_outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
        )
        self.assertEqual(no_deliverable.available_outputs, frozenset({ExecutorOutput.ARTIFACT_FILES}))

    def test_execution_result_rejects_inconsistent_channels_and_failures(self) -> None:
        with self.assertRaises(ValueError):
            self._result(status="unknown")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self._result(available_outputs={"unknown"})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self._result(output_text=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self._result(output_text="answer")
        with self.assertRaises(ValueError):
            self._result(
                available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}),
                output_text=None,
            )
        with self.assertRaises(ValueError):
            self._result(status=ExecutionStatus.FAILED)
        with self.assertRaises(ValueError):
            self._result(
                status=ExecutionStatus.FAILED,
                failure=Failure(FailureKind.PROCESS, "process_exit", FailureImpact.RUN),
                available_outputs=frozenset({ExecutorOutput.ARTIFACT_FILES}),
            )
        with self.assertRaises(TypeError):
            self._result(failure=object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self._result(failure=Failure(FailureKind.PROCESS, "process_exit", FailureImpact.TASK))
        with self.assertRaises(ValueError):
            self._result(
                status=ExecutionStatus.FAILED,
                failure=Failure(FailureKind.PROCESS, "process_exit", FailureImpact.TASK),
            )

    def test_execution_result_requires_run_failure_for_systemic_adapter_failures(self) -> None:
        failure = Failure(FailureKind.INTEGRITY, "reference_mutation", FailureImpact.RUN)
        result = self._result(status=ExecutionStatus.FAILED, failure=failure)
        self.assertIs(result.failure, failure)
        self.assertEqual(result.available_outputs, frozenset())

    def test_failure_records_are_immutable_and_systemic_kinds_cannot_be_task_impact(self) -> None:
        failure = Failure(FailureKind.PROCESS, "executor_exit", FailureImpact.TASK)
        self.assertEqual(failure.code, "executor_exit")
        self.assertEqual(failure.impact, FailureImpact.TASK)
        with self.assertRaises(ValueError):
            Failure(FailureKind.AUTH, "login_required", FailureImpact.TASK)
        with self.assertRaises(ValueError):
            Failure(FailureKind.PROCESS, "provider failure", FailureImpact.TASK)
        with self.assertRaises(ValueError):
            Failure(FailureKind.INTERNAL, "secret=must_not_persist", FailureImpact.RUN)
        with self.assertRaises(ValueError):
            Failure(FailureKind.INTERNAL, "stable_code_" + ("x" * 128), FailureImpact.RUN)
        with self.assertRaises(ValueError):
            Failure("unknown", "stable_code", FailureImpact.TASK)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Failure(FailureKind.PROCESS, "stable_code", "unknown")  # type: ignore[arg-type]

    def test_run_abort_requires_run_impact_and_exposes_only_stable_code(self) -> None:
        failure = Failure(FailureKind.AUTH, "login_required", FailureImpact.RUN)
        raised = RunAbort(failure)
        self.assertIs(raised.failure, failure)
        self.assertEqual(str(raised), "login_required")
        with self.assertRaises(ValueError):
            RunAbort(Failure(FailureKind.PROCESS, "task_failed", FailureImpact.TASK))
        with self.assertRaises(TypeError):
            RunAbort(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
