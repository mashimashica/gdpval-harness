# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import eval_harness.local_runner as local_runner
from eval_harness.benchmarks.base import BenchmarkTask
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    PreflightResult,
    TaskSpec,
)
from eval_harness.executors.claude_code import ClaudeCodeExecutor
from eval_harness.executors.codex import CodexExecutor
from eval_harness.executors.cursor import CursorExecutor
from eval_harness.interventions import PromptOverlayIntervention
from eval_harness.interventions.base import InterventionPreflightResult
from eval_harness.judges.base import JudgeRequest
from eval_harness.judges.codex import CodexJudgeExecutor
from eval_harness.local_judge_runner import _candidate_task_prompt
from eval_harness.local_runner import (
    _application_evidence,
    _condition_instructions,
    _executor_environment,
    _validate_resume_condition,
    build_task_prompt,
)


class _FakeLocalBenchmark:
    def __init__(self, prompt: str = "Base GDPval task") -> None:
        self.prompt = prompt
        self.prepared = 0

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        self.prepared += 1

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        if limit != 1:
            raise AssertionError(f"expected one fake task, got {limit}")
        return [BenchmarkTask(execution=TaskSpec("task", self.prompt))]

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        del task, workspace
        return []


class _CapturingLocalExecutor:
    name = "fake-local"
    invocation_mode = "fake"
    tool_permission_mode = "fake"

    def __init__(self) -> None:
        self.preflight_calls = 0
        self.requests: list[ExecutionRequest] = []

    def preflight(self) -> PreflightResult:
        self.preflight_calls += 1
        return PreflightResult(executor=self.name, ok=True, version="fake-1", auth_mode="fake")

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        return ExecutionResult(
            runtime="test",
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="fake-1",
            invocation_mode=self.invocation_mode,
            auth_mode="fake",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=ExecutionStatus.NO_DELIVERABLE,
            started_at="started",
            finished_at="finished",
            exit_code=0,
            available_outputs=frozenset(),
            failure=None,
        )


class ExperimentConditionTests(unittest.TestCase):
    def _execution_request(self, root: Path) -> ExecutionRequest:
        workspace = root / "workspace"
        executor_dir = root / "executor"
        return ExecutionRequest(
            task=TaskSpec(task_id="task", prompt="Base GDPval task"),
            workspace=workspace,
            deliverables_dir=workspace / "deliverables",
            executor_dir=executor_dir,
        )

    def _judge_request(self, root: Path) -> JudgeRequest:
        workspace = root / "judge-workspace"
        return JudgeRequest(
            task_id="task",
            task_prompt="Judge this task",
            workspace=workspace,
            reference_dir=workspace / "reference_files",
            submission_a_dir=workspace / "submission_a",
            submission_b_dir=workspace / "submission_b",
            executor_dir=root / "judge-executor",
            trial_index=0,
            swapped=False,
        )

    def test_condition_file_is_external_and_label_is_not_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition.md"
            condition.write_text("Use the supplied work-design method.\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"GDPVAL_CONDITION_FILE": str(condition), "GDPVAL_CONDITION": "secret-label"},
                clear=True,
            ):
                instructions = _condition_instructions()
            prompt = build_task_prompt(
                TaskSpec(task_id="task", prompt="Base GDPval task"),
                root,
                network_policy="disabled",
                condition_instructions=instructions,
            )
            self.assertIn("Use the supplied work-design method.", prompt)
            self.assertEqual(prompt.count("[BEGIN INTERVENTION PROMPT OVERLAY]"), 1)
            self.assertNotIn("secret-label", prompt)
            self.assertEqual(prompt.rsplit("\nTask:\n", 1)[1], "Base GDPval task\n")

    def test_condition_label_and_source_stay_out_of_application_and_executor_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "condition-source-sentinel.md"
            source.write_text("Use the supplied work-design method.\n", encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            label = "condition-label-sentinel"
            with patch.dict(
                os.environ,
                {
                    "GDPVAL_CONDITION": label,
                    "GDPVAL_CONDITION_FILE": str(source),
                    "GDPVAL_CONDITION_APPLIED": "true",
                },
                clear=True,
            ):
                environment = _executor_environment()

            intervention = PromptOverlayIntervention(source)
            self.assertTrue(intervention.preflight().ok)
            wrapped = TaskSpec(
                "task",
                build_task_prompt(
                    TaskSpec("task", "Base GDPval task"),
                    workspace,
                    network_policy="disabled",
                ),
            )
            application = intervention.apply(wrapped, workspace, application_run_id="opaque-application-run")
            application_payload = json.dumps(_application_evidence(application), sort_keys=True)

            self.assertIn("Use the supplied work-design method.", application.task.prompt)
            self.assertEqual(application.task.prompt.count("[BEGIN INTERVENTION PROMPT OVERLAY]"), 1)
            self.assertNotIn(label, application.task.prompt)
            self.assertNotIn(str(source), application.task.prompt)
            self.assertNotIn(label, application_payload)
            self.assertNotIn(str(source), application_payload)
            self.assertNotIn(label, environment)
            self.assertNotIn(str(source), environment)
            self.assertNotIn("GDPVAL_CONDITION", environment)
            self.assertNotIn("GDPVAL_CONDITION_FILE", environment)
            self.assertNotIn("GDPVAL_CONDITION_APPLIED", environment)

            request = ExecutionRequest(
                task=application.task,
                workspace=workspace,
                deliverables_dir=workspace / "deliverables",
                executor_dir=root / "executor",
                environment=environment,
            )
            for executor in (
                CodexExecutor(command="codex"),
                ClaudeCodeExecutor(command="claude"),
                CursorExecutor(command="agent"),
            ):
                argv = " ".join(executor.build_command(request))
                self.assertNotIn(label, argv)
                self.assertNotIn(str(source), argv)

    def test_run_applies_overlay_once_and_keeps_canonical_prompt_and_metadata_outer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "condition-source-sentinel.md"
            source.write_text("Use the supplied work-design method.\n", encoding="utf-8")
            out = root / "run"
            label = "condition-label-sentinel"
            benchmark = _FakeLocalBenchmark()
            executor = _CapturingLocalExecutor()
            intervention = PromptOverlayIntervention(source)
            with (
                patch.dict(
                    os.environ,
                    {
                        "GDPVAL_EXECUTOR": "fake-local",
                        "GDPVAL_CONDITION": label,
                        "GDPVAL_CONDITION_FILE": str(source),
                        "GDPVAL_CONDITION_APPLIED": "true",
                        "GDPVAL_WRITE_METADATA": "0",
                        "LIMIT": "1",
                        "OUT": str(out),
                    },
                    clear=True,
                ),
                patch.object(local_runner, "_build_intervention", return_value=intervention),
                patch.object(local_runner, "_benchmark", return_value=benchmark),
                patch.object(local_runner, "_executor", return_value=executor),
                patch.object(intervention, "apply", wraps=intervention.apply) as apply,
            ):
                status = local_runner.run()

            self.assertEqual(status, 0)
            self.assertEqual(executor.preflight_calls, 1)
            self.assertEqual(len(executor.requests), 1)
            self.assertEqual(apply.call_count, 1)
            request = executor.requests[0]
            self.assertEqual(request.task.task_id, "task")
            self.assertEqual(request.task.prompt.count("[BEGIN INTERVENTION PROMPT OVERLAY]"), 1)
            self.assertEqual(request.task.prompt.count("Use the supplied work-design method."), 1)
            self.assertNotIn(label, request.task.prompt)
            self.assertNotIn(str(source), request.task.prompt)
            self.assertNotIn(label, request.environment)
            self.assertNotIn(str(source), request.environment)
            self.assertNotIn("GDPVAL_CONDITION", request.environment)
            self.assertNotIn("GDPVAL_CONDITION_FILE", request.environment)
            self.assertNotIn("GDPVAL_CONDITION_APPLIED", request.environment)

            canonical = (out / "tasks" / "task" / "executor" / "task-prompt.txt").read_text(encoding="utf-8")
            self.assertEqual(canonical, "Base GDPval task")
            metadata = json.loads((out / "tasks" / "task" / "executor" / "metadata.json").read_text())
            evidence = json.dumps(metadata["intervention_application"], sort_keys=True)
            self.assertNotIn(label, evidence)
            self.assertNotIn(str(source), evidence)
            self.assertTrue(metadata["intervention_application"]["application_run_id"])

            for path in (out / "tasks" / "task").rglob("*"):
                if path.is_file():
                    contents = path.read_text(encoding="utf-8")
                    self.assertNotIn(label, contents)
                    self.assertNotIn(str(source), contents)

    def test_invalid_condition_fails_before_executor_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "invalid-condition.md"
            source.write_bytes(b"\xff")
            out = root / "run"
            executor = _CapturingLocalExecutor()
            with (
                patch.dict(
                    os.environ,
                    {
                        "GDPVAL_EXECUTOR": "fake-local",
                        "GDPVAL_CONDITION_FILE": str(source),
                        "GDPVAL_WRITE_METADATA": "0",
                        "LIMIT": "1",
                        "OUT": str(out),
                    },
                    clear=True,
                ),
                patch.object(local_runner, "_executor", return_value=executor),
            ):
                status = local_runner.run()

            self.assertEqual(status, 2)
            self.assertEqual(executor.preflight_calls, 0)
            self.assertFalse(out.exists())

    def test_tampered_condition_fails_closed_before_executor_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "condition.md"
            source.write_text("original intervention\n", encoding="utf-8")
            out = root / "run"
            executor = _CapturingLocalExecutor()

            class TamperingPromptOverlay(PromptOverlayIntervention):
                def preflight(self) -> InterventionPreflightResult:
                    result = super().preflight()
                    if result.ok:
                        self.source.write_text("tampered intervention\n", encoding="utf-8")
                    return result

            intervention = TamperingPromptOverlay(source)
            with (
                patch.dict(
                    os.environ,
                    {
                        "GDPVAL_EXECUTOR": "fake-local",
                        "GDPVAL_CONDITION_FILE": str(source),
                        "GDPVAL_WRITE_METADATA": "0",
                        "LIMIT": "1",
                        "OUT": str(out),
                    },
                    clear=True,
                ),
                patch.object(local_runner, "_build_intervention", return_value=intervention),
                patch.object(local_runner, "_benchmark", return_value=_FakeLocalBenchmark()),
                patch.object(local_runner, "_executor", return_value=executor),
            ):
                status = local_runner.run()

            self.assertEqual(status, 1)
            self.assertEqual(executor.preflight_calls, 1)
            self.assertEqual(executor.requests, [])
            metadata = json.loads((out / "tasks" / "task" / "executor" / "metadata.json").read_text())
            self.assertEqual(metadata["harness_error"], "intervention application failed before executor")
            self.assertEqual(metadata["intervention_error_type"], "RuntimeError")
            self.assertNotIn(str(source), json.dumps(metadata))

    def test_condition_may_contain_task_marker_when_canonical_prompt_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            executor_dir = run_root / "tasks" / "one" / "executor"
            executor_dir.mkdir(parents=True)
            (executor_dir / "task-prompt.txt").write_text("Base GDPval task", encoding="utf-8")
            (executor_dir / "prompt.txt").write_text(
                "Wrapper\n<condition>\nTemplate\nTask:\nplaceholder\n</condition>\n\nTask:\nBase GDPval task\n",
                encoding="utf-8",
            )
            deliverables = run_root / "deliverables"
            deliverables.mkdir()
            self.assertEqual(_candidate_task_prompt(deliverables, "task_one"), "Base GDPval task")

    def test_condition_file_must_be_nonempty_utf8_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.md"
            empty.write_text("   \n", encoding="utf-8")
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(empty)}, clear=True):
                with self.assertRaisesRegex(ValueError, "empty"):
                    _condition_instructions()

            invalid = root / "invalid.bin"
            invalid.write_bytes(b"\xff")
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(invalid)}, clear=True):
                with self.assertRaisesRegex(ValueError, "UTF-8"):
                    _condition_instructions()

            large = root / "large.md"
            large.write_bytes(b"x" * (1024 * 1024 + 1))
            with patch.dict(os.environ, {"GDPVAL_CONDITION_FILE": str(large)}, clear=True):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    _condition_instructions()

    def test_resume_requires_identical_condition_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition.md"
            condition.write_text("same intervention\n", encoding="utf-8")
            digest = hashlib.sha256(condition.read_bytes()).hexdigest()
            (root / "run-metadata.json").write_text(
                json.dumps(
                    {
                        "configuration": {
                            "condition": "treatment",
                            "condition_file_sha256": digest,
                            "condition_applied_to_prompt": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            env = {
                "RESUME": "1",
                "GDPVAL_CONDITION": "treatment",
                "GDPVAL_CONDITION_FILE": str(condition),
            }
            with patch.dict(os.environ, env, clear=True):
                _validate_resume_condition(root)

            condition.write_text("changed intervention\n", encoding="utf-8")
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    _validate_resume_condition(root)

    def test_conditioned_resume_without_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition.md"
            condition.write_text("intervention\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"RESUME": "1", "GDPVAL_CONDITION": "treatment", "GDPVAL_CONDITION_FILE": str(condition)},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "requires the existing run-metadata"):
                    _validate_resume_condition(root)

    def test_codex_policy_forces_chatgpt_and_disables_web_search_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = CodexExecutor(network_enabled=False).build_command(self._execution_request(Path(tmp)))
        joined = " ".join(command)
        self.assertIn('forced_login_method="chatgpt"', joined)
        self.assertIn('web_search="disabled"', joined)
        self.assertIn('approval_policy="never"', joined)

    def test_codex_policy_does_not_force_web_search_off_when_network_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = CodexExecutor(network_enabled=True).build_command(self._execution_request(Path(tmp)))
        self.assertNotIn('web_search="disabled"', " ".join(command))
        self.assertIn('forced_login_method="chatgpt"', " ".join(command))

    def test_codex_judge_forces_chatgpt_disables_web_and_keeps_read_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = CodexJudgeExecutor().build_command(self._judge_request(Path(tmp)))
        joined = " ".join(command)
        self.assertIn('forced_login_method="chatgpt"', joined)
        self.assertIn('web_search="disabled"', joined)
        self.assertIn('default_permissions="gdpval-harness-blind-judge"', joined)
        self.assertIn('":root"="deny"', joined)
        self.assertIn('":minimal"="read"', joined)
        self.assertNotIn("--sandbox read-only", joined)


if __name__ == "__main__":
    unittest.main()
