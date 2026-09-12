# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast
from unittest.mock import patch

import eval_harness.runner as runner_module
from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.capabilities import ExecutorOutput
from eval_harness.evaluators.base import (
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
    EvaluatorType,
)
from eval_harness.evaluators.gdpval import GDPvalExternalEvaluator
from eval_harness.evaluators.pairwise import PairwiseJudgeEvaluator
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from eval_harness.failures import Failure, FailureImpact, FailureKind, RunAbort
from eval_harness.interventions.agent_skill import AgentSkillIntervention, load_agent_skill_bundle
from eval_harness.interventions.base import (
    ApplicationMapping,
    Intervention,
    InterventionApplication,
    InterventionBundle,
    InterventionFile,
    InterventionManifest,
    InterventionPreflightResult,
    InterventionType,
    compute_bundle_sha256,
)
from eval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult
from eval_harness.judges.pairwise import discover_tasks
from eval_harness.local_judge_runner import _candidate_task_prompt
from eval_harness.provenance import RepositoryProvenance, canonical_json_sha256
from eval_harness.reasoning import ReasoningEffortOption
from eval_harness.runner import run_benchmark


class _ExecutorOptions(TypedDict, total=False):
    result_executor: str
    result_invocation_mode: str
    result_version: str
    result_auth_mode: str


JsonObject = dict[str, object]


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AssertionError(f"expected a JSON object with string keys, got {type(value).__name__}")
    return cast(JsonObject, value)


def _load_json_object(path: Path) -> JsonObject:
    return _json_object(json.loads(path.read_text(encoding="utf-8")))


class FakeBenchmark(Benchmark):
    name = "fake"
    revision = "test-revision"

    def __init__(
        self,
        task_count: int = 3,
        *,
        events: list[str] | None = None,
        evaluation_metadata: dict[str, object] | None = None,
        prompt_prefix: str = "",
    ) -> None:
        self.task_count = task_count
        self.events = events if events is not None else []
        self.evaluation_metadata = evaluation_metadata or {}
        self.prompt_prefix = prompt_prefix

    def is_prepared(self) -> bool:
        return True

    def prepare(self) -> None:
        self.events.append("benchmark-prepare")
        return None

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        return [
            BenchmarkTask(
                execution=TaskSpec(task_id=f"task-{index}", prompt=f"{self.prompt_prefix}prompt-{index}"),
                evaluation=dict(self.evaluation_metadata),
            )
            for index in range(min(limit, self.task_count))
        ]

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        self.events.append(f"materialize:{task.execution.task_id}")
        (workspace / "input.txt").write_text(task.execution.prompt, encoding="utf-8")
        return ["input.txt"]

    def execution_task(self, task: BenchmarkTask, workspace: Path, *, network_policy: str) -> TaskSpec:
        del workspace, network_policy
        self.events.append(f"execution-task:{task.execution.task_id}")
        return task.execution


class FakeIntervention(Intervention):
    name = "fake-intervention"
    intervention_type = InterventionType.PROMPT_OVERLAY

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        fail_task_id: str | None = None,
        invalid_task_id: str | None = None,
        mismatch_application: str | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.fail_task_id = fail_task_id
        self.invalid_task_id = invalid_task_id
        self.mismatch_application = mismatch_application
        self.preflight_calls = 0
        self.apply_calls = 0
        self.tasks: list[TaskSpec] = []
        self._bundle: InterventionBundle | None = None

    def preflight(self) -> InterventionPreflightResult:
        self.events.append("intervention-preflight")
        self.preflight_calls += 1
        content = b"secret intervention source"
        entry = InterventionFile(path="overlay.txt", size=len(content), sha256="".join(["a"] * 64))
        manifest = InterventionManifest(
            intervention_id="fake-intervention-id",
            intervention_type=self.intervention_type,
            source_revision="source-revision",
            revision_status="available",
            files=(entry,),
            bundle_sha256=compute_bundle_sha256(((entry.path, content),)),
            application=ApplicationMapping(method="prompt-overlay", target="task.prompt"),
        )
        self._bundle = InterventionBundle(root=Path(__file__).resolve().parent, manifest=manifest)
        return InterventionPreflightResult(
            name=self.name,
            intervention_type=self.intervention_type,
            ok=True,
            bundle=self._bundle,
            details=("fake intervention ready", "/external/secret/intervention-source"),
        )

    def validate_task(self, task: TaskSpec) -> None:
        self.events.append(f"validate:{task.task_id}")
        if task.task_id == self.invalid_task_id:
            raise ValueError("invalid intervention task")

    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        self.events.append(f"apply:{task.task_id}")
        self.apply_calls += 1
        self.tasks.append(task)
        if task.task_id == self.fail_task_id:
            raise RuntimeError("intervention application secret failure")
        assert self._bundle is not None
        bundle_sha256 = self._bundle.manifest.bundle_sha256
        manifest_sha256 = self._bundle.manifest.manifest_sha256 or ""
        application_mapping = self._bundle.manifest.application
        if self.mismatch_application == "bundle":
            bundle_sha256 = "0" * 64
        elif self.mismatch_application == "manifest":
            manifest_sha256 = "1" * 64
        elif self.mismatch_application == "mapping":
            application_mapping = ApplicationMapping(method="files", target="workspace")
        return InterventionApplication(
            application_run_id=application_run_id,
            task=TaskSpec(task.task_id, f"[OVERLAY FOR CONDITION SECRET]\n{task.prompt}"),
            materialized_files=(
                InterventionFile(
                    path="injected.txt",
                    size=7,
                    sha256="".join(["b"] * 64),
                ),
            ),
            bundle_sha256=bundle_sha256,
            manifest_sha256=manifest_sha256,
            application=application_mapping,
        )


class FakeEvaluator(Evaluator):
    name = "fake-evaluator"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def __init__(
        self,
        *,
        fail: bool = False,
        statuses: tuple[EvaluationStatus, ...] | None = None,
        invalid_plan_task_id: str | None = None,
        events: list[str] | None = None,
        preflight_details: tuple[str, ...] = (),
        judge_fields: dict[str, str | None] | None = None,
    ) -> None:
        self.fail = fail
        self.statuses = statuses or (EvaluationStatus.COMPLETED,)
        self.invalid_plan_task_id = invalid_plan_task_id
        self.events = events if events is not None else []
        self.calls = 0
        self.preflight_calls = 0
        self.preflight_details = preflight_details
        self.judge_fields = dict(judge_fields or {})

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        self.events.append("evaluator-preflight")
        self.preflight_calls += 1
        return EvaluatorPreflightResult(
            self.name,
            self.evaluator_type,
            True,
            version="fake-eval-1",
            details=self.preflight_details,
            **self.judge_fields,
        )

    def validate_plan(self, plan: EvaluationPlan) -> None:
        if plan.task_id == self.invalid_plan_task_id:
            raise ValueError("invalid evaluation plan")

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls += 1
        self.events.append(f"evaluate:{request.task_id}")
        if self.fail:
            raise RuntimeError("evaluator test failure")
        status = self.statuses[min(self.calls - 1, len(self.statuses) - 1)]
        return EvaluationResult(
            task_id=request.task_id,
            status=status,
            metrics={"accuracy": 1.0},
            details={"canonical_prompt": request.task_prompt},
        )


class FakeExecutor(Executor):
    name = "fake-executor"
    runtime = "test"
    invocation_mode = "fake"
    network_access_enabled: bool = False
    reasoning_effort: ReasoningEffortOption = None

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        outside_deliverables: bool = False,
        events: list[str] | None = None,
        result_executor: str | None = None,
        result_invocation_mode: str | None = None,
        result_version: str | None = "fake-1",
        result_auth_mode: str | None = "fake-local",
        result_metadata: dict[str, object] | None = None,
        result_output_text: str | None = "answer",
        result_status: ExecutionStatus = ExecutionStatus.NO_DELIVERABLE,
    ) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.outside_deliverables = outside_deliverables
        self.events = events if events is not None else []
        self.tasks: list[TaskSpec] = []
        self.environments: list[dict[str, str]] = []
        self.preflight_calls = 0
        self.result_executor = result_executor
        self.result_invocation_mode = result_invocation_mode
        self.result_version = result_version
        self.result_auth_mode = result_auth_mode
        self.result_metadata = result_metadata if result_metadata is not None else {"fake": True}
        self.result_output_text = result_output_text
        self.result_status = result_status

    def preflight(self) -> PreflightResult:
        self.events.append("executor-preflight")
        self.preflight_calls += 1
        return PreflightResult(
            executor=self.name,
            ok=True,
            version="fake-1",
            auth_mode="fake-local",
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls += 1
        self.events.append(f"executor:{request.task.task_id}")
        self.tasks.append(request.task)
        self.environments.append(dict(request.environment))
        failed = self.fail_on_call == self.calls
        status = (
            ExecutionStatus.INTERRUPTED
            if failed and self.result_status is ExecutionStatus.INTERRUPTED
            else ExecutionStatus.FAILED
            if failed
            else self.result_status
        )
        output_text = None if failed else self.result_output_text
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        (request.executor_dir / "stdout.log").write_text("fake\n", encoding="utf-8")
        deliverables_dir = request.deliverables_dir
        if self.outside_deliverables:
            deliverables_dir = request.workspace.parent.parent / "outside-deliverables"
            deliverables_dir.mkdir(parents=True, exist_ok=True)
        return ExecutionResult(
            runtime="test",
            task_id=request.task.task_id,
            executor=self.result_executor or self.name,
            executor_version=self.result_version,
            invocation_mode=self.result_invocation_mode or self.invocation_mode,
            # ``None`` is retained by the null-result fixture to test the
            # runner's persisted identity handling.
            auth_mode=cast(str, self.result_auth_mode),
            workspace=request.workspace,
            deliverables_dir=deliverables_dir,
            status=status,
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=1 if failed else 0,
            available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}) if output_text is not None else frozenset(),
            failure=(
                None
                if not failed
                else Failure(
                    FailureKind.INTERRUPTED if status is ExecutionStatus.INTERRUPTED else FailureKind.PROCESS,
                    "interrupted" if status is ExecutionStatus.INTERRUPTED else "test_failure",
                    FailureImpact.RUN,
                )
            ),
            output_text=output_text,
            metadata=self.result_metadata,
        )


class PlanOnlyJudge(JudgeExecutor):
    name = "plan-only-judge"
    invocation_mode = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        del environment
        return JudgePreflightResult(judge_executor=self.name, ok=True, version="fake-judge-1")

    def judge(self, request: JudgeRequest) -> JudgeResult:
        del request
        self.calls += 1
        raise AssertionError("pairwise plan validation must happen before judge calls")


class GenericRunnerTests(unittest.TestCase):
    def test_explicit_intervention_order_evidence_and_prompt_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []
            benchmark = FakeBenchmark(task_count=1, events=events, evaluation_metadata={"secret": "sentinel"})
            evaluator = FakeEvaluator(events=events)
            intervention = FakeIntervention(events=events)
            executor = FakeExecutor(events=events)

            with patch.dict(
                os.environ,
                {
                    "GDPVAL_CONDITION": "condition-label-sentinel",
                    "GDPVAL_CONDITION_FILE": "/external/secret/condition-source",
                    "GDPVAL_CONDITION_APPLIED": "true",
                },
            ):
                run_benchmark(
                    benchmark,
                    evaluator,
                    executor,
                    out_dir=root / "run",
                    limit=1,
                    intervention=intervention,
                )

            self.assertEqual(
                events[:3],
                ["evaluator-preflight", "intervention-preflight", "executor-preflight"],
            )
            self.assertLess(events.index("materialize:task-0"), events.index("execution-task:task-0"))
            self.assertLess(events.index("execution-task:task-0"), events.index("apply:task-0"))
            self.assertLess(events.index("apply:task-0"), events.index("executor:task-0"))
            self.assertLess(events.index("executor:task-0"), events.index("evaluate:task-0"))
            self.assertEqual(executor.tasks[0].task_id, "task-0")
            self.assertIn("OVERLAY FOR CONDITION SECRET", executor.tasks[0].prompt)
            executor_environment = executor.environments[0]
            for key in ("GDPVAL_CONDITION", "GDPVAL_CONDITION_FILE", "GDPVAL_CONDITION_APPLIED"):
                self.assertNotIn(key, executor_environment)
            serialized_environment = json.dumps(executor_environment)
            self.assertNotIn("condition-label-sentinel", serialized_environment)
            self.assertNotIn("/external/secret/condition-source", serialized_environment)
            self.assertNotIn("/external/secret/intervention-source", serialized_environment)
            self.assertNotIn("OVERLAY FOR CONDITION SECRET", serialized_environment)

            run_dir = root / "run"
            canonical = run_dir / "tasks" / "task-0" / "executor" / "task-prompt.txt"
            self.assertEqual(canonical.read_text(encoding="utf-8"), "prompt-0")
            metadata = json.loads((run_dir / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["intervention"]["id"], "fake-intervention-id")
            self.assertEqual(metadata["intervention"]["type"], "prompt-overlay")
            self.assertEqual(
                metadata["intervention"]["files"],
                [{"path": "overlay.txt", "size": len(b"secret intervention source"), "sha256": "a" * 64}],
            )
            self.assertEqual(
                metadata["intervention"]["application"],
                {"method": "prompt-overlay", "target": "task.prompt"},
            )
            self.assertNotIn("/external/secret/intervention-source", json.dumps(metadata))
            self.assertNotIn("sentinel", json.dumps(metadata))

            row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
            evidence = row["intervention"]
            self.assertEqual(evidence["status"], "applied")
            self.assertEqual(evidence["materialized_files"][0]["path"], "injected.txt")
            self.assertEqual(evidence["materialized_files"][0]["size"], 7)
            self.assertNotIn("/external/secret/intervention-source", json.dumps(row))
            self.assertNotIn("sentinel", json.dumps(row))
            self.assertNotIn("task-0", evidence["application_run_id"])

    def test_agent_skill_workspace_reference_is_portable_and_outer_only_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_parent = root / "source-parent-sentinel"
            skill_dir = source_parent / "demo-skill-sentinel"
            references_dir = skill_dir / "references"
            references_dir.mkdir(parents=True)
            skill_content = (
                "---\n"
                "name: demo-skill-sentinel\n"
                "description: A deterministic workspace reference test skill.\n"
                "---\n\n"
                "# skill-content-sentinel\n\n"
                "Use this reviewed skill.\n"
            )
            resource_content = "resource-content-sentinel\n"
            (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
            (references_dir / "guide.txt").write_text(resource_content, encoding="utf-8")
            source_reference = str(skill_dir.resolve())
            intervention_id = "agent-skill-id-sentinel"
            source_revision = "agent-skill-revision-sentinel"
            bundle = load_agent_skill_bundle(
                skill_dir,
                intervention_id=intervention_id,
                source_revision=source_revision,
            )
            intervention = AgentSkillIntervention(bundle, source_reference=source_reference)
            benchmark = FakeBenchmark(evaluation_metadata={"evaluation-metadata-sentinel": "secret"})
            executor = FakeExecutor()

            with patch.dict(
                os.environ,
                {
                    "GDPVAL_CONDITION": "condition-label-sentinel",
                    "GDPVAL_CONDITION_FILE": str(root / "condition-source-sentinel"),
                    "GDPVAL_CONDITION_APPLIED": "true",
                },
            ):
                run_benchmark(
                    benchmark,
                    FakeEvaluator(),
                    executor,
                    out_dir=root / "run",
                    limit=1,
                    intervention=intervention,
                )

            run_dir = root / "run"
            derived_prompt = executor.tasks[0].prompt
            skill_reference = ".gdpval/interventions/demo-skill-sentinel/SKILL.md"
            self.assertIn(skill_reference, derived_prompt)
            self.assertIn("prompt-0", derived_prompt)
            for value in (
                source_reference,
                intervention_id,
                source_revision,
                "evaluation-metadata-sentinel",
                "condition-label-sentinel",
            ):
                self.assertNotIn(value, derived_prompt)
            self.assertEqual(
                (run_dir / "tasks" / "task-0" / "executor" / "task-prompt.txt").read_text(encoding="utf-8"),
                "prompt-0",
            )

            workspace = run_dir / "tasks" / "task-0" / "workspace"
            workspace_files = sorted(
                path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()
            )
            workspace_contents = {path: (workspace / path).read_text(encoding="utf-8") for path in workspace_files}
            self.assertIn(skill_reference, workspace_files)
            self.assertIn(".gdpval/interventions/demo-skill-sentinel/references/guide.txt", workspace_files)
            self.assertEqual(workspace_contents[skill_reference], skill_content)
            self.assertEqual(
                workspace_contents[".gdpval/interventions/demo-skill-sentinel/references/guide.txt"],
                resource_content,
            )
            serialized_workspace = json.dumps({"files": workspace_files, "contents": workspace_contents})
            for value in (
                source_reference,
                intervention_id,
                source_revision,
                "evaluation-metadata-sentinel",
                "condition-label-sentinel",
            ):
                self.assertNotIn(value, serialized_workspace)

            executor_environment = executor.environments[0]
            for key in ("GDPVAL_CONDITION", "GDPVAL_CONDITION_FILE", "GDPVAL_CONDITION_APPLIED"):
                self.assertNotIn(key, executor_environment)
            serialized_environment = json.dumps(executor_environment)
            for value in (
                source_reference,
                intervention_id,
                source_revision,
                "evaluation-metadata-sentinel",
                "condition-label-sentinel",
            ):
                self.assertNotIn(value, serialized_environment)

            metadata = json.loads((run_dir / "run-metadata.json").read_text(encoding="utf-8"))
            descriptor = metadata["intervention"]
            self.assertEqual(descriptor["source_reference"], source_reference)
            self.assertEqual(descriptor["id"], intervention_id)
            self.assertEqual(descriptor["revision"], source_revision)
            self.assertTrue(descriptor["bundle_sha256"])
            self.assertTrue(descriptor["manifest_sha256"])
            self.assertIn(source_reference, json.dumps(metadata))

            row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
            task_result = json.loads((run_dir / "tasks" / "task-0" / "result.json").read_text(encoding="utf-8"))
            for payload in (row, task_result):
                serialized = json.dumps(payload)
                self.assertNotIn(source_reference, serialized)
                self.assertNotIn("source_reference", payload["intervention"])
                self.assertNotIn("evaluation-metadata-sentinel", serialized)
                self.assertNotIn("condition-label-sentinel", serialized)
            self.assertEqual(row["intervention"]["id"], intervention_id)
            self.assertEqual(row["intervention"]["revision"], source_revision)
            self.assertEqual(
                row["intervention"]["application"],
                {"method": "workspace-reference", "target": skill_reference},
            )
            self.assertIn(skill_reference, {item["path"] for item in row["intervention"]["materialized_files"]})

    def test_agent_skill_source_output_overlap_fails_before_executor_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_parent = root / "source-parent-sentinel"
            skill_dir = source_parent / "demo-skill-sentinel"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo-skill-sentinel\ndescription: overlap test\n---\n",
                encoding="utf-8",
            )
            (skill_dir / "resource.txt").write_text("resource-sentinel\n", encoding="utf-8")
            before = {
                path.relative_to(source_parent).as_posix(): path.read_bytes()
                for path in source_parent.rglob("*")
                if path.is_file()
            }
            alias = root / "source-alias-sentinel"
            alias.symlink_to(source_parent, target_is_directory=True)

            for output in (skill_dir / "run-output", alias / skill_dir.name / "run-output"):
                intervention = AgentSkillIntervention(
                    load_agent_skill_bundle(skill_dir),
                    source_reference=str(skill_dir.resolve()),
                )
                executor = FakeExecutor()
                with self.assertRaisesRegex(ValueError, "source and output paths must be separate"):
                    run_benchmark(
                        FakeBenchmark(task_count=1),
                        FakeEvaluator(),
                        executor,
                        out_dir=output,
                        limit=1,
                        intervention=intervention,
                    )
                self.assertEqual(executor.preflight_calls, 0)
                self.assertEqual(executor.calls, 0)
                self.assertFalse(output.exists())
                self.assertFalse((output / "run-metadata.json").exists())
                self.assertFalse((output / "tasks").exists())

            after = {
                path.relative_to(source_parent).as_posix(): path.read_bytes()
                for path in source_parent.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_agent_skill_relative_output_path_materializes_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "relative-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: relative-skill\ndescription: relative output test\n---\n",
                encoding="utf-8",
            )
            intervention = AgentSkillIntervention(load_agent_skill_bundle(skill_dir))
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    FakeExecutor(),
                    out_dir=Path("relative-run"),
                    limit=1,
                    intervention=intervention,
                )
            finally:
                os.chdir(original_cwd)

            target = (
                root
                / "relative-run"
                / "tasks"
                / "task-0"
                / "workspace"
                / ".gdpval"
                / "interventions"
                / "relative-skill"
                / "SKILL.md"
            )
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                (skill_dir / "SKILL.md").read_text(encoding="utf-8"),
            )

    def test_none_intervention_is_the_default_identity_application(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            run_benchmark(FakeBenchmark(task_count=1), FakeEvaluator(), executor, out_dir=out, limit=1)

            self.assertEqual(executor.tasks[0].prompt, "prompt-0")
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["intervention"]["id"], "none")
            self.assertEqual(metadata["intervention"]["type"], "none")
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["intervention"]["status"], "applied")
            self.assertEqual(row["intervention"]["materialized_files"], [])

    def test_run_persists_each_task_and_aggregates_completed_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            evaluator = FakeEvaluator()

            summary = run_benchmark(FakeBenchmark(), evaluator, executor, out_dir=out, limit=2)

            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.task_count, 2)
            self.assertEqual(summary.metrics, {"accuracy": 1.0})
            self.assertEqual(summary.evaluation_status_counts, {"completed": 2})
            self.assertEqual(set(summary.application_run_ids), {"task-0", "task-1"})
            self.assertEqual(evaluator.preflight_calls, 1)
            self.assertEqual(evaluator.calls, 2)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                summary.application_run_ids,
                {row["task_id"]: row["intervention"]["application_run_id"] for row in rows},
            )
            self.assertEqual(rows[0]["materialized"], ["input.txt"])
            self.assertEqual(rows[0]["evaluation"]["metrics"], {"accuracy": 1.0})
            self.assertEqual(rows[0]["evaluation"]["status"], "completed")
            self.assertEqual(
                (out / "tasks" / "task-0" / "executor" / "task-prompt.txt").read_text(encoding="utf-8"),
                "prompt-0",
            )
            self.assertTrue((out / "tasks" / "task-0" / "result.json").is_file())
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_tasks"], 2)
            self.assertEqual(metadata["evaluation_status_counts"], {"completed": 2})
            self.assertEqual(metadata["evaluator_id"], "fake-evaluator")

    def test_external_runtime_root_persists_task_runtime_and_keeps_judge_output_in_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            runtime = root / "runtime"
            summary = run_benchmark(
                FakeBenchmark(task_count=1),
                GDPvalExternalEvaluator(),
                FakeExecutor(),
                out_dir=out,
                limit=1,
                runtime_root=runtime,
            )

            runtime_task = runtime / "tasks" / "task-0"
            self.assertEqual(summary.runtime_root, runtime.resolve())
            self.assertEqual(summary.out_dir, out)
            self.assertTrue((runtime_task / "workspace" / "input.txt").is_file())
            self.assertTrue((runtime_task / "executor" / "stdout.log").is_file())
            self.assertTrue((runtime_task / "executor" / "task-prompt.txt").is_file())
            self.assertTrue((runtime_task / "result.json").is_file())
            self.assertTrue((runtime_task / "workspace" / "deliverables").is_dir())
            self.assertFalse((out / "tasks").exists())
            self.assertTrue((out / "results.jsonl").is_file())
            self.assertTrue((out / "run-metadata.json").is_file())
            self.assertTrue((out / "deliverables" / "task_task-0" / "repeat_0").is_dir())

            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 4)
            self.assertEqual(metadata["runtime_root"], str(runtime.resolve()))
            self.assertEqual(metadata["runtime_layout"], "external-persistent")

    def test_external_runtime_workspace_is_separate_from_control_profile_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control"
            profile = control / "profile"
            profile.mkdir(parents=True)
            (control / "AGENTS.md").write_text("control-profile-sentinel\n", encoding="utf-8")
            out = profile / "run-output"
            runtime = root / "neutral-runtime"
            assigned_workspaces: list[Path] = []

            class CapturingExecutor(FakeExecutor):
                def execute(self, request: ExecutionRequest) -> ExecutionResult:
                    assigned_workspaces.append(request.workspace)
                    return super().execute(request)

            run_benchmark(
                FakeBenchmark(task_count=1),
                FakeEvaluator(),
                CapturingExecutor(),
                out_dir=out,
                limit=1,
                runtime_root=runtime,
            )

            self.assertEqual(len(assigned_workspaces), 1)
            workspace = assigned_workspaces[0].resolve()
            runtime_resolved = runtime.resolve()
            control_resolved = control.resolve()
            self.assertIn(runtime_resolved, workspace.parents)
            self.assertNotIn(control_resolved, workspace.parents)
            # Harness path separation only; this does not claim OS read confinement.
            self.assertEqual(list(runtime.rglob("AGENTS.md")), [])
            self.assertFalse((workspace / "AGENTS.md").exists())

    def test_default_runtime_metadata_maps_to_canonical_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            summary = run_benchmark(FakeBenchmark(task_count=1), FakeEvaluator(), FakeExecutor(), out_dir=out, limit=1)

            self.assertEqual(summary.runtime_root, out.resolve())
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 4)
            self.assertEqual(metadata["runtime_root"], str(out.resolve()))
            self.assertEqual(metadata["runtime_layout"], "run-output")

    def test_external_runtime_root_must_be_fresh_and_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            runtime = root / "runtime"
            runtime.mkdir()
            marker = runtime / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            executor = FakeExecutor()

            with self.assertRaises(FileExistsError):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=1,
                    runtime_root=runtime,
                )

            self.assertEqual(executor.preflight_calls, 0)
            self.assertFalse(out.exists())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

            symlink_target = root / "runtime-target"
            symlink_target.mkdir()
            symlink_marker = symlink_target / "keep.txt"
            symlink_marker.write_text("keep", encoding="utf-8")
            runtime_link = root / "runtime-link"
            runtime_link.symlink_to(symlink_target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=1,
                    runtime_root=runtime_link,
                )
            self.assertFalse(out.exists())
            self.assertEqual(symlink_marker.read_text(encoding="utf-8"), "keep")

            with self.assertRaises(ValueError):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=1,
                    runtime_root=out / "runtime",
                )
            self.assertFalse(out.exists())

    def test_external_runtime_root_intervention_source_overlap_fails_before_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            skill = source / "skill"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                "---\nname: skill\ndescription: overlap\n---\n",
                encoding="utf-8",
            )
            runtime = skill / "runtime"
            out = root / "run"
            executor = FakeExecutor()
            intervention = AgentSkillIntervention(load_agent_skill_bundle(skill))

            with self.assertRaisesRegex(ValueError, "source and output paths must be separate"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=1,
                    intervention=intervention,
                    runtime_root=runtime,
                )

            self.assertEqual(executor.preflight_calls, 0)
            self.assertFalse(out.exists())
            self.assertFalse(runtime.exists())

    def test_external_status_is_visible_and_not_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            summary = run_benchmark(
                FakeBenchmark(task_count=1),
                FakeEvaluator(statuses=(EvaluationStatus.EXTERNAL,)),
                FakeExecutor(),
                out_dir=out,
                limit=1,
            )
            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.metrics, {})
            self.assertEqual(summary.evaluation_status_counts, {"external": 1})
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["evaluation"]["status"], "external")

    def test_gdpval_handoff_matches_existing_consumers_and_task_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            benchmark = FakeBenchmark(task_count=1)
            benchmark.name = "gdpval"
            summary = run_benchmark(
                benchmark,
                GDPvalExternalEvaluator(),
                FakeExecutor(),
                out_dir=out,
                limit=1,
            )
            self.assertEqual(summary.evaluation_status_counts, {"external": 1})
            deliverables = out / "deliverables"
            discovered = discover_tasks(deliverables)
            self.assertEqual(set(discovered), {"task_task-0"})
            self.assertEqual(discovered["task_task-0"].name, "repeat_0")
            self.assertEqual(_candidate_task_prompt(deliverables, "task_task-0"), "prompt-0")

    def test_evaluator_preflight_precedes_executor_and_benchmark(self) -> None:
        events: list[str] = []

        class OrderedBenchmark(FakeBenchmark):
            def prepare(self) -> None:
                events.append("benchmark")

        class OrderedEvaluator(FakeEvaluator):
            def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
                events.append("evaluator")
                return super().preflight(run_dir)

        class OrderedExecutor(FakeExecutor):
            def preflight(self) -> PreflightResult:
                events.append("executor")
                return super().preflight()

        with tempfile.TemporaryDirectory() as tmp:
            run_benchmark(
                OrderedBenchmark(task_count=1),
                OrderedEvaluator(),
                OrderedExecutor(),
                out_dir=Path(tmp) / "run",
                limit=1,
            )
        self.assertEqual(events[:3], ["evaluator", "executor", "benchmark"])

    def test_evaluator_preflight_failure_stops_before_executor_or_benchmark(self) -> None:
        class NotReadyEvaluator(FakeEvaluator):
            def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
                del run_dir
                self.preflight_calls += 1
                return EvaluatorPreflightResult(
                    self.name,
                    self.evaluator_type,
                    False,
                    details=("math-verify==0.8.0 is unavailable",),
                )

        class RecordingBenchmark(FakeBenchmark):
            def __init__(self) -> None:
                super().__init__(task_count=1)
                self.prepared = False

            def prepare(self) -> None:
                self.prepared = True

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            benchmark = RecordingBenchmark()
            executor = FakeExecutor()
            with self.assertRaisesRegex(RuntimeError, "math-verify==0.8.0"):
                run_benchmark(benchmark, NotReadyEvaluator(), executor, out_dir=out, limit=1)
            self.assertFalse(benchmark.prepared)
            self.assertEqual(executor.calls, 0)
            self.assertFalse(out.exists())

    def test_intervention_preflight_failure_stops_before_executor_or_benchmark(self) -> None:
        class NotReadyIntervention(FakeIntervention):
            def preflight(self) -> InterventionPreflightResult:
                self.events.append("intervention-preflight")
                return InterventionPreflightResult(
                    name=self.name,
                    intervention_type=self.intervention_type,
                    ok=False,
                    details=("intervention source unavailable",),
                )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            events: list[str] = []
            benchmark = FakeBenchmark(events=events)
            executor = FakeExecutor(events=events)
            with self.assertRaisesRegex(RuntimeError, "intervention source unavailable"):
                run_benchmark(
                    benchmark,
                    FakeEvaluator(events=events),
                    executor,
                    out_dir=out,
                    limit=1,
                    intervention=NotReadyIntervention(events=events),
                )
            self.assertEqual(executor.calls, 0)
            self.assertEqual(events[:2], ["evaluator-preflight", "intervention-preflight"])
            self.assertFalse(out.exists())

    def test_all_evaluation_plans_validate_before_any_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            evaluator = FakeEvaluator(invalid_plan_task_id="task-1")
            with self.assertRaisesRegex(ValueError, "invalid evaluation plan"):
                run_benchmark(
                    FakeBenchmark(task_count=2),
                    evaluator,
                    executor,
                    out_dir=out,
                    limit=2,
                )
            self.assertEqual(executor.calls, 0)
            self.assertFalse(out.exists())

    def test_later_intervention_task_validation_stops_before_any_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            with self.assertRaisesRegex(ValueError, "invalid intervention task"):
                run_benchmark(
                    FakeBenchmark(task_count=2),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=2,
                    intervention=FakeIntervention(invalid_task_id="task-1"),
                )
            self.assertEqual(executor.calls, 0)
            self.assertFalse(out.exists())

    def test_intervention_failure_persists_evidence_and_stops_later_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            intervention = FakeIntervention(fail_task_id="task-1")
            with self.assertRaisesRegex(RuntimeError, "intervention application secret failure"):
                run_benchmark(
                    FakeBenchmark(task_count=3),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=3,
                    intervention=intervention,
                )

            self.assertEqual(executor.calls, 1)
            self.assertEqual([task.task_id for task in executor.tasks], ["task-0"])
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["intervention"]["status"], "applied")
            self.assertEqual(rows[1]["intervention"]["status"], "failed")
            self.assertEqual(rows[1]["execution"], None)
            self.assertNotIn("secret failure", json.dumps(rows[1]))
            self.assertTrue((out / "tasks" / "task-0" / "executor" / "stdout.log").is_file())
            self.assertTrue((out / "tasks" / "task-0" / "result.json").is_file())
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["failure"]["phase"], "intervention_apply")

    def test_intervention_application_mismatch_fails_before_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            with self.assertRaisesRegex(ValueError, "mismatched bundle_sha256"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    limit=1,
                    intervention=FakeIntervention(mismatch_application="bundle"),
                )

            self.assertEqual(executor.calls, 0)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["intervention"]["status"], "failed")
            self.assertIsNone(rows[0]["execution"])

    def test_pairwise_injection_rejects_generic_single_candidate_plan_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor()
            judge = PlanOnlyJudge()
            evaluator = PairwiseJudgeEvaluator(judge)
            with self.assertRaisesRegex(ValueError, "exactly two candidates"):
                run_benchmark(FakeBenchmark(task_count=2), evaluator, executor, out_dir=out, limit=2)
            self.assertEqual(executor.calls, 0)
            self.assertEqual(judge.calls, 0)
            self.assertFalse(out.exists())

    def test_evaluator_exception_persists_completed_execution_and_fails_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "evaluator test failure"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(fail=True),
                    FakeExecutor(),
                    out_dir=out,
                    limit=1,
                )

            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["execution"]["status"], "no_deliverable")
            self.assertEqual(row["evaluation"]["status"], "failed")
            self.assertEqual(row["evaluation"]["metrics"], {})
            self.assertEqual(row["evaluation"]["outcomes"], {})
            persisted = json.loads((out / "tasks" / "task-0" / "result.json").read_text())
            self.assertEqual(persisted["evaluation"]["status"], "failed")
            metadata = json.loads((out / "run-metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["evaluation_status_counts"], {"failed": 1})
            self.assertNotIn("evaluator test failure", json.dumps(row))

    def test_executor_failure_stops_before_evaluation_and_aborts_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            executor = FakeExecutor(fail_on_call=1)
            evaluator = FakeEvaluator()

            with self.assertRaises(RunAbort) as raised:
                run_benchmark(FakeBenchmark(), evaluator, executor, out_dir=out, limit=3)

            self.assertEqual(str(raised.exception), "test_failure")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(evaluator.calls, 0)
            rows = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            row = json.loads(rows[0])
            self.assertEqual(row["execution"]["status"], "failed")
            self.assertEqual(row["execution"]["available_outputs"], [])
            self.assertEqual(
                row["execution"]["failure"],
                {"kind": "process", "code": "test_failure", "impact": "run"},
            )
            self.assertEqual(row["evaluation"]["status"], "skipped")
            self.assertEqual(row["evaluation"]["metrics"], {})
            self.assertEqual(row["evaluation"]["outcomes"], {})
            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["evaluation_status_counts"], {"skipped": 1})

            interrupt_out = Path(tmp) / "interrupt-run"
            interrupt_executor = FakeExecutor(fail_on_call=1, result_status=ExecutionStatus.INTERRUPTED)
            with self.assertRaises(RunAbort) as interrupted:
                run_benchmark(FakeBenchmark(), FakeEvaluator(), interrupt_executor, out_dir=interrupt_out, limit=3)
            self.assertEqual(str(interrupted.exception), "interrupted")
            interrupt_metadata = json.loads((interrupt_out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(interrupt_metadata["status"], "interrupted")

    def test_reproducibility_metadata_is_typed_and_secret_safe(self) -> None:
        repository = RepositoryProvenance("a" * 40, "available", "clean")
        evaluator = FakeEvaluator(
            preflight_details=("preflight-secret",),
            judge_fields={
                "judge_executor": "judge-executor",
                "judge_executor_version": "judge-version",
                "judge_auth_mode": "judge-auth",
                "judge_model": "judge-model",
            },
        )
        evaluator.evaluator_type = EvaluatorType.LLM_RUBRIC
        executor = FakeExecutor(
            result_metadata={
                "credential": "credential-secret",
                "command": "command-secret",
                "environment": "environment-secret",
            },
            result_output_text="output-secret",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            with (
                patch.object(runner_module, "repository_provenance", return_value=repository),
                patch.dict(os.environ, {"HARNESS_CREDENTIAL_SECRET": "environment-secret"}),
            ):
                summary = run_benchmark(
                    FakeBenchmark(task_count=1),
                    evaluator,
                    executor,
                    out_dir=out,
                    limit=1,
                    model="model-name",
                )

            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            execution = row["execution"]
            self.assertEqual(summary.status, "completed")
            self.assertTrue((out / "tasks" / "task-0" / "executor" / "stdout.log").is_file())
            self.assertTrue((out / "tasks" / "task-0" / "result.json").is_file())
            self.assertEqual(metadata["schema_version"], 4)
            self.assertIsInstance(metadata["finished_at"], str)
            self.assertTrue(metadata["finished_at"])
            self.assertEqual(
                metadata["repository"],
                {
                    "commit": "a" * 40,
                    "revision_status": "available",
                    "worktree_status": "clean",
                },
            )
            self.assertEqual(metadata["benchmark_revision_status"], "available")
            self.assertEqual(metadata["evaluator"]["preflight_details"], [])
            self.assertEqual(metadata["evaluator"]["preflight_detail_count"], 1)
            self.assertEqual(
                metadata["evaluator"]["judge"],
                {
                    "applicable": True,
                    "executor": "judge-executor",
                    "version": "judge-version",
                    "auth_mode": "judge-auth",
                    "model": "judge-model",
                },
            )
            self.assertEqual(metadata["judge"], metadata["evaluator"]["judge"])
            self.assertEqual(
                set(metadata["executor_descriptor"]),
                {"id", "version", "invocation_mode", "auth_mode", "model", "network_policy"},
            )
            self.assertEqual(execution["metadata"], {})
            self.assertNotIn("output_text", execution)
            self.assertEqual(row["task_sha256"], hashlib.sha256(b"prompt-0").hexdigest())
            self.assertEqual(summary.application_run_ids, {"task-0": row["intervention"]["application_run_id"]})
            self.assertEqual(
                metadata["configuration_sha256"],
                canonical_json_sha256(metadata["configuration"]),
            )
            self.assertEqual(
                metadata["run_fingerprint_sha256"],
                canonical_json_sha256(
                    {
                        "configuration_sha256": metadata["configuration_sha256"],
                        "repository": metadata["repository"],
                        "tasks": metadata["tasks"],
                    }
                ),
            )
            configuration_text = json.dumps(metadata["configuration"], sort_keys=True)
            self.assertNotIn(str(out), configuration_text)
            self.assertNotIn("preflight-secret", configuration_text)
            self.assertNotIn("status", metadata["configuration"]["intervention"])
            self.assertNotIn("source_reference", metadata["configuration"]["intervention"])
            serialized = json.dumps({"metadata": metadata, "row": row})
            for secret in (
                "preflight-secret",
                "credential-secret",
                "command-secret",
                "environment-secret",
                "output-secret",
            ):
                self.assertNotIn(secret, serialized)

    def test_judge_nonapplicability_and_unavailable_benchmark_revision_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            benchmark = FakeBenchmark(task_count=1)
            # Preserve an unavailable benchmark revision in this fixture.
            benchmark.revision = cast(str, None)
            run_benchmark(
                benchmark,
                FakeEvaluator(judge_fields={"judge_executor": "ignored-non-judge"}),
                FakeExecutor(),
                out_dir=out,
                limit=1,
            )

            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["benchmark_revision_status"], "unavailable")
            self.assertIsNone(metadata["benchmark_revision"])
            self.assertEqual(metadata["configuration"]["benchmark"]["revision_status"], "unavailable")
            self.assertEqual(
                metadata["judge"],
                {
                    "applicable": False,
                    "executor": None,
                    "version": None,
                    "auth_mode": None,
                    "model": None,
                },
            )

    def test_judge_applicability_is_typed_even_when_judge_details_are_unavailable(self) -> None:
        evaluator = FakeEvaluator()
        evaluator.evaluator_type = EvaluatorType.PAIRWISE
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            run_benchmark(FakeBenchmark(task_count=1), evaluator, FakeExecutor(), out_dir=out, limit=1)

            metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["judge"],
                {
                    "applicable": True,
                    "executor": None,
                    "version": None,
                    "auth_mode": None,
                    "model": None,
                },
            )

    def test_duplicate_and_safe_task_id_collisions_fail_before_execution_or_roots(self) -> None:
        class CollisionBenchmark(FakeBenchmark):
            def __init__(self, task_ids: tuple[str, ...]) -> None:
                super().__init__(task_count=len(task_ids))
                self.task_ids = task_ids

            def load_tasks(self, limit: int) -> list[BenchmarkTask]:
                del limit
                return [
                    BenchmarkTask(TaskSpec(task_id=task_id, prompt=f"prompt-{index}"))
                    for index, task_id in enumerate(self.task_ids)
                ]

        cases = (
            (("duplicate", "duplicate"), "duplicate task_id"),
            (("a/b", "a?b"), "collide after safe normalization"),
        )
        for index, (task_ids, message) in enumerate(cases):
            with self.subTest(task_ids=task_ids), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / "out"
                runtime = root / "runtime"
                executor = FakeExecutor()
                with self.assertRaisesRegex(ValueError, message):
                    run_benchmark(
                        CollisionBenchmark(task_ids),
                        FakeEvaluator(),
                        executor,
                        out_dir=out,
                        runtime_root=runtime,
                        limit=len(task_ids),
                    )
                self.assertEqual(executor.calls, 0)
                self.assertFalse(out.exists())
                self.assertFalse(runtime.exists())

    def test_reproducibility_hashes_ignore_paths_timestamps_and_random_application_ids(self) -> None:
        repository = RepositoryProvenance("b" * 40, "available", "dirty")

        def run_once(root: Path, *, model: str | None = None, prompt_prefix: str = "") -> dict[str, object]:
            with patch.object(runner_module, "repository_provenance", return_value=repository):
                run_benchmark(
                    FakeBenchmark(task_count=1, prompt_prefix=prompt_prefix),
                    FakeEvaluator(),
                    FakeExecutor(),
                    out_dir=root / "run",
                    limit=1,
                    model=model,
                )
            return _load_json_object(root / "run" / "run-metadata.json")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = run_once(root / "first")
            second = run_once(root / "second")
            self.assertEqual(first["configuration_sha256"], second["configuration_sha256"])
            self.assertEqual(first["run_fingerprint_sha256"], second["run_fingerprint_sha256"])
            self.assertEqual(first["tasks"], second["tasks"])

            changed_config = run_once(root / "config", model="different-model")
            self.assertNotEqual(first["configuration_sha256"], changed_config["configuration_sha256"])
            self.assertNotEqual(first["run_fingerprint_sha256"], changed_config["run_fingerprint_sha256"])

            changed_task = run_once(root / "task", prompt_prefix="changed-")
            self.assertEqual(first["configuration_sha256"], changed_task["configuration_sha256"])
            self.assertNotEqual(first["run_fingerprint_sha256"], changed_task["run_fingerprint_sha256"])
            self.assertNotEqual(first["tasks"], changed_task["tasks"])

    def test_repository_unavailable_and_dirty_records_are_safe(self) -> None:
        cases = (
            RepositoryProvenance(None, "unavailable", "unavailable"),
            RepositoryProvenance("c" * 40, "available", "dirty"),
        )
        for index, repository in enumerate(cases):
            with self.subTest(repository=repository), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                with patch.object(runner_module, "repository_provenance", return_value=repository):
                    run_benchmark(
                        FakeBenchmark(task_count=1),
                        FakeEvaluator(),
                        FakeExecutor(),
                        out_dir=out,
                        limit=1,
                    )
                metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["repository"]["revision_status"], repository.revision_status)
                self.assertEqual(metadata["repository"]["worktree_status"], repository.worktree_status)
                self.assertEqual(metadata["repository"]["commit"], repository.commit)

    def test_executor_identity_and_typed_result_mismatches_persist_before_stopping(self) -> None:
        cases: tuple[tuple[_ExecutorOptions, str], ...] = (
            ({"result_executor": "tampered-executor"}, "mismatched executor"),
            ({"result_invocation_mode": "tampered-mode"}, "mismatched invocation_mode"),
            ({"result_version": "tampered-version"}, "mismatched executor_version"),
            ({"result_auth_mode": "tampered-auth"}, "mismatched auth_mode"),
        )
        for index, (executor_options, message) in enumerate(cases):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / f"run-{index}"
                evaluator = FakeEvaluator()
                with self.assertRaisesRegex(ValueError, message):
                    run_benchmark(
                        FakeBenchmark(task_count=1),
                        evaluator,
                        FakeExecutor(**executor_options),
                        out_dir=out,
                        limit=1,
                    )
                self.assertEqual(evaluator.calls, 0)
                row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
                metadata = json.loads((out / "run-metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(row["evaluation"]["status"], "failed")
                self.assertEqual(metadata["status"], "failed")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "null-result-fields"
            summary = run_benchmark(
                FakeBenchmark(task_count=1),
                FakeEvaluator(),
                FakeExecutor(result_version=None, result_auth_mode=None),
                out_dir=out,
                limit=1,
            )
            self.assertEqual(summary.status, "completed")
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertIsNone(row["execution"]["executor_version"])
            self.assertIsNone(row["execution"]["auth_mode"])

    def test_executor_preflight_identity_mismatch_stops_before_execution_or_roots(self) -> None:
        class TamperedPreflightExecutor(FakeExecutor):
            def preflight(self) -> PreflightResult:
                result = super().preflight()
                return PreflightResult(
                    executor="tampered-preflight-executor",
                    ok=result.ok,
                    version=result.version,
                    auth_mode=result.auth_mode,
                    details=result.details,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            runtime = root / "runtime"
            executor = TamperedPreflightExecutor()
            with self.assertRaisesRegex(ValueError, "preflight returned a mismatched executor"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    FakeEvaluator(),
                    executor,
                    out_dir=out,
                    runtime_root=runtime,
                    limit=1,
                )
            self.assertEqual(executor.calls, 0)
            self.assertFalse(out.exists())
            self.assertFalse(runtime.exists())

    def test_mismatched_deliverables_are_rejected_before_evaluator_and_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            evaluator = FakeEvaluator()
            with self.assertRaisesRegex(ValueError, "deliverables"):
                run_benchmark(
                    FakeBenchmark(task_count=1),
                    evaluator,
                    FakeExecutor(outside_deliverables=True),
                    out_dir=out,
                    limit=1,
                )
            self.assertEqual(evaluator.calls, 0)
            row = json.loads((out / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["evaluation"]["status"], "failed")
            self.assertFalse((out / "deliverables").exists())

    def test_existing_run_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_benchmark(FakeBenchmark(), FakeEvaluator(), FakeExecutor(), out_dir=out, limit=1)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
