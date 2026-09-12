# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Mapping, cast
from unittest.mock import patch

from gdpval_harness.benchmarks.aime26 import AIME26Benchmark
from gdpval_harness.benchmarks.base import Benchmark, BenchmarkTask
from gdpval_harness.benchmarks.bigcodebench import BigCodeBenchBenchmark
from gdpval_harness.benchmarks.gdpval import GDPvalBenchmark
from gdpval_harness.benchmarks.registry import create_benchmark, get_benchmark_descriptor, list_benchmarks
from gdpval_harness.evaluators.aime26 import AIME26Evaluator, _math_verify_preflight, _native_math_evaluate
from gdpval_harness.evaluators.base import (
    EvaluationCandidate,
    EvaluationPlan,
    EvaluationRequest,
    EvaluationStatus,
    require_candidates,
    require_one_candidate,
    require_two_candidates,
)
from gdpval_harness.evaluators.bigcodebench import BigCodeBenchEvaluator, _native_bigcodebench_evaluate
from gdpval_harness.evaluators.exact import ExactMatchEvaluator
from gdpval_harness.evaluators.gdpval import GDPvalExternalEvaluator
from gdpval_harness.evaluators.pairwise import PairwiseJudgeEvaluator, sanitize_environment
from gdpval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec
from gdpval_harness.interventions.agent_skill import AgentSkillIntervention, load_agent_skill_bundle
from gdpval_harness.interventions.base import (
    ApplicationMapping,
    InterventionApplication,
    InterventionBundle,
    InterventionFile,
    InterventionManifest,
    InterventionType,
    canonical_manifest_bytes,
    ensure_destination_parents,
    ensure_source_output_separation,
    file_evidence,
    revision_fields,
)
from gdpval_harness.interventions.files import FilesIntervention
from gdpval_harness.interventions.none import NoneIntervention
from gdpval_harness.interventions.prompt_overlay import (
    MAX_PROMPT_OVERLAY_BYTES,
    PromptOverlayIntervention,
    apply_prompt_overlay,
)
from gdpval_harness.interventions.registry import create_intervention, get_intervention
from gdpval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult, Verdict
from gdpval_harness.judges.pairwise import (
    aggregate,
    build_judge_prompt,
    discover_tasks,
    initial_swap,
    matched_tasks,
    normalize_verdict,
    prepare_trial,
    tree_hash,
    validate_reference_equivalence,
    write_trial_metadata,
)


def _execution_result(
    root: Path,
    *,
    task_id: str = "task-1",
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    output_text: str | None = "answer",
    deliverables: Path | None = None,
) -> ExecutionResult:
    workspace = root / "workspace"
    return ExecutionResult(
        task_id=task_id,
        executor="fake",
        executor_version="fake-1",
        invocation_mode="fake",
        auth_mode="fake",
        workspace=workspace,
        deliverables_dir=deliverables or workspace / "deliverables",
        status=status,
        started_at="2026-09-12T00:00:00+00:00",
        finished_at="2026-09-12T00:00:01+00:00",
        exit_code=0 if status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE} else 1,
        output_text=output_text,
    )


def _candidate(
    root: Path,
    candidate_id: str = "policy",
    *,
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    output_text: str | None = "answer",
) -> EvaluationCandidate:
    result = _execution_result(root, status=status, output_text=output_text)
    return EvaluationCandidate(candidate_id, result, result.deliverables_dir)


def _skill_source(root: Path, *, name: str = "demo-skill", body: str = "Use the supplied files.\n") -> Path:
    source = root / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(f"---\nname: {name}\ndescription: A test skill\n---\n\n{body}", encoding="utf-8")
    (source / "helper.txt").write_text("helper\n", encoding="utf-8")
    return source


class InterventionBoundaryTests(unittest.TestCase):
    def test_contract_records_reject_invalid_paths_hashes_and_mappings(self) -> None:
        with self.assertRaises(ValueError):
            InterventionFile("", 0, "0" * 64)
        with self.assertRaises(ValueError):
            InterventionFile("ok", -1, "0" * 64)
        with self.assertRaises(ValueError):
            InterventionFile("ok", 0, "A" * 64)
        with self.assertRaises(ValueError):
            ApplicationMapping("", None)
        with self.assertRaises(ValueError):
            ApplicationMapping("method", "")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = ApplicationMapping("files", ".")
            manifest = InterventionManifest(
                intervention_id="id",
                intervention_type=InterventionType.FILES,
                source_revision=None,
                revision_status="unavailable",
                files=(file_evidence("a.txt", b"a"),),
                bundle_sha256="0" * 64,
                application=app,
            )
            self.assertEqual(canonical_manifest_bytes(manifest), canonical_manifest_bytes(manifest))
            with self.assertRaises(ValueError):
                InterventionManifest(
                    intervention_id="id",
                    intervention_type=InterventionType.FILES,
                    source_revision=None,
                    revision_status="unavailable",
                    files=(file_evidence("a.txt", b"a"),),
                    bundle_sha256="0" * 64,
                    application=app,
                    manifest_sha256="1" * 64,
                )
            with self.assertRaises(ValueError):
                InterventionApplication(
                    application_run_id="run",
                    task=TaskSpec("task", "prompt"),
                    materialized_files=(file_evidence("../unsafe", b"x"),),
                    bundle_sha256="0" * 64,
                    manifest_sha256="0" * 64,
                    application=app,
                )
            self.assertTrue(root.is_dir())

    def test_intervention_destination_and_source_separation_reject_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            with self.assertRaises(ValueError):
                ensure_destination_parents(workspace, ["nested/../bad.txt"])
            (workspace / "nested").write_text("file", encoding="utf-8")
            with self.assertRaises(ValueError):
                ensure_destination_parents(workspace, ["nested/file.txt"])
            source = root / "source"
            source.mkdir()
            bundle = InterventionBundle(
                source,
                InterventionManifest(
                    intervention_id="files",
                    intervention_type=InterventionType.FILES,
                    source_revision=None,
                    revision_status="unavailable",
                    files=(file_evidence("a.txt", b"a"),),
                    bundle_sha256="0" * 64,
                    application=ApplicationMapping("files", "."),
                ),
            )
            with self.assertRaises(ValueError):
                ensure_source_output_separation(bundle, source / "nested")
            with self.assertRaises(ValueError):
                ensure_source_output_separation(bundle, source)
            self.assertEqual(revision_fields(None, applicable=False), (None, "not-applicable"))
            self.assertEqual(revision_fields(None, applicable=True), (None, "unavailable"))
            self.assertEqual(revision_fields("rev", applicable=True), ("rev", "available"))
            with self.assertRaises(ValueError):
                revision_fields("", applicable=True)

    def test_files_intervention_preflight_rejects_empty_symlink_special_and_collision_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, setup in (
                ("missing", lambda path: None),
                ("empty", lambda path: path.mkdir()),
            ):
                source = root / name
                setup(source)
                result = FilesIntervention(source).preflight()
                self.assertFalse(result.ok)
            source = root / "symlink-root-target"
            source.mkdir()
            symlink = root / "symlink-root"
            symlink.symlink_to(source, target_is_directory=True)
            self.assertFalse(FilesIntervention(symlink).preflight().ok)

            source = root / "unsafe"
            source.mkdir()
            (source / "deliverables").write_text("reserved", encoding="utf-8")
            self.assertFalse(FilesIntervention(source).preflight().ok)

            source = root / "collision"
            source.mkdir()
            (source / "Readme").write_text("a", encoding="utf-8")
            (source / "README").write_text("b", encoding="utf-8")
            collision = FilesIntervention(source).preflight()
            self.assertFalse(collision.ok)
            self.assertIn("collision", collision.details[0])

            source = root / "special"
            source.mkdir()
            fifo = source / "pipe"
            os.mkfifo(fifo)
            special = FilesIntervention(source).preflight()
            self.assertFalse(special.ok)
            self.assertIn("special file", special.details[0])

    def test_files_intervention_requires_preflight_detects_tamper_and_destination_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "nested").mkdir()
            (source / "nested" / "file.txt").write_text("before", encoding="utf-8")
            intervention = FilesIntervention(source, source_revision="rev")
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                intervention.apply(TaskSpec("task", "prompt"), root / "workspace", application_run_id="run")
            ready = intervention.preflight()
            self.assertTrue(ready.ok)
            (source / "nested" / "file.txt").write_text("after", encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                intervention.apply(TaskSpec("task", "prompt"), workspace, application_run_id="run")

            source2 = root / "source2"
            source2.mkdir()
            (source2 / "file.txt").write_text("content", encoding="utf-8")
            intervention2 = FilesIntervention(source2)
            self.assertTrue(intervention2.preflight().ok)
            destination = root / "workspace2"
            destination.mkdir()
            (destination / "file.txt").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                intervention2.apply(TaskSpec("task", "prompt"), destination, application_run_id="run")

    def test_prompt_overlay_validation_and_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = PromptOverlayIntervention(root / "missing")
            self.assertFalse(missing.preflight().ok)
            directory = root / "directory"
            directory.mkdir()
            self.assertFalse(PromptOverlayIntervention(directory).preflight().ok)
            binary = root / "binary"
            binary.write_bytes(b"\xff")
            self.assertFalse(PromptOverlayIntervention(binary).preflight().ok)
            empty = root / "empty"
            empty.write_text(" \n", encoding="utf-8")
            self.assertFalse(PromptOverlayIntervention(empty).preflight().ok)
            huge = root / "huge"
            huge.write_bytes(b"x" * (MAX_PROMPT_OVERLAY_BYTES + 1))
            self.assertFalse(PromptOverlayIntervention(huge).preflight().ok)

            source = root / "overlay.txt"
            source.write_text("review this", encoding="utf-8")
            intervention = PromptOverlayIntervention(source)
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                intervention.apply(TaskSpec("task", "original"), root, application_run_id="run")
            self.assertTrue(intervention.preflight().ok)
            (source).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                intervention.apply(TaskSpec("task", "original"), root, application_run_id="run")
            self.assertEqual(apply_prompt_overlay(TaskSpec("task", "body"), "prefix").task_id, "task")

    def test_agent_skill_validation_and_apply_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing"
            with self.assertRaisesRegex(ValueError, "invalid Agent Skill"):
                load_agent_skill_bundle(missing)
            source = _skill_source(root)
            intervention = AgentSkillIntervention.from_source(source, source_revision="rev")
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                intervention.apply(TaskSpec("task", "body"), root / "workspace", application_run_id="run")
            ready = intervention.preflight()
            self.assertTrue(ready.ok)
            workspace = root / "workspace"
            workspace.mkdir()
            application = intervention.apply(TaskSpec("task", "body"), workspace, application_run_id="run")
            self.assertIn("BEGIN AGENT SKILL", application.task.prompt)
            self.assertEqual(
                (workspace / ".gdpval" / "interventions" / "demo-skill" / "SKILL.md").read_text()[:3], "---"
            )
            with self.assertRaises(FileExistsError):
                intervention.apply(TaskSpec("task", "body"), workspace, application_run_id="run-2")

            bad_name = root / "BadName"
            bad_name.mkdir()
            (bad_name / "SKILL.md").write_text("---\nname: BadName\ndescription: bad\n---\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid Agent Skill"):
                load_agent_skill_bundle(bad_name)

            unsupported = root / "unsupported"
            unsupported.mkdir()
            (unsupported / "SKILL.md").write_text(
                "---\nname: unsupported\ndescription: bad\nunknown: value\n---\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_agent_skill_bundle(unsupported)

            malformed = root / "malformed"
            malformed.mkdir()
            (malformed / "SKILL.md").write_bytes(b"---\nname: [\n---\n")
            with self.assertRaisesRegex(ValueError, "invalid Agent Skill"):
                load_agent_skill_bundle(malformed)

    def test_intervention_factories_and_none_contract_failures(self) -> None:
        self.assertIsInstance(create_intervention("none"), NoneIntervention)
        self.assertIsInstance(get_intervention(InterventionType.NONE), NoneIntervention)
        none = NoneIntervention()
        with self.assertRaisesRegex(RuntimeError, "preflight"):
            none.apply(TaskSpec("task", "prompt"), Path("."), application_run_id="run")
        self.assertTrue(none.preflight().ok)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unknown intervention"):
                create_intervention("unknown")
            with self.assertRaisesRegex(ValueError, "requires an explicit source"):
                create_intervention("files")
            with self.assertRaisesRegex(ValueError, "does not accept"):
                create_intervention("none", source=Path(tmp) / "source")
            source = Path(tmp) / "overlay.txt"
            source.write_text("overlay", encoding="utf-8")
            self.assertIsInstance(create_intervention("prompt-overlay", source=source), PromptOverlayIntervention)


class PairwiseJudgeBoundaryTests(unittest.TestCase):
    @staticmethod
    def _candidate(root: Path, name: str, *, reference: str | None = "source") -> Path:
        candidate = root / name
        task = candidate / "task_1" / "repeat_0"
        task.mkdir(parents=True)
        (task / "answer.txt").write_text(f"{name}\n", encoding="utf-8")
        (candidate / "finish_params.json").write_text("ignored\n", encoding="utf-8")
        if reference is not None:
            (candidate / "reference_files").mkdir()
            (candidate / "reference_files" / "source.txt").write_text(reference, encoding="utf-8")
        return candidate

    def test_pairwise_discovery_matching_staging_and_metadata_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_a = self._candidate(root, "candidate-a")
            candidate_b = self._candidate(root, "candidate-b")
            self.assertEqual(tuple(discover_tasks(candidate_a)), ("task_1",))
            self.assertEqual(len(matched_tasks(candidate_a, candidate_b)), 1)
            validate_reference_equivalence(candidate_a, candidate_b)
            before = tree_hash(candidate_a / "reference_files")

            prepared = prepare_trial(
                root / "out",
                "task-1",
                candidate_a,
                candidate_b,
                trial_index=0,
                seed=42,
            )
            self.assertTrue(prepared.workspace.is_dir())
            self.assertTrue((prepared.reference_dir / "source.txt").is_file())
            self.assertFalse((prepared.submission_a_dir / "finish_params.json").exists())
            self.assertEqual(tree_hash(candidate_a / "reference_files"), before)
            self.assertIn("BOXED[TIE]", build_judge_prompt("do the work"))
            self.assertEqual(normalize_verdict(Verdict.A, swapped=False), Verdict.A)
            self.assertEqual(normalize_verdict(Verdict.A, swapped=True), Verdict.B)
            self.assertEqual(normalize_verdict(Verdict.TIE, swapped=True), Verdict.TIE)
            self.assertEqual(aggregate([]), {"trials": 0, "wins_a": 0, "wins_b": 0, "ties": 0, "score_a": 0.0})
            self.assertFalse(initial_swap("task-1", 42))
            metadata = prepared.executor_dir / "metadata.json"
            write_trial_metadata(metadata, {"trial": prepared.trial_index, "swapped": prepared.swapped})
            self.assertEqual(json.loads(metadata.read_text())["trial"], 0)

    def test_pairwise_discovery_and_reference_checks_reject_unsafe_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_a = self._candidate(root, "candidate-a")
            candidate_b = self._candidate(root, "candidate-b")
            (candidate_b / "task_2" / "repeat_0").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "task sets differ"):
                matched_tasks(candidate_a, candidate_b)

            candidate_b = self._candidate(root, "candidate-c", reference="different")
            with self.assertRaisesRegex(ValueError, "reference files differ"):
                validate_reference_equivalence(candidate_a, candidate_b)
            candidate_d = self._candidate(root, "candidate-d", reference=None)
            with self.assertRaisesRegex(ValueError, "presence differs"):
                validate_reference_equivalence(candidate_a, candidate_d)

            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (candidate_a / "task_1" / "repeat_0" / "escape").symlink_to(outside)
            candidate_e = self._candidate(root, "candidate-e")
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_trial(root / "unsafe", "task-1", candidate_a, candidate_e, trial_index=0, seed=1)

            with self.assertRaisesRegex(ValueError, "directory not found"):
                discover_tasks(root / "missing")


class EvaluatorBoundaryTests(unittest.TestCase):
    def test_exact_evaluator_is_trimmed_deterministic_and_zero_on_failed_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = ExactMatchEvaluator()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 2))
            with self.assertRaisesRegex(ValueError, "expected_answer"):
                evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 1))
            preflight = evaluator.preflight(root / "run")
            self.assertTrue(preflight.ok)
            self.assertEqual(preflight.version, "1")
            matched = evaluator.evaluate(
                EvaluationRequest(
                    "task-1",
                    "prompt",
                    {"expected_answer": " answer "},
                    (_candidate(root, output_text="\nanswer\t"),),
                )
            )
            self.assertEqual(matched.metrics, {"exact_match": 1.0})
            self.assertTrue(matched.outcomes["matched"])
            failed = evaluator.evaluate(
                EvaluationRequest(
                    "task-1",
                    "prompt",
                    {"expected_answer": "answer"},
                    (_candidate(root, status=ExecutionStatus.FAILED, output_text="answer"),),
                )
            )
            self.assertEqual(failed.metrics, {"exact_match": 0.0})
            with self.assertRaisesRegex(ValueError, "expected_answer"):
                evaluator.evaluate(EvaluationRequest("task-1", "prompt", candidates=(_candidate(root),)))

    def test_evaluation_contracts_normalize_candidates_and_reject_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _execution_result(root)
            request = EvaluationRequest(
                "task-1",
                "prompt",
                candidates=cast(tuple[EvaluationCandidate, ...], (result,)),
            )
            self.assertEqual(request.candidates[0].candidate_id, "candidate_0")
            self.assertEqual(request.canonical_task_prompt, "prompt")
            self.assertEqual(request.evaluator_metadata, {})
            self.assertIsNone(request.evaluator_artifact_destination)
            self.assertEqual(require_one_candidate(request).result, result)
            self.assertEqual(require_candidates(request, 1), request.candidates)
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                require_two_candidates(request)
            other = EvaluationRequest(
                "task-1",
                "prompt",
                candidates=(EvaluationCandidate("a", result), EvaluationCandidate("a", result)),
            )
            with self.assertRaisesRegex(ValueError, "distinct"):
                require_two_candidates(other)
            with self.assertRaisesRegex(ValueError, "does not match"):
                require_one_candidate(
                    EvaluationRequest(
                        "different",
                        "prompt",
                        candidates=cast(tuple[EvaluationCandidate, ...], (result,)),
                    )
                )
            with self.assertRaises(ValueError):
                EvaluationPlan("task", "prompt", {}, -1)
            plan = EvaluationPlan("task", "prompt", {"key": "value"}, 1)
            self.assertEqual(plan.metadata["key"], "value")

    def test_aime_and_bigcode_native_evaluators_cover_metadata_and_output_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aime = AIME26Evaluator()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                aime.validate_plan(EvaluationPlan("task", "prompt", {}, 2))
            with self.assertRaisesRegex(ValueError, "expected_answer"):
                aime.validate_plan(EvaluationPlan("task", "prompt", {}, 1))
            with patch(
                "gdpval_harness.evaluators.aime26._math_verify_preflight",
                return_value=(False, "missing", None),
            ):
                self.assertFalse(aime.preflight().ok)
            with patch(
                "gdpval_harness.evaluators.aime26._math_verify_preflight",
                return_value=(True, "ready", "0.8.0"),
            ):
                aime.preflight()
            empty = EvaluationRequest(
                "task-1", "prompt", {"expected_answer": "1"}, (_candidate(root, output_text=""),)
            )
            empty_result = aime.evaluate(empty)
            self.assertEqual(empty_result.details["reason"], "empty executor output")
            failed = EvaluationRequest(
                "task-1",
                "prompt",
                {"expected_answer": "1"},
                (_candidate(root, status=ExecutionStatus.FAILED),),
            )
            self.assertEqual(aime.evaluate(failed).metrics, {"accuracy": 0.0})

            bigcode = BigCodeBenchEvaluator(resource_dir=root / "grader")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                bigcode.validate_plan(EvaluationPlan("task", "prompt", {}, 2))
            with self.assertRaisesRegex(ValueError, "metadata keys"):
                bigcode.validate_plan(EvaluationPlan("task", "prompt", {}, 1))
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                bigcode.evaluate(
                    EvaluationRequest(
                        "task-1",
                        "prompt",
                        {"test": "t", "entry_point": "f", "code_prompt": "def f():"},
                        (_candidate(root),),
                    )
                )
            with patch(
                "resources_servers.bigcodebench.code_extraction.preprocess_code_completion",
                return_value="",
            ):
                self.assertEqual(
                    _native_bigcodebench_evaluate(
                        "no code here",
                        {"test": "t", "entry_point": "f", "code_prompt": "def f():"},
                        resource_dir=root,
                    )["status"],
                    "no_code_block",
                )

    def test_aime_evaluator_is_native_only_and_requires_ready_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aime = AIME26Evaluator()
            request = EvaluationRequest(
                "task-1",
                "prompt",
                {"expected_answer": "42"},
                (_candidate(root, output_text="\\boxed{42}"),),
            )
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                aime.evaluate(request)
            with patch(
                "gdpval_harness.evaluators.aime26._math_verify_preflight",
                return_value=(True, "native ready", "0.8.0"),
            ):
                preflight = aime.preflight(root / "run")
            self.assertTrue(preflight.ok)
            self.assertEqual(preflight.version, "0.8.0")
            self.assertEqual(preflight.revision, None)
            with patch(
                "gdpval_harness.evaluators.aime26._native_math_evaluate",
                return_value=(1.0, "42"),
            ) as native:
                result = aime.evaluate(request)
            self.assertEqual(result.metrics, {"accuracy": 1.0})
            self.assertEqual(result.details["extracted_answer"], "42")
            self.assertFalse(result.details["llm_judge_used"])
            native.assert_called_once_with("42", "\\boxed{42}")
            with self.assertRaisesRegex(ValueError, "expected_answer"):
                aime.evaluate(EvaluationRequest("task-1", "prompt", candidates=request.candidates))

            failed_request = EvaluationRequest(
                "task-1",
                "prompt",
                {"expected_answer": "42"},
                (_candidate(root, status=ExecutionStatus.TIMED_OUT, output_text="\\boxed{42}"),),
            )
            with patch("gdpval_harness.evaluators.aime26._native_math_evaluate") as native:
                failed_result = aime.evaluate(failed_request)
            self.assertEqual(failed_result.metrics, {"accuracy": 0.0})
            native.assert_not_called()

    def test_aime_native_helper_and_dependency_preflight_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            del tmp

            class FakeHelper:
                class LatexExtractionConfig:
                    pass

                class ExprExtractionConfig:
                    pass

                def __init__(self) -> None:
                    self.boxed: str | None = None

                def _extract_last_boxed_answer(self, output: str) -> str | None:
                    del output
                    return self.boxed

                def math_metric(self, **kwargs: object) -> object:
                    self.metric_kwargs = kwargs
                    return "metric"

                def _run_math_verify(self, verifier: object, expected: str, output: str) -> tuple[float, str]:
                    self.run_args = (verifier, expected, output)
                    return 1.0, "42"

            helper = FakeHelper()
            with patch(
                "gdpval_harness.evaluators.aime26.importlib.import_module",
                return_value=helper,
            ):
                self.assertEqual(_native_math_evaluate("42", "no boxed answer"), (0.0, None))
                helper.boxed = "42"
                self.assertEqual(_native_math_evaluate("42", "\\boxed{42}"), (1.0, "42"))
                self.assertEqual(helper.run_args[1:], ("42", "\\boxed{42}"))

            module = "gdpval_harness.evaluators.aime26"
            with patch(f"{module}.importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError()):
                missing = _math_verify_preflight()
            self.assertFalse(missing[0])
            with patch(f"{module}.importlib.metadata.version", side_effect=RuntimeError("metadata broken")):
                unavailable = _math_verify_preflight()
            self.assertFalse(unavailable[0])
            with patch(f"{module}.importlib.metadata.version", return_value="0.7.0"):
                wrong_version = _math_verify_preflight()
            self.assertFalse(wrong_version[0])
            with (
                patch(f"{module}.importlib.metadata.version", return_value="0.8.0"),
                patch(f"{module}.importlib.import_module", side_effect=ImportError("helper unavailable")),
            ):
                helper_missing = _math_verify_preflight()
            self.assertFalse(helper_missing[0])
            with (
                patch(f"{module}.importlib.metadata.version", return_value="0.8.0"),
                patch(f"{module}.importlib.import_module", return_value=object()),
            ):
                ready = _math_verify_preflight()
            self.assertTrue(ready[0])
            self.assertEqual(ready[2], "0.8.0")

    def test_bigcode_preflight_and_evaluation_never_install_or_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grader = root / "grader"
            grader.mkdir()
            evaluator = BigCodeBenchEvaluator(resource_dir=grader)
            overlap = evaluator.preflight(grader / "run")
            self.assertFalse(overlap.ok)
            self.assertIn("separate", overlap.details[0])
            self.assertFalse(evaluator.preflight(root / "run").ok)
            (grader / "bcb_runner.py").write_text("# deterministic fake runner\n", encoding="utf-8")
            fake_python = grader / "python"
            fake_python.write_text("fake", encoding="utf-8")
            with patch(
                "resources_servers.bigcodebench.setup_bcb_venv.ensure_bcb_venv",
                return_value=fake_python,
            ) as ensure:
                ready = evaluator.preflight(root / "run")
            self.assertTrue(ready.ok)
            ensure.assert_called_once()
            self.assertIn("dedicated Python 3.10", ready.details[0])

            candidate = _candidate(root, output_text="code")
            failed_request = EvaluationRequest(
                "task-1",
                "prompt",
                {"test": "t", "entry_point": "f", "code_prompt": "def f():"},
                (candidate,),
            )
            failed = evaluator.evaluate(
                EvaluationRequest(
                    failed_request.task_id,
                    failed_request.task_prompt,
                    failed_request.metadata,
                    (_candidate(root, status=ExecutionStatus.FAILED, output_text="code"),),
                )
            )
            self.assertEqual(failed.metrics, {"pass_rate": 0.0})
            self.assertFalse(failed.details["grader_invoked"])
            empty = evaluator.evaluate(
                EvaluationRequest(
                    failed_request.task_id,
                    failed_request.task_prompt,
                    failed_request.metadata,
                    (_candidate(root, output_text="   "),),
                )
            )
            self.assertEqual(empty.details["status"], "empty_output")
            with patch(
                "gdpval_harness.evaluators.bigcodebench._native_bigcodebench_evaluate",
                return_value={"reward": 1.0, "status": "pass", "extracted_model_code": "return 1", "details": None},
            ) as native:
                passed = evaluator.evaluate(failed_request)
            self.assertEqual(passed.metrics, {"pass_rate": 1.0})
            self.assertTrue(passed.details["grader_invoked"])
            native.assert_called_once()
            with self.assertRaisesRegex(ValueError, "metadata"):
                evaluator.evaluate(
                    EvaluationRequest(
                        "task-1",
                        "prompt",
                        {},
                        (candidate,),
                    )
                )
            overlapping_artifacts = grader / "candidate-artifacts"
            overlapping_artifacts.mkdir()
            (overlapping_artifacts / "answer.txt").write_text("answer", encoding="utf-8")
            overlapping_result = _execution_result(root, deliverables=overlapping_artifacts)
            overlapping_result = ExecutionResult(
                task_id=overlapping_result.task_id,
                executor=overlapping_result.executor,
                executor_version=overlapping_result.executor_version,
                invocation_mode=overlapping_result.invocation_mode,
                auth_mode=overlapping_result.auth_mode,
                workspace=grader,
                deliverables_dir=overlapping_result.deliverables_dir,
                status=ExecutionStatus.COMPLETED,
                started_at=overlapping_result.started_at,
                finished_at=overlapping_result.finished_at,
                exit_code=0,
                output_text="code",
            )
            with self.assertRaisesRegex(RuntimeError, "separate"):
                evaluator.evaluate(
                    EvaluationRequest(
                        "task-1",
                        "prompt",
                        failed_request.metadata,
                        (EvaluationCandidate("candidate", overlapping_result, overlapping_artifacts),),
                    )
                )

    def test_bigcode_native_grader_timeout_oserror_and_malformed_json_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resource = root / "grader"
            resource.mkdir()
            python = resource / "python"
            python.write_text("fake", encoding="utf-8")
            metadata = {"test": "test", "entry_point": "solve", "code_prompt": "def solve():"}
            with patch(
                "resources_servers.bigcodebench.code_extraction.preprocess_code_completion",
                return_value="return 1",
            ):
                with patch(
                    "gdpval_harness.evaluators.bigcodebench.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd=["fake"], timeout=1),
                ):
                    timeout = _native_bigcodebench_evaluate(
                        "```python\nreturn 1\n```", metadata, resource_dir=resource, bcb_python=python
                    )
                self.assertEqual(timeout["status"], "timeout")
                with patch(
                    "gdpval_harness.evaluators.bigcodebench.subprocess.run",
                    side_effect=OSError("grader missing"),
                ):
                    failed = _native_bigcodebench_evaluate(
                        "```python\nreturn 1\n```", metadata, resource_dir=resource, bcb_python=python
                    )
                self.assertEqual(failed["status"], "error")
                with patch(
                    "gdpval_harness.evaluators.bigcodebench.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, stdout="not json", stderr="stderr"),
                ):
                    malformed = _native_bigcodebench_evaluate(
                        "```python\nreturn 1\n```", metadata, resource_dir=resource, bcb_python=python
                    )
                self.assertEqual(malformed["status"], "error")

    def test_bigcode_native_grader_reports_pass_and_failure_with_replacement_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {"test": "assert f()", "entry_point": "f", "code_prompt": "def f():"}
            fake_python = root / "python"
            fake_python.write_text("fake", encoding="utf-8")
            with patch(
                "resources_servers.bigcodebench.code_extraction.preprocess_code_completion",
                return_value="return 1",
            ):
                with patch(
                    "gdpval_harness.evaluators.bigcodebench.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess(
                            [], 0, stdout=json.dumps({"status": "pass", "details": {"ok": True}}), stderr=""
                        ),
                        subprocess.CompletedProcess(
                            [], 0, stdout=json.dumps({"status": "fail", "details": {"ok": False}}), stderr=""
                        ),
                    ],
                ) as run:
                    passed = _native_bigcodebench_evaluate(
                        "```python\nreturn 1\n```", metadata, resource_dir=root, bcb_python=fake_python
                    )
                    failed = _native_bigcodebench_evaluate(
                        "```python\nreturn 1\n```",
                        metadata,
                        resource_dir=root,
                        bcb_python=fake_python,
                    )
                self.assertEqual(passed["reward"], 1.0)
                self.assertEqual(passed["status"], "pass")
                self.assertEqual(failed["reward"], 0.0)
                self.assertEqual(failed["status"], "fail")
                self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_gdpval_external_evaluator_publishes_and_rejects_unsafe_handoffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = GDPvalExternalEvaluator()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 2, root / "out"))
            with self.assertRaisesRegex(ValueError, "artifact destination"):
                evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 1))
            preflight = evaluator.preflight(root / "run")
            self.assertTrue(preflight.ok)
            workspace = root / "workspace"
            deliverables = workspace / "deliverables"
            deliverables.mkdir(parents=True)
            (deliverables / "answer.txt").write_text("answer", encoding="utf-8")
            (workspace / "reference_files").mkdir()
            (workspace / "reference_files" / "source.txt").write_text("source", encoding="utf-8")
            result = _execution_result(root, deliverables=deliverables)
            request = EvaluationRequest(
                "task-1",
                "prompt",
                candidates=(EvaluationCandidate("policy", result, deliverables),),
                artifact_dir=root / "out",
            )
            published = evaluator.evaluate(request)
            self.assertEqual(published.status, EvaluationStatus.EXTERNAL)
            self.assertEqual((root / "out" / "answer.txt").read_text(), "answer")
            self.assertTrue((root / "out" / "reference_files" / "source.txt").is_file())
            self.assertTrue((root / "out" / "finish_params.json").is_file())
            finish = json.loads((root / "out" / "finish_params.json").read_text(encoding="utf-8"))
            self.assertEqual(finish["executor"], "fake")
            self.assertEqual(finish["status"], "completed")
            self.assertEqual(finish["files"], ["answer.txt"])
            self.assertNotIn("prompt", finish)
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(request)
            with self.assertRaisesRegex(ValueError, "artifact_dir"):
                evaluator.evaluate(
                    EvaluationRequest(
                        "task-1",
                        "prompt",
                        candidates=(EvaluationCandidate("policy", result, deliverables),),
                    )
                )
            skipped = evaluator.evaluate(
                EvaluationRequest(
                    "task-1",
                    "prompt",
                    candidates=(_candidate(root, status=ExecutionStatus.FAILED),),
                    artifact_dir=root / "skipped",
                )
            )
            self.assertEqual(skipped.status, EvaluationStatus.SKIPPED)

    def test_gdpval_handoff_rejects_reserved_symlink_and_overlap_paths_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = GDPvalExternalEvaluator()
            workspace = root / "workspace"
            deliverables = workspace / "deliverables"
            deliverables.mkdir(parents=True)
            (deliverables / "answer.txt").write_text("answer", encoding="utf-8")
            result = _execution_result(root, deliverables=deliverables)

            def request(destination: Path) -> EvaluationRequest:
                return EvaluationRequest(
                    "task-1",
                    "prompt",
                    candidates=(EvaluationCandidate("policy", result, deliverables),),
                    artifact_dir=destination,
                )

            reserved = deliverables / "finish_params.json"
            reserved.write_text("do not copy", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved"):
                evaluator.evaluate(request(root / "reserved-out"))
            reserved.unlink()

            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (deliverables / "escape.txt").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                evaluator.evaluate(request(root / "symlink-out"))
            (deliverables / "escape.txt").unlink()

            reference = workspace / "reference_files"
            outside_reference = root / "outside-reference"
            outside_reference.mkdir()
            reference.symlink_to(outside_reference, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked GDPval reference"):
                evaluator.evaluate(request(root / "reference-out"))
            reference.unlink()

            with self.assertRaisesRegex(ValueError, "separate"):
                evaluator.evaluate(request(deliverables / "nested"))

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                evaluator.evaluate(request(linked_parent / "out"))

            destination = root / "atomic-out"
            with patch(
                "gdpval_harness.evaluators.gdpval.os.replace",
                side_effect=OSError("publish interrupted"),
            ):
                with self.assertRaisesRegex(OSError, "publish interrupted"):
                    evaluator.evaluate(request(destination))
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".atomic-out.staging-*")), [])

    def test_gdpval_copy_helpers_keep_bytes_and_reject_nonempty_or_special_boundaries(self) -> None:
        from gdpval_harness.evaluators import gdpval as gdpval_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "nested").mkdir()
            (source / "nested" / "answer.txt").write_bytes(b"bytes\xff\n")
            destination = root / "destination"
            copied = gdpval_module._copy_tree_bytes(source, destination)
            self.assertEqual(copied, ["nested/answer.txt"])
            self.assertEqual((destination / "nested" / "answer.txt").read_bytes(), b"bytes\xff\n")
            with self.assertRaises(FileExistsError):
                gdpval_module._copy_tree_bytes(source, destination, allow_existing_destination=True)

            missing_reference = root / "missing-reference"
            gdpval_module._copy_reference_tree(missing_reference, root / "missing-destination")
            source_file = root / "reference-file"
            source_file.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                gdpval_module._copy_reference_tree(source_file, root / "bad-reference")
            marker = root / "marker.json"
            gdpval_module._write_durable_json(marker, {"ok": True})
            with self.assertRaises(FileExistsError):
                gdpval_module._write_durable_json(marker, {"overwrite": True})

            fifo = source / "pipe"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                gdpval_module._copy_tree_bytes(source, root / "special-destination")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                gdpval_module._assert_no_symlink_components(root / "missing", allow_missing_final=False)
            gdpval_module._fsync_directory(source / "nested" / "answer.txt")

    def test_pairwise_evaluator_preflight_sanitizes_environment_and_rejects_boundaries(self) -> None:
        class FakeJudge:
            name = "fake-judge"

            def preflight(self, *, environment: Mapping[str, str]) -> JudgePreflightResult:
                return JudgePreflightResult(
                    "fake-judge", True, version="1", auth_mode="fake", details=(environment["PATH"],)
                )

            def judge(self, request: object) -> JudgeResult:
                raise AssertionError(f"judge should not be called: {request}")

        self.assertEqual(sanitize_environment({"PATH": "/bin", "SECRET": "hidden"}), {"PATH": "/bin"})
        evaluator = PairwiseJudgeEvaluator(
            cast(JudgeExecutor, FakeJudge()), trials=1, environment={"PATH": "/bin", "SECRET": "hidden"}
        )
        with self.assertRaisesRegex(ValueError, "exactly two"):
            evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 1, Path("out")))
        with self.assertRaisesRegex(ValueError, "artifact destination"):
            evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 2))
        self.assertTrue(evaluator.preflight().ok)
        with self.assertRaisesRegex(ValueError, "terminal"):
            evaluator.evaluate(
                EvaluationRequest(
                    "task-1",
                    "prompt",
                    candidates=(
                        _candidate(Path(tempfile.gettempdir()), status=ExecutionStatus.FAILED),
                        _candidate(Path(tempfile.gettempdir()), "b", status=ExecutionStatus.COMPLETED),
                    ),
                    artifact_dir=Path(tempfile.gettempdir()) / "pairwise-output-does-not-matter",
                )
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            PairwiseJudgeEvaluator(cast(JudgeExecutor, FakeJudge()), trials=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            PairwiseJudgeEvaluator(cast(JudgeExecutor, FakeJudge()), timeout_seconds=0)

    def test_pairwise_evaluator_records_blind_judge_provenance_and_preserves_metadata(self) -> None:
        class RecordingJudge:
            name = "recording-judge"

            def __init__(self) -> None:
                self.requests: list[object] = []

            def preflight(self, *, environment: Mapping[str, str]) -> JudgePreflightResult:
                self.environment = dict(environment)
                return JudgePreflightResult(
                    "recording-judge",
                    True,
                    version="judge-1",
                    auth_mode="subscription",
                    details=("deterministic fake",),
                )

            def judge(self, request: object) -> JudgeResult:
                self.requests.append(request)
                assert isinstance(request, JudgeRequest)
                request.executor_dir.mkdir(parents=True, exist_ok=True)
                (request.executor_dir / "metadata.json").write_text('{"owner": "judge"}\n', encoding="utf-8")
                return JudgeResult(
                    task_id=request.task_id,
                    trial_index=request.trial_index,
                    judge_executor=self.name,
                    verdict=Verdict.A if request.trial_index == 0 else Verdict.TIE,
                    executor_version="judge-1",
                    invocation_mode="fake-judge",
                    auth_mode="subscription",
                    started_at="2026-09-12T00:00:00+00:00",
                    finished_at="2026-09-12T00:00:01+00:00",
                    exit_code=0,
                    stdout_path=request.executor_dir / "stdout.log",
                    stderr_path=request.executor_dir / "stderr.log",
                    metadata={"blind": True},
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "candidate-a"
            second_dir = root / "candidate-b"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "answer.txt").write_text("a\n", encoding="utf-8")
            (second_dir / "answer.txt").write_text("b\n", encoding="utf-8")
            judge = RecordingJudge()
            evaluator = PairwiseJudgeEvaluator(
                cast(JudgeExecutor, judge),
                trials=2,
                seed=7,
                model="judge-model",
                environment={"PATH": "/bin", "SECRET": "do-not-forward"},
            )
            preflight = evaluator.preflight(root / "run")
            self.assertTrue(preflight.ok)
            self.assertEqual(preflight.judge_executor, "recording-judge")
            self.assertEqual(preflight.judge_executor_version, "judge-1")
            self.assertEqual(preflight.judge_auth_mode, "subscription")
            self.assertEqual(judge.environment, {"PATH": "/bin"})
            request = EvaluationRequest(
                "task-1",
                "compare these submissions",
                candidates=(
                    _candidate(root, "a", output_text="a"),
                    _candidate(root, "b", output_text="b"),
                ),
                artifact_dir=root / "pairwise-output",
            )
            request = EvaluationRequest(
                request.task_id,
                request.task_prompt,
                candidates=(
                    EvaluationCandidate("candidate-a", request.candidates[0].execution, first_dir),
                    EvaluationCandidate("candidate-b", request.candidates[1].execution, second_dir),
                ),
                artifact_dir=request.artifact_dir,
            )
            result = evaluator.evaluate(request)
            self.assertEqual(result.status, EvaluationStatus.COMPLETED)
            self.assertEqual(result.outcomes["counts"], {"trials": 2, "wins_a": 1, "wins_b": 0, "ties": 1})
            self.assertEqual(result.details["judge_executor"], "recording-judge")
            self.assertEqual(result.details["judge_model"], "judge-model")
            self.assertEqual(len(judge.requests), 2)
            for trial in range(2):
                trial_dir = root / "pairwise-output" / "judge" / "tasks" / "task-1" / f"trial_{trial}" / "executor"
                self.assertEqual(
                    (trial_dir / "metadata.json").read_text(encoding="utf-8"),
                    '{"owner": "judge"}\n',
                )
                metadata = json.loads((trial_dir / "harness-metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["judge_executor_version"], "judge-1")
                self.assertEqual(metadata["judge_auth_mode"], "subscription")
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(request)

    def test_pairwise_evaluator_fails_closed_and_persists_judge_failures(self) -> None:
        class FailingJudge:
            name = "failing-judge"

            def __init__(self, mode: str) -> None:
                self.mode = mode

            def preflight(self, environment: Mapping[str, str]) -> JudgePreflightResult:
                return JudgePreflightResult(self.name, True, version="1", auth_mode="fake")

            def judge(self, request: object) -> JudgeResult:
                assert isinstance(request, JudgeRequest)
                if self.mode == "raise":
                    raise OSError("judge process unavailable")
                verdict = None if self.mode == "missing" else Verdict.A
                exit_code = 4 if self.mode == "nonzero" else 0
                task_id = "other" if self.mode == "mismatch" else request.task_id
                return JudgeResult(
                    task_id=task_id,
                    trial_index=request.trial_index,
                    judge_executor=self.name,
                    verdict=verdict,
                    executor_version="1",
                    invocation_mode="fake",
                    auth_mode="fake",
                    started_at="start",
                    finished_at="finish",
                    exit_code=exit_code,
                    stdout_path=request.executor_dir / "stdout.log",
                    stderr_path=request.executor_dir / "stderr.log",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "answer.txt").write_text("a", encoding="utf-8")
            (second_dir / "answer.txt").write_text("b", encoding="utf-8")
            base_request = EvaluationRequest(
                "task-1",
                "prompt",
                candidates=(
                    EvaluationCandidate("a", _execution_result(root / "a"), first_dir),
                    EvaluationCandidate("b", _execution_result(root / "b"), second_dir),
                ),
                artifact_dir=root / "output",
            )
            for mode, expected in (
                ("missing", "judge result is missing a valid verdict"),
                ("nonzero", "judge executor returned a nonzero exit code"),
                ("mismatch", "judge result task id mismatch"),
                ("raise", "judge_call"),
            ):
                judge = FailingJudge(mode)
                evaluator = PairwiseJudgeEvaluator(cast(JudgeExecutor, judge), trials=1)
                evaluator.preflight()
                request = EvaluationRequest(
                    base_request.task_id,
                    base_request.task_prompt,
                    candidates=base_request.candidates,
                    artifact_dir=root / f"output-{mode}",
                )
                with self.assertRaisesRegex(RuntimeError, "pairwise judge failed"):
                    evaluator.evaluate(request)
                metadata = next(
                    path
                    for path in (root / f"output-{mode}").rglob("*.json")
                    if path.name in {"metadata.json", "harness-metadata.json"}
                )
                persisted = json.loads(metadata.read_text(encoding="utf-8"))
                if mode == "raise":
                    self.assertEqual(persisted["error_details"]["phase"], expected)
                else:
                    self.assertEqual(persisted["error"], expected)

    def test_pairwise_preflight_supports_legacy_no_argument_judges(self) -> None:
        class LegacyJudge:
            name = "legacy"

            def preflight(self) -> JudgePreflightResult:
                return JudgePreflightResult(self.name, True, version="old", auth_mode="account")

            def judge(self, request: object) -> JudgeResult:
                raise AssertionError(request)

        evaluator = PairwiseJudgeEvaluator(cast(JudgeExecutor, LegacyJudge()), trials=1)
        result = evaluator.preflight()
        self.assertTrue(result.ok)
        self.assertEqual(result.judge_executor, "legacy")
        self.assertEqual(result.judge_executor_version, "old")

    def test_pairwise_evaluator_rejects_candidate_and_output_path_boundaries(self) -> None:
        class NoopJudge:
            name = "noop"

            def preflight(self) -> JudgePreflightResult:
                return JudgePreflightResult(self.name, True, version="1", auth_mode="fake")

            def judge(self, request: object) -> JudgeResult:
                raise AssertionError(request)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "answer.txt").write_text("a", encoding="utf-8")
            (second_dir / "answer.txt").write_text("b", encoding="utf-8")
            first = EvaluationCandidate("a", _execution_result(root / "a"), first_dir)
            second = EvaluationCandidate("b", _execution_result(root / "b"), second_dir)

            def evaluate_with(
                destination: Path | None, candidates: tuple[EvaluationCandidate, ...] = (first, second)
            ) -> None:
                evaluator = PairwiseJudgeEvaluator(cast(JudgeExecutor, NoopJudge()), trials=1)
                evaluator.preflight()
                evaluator.evaluate(
                    EvaluationRequest("task-1", "prompt", candidates=candidates, artifact_dir=destination)
                )

            with self.assertRaisesRegex(ValueError, "missing"):
                evaluate_with(root / "missing-candidate", (EvaluationCandidate("a", first.execution), second))
            file_path = root / "candidate-file"
            file_path.write_text("file", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not found"):
                evaluate_with(root / "file-candidate", (EvaluationCandidate("a", first.execution, file_path), second))
            real = root / "real-candidate"
            real.mkdir()
            symlink = root / "candidate-link"
            symlink.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                evaluate_with(root / "symlink-candidate", (EvaluationCandidate("a", first.execution, symlink), second))
            with self.assertRaisesRegex(ValueError, "artifact_dir"):
                evaluate_with(None)
            existing_file = root / "existing-file"
            existing_file.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                evaluate_with(existing_file)
            existing_dir = root / "existing-dir"
            existing_dir.mkdir()
            with self.assertRaises(FileExistsError):
                evaluate_with(existing_dir)
            linked_parent = root / "linked-parent"
            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                evaluate_with(linked_parent / "output")
            with self.assertRaisesRegex(ValueError, "separate"):
                evaluate_with(first_dir / "nested-output")

    def test_pairwise_evaluator_persists_interruptions_and_refuses_metadata_overwrite(self) -> None:
        class InterruptJudge:
            name = "interrupt"

            def preflight(self, environment: Mapping[str, str]) -> JudgePreflightResult:
                del environment
                return JudgePreflightResult(self.name, True)

            def judge(self, request: object) -> JudgeResult:
                assert isinstance(request, JudgeRequest)
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "answer.txt").write_text("a", encoding="utf-8")
            (second_dir / "answer.txt").write_text("b", encoding="utf-8")
            candidates = (
                EvaluationCandidate("a", _execution_result(root / "a"), first_dir),
                EvaluationCandidate("b", _execution_result(root / "b"), second_dir),
            )
            evaluator = PairwiseJudgeEvaluator(cast(JudgeExecutor, InterruptJudge()), trials=1)
            evaluator.preflight()
            with self.assertRaises(KeyboardInterrupt):
                evaluator.evaluate(
                    EvaluationRequest("task-1", "prompt", candidates=candidates, artifact_dir=root / "output")
                )
            metadata = next((root / "output").rglob("metadata.json"))
            self.assertEqual(json.loads(metadata.read_text())["error_type"], "KeyboardInterrupt")

            from gdpval_harness.evaluators import pairwise as pairwise_module

            trial_dir = root / "trial-metadata"
            trial_dir.mkdir()
            (trial_dir / "metadata.json").write_text("judge", encoding="utf-8")
            (trial_dir / "harness-metadata.json").write_text("harness", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                pairwise_module._write_trial_metadata_preserving(trial_dir, {"new": True})
            self.assertEqual(pairwise_module._jsonable_verdict(None), None)
            self.assertEqual(pairwise_module._jsonable_verdict(Verdict.A), "A")
            self.assertEqual(pairwise_module._jsonable_verdict(cast(Verdict, "TIE")), "TIE")


class BenchmarkBoundaryTests(unittest.TestCase):
    def test_benchmark_base_execution_task_keeps_canonical_task(self) -> None:
        class StubBenchmark(Benchmark):
            name = "stub"
            revision = None

            def is_prepared(self) -> bool:
                return True

            def prepare(self) -> None:
                return None

            def load_tasks(self, limit: int) -> list[BenchmarkTask]:
                return [] if limit == 0 else [BenchmarkTask(TaskSpec("task", "prompt"))]

            def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
                return []

        task = BenchmarkTask(TaskSpec("task", "prompt"), evaluation={"expected": "answer"})
        self.assertEqual(
            StubBenchmark().execution_task(task, Path("workspace"), network_policy="disabled"), task.execution
        )

    def _prepare_script(self, root: Path, name: str, *, mode: str) -> Path:
        script = root / f"{name}-prepare.py"
        if mode == "success":
            script.write_text(
                f"from pathlib import Path\nPath({name!r} + '.jsonl').write_text('{{\"ok\": true}}\\n')\n",
                encoding="utf-8",
            )
        elif mode == "failure":
            script.write_text("raise SystemExit(3)\n", encoding="utf-8")
        else:
            script.write_text("pass\n", encoding="utf-8")
        return script

    def test_benchmark_registry_and_prepare_failures_are_explicit(self) -> None:
        self.assertEqual([item.name for item in list_benchmarks()], ["aime26", "bigcodebench", "gdpval"])
        self.assertEqual(get_benchmark_descriptor("aime26").network_requirement.startswith("disabled"), True)
        with self.assertRaisesRegex(ValueError, "unknown benchmark"):
            get_benchmark_descriptor("missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls, name in (
                (AIME26Benchmark, "aime"),
                (BigCodeBenchBenchmark, "bigcode"),
                (GDPvalBenchmark, "gdpval"),
            ):
                dataset = root / f"{name}.jsonl"
                benchmark = cls(root=root, dataset_path=dataset, prepare_script=root / "missing.py")
                self.assertFalse(benchmark.is_prepared())
                with self.assertRaisesRegex(RuntimeError, "prepare script not found"):
                    benchmark.prepare()
                benchmark.prepare_script = self._prepare_script(root, name, mode="failure")
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()
                benchmark.prepare_script = self._prepare_script(root, name, mode="empty")
                with self.assertRaisesRegex(RuntimeError, "failed to prepare"):
                    benchmark.prepare()
                benchmark.prepare_script = self._prepare_script(root, name, mode="success")
                benchmark.prepare()
                self.assertTrue(dataset.is_file())
                benchmark.prepare()
            self.assertIsInstance(create_benchmark("gdpval", root=root), GDPvalBenchmark)
            self.assertIsInstance(create_benchmark("aime26", root=root), AIME26Benchmark)
            self.assertIsInstance(create_benchmark("bigcodebench", root=root), BigCodeBenchBenchmark)
            with self.assertRaisesRegex(ValueError, "unknown benchmark"):
                create_benchmark("missing", root=root)

    def test_benchmark_loaders_reject_empty_and_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aime = AIME26Benchmark(root=root, dataset_path=root / "aime.jsonl", prepare_script=root / "prepare.py")
            bigcode = BigCodeBenchBenchmark(
                root=root, dataset_path=root / "big.jsonl", prepare_script=root / "prepare.py"
            )
            gdpval = GDPvalBenchmark(root=root, dataset_path=root / "gdp.jsonl", prepare_script=root / "prepare.py")
            for benchmark in (aime, bigcode, gdpval):
                benchmark.dataset_path.write_text("\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "no .* tasks"):
                    benchmark.load_tasks(1)
                with self.assertRaises(ValueError):
                    benchmark.load_tasks(0)
            bigcode.dataset_path.write_text(
                json.dumps({"question": "q", "verifier_metadata": {}}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "missing task_id"):
                bigcode.load_tasks(1)
            gdpval.dataset_path.write_text(json.dumps({"task_id": "x", "prompt": "p"}) + "\n", encoding="utf-8")
            task = gdpval.load_tasks(1)[0]
            self.assertEqual(gdpval.materialize(task, root / "workspace"), [])
            prompt = gdpval.execution_task(task, root / "workspace", network_policy="disabled").prompt
            self.assertIn("Network policy for model-generated tools: disabled", prompt)

    def test_benchmark_loaders_preserve_task_and_evaluator_boundaries_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aime = AIME26Benchmark(root=root, dataset_path=root / "aime.jsonl", prepare_script=root / "prepare.py")
            aime.dataset_path.write_text(
                "\n" + json.dumps({"question": "What is 2+2?", "expected_answer": 4}) + "\n",
                encoding="utf-8",
            )
            aime_tasks = aime.load_tasks(1)
            self.assertEqual(aime_tasks[0].execution.task_id, "aime26-02")
            self.assertIn("\\boxed", aime_tasks[0].execution.prompt)
            self.assertEqual(aime_tasks[0].evaluation["expected_answer"], "4")
            aime_workspace = root / "aime-workspace"
            self.assertEqual(aime.materialize(aime_tasks[0], aime_workspace), [])
            self.assertTrue(aime_workspace.is_dir())

            bigcode = BigCodeBenchBenchmark(
                root=root, dataset_path=root / "big.jsonl", prepare_script=root / "prepare.py"
            )
            bigcode.dataset_path.write_text(
                json.dumps(
                    {
                        "question": "write solve",
                        "verifier_metadata": {
                            "task_id": "bcb-1",
                            "test": "assert solve(2) == 4",
                            "entry_point": "solve",
                            "code_prompt": "def solve(x):",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            bigcode_task = bigcode.load_tasks(1)[0]
            self.assertEqual(bigcode_task.execution.task_id, "bcb-1")
            self.assertIn("write solve", bigcode_task.execution.prompt)
            self.assertEqual(bigcode_task.evaluation["entry_point"], "solve")
            self.assertEqual(bigcode.materialize(bigcode_task, root / "bcb-workspace"), [])

            gdpval = GDPvalBenchmark(root=root, dataset_path=root / "gdp.jsonl", prepare_script=root / "prepare.py")
            gdpval.dataset_path.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "task_id": "g-1",
                                "prompt": "review",
                                "sector": "legal",
                                "occupation": "analyst",
                                "reference_files": '["reference_files/source.txt"]',
                                "reference_file_urls": '["file://offline/source.txt"]',
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "g-2",
                                "prompt": "draft",
                                "reference_files": "reference_files/none.txt",
                                "reference_file_urls": "not-json",
                            }
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            gdp_tasks = gdpval.load_tasks(2)
            self.assertEqual(gdp_tasks[0].materialization["reference_files"], ("reference_files/source.txt",))
            self.assertEqual(gdp_tasks[1].materialization["reference_files"], ("reference_files/none.txt",))
            workspace = root / "gdp-workspace"

            def fake_download(files: list[str], urls: list[str], destination: Path) -> list[str]:
                self.assertEqual(files, ["reference_files/source.txt"])
                self.assertEqual(urls, ["file://offline/source.txt"])
                target = destination / files[0]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("reference", encoding="utf-8")
                return files

            with patch(
                "responses_api_agents.stirrup_agent.tasks.gdpval._download_reference_files",
                side_effect=fake_download,
            ):
                self.assertEqual(gdpval.materialize(gdp_tasks[0], workspace), ["reference_files/source.txt"])
            execution = gdpval.execution_task(gdp_tasks[0], workspace, network_policy="disabled")
            self.assertIn("reference_files/source.txt", execution.prompt)
            self.assertIn("legal", str(gdp_tasks[0].evaluation["sector"]))

    def test_gdpval_reference_materialization_rejects_mismatch_and_unsafe_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "gdp.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "task_id": "x",
                        "prompt": "p",
                        "reference_files": ["reference_files/a.txt"],
                        "reference_file_urls": ["https://example.invalid/a.txt"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark = GDPvalBenchmark(root=root, dataset_path=dataset, prepare_script=root / "prepare.py")
            task = benchmark.load_tasks(1)[0]
            with self.assertRaisesRegex(RuntimeError, "count mismatch"):
                benchmark.materialize(
                    task.__class__(
                        task.execution, {"reference_files": ("a", "b"), "reference_file_urls": ("url",)}, {}
                    ),
                    root / "workspace",
                )
            with patch(
                "responses_api_agents.stirrup_agent.tasks.gdpval._download_reference_files",
                return_value=["../escape.txt"],
            ):
                with self.assertRaisesRegex(RuntimeError, "unsafe"):
                    benchmark.materialize(task, root / "workspace")


if __name__ == "__main__":
    unittest.main()
