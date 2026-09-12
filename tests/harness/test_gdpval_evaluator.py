# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval_harness.capabilities import ExecutorOutput
from eval_harness.evaluators.base import EvaluationCandidate, EvaluationPlan, EvaluationRequest, EvaluationStatus
from eval_harness.evaluators.gdpval import GDPvalExternalEvaluator
from eval_harness.executors.base import ExecutionResult, ExecutionStatus
from eval_harness.failures import Failure, FailureImpact, FailureKind


def _execution(root: Path, *, status: ExecutionStatus = ExecutionStatus.COMPLETED) -> ExecutionResult:
    workspace = root / "workspace"
    deliverables = workspace / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    successful = status in {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
    output_text = "answer" if successful else None
    return ExecutionResult(
        task_id="task/x",
        executor="fake",
        executor_version="fake-1",
        invocation_mode="fake",
        auth_mode="local",
        workspace=workspace,
        deliverables_dir=deliverables,
        status=status,
        started_at="2026-09-11T00:00:00+00:00",
        finished_at="2026-09-11T00:00:01+00:00",
        exit_code=0 if successful else 1,
        available_outputs=frozenset({ExecutorOutput.FINAL_TEXT}) if output_text is not None else frozenset(),
        failure=None if successful else Failure(FailureKind.PROCESS, "test_failure", FailureImpact.RUN),
        output_text=output_text,
    )


class GDPvalEvaluatorTests(unittest.TestCase):
    def test_plan_requires_one_candidate_and_artifact_destination(self) -> None:
        evaluator = GDPvalExternalEvaluator()
        evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 1, Path("out")))
        with self.assertRaisesRegex(ValueError, "exactly one candidate"):
            evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 2, Path("out")))
        with self.assertRaisesRegex(ValueError, "artifact destination"):
            evaluator.validate_plan(EvaluationPlan("task", "prompt", {}, 1))

    def test_external_handoff_copies_submission_and_references_without_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _execution(root)
            (result.deliverables_dir / "submitted.txt").write_bytes(b"submitted bytes")
            (result.workspace / "reference_files").mkdir()
            (result.workspace / "reference_files" / "source.txt").write_bytes(b"reference bytes")
            destination = root / "run" / "deliverables" / "task_task_x" / "repeat_0"
            request = EvaluationRequest(
                task_id="task/x",
                task_prompt="canonical prompt",
                metadata={"rubric": "secret", "reference_answer": "secret"},
                candidates=(EvaluationCandidate("policy", result, result.deliverables_dir),),
                artifact_dir=destination,
            )

            evaluation = GDPvalExternalEvaluator().evaluate(request)

            self.assertEqual(evaluation.status, EvaluationStatus.EXTERNAL)
            self.assertEqual(evaluation.metrics, {})
            self.assertEqual(evaluation.outcomes, {})
            evaluation_detail = evaluation.details["evaluation"]
            if not isinstance(evaluation_detail, str):
                raise AssertionError("expected a string evaluation detail")
            self.assertIn("rubric/pairwise", evaluation_detail)
            self.assertEqual((destination / "submitted.txt").read_bytes(), b"submitted bytes")
            self.assertEqual((destination / "reference_files" / "source.txt").read_bytes(), b"reference bytes")
            finish = json.loads((destination / "finish_params.json").read_text(encoding="utf-8"))
            self.assertEqual(finish["files"], ["submitted.txt"])
            self.assertNotIn("secret", json.dumps(finish))
            self.assertFalse(list(destination.parent.glob(".repeat_0.staging-*")))

    def test_existing_destination_is_never_overwritten_and_staging_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _execution(root)
            (result.deliverables_dir / "new.txt").write_text("new", encoding="utf-8")
            destination = root / "handoff"
            destination.mkdir()
            (destination / "marker").write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                GDPvalExternalEvaluator().evaluate(
                    EvaluationRequest(
                        task_id="task/x",
                        task_prompt="prompt",
                        candidates=(EvaluationCandidate("policy", result),),
                        artifact_dir=destination,
                    )
                )

            self.assertEqual((destination / "marker").read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(root.glob(".handoff.staging-*")), [])

    def test_source_symlink_and_failed_execution_do_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _execution(root)
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (result.deliverables_dir / "escape").symlink_to(outside)
            destination = root / "handoff"
            with self.assertRaisesRegex(ValueError, "symlink"):
                GDPvalExternalEvaluator().evaluate(
                    EvaluationRequest(
                        task_id="task/x",
                        task_prompt="prompt",
                        candidates=(EvaluationCandidate("policy", result),),
                        artifact_dir=destination,
                    )
                )
            self.assertFalse(destination.exists())

            failed = _execution(root, status=ExecutionStatus.FAILED)
            failed_destination = root / "failed-handoff"
            evaluation = GDPvalExternalEvaluator().evaluate(
                EvaluationRequest(
                    task_id="task/x",
                    task_prompt="prompt",
                    candidates=(EvaluationCandidate("policy", failed),),
                    artifact_dir=failed_destination,
                )
            )
            self.assertEqual(evaluation.status, EvaluationStatus.SKIPPED)
            self.assertFalse(failed_destination.exists())

    def test_reserved_submitted_names_are_rejected_and_reference_finish_bytes_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _execution(root)
            reference = result.workspace / "reference_files"
            reference.mkdir()
            (reference / "finish_params.json").write_bytes(b"reference marker bytes")
            (result.deliverables_dir / "finish_params.json").write_text("executor marker", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved"):
                GDPvalExternalEvaluator().evaluate(
                    EvaluationRequest(
                        task_id="task/x",
                        task_prompt="prompt",
                        candidates=(EvaluationCandidate("policy", result),),
                        artifact_dir=root / "handoff",
                    )
                )
            (result.deliverables_dir / "finish_params.json").unlink()
            destination = root / "reference-handoff"
            GDPvalExternalEvaluator().evaluate(
                EvaluationRequest(
                    task_id="task/x",
                    task_prompt="prompt",
                    candidates=(EvaluationCandidate("policy", result),),
                    artifact_dir=destination,
                )
            )
            self.assertEqual(
                (destination / "reference_files" / "finish_params.json").read_bytes(), b"reference marker bytes"
            )


if __name__ == "__main__":
    unittest.main()
