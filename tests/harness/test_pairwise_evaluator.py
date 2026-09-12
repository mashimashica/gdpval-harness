# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

from eval_harness.evaluators.base import EvaluationCandidate, EvaluationRequest, EvaluationStatus
from eval_harness.evaluators.pairwise import PairwiseJudgeEvaluator
from eval_harness.executors.base import ExecutionResult, ExecutionStatus
from eval_harness.judges.base import JudgeExecutor, JudgePreflightResult, JudgeRequest, JudgeResult, Verdict


class _CandidateOutcome(TypedDict):
    candidate_id: str


class FakeJudge(JudgeExecutor):
    name = "fake-judge"
    invocation_mode = "fake"

    def __init__(self, *, verdict: Verdict = Verdict.A, exit_code: int | None = 0) -> None:
        self.verdict = verdict
        self.exit_code = exit_code
        self.preflight_environment: Mapping[str, str] | None = None
        self.requests: list[JudgeRequest] = []

    def preflight(self, environment: Mapping[str, str] | None = None) -> JudgePreflightResult:
        self.preflight_environment = environment
        return JudgePreflightResult(
            judge_executor=self.name,
            ok=True,
            version="judge-1",
            auth_mode="fake-subscription",
            details=("fake ready",),
        )

    def judge(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        (request.executor_dir / "stdout.log").write_text("judge output", encoding="utf-8")
        return JudgeResult(
            task_id=request.task_id,
            trial_index=request.trial_index,
            judge_executor=self.name,
            verdict=self.verdict,
            executor_version="judge-1",
            invocation_mode=self.invocation_mode,
            auth_mode="fake-subscription",
            started_at="2026-09-11T00:00:00+00:00",
            finished_at="2026-09-11T00:00:01+00:00",
            exit_code=self.exit_code,
            stdout_path=request.executor_dir / "stdout.log",
            stderr_path=request.executor_dir / "stderr.log",
        )


def _execution(root: Path, label: str) -> ExecutionResult:
    return ExecutionResult(
        task_id="task/x",
        executor=label,
        executor_version="1",
        invocation_mode="fake",
        auth_mode="local",
        workspace=root,
        deliverables_dir=root,
        status=ExecutionStatus.COMPLETED,
        started_at="s",
        finished_at="f",
        exit_code=0,
    )


def _candidate(root: Path, name: str, content: str) -> EvaluationCandidate:
    directory = root / name
    (directory / "reference_files").mkdir(parents=True)
    (directory / "reference_files" / "ref.txt").write_text("same", encoding="utf-8")
    (directory / "answer.txt").write_text(content, encoding="utf-8")
    return EvaluationCandidate(name, _execution(directory, name), directory)


class PairwiseEvaluatorTests(unittest.TestCase):
    def test_evaluate_requires_preflight_before_any_judge_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate(root, "a", "a")
            second = _candidate(root, "b", "b")
            judge = FakeJudge()
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                PairwiseJudgeEvaluator(judge).evaluate(
                    EvaluationRequest("task/x", "prompt", candidates=(first, second), artifact_dir=root / "out")
                )
            self.assertEqual(judge.requests, [])

    def test_preflight_maps_injected_judge_and_trials_are_anonymous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate(root, "candidate-secret-a", "a")
            second = _candidate(root, "candidate-secret-b", "b")
            judge = FakeJudge()
            evaluator = PairwiseJudgeEvaluator(
                judge,
                trials=2,
                seed=42,
                model="judge-model",
                timeout_seconds=12,
                environment={"PATH": "/bin", "SECRET_CANDIDATE": "redact"},
            )

            preflight = evaluator.preflight(root / "run")
            result = evaluator.evaluate(
                EvaluationRequest("task/x", "canonical task", candidates=(first, second), artifact_dir=root / "out")
            )

            self.assertTrue(preflight.ok)
            self.assertEqual(preflight.judge_executor, "fake-judge")
            self.assertEqual(preflight.judge_executor_version, "judge-1")
            self.assertEqual(preflight.judge_auth_mode, "fake-subscription")
            self.assertEqual(result.status, EvaluationStatus.COMPLETED)
            self.assertEqual(result.metrics, {})
            candidate_a = cast(_CandidateOutcome, result.outcomes["candidate_a"])
            trial_verdicts = cast(list[object], result.outcomes["trial_verdicts"])
            self.assertEqual(candidate_a["candidate_id"], "candidate-secret-a")
            self.assertEqual(len(trial_verdicts), 2)
            self.assertEqual(judge.preflight_environment, {"PATH": "/bin"})
            self.assertEqual(len(judge.requests), 2)
            request = judge.requests[0]
            self.assertIn("BOXED[", request.task_prompt)
            self.assertTrue(request.submission_a_dir.is_dir())
            self.assertTrue(request.submission_b_dir.is_dir())
            self.assertNotIn("candidate-secret-a", str(request.workspace))
            self.assertNotIn("candidate-secret-b", str(request.workspace))
            self.assertFalse((request.submission_a_dir / "reference_files").exists())
            self.assertEqual((request.reference_dir / "ref.txt").read_text(encoding="utf-8"), "same")
            self.assertTrue(
                (root / "out" / "judge" / "tasks" / "task_x" / "trial_0" / "executor" / "metadata.json").is_file()
            )

    def test_missing_verdict_or_judge_failure_preserves_trial_logs_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate(root, "a", "a")
            second = _candidate(root, "b", "b")
            # Preserve the invalid runtime verdict to exercise fail-closed handling.
            judge = FakeJudge(verdict=cast(Verdict, None), exit_code=1)
            evaluator = PairwiseJudgeEvaluator(judge, trials=1)
            evaluator.preflight(root / "run")
            with self.assertRaisesRegex(RuntimeError, "failed closed"):
                evaluator.evaluate(
                    EvaluationRequest("task/x", "prompt", candidates=(first, second), artifact_dir=root / "out")
                )
            executor = root / "out" / "judge" / "tasks" / "task_x" / "trial_0" / "executor"
            self.assertEqual((executor / "stdout.log").read_text(encoding="utf-8"), "judge output")
            self.assertTrue((executor / "metadata.json").is_file())

    def test_nonterminal_candidate_is_rejected_without_judge_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate(root, "a", "a")
            second = _candidate(root, "b", "b")
            assert second.artifacts_dir is not None
            failed = _execution(second.artifacts_dir, "b")
            failed = ExecutionResult(
                task_id=failed.task_id,
                executor=failed.executor,
                executor_version=failed.executor_version,
                invocation_mode=failed.invocation_mode,
                auth_mode=failed.auth_mode,
                workspace=failed.workspace,
                deliverables_dir=failed.deliverables_dir,
                status=ExecutionStatus.FAILED,
                started_at=failed.started_at,
                finished_at=failed.finished_at,
                exit_code=1,
            )
            second = EvaluationCandidate("b", failed, second.artifacts_dir)
            judge = FakeJudge()
            evaluator = PairwiseJudgeEvaluator(judge)
            evaluator.preflight(root / "run")
            with self.assertRaisesRegex(ValueError, "terminal"):
                evaluator.evaluate(
                    EvaluationRequest("task/x", "prompt", candidates=(first, second), artifact_dir=root / "out")
                )
            self.assertEqual(judge.requests, [])

    def test_judge_exception_metadata_does_not_persist_secret_text(self) -> None:
        class SecretJudge(FakeJudge):
            def judge(self, request: JudgeRequest) -> JudgeResult:
                request.executor_dir.mkdir(parents=True, exist_ok=True)
                raise RuntimeError("SECRET_SENTINEL")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate(root, "a", "a")
            second = _candidate(root, "b", "b")
            judge = SecretJudge()
            evaluator = PairwiseJudgeEvaluator(judge)
            evaluator.preflight(root / "run")
            with self.assertRaisesRegex(RuntimeError, "failed") as caught:
                evaluator.evaluate(
                    EvaluationRequest("task/x", "prompt", candidates=(first, second), artifact_dir=root / "out")
                )
            self.assertNotIn("SECRET_SENTINEL", str(caught.exception))
            trial_metadata = (
                root / "out" / "judge" / "tasks" / "task_x" / "trial_0" / "executor" / "metadata.json"
            ).read_text()
            self.assertNotIn("SECRET_SENTINEL", trial_metadata)

    def test_existing_harness_metadata_is_never_overwritten(self) -> None:
        class CollisionJudge(FakeJudge):
            def judge(self, request: JudgeRequest) -> JudgeResult:
                request.executor_dir.mkdir(parents=True, exist_ok=True)
                (request.executor_dir / "metadata.json").write_text("judge metadata", encoding="utf-8")
                (request.executor_dir / "harness-metadata.json").write_text("preserve", encoding="utf-8")
                return super().judge(request)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate(root, "a", "a")
            second = _candidate(root, "b", "b")
            judge = CollisionJudge()
            evaluator = PairwiseJudgeEvaluator(judge, trials=1)
            evaluator.preflight(root / "run")
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(
                    EvaluationRequest("task/x", "prompt", candidates=(first, second), artifact_dir=root / "out")
                )
            metadata = root / "out" / "judge" / "tasks" / "task_x" / "trial_0" / "executor"
            self.assertEqual((metadata / "metadata.json").read_text(encoding="utf-8"), "judge metadata")
            self.assertEqual((metadata / "harness-metadata.json").read_text(encoding="utf-8"), "preserve")

    def test_output_root_must_be_new_and_candidate_reference_files_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate(root, "a", "a")
            second = _candidate(root, "b", "b")
            output = root / "out"
            output.mkdir()
            evaluator = PairwiseJudgeEvaluator(FakeJudge())
            evaluator.preflight(root / "run")
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(
                    EvaluationRequest("task/x", "prompt", candidates=(first, second), artifact_dir=output)
                )
            assert second.artifacts_dir is not None
            (second.artifacts_dir / "reference_files" / "ref.txt").write_text("different", encoding="utf-8")
            evaluator = PairwiseJudgeEvaluator(FakeJudge())
            evaluator.preflight(root / "run")
            with self.assertRaisesRegex(ValueError, "reference"):
                evaluator.evaluate(
                    EvaluationRequest("task/x", "prompt", candidates=(first, second), artifact_dir=root / "new-out")
                )


if __name__ == "__main__":
    unittest.main()
