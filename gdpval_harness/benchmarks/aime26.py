# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkEvaluation, BenchmarkTask, EvaluatorType
from gdpval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec


_MATH_PROMPT = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{question}"
_TERMINAL_SUCCESS = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}


def _native_math_evaluate(expected_answer: str, generated_answer: str) -> tuple[float, str | None]:
    """Reuse the benchmark's existing math_verify implementation without its optional LLM fallback."""
    from resources_servers.math_with_judge.app import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        _extract_last_boxed_answer,
        _run_math_verify,
        math_metric,
    )

    boxed_answer = _extract_last_boxed_answer(generated_answer)
    if boxed_answer is None or not boxed_answer.strip():
        return 0.0, None

    verifier = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    return _run_math_verify(verifier, expected_answer, generated_answer)


class AIME26Benchmark(Benchmark):
    name = "aime26"
    # Upstream prepare.py currently loads MathArena/aime_2026 without a pinned dataset revision.
    revision = None
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def __init__(self, *, root: Path, dataset_path: Path, prepare_script: Path) -> None:
        self.root = root
        self.dataset_path = dataset_path
        self.prepare_script = prepare_script

    def is_prepared(self) -> bool:
        return self.dataset_path.is_file()

    def prepare(self) -> None:
        if self.is_prepared():
            return
        if not self.prepare_script.is_file():
            raise RuntimeError(f"AIME26 prepare script not found: {self.prepare_script}")
        result = subprocess.run([sys.executable, str(self.prepare_script)], cwd=self.root, check=False)
        if result.returncode != 0 or not self.is_prepared():
            raise RuntimeError("failed to prepare AIME26 benchmark data")

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        if limit <= 0:
            raise ValueError("benchmark task limit must be positive")
        tasks: list[BenchmarkTask] = []
        with self.dataset_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                question = str(row["question"])
                tasks.append(
                    BenchmarkTask(
                        execution=TaskSpec(
                            task_id=f"aime26-{index:02d}",
                            prompt=_MATH_PROMPT.format(question=question),
                        ),
                        evaluation={
                            "question": question,
                            "expected_answer": str(row["expected_answer"]),
                        },
                    )
                )
                if len(tasks) >= limit:
                    break
        if not tasks:
            raise RuntimeError(f"no AIME26 tasks found in {self.dataset_path}")
        return tasks

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        del task
        workspace.mkdir(parents=True, exist_ok=True)
        return []

    def evaluate(self, task: BenchmarkTask, result: ExecutionResult) -> BenchmarkEvaluation:
        expected_answer = str(task.evaluation["expected_answer"])
        if result.status not in _TERMINAL_SUCCESS:
            return BenchmarkEvaluation(
                task_id=task.execution.task_id,
                metrics={"accuracy": 0.0},
                details={
                    "expected_answer": expected_answer,
                    "extracted_answer": None,
                    "execution_status": result.status.value,
                    "verifier": "math_with_judge/library",
                    "llm_judge_used": False,
                },
            )
        output_text = result.output_text or ""
        if not output_text.strip():
            return BenchmarkEvaluation(
                task_id=task.execution.task_id,
                metrics={"accuracy": 0.0},
                details={
                    "expected_answer": expected_answer,
                    "extracted_answer": None,
                    "execution_status": result.status.value,
                    "verifier": "math_with_judge/library",
                    "llm_judge_used": False,
                    "reason": "empty executor output",
                },
            )

        reward, extracted_answer = _native_math_evaluate(expected_answer, output_text)
        return BenchmarkEvaluation(
            task_id=task.execution.task_id,
            metrics={"accuracy": float(reward)},
            details={
                "expected_answer": expected_answer,
                "extracted_answer": extracted_answer,
                "execution_status": result.status.value,
                "verifier": "math_with_judge/library",
                "llm_judge_used": False,
            },
        )
