# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from gdpval_harness.benchmarks.base import Benchmark, BenchmarkEvaluation, BenchmarkTask, EvaluatorType
from gdpval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec


_PROMPT = "Generate an executable Python function generated from the given prompt.\n\n{question}"
_TERMINAL_SUCCESS = {ExecutionStatus.COMPLETED, ExecutionStatus.NO_DELIVERABLE}
_BCB_PYTHON_VERSION = "3.10"
_MAX_AS_LIMIT = 30 * 1024
_MAX_DATA_LIMIT = 30 * 1024
_MAX_STACK_LIMIT = 10
_MIN_TIME_LIMIT = 1.0
_GT_TIME_LIMIT = 20.0
_SUBPROCESS_TIMEOUT = 240.0


def _is_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _native_bigcodebench_evaluate(
    output_text: str,
    verifier_metadata: Mapping[str, object],
    *,
    resource_dir: Path,
) -> dict[str, object]:
    """Run the existing BigCodeBench extractor and bcb_runner in its dedicated evaluation venv."""
    from resources_servers.bigcodebench.code_extraction import preprocess_code_completion
    from resources_servers.bigcodebench.setup_bcb_venv import ensure_bcb_venv

    extracted = preprocess_code_completion(output_text)
    if not extracted:
        return {
            "reward": 0.0,
            "status": "no_code_block",
            "extracted_model_code": None,
            "details": None,
        }

    code_prompt = str(verifier_metadata["code_prompt"])
    calibrated = code_prompt + "\n    pass\n" + extracted
    bcb_python = ensure_bcb_venv(resource_dir / ".bcb_venv", _BCB_PYTHON_VERSION)
    runner_path = resource_dir / "bcb_runner.py"
    payload = json.dumps(
        {
            "code": calibrated,
            "test_code": str(verifier_metadata["test"]),
            "entry_point": str(verifier_metadata["entry_point"]),
            "max_as_limit": _MAX_AS_LIMIT,
            "max_data_limit": _MAX_DATA_LIMIT,
            "max_stack_limit": _MAX_STACK_LIMIT,
            "min_time_limit": _MIN_TIME_LIMIT,
            "gt_time_limit": _GT_TIME_LIMIT,
        }
    )

    try:
        completed = subprocess.run(
            [str(bcb_python), str(runner_path)],
            input=payload,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_SUBPROCESS_TIMEOUT,
            cwd=resource_dir,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "reward": 0.0,
            "status": "timeout",
            "extracted_model_code": extracted,
            "details": {"reason": "outer_subprocess_timeout"},
        }
    except OSError as exc:
        return {
            "reward": 0.0,
            "status": "error",
            "extracted_model_code": extracted,
            "details": {"reason": str(exc)},
        }

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "reward": 0.0,
            "status": "error",
            "extracted_model_code": extracted,
            "details": {
                "returncode": completed.returncode,
                "stderr": completed.stderr[:2000],
                "stdout": completed.stdout[:2000],
            },
        }

    status = result.get("status")
    return {
        "reward": 1.0 if status == "pass" else 0.0,
        "status": status,
        "extracted_model_code": extracted,
        "details": result.get("details"),
    }


class BigCodeBenchBenchmark(Benchmark):
    name = "bigcodebench"
    revision = "v0.1.4"
    evaluator_type = EvaluatorType.EXECUTABLE_TESTS

    def __init__(
        self,
        *,
        root: Path,
        dataset_path: Path,
        prepare_script: Path,
        resource_dir: Path | None = None,
    ) -> None:
        self.root = root
        self.dataset_path = dataset_path
        self.prepare_script = prepare_script
        self.resource_dir = resource_dir or root / "resources_servers" / "bigcodebench"

    def is_prepared(self) -> bool:
        return self.dataset_path.is_file()

    def prepare(self) -> None:
        if self.is_prepared():
            return
        if not self.prepare_script.is_file():
            raise RuntimeError(f"BigCodeBench prepare script not found: {self.prepare_script}")
        result = subprocess.run([sys.executable, str(self.prepare_script)], cwd=self.root, check=False)
        if result.returncode != 0 or not self.is_prepared():
            raise RuntimeError("failed to prepare BigCodeBench benchmark data")

    def load_tasks(self, limit: int) -> list[BenchmarkTask]:
        if limit <= 0:
            raise ValueError("benchmark task limit must be positive")
        tasks: list[BenchmarkTask] = []
        with self.dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                verifier_metadata = row.get("verifier_metadata")
                if not isinstance(verifier_metadata, dict):
                    raise RuntimeError("BigCodeBench task is missing verifier_metadata")
                for key in ("task_id", "test", "entry_point", "code_prompt"):
                    if key not in verifier_metadata:
                        raise RuntimeError(f"BigCodeBench verifier_metadata is missing {key}")
                question = str(row["question"])
                tasks.append(
                    BenchmarkTask(
                        execution=TaskSpec(
                            task_id=str(verifier_metadata["task_id"]),
                            prompt=_PROMPT.format(question=question),
                        ),
                        evaluation=dict(verifier_metadata),
                    )
                )
                if len(tasks) >= limit:
                    break
        if not tasks:
            raise RuntimeError(f"no BigCodeBench tasks found in {self.dataset_path}")
        return tasks

    def materialize(self, task: BenchmarkTask, workspace: Path) -> list[str]:
        del task
        workspace.mkdir(parents=True, exist_ok=True)
        return []

    def evaluate(self, task: BenchmarkTask, result: ExecutionResult) -> BenchmarkEvaluation:
        if result.status not in _TERMINAL_SUCCESS:
            return BenchmarkEvaluation(
                task_id=task.execution.task_id,
                metrics={"pass_rate": 0.0},
                details={
                    "execution_status": result.status.value,
                    "grader": "resources_servers/bigcodebench/bcb_runner.py",
                    "grader_invoked": False,
                },
            )

        output_text = result.output_text or ""
        if not output_text.strip():
            return BenchmarkEvaluation(
                task_id=task.execution.task_id,
                metrics={"pass_rate": 0.0},
                details={
                    "execution_status": result.status.value,
                    "status": "empty_output",
                    "grader": "resources_servers/bigcodebench/bcb_runner.py",
                    "grader_invoked": False,
                },
            )

        workspace = result.workspace.resolve()
        grader_root = self.resource_dir.resolve()
        if _is_inside(workspace, grader_root) or _is_inside(grader_root, workspace):
            raise RuntimeError("BigCodeBench grader directory must be separate from the executor workspace")

        native = _native_bigcodebench_evaluate(output_text, task.evaluation, resource_dir=grader_root)
        return BenchmarkEvaluation(
            task_id=task.execution.task_id,
            metrics={"pass_rate": float(native["reward"])},
            details={
                "execution_status": result.status.value,
                "status": native.get("status"),
                "extracted_model_code": native.get("extracted_model_code"),
                "grader_details": native.get("details"),
                "grader": "resources_servers/bigcodebench/bcb_runner.py",
                "grader_invoked": True,
                "grader_root": str(grader_root),
                "executor_workspace": str(workspace),
            },
        )
