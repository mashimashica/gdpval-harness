# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory and descriptors for generic-runner evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from eval_harness.evaluators.base import Evaluator, EvaluatorType


if TYPE_CHECKING:
    from eval_harness.judges.base import JudgeExecutor


@dataclass(frozen=True)
class EvaluatorDescriptor:
    """Stable CLI metadata for an evaluator implementation."""

    name: str
    benchmark: str
    evaluator_type: EvaluatorType
    status: str
    version: str | None = None
    revision: str | None = None
    requirements: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    isolation_requirement: str | None = None
    judge: str | None = None


_EVALUATORS = {
    "gdpval": EvaluatorDescriptor(
        name="gdpval-external",
        benchmark="gdpval",
        evaluator_type=EvaluatorType.LLM_RUBRIC,
        status=(
            "external handoff; execution exports judge-compatible artifacts, while the real GDPval "
            "rubric/pairwise evaluation remains external"
        ),
        version="1",
        revision="external-handoff-v1",
        isolation_requirement="publishes artifacts only; the existing rubric/pairwise path remains external",
        judge="existing GDPval rubric/pairwise path",
    ),
    "aime26": EvaluatorDescriptor(
        name="aime26-native",
        benchmark="aime26",
        evaluator_type=EvaluatorType.BENCHMARK_NATIVE,
        status="native math verification with pinned local dependency",
        version="0.8.0",
        requirements=("math-verify==0.8.0",),
        assets=("resources_servers/math_with_judge/requirements.txt",),
        isolation_requirement="local evaluator in the generic runner process",
    ),
    "bigcodebench": EvaluatorDescriptor(
        name="bigcodebench-tests",
        benchmark="bigcodebench",
        evaluator_type=EvaluatorType.EXECUTABLE_TESTS,
        status="native executable tests in the dedicated grader environment",
        version="1",
        revision="v0.1.4",
        assets=("resources_servers/bigcodebench/.bcb_venv",),
        isolation_requirement="separate BigCodeBench grader venv and subprocess",
    ),
}


def list_evaluator_descriptors() -> tuple[EvaluatorDescriptor, ...]:
    return tuple(_EVALUATORS[name] for name in sorted(_EVALUATORS))


def list_evaluators() -> tuple[EvaluatorDescriptor, ...]:
    """Compatibility spelling used by CLI callers."""

    return list_evaluator_descriptors()


def get_evaluator_descriptor(name: str) -> EvaluatorDescriptor:
    """Return metadata by benchmark key or evaluator implementation name."""

    if name in _EVALUATORS:
        return _EVALUATORS[name]
    for descriptor in _EVALUATORS.values():
        if descriptor.name == name:
            return descriptor
    available = ", ".join(sorted(_EVALUATORS))
    raise ValueError(f"unknown evaluator or benchmark {name!r}; available: {available}")


def create_evaluator(
    name: str,
    *,
    root: Path | None = None,
) -> Evaluator:
    """Create the evaluator associated with a benchmark.

    Pairwise judging is intentionally absent from this benchmark-key factory;
    callers must construct :class:`PairwiseJudgeEvaluator` explicitly and
    inject a JudgeExecutor.  In particular, choosing ``gdpval`` can never
    instantiate a judge as a side effect.
    """

    descriptor = get_evaluator_descriptor(name)
    repo_root = root or Path(__file__).resolve().parents[2]
    if descriptor.benchmark == "gdpval":
        from eval_harness.evaluators.gdpval import GDPvalExternalEvaluator

        return GDPvalExternalEvaluator()
    if descriptor.benchmark == "aime26":
        from eval_harness.evaluators.aime26 import AIME26Evaluator

        return AIME26Evaluator()
    if descriptor.benchmark == "bigcodebench":
        from eval_harness.evaluators.bigcodebench import BigCodeBenchEvaluator

        return BigCodeBenchEvaluator(resource_dir=repo_root / "resources_servers" / "bigcodebench")
    # The descriptor table is static, so this is defensive only.
    raise AssertionError(f"no evaluator factory for {descriptor.benchmark!r}")


def create_pairwise_evaluator(
    judge_executor: JudgeExecutor,
    *,
    trials: int = 2,
    seed: int = 42,
    model: str | None = None,
    timeout_seconds: float = 3600.0,
    environment: dict[str, str] | None = None,
) -> Evaluator:
    """Explicit opt-in helper; never used by the GDPval CLI default path."""

    from eval_harness.evaluators.pairwise import PairwiseJudgeEvaluator

    return PairwiseJudgeEvaluator(
        judge_executor,
        trials=trials,
        seed=seed,
        model=model,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )


__all__ = [
    "EvaluatorDescriptor",
    "create_evaluator",
    "create_pairwise_evaluator",
    "get_evaluator_descriptor",
    "list_evaluator_descriptors",
    "list_evaluators",
]
