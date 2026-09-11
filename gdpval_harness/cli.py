# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from gdpval_harness.benchmarks.registry import create_benchmark, get_benchmark_descriptor, list_benchmarks
from gdpval_harness.builders import ExecutorSkillBuilder
from gdpval_harness.evaluators.registry import create_evaluator, get_evaluator_descriptor
from gdpval_harness.executors.registry import create_executor, get_executor_descriptor, list_executors
from gdpval_harness.experiments.base import ExperimentRunConfig
from gdpval_harness.experiments.profile import load_experiment_profile
from gdpval_harness.experiments.runner import run_builder_experiment
from gdpval_harness.interventions.registry import create_intervention
from gdpval_harness.runner import run_benchmark


def _default_out(benchmark: str, executor: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / "eval" / benchmark / f"{stamp}-{executor}"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="./eval", description="Local multi-benchmark evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("benchmarks", help="List registered benchmarks and support requirements")
    subparsers.add_parser("executors", help="List registered executors")

    run_parser = subparsers.add_parser("run", help="Run one benchmark with a local executor")
    run_parser.add_argument("benchmark")
    run_parser.add_argument("--executor", default="codex")
    run_parser.add_argument("--limit", required=True, type=int)
    run_parser.add_argument("--model")
    run_parser.add_argument("--out", type=Path)
    run_parser.add_argument("--executor-timeout", type=float, default=12600.0)
    run_parser.add_argument("--network", action="store_true", help="Explicitly enable policy-executor network access")
    run_parser.add_argument("--claude-max-turns", type=int, default=250)
    run_parser.add_argument(
        "--intervention",
        choices=("none", "prompt-overlay", "files", "agent-skill"),
        default="none",
        help=(
            "Apply a registered intervention before execution; agent-skill uses the portable "
            "workspace-reference method (default: none)"
        ),
    )
    run_parser.add_argument(
        "--intervention-source",
        type=Path,
        help="Source file/directory for the selected non-none intervention",
    )

    experiment_parser = subparsers.add_parser("experiment", help="Run a Builder experiment profile")
    experiment_parser.add_argument("profile", type=Path, metavar="PROFILE")
    experiment_parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        metavar="INPUT_ID=DIR",
        help="Bind one profile input ID to its source directory; repeat for each input",
    )
    experiment_parser.add_argument("--limit", required=True, type=_positive_int)
    experiment_parser.add_argument("--order-seed", required=True, type=_nonnegative_int)
    experiment_parser.add_argument("--out", required=True, type=Path)
    experiment_parser.add_argument("--runtime-root", required=True, type=Path)
    experiment_parser.add_argument("--builder-executor", default="codex")
    experiment_parser.add_argument("--executor", default="codex", dest="executor")
    experiment_parser.add_argument("--builder-model", dest="builder_model")
    experiment_parser.add_argument("--model", dest="model")
    experiment_parser.add_argument("--builder-timeout", type=_positive_float, default=12600.0)
    experiment_parser.add_argument("--executor-timeout", type=_positive_float, default=12600.0)
    experiment_parser.add_argument(
        "--builder-network",
        action="store_true",
        help="Explicitly enable network access for Builder execution",
    )
    experiment_parser.add_argument(
        "--network",
        action="store_true",
        help="Explicitly enable network access for application execution",
    )
    experiment_parser.add_argument("--builder-claude-max-turns", type=_positive_int, default=250)
    experiment_parser.add_argument("--claude-max-turns", type=_positive_int, default=250)
    return parser


def _print_benchmarks() -> None:
    for descriptor in list_benchmarks():
        evaluator = get_evaluator_descriptor(descriptor.name)
        executors = ",".join(descriptor.supported_executors)
        assets = ",".join(descriptor.assets)
        evaluator_assets = ",".join(evaluator.assets)
        requirements = ",".join(evaluator.requirements)
        print(
            f"{descriptor.name}\tbenchmark_status={descriptor.status}\t"
            f"evaluator={evaluator.name}\tevaluator_status={evaluator.status}\t"
            f"type={evaluator.evaluator_type.value}\texecutors={executors}\t"
            f"benchmark_assets={assets}\tevaluator_assets={evaluator_assets}\t"
            f"requirements={requirements}\tisolation={evaluator.isolation_requirement or 'none'}\t"
            f"sandbox={descriptor.sandbox_requirement}\tnetwork={descriptor.network_requirement}"
        )


def _print_executors() -> None:
    for descriptor in list_executors():
        print(
            f"{descriptor.name}\t{descriptor.runtime}\t{descriptor.usage_mode}\t"
            f"{descriptor.generic_runner_status}"
        )


def _run(args: argparse.Namespace) -> int:
    benchmark_descriptor = get_benchmark_descriptor(args.benchmark)
    get_executor_descriptor(args.executor)
    if args.executor not in benchmark_descriptor.supported_executors:
        supported = ", ".join(benchmark_descriptor.supported_executors)
        raise ValueError(
            f"benchmark {args.benchmark!r} does not support executor {args.executor!r} through ./eval; "
            f"supported: {supported}"
        )
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.executor_timeout <= 0:
        raise ValueError("--executor-timeout must be positive")
    if args.claude_max_turns <= 0:
        raise ValueError("--claude-max-turns must be positive")
    if args.intervention == "none" and args.intervention_source is not None:
        raise ValueError("--intervention-source requires --intervention prompt-overlay or files")
    if args.intervention != "none" and args.intervention_source is None:
        raise ValueError(f"--intervention-source is required for --intervention {args.intervention}")

    intervention = create_intervention(args.intervention, source=args.intervention_source)
    benchmark = create_benchmark(args.benchmark)
    evaluator = create_evaluator(args.benchmark)
    executor = create_executor(
        args.executor,
        network_enabled=args.network,
        claude_max_turns=args.claude_max_turns,
    )
    out_dir = args.out or _default_out(args.benchmark, args.executor)
    summary = run_benchmark(
        benchmark,
        evaluator,
        executor,
        out_dir=out_dir,
        limit=args.limit,
        model=args.model,
        timeout_seconds=args.executor_timeout,
        intervention=intervention,
    )
    print(
        json.dumps(
            {
                "benchmark": summary.benchmark,
                "executor": summary.executor,
                "out": str(summary.out_dir),
                "status": summary.status,
                "task_count": summary.task_count,
                "metrics": dict(summary.metrics),
                "evaluation_status_counts": dict(summary.evaluation_status_counts),
            },
            sort_keys=True,
        )
    )
    return 0 if summary.status == "completed" else 1


def _parse_input_roots(bindings: list[str]) -> dict[str, Path]:
    source_roots: dict[str, Path] = {}
    for binding in bindings:
        input_id, separator, directory = binding.partition("=")
        input_id = input_id.strip()
        directory = directory.strip()
        if not separator or not input_id or not directory:
            raise ValueError("--input-root must use a non-empty INPUT_ID=DIR binding")
        if input_id in source_roots:
            raise ValueError(f"duplicate --input-root binding for input {input_id!r}")
        source_roots[input_id] = Path(directory)
    return source_roots


def _planned_absolute_path(path: Path, *, label: str) -> Path:
    try:
        planned = path.absolute()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"could not resolve {label}: {path}") from exc
    if not planned.is_absolute():
        raise ValueError(f"{label} must resolve to an absolute path")
    return planned


def _experiment(args: argparse.Namespace) -> int:
    loaded_profile = load_experiment_profile(args.profile.absolute())
    profile = loaded_profile.profile
    source_roots = _parse_input_roots(args.input_root)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.order_seed < 0:
        raise ValueError("--order-seed must be non-negative")
    for label, value in (
        ("--builder-timeout", args.builder_timeout),
        ("--executor-timeout", args.executor_timeout),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    if args.builder_claude_max_turns <= 0:
        raise ValueError("--builder-claude-max-turns must be positive")
    if args.claude_max_turns <= 0:
        raise ValueError("--claude-max-turns must be positive")

    benchmark_name = profile.benchmark
    benchmark_descriptor = get_benchmark_descriptor(benchmark_name)
    get_executor_descriptor(args.builder_executor)
    get_executor_descriptor(args.executor)
    if args.executor not in benchmark_descriptor.supported_executors:
        supported = ", ".join(benchmark_descriptor.supported_executors)
        raise ValueError(
            f"benchmark {benchmark_name!r} does not support executor {args.executor!r} through ./eval; "
            f"supported: {supported}"
        )

    benchmark = create_benchmark(benchmark_name)
    evaluator = create_evaluator(benchmark_name)
    builder_executor = create_executor(
        args.builder_executor,
        network_enabled=args.builder_network,
        claude_max_turns=args.builder_claude_max_turns,
    )
    application_executor = create_executor(
        args.executor,
        network_enabled=args.network,
        claude_max_turns=args.claude_max_turns,
    )
    builder = ExecutorSkillBuilder(builder_executor)
    run_config = ExperimentRunConfig(
        builder_executor=builder_executor.name,
        application_executor=application_executor.name,
        evaluator=evaluator.name,
        builder_model=args.builder_model,
        application_model=args.model,
        builder_timeout_seconds=args.builder_timeout,
        application_timeout_seconds=args.executor_timeout,
        builder_network_enabled=args.builder_network,
        application_network_enabled=args.network,
        limit=args.limit,
        order_seed=args.order_seed,
    )
    out_dir = _planned_absolute_path(args.out, label="--out")
    runtime_root = _planned_absolute_path(args.runtime_root, label="--runtime-root")
    summary = run_builder_experiment(
        loaded_profile,
        run_config,
        benchmark,
        evaluator,
        builder,
        application_executor,
        source_roots=source_roots,
        out_dir=out_dir,
        runtime_root=runtime_root,
    )
    status = getattr(summary.status, "value", summary.status)
    status = str(status)
    if status not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"unexpected experiment status: {status!r}")
    print(
        json.dumps(
            {
                "application_executor": run_config.application_executor,
                "arm_count": summary.arm_count,
                "benchmark": profile.benchmark,
                "builder_executor": run_config.builder_executor,
                "completed_applications": summary.completed_applications,
                "out": str(summary.out_dir),
                "profile": profile.profile_id,
                "runtime_root": str(summary.runtime_root),
                "status": status,
                "task_count": summary.task_count,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmarks":
            _print_benchmarks()
            return 0
        if args.command == "executors":
            _print_executors()
            return 0
        if args.command == "run":
            return _run(args)
        if args.command == "experiment":
            return _experiment(args)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
