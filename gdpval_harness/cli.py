# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gdpval_harness.benchmarks.registry import create_benchmark, get_benchmark_descriptor, list_benchmarks
from gdpval_harness.evaluators.registry import create_evaluator, get_evaluator_descriptor
from gdpval_harness.executors.registry import create_executor, get_executor_descriptor, list_executors
from gdpval_harness.interventions.registry import create_intervention
from gdpval_harness.runner import run_benchmark


def _default_out(benchmark: str, executor: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / "eval" / benchmark / f"{stamp}-{executor}"


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
        choices=("none", "prompt-overlay", "files"),
        default="none",
        help="Apply a registered intervention before execution (default: none)",
    )
    run_parser.add_argument(
        "--intervention-source",
        type=Path,
        help="Source file/directory for the selected non-none intervention",
    )
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
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
