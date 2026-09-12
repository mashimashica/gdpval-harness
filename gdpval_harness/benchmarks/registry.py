# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gdpval_harness.benchmarks.aime26 import AIME26Benchmark
from gdpval_harness.benchmarks.base import Benchmark
from gdpval_harness.benchmarks.bigcodebench import BigCodeBenchBenchmark
from gdpval_harness.benchmarks.gdpval import GDPvalBenchmark


@dataclass(frozen=True)
class BenchmarkDescriptor:
    name: str
    status: str
    supported_executors: tuple[str, ...]
    assets: tuple[str, ...]
    sandbox_requirement: str
    network_requirement: str


_BENCHMARKS = {
    "gdpval": BenchmarkDescriptor(
        name="gdpval",
        status="supported; task execution and reference-file materialization",
        supported_executors=("codex", "claude-code", "cursor"),
        assets=("benchmarks/gdpval/data/gdpval_benchmark.jsonl", "reference_files"),
        sandbox_requirement="per-task writable workspace; reference files must remain unchanged",
        network_requirement=(
            "disabled for policy execution by default; preparation/reference download may require network"
        ),
    ),
    "aime26": BenchmarkDescriptor(
        name="aime26",
        status="supported; task execution and prompt materialization",
        supported_executors=("codex",),
        assets=("benchmarks/aime26/data/aime26_benchmark.jsonl",),
        sandbox_requirement="per-task writable workspace",
        network_requirement="disabled for policy execution; preparation may require network",
    ),
    "bigcodebench": BenchmarkDescriptor(
        name="bigcodebench",
        status="supported; task execution and prompt materialization",
        supported_executors=("codex",),
        assets=("benchmarks/bigcodebench/data/bigcodebench_benchmark.jsonl",),
        sandbox_requirement="per-task writable workspace",
        network_requirement="disabled for policy execution; preparation may require network",
    ),
}


def list_benchmarks() -> tuple[BenchmarkDescriptor, ...]:
    return tuple(_BENCHMARKS[name] for name in sorted(_BENCHMARKS))


def get_benchmark_descriptor(name: str) -> BenchmarkDescriptor:
    try:
        return _BENCHMARKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BENCHMARKS))
        raise ValueError(f"unknown benchmark {name!r}; available: {available}") from exc


def create_benchmark(name: str, *, root: Path | None = None) -> Benchmark:
    repo_root = root or Path(__file__).resolve().parents[2]
    if name == "gdpval":
        return GDPvalBenchmark(
            root=repo_root,
            dataset_path=repo_root / "benchmarks" / "gdpval" / "data" / "gdpval_benchmark.jsonl",
            prepare_script=repo_root / "benchmarks" / "gdpval" / "prepare.py",
        )
    if name == "aime26":
        return AIME26Benchmark(
            root=repo_root,
            dataset_path=repo_root / "benchmarks" / "aime26" / "data" / "aime26_benchmark.jsonl",
            prepare_script=repo_root / "benchmarks" / "aime26" / "prepare.py",
        )
    if name == "bigcodebench":
        return BigCodeBenchBenchmark(
            root=repo_root,
            dataset_path=repo_root / "benchmarks" / "bigcodebench" / "data" / "bigcodebench_benchmark.jsonl",
            prepare_script=repo_root / "benchmarks" / "bigcodebench" / "prepare.py",
        )
    get_benchmark_descriptor(name)
    raise AssertionError("unreachable")
