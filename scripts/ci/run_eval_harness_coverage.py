# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the deterministic Eval Harness suite under subprocess-aware coverage."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[2]
MINIMUM_COVERAGE_PERCENT = 96


def _summary_integer(summary: Mapping[str, object], field: str) -> int:
    value = summary.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"coverage summary field {field!r} must be an integer")
    return value


def coverage_summary_passes(summary: Mapping[str, object]) -> bool:
    covered_lines = _summary_integer(summary, "covered_lines")
    num_statements = _summary_integer(summary, "num_statements")
    if covered_lines < 0 or num_statements <= 0 or covered_lines > num_statements:
        raise ValueError("coverage summary has invalid statement totals")
    return covered_lines * 100 >= num_statements * MINIMUM_COVERAGE_PERCENT


def check_coverage_json(path: Path) -> None:
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError("coverage JSON must contain an object")
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("coverage JSON must contain a totals object")
    covered_lines = _summary_integer(totals, "covered_lines")
    num_statements = _summary_integer(totals, "num_statements")
    if not coverage_summary_passes(totals):
        raise ValueError(
            f"{covered_lines}/{num_statements} statements is below the exact {MINIMUM_COVERAGE_PERCENT}% threshold"
        )
    print(
        f"coverage guard passed: {covered_lines}/{num_statements} statements meet the exact "
        f"{MINIMUM_COVERAGE_PERCENT}% threshold"
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        if len(arguments) != 2 or arguments[0] != "--check-summary":
            print(f"usage: {Path(__file__).name} [--check-summary COVERAGE_JSON]", file=sys.stderr)
            return 2
        try:
            check_coverage_json(Path(arguments[1]))
        except (OSError, ValueError) as error:
            print(f"coverage guard failed: {error}", file=sys.stderr)
            return 1
        return 0

    result = subprocess.run(["bash", "tests/harness/test_gdpval_cli.sh"], cwd=ROOT, check=False)
    return result.returncode if result.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
