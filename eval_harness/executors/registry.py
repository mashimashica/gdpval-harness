# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from eval_harness.executors.base import Executor
from eval_harness.reasoning import ReasoningEffortOption, validate_executor_reasoning_effort


@dataclass(frozen=True)
class ExecutorDescriptor:
    name: str
    runtime: str
    usage_mode: str
    generic_runner_status: str


_EXECUTORS = {
    "claude-code": ExecutorDescriptor(
        name="claude-code",
        runtime="Claude Code (local)",
        usage_mode="Claude subscription login only",
        generic_runner_status="supported where listed by benchmark",
    ),
    "codex": ExecutorDescriptor(
        name="codex",
        runtime="Codex CLI (local)",
        usage_mode="ChatGPT subscription login only",
        generic_runner_status="supported",
    ),
    "cursor": ExecutorDescriptor(
        name="cursor",
        runtime="Cursor Agent CLI (local)",
        usage_mode="Cursor account/plan usage; API-key auth rejected",
        generic_runner_status="supported where listed by benchmark",
    ),
    "stirrup": ExecutorDescriptor(
        name="stirrup",
        runtime="NeMo Gym / Stirrup",
        usage_mode="provider-backed model API",
        generic_runner_status="legacy ./gdpval path only",
    ),
}


def get_executor_descriptor(name: str) -> ExecutorDescriptor:
    try:
        return _EXECUTORS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_EXECUTORS))
        raise ValueError(f"unknown executor {name!r}; available: {available}") from exc


def list_executors() -> tuple[ExecutorDescriptor, ...]:
    return tuple(_EXECUTORS[name] for name in sorted(_EXECUTORS))


def create_executor(
    name: str,
    *,
    network_enabled: bool = False,
    claude_max_turns: int = 250,
    reasoning_effort: ReasoningEffortOption = None,
) -> Executor:
    validate_executor_reasoning_effort(name, reasoning_effort)
    if name == "codex":
        from eval_harness.executors.codex import CodexExecutor

        return CodexExecutor(network_enabled=network_enabled, reasoning_effort=reasoning_effort)
    if name == "claude-code":
        from eval_harness.executors.claude_code import ClaudeCodeExecutor

        return ClaudeCodeExecutor(network_enabled=network_enabled, max_turns=claude_max_turns)
    if name == "cursor":
        from eval_harness.executors.cursor import CursorExecutor

        return CursorExecutor(network_enabled=network_enabled)
    descriptor = get_executor_descriptor(name)
    raise ValueError(f"executor {descriptor.name!r} is not available through the generic local runner")


def main() -> None:
    for descriptor in list_executors():
        print(f"{descriptor.name}\t{descriptor.runtime}\t{descriptor.usage_mode}\t{descriptor.generic_runner_status}")


if __name__ == "__main__":
    main()
