# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutorDescriptor:
    name: str
    runtime: str
    usage_mode: str


_EXECUTORS = {
    "claude-code": ExecutorDescriptor(
        name="claude-code",
        runtime="Claude Code (local)",
        usage_mode="Claude subscription login only",
    ),
    "codex": ExecutorDescriptor(
        name="codex",
        runtime="Codex CLI (local)",
        usage_mode="ChatGPT subscription login only",
    ),
    "stirrup": ExecutorDescriptor(
        name="stirrup",
        runtime="NeMo Gym / Stirrup",
        usage_mode="provider-backed model API",
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


def main() -> None:
    for descriptor in list_executors():
        print(f"{descriptor.name}\t{descriptor.runtime}\t{descriptor.usage_mode}")


if __name__ == "__main__":
    main()
