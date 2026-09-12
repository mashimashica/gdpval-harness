# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from subprocess import CompletedProcess
from typing import cast
from unittest.mock import patch

import eval_harness.provenance as provenance_module
from eval_harness.executors.base import ExecutionResult, ExecutionStatus, TaskSpec
from eval_harness.provenance import (
    RepositoryProvenance,
    canonical_json_sha256,
    execution_record,
    repository_provenance,
    task_sha256,
)


_SHA1 = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
_SHA256 = "0123456789abcdef" * 4


class _ForbiddenMapping(Mapping[str, object]):
    def __getitem__(self, key: object) -> object:
        raise AssertionError(f"metadata key accessed: {key!r}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("metadata iterated")

    def __len__(self) -> int:
        raise AssertionError("metadata length inspected")


class _StringSentinel:
    def __str__(self) -> str:
        raise AssertionError("arbitrary object was coerced")


class ProvenanceTests(unittest.TestCase):
    def test_public_exports_and_repository_provenance_invariants(self) -> None:
        self.assertEqual(
            tuple(provenance_module.__all__),
            (
                "RepositoryProvenance",
                "canonical_json_sha256",
                "task_sha256",
                "repository_provenance",
                "execution_record",
            ),
        )
        self.assertEqual(
            RepositoryProvenance(_SHA1, "available", "clean"),
            RepositoryProvenance(_SHA1, "available", "clean"),
        )
        self.assertEqual(RepositoryProvenance(_SHA256, "available", "unavailable").worktree_status, "unavailable")
        with self.assertRaises(FrozenInstanceError):
            setattr(RepositoryProvenance(_SHA1, "available", "clean"), "commit", _SHA256)

        invalid: tuple[tuple[object, object, object], ...] = (
            (None, "available", "clean"),
            (None, [], "clean"),
            (None, "unavailable", []),
            ("A" * 40, "available", "clean"),
            ("a" * 39, "available", "clean"),
            ("a" * 41, "available", "clean"),
            ("a" * 40, "available", "unknown"),
            ("a" * 40, "unknown", "unavailable"),
            (_SHA1, "unavailable", "unavailable"),
            (None, "unavailable", "clean"),
            (None, "unavailable", "dirty"),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                # These values intentionally cross the runtime-invalid constructor boundary.
                RepositoryProvenance(*cast(tuple[str | None, str, str], values))

    def test_canonical_json_hash_is_ordered_unicode_and_compact(self) -> None:
        first = {"z": ["é", 3], "a": {"b": True, "a": None}}
        second = {"a": {"a": None, "b": True}, "z": ["é", 3]}
        encoded = json.dumps(
            first,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected = hashlib.sha256(encoded).hexdigest()
        self.assertEqual(canonical_json_sha256(first), expected)
        self.assertEqual(canonical_json_sha256(first), canonical_json_sha256(second))
        self.assertEqual(encoded, '{"a":{"a":null,"b":true},"z":["é",3]}'.encode("utf-8"))

    def test_canonical_json_hash_rejects_nonfinite_and_arbitrary_values(self) -> None:
        for nonfinite in (math.nan, math.inf, -math.inf):
            with self.subTest(value=nonfinite), self.assertRaises(ValueError):
                canonical_json_sha256(nonfinite)
        for unsupported in ({"bytes": b"secret"}, {"set": {1, 2}}, _StringSentinel()):
            with self.subTest(value=type(unsupported).__name__), self.assertRaises(TypeError):
                canonical_json_sha256(unsupported)

    def test_task_hash_uses_only_exact_prompt_utf8_bytes(self) -> None:
        prompt = "héllo\n世界"
        expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.assertEqual(task_sha256(TaskSpec("task-a", prompt)), expected)
        self.assertEqual(task_sha256(TaskSpec("credential-task", prompt)), expected)
        self.assertNotEqual(task_sha256(TaskSpec("task-a", prompt + "!")), expected)
        with self.assertRaises(TypeError):
            # Preserve the invalid runtime object at the task hashing boundary.
            task_sha256(cast(TaskSpec, object()))

        class ChildTask(TaskSpec):
            pass

        with self.assertRaises(TypeError):
            task_sha256(ChildTask("task", prompt))
        with self.assertRaises(TypeError):
            # Preserve the invalid runtime prompt at the task hashing boundary.
            task_sha256(TaskSpec("task", cast(str, 3)))

    def test_repository_provenance_observes_clean_dirty_and_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repo"
            repository.mkdir()
            tracked = repository / "tracked.txt"
            tracked.write_text("committed", encoding="utf-8")
            self._git(repository, "init")
            self._git(repository, "config", "user.email", "provenance@example.invalid")
            self._git(repository, "config", "user.name", "Provenance Test")
            self._git(repository, "add", "tracked.txt")
            self._git(repository, "commit", "-m", "initial")
            commit = self._git(repository, "rev-parse", "HEAD").stdout.decode("ascii").strip()

            observed = repository_provenance(repository)
            self.assertEqual(observed, RepositoryProvenance(commit, "available", "clean"))

            tracked.write_text("dirty", encoding="utf-8")
            self.assertEqual(repository_provenance(repository).worktree_status, "dirty")
            (repository / "untracked.txt").write_text("untracked", encoding="utf-8")
            self.assertEqual(repository_provenance(repository).worktree_status, "dirty")

            unavailable = repository_provenance(Path(temporary) / "missing")
            self.assertEqual(unavailable, RepositoryProvenance(None, "unavailable", "unavailable"))

    def test_repository_commands_are_read_only_bounded_and_secret_safe(self) -> None:
        start = Path("/tmp/provenance-secret-source")
        head = CompletedProcess([], 0, stdout=f"{_SHA1}\n", stderr="secret stderr")
        status = CompletedProcess([], 0, stdout=" M private-name.txt\n", stderr="secret status stderr")
        original_optional = os.environ.get("GIT_OPTIONAL_LOCKS")
        original_prompt = os.environ.get("GIT_TERMINAL_PROMPT")
        with patch("eval_harness.provenance.subprocess.run", side_effect=[head, status]) as run:
            observed = repository_provenance(start)

        self.assertEqual(observed, RepositoryProvenance(_SHA1, "available", "dirty"))
        self.assertNotIn("private-name.txt", repr(observed))
        self.assertNotIn("secret", repr(observed))
        self.assertEqual(run.call_count, 2)
        expected_prefix = [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(start),
        ]
        self.assertEqual(run.call_args_list[0].args[0], [*expected_prefix, "rev-parse", "--verify", "HEAD"])
        self.assertEqual(
            run.call_args_list[1].args[0],
            [*expected_prefix, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=all"],
        )
        for call in run.call_args_list:
            kwargs = call.kwargs
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "replace")
            self.assertFalse(kwargs["check"])
            self.assertGreater(kwargs["timeout"], 0)
            self.assertTrue(math.isfinite(kwargs["timeout"]))
            self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(os.environ.get("GIT_OPTIONAL_LOCKS"), original_optional)
        self.assertEqual(os.environ.get("GIT_TERMINAL_PROMPT"), original_prompt)

    def test_repository_failures_are_unavailable_without_status_details(self) -> None:
        start = Path("/tmp/provenance-source")
        cases = (
            CompletedProcess([], 1, stdout="not-a-head", stderr="failure details"),
            CompletedProcess([], 0, stdout="malformed-head\nextra", stderr=""),
        )
        for head in cases:
            with self.subTest(head=head), patch("eval_harness.provenance.subprocess.run", return_value=head) as run:
                observed = repository_provenance(start)
            self.assertEqual(observed, RepositoryProvenance(None, "unavailable", "unavailable"))
            self.assertEqual(run.call_count, 1)

        head = CompletedProcess([], 0, stdout=f"{_SHA1}\n", stderr="")
        status_failure = CompletedProcess([], 1, stdout="", stderr="status secret")
        with patch("eval_harness.provenance.subprocess.run", side_effect=[head, status_failure]) as run:
            observed = repository_provenance(start)
        self.assertEqual(observed, RepositoryProvenance(_SHA1, "available", "unavailable"))
        self.assertEqual(run.call_count, 2)

    def test_execution_record_preserves_typed_fields_and_discards_metadata(self) -> None:
        result = ExecutionResult(
            task_id="secret-task-id",
            executor="executor",
            executor_version="version",
            invocation_mode="command",
            auth_mode="credential-mode",
            workspace=Path("/private/workspace"),
            deliverables_dir=Path("/private/deliverables"),
            status=ExecutionStatus.COMPLETED,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            exit_code=0,
            output_text="secret output text",
            metadata=_ForbiddenMapping(),
        )

        record = execution_record(result)

        self.assertEqual(
            set(record),
            {
                "status",
                "executor",
                "executor_version",
                "invocation_mode",
                "auth_mode",
                "workspace",
                "deliverables_dir",
                "started_at",
                "finished_at",
                "exit_code",
                "output_text_present",
                "metadata",
            },
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["workspace"], "/private/workspace")
        self.assertEqual(record["deliverables_dir"], "/private/deliverables")
        self.assertTrue(record["output_text_present"])
        self.assertEqual(record["metadata"], {})
        for forbidden_key in ("task_id", "environment", "command", "credential", "credentials", "output_text"):
            self.assertNotIn(forbidden_key, record)
        self.assertNotIn("secret-task-id", repr(record))
        self.assertNotIn("secret output text", repr(record))

        with self.assertRaises(TypeError):
            # Preserve the invalid runtime object at the execution record boundary.
            execution_record(cast(ExecutionResult, object()))

    @staticmethod
    def _git(root: Path, *arguments: str) -> CompletedProcess[bytes]:
        return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
