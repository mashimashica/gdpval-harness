# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from subprocess import CompletedProcess
from types import MappingProxyType
from typing import cast
from unittest.mock import Mock, patch

import eval_harness.experiments.profile as profile_module
from eval_harness.builders.base import BuilderInputBundle
from eval_harness.builders.inputs import load_builder_input_bundle
from eval_harness.experiments.base import ExperimentArm, ExperimentInputSpec, ExperimentProfile
from eval_harness.experiments.profile import load_experiment_inputs, load_experiment_profile


_HEAD = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret


class _DuplicateKeyMapping(Mapping[str, Path]):
    def __init__(self, root: Path) -> None:
        self.root = root

    def __getitem__(self, key: str) -> Path:
        if key != "input-a":
            raise KeyError(key)
        return self.root

    def __iter__(self) -> Iterator[str]:
        yield "input-a"
        yield "input-a"

    def __len__(self) -> int:
        return 2


class ExperimentProfileLoaderTests(unittest.TestCase):
    def _payload(
        self,
        *,
        input_values: list[dict[str, object]] | None = None,
        arms: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "profile_id": "profile-a",
            "benchmark": "benchmark-a",
            "inputs": input_values
            if input_values is not None
            else [
                {
                    "input_id": "input-a",
                    "input_type": "files",
                    "source_revision": None,
                    "revision_status": "unavailable",
                    "allowed_files": ["allowed.txt"],
                }
            ],
            "arms": arms if arms is not None else [{"arm_id": "arm-a", "builder_inputs": ["input-a"]}],
        }

    def _write_profile(self, root: Path, payload: dict[str, object]) -> tuple[Path, bytes]:
        source = root / "profile.json"
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        source.write_bytes(raw)
        return source, raw

    def _source(self, root: Path, name: str = "source") -> Path:
        source = root / name
        source.mkdir()
        (source / "allowed.txt").write_bytes(b"allowed")
        (source / "decoy.txt").write_bytes(b"must not enter")
        return source

    def _profile(self, *specs: ExperimentInputSpec) -> ExperimentProfile:
        if not specs:
            specs = (ExperimentInputSpec("input-a", "files", None, "unavailable", ("allowed.txt",)),)
        return ExperimentProfile(
            schema_version=1,
            profile_id="profile-a",
            benchmark="benchmark-a",
            inputs=specs,
            arms=tuple(ExperimentArm(f"arm-{index}", (spec.input_id,)) for index, spec in enumerate(specs)),
        )

    def test_loads_contract_objects_and_hashes_exact_profile_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, raw = self._write_profile(Path(temporary), self._payload())

            loaded = load_experiment_profile(source)

            self.assertEqual(loaded.source, source)
            self.assertEqual(loaded.sha256, hashlib.sha256(raw).hexdigest())
            self.assertIsInstance(loaded.profile, ExperimentProfile)
            self.assertEqual(loaded.profile.inputs[0].allowed_files, ("allowed.txt",))

    def test_strict_json_rejects_duplicates_unknown_missing_and_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = json.dumps(self._payload(), separators=(",", ":"))
            malformed = (
                '{"schema_version":1,"profile_id":"profile-a","benchmark":"benchmark-a",'
                '"inputs":[{"input_id":"input-a","input_type":"files","source_revision":null,'
                '"revision_status":"unavailable","allowed_files":["allowed.txt"],'
                '"allowed_files":["decoy.txt"]}],"arms":[{"arm_id":"arm-a",'
                '"builder_inputs":["input-a"]}]}'
            )
            cases = {
                "duplicate": malformed,
                "unknown": json.dumps(self._payload() | {"secret-member": "must not echo"}),
                "missing": json.dumps({key: value for key, value in self._payload().items() if key != "arms"}),
                "nan": valid.replace('"schema_version":1', '"schema_version":NaN'),
                "infinite": valid.replace('"schema_version":1', '"schema_version":1e999'),
            }
            for name, content in cases.items():
                with self.subTest(case=name):
                    path = root / f"{name}.json"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError) as error:
                        load_experiment_profile(path)
                    self.assertNotIn("must not echo", str(error.exception))

            nested_unknown = self._payload()
            nested_inputs_value = nested_unknown["inputs"]
            if not isinstance(nested_inputs_value, list) or not nested_inputs_value:
                raise AssertionError("expected a non-empty input list")
            first_input = nested_inputs_value[0]
            if not isinstance(first_input, dict) or any(not isinstance(key, str) for key in first_input):
                raise AssertionError("expected an input object with string keys")
            nested_unknown["inputs"] = [dict(first_input, extra="unknown")]
            path = root / "nested-unknown.json"
            path.write_text(json.dumps(nested_unknown), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_profile(path)

    def test_rejects_invalid_utf8_symlinks_and_oversized_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_utf8 = root / "invalid.json"
            invalid_utf8.write_bytes(b"{\xff}")
            with self.assertRaises(ValueError):
                load_experiment_profile(invalid_utf8)

            target, _ = self._write_profile(root, self._payload())
            link = root / "profile-link.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                load_experiment_profile(link)

            oversized = root / "oversized.json"
            oversized.write_bytes(b"{" + b"x" * (1024 * 1024) + b"}")
            self.assertGreater(oversized.stat().st_size, 1024 * 1024)
            with self.assertRaises(ValueError):
                load_experiment_profile(oversized)

    def test_source_bindings_require_exact_unique_input_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            profile = self._profile()
            for bindings in ({}, {"input-a": source, "extra": source}, _DuplicateKeyMapping(source)):
                with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                    load_experiment_inputs(profile, bindings)

            with self.assertRaises(TypeError):
                # Preserve the invalid runtime binding type for the loader boundary.
                load_experiment_inputs(profile, cast(Mapping[str, Path], [("input-a", source)]))

    def test_unavailable_input_uses_only_explicit_allowlist_and_returns_mapping_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source(Path(temporary))
            profile = self._profile()

            bundles = load_experiment_inputs(profile, {"input-a": source})

            self.assertIsInstance(bundles, MappingProxyType)
            bundle = bundles["input-a"]
            self.assertIsInstance(bundle, BuilderInputBundle)
            self.assertEqual(tuple(file.path for file in bundle.manifest.files), ("allowed.txt",))
            self.assertNotIn("decoy.txt", tuple(file.path for file in bundle.manifest.files))
            self.assertEqual(bundle.manifest.source_revision, None)
            self.assertEqual(bundle.manifest.revision_status, "unavailable")
            with self.assertRaises(TypeError):
                # Preserve the immutable mapping boundary at runtime.
                cast(MutableMapping[str, BuilderInputBundle], bundles)["input-b"] = bundle

    def test_available_revision_checks_exact_git_head_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            profile = self._profile(ExperimentInputSpec("input-a", "files", _HEAD, "available", ("allowed.txt",)))
            environment_before = os.environ.get("GIT_OPTIONAL_LOCKS")
            completed = [
                CompletedProcess([], 0, stdout=f"{_HEAD}\n", stderr=""),
                CompletedProcess([], 0, stdout=f"{_HEAD}\n", stderr=""),
                CompletedProcess([], 0, stdout=b"allowed", stderr=b""),
            ]

            with patch("eval_harness.experiments.profile.subprocess.run", side_effect=completed) as git_run:
                bundles = load_experiment_inputs(profile, {"input-a": source})

            self.assertEqual(git_run.call_count, 3)
            command = git_run.call_args_list[0].args[0]
            self.assertEqual(
                command,
                ["git", "--no-optional-locks", "-C", str(source), "rev-parse", "--verify", "HEAD"],
            )
            kwargs = git_run.call_args_list[0].kwargs
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "replace")
            self.assertFalse(kwargs["check"])
            self.assertTrue(kwargs["timeout"] > 0)
            self.assertTrue(kwargs["timeout"] < float("inf"))
            self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(os.environ.get("GIT_OPTIONAL_LOCKS"), environment_before)
            self.assertEqual(bundles["input-a"].manifest.source_revision, _HEAD)

            blob_command = git_run.call_args_list[2].args[0]
            self.assertEqual(
                blob_command,
                ["git", "--no-optional-locks", "-C", str(source), "cat-file", "blob", f"{_HEAD}:./allowed.txt"],
            )
            blob_kwargs = git_run.call_args_list[2].kwargs
            self.assertTrue(blob_kwargs["capture_output"])
            self.assertFalse(blob_kwargs["check"])
            self.assertTrue(blob_kwargs["timeout"] > 0)
            self.assertTrue(blob_kwargs["timeout"] < float("inf"))
            self.assertEqual(blob_kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")

    def test_commit_blob_errors_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source(Path(temporary))
            profile = self._profile(ExperimentInputSpec("input-a", "files", _HEAD, "available", ("allowed.txt",)))
            for bad_result in (
                CompletedProcess([], 1, stdout=b"allowed", stderr=b"failure"),
                CompletedProcess([], 0, stdout="allowed", stderr=""),
            ):
                with self.subTest(result=bad_result):
                    results = [
                        CompletedProcess([], 0, stdout=f"{_HEAD}\n", stderr=""),
                        CompletedProcess([], 0, stdout=f"{_HEAD}\n", stderr=""),
                        bad_result,
                    ]
                    with (
                        patch("eval_harness.experiments.profile.subprocess.run", side_effect=results) as git_run,
                        self.assertRaises(ValueError),
                    ):
                        load_experiment_inputs(profile, {"input-a": source})
                    self.assertEqual(git_run.call_count, 3)

    def test_available_bundle_matches_pinned_commit_and_rejects_selected_worktree_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "repo"
            source.mkdir()
            selected = source / "selected.txt"
            unrelated = source / "unrelated.txt"
            selected.write_bytes(b"committed selected")
            unrelated.write_bytes(b"committed unrelated")
            self._git(source, "init")
            self._git(source, "config", "user.email", "profile-loader@example.invalid")
            self._git(source, "config", "user.name", "Profile Loader")
            self._git(source, "add", "selected.txt", "unrelated.txt")
            self._git(source, "commit", "-m", "initial")
            revision = self._git(source, "rev-parse", "HEAD").stdout.decode("ascii").strip()
            clean_profile = self._profile(
                ExperimentInputSpec("input-a", "files", revision, "available", ("selected.txt",))
            )

            loaded = load_experiment_inputs(clean_profile, {"input-a": source})
            self.assertEqual(
                loaded["input-a"].manifest.files[0].sha256,
                hashlib.sha256(b"committed selected").hexdigest(),
            )

            selected.write_bytes(b"modified selected")
            with self.assertRaises(ValueError):
                load_experiment_inputs(clean_profile, {"input-a": source})
            selected.unlink()
            selected.write_bytes(b"committed selected")

            untracked = source / "untracked.txt"
            untracked.write_bytes(b"untracked selected")
            untracked_profile = self._profile(
                ExperimentInputSpec("input-a", "files", revision, "available", ("untracked.txt",))
            )
            with self.assertRaises(ValueError):
                load_experiment_inputs(untracked_profile, {"input-a": source})

            unrelated.write_bytes(b"modified unrelated")
            (source / "unrelated-untracked.txt").write_bytes(b"unrelated untracked")
            loaded = load_experiment_inputs(clean_profile, {"input-a": source})
            self.assertEqual(loaded["input-a"].manifest.files[0].path, "selected.txt")

    @staticmethod
    def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        )

    def test_available_revision_is_checked_again_after_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source(Path(temporary))
            profile = self._profile(ExperimentInputSpec("input-a", "files", _HEAD, "available", ("allowed.txt",)))
            results = [
                CompletedProcess([], 0, stdout=f"{_HEAD}\n", stderr=""),
                CompletedProcess([], 0, stdout=("f" * 40) + "\n", stderr=""),
            ]
            with (
                patch("eval_harness.experiments.profile.subprocess.run", side_effect=results) as git_run,
                patch.object(profile_module, "load_builder_input_bundle", wraps=load_builder_input_bundle) as loader,
                self.assertRaises(ValueError),
            ):
                load_experiment_inputs(profile, {"input-a": source})
            self.assertEqual(git_run.call_count, 2)
            self.assertEqual(loader.call_count, 1)

    def test_available_revision_rejects_mismatch_malformed_output_and_git_failure_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            profile = self._profile(ExperimentInputSpec("input-a", "files", _HEAD, "available", ("allowed.txt",)))
            for result in (
                CompletedProcess([], 0, stdout=("f" * 40) + "\n", stderr=""),
                CompletedProcess([], 0, stdout=("A" * 40) + "\n", stderr=""),
                CompletedProcess([], 0, stdout=f"{_HEAD}\nextra\n", stderr=""),
                CompletedProcess([], 1, stdout=f"{_HEAD}\n", stderr="failed"),
            ):
                with self.subTest(result=result):
                    loader = Mock(wraps=load_builder_input_bundle)
                    with (
                        patch("eval_harness.experiments.profile.subprocess.run", return_value=result) as git_run,
                        patch.object(profile_module, "load_builder_input_bundle", loader),
                        self.assertRaises(ValueError),
                    ):
                        load_experiment_inputs(profile, {"input-a": source})
                    self.assertEqual(git_run.call_count, 1)
                    self.assertEqual(loader.call_count, 0)

            malformed_bytes = CompletedProcess([], 0, stdout=b"\xff\n", stderr=b"\xfe")
            with (
                patch("eval_harness.experiments.profile.subprocess.run", return_value=malformed_bytes),
                self.assertRaises(ValueError),
            ):
                load_experiment_inputs(profile, {"input-a": source})

    def test_expected_bundle_hash_is_checked_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source(Path(temporary))
            baseline = load_builder_input_bundle(
                source,
                input_id="input-a",
                input_type="files",
                allowed_files=("allowed.txt",),
            )
            matching = self._profile(
                ExperimentInputSpec(
                    "input-a",
                    "files",
                    None,
                    "unavailable",
                    ("allowed.txt",),
                    baseline.manifest.bundle_sha256,
                )
            )
            loaded = load_experiment_inputs(matching, {"input-a": source})
            self.assertEqual(loaded["input-a"].manifest.bundle_sha256, baseline.manifest.bundle_sha256)

            mismatch = self._profile(
                ExperimentInputSpec("input-a", "files", None, "unavailable", ("allowed.txt",), "a" * 64)
            )
            with self.assertRaises(ValueError):
                load_experiment_inputs(mismatch, {"input-a": source})

    def test_all_inputs_are_loaded_and_validated_before_immutable_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._source(root, "first")
            second = self._source(root, "second")
            specs = (
                ExperimentInputSpec("input-a", "files", None, "unavailable", ("allowed.txt",)),
                ExperimentInputSpec("input-b", "files", None, "unavailable", ("allowed.txt",)),
            )
            profile = self._profile(*specs)
            real_loader = load_builder_input_bundle
            loader = Mock(side_effect=lambda source, **kwargs: real_loader(source, **kwargs))

            with patch.object(profile_module, "load_builder_input_bundle", loader):
                bundles = load_experiment_inputs(profile, {"input-a": first, "input-b": second})

            self.assertEqual(loader.call_count, 2)
            self.assertEqual(tuple(bundles), ("input-a", "input-b"))
            self.assertIsInstance(bundles, MappingProxyType)


if __name__ == "__main__":
    unittest.main()
