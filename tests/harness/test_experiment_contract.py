# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import gdpval_harness.experiments as experiments
import gdpval_harness.experiments.profile as profile_module
import gdpval_harness.experiments.runner as runner_module
from gdpval_harness.experiments import (
    ExperimentArm,
    ExperimentInputSpec,
    ExperimentProfile,
    ExperimentRunConfig,
    ExperimentRunSummary,
    LoadedExperimentProfile,
    base,
    load_experiment_inputs,
    load_experiment_profile,
    run_builder_experiment,
)


_DIGEST = "a" * 64
_VALID_CONTRACT_NAMES = (
    "ExperimentInputSpec",
    "ExperimentArm",
    "ExperimentProfile",
    "LoadedExperimentProfile",
    "ExperimentRunSummary",
    "ExperimentRunConfig",
)
_VALID_PUBLIC_NAMES = _VALID_CONTRACT_NAMES + (
    "load_experiment_profile",
    "load_experiment_inputs",
    "run_builder_experiment",
)


class ExperimentContractTests(unittest.TestCase):
    def input(self, input_id: str = "input-a", files: tuple[str, ...] = ("src/a.txt",)) -> ExperimentInputSpec:
        return ExperimentInputSpec(input_id, "files", None, "unavailable", files, _DIGEST)

    def arm(self, arm_id: str = "arm-a", builder_inputs: tuple[str, ...] = ("input-a",)) -> ExperimentArm:
        return ExperimentArm(arm_id, builder_inputs)

    def profile(
        self,
        inputs: tuple[ExperimentInputSpec, ...] = (),
        arms: tuple[ExperimentArm, ...] = (),
    ) -> ExperimentProfile:
        return ExperimentProfile(
            1,
            "profile-a",
            "benchmark-a",
            inputs or (self.input(),),
            arms or (self.arm(),),
        )

    def config(self, **overrides: object) -> ExperimentRunConfig:
        values: dict[str, object] = {
            "builder_executor": "builder-a",
            "application_executor": "application-a",
            "evaluator": "evaluator-a",
            "builder_model": None,
            "application_model": "model-a",
            "builder_timeout_seconds": 1.0,
            "application_timeout_seconds": 2.0,
            "builder_network_enabled": False,
            "application_network_enabled": True,
            "limit": 1,
            "order_seed": 0,
        }
        values.update(overrides)
        return ExperimentRunConfig(**values)  # type: ignore[arg-type]

    def test_public_shapes_exports_and_frozen_records_are_exact(self) -> None:
        self.assertEqual(tuple(base.__all__), _VALID_CONTRACT_NAMES)
        self.assertEqual(tuple(experiments.__all__), _VALID_PUBLIC_NAMES)
        self.assertIs(experiments.load_experiment_profile, profile_module.load_experiment_profile)
        self.assertIs(experiments.load_experiment_inputs, profile_module.load_experiment_inputs)
        self.assertIs(experiments.run_builder_experiment, runner_module.run_builder_experiment)
        self.assertIs(load_experiment_profile, profile_module.load_experiment_profile)
        self.assertIs(load_experiment_inputs, profile_module.load_experiment_inputs)
        self.assertIs(run_builder_experiment, runner_module.run_builder_experiment)
        expected_fields = {
            ExperimentInputSpec: (
                "input_id",
                "input_type",
                "source_revision",
                "revision_status",
                "allowed_files",
                "expected_bundle_sha256",
            ),
            ExperimentArm: ("arm_id", "builder_inputs"),
            ExperimentProfile: ("schema_version", "profile_id", "benchmark", "inputs", "arms"),
            LoadedExperimentProfile: ("profile", "source", "sha256"),
            ExperimentRunSummary: (
                "profile_id",
                "benchmark",
                "out_dir",
                "runtime_root",
                "status",
                "task_count",
                "arm_count",
                "completed_applications",
            ),
            ExperimentRunConfig: (
                "builder_executor",
                "application_executor",
                "evaluator",
                "builder_model",
                "application_model",
                "builder_timeout_seconds",
                "application_timeout_seconds",
                "builder_network_enabled",
                "application_network_enabled",
                "limit",
                "order_seed",
            ),
        }
        for record, fields in expected_fields.items():
            with self.subTest(record=record.__name__):
                self.assertEqual(tuple(record.__dataclass_fields__), fields)
                self.assertTrue(record.__dataclass_params__.frozen)

        input_spec = self.input(files=("z/file.txt", "a/file.txt"))
        self.assertEqual(input_spec.allowed_files, ("a/file.txt", "z/file.txt"))
        self.assertIsInstance(input_spec.allowed_files, tuple)
        with self.assertRaises(FrozenInstanceError):
            input_spec.input_id = "changed"  # type: ignore[misc]

        arm = self.arm(builder_inputs=["input-a"])  # type: ignore[arg-type]
        self.assertEqual(arm.builder_inputs, ("input-a",))
        profile = self.profile()
        self.assertIsInstance(profile.inputs, tuple)
        self.assertIsInstance(profile.arms, tuple)

    def test_input_revision_hash_and_identifier_invariants(self) -> None:
        self.assertEqual(
            self.input("input-1", files=("a.txt",)).source_revision,
            None,
        )
        available = ExperimentInputSpec("input-1", "dataset", "rev-1", "available", ("a.txt",))
        self.assertEqual(available.source_revision, "rev-1")

        invalid_identifiers = ("", "A", "-leading", "bad space", "bad/slash", "x" * 65, 1, True)
        for invalid in invalid_identifiers:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ExperimentInputSpec(invalid, "files", None, "unavailable", ("a.txt",))  # type: ignore[arg-type]
            with self.subTest(profile_id=invalid), self.assertRaises(ValueError):
                ExperimentProfile(1, invalid, "benchmark-a", (self.input(),), (self.arm(),))  # type: ignore[arg-type]
            with self.subTest(arm_id=invalid), self.assertRaises(ValueError):
                ExperimentArm(invalid, ("input-a",))  # type: ignore[arg-type]

        for invalid in ("", "A", "bad space", "x" * 65, 1, True):
            with self.subTest(benchmark=invalid), self.assertRaises(ValueError):
                ExperimentProfile(1, "profile-a", invalid, (self.input(),), (self.arm(),))  # type: ignore[arg-type]
        for invalid in ("", "A", "bad space", "x" * 65, 1, True):
            with self.subTest(builder_input=invalid), self.assertRaises((TypeError, ValueError)):
                ExperimentArm("arm-a", (invalid,))  # type: ignore[arg-type]

        for schema_version in (0, 2, -1, True, 1.0, "1"):
            with self.subTest(schema_version=schema_version), self.assertRaises((TypeError, ValueError)):
                ExperimentProfile(
                    schema_version, "profile-a", "benchmark-a", (self.input(),), (self.arm(),)
                )  # type: ignore[arg-type]

        for source_revision, revision_status in (
            (None, "available"),
            ("", "available"),
            (" ", "available"),
            ("rev", "unavailable"),
            (None, "unknown"),
            ("rev", "AVAILABLE"),
        ):
            with self.subTest(source_revision=source_revision, revision_status=revision_status), self.assertRaises(
                ValueError
            ):
                ExperimentInputSpec("input-a", "files", source_revision, revision_status, ("a.txt",))

        for digest in ("A" * 64, "g" * 64, "a" * 63, "a" * 65, 1):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                ExperimentInputSpec(
                    "input-a", "files", None, "unavailable", ("a.txt",), digest
                )  # type: ignore[arg-type]

    def test_allowed_file_paths_reject_unsafe_forms_and_collisions(self) -> None:
        invalid_paths = (
            "",
            "../escape.txt",
            "a/../escape.txt",
            "/absolute.txt",
            "//absolute.txt",
            r"folder\file.txt",
            "folder//file.txt",
            "folder/",
            "./file.txt",
            "folder/./file.txt",
            "a\x00b.txt",
            "e\u0301.txt",
            "C:/drive.txt",
        )
        for path in invalid_paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.input(files=(path,))

        with self.assertRaises(ValueError):
            self.input(files=())
        with self.assertRaises(TypeError):
            self.input(files=("a.txt", 3))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ExperimentInputSpec("input-a", "files", None, "unavailable", "a.txt")  # type: ignore[arg-type]

        collision_cases = (
            ("A.txt", "a.txt"),
            ("e\u0301.txt", "é.txt"),
            ("A/x", "a/y"),
            ("dir/A", "DIR/a"),
            ("a", "a/b"),
            ("A", "a/b"),
        )
        for files in collision_cases:
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.input(files=files)

    def test_profile_relations_preserve_order_and_reject_invalid_graphs(self) -> None:
        first = self.input("input-a", ("a.txt",))
        second = self.input("input-b", ("b.txt",))
        arm_a = self.arm("arm-a", ("input-b",))
        arm_b = self.arm("arm-b", ("input-a",))
        profile = self.profile(inputs=(first, second), arms=(arm_a, arm_b))
        self.assertEqual(profile.inputs, (first, second))
        self.assertEqual(profile.arms, (arm_a, arm_b))
        self.assertEqual(profile.arms[0].builder_inputs, ("input-b",))

        invalid_profiles = (
            ((first, first), (self.arm(),)),
            ((first,), (self.arm("arm-a", ("unknown",)),)),
            ((first, second), (self.arm("arm-a", ("input-a",)),)),
            ((first,), (self.arm("arm-a"), self.arm("arm-a"))),
            ((), (self.arm(),)),
            ((first,), ()),
        )
        for inputs, arms in invalid_profiles:
            with self.subTest(inputs=inputs, arms=arms), self.assertRaises(ValueError):
                ExperimentProfile(1, "profile-a", "benchmark-a", inputs, arms)
        with self.assertRaises(ValueError):
            ExperimentArm("arm-a", ("input-a", "input-a"))

        with self.assertRaises(TypeError):
            ExperimentProfile(1, "profile-a", "benchmark-a", (object(),), (self.arm(),))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ExperimentProfile(1, "profile-a", "benchmark-a", (first,), (object(),))  # type: ignore[arg-type]

    def test_loaded_profile_and_run_summary_normalize_paths_and_validate_counts(self) -> None:
        profile = self.profile()
        loaded = LoadedExperimentProfile(profile, "profiles/profile.json", _DIGEST)
        self.assertEqual(loaded.source, Path("profiles/profile.json"))
        with self.assertRaises(FrozenInstanceError):
            loaded.source = Path("changed")  # type: ignore[misc]
        with self.assertRaises(TypeError):
            LoadedExperimentProfile(object(), Path("profile.json"), _DIGEST)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LoadedExperimentProfile(profile, Path("profile.json"), "A" * 64)

        summary = ExperimentRunSummary("profile-a", "benchmark-a", "out", "runtime", "completed", 2, 3, 6)
        self.assertEqual(summary.out_dir, Path("out"))
        self.assertEqual(summary.runtime_root, Path("runtime"))
        for status in ("completed", "failed", "interrupted"):
            self.assertEqual(
                ExperimentRunSummary("profile-a", "benchmark-a", Path("out"), Path("runtime"), status, 0, 0, 0).status,
                status,
            )
        for status in ("", "pending", "COMPLETED"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                ExperimentRunSummary("profile-a", "benchmark-a", Path("out"), Path("runtime"), status, 0, 0, 0)
        for counts in ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 1, 2), (True, 1, 0)):
            with self.subTest(counts=counts), self.assertRaises((TypeError, ValueError)):
                ExperimentRunSummary(
                    "profile-a", "benchmark-a", Path("out"), Path("runtime"), "failed", *counts
                )  # type: ignore[arg-type]

    def test_run_config_is_shared_and_rejects_invalid_values(self) -> None:
        config = self.config()
        self.assertEqual(config.builder_timeout_seconds, 1.0)
        self.assertEqual(config.application_network_enabled, True)
        with self.assertRaises(FrozenInstanceError):
            config.limit = 2  # type: ignore[misc]

        for field in ("builder_executor", "application_executor", "evaluator"):
            for value in ("", "UPPER", "bad value", "x" * 65, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.config(**{field: value})
        for field in ("builder_model", "application_model"):
            for value in ("", " ", 3, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.config(**{field: value})
        for field in ("builder_timeout_seconds", "application_timeout_seconds"):
            for value in (0, -1, math.nan, math.inf, -math.inf, True, "1"):
                with self.subTest(field=field, value=value), self.assertRaises((TypeError, ValueError)):
                    self.config(**{field: value})
        for field in ("builder_network_enabled", "application_network_enabled"):
            with self.subTest(field=field), self.assertRaises(TypeError):
                self.config(**{field: 1})
        for value in (0, -1, True, 1.0, "1"):
            with self.subTest(limit=value), self.assertRaises((TypeError, ValueError)):
                self.config(limit=value)
        for value in (-1, 2**63, True, 1.0, "1"):
            with self.subTest(order_seed=value), self.assertRaises((TypeError, ValueError)):
                self.config(order_seed=value)
        self.assertEqual(self.config(order_seed=2**63 - 1).order_seed, 2**63 - 1)


if __name__ == "__main__":
    unittest.main()
