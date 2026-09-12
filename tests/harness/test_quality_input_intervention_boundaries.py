# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence, cast
from unittest.mock import patch

import eval_harness.builders.inputs as inputs_module
import eval_harness.experiments.profile as profile_module
import eval_harness.experiments.runner as runner_module
import eval_harness.interventions.agent_skill as skill_module
import eval_harness.interventions.base as intervention_base
import eval_harness.interventions.files as files_module
import eval_harness.interventions.prompt_overlay as overlay_module
from eval_harness.benchmarks.base import Benchmark, BenchmarkTask
from eval_harness.builders.base import (
    Builder,
    BuilderInputBundle,
    BuilderInputManifest,
    BuilderPreflightResult,
    BuildFailurePhase,
    BuildRequest,
    BuildResult,
    BuildStatus,
)
from eval_harness.evaluators.base import (
    EvaluationPlan,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    EvaluatorPreflightResult,
    EvaluatorType,
)
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from eval_harness.experiments.base import (
    ExperimentArm,
    ExperimentInputSpec,
    ExperimentProfile,
    ExperimentRunConfig,
    LoadedExperimentProfile,
)
from eval_harness.interventions.base import (
    ApplicationMapping,
    InterventionApplication,
    InterventionBundle,
    InterventionFile,
    InterventionManifest,
    InterventionType,
    compute_bundle_sha256,
    file_evidence,
)
from eval_harness.provenance import canonical_json_sha256
from eval_harness.reasoning import ReasoningEffortOption
from eval_harness.runner import RunSummary


def _file(path: str = "input.txt", content: bytes = b"input") -> InterventionFile:
    return file_evidence(path, content)


def _manifest(
    *,
    intervention_id: str = "intervention",
    intervention_type: InterventionType = InterventionType.FILES,
    content: bytes = b"input",
    application: ApplicationMapping | None = None,
) -> InterventionManifest:
    entry = _file(content=content)
    return InterventionManifest(
        intervention_id=intervention_id,
        intervention_type=intervention_type,
        source_revision=None,
        revision_status="unavailable",
        files=(entry,),
        bundle_sha256=compute_bundle_sha256(((entry.path, content),)),
        application=application or ApplicationMapping("workspace-files", "."),
    )


def _input_bundle(root: Path, *, input_id: str = "input-a") -> BuilderInputBundle:
    content = b"guide"
    entry = _file("guide.txt", content)
    manifest = BuilderInputManifest(
        input_id=input_id,
        input_type="reference",
        source_revision=None,
        revision_status="unavailable",
        files=(entry,),
        bundle_sha256=compute_bundle_sha256(((entry.path, content),)),
    )
    return BuilderInputBundle(root, manifest)


def _profile(root: Path, *, revision_status: str = "unavailable") -> LoadedExperimentProfile:
    source = root / "profile.json"
    source.write_text("{}\n", encoding="utf-8")
    revision = None if revision_status == "unavailable" else "a" * 40
    input_spec = ExperimentInputSpec("input-a", "reference", revision, revision_status, ("guide.txt",))
    experiment = ExperimentProfile(
        1,
        "profile-a",
        "benchmark-a",
        (input_spec,),
        (ExperimentArm("arm-a", ("input-a",)),),
    )
    return LoadedExperimentProfile(experiment, source, "b" * 64)


def _skill(root: Path, *, name: str = "demo-skill") -> Path:
    skill = root / name
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: deterministic boundary skill\n"
        "metadata:\n"
        "  owner: tests\n"
        "---\n\n"
        "Follow the skill.\n",
        encoding="utf-8",
    )
    (skill / "references" / "guide.md").write_bytes(b"guide")
    return skill


class _Benchmark(Benchmark):
    name = "benchmark-a"
    revision: str | None = None

    def __init__(self, tasks: Sequence[BenchmarkTask]) -> None:
        self.tasks = tuple(tasks)
        self.prepared = False

    def is_prepared(self) -> bool:
        return self.prepared

    def prepare(self) -> None:
        self.prepared = True

    def load_tasks(self, limit: int) -> Sequence[BenchmarkTask]:
        return self.tasks[:limit]

    def materialize(self, task: BenchmarkTask, workspace: Path) -> Sequence[str]:
        del task, workspace
        return ()


class _Evaluator(Evaluator):
    name = "evaluator-a"
    evaluator_type = EvaluatorType.BENCHMARK_NATIVE

    def validate_plan(self, plan: EvaluationPlan) -> None:
        del plan

    def preflight(self, run_dir: Path | None = None) -> EvaluatorPreflightResult:
        del run_dir
        return EvaluatorPreflightResult(self.name, self.evaluator_type, True)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult(request.task_id, EvaluationStatus.COMPLETED)


class _Executor(Executor):
    name = "executor-a"
    runtime = "test"
    invocation_mode = "deterministic"
    network_access_enabled: bool = False
    reasoning_effort: ReasoningEffortOption = None

    def preflight(self) -> PreflightResult:
        return PreflightResult(self.name, True, version="1", auth_mode="local")

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            runtime="test",
            task_id=request.task.task_id,
            executor=self.name,
            executor_version="1",
            invocation_mode=self.invocation_mode,
            auth_mode="local",
            workspace=request.workspace,
            deliverables_dir=request.deliverables_dir,
            status=ExecutionStatus.COMPLETED,
            started_at="start",
            finished_at="finish",
            exit_code=0,
            available_outputs=frozenset(),
            failure=None,
        )


class _Builder(Builder):
    name = "builder-a"

    def preflight(self) -> BuilderPreflightResult:
        return BuilderPreflightResult(self.name, True, builder_executor="executor-a")

    def build(self, request: BuildRequest) -> BuildResult:
        return BuildResult(
            request.build_run_id,
            request.task.task_id,
            self.name,
            BuildStatus.FAILED,
            tuple(item.manifest for item in request.inputs),
            failure_phase=BuildFailurePhase.PREFLIGHT,
        )


class _PathOnly:
    def __init__(self, path: str) -> None:
        self.path = path


class _ExplodingAttribute:
    def __getattribute__(self, name: str) -> object:
        raise RuntimeError(f"attribute access failed: {name}")


class InterventionBoundaryTests(unittest.TestCase):
    def test_base_contracts_reject_invalid_records_and_preserve_hashes(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        with self.assertRaises(ValueError):
            InterventionFile("", 1, digest)
        with self.assertRaises(ValueError):
            InterventionFile("x", -1, digest)
        with self.assertRaises(ValueError):
            InterventionFile("x", 1, "A" * 64)
        with self.assertRaises(ValueError):
            ApplicationMapping("", ".")
        with self.assertRaises(ValueError):
            ApplicationMapping("files", "")

        entry = _file()
        application = ApplicationMapping("files", ".")
        manifest = _manifest(application=application)
        self.assertEqual(
            manifest.manifest_sha256, hashlib.sha256(intervention_base.canonical_manifest_bytes(manifest)).hexdigest()
        )
        with self.assertRaises(ValueError):
            InterventionManifest(
                "", InterventionType.FILES, None, "unavailable", (entry,), manifest.bundle_sha256, application
            )
        with self.assertRaises(TypeError):
            InterventionManifest(
                "id",
                InterventionType.FILES,
                None,
                "unavailable",
                cast(Sequence[InterventionFile], (object(),)),
                manifest.bundle_sha256,
                application,
            )
        with self.assertRaises(TypeError):
            InterventionManifest(
                "id",
                InterventionType.FILES,
                None,
                "unavailable",
                (entry,),
                manifest.bundle_sha256,
                cast(ApplicationMapping, object()),
            )
        with self.assertRaises(ValueError):
            replace(manifest, manifest_sha256="0" * 64)

        task = TaskSpec("task-a", "prompt")
        with self.assertRaises(ValueError):
            InterventionApplication("", task, (), manifest.bundle_sha256, manifest.manifest_sha256 or "", application)
        with self.assertRaises(TypeError):
            InterventionApplication(
                "run",
                cast(TaskSpec, object()),
                (),
                manifest.bundle_sha256,
                manifest.manifest_sha256 or "",
                application,
            )
        with self.assertRaises(TypeError):
            InterventionApplication(
                "run",
                task,
                cast(Sequence[InterventionFile], (object(),)),
                manifest.bundle_sha256,
                manifest.manifest_sha256 or "",
                application,
            )
        with self.assertRaises(TypeError):
            InterventionApplication(
                "run",
                task,
                (),
                manifest.bundle_sha256,
                manifest.manifest_sha256 or "",
                cast(ApplicationMapping, object()),
            )

        self.assertEqual(intervention_base.revision_fields(None, applicable=False), (None, "not-applicable"))
        self.assertEqual(intervention_base.revision_fields(None, applicable=True), (None, "unavailable"))
        with self.assertRaises(ValueError):
            intervention_base.revision_fields("", applicable=True)

    def test_base_destination_and_source_boundaries_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "file").write_text("sentinel", encoding="utf-8")
            with self.assertRaises(ValueError):
                intervention_base.ensure_destination_parents(workspace, ("file/nested.txt",))
            outside = root / "outside"
            outside.mkdir()
            (workspace / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                intervention_base.ensure_destination_parents(workspace, ("link/file.txt",))

            bundle = InterventionBundle(root, _manifest())
            with self.assertRaises(ValueError):
                intervention_base.ensure_source_output_separation(bundle, root)
            with self.assertRaises(ValueError):
                intervention_base.ensure_source_output_separation(bundle, root / "child")
            source_link = root / "source-link"
            source_link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                intervention_base.ensure_source_output_separation(
                    InterventionBundle(source_link, _manifest()), workspace
                )
            with self.assertRaises(ValueError):
                intervention_base.ensure_source_output_separation(cast(InterventionBundle, object()), workspace)

    def test_prompt_overlay_validates_text_source_and_rehashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "overlay.txt"
            source.write_text("reviewed instruction", encoding="utf-8")
            task = TaskSpec("task-a", "original")
            with self.assertRaises(TypeError):
                overlay_module.apply_prompt_overlay(cast(TaskSpec, object()), "text")
            with self.assertRaises(ValueError):
                overlay_module.apply_prompt_overlay(task, " \t")
            applied = overlay_module.apply_prompt_overlay(task, "reviewed")
            self.assertEqual(applied.task_id, task.task_id)
            self.assertTrue(applied.prompt.endswith("\n\noriginal"))

            intervention = overlay_module.PromptOverlayIntervention(source, source_revision="rev-1")
            preflight = intervention.preflight()
            self.assertTrue(preflight.ok, preflight.details)
            application = intervention.apply(task, root / "unused", application_run_id="run-a")
            self.assertIn("reviewed instruction", application.task.prompt)
            self.assertEqual(application.task.task_id, task.task_id)
            with self.assertRaises(TypeError):
                intervention.validate_task(cast(TaskSpec, object()))

            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                intervention.apply(task, root / "unused", application_run_id="run-b")

            directory = root / "directory"
            directory.mkdir()
            self.assertFalse(overlay_module.PromptOverlayIntervention(directory).preflight().ok)
            empty = root / "empty.txt"
            empty.write_text(" \n", encoding="utf-8")
            self.assertFalse(overlay_module.PromptOverlayIntervention(empty).preflight().ok)
            invalid = root / "invalid.txt"
            invalid.write_bytes(b"\xff")
            self.assertFalse(overlay_module.PromptOverlayIntervention(invalid).preflight().ok)
            with self.assertRaisesRegex(RuntimeError, "successful intervention preflight"):
                overlay_module.PromptOverlayIntervention(source).apply(task, root, application_run_id="run-c")

            oversized = root / "oversized.txt"
            oversized.write_bytes(b"x")
            with patch.object(Path, "read_bytes", return_value=b"x" * (overlay_module.MAX_PROMPT_OVERLAY_BYTES + 1)):
                self.assertFalse(overlay_module.PromptOverlayIntervention(oversized).preflight().ok)

    def test_files_intervention_scan_and_copy_roll_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "nested").mkdir(parents=True)
            (source / "nested" / "value.txt").write_bytes(b"value")
            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            self.assertFalse(files_module.FilesIntervention(source_link).preflight().ok)
            with self.assertRaises(ValueError):
                files_module._read_regular_file(source_link)
            with self.assertRaises(ValueError):
                files_module._read_regular_file(source / "nested")

            workspace = root / "workspace"
            workspace.mkdir()
            files_module._validate_destination_collisions(workspace, ("missing/value.txt",))
            (workspace / "MISSING").mkdir()
            with self.assertRaisesRegex(ValueError, "casefold"):
                files_module._validate_destination_collisions(workspace, ("missing/value.txt",))
            (workspace / "MISSING").rmdir()

            (workspace / "file").write_text("sentinel", encoding="utf-8")
            with self.assertRaises(ValueError):
                files_module._mkdir_missing_parents(workspace, ("file/value.txt",), [])

            destination = root / "copy.txt"
            with (
                patch.object(os, "fsync", side_effect=OSError("fsync failed")),
                self.assertRaises(OSError),
            ):
                files_module._copy_exclusive(destination, b"copy")
            self.assertFalse(destination.exists())

            intervention = files_module.FilesIntervention(source)
            self.assertTrue(intervention.preflight().ok)
            workspace = root / "apply-workspace"
            workspace.mkdir()

            def write_wrong(path: Path, content: bytes) -> None:
                path.write_bytes(content + b"wrong")

            with (
                patch.object(files_module, "_copy_exclusive", side_effect=write_wrong),
                self.assertRaisesRegex(RuntimeError, "verification"),
            ):
                intervention.apply(TaskSpec("task-a", "prompt"), workspace, application_run_id="run")
            self.assertEqual(tuple(workspace.iterdir()), ())

            unreadable = root / "unreadable"
            unreadable.mkdir()
            with patch.object(Path, "iterdir", side_effect=OSError("directory unavailable")):
                self.assertFalse(files_module.FilesIntervention(unreadable).preflight().ok)

    def test_agent_skill_loader_and_application_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            with self.assertRaisesRegex(ValueError, "invalid Agent Skill bundle"):
                skill_module.load_agent_skill_bundle(missing)
            regular = root / "regular"
            regular.write_text("file", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid Agent Skill bundle"):
                skill_module.load_agent_skill_bundle(regular)

            no_skill = root / "no-skill"
            no_skill.mkdir()
            with self.assertRaisesRegex(ValueError, "invalid Agent Skill bundle"):
                skill_module.load_agent_skill_bundle(no_skill)

            invalid_frontmatter = (
                b"not-frontmatter\n",
                b"---\nname: demo\n",
                b"---\n[broken\n---\n",
                b"---\n- item\n---\n",
                b"---\n123: value\n---\n",
                b"---\nunknown: value\n---\n",
                b"\xff",
            )
            for index, content in enumerate(invalid_frontmatter):
                skill = root / f"invalid-{index}"
                skill.mkdir()
                (skill / "SKILL.md").write_bytes(content)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    skill_module.load_agent_skill_bundle(skill)

            skill = _skill(root)
            bundle = skill_module.load_agent_skill_bundle(skill)
            intervention = skill_module.AgentSkillIntervention(bundle)
            with self.assertRaisesRegex(RuntimeError, "successful Agent Skill preflight"):
                intervention.apply(TaskSpec("task-a", "prompt"), root / "workspace", application_run_id="run")
            self.assertTrue(intervention.preflight().ok)
            workspace = root / "workspace"
            workspace.mkdir()
            application = intervention.apply(TaskSpec("task-a", "prompt"), workspace, application_run_id="run")
            self.assertIn(".gdpval/interventions/demo-skill/SKILL.md", application.task.prompt)
            self.assertTrue((workspace / ".gdpval/interventions/demo-skill/SKILL.md").is_file())

            (skill / "references" / "guide.md").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                intervention.apply(TaskSpec("task-a", "prompt"), root / "other-workspace", application_run_id="run-2")

            with self.assertRaises(ValueError):
                skill_module.AgentSkillIntervention(bundle, source_reference=" \x00 ")
            with self.assertRaises(TypeError):
                intervention.validate_task(cast(TaskSpec, object()))

            with self.assertRaises(ValueError):
                skill_module._skill_name_from_manifest(
                    replace(bundle.manifest, application=ApplicationMapping("skill", "bad"))
                )
            with self.assertRaises(ValueError):
                skill_module._skill_name_from_manifest(
                    replace(
                        bundle.manifest, application=ApplicationMapping("skill", ".gdpval/interventions/BAD/SKILL.md")
                    )
                )
            with self.assertRaises(RuntimeError):
                skill_module._source_entries_from_manifest(bundle.manifest, skill)

            collision_workspace = root / "collision-workspace"
            collision_workspace.mkdir()
            (collision_workspace / ".GDPVAL").mkdir()
            with self.assertRaisesRegex(ValueError, "collision"):
                skill_module._validate_destination_collisions(
                    collision_workspace, (".gdpval/interventions/demo-skill/SKILL.md",)
                )

            parent_file = root / "parent-file"
            parent_file.write_text("file", encoding="utf-8")
            with self.assertRaises(ValueError):
                skill_module._mkdir_missing_parents(parent_file.parent, ("parent-file/nested.txt",), [])

            destination = root / "copy.txt"
            with (
                patch.object(os, "fsync", side_effect=OSError("fsync failed")),
                self.assertRaises(OSError),
            ):
                skill_module._copy_exclusive(destination, b"copy")
            self.assertFalse(destination.exists())

    def test_profile_parser_and_git_boundaries_reject_malformed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "profile.json"
            source.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                profile_module._canonical_profile_path(cast(Path | str, object()))
            with self.assertRaises(ValueError):
                profile_module._require_object([])
            with self.assertRaises(ValueError):
                profile_module._require_exact_keys({}, frozenset({"required"}))
            with self.assertRaises(profile_module._DuplicateObjectKey):
                profile_module._reject_duplicate_object_keys([("key", 1), ("key", 2)])
            with self.assertRaises(profile_module._NonFiniteNumber):
                profile_module._parse_finite_float("inf")
            with self.assertRaises(profile_module._NonFiniteNumber):
                profile_module._reject_non_finite_number("NaN")
            with self.assertRaises(ValueError):
                profile_module._decode_profile(b"\xff")

            loaded = _profile(root)
            bundle_root = root / "input"
            bundle_root.mkdir()
            (bundle_root / "guide.txt").write_bytes(b"guide")
            bundle = _input_bundle(bundle_root)
            with self.assertRaises(TypeError):
                profile_module._validate_source_bindings(loaded.profile, cast(Mapping[str, Path | str], object()))
            with self.assertRaises(ValueError):
                profile_module._validate_source_bindings(loaded.profile, {"wrong": bundle_root})
            with self.assertRaises(ValueError):
                profile_module._validate_loaded_bundle(object(), loaded.profile.inputs[0])
            wrong_type = replace(loaded.profile.inputs[0], input_type="wrong")
            with self.assertRaises(ValueError):
                profile_module._validate_loaded_bundle(bundle, wrong_type)
            wrong_revision = replace(loaded.profile.inputs[0], source_revision="rev", revision_status="available")
            with self.assertRaises(ValueError):
                profile_module._validate_loaded_bundle(bundle, wrong_revision)
            wrong_hash = replace(loaded.profile.inputs[0], expected_bundle_sha256="0" * 64)
            with self.assertRaises(ValueError):
                profile_module._validate_loaded_bundle(bundle, wrong_hash)

            revision = "a" * 40
            malformed_outputs: tuple[object, ...] = (
                subprocess.CompletedProcess[bytes](args=["git"], returncode=1, stdout=b""),
                subprocess.CompletedProcess[bytes](args=["git"], returncode=0, stdout=b"wrong\nextra\n"),
                subprocess.CompletedProcess[bytes](args=["git"], returncode=0, stdout=cast(bytes, object())),
            )
            for result in malformed_outputs:
                with (
                    patch.object(subprocess, "run", return_value=result),
                    self.assertRaises(ValueError),
                ):
                    profile_module._git_head(bundle_root, revision)
            with (
                patch.object(subprocess, "run", side_effect=OSError("git unavailable")),
                self.assertRaises(ValueError),
            ):
                profile_module._git_head(bundle_root, revision)

            bad_blob = subprocess.CompletedProcess[bytes](args=["git"], returncode=0, stdout=cast(bytes, object()))
            with patch.object(subprocess, "run", return_value=bad_blob), self.assertRaises(ValueError):
                profile_module._git_commit_blob(bundle_root, revision, "guide.txt")
            with (
                patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 1)),
                self.assertRaises(ValueError),
            ):
                profile_module._git_commit_blob(bundle_root, revision, "guide.txt")

            valid_payload = {
                "schema_version": 1,
                "profile_id": "profile-a",
                "benchmark": "benchmark-a",
                "inputs": [
                    {
                        "input_id": "input-a",
                        "input_type": "reference",
                        "source_revision": None,
                        "revision_status": "unavailable",
                        "allowed_files": ["guide.txt"],
                    }
                ],
                "arms": [{"arm_id": "arm-a", "builder_inputs": ["input-a"]}],
            }
            profile_path = root / "valid-profile.json"
            profile_path.write_text(json.dumps(valid_payload), encoding="utf-8")
            self.assertEqual(profile_module.load_experiment_profile(profile_path).profile.profile_id, "profile-a")

    def test_builder_input_boundaries_reject_invalid_paths_and_roll_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "value.txt").write_bytes(b"value")
            with self.assertRaises(TypeError):
                inputs_module._validate_relative_path(object(), label="path")
            with self.assertRaises(ValueError):
                inputs_module._validate_relative_path("\ud800", label="path")
            for allowed in (cast(Sequence[str], "value.txt"), cast(Sequence[str], object()), ()):
                with self.subTest(allowed=allowed), self.assertRaises((TypeError, ValueError)):
                    inputs_module._validate_allowlist(allowed)

            with self.assertRaises(ValueError):
                inputs_module._canonical_existing_directory(root / "missing", label="directory")
            file_path = root / "file"
            file_path.write_text("file", encoding="utf-8")
            with self.assertRaises(ValueError):
                inputs_module._canonical_existing_directory(file_path, label="directory")
            link = root / "link"
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                inputs_module._canonical_existing_directory(link, label="directory")
            with self.assertRaises(ValueError):
                inputs_module._load_source_root(cast(Path | str, object()))

            entry = _file("value.txt", b"value")
            with self.assertRaises(ValueError):
                inputs_module._validate_manifest_files(())
            with self.assertRaises(TypeError):
                inputs_module._validate_manifest_files(cast(Sequence[InterventionFile], (object(),)))
            unsorted = (entry, _file("a.txt", b"a"))
            with self.assertRaises(ValueError):
                inputs_module._validate_manifest_files(unsorted)
            prefix_files = (_file("dir", b"dir"), _file("dir/file.txt", b"file"))
            with self.assertRaises(ValueError):
                inputs_module._validate_manifest_files(prefix_files)

            bundle = inputs_module.load_builder_input_bundle(
                source,
                input_id="input-a",
                input_type="reference",
                allowed_files=("value.txt",),
            )
            forged_manifest = object.__new__(BuilderInputManifest)
            for field_name in (
                "input_id",
                "input_type",
                "source_revision",
                "revision_status",
                "files",
            ):
                object.__setattr__(forged_manifest, field_name, getattr(bundle.manifest, field_name))
            object.__setattr__(forged_manifest, "bundle_sha256", bundle.manifest.bundle_sha256)
            object.__setattr__(forged_manifest, "manifest_sha256", "0" * 64)
            with self.assertRaises(ValueError):
                inputs_module._validate_manifest_hash(forged_manifest)
            stale_manifest = object.__new__(BuilderInputManifest)
            for field_name in (
                "input_id",
                "input_type",
                "source_revision",
                "revision_status",
                "files",
                "manifest_sha256",
            ):
                object.__setattr__(stale_manifest, field_name, getattr(bundle.manifest, field_name))
            object.__setattr__(stale_manifest, "bundle_sha256", "0" * 64)
            with self.assertRaises(RuntimeError):
                inputs_module._snapshot_manifest_files(source, stale_manifest)

            workspace = root / "workspace"
            workspace.mkdir()
            namespace = workspace / "reference_files"
            namespace.mkdir()
            with self.assertRaises(FileExistsError):
                inputs_module.stage_builder_inputs((bundle,), workspace)
            empty_workspace = root / "empty-workspace"
            empty_workspace.mkdir()
            self.assertEqual(inputs_module.stage_builder_inputs((), empty_workspace), ())
            with self.assertRaises(TypeError):
                inputs_module.stage_builder_inputs(cast(Sequence[BuilderInputBundle], "bundle"), empty_workspace)

            staged_workspace = root / "staged-workspace"
            staged_workspace.mkdir()
            staged = inputs_module.stage_builder_inputs((bundle,), staged_workspace)
            with self.assertRaises(ValueError):
                inputs_module.verify_staged_builder_inputs(staged, root / "missing-workspace")
            with self.assertRaises(TypeError):
                inputs_module.verify_staged_builder_inputs(
                    cast(Sequence[inputs_module.StagedBuilderInput], object()), staged_workspace
                )
            target_dir = staged_workspace / "reference_files" / "builder-inputs" / "input-001"
            target_dir.chmod(0o755)
            (target_dir / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(ValueError):
                inputs_module.verify_staged_builder_inputs(staged, staged_workspace)

    def test_builder_input_failure_isolation_covers_source_and_workspace_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            value = source / "value.txt"
            value.write_bytes(b"value")
            directory = root / "directory"
            directory.mkdir()
            resolved_target = root / "resolved-target"
            resolved_target.mkdir()

            with self.assertRaises(ValueError):
                inputs_module.StagedBuilderInput("", "target", ())
            with self.assertRaises(ValueError):
                inputs_module.StagedBuilderInput("input", "", ())
            with self.assertRaises(TypeError):
                inputs_module.StagedBuilderInput("input", "target", cast(Sequence[InterventionFile], object()))
            with self.assertRaises(TypeError):
                inputs_module.StagedBuilderInput("input", "target", (cast(InterventionFile, object()),))

            invalid_paths = ("", "a\x00b", "a\\b", "/absolute", "C:/absolute", "./", "..", "a/../b", "e\u0301.txt")
            for invalid_path in invalid_paths:
                with self.subTest(invalid_path=invalid_path), self.assertRaises(ValueError):
                    inputs_module._validate_relative_path(invalid_path, label="path")

            with patch.object(Path, "resolve", side_effect=OSError("resolve failed")), self.assertRaises(ValueError):
                inputs_module._canonical_existing_directory(directory, label="directory")
            with patch.object(Path, "resolve", return_value=value), self.assertRaises(ValueError):
                inputs_module._canonical_existing_directory(directory, label="directory")
            with patch.object(Path, "resolve", return_value=resolved_target), self.assertRaises(ValueError):
                inputs_module._canonical_existing_directory(directory, label="directory")

            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                inputs_module._load_source_root(source_link)
            with (
                patch.object(Path, "resolve", side_effect=OSError("source resolve failed")),
                self.assertRaises(ValueError),
            ):
                inputs_module._load_source_root(source)
            with patch.object(Path, "resolve", return_value=value), self.assertRaises(ValueError):
                inputs_module._load_source_root(source)

            missing = root / "missing"
            with self.assertRaises(ValueError):
                inputs_module._regular_file_metadata(missing, label="file")
            with self.assertRaises(ValueError):
                inputs_module._regular_file_metadata(directory, label="file")
            with self.assertRaises(ValueError):
                inputs_module._directory_metadata(missing, label="directory")
            with self.assertRaises(ValueError):
                inputs_module._directory_metadata(value, label="directory")

            with (
                patch.object(os, "fstat", return_value=os.stat(directory)),
                self.assertRaises(ValueError),
            ):
                inputs_module._read_regular_file(value, label="value")
            with (
                patch.object(os, "open", side_effect=OSError("open failed")),
                self.assertRaises(ValueError),
            ):
                inputs_module._read_regular_file(value, label="value")
            with (
                patch.object(Path, "lstat", side_effect=[os.stat(value), OSError("stat failed")]),
                self.assertRaises(ValueError),
            ):
                inputs_module._read_regular_file(value, label="value")
            with (
                patch.object(Path, "lstat", side_effect=[os.stat(value), os.stat(directory)]),
                self.assertRaises(ValueError),
            ):
                inputs_module._read_regular_file(value, label="value")

            bundle = inputs_module.load_builder_input_bundle(
                source,
                input_id="input-a",
                input_type="reference",
                allowed_files=("value.txt",),
            )
            with self.assertRaises(TypeError):
                inputs_module._validate_manifest_files(cast(Sequence[InterventionFile], object()))
            with self.assertRaises(ValueError):
                inputs_module._validate_manifest_files((_file("value.txt"), _file("value.txt")))
            with (
                patch.object(
                    inputs_module, "canonical_builder_input_manifest_bytes", side_effect=TypeError("bad manifest")
                ),
                self.assertRaises(ValueError),
            ):
                inputs_module._manifest_hash(bundle.manifest)
            value.unlink()
            with self.assertRaises(RuntimeError):
                inputs_module._snapshot_manifest_files(source, bundle.manifest)

            for number in (0, 1000):
                with self.subTest(number=number), self.assertRaises(ValueError):
                    inputs_module._target_for_number(number)
            with self.assertRaises(ValueError):
                inputs_module._validate_target("reference_files/builder-inputs/input-001", expected="other")
            with self.assertRaises(ValueError):
                inputs_module._validate_target("other/input-001")
            with self.assertRaises(ValueError):
                inputs_module._validate_target("reference_files/builder-inputs/input-1")

            with self.assertRaises(TypeError):
                inputs_module._validate_bundle_for_staging(cast(BuilderInputBundle, object()))
            forged_bundle = object.__new__(BuilderInputBundle)
            object.__setattr__(forged_bundle, "root", "not-a-path")
            object.__setattr__(forged_bundle, "manifest", bundle.manifest)
            with self.assertRaises(TypeError):
                inputs_module._validate_bundle_for_staging(forged_bundle)
            object.__setattr__(forged_bundle, "root", source)
            object.__setattr__(forged_bundle, "manifest", object())
            with self.assertRaises(TypeError):
                inputs_module._validate_bundle_for_staging(forged_bundle)

            self.assertFalse(inputs_module._path_exists_without_following(missing))
            with patch.object(Path, "lstat", side_effect=OSError("inspect failed")), self.assertRaises(ValueError):
                inputs_module._path_exists_without_following(missing)

            workspace = root / "workspace"
            workspace.mkdir()
            namespace = workspace / "reference_files"
            namespace.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                inputs_module._validate_destination_namespace(workspace, ())
            namespace.unlink()
            (workspace / "REFERENCE_FILES").mkdir()
            with self.assertRaises(FileExistsError):
                inputs_module._validate_destination_namespace(workspace, ())
            (workspace / "REFERENCE_FILES").rmdir()
            with patch.object(Path, "iterdir", side_effect=OSError("scan failed")), self.assertRaises(ValueError):
                inputs_module._validate_destination_namespace(workspace, ())

            destination = root / "copy-failure.txt"
            with (
                patch.object(os, "open", return_value=123),
                patch.object(os, "fdopen", side_effect=OSError("fdopen failed")),
                patch.object(os, "close", side_effect=OSError("close failed")),
                self.assertRaises(OSError),
            ):
                inputs_module._copy_exclusive(destination, b"content")
            self.assertFalse(destination.exists())
            with self.assertRaises(FileExistsError):
                inputs_module._mkdir_exclusive(workspace)
            with patch.object(os, "chmod", side_effect=OSError("chmod failed")):
                inputs_module._make_directory_read_only(workspace)

            with patch.object(Path, "lstat", side_effect=FileNotFoundError):
                inputs_module._remove_tree_without_following(missing)
            with patch.object(Path, "lstat", side_effect=OSError("lstat failed")):
                inputs_module._remove_tree_without_following(missing)
            link = root / "tree-link"
            link.symlink_to(source, target_is_directory=True)
            with patch.object(Path, "unlink", side_effect=OSError("unlink failed")):
                inputs_module._remove_tree_without_following(link)
            tree = root / "tree"
            tree.mkdir()
            with (
                patch.object(os, "chmod", side_effect=OSError("chmod failed")),
                patch.object(Path, "iterdir", side_effect=OSError("iter failed")),
                patch.object(Path, "rmdir", side_effect=OSError("rmdir failed")),
            ):
                inputs_module._remove_tree_without_following(tree)

            broken_inputs = cast(Sequence[BuilderInputBundle], object())
            with self.assertRaises(TypeError):
                inputs_module.stage_builder_inputs(broken_inputs, workspace)
            with self.assertRaises(TypeError):
                inputs_module.stage_builder_inputs((), cast(Path, object()))
            with self.assertRaises(ValueError):
                inputs_module.stage_builder_inputs((bundle, bundle), workspace)
            with patch.object(inputs_module, "_MAX_TARGET_NUMBER", 0), self.assertRaises(ValueError):
                inputs_module.stage_builder_inputs((bundle,), workspace)

            with self.assertRaises(TypeError):
                inputs_module.verify_staged_builder_inputs(
                    cast(Sequence[inputs_module.StagedBuilderInput], "bad"), workspace
                )
            with self.assertRaises(TypeError):
                inputs_module.verify_staged_builder_inputs((), cast(Path, object()))
            with self.assertRaises(TypeError):
                inputs_module.verify_staged_builder_inputs(
                    (cast(inputs_module.StagedBuilderInput, object()),), workspace
                )
            staged_record = inputs_module.StagedBuilderInput("input-a", "target", ())
            with self.assertRaises(ValueError):
                inputs_module.verify_staged_builder_inputs((staged_record, staged_record), workspace)

            forged_record = object.__new__(inputs_module.StagedBuilderInput)
            object.__setattr__(forged_record, "input_id", "input-a")
            object.__setattr__(forged_record, "target", "reference_files/builder-inputs/input-001")
            object.__setattr__(
                forged_record,
                "materialized_files",
                (cast(InterventionFile, _PathOnly("reference_files/builder-inputs/input-001/value.txt")),),
            )
            with self.assertRaises(TypeError):
                inputs_module.verify_staged_builder_inputs((forged_record,), workspace)

            empty_namespace_workspace = root / "empty-namespace-workspace"
            empty_namespace_workspace.mkdir()
            (empty_namespace_workspace / "reference_files").mkdir()
            with self.assertRaises(ValueError):
                inputs_module.verify_staged_builder_inputs((), empty_namespace_workspace)
            missing_namespace_workspace = root / "missing-namespace-workspace"
            missing_namespace_workspace.mkdir()
            missing_file = _file("reference_files/builder-inputs/input-001/value.txt", b"value")
            missing_record = inputs_module.StagedBuilderInput(
                "input-a", "reference_files/builder-inputs/input-001", (missing_file,)
            )
            with self.assertRaises(ValueError):
                inputs_module.verify_staged_builder_inputs((missing_record,), missing_namespace_workspace)

            scan_workspace = root / "scan-workspace"
            scan_workspace.mkdir()
            scan_namespace = scan_workspace / "reference_files"
            scan_namespace.mkdir()
            with patch.object(Path, "iterdir", side_effect=OSError("scan failed")), self.assertRaises(ValueError):
                inputs_module._walk_namespace(scan_namespace, "reference_files", set(), set())
            scan_child = scan_namespace / "child.txt"
            scan_child.write_text("child", encoding="utf-8")
            with (
                patch.object(Path, "lstat", side_effect=[os.stat(scan_namespace), OSError("child changed")]),
                self.assertRaises(ValueError),
            ):
                inputs_module._walk_namespace(scan_namespace, "reference_files", set(), set())

            materialized = _file("reference_files/builder-inputs/input-001/value.txt", b"value")
            with self.assertRaises(ValueError):
                inputs_module._read_materialized_file(root / "missing-materialized", materialized)
            with self.assertRaises(ValueError):
                inputs_module._read_materialized_file(directory, materialized)
            value.write_bytes(b"value")
            with (
                patch.object(os, "fstat", return_value=os.stat(directory)),
                self.assertRaises(ValueError),
            ):
                inputs_module._read_materialized_file(value, materialized)
            with (
                patch.object(Path, "lstat", side_effect=[os.stat(value), OSError("verify stat failed")]),
                self.assertRaises(ValueError),
            ):
                inputs_module._read_materialized_file(value, materialized)

            extra_workspace = root / "extra-directory-workspace"
            extra_workspace.mkdir()
            extra_namespace = extra_workspace / "reference_files" / "builder-inputs" / "input-001"
            extra_namespace.mkdir(parents=True)
            (extra_namespace / "value.txt").write_bytes(b"value")
            (extra_namespace / "unexpected").mkdir()
            exact_record = inputs_module.StagedBuilderInput(
                "input-a", "reference_files/builder-inputs/input-001", (materialized,)
            )
            with self.assertRaises(ValueError):
                inputs_module.verify_staged_builder_inputs((exact_record,), extra_workspace)

    def test_runner_validation_helpers_cover_planning_identity_and_durability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            source = root / "input"
            source.mkdir()
            (source / "guide.txt").write_bytes(b"guide")
            bundle = _input_bundle(source)
            task = BenchmarkTask(TaskSpec("task-a", "prompt"))
            benchmark = _Benchmark((task,))
            run_config = ExperimentRunConfig(
                "executor-a",
                "executor-a",
                "evaluator-a",
                None,
                None,
                1.0,
                1.0,
                False,
                False,
                1,
                3,
            )
            with self.assertRaises(ValueError):
                runner_module._canonical_existing(root / "missing", label="path")
            with self.assertRaises(ValueError):
                runner_module._canonical_existing(source, label="file", directory=False)
            planned = root / "planned"
            self.assertEqual(runner_module._canonical_planned_root(planned, label="planned"), planned)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                runner_module._canonical_planned_root(existing, label="planned")

            with self.assertRaises(ValueError):
                runner_module._validate_path_namespace(root, root, profile.source, {"input-a": source})
            with self.assertRaises(ValueError):
                runner_module._validate_path_namespace(
                    root / "out", root / "runtime", profile.source, {"input-a": root / "out"}
                )

            items = runner_module._make_schedule((task,), profile.profile.arms, 3, root / "out", root / "runtime")
            self.assertEqual(len(items), 1)
            with self.assertRaises(ValueError):
                runner_module._validate_planned_paths((items[0].output_root, items[0].output_root))
            existing_path = root / "existing-path"
            existing_path.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner_module._validate_planned_paths((existing_path,))

            with patch.object(runner_module.secrets, "token_hex", return_value="same"):
                two_tasks = (task, BenchmarkTask(TaskSpec("task-b", "prompt")))
                with self.assertRaises(ValueError):
                    runner_module._make_schedule(
                        two_tasks, profile.profile.arms, 3, root / "out-2", root / "runtime-2"
                    )

            summary = RunSummary(
                benchmark.name,
                "executor-a",
                items[0].output_root,
                items[0].application_root,
                "completed",
                1,
                {},
                {},
                {"task-a": "run-a"},
            )
            self.assertEqual(runner_module._application_run_id(summary, items[0]), "run-a")
            for bad in ({}, {"other": "run"}, {"task-a": ""}, {"task-a": items[0].schedule_id}):
                with self.subTest(bad=bad):
                    invalid = replace(summary, application_run_ids=bad)
                    with self.assertRaises((TypeError, ValueError)):
                        runner_module._application_run_id(invalid, items[0])

            metadata_path = root / "metadata.json"
            metadata: dict[str, object] = {"entries": [{"task_id": "task-a"}]}
            runner_module._persist_metadata(metadata_path, metadata, status="running", completed=0, finished=False)
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["status"], "running")
            with (
                patch.object(os, "replace", side_effect=OSError("replace failed")),
                self.assertRaises(OSError),
            ):
                runner_module._write_json(metadata_path, {"payload": True})
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["status"], "running")

            self.assertEqual(runner_module._safe_schedule_id("safe-id"), "safe-id")
            for bad_id in ("", ".", "..", "a/b", "a\\b", "a\x00b"):
                with self.subTest(bad_id=bad_id), self.assertRaises(ValueError):
                    runner_module._safe_schedule_id(bad_id)

            self.assertEqual(runner_module._validate_tasks(benchmark, 1), (task,))
            empty_benchmark = _Benchmark(())
            with self.assertRaises(ValueError):
                runner_module._validate_tasks(empty_benchmark, 1)

            descriptors = runner_module._configuration_payload(
                profile,
                run_config,
                benchmark,
                {"input-a": bundle},
                _Builder(),
                BuilderPreflightResult("builder-a", True, "executor-a"),
                PreflightResult("executor-a", True, "1", "local"),
                EvaluatorPreflightResult("evaluator-a", EvaluatorType.BENCHMARK_NATIVE, True),
                _Executor(),
            )
            self.assertEqual(canonical_json_sha256(descriptors), canonical_json_sha256(descriptors))

    def test_runner_loader_component_and_task_contracts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile(root)
            source = root / "input"
            source.mkdir()
            (source / "guide.txt").write_bytes(b"guide")
            bundle = _input_bundle(source)
            task = BenchmarkTask(TaskSpec("task-a", "prompt"))
            benchmark = _Benchmark((task,))
            evaluator = _Evaluator()
            builder = _Builder()
            application_executor = _Executor()
            config = ExperimentRunConfig(
                "executor-a",
                "executor-a",
                "evaluator-a",
                None,
                None,
                1.0,
                1.0,
                False,
                False,
                1,
                3,
            )

            with patch.object(Path, "lstat", side_effect=OSError("lstat failed")), self.assertRaises(ValueError):
                runner_module._canonical_existing(source, label="source")
            with self.assertRaises(ValueError):
                runner_module._canonical_existing(profile.source, label="source", directory=True)
            with (
                patch.object(Path, "resolve", side_effect=RuntimeError("resolve failed")),
                self.assertRaises(ValueError),
            ):
                runner_module._canonical_existing(source, label="source")
            with self.assertRaises(ValueError):
                runner_module._canonical_planned_root(Path("relative-planned"), label="planned")
            with self.assertRaises(ValueError):
                runner_module._canonical_planned_root(root / "missing" / "planned", label="planned")
            with self.assertRaises(ValueError):
                runner_module._validate_path_namespace(
                    root / "out", root / "runtime", root / "out" / "profile.json", {}
                )
            with self.assertRaises(TypeError):
                runner_module._require_instance(object(), _Builder, "builder")
            with self.assertRaises(TypeError):
                runner_module._require_exact_result(object(), RunSummary, "summary")

            with self.assertRaises(TypeError):
                runner_module._validate_loaded_inputs(profile, cast(Mapping[str, Path | str], object()), {})
            with self.assertRaises(ValueError):
                runner_module._validate_loaded_inputs(profile, {}, {})
            with self.assertRaises(TypeError):
                runner_module._validate_loaded_inputs(
                    profile, {"input-a": cast(Path | str, object())}, {"input-a": bundle}
                )
            with self.assertRaises(TypeError):
                runner_module._validate_loaded_inputs(profile, {"input-a": source}, object())
            with self.assertRaises(TypeError):
                runner_module._validate_loaded_inputs(profile, {"input-a": source}, {"": bundle})
            with self.assertRaises(TypeError):
                runner_module._validate_loaded_inputs(profile, {"input-a": source}, {"input-a": object()})
            with self.assertRaises(ValueError):
                runner_module._validate_loaded_inputs(profile, {"input-a": source}, {"other": bundle})

            other_source = root / "other-source"
            other_source.mkdir()
            wrong_root_bundle = BuilderInputBundle(other_source, bundle.manifest)
            with self.assertRaises(ValueError):
                runner_module._validate_loaded_inputs(profile, {"input-a": source}, {"input-a": wrong_root_bundle})

            class DerivedBundle(BuilderInputBundle):
                pass

            with self.assertRaises(TypeError):
                runner_module._validate_loaded_inputs(
                    profile,
                    {"input-a": source},
                    {"input-a": DerivedBundle(source, bundle.manifest)},
                )

            class DerivedManifest(BuilderInputManifest):
                pass

            derived_manifest = DerivedManifest(
                bundle.manifest.input_id,
                bundle.manifest.input_type,
                bundle.manifest.source_revision,
                bundle.manifest.revision_status,
                bundle.manifest.files,
                bundle.manifest.bundle_sha256,
            )
            with self.assertRaises(TypeError):
                runner_module._validate_loaded_inputs(
                    profile,
                    {"input-a": source},
                    {"input-a": BuilderInputBundle(source, derived_manifest)},
                )

            manifest_variants = (
                BuilderInputManifest(
                    "other-input",
                    bundle.manifest.input_type,
                    bundle.manifest.source_revision,
                    bundle.manifest.revision_status,
                    bundle.manifest.files,
                    bundle.manifest.bundle_sha256,
                ),
                BuilderInputManifest(
                    bundle.manifest.input_id,
                    "other-type",
                    bundle.manifest.source_revision,
                    bundle.manifest.revision_status,
                    bundle.manifest.files,
                    bundle.manifest.bundle_sha256,
                ),
                BuilderInputManifest(
                    bundle.manifest.input_id,
                    bundle.manifest.input_type,
                    "revision",
                    "available",
                    bundle.manifest.files,
                    bundle.manifest.bundle_sha256,
                ),
                BuilderInputManifest(
                    bundle.manifest.input_id,
                    bundle.manifest.input_type,
                    bundle.manifest.source_revision,
                    bundle.manifest.revision_status,
                    (_file("other.txt"),),
                    bundle.manifest.bundle_sha256,
                ),
            )
            for variant in manifest_variants:
                with self.subTest(variant=variant), self.assertRaises(ValueError):
                    runner_module._validate_loaded_inputs(
                        profile, {"input-a": source}, {"input-a": BuilderInputBundle(source, variant)}
                    )
            expected_profile = LoadedExperimentProfile(
                replace(
                    profile.profile,
                    inputs=(replace(profile.profile.inputs[0], expected_bundle_sha256="0" * 64),),
                ),
                profile.source,
                profile.sha256,
            )
            with self.assertRaises(ValueError):
                runner_module._validate_loaded_inputs(expected_profile, {"input-a": source}, {"input-a": bundle})

            with self.assertRaises(ValueError):
                runner_module._validate_profile_and_components(
                    replace(profile, profile=replace(profile.profile, benchmark="other")),
                    config,
                    benchmark,
                    evaluator,
                    builder,
                    application_executor,
                    root,
                )
            with self.assertRaises(ValueError):
                runner_module._validate_profile_and_components(
                    profile,
                    replace(config, evaluator="other"),
                    benchmark,
                    evaluator,
                    builder,
                    application_executor,
                    root,
                )
            with self.assertRaises(ValueError):
                runner_module._validate_profile_and_components(
                    profile,
                    replace(config, application_executor="other"),
                    benchmark,
                    evaluator,
                    builder,
                    application_executor,
                    root,
                )
            builder_effort_config = replace(config)
            object.__setattr__(builder_effort_config, "builder_reasoning_effort", "low")
            with self.assertRaises(ValueError):
                runner_module._validate_profile_and_components(
                    profile,
                    builder_effort_config,
                    benchmark,
                    evaluator,
                    builder,
                    application_executor,
                    root,
                )
            application_effort_config = replace(config)
            object.__setattr__(application_effort_config, "application_reasoning_effort", "low")
            with self.assertRaises(ValueError):
                runner_module._validate_profile_and_components(
                    profile,
                    application_effort_config,
                    benchmark,
                    evaluator,
                    builder,
                    application_executor,
                    root,
                )

            with (
                patch.object(
                    evaluator,
                    "preflight",
                    return_value=EvaluatorPreflightResult("evaluator-a", EvaluatorType.BENCHMARK_NATIVE, False),
                ),
                self.assertRaises(RuntimeError),
            ):
                runner_module._validate_profile_and_components(
                    profile, config, benchmark, evaluator, builder, application_executor, root
                )
            with (
                patch.object(
                    application_executor,
                    "preflight",
                    return_value=PreflightResult("executor-a", False),
                ),
                self.assertRaises(RuntimeError),
            ):
                runner_module._validate_profile_and_components(
                    profile, config, benchmark, evaluator, builder, application_executor, root
                )
            with (
                patch.object(
                    builder,
                    "preflight",
                    return_value=BuilderPreflightResult("builder-a", False, "executor-a"),
                ),
                self.assertRaises(RuntimeError),
            ):
                runner_module._validate_profile_and_components(
                    profile, config, benchmark, evaluator, builder, application_executor, root
                )
            with (
                patch.object(
                    builder,
                    "preflight",
                    return_value=BuilderPreflightResult("other-builder", True, "executor-a"),
                ),
                self.assertRaises(RuntimeError),
            ):
                runner_module._validate_profile_and_components(
                    profile, config, benchmark, evaluator, builder, application_executor, root
                )

            class TooManyTasks(_Benchmark):
                def load_tasks(self, limit: int) -> Sequence[BenchmarkTask]:
                    del limit
                    return self.tasks + (task,)

            with self.assertRaises(ValueError):
                runner_module._validate_tasks(TooManyTasks((task,)), 1)

            class BrokenTasks(_Benchmark):
                def load_tasks(self, limit: int) -> Sequence[BenchmarkTask]:
                    del limit
                    return cast(Sequence[BenchmarkTask], object())

            with self.assertRaises(TypeError):
                runner_module._validate_tasks(BrokenTasks((task,)), 1)

            class DerivedTaskSpec(TaskSpec):
                pass

            class DerivedBenchmarkTask(BenchmarkTask):
                pass

            with self.assertRaises(TypeError):
                runner_module._validate_tasks(_Benchmark((DerivedBenchmarkTask(task.execution),)), 1)
            with self.assertRaises(TypeError):
                runner_module._validate_tasks(_Benchmark((BenchmarkTask(DerivedTaskSpec("task-a", "prompt")),)), 1)
            with self.assertRaises(ValueError):
                runner_module._validate_tasks(_Benchmark((BenchmarkTask(TaskSpec("", "prompt")),)), 1)
            duplicate = BenchmarkTask(TaskSpec("task-a", "other"))
            with self.assertRaises(ValueError):
                runner_module._validate_tasks(_Benchmark((task, duplicate)), 2)
            with self.assertRaises(ValueError):
                runner_module._safe_schedule_id(object())

            selected = runner_module._SelectedTaskBenchmark(benchmark, task)
            self.assertTrue(selected.is_prepared())
            with self.assertRaises(ValueError):
                selected.load_tasks(2)
            other_task = BenchmarkTask(TaskSpec("other", "prompt"))
            with self.assertRaises(ValueError):
                selected.materialize(other_task, root)
            with self.assertRaises(ValueError):
                selected.execution_task(other_task, root, network_policy="disabled")

            descriptor_config = replace(config)
            object.__setattr__(descriptor_config, "builder_reasoning_effort", "low")
            object.__setattr__(descriptor_config, "application_reasoning_effort", "high")
            self.assertEqual(
                runner_module._config_payload(descriptor_config)["builder_reasoning_effort_requested"], "low"
            )
            self.assertEqual(
                runner_module._config_payload(descriptor_config)["application_reasoning_effort_requested"], "high"
            )
            self.assertIsNone(runner_module._enum_text(_ExplodingAttribute()))
            self.assertIsNone(runner_module._executor_invocation_mode(cast(Executor, _ExplodingAttribute())))
            with (
                patch.object(runner_module, "task_layout", side_effect=OSError("layout failed")),
                self.assertRaises(ValueError),
            ):
                runner_module._plan_artifact_dir(root, task, root)
            with (
                patch.object(Path, "resolve", side_effect=RuntimeError("planned resolve failed")),
                self.assertRaises(ValueError),
            ):
                runner_module._validate_planned_paths((root / "planned",))

            item = runner_module._make_schedule((task,), profile.profile.arms, 3, root / "out", root / "runtime")[0]
            summary = RunSummary(
                benchmark.name,
                application_executor.name,
                item.output_root,
                item.application_root,
                "completed",
                1,
                {},
                {},
                {task.execution.task_id: "run"},
            )
            invalid_summary = replace(summary, application_run_ids=cast(Mapping[str, str], "bad"))
            with self.assertRaises(TypeError):
                runner_module._application_run_id(invalid_summary, item)

            artifact_root = root / "artifact"
            artifact_root.mkdir()
            runtime_root = root / "runtime"
            runtime_root.mkdir()
            request = BuildRequest("run", task.execution, (bundle,), runtime_root, artifact_root)
            execution = ExecutionResult(
                runtime="test",
                task_id=task.execution.task_id,
                executor=builder.name,
                executor_version="1",
                invocation_mode="deterministic",
                auth_mode="local",
                workspace=runtime_root,
                deliverables_dir=artifact_root,
                status=ExecutionStatus.COMPLETED,
                started_at="start",
                finished_at="finish",
                exit_code=0,
                available_outputs=frozenset(),
                failure=None,
            )
            manifest = _manifest()
            valid_bundle = InterventionBundle(artifact_root / "skill", manifest)

            def forged_result(bundle_value: object, execution_value: object) -> BuildResult:
                result = object.__new__(BuildResult)
                object.__setattr__(result, "build_run_id", request.build_run_id)
                object.__setattr__(result, "task_id", request.task.task_id)
                object.__setattr__(result, "builder", builder.name)
                object.__setattr__(result, "status", BuildStatus.COMPLETED)
                object.__setattr__(result, "inputs", tuple(item.manifest for item in request.inputs))
                object.__setattr__(result, "execution", execution_value)
                object.__setattr__(result, "bundle", bundle_value)
                object.__setattr__(result, "failure_phase", None)
                return result

            with self.assertRaises(TypeError):
                runner_module._validate_build_result(forged_result(None, execution), request, builder)
            with self.assertRaises(ValueError):
                runner_module._validate_build_result(forged_result(valid_bundle, None), request, builder)
            with self.assertRaises(ValueError):
                runner_module._validate_build_result(
                    forged_result(InterventionBundle(None, manifest), execution), request, builder
                )


if __name__ == "__main__":
    unittest.main()
