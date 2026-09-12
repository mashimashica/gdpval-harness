# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Iterator, overload
from unittest.mock import patch

import eval_harness.builders.artifact as artifact_module
import eval_harness.builders.base as base_module
import eval_harness.builders.executor_skill as executor_skill_module
import eval_harness.builders.inputs as inputs_module
import eval_harness.builders.prompt as prompt_module
from eval_harness.builders.artifact import ArtifactHandoffError, GeneratedSkillValidationError
from eval_harness.builders.base import (
    BuilderInputBundle,
    BuilderInputManifest,
    BuilderPreflightResult,
    BuildFailurePhase,
    BuildRequest,
    BuildResult,
    BuildStatus,
)
from eval_harness.builders.executor_skill import ExecutorSkillBuilder
from eval_harness.builders.inputs import (
    StagedBuilderInput,
    load_builder_input_bundle,
    stage_builder_inputs,
    verify_staged_builder_inputs,
)
from eval_harness.executors.base import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    Executor,
    PreflightResult,
    TaskSpec,
)
from eval_harness.interventions import InterventionBundle, load_agent_skill_bundle
from eval_harness.interventions.base import (
    InterventionFile,
    compute_bundle_sha256,
    file_evidence,
)


class ReliabilityExecutor(Executor):
    name = "reliability-executor"
    invocation_mode = "deterministic"

    def __init__(
        self,
        *,
        status: ExecutionStatus = ExecutionStatus.COMPLETED,
        preflight_ok: bool = True,
        preflight_exception: Exception | None = None,
        raise_on_execute: Exception | None = None,
        result_task_id: str | None = None,
        result_executor: str | None = None,
        result_workspace: Path | None = None,
        result_deliverables: Path | None = None,
    ) -> None:
        self.status = status
        self.preflight_ok = preflight_ok
        self.preflight_exception = preflight_exception
        self.raise_on_execute = raise_on_execute
        self.result_task_id = result_task_id
        self.result_executor = result_executor
        self.result_workspace = result_workspace
        self.result_deliverables = result_deliverables
        self.requests: list[ExecutionRequest] = []

    def preflight(self) -> PreflightResult:
        if self.preflight_exception is not None:
            raise self.preflight_exception
        return PreflightResult(
            executor=self.name,
            ok=self.preflight_ok,
            version="reliability-1",
            auth_mode="local",
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        request.executor_dir.mkdir(parents=True, exist_ok=True)
        (request.executor_dir / "stdout.log").write_text("builder log", encoding="utf-8")
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        if self.status is ExecutionStatus.COMPLETED:
            skill = request.deliverables_dir / "reliability-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                "---\nname: reliability-skill\ndescription: reliable skill\n---\n\nUse it.\n",
                encoding="utf-8",
            )
        return ExecutionResult(
            task_id=self.result_task_id or request.task.task_id,
            executor=self.result_executor or self.name,
            executor_version="reliability-1",
            invocation_mode=self.invocation_mode,
            auth_mode="local",
            workspace=self.result_workspace or request.workspace,
            deliverables_dir=self.result_deliverables or request.deliverables_dir,
            status=self.status,
            started_at="2026-09-12T00:00:00+00:00",
            finished_at="2026-09-12T00:00:01+00:00",
            exit_code=0,
        )


class BuilderReliabilityTests(unittest.TestCase):
    def _source(self, root: Path, name: str = "source") -> Path:
        source = root / name
        (source / "nested").mkdir(parents=True)
        (source / "a.txt").write_bytes(b"alpha")
        (source / "nested" / "b.txt").write_bytes(b"bravo")
        return source

    def _bundle(self, source: Path, input_id: str = "input-one") -> BuilderInputBundle:
        return load_builder_input_bundle(
            source,
            input_id=input_id,
            input_type="creation-files",
            allowed_files=("nested/b.txt", "a.txt"),
        )

    def _request(self, root: Path, executor: ReliabilityExecutor | None = None) -> BuildRequest:
        source = self._source(root)
        control = root / "control"
        control.mkdir()
        bundle = self._bundle(source)
        return BuildRequest(
            build_run_id="build-reliability",
            task=TaskSpec("task-reliability", "create a reusable skill"),
            inputs=(bundle,),
            runtime_root=control / "runtime",
            artifact_root=control / "artifact",
            model="local-model",
            timeout_seconds=4.0,
        )

    def _execution(self, status: ExecutionStatus = ExecutionStatus.COMPLETED) -> ExecutionResult:
        return ExecutionResult(
            task_id="task-reliability",
            executor="reliability-executor",
            executor_version="1",
            invocation_mode="deterministic",
            auth_mode="local",
            workspace=Path("/tmp/workspace"),
            deliverables_dir=Path("/tmp/deliverables"),
            status=status,
            started_at="2026-09-12T00:00:00+00:00",
            finished_at="2026-09-12T00:00:01+00:00",
            exit_code=0,
        )

    def test_manifest_and_build_records_reject_ambiguous_contracts(self) -> None:
        content = b"payload"
        item_a = file_evidence("a.txt", content)
        item_b = file_evidence("b.txt", content)
        bundle_hash = compute_bundle_sha256(((item_a.path, content), (item_b.path, content)))
        valid = BuilderInputManifest(
            input_id="input-one",
            input_type="creation-files",
            source_revision=None,
            revision_status="unavailable",
            files=(item_a, item_b),
            bundle_sha256=bundle_hash,
        )
        self.assertEqual(
            valid.manifest_sha256,
            hashlib.sha256(base_module.canonical_builder_input_manifest_bytes(valid)).hexdigest(),
        )

        for path in ("", "../escape", "nested\\file", "/absolute", "a//b", "a/./b", "a/../b", "C:/file"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                base_module._validate_manifest_path(path)
        with self.assertRaises(ValueError):
            base_module._validate_manifest_path("\ud800")
        for source_revision, revision_status in (
            (None, "available"),
            ("revision", "unavailable"),
            ("revision", "not-a-status"),
        ):
            with (
                self.subTest(source_revision=source_revision, revision_status=revision_status),
                self.assertRaises(ValueError),
            ):
                base_module._validate_revision(source_revision, revision_status)
        with self.assertRaises(ValueError):
            BuilderInputManifest("id", "type", None, "unavailable", (item_b, item_a), bundle_hash)
        prefix_parent = file_evidence("a", content)
        with self.assertRaises(ValueError):
            BuilderInputManifest(
                "id", "type", None, "unavailable", (prefix_parent, file_evidence("a/child.txt", content)), bundle_hash
            )
        collision = file_evidence("A.txt", content)
        with self.assertRaises(ValueError):
            BuilderInputManifest("id", "type", None, "unavailable", (collision, item_a), bundle_hash)
        with self.assertRaises(ValueError):
            BuilderInputManifest("id", "type", None, "unavailable", (item_a,), "0" * 64, "1" * 64)

        execution = self._execution()
        for status, phase, execution_value in (
            (BuildStatus.FAILED, None, None),
            (BuildStatus.FAILED, BuildFailurePhase.PREFLIGHT, execution),
            (BuildStatus.TIMED_OUT, BuildFailurePhase.EXECUTION, self._execution(ExecutionStatus.FAILED)),
            (BuildStatus.NO_ARTIFACT, BuildFailurePhase.ARTIFACT_VALIDATION, execution),
            (BuildStatus.FAILED, BuildFailurePhase.ARTIFACT_HANDOFF, self._execution(ExecutionStatus.FAILED)),
        ):
            with self.subTest(status=status, phase=phase), self.assertRaises(ValueError):
                BuildResult(
                    build_run_id="run",
                    task_id="task-reliability",
                    builder="builder",
                    status=status,
                    inputs=(valid,),
                    execution=execution_value,
                    failure_phase=phase,
                )

    def test_contract_dataclasses_revalidate_forged_runtime_values(self) -> None:
        content = b"payload"
        item = file_evidence("a.txt", content)
        manifest = BuilderInputManifest(
            "input-one",
            "creation-files",
            None,
            "unavailable",
            (item,),
            compute_bundle_sha256(((item.path, content),)),
        )
        bundle = BuilderInputBundle(Path("/tmp/source"), manifest)
        object.__setattr__(bundle, "root", object())
        with self.assertRaises(TypeError):
            bundle.__post_init__()
        object.__setattr__(bundle, "root", Path("/tmp/source"))
        object.__setattr__(bundle, "manifest", object())
        with self.assertRaises(TypeError):
            bundle.__post_init__()
        object.__setattr__(bundle, "manifest", manifest)

        broken_manifest = BuilderInputManifest(
            "input-one",
            "creation-files",
            None,
            "unavailable",
            (item,),
            manifest.bundle_sha256,
        )
        object.__setattr__(broken_manifest, "files", None)
        with self.assertRaises(TypeError):
            broken_manifest.__post_init__()

        request = BuildRequest(
            "run", TaskSpec("task", "prompt"), (bundle,), Path("/tmp/runtime"), Path("/tmp/artifact")
        )
        object.__setattr__(request, "inputs", None)
        with self.assertRaises(TypeError):
            request.__post_init__()
        object.__setattr__(request, "inputs", (bundle,))
        object.__setattr__(request, "runtime_root", object())
        with self.assertRaises(TypeError):
            request.__post_init__()

        preflight = BuilderPreflightResult("builder", True, details=("ready",))
        object.__setattr__(preflight, "details", None)
        with self.assertRaises(TypeError):
            preflight.__post_init__()

        result = BuildResult(
            "run", "task", "builder", BuildStatus.FAILED, (manifest,), failure_phase=BuildFailurePhase.EXECUTION
        )
        object.__setattr__(result, "status", object())
        with self.assertRaises(ValueError):
            result.__post_init__()
        object.__setattr__(result, "status", BuildStatus.FAILED)
        object.__setattr__(result, "inputs", None)
        with self.assertRaises(TypeError):
            result.__post_init__()
        object.__setattr__(result, "inputs", (manifest,))
        object.__setattr__(result, "execution", object())
        with self.assertRaises(TypeError):
            result.__post_init__()
        object.__setattr__(result, "execution", None)
        object.__setattr__(result, "bundle", object())
        with self.assertRaises(TypeError):
            result.__post_init__()
        object.__setattr__(result, "bundle", None)
        object.__setattr__(result, "failure_phase", object())
        with self.assertRaises(ValueError):
            result.__post_init__()

    def test_input_boundary_handles_broken_iterables_and_readback_failures(self) -> None:
        class BrokenSequence(Sequence[InterventionFile]):
            @overload
            def __getitem__(self, index: int) -> InterventionFile: ...

            @overload
            def __getitem__(self, index: slice) -> Sequence[InterventionFile]: ...

            def __getitem__(self, index: int | slice) -> InterventionFile | Sequence[InterventionFile]:
                raise TypeError(f"broken index {index}")

            def __len__(self) -> int:
                return 1

            def __iter__(self) -> Iterator[InterventionFile]:
                raise TypeError("broken iterator")
                yield from ()

        class BrokenStringSequence(Sequence[str]):
            @overload
            def __getitem__(self, index: int) -> str: ...

            @overload
            def __getitem__(self, index: slice) -> Sequence[str]: ...

            def __getitem__(self, index: int | slice) -> str | Sequence[str]:
                raise TypeError(f"broken index {index}")

            def __len__(self) -> int:
                return 1

            def __iter__(self) -> Iterator[str]:
                raise TypeError("broken iterator")
                yield from ()

        with self.assertRaises(TypeError):
            StagedBuilderInput("input", "target", BrokenSequence())
        with self.assertRaises(ValueError):
            StagedBuilderInput("", "target", ())
        with self.assertRaises(ValueError):
            StagedBuilderInput("input", "", ())
        with self.assertRaises(TypeError):
            inputs_module._validate_relative_path(object(), label="path")
        with self.assertRaises(TypeError):
            inputs_module._validate_allowlist(BrokenStringSequence())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            source.joinpath("a.txt").unlink()
            with self.assertRaises(RuntimeError):
                inputs_module._snapshot_manifest_files(source, bundle.manifest)
            long_path = root / ("x" * 5000)
            with self.assertRaises(ValueError):
                inputs_module._path_exists_without_following(long_path)
            destination = root / "missing-parent" / "file.txt"
            with self.assertRaises(OSError):
                inputs_module._copy_exclusive(destination, b"bytes")
            directory = root / "directory"
            directory.mkdir()
            with self.assertRaises(FileExistsError):
                inputs_module._mkdir_exclusive(directory)
            with patch.object(os, "chmod", side_effect=OSError("read-only boundary")):
                inputs_module._make_directory_read_only(directory)

    def test_builder_destination_verification_handles_missing_symlink_and_special_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            workspace = root / "workspace"
            workspace.mkdir()
            staged = stage_builder_inputs((bundle,), workspace)
            expected = staged[0].materialized_files[0]
            destination = workspace / expected.path
            with self.assertRaises(ValueError):
                inputs_module._read_materialized_file(root / "missing", expected)
            symlink = root / "destination-link"
            symlink.symlink_to(destination)
            with self.assertRaises(ValueError):
                inputs_module._read_materialized_file(symlink, expected)
            os.chmod(workspace / "reference_files", 0o755)
            os.chmod(workspace / "reference_files" / "builder-inputs", 0o755)
            os.chmod(workspace / staged[0].target, 0o755)
            os.chmod(destination, 0o644)
            destination.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                inputs_module._read_materialized_file(destination, expected)
            destination.write_bytes(b"alpha")
            verify_staged_builder_inputs(staged, workspace)

            if hasattr(os, "mkfifo"):
                special = root / "special"
                os.mkfifo(special)
                with self.assertRaises(ValueError):
                    inputs_module._read_materialized_file(special, expected)

    def test_input_source_boundary_rejects_special_files_and_path_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            workspace = root / "workspace"
            workspace.mkdir()

            with self.assertRaises(ValueError):
                inputs_module._validate_allowlist(())
            with self.assertRaises(ValueError):
                inputs_module._validate_relative_path("\ud800", label="path")
            with self.assertRaises(ValueError):
                inputs_module._target_for_number(0)
            with self.assertRaises(ValueError):
                inputs_module._target_for_number(1000)
            for target in (
                "reference_files/builder-inputs/input-000",
                "reference_files/builder-inputs/input-01",
                "reference_files/builder-inputs/input-00a",
                "reference_files/other/input-001",
            ):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    inputs_module._validate_target(target)

            with self.assertRaises(ValueError):
                inputs_module._canonical_existing_directory(root / "missing", label="source")
            regular = root / "regular"
            regular.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(ValueError):
                inputs_module._load_source_root(regular)
            link = root / "source-link"
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                inputs_module._load_source_root(link)
            with self.assertRaises(ValueError):
                inputs_module._regular_file_metadata(root / "missing", label="file")
            with self.assertRaises(ValueError):
                inputs_module._directory_metadata(root / "missing", label="directory")
            if hasattr(os, "mkfifo"):
                fifo = source / "fifo"
                os.mkfifo(fifo)
                with self.assertRaises(ValueError):
                    inputs_module._regular_file_metadata(fifo, label="fifo")

            bundle = self._bundle(source)
            with self.assertRaises(ValueError):
                inputs_module._validate_destination_namespace(workspace, (workspace,))
            namespace_alias = root / "namespace-target"
            namespace_alias.mkdir()
            (workspace / "Reference_Files").mkdir()
            with self.assertRaises(FileExistsError):
                inputs_module._validate_destination_namespace(workspace, (source,))
            (workspace / "Reference_Files").rmdir()
            (workspace / "reference_files").symlink_to(namespace_alias, target_is_directory=True)
            with self.assertRaises(ValueError):
                inputs_module._validate_destination_namespace(workspace, (source,))

            self.assertEqual(inputs_module._validate_bundle_for_staging(bundle)[0], source.resolve())

    def test_snapshot_hashes_and_staging_rollback_preserve_source_and_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            changed_size = replace(bundle.manifest.files[0], size=bundle.manifest.files[0].size + 1)
            forged_size = replace(
                bundle.manifest, files=(changed_size, bundle.manifest.files[1]), manifest_sha256=None
            )
            with self.assertRaises(RuntimeError):
                inputs_module._snapshot_manifest_files(source, forged_size)
            changed_hash = replace(bundle.manifest.files[0], sha256="0" * 64)
            forged_hash = replace(
                bundle.manifest, files=(changed_hash, bundle.manifest.files[1]), manifest_sha256=None
            )
            with self.assertRaises(RuntimeError):
                inputs_module._snapshot_manifest_files(source, forged_hash)
            forged_bundle = replace(bundle.manifest, bundle_sha256="0" * 64, manifest_sha256=None)
            with self.assertRaises(RuntimeError):
                inputs_module._snapshot_manifest_files(source, forged_bundle)

            workspace = root / "workspace"
            workspace.mkdir()
            staged = stage_builder_inputs((bundle,), workspace)
            self.assertEqual(staged[0].target, "reference_files/builder-inputs/input-001")
            self.assertEqual(
                sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*")),
                [
                    "reference_files",
                    "reference_files/builder-inputs",
                    "reference_files/builder-inputs/input-001",
                    "reference_files/builder-inputs/input-001/a.txt",
                    "reference_files/builder-inputs/input-001/nested",
                    "reference_files/builder-inputs/input-001/nested/b.txt",
                ],
            )
            verify_staged_builder_inputs(staged, workspace)
            with self.assertRaises(ValueError):
                inputs_module._read_materialized_file(
                    workspace / staged[0].materialized_files[0].path,
                    replace(staged[0].materialized_files[0], sha256="0" * 64),
                )

            namespace = workspace / "reference_files"
            for directory in (namespace, namespace / "builder-inputs", namespace / "builder-inputs" / "input-001"):
                os.chmod(directory, 0o755)
            (workspace / "reference_files" / "builder-inputs" / "input-001" / "a.txt").unlink()
            with self.assertRaises(ValueError):
                verify_staged_builder_inputs(staged, workspace)
            self.assertEqual((source / "a.txt").read_bytes(), b"alpha")

            empty = root / "empty-workspace"
            empty.mkdir()
            self.assertEqual(stage_builder_inputs((), empty), ())
            verify_staged_builder_inputs((), empty)
            self.assertFalse((empty / "reference_files").exists())

    def test_verify_rejects_duplicate_ids_targets_and_namespace_special_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            bundle = self._bundle(source)
            workspace = root / "workspace"
            workspace.mkdir()
            staged = stage_builder_inputs((bundle,), workspace)
            record = staged[0]
            with self.assertRaises(ValueError):
                verify_staged_builder_inputs((record, replace(record, input_id=record.input_id)), workspace)
            with self.assertRaises(ValueError):
                verify_staged_builder_inputs(
                    (replace(record, target="reference_files/builder-inputs/input-002"),), workspace
                )
            unsorted = replace(
                record,
                materialized_files=tuple(reversed(record.materialized_files)),
            )
            with self.assertRaises(ValueError):
                verify_staged_builder_inputs((unsorted,), workspace)
            with self.assertRaises(ValueError):
                verify_staged_builder_inputs(
                    (
                        replace(
                            record,
                            materialized_files=(replace(record.materialized_files[0], path=record.target),),
                        ),
                    ),
                    workspace,
                )

            special = workspace / "reference_files" / "builder-inputs" / "input-001" / "special"
            if hasattr(os, "mkfifo"):
                for directory in (
                    workspace / "reference_files",
                    workspace / "reference_files" / "builder-inputs",
                    workspace / "reference_files" / "builder-inputs" / "input-001",
                ):
                    os.chmod(directory, 0o755)
                os.mkfifo(special)
                with self.assertRaises(ValueError):
                    verify_staged_builder_inputs(staged, workspace)

            inputs_module._remove_tree_without_following(root / "missing")
            link_target = root / "link-target"
            link_target.mkdir()
            link = root / "link"
            link.symlink_to(link_target, target_is_directory=True)
            inputs_module._remove_tree_without_following(link)
            self.assertFalse(link.exists())

    def test_sealed_artifact_rejects_invalid_shapes_and_keeps_partial_output_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deliverables = root / "deliverables"
            deliverables.mkdir()
            skill = deliverables / "reliability-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                "---\nname: reliability-skill\ndescription: valid\n---\n\ncontent\n",
                encoding="utf-8",
            )
            artifact_root = root / "artifact"
            sealed = artifact_module.seal_generated_skill(deliverables, artifact_root)
            self.assertEqual(sealed.root, (artifact_root / "reliability-skill").resolve())

            for path in (root / "missing", root / "file.txt"):
                if path.name == "file.txt":
                    path.write_text("file", encoding="utf-8")
                with self.subTest(path=path), self.assertRaises(ValueError):
                    artifact_module._existing_canonical_directory(path, label="candidate")
            symlink_target = root / "symlink-target"
            symlink_target.mkdir()
            symlink = root / "symlink"
            symlink.symlink_to(symlink_target, target_is_directory=True)
            with self.assertRaises(ValueError):
                artifact_module._existing_canonical_directory(symlink, label="candidate")

            with self.assertRaises(ValueError):
                artifact_module._validate_manifest_paths(())
            unsafe = InterventionFile("../escape", 1, "0" * 64)
            with self.assertRaises(ValueError):
                artifact_module._validate_manifest_paths((unsafe,))
            duplicate = InterventionFile("same.txt", 1, "0" * 64)
            with self.assertRaises(ValueError):
                artifact_module._validate_manifest_paths((duplicate, duplicate))

            source_item = sealed.manifest.files[0]
            with self.assertRaises(RuntimeError):
                artifact_module._source_file(root / "missing", source_item)
            destination = root / "destination.txt"
            destination.write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                artifact_module._verify_destination(destination, source_item, b"expected")

            existing = root / "existing"
            existing.mkdir()
            (existing / "keep").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                artifact_module.seal_generated_skill(deliverables, existing)
            self.assertEqual((existing / "keep").read_text(encoding="utf-8"), "keep")

            fresh_deliverables = root / "fresh-deliverables"
            fresh_deliverables.mkdir()
            fresh_skill = fresh_deliverables / "fresh-skill"
            fresh_skill.mkdir()
            (fresh_skill / "SKILL.md").write_text(
                "---\nname: fresh-skill\ndescription: valid\n---\n\ncontent\n",
                encoding="utf-8",
            )
            with patch.object(artifact_module, "_copy_exclusive", side_effect=OSError("copy boundary failure")):
                with self.assertRaises(ArtifactHandoffError):
                    artifact_module.seal_generated_skill(fresh_deliverables, root / "partial")
            self.assertFalse((root / "partial").exists())
            self.assertTrue(fresh_skill.joinpath("SKILL.md").is_file())

    def test_artifact_filesystem_boundaries_fail_closed_without_following_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            long_path = root / ("x" * 5000)
            with self.assertRaises(ValueError):
                artifact_module._existing_canonical_directory(long_path, label="long")
            with self.assertRaises(ValueError):
                artifact_module._path_exists_without_following(long_path)
            canonical_dir = root / "canonical"
            canonical_dir.mkdir()
            with self.assertRaises(ValueError):
                artifact_module._existing_canonical_directory(canonical_dir / "..", label="noncanonical")

            deliverables = root / "deliverables"
            deliverables.mkdir()
            skill = deliverables / "nested-skill"
            (skill / "references").mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                "---\nname: nested-skill\ndescription: nested\n---\n\ncontent\n",
                encoding="utf-8",
            )
            (skill / "references" / "guide.md").write_text("guide", encoding="utf-8")
            source_bundle = load_agent_skill_bundle(skill)
            nested_item = next(item for item in source_bundle.manifest.files if item.path == "references/guide.md")
            with self.assertRaises(RuntimeError):
                artifact_module._source_file(root / "missing", nested_item)
            alias_root = root / "alias-root"
            alias_root.mkdir()
            (alias_root / "references").symlink_to(skill / "references", target_is_directory=True)
            aliased_bundle = InterventionBundle(alias_root, source_bundle.manifest)
            with self.assertRaises(RuntimeError):
                artifact_module._source_file(aliased_bundle.root or root, nested_item)
            missing_item = InterventionFile("references/missing.md", 1, "0" * 64)
            with self.assertRaises(RuntimeError):
                artifact_module._source_file(skill, missing_item)
            with self.assertRaises(ValueError):
                artifact_module._source_entries(InterventionBundle(None, source_bundle.manifest), (nested_item,))

            destination = root / "copy.bin"
            with patch.object(os, "fsync", side_effect=OSError("fsync boundary")):
                with self.assertRaises(OSError):
                    artifact_module._copy_exclusive(destination, b"copy")
            self.assertFalse(destination.exists())
            with self.assertRaises(RuntimeError):
                artifact_module._verify_destination(root / "missing.bin", nested_item, b"guide")

            with self.assertRaises(ValueError):
                artifact_module._validate_sealed_bundle(
                    source_bundle,
                    InterventionBundle(None, source_bundle.manifest),
                    root / "artifact",
                )
            artifact_root = root / "artifact-root"
            artifact_root.mkdir()
            with self.assertRaises(ValueError):
                artifact_module._validate_sealed_bundle(
                    source_bundle,
                    source_bundle,
                    artifact_root,
                )
            with self.assertRaises(ValueError):
                artifact_module._validate_sealed_bundle(
                    source_bundle,
                    InterventionBundle(source_bundle.root, replace(source_bundle.manifest, intervention_id="other")),
                    source_bundle.root or root,
                )
            artifact_module._cleanup([root / "not-created"], [root / "not-created-dir"], root / "artifact")

            invalid_deliverables = root / "invalid-deliverables"
            invalid_deliverables.mkdir()
            invalid_skill = invalid_deliverables / "invalid-skill"
            invalid_skill.mkdir()
            (invalid_skill / "SKILL.md").write_text("invalid", encoding="utf-8")
            with patch.object(
                artifact_module,
                "load_agent_skill_bundle",
                side_effect=GeneratedSkillValidationError("loader validation"),
            ):
                with self.assertRaises(GeneratedSkillValidationError):
                    artifact_module.seal_generated_skill(invalid_deliverables, root / "invalid-artifact")

            final_tamper_deliverables = root / "final-tamper-deliverables"
            final_tamper_deliverables.mkdir()
            final_skill = final_tamper_deliverables / "final-skill"
            final_skill.mkdir()
            final_file = final_skill / "SKILL.md"
            final_file.write_text("---\nname: final-skill\ndescription: final\n---\n\ncontent\n", encoding="utf-8")
            real_copy = artifact_module._copy_exclusive

            def copy_then_tamper(destination_path: Path, content: bytes) -> None:
                real_copy(destination_path, content)
                final_file.write_text(
                    "---\nname: final-skill\ndescription: changed\n---\n\ncontent\n", encoding="utf-8"
                )

            with patch.object(artifact_module, "_copy_exclusive", side_effect=copy_then_tamper):
                with self.assertRaises(ArtifactHandoffError):
                    artifact_module.seal_generated_skill(final_tamper_deliverables, root / "final-artifact")
            self.assertFalse((root / "final-artifact").exists())

    def test_prompt_and_executor_builder_narrow_contracts_fail_closed(self) -> None:
        for targets in (
            "reference_files/builder-inputs/input-001",
            ("reference_files/builder-inputs/input-001", "reference_files/builder-inputs/input-001"),
            ("reference_files/builder-inputs/input-002", "reference_files/builder-inputs/input-001"),
            ("reference_files/builder-inputs/input-000",),
            ("reference_files/builder-inputs/input-01",),
        ):
            with self.subTest(targets=targets), self.assertRaises((TypeError, ValueError)):
                prompt_module._normalize_input_targets(targets)
        self.assertIn(
            "no creation input directories",
            prompt_module.build_skill_task(TaskSpec("task", "prompt"), ()).prompt,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            executor = ReliabilityExecutor()
            builder = ExecutorSkillBuilder(executor)
            self.assertTrue(builder.preflight().ok)
            with patch.object(executor, "preflight", side_effect=RuntimeError("private preflight")):
                failed = ExecutorSkillBuilder(executor).preflight()
            self.assertFalse(failed.ok)
            self.assertEqual(failed.details, ("executor preflight raised RuntimeError",))

            with patch.object(executor, "preflight", return_value="invalid"):
                invalid = ExecutorSkillBuilder(executor).preflight()
            self.assertFalse(invalid.ok)
            self.assertEqual(invalid.details, ("executor preflight returned an invalid result",))

            execution_failure = ExecutorSkillBuilder(
                ReliabilityExecutor(raise_on_execute=RuntimeError("external executor failure"))
            ).build(request)
            self.assertEqual(execution_failure.status, BuildStatus.FAILED)
            self.assertEqual(execution_failure.failure_phase, BuildFailurePhase.EXECUTION)
            self.assertIsNone(execution_failure.execution)
            self.assertTrue((root / "control" / "runtime" / "executor" / "stdout.log").is_file())

            wrong_task = ExecutorSkillBuilder(ReliabilityExecutor(result_task_id="wrong-task")).build(
                replace(
                    request,
                    runtime_root=root / "control" / "runtime-wrong-task",
                    artifact_root=root / "control" / "artifact-wrong-task",
                )
            )
            self.assertEqual(wrong_task.failure_phase, BuildFailurePhase.EXECUTION)
            wrong_executor = ExecutorSkillBuilder(ReliabilityExecutor(result_executor="other")).build(
                replace(
                    request,
                    runtime_root=root / "control" / "runtime-wrong-executor",
                    artifact_root=root / "control" / "artifact-wrong-executor",
                )
            )
            self.assertEqual(wrong_executor.failure_phase, BuildFailurePhase.EXECUTION)

    def test_executor_skill_builder_success_preserves_task_identity_and_isolates_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            source_bytes = {
                path.relative_to(root / "source").as_posix(): path.read_bytes()
                for path in (root / "source").rglob("*")
                if path.is_file()
            }
            executor = ReliabilityExecutor()
            with patch.dict(
                os.environ,
                {
                    "GDPVAL_CONDITION": "private-condition",
                    "GDPVAL_CONDITION_FILE": "private-condition-file",
                    "GDPVAL_CONDITION_APPLIED": "private-condition-applied",
                },
                clear=False,
            ):
                result = ExecutorSkillBuilder(executor).build(request)

            self.assertEqual(result.status, BuildStatus.COMPLETED)
            self.assertEqual(result.task_id, request.task.task_id)
            self.assertIsNotNone(result.execution)
            self.assertIsNotNone(result.bundle)
            self.assertEqual(len(executor.requests), 1)
            execution_request = executor.requests[0]
            self.assertEqual(execution_request.task.task_id, request.task.task_id)
            self.assertIn("create a reusable skill", execution_request.task.prompt)
            self.assertNotIn("input-one", execution_request.task.prompt)
            self.assertNotIn("private-condition", json.dumps(execution_request.environment))
            self.assertNotIn("GDPVAL_CONDITION", execution_request.environment)
            self.assertEqual(
                {
                    path.relative_to(root / "source").as_posix(): path.read_bytes()
                    for path in (root / "source").rglob("*")
                    if path.is_file()
                },
                source_bytes,
            )
            bundle = result.bundle
            assert bundle is not None
            bundle_root = bundle.root
            assert bundle_root is not None
            artifact_root = request.artifact_root
            assert artifact_root is not None
            self.assertEqual(bundle_root.parent, artifact_root.resolve())
            self.assertTrue((bundle_root / "SKILL.md").is_file())
            self.assertTrue((request.runtime_root / "workspace" / "reference_files").is_dir())
            self.assertTrue((request.runtime_root / "executor" / "stdout.log").is_file())
            self.assertFalse((artifact_root / "unexpected").exists())

    def test_executor_skill_builder_maps_executor_statuses_without_publishing_artifacts(self) -> None:
        cases = (
            (ExecutionStatus.FAILED, BuildStatus.FAILED),
            (ExecutionStatus.TIMED_OUT, BuildStatus.TIMED_OUT),
            (ExecutionStatus.INTERRUPTED, BuildStatus.INTERRUPTED),
            (ExecutionStatus.NO_DELIVERABLE, BuildStatus.NO_ARTIFACT),
        )
        for execution_status, expected_status in cases:
            with self.subTest(execution_status=execution_status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                executor = ReliabilityExecutor(status=execution_status)
                result = ExecutorSkillBuilder(executor).build(self._request(root))
                self.assertEqual(result.status, expected_status)
                self.assertEqual(
                    result.failure_phase,
                    BuildFailurePhase.EXECUTION
                    if execution_status
                    in {ExecutionStatus.FAILED, ExecutionStatus.TIMED_OUT, ExecutionStatus.INTERRUPTED}
                    else BuildFailurePhase.ARTIFACT_VALIDATION,
                )
                self.assertIsNotNone(result.execution)
                self.assertTrue(executor.requests)
                self.assertTrue((root / "control" / "runtime").is_dir())
                self.assertFalse((root / "control" / "artifact").exists())

    def test_executor_skill_builder_fails_before_execution_for_preflight_and_root_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight_executor = ReliabilityExecutor(preflight_ok=False)
            request = self._request(root)
            preflight_failure = ExecutorSkillBuilder(preflight_executor).build(request)
            self.assertEqual(preflight_failure.status, BuildStatus.FAILED)
            self.assertEqual(preflight_failure.failure_phase, BuildFailurePhase.PREFLIGHT)
            self.assertFalse(preflight_executor.requests)
            self.assertFalse(request.runtime_root.exists())
            self.assertFalse(request.artifact_root.exists())

            exception_executor = ReliabilityExecutor(preflight_exception=RuntimeError("preflight boundary"))
            exception_request = self._request(root / "exception")
            exception_failure = ExecutorSkillBuilder(exception_executor).build(exception_request)
            self.assertEqual(exception_failure.failure_phase, BuildFailurePhase.PREFLIGHT)
            self.assertFalse(exception_executor.requests)

            existing_request = self._request(root / "existing")
            existing_request.runtime_root.mkdir()
            existing_failure = ExecutorSkillBuilder(ReliabilityExecutor()).build(existing_request)
            self.assertEqual(existing_failure.failure_phase, BuildFailurePhase.INPUT_VALIDATION)
            self.assertFalse((root / "existing" / "control" / "artifact").exists())

            overlap_request = self._request(root / "overlap")
            overlap_request = replace(
                overlap_request,
                runtime_root=root / "overlap" / "control" / "same",
                artifact_root=root / "overlap" / "control" / "same",
            )
            overlap_executor = ReliabilityExecutor()
            overlap_failure = ExecutorSkillBuilder(overlap_executor).build(overlap_request)
            self.assertEqual(overlap_failure.failure_phase, BuildFailurePhase.INPUT_VALIDATION)
            self.assertFalse(overlap_executor.requests)

    def test_executor_skill_builder_rejects_execution_identity_and_staged_input_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_workspace = root / "wrong-workspace"
            wrong_workspace.mkdir()
            result = ExecutorSkillBuilder(ReliabilityExecutor(result_workspace=wrong_workspace)).build(
                self._request(root)
            )
            self.assertEqual(result.status, BuildStatus.FAILED)
            self.assertEqual(result.failure_phase, BuildFailurePhase.EXECUTION)
            self.assertIsNotNone(result.execution)

            root = Path(temporary) / "wrong-deliverables"
            root.mkdir()
            wrong_deliverables = root / "wrong-deliverables-output"
            wrong_deliverables.mkdir()
            result = ExecutorSkillBuilder(ReliabilityExecutor(result_deliverables=wrong_deliverables)).build(
                self._request(root)
            )
            self.assertEqual(result.failure_phase, BuildFailurePhase.EXECUTION)

            tamper_root = Path(temporary) / "tampered"
            tamper_executor = ReliabilityExecutor()
            original_execute = tamper_executor.execute

            def execute_then_tamper(execution_request: ExecutionRequest) -> ExecutionResult:
                execution = original_execute(execution_request)
                staged_file = execution_request.workspace / "reference_files/builder-inputs/input-001/a.txt"
                staged_file.chmod(0o644)
                staged_file.write_bytes(b"tampered")
                return execution

            with patch.object(tamper_executor, "execute", side_effect=execute_then_tamper):
                result = ExecutorSkillBuilder(tamper_executor).build(self._request(tamper_root))
            self.assertEqual(result.status, BuildStatus.INVALID_ARTIFACT)
            self.assertEqual(result.failure_phase, BuildFailurePhase.ARTIFACT_VALIDATION)
            self.assertFalse((tamper_root / "control" / "artifact").exists())

    def test_executor_skill_builder_maps_artifact_validation_and_handoff_failures(self) -> None:
        cases = (
            (
                GeneratedSkillValidationError("invalid generated skill"),
                BuildStatus.INVALID_ARTIFACT,
                BuildFailurePhase.ARTIFACT_VALIDATION,
            ),
            (ArtifactHandoffError("handoff failed"), BuildStatus.FAILED, BuildFailurePhase.ARTIFACT_HANDOFF),
            (FileExistsError("artifact already exists"), BuildStatus.FAILED, BuildFailurePhase.ARTIFACT_HANDOFF),
            (RuntimeError("unexpected handoff failure"), BuildStatus.FAILED, BuildFailurePhase.ARTIFACT_HANDOFF),
            (None, BuildStatus.FAILED, BuildFailurePhase.ARTIFACT_HANDOFF),
        )
        for error, expected_status, expected_phase in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                executor = ReliabilityExecutor()
                seal_result = (
                    None
                    if error is None
                    else patch.object(
                        executor_skill_module,
                        "seal_generated_skill",
                        side_effect=error,
                    )
                )
                context = (
                    seal_result
                    if seal_result is not None
                    else patch.object(
                        executor_skill_module,
                        "seal_generated_skill",
                        return_value=None,
                    )
                )
                with context:
                    result = ExecutorSkillBuilder(executor).build(self._request(root))
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.failure_phase, expected_phase)
                self.assertIsNotNone(result.execution)
                self.assertFalse((root / "control" / "artifact").exists())


if __name__ == "__main__":
    unittest.main()
