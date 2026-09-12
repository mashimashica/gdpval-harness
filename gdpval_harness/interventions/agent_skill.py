# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent Skills validation and explicit workspace-reference application.

This module implements the portable reference path.  It materializes a
validated skill below ``.gdpval/interventions`` and gives the executor an
explicit relative path.  It deliberately does not stage native Codex,
Claude, or Cursor discovery directories.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from gdpval_harness.executors.base import TaskSpec
from gdpval_harness.interventions.base import (
    ApplicationMapping,
    Intervention,
    InterventionApplication,
    InterventionBundle,
    InterventionFile,
    InterventionManifest,
    InterventionPreflightResult,
    InterventionType,
    compute_bundle_sha256,
    ensure_destination_parents,
    ensure_source_output_separation,
    file_evidence,
    fsync_directory,
    revision_fields,
)


_SKILL_FRONTMATTER_FIELDS = frozenset({"name", "description", "license", "compatibility", "metadata", "allowed-tools"})
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_SKILL_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_MAX_COMPATIBILITY_LENGTH = 500
_SKILL_TARGET_PREFIX = ".gdpval/interventions/"
_SKILL_TARGET_SUFFIX = "/SKILL.md"
_SKILL_PROMPT_BEGIN = "[BEGIN AGENT SKILL WORKSPACE REFERENCE]"
_SKILL_PROMPT_END = "[END AGENT SKILL WORKSPACE REFERENCE]"


def _validate_relative_path(path: str) -> None:
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Agent Skill path must be strict UTF-8: {path!r}") from exc
    pure = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        not path
        or "\\" in path
        or pure.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"Agent Skill contains an unsafe relative path: {path!r}")


def _collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _read_skill_source(source: Path) -> tuple[Path, tuple[tuple[str, bytes], ...]]:
    if source.is_symlink():
        raise ValueError("Agent Skill source root must not be a symlink")
    try:
        root = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Agent Skill source root is unavailable: {exc}") from exc
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Agent Skill source root must be an existing non-symlink directory")

    entries: list[tuple[str, bytes]] = []
    seen: dict[str, str] = {}

    def visit(directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise ValueError(f"could not read Agent Skill source directory: {exc}") from exc
        for child in children:
            relative = child.relative_to(root).as_posix()
            _validate_relative_path(relative)
            key = _collision_key(relative)
            previous = seen.get(key)
            if previous is not None and previous != relative:
                raise ValueError(f"Agent Skill has a Unicode/casefold path collision: {previous!r}")
            seen[key] = relative
            if child.is_symlink():
                raise ValueError(f"Agent Skill contains a symlink: {relative!r}")
            child_stat = child.lstat()
            if stat.S_ISDIR(child_stat.st_mode):
                visit(child)
            elif stat.S_ISREG(child_stat.st_mode):
                try:
                    content = child.read_bytes()
                except OSError as exc:
                    raise ValueError(f"could not read Agent Skill file: {relative!r}") from exc
                entries.append((relative, content))
            else:
                raise ValueError(f"Agent Skill contains a special file: {relative!r}")

    visit(root)
    if not any(path == "SKILL.md" for path, _ in entries):
        raise ValueError("Agent Skill source must contain an exact regular SKILL.md")
    return root, tuple(sorted(entries, key=lambda entry: entry[0]))


def _parse_frontmatter(content: bytes) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SKILL.md must be strict UTF-8") from exc
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError("SKILL.md must begin with a YAML frontmatter delimiter")
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError("SKILL.md YAML frontmatter is missing its closing delimiter")
    frontmatter_text = "".join(lines[1:closing_index])
    try:
        parsed = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"SKILL.md frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    if any(not isinstance(key, str) for key in parsed):
        raise ValueError("SKILL.md frontmatter keys must be strings")
    unsupported = sorted(set(parsed) - _SKILL_FRONTMATTER_FIELDS)
    if unsupported:
        raise ValueError(f"SKILL.md has unsupported frontmatter fields: {', '.join(unsupported)}")
    return parsed


def _required_string(metadata: Mapping[str, Any], key: str, *, maximum: int | None = None) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SKILL.md frontmatter field {key!r} must be nonblank text")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"SKILL.md frontmatter field {key!r} exceeds {maximum} characters")
    return value


def _skill_name(root: Path, metadata: Mapping[str, Any]) -> str:
    name = _required_string(metadata, "name", maximum=_MAX_SKILL_NAME_LENGTH)
    if not _SKILL_NAME_RE.fullmatch(name):
        raise ValueError("SKILL.md name must use lowercase ASCII letters, digits, and single hyphens")
    if name != root.name:
        raise ValueError("SKILL.md name must exactly match the skill directory name")
    return name


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    _required_string(metadata, "description", maximum=_MAX_DESCRIPTION_LENGTH)
    if "license" in metadata:
        _required_string(metadata, "license")
    if "compatibility" in metadata:
        _required_string(metadata, "compatibility", maximum=_MAX_COMPATIBILITY_LENGTH)
    if "metadata" in metadata:
        values = metadata["metadata"]
        if not isinstance(values, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in values.items()
        ):
            raise ValueError("SKILL.md metadata must map string keys to string values")
    if "allowed-tools" in metadata:
        allowed_tools = metadata["allowed-tools"]
        if not isinstance(allowed_tools, str) or not allowed_tools.strip() or not allowed_tools.split():
            raise ValueError("SKILL.md allowed-tools must be a nonblank space-separated string")


def _build_bundle(
    source: Path | str,
    *,
    intervention_id: str | None = None,
    source_revision: str | None = None,
) -> InterventionBundle:
    root, entries = _read_skill_source(Path(source))
    skill_content = dict(entries)["SKILL.md"]
    frontmatter = _parse_frontmatter(skill_content)
    skill_name = _skill_name(root, frontmatter)
    _validate_metadata(frontmatter)
    source_revision, revision_status = revision_fields(source_revision, applicable=True)
    manifest_files = tuple(file_evidence(path, content) for path, content in entries)
    manifest = InterventionManifest(
        intervention_id=intervention_id or skill_name,
        intervention_type=InterventionType.AGENT_SKILL,
        source_revision=source_revision,
        revision_status=revision_status,
        files=manifest_files,
        bundle_sha256=compute_bundle_sha256(entries),
        application=ApplicationMapping(
            method="workspace-reference",
            target=f"{_SKILL_TARGET_PREFIX}{skill_name}{_SKILL_TARGET_SUFFIX}",
        ),
    )
    return InterventionBundle(root=root, manifest=manifest)


def load_agent_skill_bundle(
    source: Path | str,
    *,
    intervention_id: str | None = None,
    source_revision: str | None = None,
) -> InterventionBundle:
    """Load one validated Agent Skill directory into an immutable bundle."""

    try:
        return _build_bundle(source, intervention_id=intervention_id, source_revision=source_revision)
    except (OSError, TypeError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("invalid Agent Skill bundle:"):
            raise
        raise ValueError(f"invalid Agent Skill bundle: {exc}") from exc


def _skill_name_from_manifest(manifest: InterventionManifest) -> str:
    target = manifest.application.target
    if (
        not isinstance(target, str)
        or not target.startswith(_SKILL_TARGET_PREFIX)
        or not target.endswith(_SKILL_TARGET_SUFFIX)
    ):
        raise ValueError("Agent Skill manifest has an invalid workspace-reference target")
    name = target[len(_SKILL_TARGET_PREFIX) : -len(_SKILL_TARGET_SUFFIX)]
    if not name or "/" in name or "\\" in name or not _SKILL_NAME_RE.fullmatch(name):
        raise ValueError("Agent Skill manifest target has an invalid skill name")
    return name


def _validated_bundle(bundle: InterventionBundle) -> tuple[InterventionBundle, str]:
    if not isinstance(bundle, InterventionBundle) or not isinstance(bundle.root, Path):
        raise ValueError("Agent Skill intervention requires a filesystem bundle")
    root = bundle.root
    if root.is_symlink() or not root.is_dir() or root != root.resolve(strict=True):
        raise ValueError("Agent Skill bundle root must be a canonical non-symlink directory")
    manifest = bundle.manifest
    if manifest.intervention_type is not InterventionType.AGENT_SKILL:
        raise ValueError("Agent Skill bundle manifest has the wrong intervention type")
    skill_name = _skill_name_from_manifest(manifest)
    expected = _build_bundle(
        root,
        intervention_id=manifest.intervention_id,
        source_revision=manifest.source_revision,
    )
    if expected.manifest != manifest:
        raise ValueError("Agent Skill bundle manifest does not match its source contents")
    if _skill_name_from_manifest(expected.manifest) != skill_name:
        raise ValueError("Agent Skill bundle target changed during validation")
    return expected, skill_name


def _validate_destination_collisions(workspace: Path, logical_paths: tuple[str, ...]) -> None:
    for logical_path in logical_paths:
        current = workspace
        for part in PurePosixPath(logical_path).parts:
            if not current.is_dir():
                break
            key = _collision_key(part)
            matches = [entry for entry in current.iterdir() if _collision_key(entry.name) == key]
            if len(matches) > 1 or (matches and matches[0].name != part):
                raise ValueError(f"Agent Skill destination has a Unicode/casefold collision: {part!r}")
            if not matches:
                break
            entry = matches[0]
            if entry.is_symlink():
                raise ValueError(f"Agent Skill destination parent is a symlink: {part!r}")
            if part != PurePosixPath(logical_path).parts[-1] and not entry.is_dir():
                raise ValueError(f"Agent Skill destination parent is not a directory: {part!r}")
            current = entry


def _mkdir_missing_parents(workspace: Path, logical_paths: tuple[str, ...], created: list[Path]) -> None:
    parents: set[Path] = set()
    for logical_path in logical_paths:
        current = workspace
        for part in PurePosixPath(logical_path).parts[:-1]:
            current = current / part
            parents.add(current)
    for parent in sorted(parents, key=lambda path: len(path.parts)):
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("Agent Skill destination parent is not a directory")
            continue
        try:
            parent.mkdir()
        except FileExistsError:
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("Agent Skill destination parent is not a directory") from None
        else:
            created.append(parent)


def _copy_exclusive(destination: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _skill_task(task: TaskSpec, skill_name: str) -> TaskSpec:
    skill_root = f".gdpval/interventions/{skill_name}"
    prefix = (
        f"{_SKILL_PROMPT_BEGIN}\n"
        f"Read and follow the Agent Skill instructions in `{skill_root}/SKILL.md`.\n"
        f"Resolve every relative file referenced by that SKILL.md from the Skill root `{skill_root}/`.\n"
        f"{_SKILL_PROMPT_END}\n\n"
    )
    return TaskSpec(task_id=task.task_id, prompt=prefix + task.prompt)


class AgentSkillIntervention(Intervention):
    """Apply a validated Agent Skill through an explicit workspace reference."""

    name = "agent-skill"
    intervention_type = InterventionType.AGENT_SKILL

    def __init__(self, bundle: InterventionBundle, *, source_reference: str | None = None) -> None:
        if source_reference is not None and (
            not isinstance(source_reference, str) or not source_reference.strip() or "\x00" in source_reference
        ):
            raise ValueError("Agent Skill source_reference must be nonblank text without NUL bytes")
        self.bundle = bundle
        self.source_reference = source_reference
        self._ready_bundle: InterventionBundle | None = None
        self._skill_name: str | None = None

    @classmethod
    def from_source(
        cls,
        source: Path | str,
        *,
        intervention_id: str | None = None,
        source_revision: str | None = None,
    ) -> "AgentSkillIntervention":
        bundle = load_agent_skill_bundle(
            source,
            intervention_id=intervention_id,
            source_revision=source_revision,
        )
        return cls(bundle, source_reference=str(bundle.root))

    def preflight(self) -> InterventionPreflightResult:
        self._ready_bundle = None
        self._skill_name = None
        try:
            bundle, skill_name = _validated_bundle(self.bundle)
        except (OSError, TypeError, ValueError) as exc:
            return InterventionPreflightResult(
                name=self.name,
                intervention_type=self.intervention_type,
                ok=False,
                details=(f"Agent Skill preflight failed: {exc}",),
            )
        self._ready_bundle = bundle
        self._skill_name = skill_name
        return InterventionPreflightResult(
            name=self.name,
            intervention_type=self.intervention_type,
            ok=True,
            bundle=bundle,
            details=("Agent Skill workspace-reference bundle is ready",),
        )

    def validate_task(self, task: TaskSpec) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError("intervention task must be a TaskSpec")

    def apply(self, task: TaskSpec, workspace: Path, *, application_run_id: str) -> InterventionApplication:
        self.validate_task(task)
        bundle = self._require_bundle()
        try:
            current, skill_name = _validated_bundle(bundle)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError("Agent Skill source changed after preflight") from exc
        if current.manifest != bundle.manifest or skill_name != self._skill_name:
            raise RuntimeError("Agent Skill source changed after preflight")

        workspace = Path(workspace)
        if workspace.is_symlink() or not workspace.is_dir():
            raise ValueError("Agent Skill workspace must be an existing non-symlink directory")
        try:
            workspace = workspace.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Agent Skill workspace must be an existing directory") from exc
        ensure_source_output_separation(bundle, workspace)

        prefix = PurePosixPath(".gdpval", "interventions", skill_name)
        logical_paths = tuple((prefix / PurePosixPath(item.path)).as_posix() for item in bundle.manifest.files)
        destination_root = workspace / prefix
        if destination_root.is_symlink() or destination_root.exists():
            raise FileExistsError("Agent Skill workspace-reference destination already exists")
        _validate_destination_collisions(workspace, logical_paths)
        ensure_destination_parents(workspace, logical_paths)

        entries = _source_entries_from_manifest(bundle.manifest, bundle.root)
        created_files: list[Path] = []
        created_dirs: list[Path] = []
        try:
            _mkdir_missing_parents(workspace, logical_paths, created_dirs)
            materialized: list[InterventionFile] = []
            for item, (_, content), logical_path in zip(bundle.manifest.files, entries, logical_paths, strict=True):
                destination = workspace / PurePosixPath(logical_path)
                _copy_exclusive(destination, content)
                created_files.append(destination)
                if destination.is_symlink() or destination.read_bytes() != content:
                    raise RuntimeError(f"Agent Skill destination verification failed: {logical_path!r}")
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                if digest != item.sha256:
                    raise RuntimeError(f"Agent Skill destination hash verification failed: {logical_path!r}")
                fsync_directory(destination.parent)
                materialized.append(file_evidence(logical_path, content))
            fsync_directory(workspace)
        except Exception:
            for destination in reversed(created_files):
                try:
                    destination.unlink()
                except OSError:
                    pass
            for directory in sorted(created_dirs, key=lambda path: len(path.parts), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            fsync_directory(workspace)
            raise

        return InterventionApplication(
            application_run_id=application_run_id,
            task=_skill_task(task, skill_name),
            materialized_files=tuple(materialized),
            bundle_sha256=bundle.manifest.bundle_sha256,
            manifest_sha256=bundle.manifest.manifest_sha256 or "",
            application=bundle.manifest.application,
        )

    def _require_bundle(self) -> InterventionBundle:
        if self._ready_bundle is None or self._skill_name is None:
            raise RuntimeError("successful Agent Skill preflight is required before apply")
        return self._ready_bundle


def _source_entries_from_manifest(manifest: InterventionManifest, root: Path) -> tuple[tuple[str, bytes], ...]:
    entries: list[tuple[str, bytes]] = []
    for item in manifest.files:
        path = root / PurePosixPath(item.path)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Agent Skill source changed after preflight")
        content = path.read_bytes()
        if len(content) != item.size or hashlib.sha256(content).hexdigest() != item.sha256:
            raise RuntimeError("Agent Skill source changed after preflight")
        entries.append((item.path, content))
    if compute_bundle_sha256(entries) != manifest.bundle_sha256:
        raise RuntimeError("Agent Skill source changed after preflight")
    return tuple(entries)


__all__ = ["AgentSkillIntervention", "load_agent_skill_bundle"]
