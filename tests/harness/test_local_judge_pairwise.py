# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gdpval_harness.judges.base import Verdict
from gdpval_harness.judges.pairwise import (
    aggregate,
    matched_tasks,
    normalize_verdict,
    parse_verdict,
    prepare_trial,
    validate_reference_equivalence,
)


class LocalPairwiseTests(unittest.TestCase):
    def _candidate(self, root: Path, name: str, *, task: str = "task_x", ref: str = "same") -> Path:
        candidate = root / name
        repeat = candidate / task / "repeat_0"
        (repeat / "reference_files").mkdir(parents=True)
        (repeat / "reference_files" / "ref.txt").write_text(ref)
        (repeat / "artifact.txt").write_text(name)
        (repeat / "finish_params.json").write_text("{}")
        return candidate

    def test_task_sets_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._candidate(root, "a", task="task_a")
            b = self._candidate(root, "b", task="task_b")
            with self.assertRaises(ValueError):
                matched_tasks(a, b)

    def test_reference_files_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._candidate(root, "a", ref="one") / "task_x" / "repeat_0"
            b = self._candidate(root, "b", ref="two") / "task_x" / "repeat_0"
            with self.assertRaises(ValueError):
                validate_reference_equivalence(a, b)

    def test_symlinked_repeat_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            (outside / "artifact.txt").write_text("secret")
            candidate = root / "candidate"
            task = candidate / "task_x"
            task.mkdir(parents=True)
            (task / "repeat_0").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked repeat"):
                matched_tasks(candidate, candidate)

    def test_symlinked_reference_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._candidate(root, "a") / "task_x" / "repeat_0"
            b = self._candidate(root, "b") / "task_x" / "repeat_0"
            outside = root / "outside-ref"
            outside.mkdir()
            (outside / "ref.txt").write_text("same")
            for candidate in (a, b):
                ref = candidate / "reference_files"
                for child in ref.iterdir():
                    child.unlink()
                ref.rmdir()
                ref.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked reference"):
                validate_reference_equivalence(a, b)

    def test_trial_workspace_is_anonymous_and_omits_bookkeeping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_root = self._candidate(root, "plain-secret-label")
            b_root = self._candidate(root, "alps-secret-label")
            _, a, b = matched_tasks(a_root, b_root)[0]
            trial = prepare_trial(root / "out", "task_x", a, b, trial_index=0, seed=42)
            self.assertTrue((trial.submission_a_dir / "artifact.txt").is_file())
            self.assertTrue((trial.submission_b_dir / "artifact.txt").is_file())
            self.assertFalse((trial.submission_a_dir / "finish_params.json").exists())
            self.assertFalse((trial.submission_b_dir / "reference_files").exists())
            self.assertNotIn("plain-secret-label", str(trial.workspace))
            self.assertNotIn("alps-secret-label", str(trial.workspace))

    def test_trial_order_alternates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_root = self._candidate(root, "a")
            b_root = self._candidate(root, "b")
            _, a, b = matched_tasks(a_root, b_root)[0]
            first = prepare_trial(root / "out", "task_x", a, b, trial_index=0, seed=9)
            second = prepare_trial(root / "out", "task_x", a, b, trial_index=1, seed=9)
            self.assertNotEqual(first.swapped, second.swapped)

    def test_verdict_parser_and_normalization(self) -> None:
        self.assertEqual(parse_verdict("reason\nBOXED[A]"), Verdict.A)
        self.assertEqual(normalize_verdict(Verdict.A, True), Verdict.B)
        self.assertEqual(normalize_verdict(Verdict.TIE, True), Verdict.TIE)
        with self.assertRaises(ValueError):
            parse_verdict("BOXED[A]\nBOXED[B]")
        with self.assertRaises(ValueError):
            parse_verdict("reason\nUNBOXED[A]")
        with self.assertRaises(ValueError):
            parse_verdict("BOXED[A]\ntrailing commentary")

    def test_aggregation_preserves_ties(self) -> None:
        result = aggregate([Verdict.A, Verdict.B, Verdict.TIE, Verdict.A])
        self.assertEqual(result["wins_a"], 2)
        self.assertEqual(result["wins_b"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["score_a"], 0.625)


if __name__ == "__main__":
    unittest.main()
