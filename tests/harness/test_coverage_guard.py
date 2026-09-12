# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from scripts.ci.run_eval_harness_coverage import coverage_summary_passes


class CoverageGuardTests(unittest.TestCase):
    def test_rejects_just_below_integer_boundary(self) -> None:
        self.assertFalse(coverage_summary_passes({"covered_lines": 6440, "num_statements": 6709}))

    def test_accepts_minimum_passing_integer_boundary(self) -> None:
        self.assertTrue(coverage_summary_passes({"covered_lines": 6441, "num_statements": 6709}))
