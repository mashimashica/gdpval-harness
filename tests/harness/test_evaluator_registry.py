# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from eval_harness.evaluators.base import EvaluatorType
from eval_harness.evaluators.registry import get_evaluator_descriptor


class EvaluatorRegistryTests(unittest.TestCase):
    def test_native_evaluator_versions_and_assets_are_advertised(self) -> None:
        aime = get_evaluator_descriptor("aime26")
        self.assertEqual(aime.version, "0.8.0")
        self.assertEqual(aime.requirements, ("math-verify==0.8.0",))
        self.assertEqual(aime.assets, ("resources_servers/math_with_judge/requirements.txt",))

        bigcode = get_evaluator_descriptor("bigcodebench")
        self.assertEqual(bigcode.version, "1")
        self.assertEqual(bigcode.assets, ("resources_servers/bigcodebench/.bcb_venv",))
        self.assertEqual(bigcode.evaluator_type, EvaluatorType.EXECUTABLE_TESTS)

    def test_gdpval_descriptor_is_external_and_has_no_default_judge(self) -> None:
        gdpval = get_evaluator_descriptor("gdpval")
        self.assertIn("rubric/pairwise evaluation remains external", gdpval.status)
        self.assertEqual(gdpval.judge, "existing GDPval rubric/pairwise path")


if __name__ == "__main__":
    unittest.main()
