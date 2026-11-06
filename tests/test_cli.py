# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for dependency-light command-line behavior."""

from __future__ import annotations

import contextlib
import io
import json
import unittest

from cpo_stcm.cli import main


class CliTests(unittest.TestCase):
    """Validate deterministic commands that do not require model assets."""

    def test_demo_emits_valid_json(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exit_code = main(["demo", "--calibration-size", "32"])
        payload = json.loads(stream.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["calibration_size"], 32)
        self.assertEqual(len(payload["p_values"]), 3)

    def test_version_emits_package_version(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exit_code = main(["version"])
        self.assertEqual(exit_code, 0)
        self.assertRegex(stream.getvalue().strip(), r"^\d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
