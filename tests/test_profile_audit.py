# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for capture-profile calibration coverage auditing."""

from __future__ import annotations

import unittest

from cpo_stcm.profile_audit import summarize_capture_profiles


class ProfileAuditTests(unittest.TestCase):
    """Validate sequence-level coverage accounting."""

    def test_roles_count_unique_sequences_per_profile(self) -> None:
        rows = [
            {
                "batch_id": "A",
                "sequence_id": "A/1",
                "role": "normal_calibration",
            },
            {
                "batch_id": "A",
                "sequence_id": "A/1",
                "role": "normal_calibration",
            },
            {
                "batch_id": "A",
                "sequence_id": "A/2",
                "role": "normal_calibration",
            },
            {
                "batch_id": "B",
                "sequence_id": "B/1",
                "role": "domain_test",
            },
        ]

        summary = summarize_capture_profiles(
            rows,
            batch_to_profile={"A": "standard", "B": "profile_b"},
            default_profile="standard",
            minimum_sequences=2,
        )

        self.assertEqual(
            summary["profiles"]["standard"]["roles"][
                "normal_calibration"
            ]["windows"],
            3,
        )
        self.assertEqual(
            summary["profiles"]["standard"]["roles"][
                "normal_calibration"
            ]["sequences"],
            2,
        )
        self.assertTrue(
            summary["profiles"]["standard"]["calibration_ready"]
        )
        self.assertFalse(
            summary["profiles"]["profile_b"]["calibration_ready"]
        )
        self.assertEqual(
            summary["unseen_test_profiles"],
            ["profile_b"],
        )


if __name__ == "__main__":
    unittest.main()
