# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for blinded directional-order annotation sampling."""

from __future__ import annotations

import unittest

from cpo_stcm.annotation import (
    requires_second_review,
    select_annotation_candidates,
)


class AnnotationSelectionTests(unittest.TestCase):
    """Validate deterministic and non-overlapping review sampling."""

    def setUp(self) -> None:
        self.rows = [
            {
                "sample_id": f"sample_{index:03d}",
                "source": "charge",
                "role": "test" if index < 20 else "domain_test",
                "subclass": "animated",
                "transform": "none",
            }
            for index in range(30)
        ]
        self.scores = {
            row["sample_id"]: float(index)
            for index, row in enumerate(self.rows)
        }

    def test_selection_is_deterministic_and_disjoint(self) -> None:
        first = select_annotation_candidates(
            self.rows,
            scores=self.scores,
            seed=20261105,
            random_count=10,
            enriched_count=10,
        )
        second = select_annotation_candidates(
            self.rows,
            scores=self.scores,
            seed=20261105,
            random_count=10,
            enriched_count=10,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({row["sample_id"] for row in first}), 20)
        self.assertEqual(
            {row["sampling_stratum"] for row in first},
            {"random_audit", "model_enriched"},
        )

    def test_selection_excludes_transformed_and_non_animation_rows(self) -> None:
        rows = self.rows + [
            {
                "sample_id": "transformed",
                "source": "charge",
                "role": "test",
                "subclass": "temporal_reverse",
                "transform": "reverse",
            },
            {
                "sample_id": "missing_device",
                "source": "charge",
                "role": "test",
                "subclass": "no_device",
                "transform": "none",
            },
        ]
        selected = select_annotation_candidates(
            rows,
            scores=self.scores,
            seed=20261105,
            random_count=5,
            enriched_count=5,
        )

        self.assertNotIn(
            "transformed",
            {row["sample_id"] for row in selected},
        )
        self.assertNotIn(
            "missing_device",
            {row["sample_id"] for row in selected},
        )

    def test_second_review_assignment_is_not_biased_by_selection_hash(
        self,
    ) -> None:
        flags = [
            requires_second_review(
                20261105,
                f"{index:016x}",
                fraction=0.20,
            )
            for index in range(2000)
        ]

        self.assertAlmostEqual(sum(flags) / len(flags), 0.20, delta=0.03)


if __name__ == "__main__":
    unittest.main()
