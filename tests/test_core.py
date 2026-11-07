# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests for calibration and model primitives."""

from __future__ import annotations

import unittest

import numpy as np

from cpo_stcm.conformal import (
    block_conformal_p_values,
    mondrian_conformal_p_values,
    observability_stratum,
    profile_mondrian_conformal_p_values,
    source_balanced_conformal_p_values,
)
from cpo_stcm.evaluation import (
    binary_metrics,
    conformal_p_values,
    empirical_two_sided_anomaly_evidence,
    empirical_upper_tail_anomaly_evidence,
)
from cpo_stcm.static_evidence import crop_array_subimage


class EvaluationTests(unittest.TestCase):
    """Test score orientation and metric helpers."""

    def test_upper_tail_evidence_is_high_for_anomalies(self) -> None:
        reference = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        values = np.asarray([0.2, 0.9], dtype=np.float64)
        evidence = empirical_upper_tail_anomaly_evidence(reference, values)
        self.assertLess(evidence[0], evidence[1])
        self.assertAlmostEqual(float(evidence[1]), 0.8)

    def test_two_sided_evidence_marks_both_tails(self) -> None:
        reference = np.linspace(0.0, 1.0, 101)
        values = np.asarray([0.5, -1.0, 2.0], dtype=np.float64)
        evidence = empirical_two_sided_anomaly_evidence(reference, values)
        self.assertLess(evidence[0], evidence[1])
        self.assertLess(evidence[0], evidence[2])

    def test_conformal_p_values_are_monotone(self) -> None:
        calibration = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
        values = np.asarray([0.05, 0.2, 0.4], dtype=np.float64)
        p_values = conformal_p_values(calibration, values)
        self.assertGreater(p_values[0], p_values[1])
        self.assertGreater(p_values[1], p_values[2])

    def test_binary_metrics_shapes(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
        metrics = binary_metrics(labels, scores, 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertAlmostEqual(float(metrics["recall"]), 1.0)

    def test_observability_strata(self) -> None:
        self.assertEqual(observability_stratum(0.0), 0)
        self.assertEqual(observability_stratum(0.75), 1)
        self.assertEqual(observability_stratum(1.0), 2)

    def test_mondrian_backoff_and_source_balance(self) -> None:
        calibration = np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float64)
        strata = np.asarray([0, 0, 1, 1], dtype=np.int64)
        values = np.asarray([0.3, 0.95], dtype=np.float64)
        query_strata = np.asarray([0, 1], dtype=np.int64)
        p_values = mondrian_conformal_p_values(
            calibration,
            strata,
            values,
            query_strata,
            minimum_stratum_size=2,
        )
        self.assertEqual(len(p_values), 2)
        weighted = source_balanced_conformal_p_values(
            calibration,
            ["a", "a", "b", "b"],
            values,
        )
        self.assertTrue(np.all((weighted > 0) & (weighted <= 1)))
        block_p = block_conformal_p_values(
            calibration,
            ["a", "a", "b", "b"],
            np.asarray([0.95], dtype=np.float64),
        )
        self.assertEqual(block_p.shape, (1,))

    def test_profile_mondrian_uses_exact_then_stratum_backoff(self) -> None:
        calibration_scores = np.asarray(
            [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float64,
        )
        calibration_strata = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        calibration_profiles = np.asarray(
            ["A", "A", "A", "A", "B", "B", "B", "B"]
        )
        p_values = profile_mondrian_conformal_p_values(
            calibration_scores,
            calibration_strata,
            calibration_profiles,
            test_scores=np.asarray([0.3, 0.3]),
            test_strata=np.asarray([0, 0]),
            test_profiles=np.asarray(["A", "unseen"]),
            minimum_stratum_size=2,
            minimum_profile_stratum_size=2,
        )

        self.assertAlmostEqual(float(p_values[0]), 1.0 / 3.0)
        self.assertAlmostEqual(float(p_values[1]), 0.6)


class StaticEvidenceTests(unittest.TestCase):
    """Test array slicing for non-square geometries."""

    def test_crop_array_subimage(self) -> None:
        frame = np.arange(2 * 8 * 6, dtype=np.uint8).reshape(8, 12)
        crop = crop_array_subimage(
            frame,
            subimage_index=5,
            array_rows=2,
            array_columns=3,
            crop_height=2,
            crop_width=2,
        )
        self.assertEqual(crop.shape, (2, 2))
        expected = frame[5:7, 9:11]
        np.testing.assert_array_equal(crop, expected)


if __name__ == "__main__":
    unittest.main()
