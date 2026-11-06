# SPDX-License-Identifier: AGPL-3.0-only
"""Block-aware and partially observed conformal calibration utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def observability_stratum(hard_pass_fraction: float) -> int:
    """Map required-slot observability into three Mondrian strata."""
    if hard_pass_fraction < 0.5:
        return 0
    if hard_pass_fraction < 0.999:
        return 1
    return 2


def mondrian_conformal_p_values(
    calibration_scores: np.ndarray,
    calibration_strata: np.ndarray,
    test_scores: np.ndarray,
    test_strata: np.ndarray,
    minimum_stratum_size: int = 20,
) -> np.ndarray:
    """Compute stratum-conditional conformal p-values with global backoff."""
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    calibration_strata = np.asarray(calibration_strata, dtype=np.int64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    test_strata = np.asarray(test_strata, dtype=np.int64)
    global_p = (
        1.0
        + np.sum(
            calibration_scores[:, None] >= test_scores[None, :],
            axis=0,
        )
    ) / (len(calibration_scores) + 1.0)
    output = global_p.copy()
    for stratum in np.unique(test_strata):
        reference = calibration_scores[
            calibration_strata == stratum
        ]
        query_indices = np.flatnonzero(test_strata == stratum)
        if len(reference) < minimum_stratum_size:
            continue
        output[query_indices] = (
            1.0
            + np.sum(
                reference[:, None] >= test_scores[query_indices][None, :],
                axis=0,
            )
        ) / (len(reference) + 1.0)
    return output


def profile_mondrian_conformal_p_values(
    calibration_scores: np.ndarray,
    calibration_strata: np.ndarray,
    calibration_profiles: Sequence[str],
    test_scores: np.ndarray,
    test_strata: np.ndarray,
    test_profiles: Sequence[str],
    minimum_stratum_size: int = 20,
    minimum_profile_stratum_size: int = 10,
) -> np.ndarray:
    """Calibrate by profile and observability with hierarchical backoff.

    Exact profile-stratum calibration is used only when enough normal
    calibration blocks exist. Otherwise the procedure backs off to the
    observability stratum and finally to the pooled global reference.
    """
    scores = np.asarray(calibration_scores, dtype=np.float64)
    strata = np.asarray(calibration_strata, dtype=np.int64)
    profiles = np.asarray([str(value) for value in calibration_profiles])
    queries = np.asarray(test_scores, dtype=np.float64)
    query_strata = np.asarray(test_strata, dtype=np.int64)
    query_profiles = np.asarray([str(value) for value in test_profiles])
    if len(scores) != len(strata) or len(scores) != len(profiles):
        raise ValueError("Calibration arrays must have equal lengths")
    if len(queries) != len(query_strata) or len(queries) != len(query_profiles):
        raise ValueError("Test arrays must have equal lengths")

    output = (
        1.0
        + np.sum(scores[:, None] >= queries[None, :], axis=0)
    ) / (len(scores) + 1.0)
    for stratum in np.unique(query_strata):
        stratum_reference = scores[strata == stratum]
        stratum_indices = np.flatnonzero(query_strata == stratum)
        if len(stratum_reference) >= minimum_stratum_size:
            output[stratum_indices] = (
                1.0
                + np.sum(
                    stratum_reference[:, None]
                    >= queries[stratum_indices][None, :],
                    axis=0,
                )
            ) / (len(stratum_reference) + 1.0)

        for profile in np.unique(query_profiles[stratum_indices]):
            indices = stratum_indices[
                query_profiles[stratum_indices] == profile
            ]
            reference = scores[
                (strata == stratum)
                & (profiles == profile)
            ]
            if len(reference) < minimum_profile_stratum_size:
                continue
            output[indices] = (
                1.0
                + np.sum(
                    reference[:, None] >= queries[indices][None, :],
                    axis=0,
                )
            ) / (len(reference) + 1.0)
    return output


def source_balanced_conformal_p_values(
    calibration_scores: np.ndarray,
    calibration_groups: Sequence[str],
    test_scores: np.ndarray,
) -> np.ndarray:
    """Compute conformal p-values with equal total weight per source block."""
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    groups = np.asarray([str(group) for group in calibration_groups])
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    group_sizes = np.bincount(group_indices)
    sample_weights = 1.0 / group_sizes[group_indices]
    total_weight = float(sample_weights.sum())
    minimum_weight = float(sample_weights.min())
    output = np.empty(len(test_scores), dtype=np.float64)
    for index, score in enumerate(test_scores):
        upper_mass = float(
            sample_weights[calibration_scores >= score].sum()
        )
        output[index] = (minimum_weight + upper_mass) / (
            total_weight + minimum_weight
        )
    return output


def block_conformal_p_values(
    calibration_scores: np.ndarray,
    calibration_groups: Sequence[str],
    test_block_scores: np.ndarray,
) -> np.ndarray:
    """Return block-max conformal p-values for source-level alarms."""
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    groups = np.asarray([str(group) for group in calibration_groups])
    block_scores = np.asarray(
        [
            float(calibration_scores[groups == group].max())
            for group in np.unique(groups)
        ],
        dtype=np.float64,
    )
    test_block_scores = np.asarray(
        test_block_scores,
        dtype=np.float64,
    )
    return (
        1.0
        + np.sum(
            block_scores[:, None] >= test_block_scores[None, :],
            axis=0,
        )
    ) / (len(block_scores) + 1.0)


def conditional_false_positive_rates(
    labels: np.ndarray,
    p_values: np.ndarray,
    strata: np.ndarray,
    alpha: float,
) -> dict[str, dict[str, float | int]]:
    """Report per-stratum FPR and recall at a conformal threshold."""
    labels = np.asarray(labels, dtype=np.int64)
    p_values = np.asarray(p_values, dtype=np.float64)
    strata = np.asarray(strata, dtype=np.int64)
    prediction = p_values <= alpha
    output: dict[str, dict[str, float | int]] = {}
    for stratum in sorted(np.unique(strata)):
        indices = strata == stratum
        normal = indices & (labels == 0)
        abnormal = indices & (labels == 1)
        output[f"stratum_{int(stratum)}"] = {
            "windows": int(indices.sum()),
            "normal_windows": int(normal.sum()),
            "abnormal_windows": int(abnormal.sum()),
            "false_positive_rate": (
                float(np.mean(prediction[normal]))
                if normal.any()
                else 0.0
            ),
            "recall": (
                float(np.mean(prediction[abnormal]))
                if abnormal.any()
                else 0.0
            ),
        }
    return output
