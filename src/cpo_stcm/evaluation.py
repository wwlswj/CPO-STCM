# SPDX-License-Identifier: AGPL-3.0-only
"""Calibration utilities and grouped metrics for CPO-STCM experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def empirical_tail_probability(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Map scores to smoothed empirical upper-tail probabilities."""
    reference = np.sort(np.asarray(reference_scores, dtype=np.float64))
    values = np.asarray(scores, dtype=np.float64)
    if reference.size == 0:
        raise ValueError("Reference scores must not be empty")
    count_less = np.searchsorted(reference, values, side="left")
    return (reference.size - count_less + 1.0) / (reference.size + 1.0)


def empirical_two_sided_anomaly_evidence(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Measure rarity in either tail and orient it toward high anomaly evidence."""
    reference = np.sort(np.asarray(reference_scores, dtype=np.float64))
    values = np.asarray(scores, dtype=np.float64)
    if reference.size == 0:
        raise ValueError("Reference scores must not be empty")
    lower_tail = np.searchsorted(reference, values, side="right") / (
        reference.size + 1.0
    )
    upper_tail = (
        reference.size
        - np.searchsorted(reference, values, side="left")
        + 1.0
    ) / (reference.size + 1.0)
    rarity = np.minimum(1.0, 2.0 * np.minimum(lower_tail, upper_tail))
    return 1.0 - rarity


def conformal_p_values(
    calibration_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Compute smoothed split-conformal p-values for high-score anomalies."""
    return empirical_tail_probability(calibration_scores, scores)


def empirical_upper_tail_anomaly_evidence(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Orient an upper-tail conformal p-value toward high anomaly evidence."""
    return 1.0 - empirical_tail_probability(reference_scores, scores)


def select_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Select an F1-optimal validation threshold."""
    from sklearn.metrics import f1_score

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(scores)
    best_threshold = float(candidates[-1])
    best_f1 = -1.0
    for threshold in candidates:
        prediction = (scores >= threshold).astype(np.int64)
        current = float(f1_score(labels, prediction, zero_division=0))
        if current > best_f1 + 1e-12 or (
            abs(current - best_f1) <= 1e-12
            and threshold > best_threshold
        ):
            best_f1 = current
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Compute discrimination and fixed-threshold classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    prediction = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )
    negative_count = true_negative + false_positive
    positive_count = true_positive + false_negative
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(
            precision_score(labels, prediction, zero_division=0)
        ),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "false_positive_rate": (
            float(false_positive / negative_count)
            if negative_count
            else None
        ),
        "true_negative_rate": (
            float(true_negative / negative_count)
            if negative_count
            else None
        ),
        "auroc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) == 2
            else None
        ),
        "auprc": (
            float(average_precision_score(labels, scores))
            if len(np.unique(labels)) == 2
            else None
        ),
        "confusion_matrix": matrix.tolist(),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
    }


def anomaly_type_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    subclasses: Sequence[str],
    threshold: float,
) -> dict[str, float]:
    """Compute recall for each abnormal subclass."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    prediction = (scores >= threshold).astype(np.int64)
    output: dict[str, float] = {}
    for subclass in sorted(
        {
            str(value)
            for value, label in zip(subclasses, labels, strict=True)
            if int(label) == 1
        }
    ):
        indices = np.asarray(
            [
                index
                for index, (value, label) in enumerate(
                    zip(subclasses, labels, strict=True)
                )
                if int(label) == 1 and str(value) == subclass
            ],
            dtype=np.int64,
        )
        if indices.size:
            output[subclass] = float(np.mean(prediction[indices]))
    return output


def grouped_bootstrap_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    threshold: float,
    seed: int,
    iterations: int = 500,
) -> dict[str, list[float]]:
    """Estimate confidence intervals by resampling source sequences."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    group_array = np.asarray([str(value) for value in groups])
    unique_groups = np.unique(group_array)
    indexes_by_group = {
        group: np.flatnonzero(group_array == group)
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"f1": [], "auroc": [], "auprc": []}
    for _ in range(iterations):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate(
            [indexes_by_group[group] for group in sampled_groups]
        )
        sampled_labels = labels[indices]
        sampled_scores = scores[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_prediction = (
            sampled_scores >= threshold
        ).astype(np.int64)
        values["f1"].append(
            float(
                f1_score(
                    sampled_labels,
                    sampled_prediction,
                    zero_division=0,
                )
            )
        )
        values["auroc"].append(
            float(roc_auc_score(sampled_labels, sampled_scores))
        )
        values["auprc"].append(
            float(average_precision_score(sampled_labels, sampled_scores))
        )
    return {
        name: [
            float(np.percentile(items, 2.5)),
            float(np.percentile(items, 97.5)),
        ]
        for name, items in values.items()
        if items
    }
