# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptive partial-observation evidence for the static semantic branch."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

import numpy as np

from cpo_stcm.powicm_core import (
    SLOT_NAMES,
    Calibration,
    Detection,
    POWICMModel,
    SemanticCandidate,
    SlotCalibration,
    detections_to_candidates,
)

PATTERN_SLOT_INDEX = 4
REQUIRED_PRESENCE_RATE = 0.85


def crop_array_subimage(
    frame: np.ndarray,
    subimage_index: int,
    array_rows: int,
    array_columns: int,
    crop_height: int,
    crop_width: int,
) -> np.ndarray:
    """Crop one centered subimage from a row-major LED screen array."""
    if array_rows < 1 or array_columns < 1:
        raise ValueError("Array dimensions must be positive")
    if not 0 <= subimage_index < array_rows * array_columns:
        raise ValueError(
            f"Subimage index {subimage_index} is outside "
            f"a {array_rows}x{array_columns} array"
        )
    row_index, column_index = divmod(subimage_index, array_columns)
    sub_height = frame.shape[0] // array_rows
    sub_width = frame.shape[1] // array_columns
    top = row_index * sub_height
    left = column_index * sub_width
    subimage = frame[
        top:top + sub_height,
        left:left + sub_width,
    ]
    crop_top = max(0, (subimage.shape[0] - crop_height) // 2)
    crop_left = max(0, (subimage.shape[1] - crop_width) // 2)
    cropped = subimage[
        crop_top:crop_top + crop_height,
        crop_left:crop_left + crop_width,
    ]
    if cropped.shape == (crop_height, crop_width):
        return cropped
    canvas = np.zeros((crop_height, crop_width), dtype=subimage.dtype)
    copy_height = min(crop_height, cropped.shape[0])
    copy_width = min(crop_width, cropped.shape[1])
    canvas[:copy_height, :copy_width] = cropped[
        :copy_height,
        :copy_width,
    ]
    return canvas


def _candidate_center_y(candidate: SemanticCandidate) -> float:
    """Return a normalized candidate center coordinate."""
    return float((candidate.box[1] + candidate.box[3]) / 2.0)


def _highest_confidence(
    candidates: Iterable[SemanticCandidate],
) -> SemanticCandidate | None:
    """Return the candidate with the largest detector confidence."""
    candidate_list = list(candidates)
    if not candidate_list:
        return None
    return max(candidate_list, key=lambda item: item.confidence)


def candidates_to_slot_observations(
    candidates: Sequence[SemanticCandidate],
    image_height: int,
    minimum_confidence: float,
) -> dict[int, SemanticCandidate]:
    """Map detector candidates to semantic slots without assuming completeness."""
    eligible = [
        candidate
        for candidate in candidates
        if candidate.confidence >= minimum_confidence
    ]
    observations: dict[int, SemanticCandidate] = {}

    for slot_index, class_index in ((0, 10), (1, 11), (7, 13), (8, 14)):
        candidate = _highest_confidence(
            item
            for item in eligible
            if item.class_indices == (class_index,)
        )
        if candidate is not None:
            observations[slot_index] = candidate

    pattern_candidate = _highest_confidence(
        item
        for item in eligible
        if item.candidate_type == "pattern"
    )
    if pattern_candidate is not None:
        observations[PATTERN_SLOT_INDEX] = pattern_candidate

    digit_candidates = sorted(
        (
            item
            for item in eligible
            if item.candidate_type == "digit_block"
        ),
        key=lambda item: item.confidence,
        reverse=True,
    )[:2]
    digit_candidates.sort(key=_candidate_center_y)
    if len(digit_candidates) == 2:
        observations[2] = digit_candidates[0]
        observations[6] = digit_candidates[1]
    elif len(digit_candidates) == 1:
        center_y = _candidate_center_y(digit_candidates[0]) / image_height
        observations[2 if center_y < 0.45 else 6] = digit_candidates[0]

    line_candidates = sorted(
        (
            item
            for item in eligible
            if item.candidate_type == "line"
        ),
        key=lambda item: item.confidence,
        reverse=True,
    )[:2]
    line_candidates.sort(key=_candidate_center_y)
    if len(line_candidates) == 2:
        observations[3] = line_candidates[0]
        observations[5] = line_candidates[1]
    elif len(line_candidates) == 1:
        center_y = _candidate_center_y(line_candidates[0]) / image_height
        observations[3 if center_y < 0.45 else 5] = line_candidates[0]

    return observations


def merge_frame_observations(
    frame_observations: Sequence[dict[int, SemanticCandidate]],
) -> dict[int, SemanticCandidate]:
    """Retain the most reliable observation of each slot across a window."""
    merged: dict[int, SemanticCandidate] = {}
    for observations in frame_observations:
        for slot_index, candidate in observations.items():
            current = merged.get(slot_index)
            if current is None or candidate.confidence > current.confidence:
                merged[slot_index] = candidate
    return merged


def _normalized_geometry(
    candidate: SemanticCandidate,
    image_height: int,
    image_width: int,
) -> tuple[float, float, float]:
    """Return normalized center y, aspect ratio, and area."""
    width = max(1e-6, candidate.box[2] - candidate.box[0])
    height = max(1e-6, candidate.box[3] - candidate.box[1])
    return (
        (candidate.box[1] + candidate.box[3]) / 2.0 / image_height,
        width / height,
        width * height / (image_height * image_width),
    )


def _robust_scale(values: np.ndarray) -> float:
    """Estimate a positive scale from a small one-dimensional sample."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-8:
        return max(0.008, 1.4826 * mad)
    lower, upper = np.percentile(values, [10, 90])
    return max(0.008, float(upper - lower) / 2.563)


def _weighted_pava(
    values: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """Fit a non-decreasing sequence with weighted PAVA."""
    blocks: list[dict[str, float | int]] = []
    for value, weight in zip(values, weights, strict=True):
        blocks.append(
            {
                "mean": float(value),
                "weight": max(float(weight), 1e-6),
                "start": len(blocks),
                "end": len(blocks),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            left = blocks[-2]
            right = blocks[-1]
            total_weight = float(left["weight"]) + float(right["weight"])
            blocks[-2] = {
                "mean": (
                    float(left["mean"]) * float(left["weight"])
                    + float(right["mean"]) * float(right["weight"])
                )
                / total_weight,
                "weight": total_weight,
                "start": int(left["start"]),
                "end": int(right["end"]),
            }
            blocks.pop()
    fitted = np.zeros(len(values), dtype=np.float64)
    for block in blocks:
        fitted[int(block["start"]):int(block["end"]) + 1] = float(
            block["mean"]
        )
    return fitted


def fit_adaptive_calibration(
    window_observations: Sequence[dict[int, SemanticCandidate]],
    fallback: Calibration,
    minimum_confidence: float = 0.005,
) -> tuple[Calibration, dict[str, object]]:
    """Fit slot geometry and presence priors from normal charge windows."""
    if not window_observations:
        raise ValueError("At least one calibration window is required")
    sample_count = len(window_observations)
    geometry: dict[int, list[tuple[float, float, float]]] = {
        index: [] for index in range(len(SLOT_NAMES))
    }
    for observations in window_observations:
        for slot_index, candidate in observations.items():
            if candidate.confidence < minimum_confidence:
                continue
            geometry[slot_index].append(
                _normalized_geometry(
                    candidate,
                    fallback.image_height,
                    fallback.image_width,
                )
            )

    fallback_slots = {slot.slot_index: slot for slot in fallback.slots}
    preliminary_means: list[float] = []
    preliminary_weights: list[float] = []
    slot_payload: list[dict[str, float | bool | str | int]] = []
    for slot_index, slot_name in enumerate(SLOT_NAMES):
        values = np.asarray(geometry[slot_index], dtype=np.float64)
        presence = len(values) / sample_count
        if len(values) >= 5:
            mean_y = float(np.median(values[:, 0]))
            sigma_y = _robust_scale(values[:, 0])
            tolerance_y = float(np.clip(1.75 * sigma_y + 0.01, 0.025, 0.20))
            target_aspect = max(1e-3, float(np.median(values[:, 1])))
            target_area = max(1e-7, float(np.median(values[:, 2])))
        else:
            fallback_slot = fallback_slots[slot_index]
            mean_y = fallback_slot.mean_y
            sigma_y = fallback_slot.sigma_y
            tolerance_y = fallback_slot.tolerance_y
            target_aspect = fallback_slot.target_aspect
            target_area = fallback_slot.target_area
        required = (
            presence >= REQUIRED_PRESENCE_RATE
            and slot_index != PATTERN_SLOT_INDEX
        )
        preliminary_means.append(mean_y)
        preliminary_weights.append(max(1.0, float(len(values))))
        slot_payload.append(
            {
                "slot_index": slot_index,
                "slot_name": slot_name,
                "mean_y": mean_y,
                "sigma_y": sigma_y,
                "tolerance_y": tolerance_y,
                "target_aspect": target_aspect,
                "target_area": target_area,
                "weight": float(max(0.05, presence)),
                "required": required,
                "presence_rate": float(presence),
                "observations": int(len(values)),
            }
        )

    fitted_means = _weighted_pava(
        preliminary_means,
        preliminary_weights,
    )
    slots = tuple(
        SlotCalibration(
            slot_index=int(payload["slot_index"]),
            slot_name=str(payload["slot_name"]),
            mean_y=float(fitted_means[int(payload["slot_index"])]),
            sigma_y=float(payload["sigma_y"]),
            tolerance_y=float(payload["tolerance_y"]),
            target_aspect=float(payload["target_aspect"]),
            target_area=float(payload["target_area"]),
            weight=float(payload["weight"]),
            required=bool(payload["required"]),
        )
        for payload in slot_payload
    )
    calibration = Calibration(
        version="1.1.0-adaptive-partial",
        image_height=fallback.image_height,
        image_width=fallback.image_width,
        slots=slots,
        confidence_floor=minimum_confidence,
        max_match_cost=3.0,
        order_tolerance=0.008,
    )
    diagnostics = {
        "calibration_windows": sample_count,
        "minimum_confidence": minimum_confidence,
        "slots": slot_payload,
    }
    return calibration, diagnostics


def score_candidate_window(
    frame_candidates: Sequence[Sequence[SemanticCandidate]],
    model: POWICMModel,
    persistent_missing_frames: int,
) -> dict[str, object]:
    """Score a window by its best complete partial observation."""
    if not frame_candidates:
        raise ValueError("At least one frame is required")
    frame_results = [
        model.score(candidates)
        for candidates in frame_candidates
    ]
    anomaly_scores = np.asarray(
        [result.anomaly_score for result in frame_results],
        dtype=np.float64,
    )
    missing_counts = {
        slot_name: sum(
            slot_name in result.missing_slots
            for result in frame_results
        )
        for slot_name in SLOT_NAMES
    }
    persistent_missing = tuple(
        slot_name
        for slot_name, count in missing_counts.items()
        if count >= persistent_missing_frames
    )
    best_index = int(np.argmin(anomaly_scores))
    return {
        "static_frames": len(frame_results),
        "layout_anomaly_best": float(anomaly_scores[best_index]),
        "layout_anomaly_mean": float(anomaly_scores.mean()),
        "layout_anomaly_worst": float(anomaly_scores.max()),
        "hard_pass_fraction": float(
            np.mean(
                [result.hard_constraint_passed for result in frame_results]
            )
        ),
        "missing_slot_rate": float(
            np.mean(
                [
                    len(result.missing_slots) / len(SLOT_NAMES)
                    for result in frame_results
                ]
            )
        ),
        "persistent_missing_count": len(persistent_missing),
        "persistent_missing_slots": "|".join(persistent_missing),
        "missing_slot_counts": {
            key: int(value)
            for key, value in missing_counts.items()
        },
        "frame_anomaly_scores": anomaly_scores.tolist(),
        "best_frame_index": best_index,
    }


def detections_to_candidate_list(
    raw_detections: Sequence[dict[str, object]],
) -> list[SemanticCandidate]:
    """Convert serialized detections into semantic candidates."""
    detections = [
        Detection(
            class_index=int(item["class_index"]),
            confidence=float(item["confidence"]),
            box=tuple(float(value) for value in item["box"]),
        )
        for item in raw_detections
    ]
    return detections_to_candidates(detections)


def replace_calibration_weights(
    calibration: Calibration,
    required_slots: set[int] | None = None,
) -> Calibration:
    """Return a calibration copy with optional explicit hard slots."""
    if required_slots is None:
        return calibration
    slots = tuple(
        replace(slot, required=slot.slot_index in required_slots)
        for slot in calibration.slots
    )
    return replace(calibration, slots=slots)
