# SPDX-License-Identifier: AGPL-3.0-only
"""Partially Observed Weighted Isotonic Consistency Model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SLOT_NAMES = (
    "top_arc",
    "lightning",
    "battery_level",
    "top_line",
    "pattern",
    "bottom_line",
    "fuel_level",
    "oil_drop",
    "bottom_arc",
)
SLOT_ALLOWED_CLASSES = {
    0: frozenset((10,)),
    1: frozenset((11,)),
    2: frozenset(range(10)),
    3: frozenset((12,)),
    4: frozenset((15, 16, 17, 18)),
    5: frozenset((12,)),
    6: frozenset(range(10)),
    7: frozenset((13,)),
    8: frozenset((14,)),
}
DIGIT_SLOTS = frozenset((2, 6))
LINE_SLOTS = frozenset((3, 5))


@dataclass(frozen=True)
class Detection:
    """One detector output in pixel coordinates."""

    class_index: int
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class SemanticCandidate:
    """A detector group that may correspond to one semantic slot."""

    class_indices: tuple[int, ...]
    confidence: float
    box: tuple[float, float, float, float]
    candidate_type: str


@dataclass(frozen=True)
class SlotCalibration:
    """Expected geometry and reliability prior for one semantic slot."""

    slot_index: int
    slot_name: str
    mean_y: float
    sigma_y: float
    tolerance_y: float
    target_aspect: float
    target_area: float
    weight: float = 1.0
    required: bool = True


@dataclass(frozen=True)
class Calibration:
    """Calibration payload for all semantic slots."""

    version: str
    image_height: int
    image_width: int
    slots: tuple[SlotCalibration, ...]
    confidence_floor: float = 0.05
    max_match_cost: float = 2.5
    order_tolerance: float = 0.01


@dataclass(frozen=True)
class ScoreWeights:
    """Exponents used by the PO-WICM consistency score."""

    completeness: float = 0.5
    order: float = 1.0
    position: float = 0.5
    confidence: float = 0.5
    use_completeness: bool = True
    use_order: bool = True
    use_position: bool = True
    use_confidence: bool = True
    enforce_hard_constraints: bool = True


@dataclass(frozen=True)
class POWICMResult:
    """Full PO-WICM score and diagnostic information."""

    normal_confidence: float
    anomaly_score: float
    completeness: float
    order_consistency: float
    position_consistency: float
    confidence_consistency: float
    hard_constraint_passed: bool
    matched_slots: tuple[dict, ...]
    missing_slots: tuple[str, ...]
    inversions: tuple[dict, ...]
    position_violations: tuple[dict, ...]


def _box_union(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Return the enclosing box of one or more boxes."""
    if not boxes:
        raise ValueError("At least one box is required")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _box_geometry(
    box: tuple[float, float, float, float],
    image_height: int,
    image_width: int,
) -> tuple[float, float, float]:
    """Return normalized center y, aspect ratio, and normalized area."""
    width = max(1e-6, box[2] - box[0])
    height = max(1e-6, box[3] - box[1])
    center_y = (box[1] + box[3]) / 2 / image_height
    aspect = width / height
    area = width * height / (image_height * image_width)
    return center_y, aspect, area


def _group_digit_detections(
    detections: Sequence[Detection],
) -> list[list[Detection]]:
    """Group digit detections into horizontal number blocks."""
    digits = sorted(
        (
            detection
            for detection in detections
            if 0 <= detection.class_index <= 9
        ),
        key=lambda detection: (
            (detection.box[1] + detection.box[3]) / 2,
            detection.box[0],
        ),
    )
    groups: list[list[Detection]] = []
    for detection in digits:
        center_y = (detection.box[1] + detection.box[3]) / 2
        height = detection.box[3] - detection.box[1]
        assigned = False
        for group in groups:
            group_center = np.mean(
                [
                    (item.box[1] + item.box[3]) / 2
                    for item in group
                ]
            )
            group_height = np.mean(
                [item.box[3] - item.box[1] for item in group]
            )
            if abs(center_y - group_center) <= 0.45 * max(
                height,
                group_height,
            ):
                group.append(detection)
                assigned = True
                break
        if not assigned:
            groups.append([detection])
    return groups


def detections_to_candidates(
    detections: Sequence[Detection],
) -> list[SemanticCandidate]:
    """Convert raw class detections into semantic-slot candidates."""
    candidates: list[SemanticCandidate] = []
    for group in _group_digit_detections(detections):
        group = sorted(group, key=lambda item: item.box[0])
        union = _box_union([item.box for item in group])
        confidence = float(
            np.exp(np.mean([math.log(max(item.confidence, 1e-8)) for item in group]))
        )
        candidates.append(
            SemanticCandidate(
                class_indices=tuple(item.class_index for item in group),
                confidence=confidence,
                box=union,
                candidate_type="digit_block",
            )
        )
    for detection in detections:
        if 0 <= detection.class_index <= 9:
            continue
        if detection.class_index == 12:
            candidate_type = "line"
        elif 15 <= detection.class_index <= 18:
            candidate_type = "pattern"
        elif detection.class_index in (10, 11, 13, 14):
            candidate_type = "symbol"
        else:
            continue
        candidates.append(
            SemanticCandidate(
                class_indices=(detection.class_index,),
                confidence=float(detection.confidence),
                box=detection.box,
                candidate_type=candidate_type,
            )
        )
    return candidates


def _candidate_slots(candidate: SemanticCandidate) -> frozenset[int]:
    """Return all semantic slots compatible with one candidate."""
    if candidate.candidate_type == "digit_block":
        return DIGIT_SLOTS
    if candidate.candidate_type == "line":
        return LINE_SLOTS
    compatible = {
        slot_index
        for slot_index, allowed_classes in SLOT_ALLOWED_CLASSES.items()
        if set(candidate.class_indices).issubset(allowed_classes)
    }
    return frozenset(compatible)


def _weighted_pava(
    values: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """Apply weighted pool-adjacent-violators isotonic regression."""
    if len(values) != len(weights):
        raise ValueError("Values and weights must have equal length")
    if not values:
        raise ValueError("At least one value is required")
    blocks: list[dict[str, float | int]] = []
    for value, weight in zip(values, weights):
        blocks.append(
            {
                "mean": float(value),
                "weight": float(weight),
                "start": len(blocks),
                "end": len(blocks),
                "count": 1,
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if float(left["mean"]) <= float(right["mean"]):
                break
            merged_weight = float(left["weight"]) + float(right["weight"])
            blocks[-2] = {
                "mean": (
                    float(left["mean"]) * float(left["weight"])
                    + float(right["mean"]) * float(right["weight"])
                )
                / merged_weight,
                "weight": merged_weight,
                "start": int(left["start"]),
                "end": int(right["end"]),
                "count": int(left["count"]) + int(right["count"]),
            }
            blocks.pop()
    fitted = np.zeros(len(values), dtype=np.float64)
    for block in blocks:
        fitted[int(block["start"]): int(block["end"]) + 1] = float(
            block["mean"]
        )
    return fitted


def _slot_union_boxes(
    semantic_slots: Sequence[dict],
    image_height: int,
    image_width: int,
) -> dict[int, tuple[float, float, float, float]]:
    """Extract one normalized semantic-slot union box from metadata."""
    result: dict[int, tuple[float, float, float, float]] = {}
    for slot in semantic_slots:
        records = slot["records"]
        if not records:
            continue
        boxes = [
            tuple(float(value) for value in record["box_pixel"])
            for record in records
        ]
        union = _box_union(boxes)
        center_y, aspect, area = _box_geometry(
            union,
            image_height,
            image_width,
        )
        result[int(slot["slot_index"])] = (center_y, aspect, area)
    return result


def fit_calibration(
    calibration_samples: Iterable[dict],
    image_height: int,
    image_width: int,
) -> Calibration:
    """Estimate monotone slot priors from normal semantic-layout samples."""
    measurements: dict[int, list[tuple[float, float, float]]] = {
        index: [] for index in range(len(SLOT_NAMES))
    }
    for sample in calibration_samples:
        if not sample.get("is_normal", False):
            continue
        slot_geometry = _slot_union_boxes(
            sample["semantic_slots"],
            image_height,
            image_width,
        )
        for slot_index, geometry in slot_geometry.items():
            measurements[slot_index].append(geometry)
    medians: list[float] = []
    sigmas: list[float] = []
    aspects: list[float] = []
    areas: list[float] = []
    counts: list[float] = []
    for slot_index in range(len(SLOT_NAMES)):
        values = np.asarray(measurements[slot_index], dtype=np.float64)
        if len(values) < 5:
            raise ValueError(
                f"Insufficient calibration observations for slot {slot_index}"
            )
        median_y = float(np.median(values[:, 0]))
        mad = float(
            np.median(np.abs(values[:, 0] - median_y))
        )
        medians.append(median_y)
        sigmas.append(max(0.006, 1.4826 * mad))
        aspects.append(max(1e-3, float(np.median(values[:, 1]))))
        areas.append(max(1e-7, float(np.median(values[:, 2]))))
        counts.append(float(len(values)))
    fitted_means = _weighted_pava(medians, counts)
    slots = tuple(
        SlotCalibration(
            slot_index=index,
            slot_name=SLOT_NAMES[index],
            mean_y=float(fitted_means[index]),
            sigma_y=float(sigmas[index]),
            tolerance_y=float(max(0.012, 1.5 * sigmas[index])),
            target_aspect=aspects[index],
            target_area=areas[index],
            weight=1.0,
            required=True,
        )
        for index in range(len(SLOT_NAMES))
    )
    return Calibration(
        version="1.0.0",
        image_height=image_height,
        image_width=image_width,
        slots=slots,
    )


def save_calibration(calibration: Calibration, path: Path) -> None:
    """Write a calibration object to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(calibration)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibration(path: Path) -> Calibration:
    """Read a calibration object from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    slots = tuple(
        SlotCalibration(**slot)
        for slot in payload["slots"]
    )
    return Calibration(
        version=str(payload["version"]),
        image_height=int(payload["image_height"]),
        image_width=int(payload["image_width"]),
        slots=slots,
        confidence_floor=float(payload.get("confidence_floor", 0.05)),
        max_match_cost=float(payload.get("max_match_cost", 2.5)),
        order_tolerance=float(payload.get("order_tolerance", 0.01)),
    )


class POWICMModel:
    """Score partially observed semantic layouts with isotonic consistency."""

    def __init__(
        self,
        calibration: Calibration,
        weights: ScoreWeights | None = None,
        position_weight: float = 1.0,
        confidence_weight: float = 0.35,
        aspect_weight: float = 0.15,
        area_weight: float = 0.10,
    ):
        self.calibration = calibration
        self.weights = weights or ScoreWeights()
        self.position_weight = position_weight
        self.confidence_weight = confidence_weight
        self.aspect_weight = aspect_weight
        self.area_weight = area_weight
        self._slot_by_index = {
            slot.slot_index: slot for slot in calibration.slots
        }

    def _confidence(self, confidence: float) -> float:
        """Calibrate a detector confidence to a bounded reliability value."""
        floor = self.calibration.confidence_floor
        return float(np.clip((confidence - floor) / (1.0 - floor), 0.0, 1.0))

    def _match_cost(
        self,
        slot: SlotCalibration,
        candidate: SemanticCandidate,
    ) -> float:
        """Compute the assignment cost between one slot and one candidate."""
        if slot.slot_index not in _candidate_slots(candidate):
            return math.inf
        center_y, aspect, area = _box_geometry(
            candidate.box,
            self.calibration.image_height,
            self.calibration.image_width,
        )
        position_excess = max(
            abs(center_y - slot.mean_y) - slot.tolerance_y,
            0.0,
        )
        normalized_position = position_excess / max(slot.sigma_y, 1e-6)
        aspect_deviation = abs(
            math.log(max(aspect, 1e-6) / slot.target_aspect)
        )
        area_deviation = abs(
            math.log(max(area, 1e-8) / slot.target_area)
        )
        confidence_cost = 1.0 - self._confidence(candidate.confidence)
        return (
            self.position_weight * normalized_position
            + self.confidence_weight * confidence_cost
            + self.aspect_weight * aspect_deviation
            + self.area_weight * area_deviation
        )

    def _match_candidates(
        self,
        candidates: Sequence[SemanticCandidate],
    ) -> dict[int, SemanticCandidate]:
        """Solve one-to-one slot assignment with optional non-matches."""
        from scipy.optimize import linear_sum_assignment

        slot_count = len(self.calibration.slots)
        if not candidates:
            return {}
        cost_matrix = np.full(
            (slot_count, len(candidates) + slot_count),
            self.calibration.max_match_cost * 2,
            dtype=np.float64,
        )
        for row, slot in enumerate(self.calibration.slots):
            for column, candidate in enumerate(candidates):
                cost = self._match_cost(slot, candidate)
                if math.isfinite(cost):
                    cost_matrix[row, column] = cost
            cost_matrix[
                row,
                len(candidates) + row,
            ] = self.calibration.max_match_cost
        rows, columns = linear_sum_assignment(cost_matrix)
        matched: dict[int, SemanticCandidate] = {}
        for row, column in zip(rows, columns):
            if column >= len(candidates):
                continue
            if cost_matrix[row, column] <= self.calibration.max_match_cost:
                matched[int(row)] = candidates[int(column)]
        return matched

    def score(
        self,
        candidates: Sequence[SemanticCandidate],
    ) -> POWICMResult:
        """Compute the full PO-WICM score and diagnostics."""
        matched_candidates = self._match_candidates(candidates)
        total_weight = sum(slot.weight for slot in self.calibration.slots)
        matched_weight = sum(
            self._slot_by_index[index].weight for index in matched_candidates
        )
        completeness = matched_weight / total_weight
        hard_constraint_passed = all(
            (not slot.required) or slot.slot_index in matched_candidates
            for slot in self.calibration.slots
        )

        matched_rows: list[dict] = []
        matched_y: dict[int, float] = {}
        matched_confidences: dict[int, float] = {}
        for slot_index, candidate in sorted(matched_candidates.items()):
            slot = self._slot_by_index[slot_index]
            center_y, aspect, area = _box_geometry(
                candidate.box,
                self.calibration.image_height,
                self.calibration.image_width,
            )
            matched_y[slot_index] = center_y
            matched_confidences[slot_index] = self._confidence(
                candidate.confidence
            )
            matched_rows.append(
                {
                    "slot_index": slot_index,
                    "slot_name": slot.slot_name,
                    "class_indices": list(candidate.class_indices),
                    "confidence": candidate.confidence,
                    "center_y": center_y,
                    "aspect": aspect,
                    "area": area,
                    "box": list(candidate.box),
                }
            )

        pair_weights: list[float] = []
        weighted_violations: list[float] = []
        inversions: list[dict] = []
        matched_indices = sorted(matched_y)
        for left_position, left_index in enumerate(matched_indices):
            for right_index in matched_indices[left_position + 1:]:
                left_slot = self._slot_by_index[left_index]
                right_slot = self._slot_by_index[right_index]
                pair_weight = math.sqrt(
                    left_slot.weight
                    * right_slot.weight
                    * matched_confidences[left_index]
                    * matched_confidences[right_index]
                )
                tolerance = 0.5 * (
                    left_slot.tolerance_y + right_slot.tolerance_y
                )
                tolerance = max(
                    self.calibration.order_tolerance,
                    tolerance,
                )
                violation = (
                    matched_y[left_index] >= matched_y[right_index] - tolerance
                )
                pair_weights.append(pair_weight)
                weighted_violations.append(pair_weight * float(violation))
                if violation:
                    inversions.append(
                        {
                            "left_slot": left_slot.slot_name,
                            "right_slot": right_slot.slot_name,
                            "left_y": matched_y[left_index],
                            "right_y": matched_y[right_index],
                        }
                    )
        if sum(pair_weights) > 0:
            order_consistency = 1.0 - (
                sum(weighted_violations) / sum(pair_weights)
            )
        else:
            order_consistency = completeness

        position_violations: list[dict] = []
        position_penalties: list[float] = []
        for slot_index, center_y in matched_y.items():
            slot = self._slot_by_index[slot_index]
            excess = max(
                abs(center_y - slot.mean_y) - slot.tolerance_y,
                0.0,
            )
            penalty = (excess / max(slot.sigma_y, 1e-6)) ** 2
            position_penalties.append(penalty)
            if excess > 0:
                position_violations.append(
                    {
                        "slot_name": slot.slot_name,
                        "center_y": center_y,
                        "expected_y": slot.mean_y,
                        "excess": excess,
                    }
                )
        if position_penalties:
            position_consistency = math.exp(
                -float(np.mean(position_penalties))
            )
        else:
            position_consistency = 0.0

        if matched_confidences:
            confidence_consistency = math.exp(
                float(np.mean(np.log(
                    [
                        max(value, 1e-8)
                        for value in matched_confidences.values()
                    ]
                )))
            )
        else:
            confidence_consistency = 0.0

        components = {
            "completeness": max(completeness, 1e-8),
            "order": max(order_consistency, 1e-8),
            "position": max(position_consistency, 1e-8),
            "confidence": max(confidence_consistency, 1e-8),
        }
        exponents = {
            "completeness": self.weights.completeness,
            "order": self.weights.order,
            "position": self.weights.position,
            "confidence": self.weights.confidence,
        }
        enabled = {
            "completeness": self.weights.use_completeness,
            "order": self.weights.use_order,
            "position": self.weights.use_position,
            "confidence": self.weights.use_confidence,
        }
        log_confidence = sum(
            -exponents[name] * math.log(components[name])
            for name in components
            if enabled[name]
        )
        normal_confidence = math.exp(-log_confidence)
        if (
            self.weights.enforce_hard_constraints
            and not hard_constraint_passed
        ):
            normal_confidence = 0.0
        normal_confidence = float(np.clip(normal_confidence, 0.0, 1.0))

        return POWICMResult(
            normal_confidence=normal_confidence,
            anomaly_score=1.0 - normal_confidence,
            completeness=completeness,
            order_consistency=order_consistency,
            position_consistency=position_consistency,
            confidence_consistency=confidence_consistency,
            hard_constraint_passed=hard_constraint_passed,
            matched_slots=tuple(matched_rows),
            missing_slots=tuple(
                slot.slot_name
                for slot in self.calibration.slots
                if slot.slot_index not in matched_candidates
            ),
            inversions=tuple(inversions),
            position_violations=tuple(position_violations),
        )
