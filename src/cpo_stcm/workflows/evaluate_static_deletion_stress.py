# SPDX-License-Identifier: AGPL-3.0-only
"""Measure static-branch sensitivity to controlled semantic deletions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()

from cpo_stcm.powicm_core import (
    POWICMModel,
    SLOT_NAMES,
    _candidate_slots,
    load_calibration,
)
from cpo_stcm.static_evidence import (
    candidates_to_slot_observations,
    detections_to_candidate_list,
    score_candidate_window,
)
from cpo_stcm.paths import load_resolved_config


def load_jsonl(path: Path) -> dict[str, dict]:
    """Read the cached raw detections."""
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[str(record["base_window_id"])] = record
    return records


def frame_candidates(record: dict) -> list[list]:
    """Convert one cached record into semantic candidates per frame."""
    return [
        detections_to_candidate_list(frame["detections"])
        for frame in record["frames"]
    ]


def remove_slot_candidates(
    candidates_by_frame: list[list],
    slot_index: int,
    image_height: int,
    minimum_confidence: float,
) -> list[list]:
    """Remove the selected semantic slot from each frame."""
    output = []
    for candidates in candidates_by_frame:
        observations = candidates_to_slot_observations(
            candidates,
            image_height,
            minimum_confidence,
        )
        selected = observations.get(slot_index)
        if selected is None:
            output.append(list(candidates))
            continue
        selected_center_y = (selected.box[1] + selected.box[3]) / 2.0
        retained = []
        for candidate in candidates:
            compatible = _candidate_slots(candidate)
            if candidate.candidate_type in {"digit_block", "line"}:
                center_y = (candidate.box[1] + candidate.box[3]) / 2.0
                remove = (
                    slot_index in compatible
                    and abs(center_y - selected_center_y) <= 0.05 * image_height
                )
            else:
                remove = slot_index in compatible
            if not remove:
                retained.append(candidate)
        output.append(retained)
    return output


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cpo_stcm.yaml",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "static_deletion_stress"
            / "charge_v1"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    if args.cache is None:
        args.cache = Path(config["paths"]["raw_detections_path"])
    if args.calibration is None:
        args.calibration = Path(config["paths"]["static_calibration_path"])
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(args.calibration)
    model = POWICMModel(calibration)
    records = load_jsonl(args.cache)
    calibration_records = [
        record
        for record in records.values()
        if record["role"] == "normal_calibration"
    ]
    test_normals = [
        record
        for record in records.values()
        if record["role"] == "test"
        and record["subclass"] == "animated"
    ]
    if not calibration_records or not test_normals:
        raise RuntimeError("Missing calibration or test-normal records")
    image_height = calibration.image_height
    minimum_confidence = calibration.confidence_floor
    calibration_scores = []
    for record in calibration_records:
        result = score_candidate_window(
            frame_candidates(record),
            model,
            persistent_missing_frames=3,
        )
        calibration_scores.append(float(result["layout_anomaly_best"]))
    threshold = float(
        np.quantile(
            np.asarray(calibration_scores),
            1.0 - args.alpha,
            method="higher",
        )
    )
    original_scores = []
    for record in test_normals:
        result = score_candidate_window(
            frame_candidates(record),
            model,
            persistent_missing_frames=3,
        )
        original_scores.append(float(result["layout_anomaly_best"]))
    original_scores_array = np.asarray(original_scores)

    results = [
        {
            "condition": "unperturbed",
            "slot_name": "none",
            "windows_with_observed_slot": len(test_normals),
            "detection_rate": float(
                np.mean(original_scores_array >= threshold)
            ),
            "mean_anomaly_score": float(original_scores_array.mean()),
        }
    ]
    for slot_index, slot_name in enumerate(SLOT_NAMES):
        perturbed_scores = []
        observed_count = 0
        for record in test_normals:
            candidates = frame_candidates(record)
            observations = [
                candidates_to_slot_observations(
                    frame,
                    image_height,
                    minimum_confidence,
                )
                for frame in candidates
            ]
            if any(slot_index in item for item in observations):
                observed_count += 1
            perturbed = remove_slot_candidates(
                candidates,
                slot_index,
                image_height,
                minimum_confidence,
            )
            result = score_candidate_window(
                perturbed,
                model,
                persistent_missing_frames=3,
            )
            perturbed_scores.append(float(result["layout_anomaly_best"]))
        scores = np.asarray(perturbed_scores)
        results.append(
            {
                "condition": "semantic_deletion",
                "slot_name": slot_name,
                "windows_with_observed_slot": int(observed_count),
                "detection_rate": float(np.mean(scores >= threshold)),
                "mean_anomaly_score": float(scores.mean()),
            }
        )
    output_frame = pd.DataFrame(results)
    output_frame.to_csv(
        args.output / "deletion_stress.csv",
        index=False,
        encoding="utf-8",
    )
    report = {
        "alpha": args.alpha,
        "anomaly_threshold": threshold,
        "calibration_windows": len(calibration_records),
        "test_normal_windows": len(test_normals),
        "results": results,
    }
    (args.output / "deletion_stress_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_frame.to_string(index=False))


if __name__ == "__main__":
    main()
