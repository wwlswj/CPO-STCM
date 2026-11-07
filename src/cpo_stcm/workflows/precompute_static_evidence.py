# SPDX-License-Identifier: AGPL-3.0-only
# ruff: noqa: E402
"""Extract YOLO detections and adaptive PO-WICM evidence for charge windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))


def sanitize_cuda_path() -> None:
    """Prevent system CUDA Toolkit DLLs from shadowing PyTorch DLLs."""
    entries = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        entry
        for entry in entries
        if "nvidia gpu computing toolkit" not in entry.lower()
    )
    for key in tuple(os.environ):
        if key.upper() == "CUDA_PATH" or key.upper().startswith("CUDA_PATH_"):
            os.environ.pop(key, None)


sanitize_cuda_path()

import cv2
import numpy as np
from ultralytics import YOLO

from cpo_stcm.paths import load_resolved_config
from cpo_stcm.powicm_core import (
    POWICMModel,
    load_calibration,
    save_calibration,
)
from cpo_stcm.static_evidence import (
    candidates_to_slot_observations,
    crop_array_subimage,
    detections_to_candidate_list,
    fit_adaptive_calibration,
    merge_frame_observations,
    score_candidate_window,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_config(path: Path) -> dict:
    """Load the project YAML configuration."""
    return load_resolved_config(path)


def read_grayscale(path: Path) -> np.ndarray:
    """Read an image from a Unicode Windows path."""
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        raise RuntimeError(f"Unable to decode image: {path}")
    return image


def serialize_detections(result) -> list[dict[str, object]]:
    """Convert one Ultralytics result into a JSON-compatible list."""
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return [
        {
            "class_index": int(class_index),
            "confidence": float(confidence),
            "box": [float(value) for value in box],
        }
            for box, confidence, class_index in zip(
            boxes,
            confidences,
            classes,
                strict=True,
        )
    ]


def build_base_window_rows(
    rows: list[dict[str, str]],
    selected_roles: set[str],
) -> list[dict[str, object]]:
    """Collapse transformed rows into one record per spatial base window."""
    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(
        list
    )
    for row in rows:
        if row["source"] != "charge" or row["role"] not in selected_roles:
            continue
        key = (
            row["sequence_id"],
            row["source"],
            int(row["start_position"]),
            int(row["subimage_index"]),
        )
        grouped[key].append(row)

    output: list[dict[str, object]] = []
    subclass_priority = {
        "no_device": 3,
        "static_animation": 2,
        "animated": 1,
    }
    for key, variants in sorted(grouped.items()):
        representative = min(
            variants,
            key=lambda row: (
                row["transform"] != "none",
                row["sample_id"],
            ),
        )
        subclass = max(
            (row["subclass"] for row in variants),
            key=lambda value: subclass_priority.get(value, 0),
        )
        if subclass.startswith("temporal_"):
            subclass = "animated"
        output.append(
            {
                "base_window_id": f"{key[0]}|{key[2]}|{key[3]}",
                "sequence_id": key[0],
                "source": key[1],
                "start_position": key[2],
                "subimage_index": key[3],
                "source_split": representative["source_split"],
                "role": representative["role"],
                "subclass": subclass,
                "label_abnormal": int(
                    subclass in {"static_animation", "no_device"}
                ),
            }
        )
    return output


def load_cache(path: Path) -> dict[str, dict[str, object]]:
    """Load previously completed windows from a JSONL cache."""
    cached: dict[str, dict[str, object]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cached[str(record["base_window_id"])] = record
    return cached


def extract_window_detections(
    window: dict[str, object],
    source_row: dict[str, str],
    source_root: Path,
    detector: YOLO,
    offsets: list[int],
    array_rows: int,
    array_columns: int,
    crop_height: int,
    crop_width: int,
    confidence: float,
    device: str,
) -> dict[str, object]:
    """Extract detections from uniformly spaced frames in one window."""
    sequence_root = source_root / source_row["relative_path"]
    frame_paths = sorted(
        (
            path
            for path in sequence_root.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ),
        key=lambda path: int(path.stem),
    )
    crops: list[np.ndarray] = []
    valid_offsets: list[int] = []
    for offset in offsets:
        frame_index = int(window["start_position"]) + offset - 1
        if frame_index < 0 or frame_index >= len(frame_paths):
            continue
        frame = read_grayscale(frame_paths[frame_index])
        crops.append(
            crop_array_subimage(
                frame,
                int(window["subimage_index"]),
                array_rows,
                array_columns,
                crop_height,
                crop_width,
            )
        )
        valid_offsets.append(offset)
    if not crops:
        raise RuntimeError(
            f"No valid frames for base window {window['base_window_id']}"
        )
    results = detector.predict(
        source=[cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in crops],
        conf=confidence,
        device=device,
        verbose=False,
    )
    return {
        **window,
        "frames": [
            {
                "offset": offset,
                "detections": serialize_detections(result),
            }
            for offset, result in zip(
                valid_offsets,
                results,
                strict=True,
            )
        ],
    }


def summarize_numeric(
    values: list[float],
) -> dict[str, float | int | None]:
    """Summarize a numeric vector."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cpo_stcm.yaml",
    )
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_rows(Path(config["paths"]["source_manifest"]))
    selected_roles = {
        "normal_calibration",
        "validation",
        "test",
        "domain_test",
    }
    base_windows = build_base_window_rows(rows, selected_roles)
    if args.limit is not None:
        base_windows = base_windows[: args.limit]
    source_rows = {
        row["sequence_id"]: row
        for row in read_rows(
            Path(config["paths"]["raw_sequence_manifest"])
        )
    }
    output_root = PROJECT_ROOT / "experiments" / "static_evidence"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / "raw_detections.jsonl"
    if args.force and cache_path.exists():
        cache_path.unlink()
    cached = load_cache(cache_path)
    pending = [
        window
        for window in base_windows
        if str(window["base_window_id"]) not in cached
    ]
    detector = YOLO(str(config["paths"]["static_detector_weights"]))
    offsets = [int(value) for value in config["scoring"]["static_frame_offsets"]]
    confidence = float(config["scoring"]["static_detector_confidence"])
    with cache_path.open("a", encoding="utf-8") as cache_stream:
        for index, window in enumerate(pending, start=1):
            record = extract_window_detections(
                window,
                source_rows[str(window["sequence_id"])],
                Path(config["paths"]["raw_charge_root"]),
                detector,
                offsets,
                int(config["data"]["array_rows"]),
                int(config["data"]["array_columns"]),
                int(config["data"]["static_crop_height"]),
                int(config["data"]["static_crop_width"]),
                confidence,
                args.device,
            )
            cache_stream.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            cache_stream.flush()
            cached[str(record["base_window_id"])] = record
            if index % 25 == 0 or index == len(pending):
                print(
                    f"extracted {index}/{len(pending)} pending windows",
                    flush=True,
                )

    missing_keys = [
        window["base_window_id"]
        for window in base_windows
        if str(window["base_window_id"]) not in cached
    ]
    if missing_keys:
        raise RuntimeError(
            f"Detection cache is missing {len(missing_keys)} windows"
        )
    selected_records = [
        cached[str(window["base_window_id"])]
        for window in base_windows
    ]
    minimum_slot_confidence = float(
        config["scoring"]["static_minimum_slot_confidence"]
    )
    image_height = int(config["data"]["static_crop_height"])
    image_width = int(config["data"]["static_crop_width"])
    calibration_observations = []
    for record in selected_records:
        if record["role"] != "normal_calibration":
            continue
        frame_observations = []
        for frame in record["frames"]:
            candidates = detections_to_candidate_list(frame["detections"])
            frame_observations.append(
                candidates_to_slot_observations(
                    candidates,
                    image_height,
                    minimum_slot_confidence,
                )
            )
        calibration_observations.append(
            merge_frame_observations(frame_observations)
        )
    fallback = load_calibration(Path(config["paths"]["powicm_calibration"]))
    calibration, calibration_diagnostics = fit_adaptive_calibration(
        calibration_observations,
        fallback,
        minimum_confidence=minimum_slot_confidence,
    )
    save_calibration(calibration, output_root / "calibration.json")
    (output_root / "calibration_diagnostics.json").write_text(
        json.dumps(calibration_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = POWICMModel(calibration)
    persistent_missing_frames = int(
        config["scoring"]["persistent_missing_frames"]
    )
    output_rows = []
    for record in selected_records:
        frame_candidates = [
            detections_to_candidate_list(frame["detections"])
            for frame in record["frames"]
        ]
        result = score_candidate_window(
            frame_candidates,
            model,
            persistent_missing_frames,
        )
        output_rows.append(
            {
                "base_window_id": record["base_window_id"],
                "sequence_id": record["sequence_id"],
                "start_position": record["start_position"],
                "subimage_index": record["subimage_index"],
                "source": record["source"],
                "role": record["role"],
                "source_split": record["source_split"],
                "subclass": record["subclass"],
                "label_abnormal": record["label_abnormal"],
                "static_frames": result["static_frames"],
                "layout_anomaly_best": result["layout_anomaly_best"],
                "layout_anomaly_mean": result["layout_anomaly_mean"],
                "layout_anomaly_worst": result["layout_anomaly_worst"],
                "hard_pass_fraction": result["hard_pass_fraction"],
                "missing_slot_rate": result["missing_slot_rate"],
                "persistent_missing_count": result[
                    "persistent_missing_count"
                ],
                "persistent_missing_slots": result[
                    "persistent_missing_slots"
                ],
                "missing_slot_counts": json.dumps(
                    result["missing_slot_counts"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "frame_anomaly_scores": json.dumps(
                    result["frame_anomaly_scores"],
                ),
                "best_frame_index": result["best_frame_index"],
            }
        )
    output_path = PROJECT_ROOT / "manifests" / "static_evidence.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    role_subclass = Counter(
        f"{row['role']}/{row['subclass']}" for row in output_rows
    )
    slot_summary = {
        slot_name: {
            "presence_rate": next(
                payload["presence_rate"]
                for payload in calibration_diagnostics["slots"]
                if payload["slot_name"] == slot_name
            ),
            "required": next(
                payload["required"]
                for payload in calibration_diagnostics["slots"]
                if payload["slot_name"] == slot_name
            ),
        }
        for slot_name in [slot.slot_name for slot in calibration.slots]
    }
    summary = {
        "windows": len(output_rows),
        "role_subclass_counts": dict(sorted(role_subclass.items())),
        "detector_confidence": confidence,
        "minimum_slot_confidence": minimum_slot_confidence,
        "frame_offsets": offsets,
        "array_shape": [
            int(config["data"]["array_rows"]),
            int(config["data"]["array_columns"]),
        ],
        "crop_size": [image_height, image_width],
        "slots": slot_summary,
        "layout_anomaly_best": summarize_numeric(
            [float(row["layout_anomaly_best"]) for row in output_rows]
        ),
        "missing_slot_rate": summarize_numeric(
            [float(row["missing_slot_rate"]) for row in output_rows]
        ),
    }
    (PROJECT_ROOT / "manifests" / "static_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
