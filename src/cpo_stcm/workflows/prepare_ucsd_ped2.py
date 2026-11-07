# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare UCSD Ped2 as an independent public difference-window benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path.cwd().resolve()
EXTERNAL_ROOT = PROJECT_ROOT / "datasets" / "external"


def list_frames(directory: Path) -> list[Path]:
    """Return sorted TIFF frames from one clip directory."""
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: int(
            "".join(character for character in path.stem if character.isdigit())
        ),
    )


def read_grayscale(path: Path, height: int, width: int) -> np.ndarray:
    """Read and resize one grayscale frame."""
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        raise RuntimeError(f"Unable to decode frame: {path}")
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def read_gt_frames(directory: Path) -> dict[int, bool]:
    """Map frame index to a binary anomaly label from pixel masks."""
    labels: dict[int, bool] = {}
    if not directory.exists():
        return labels
    for path in sorted(
        (
            item
            for item in directory.iterdir()
            if item.suffix.lower()
            in {".bmp", ".png", ".tif", ".tiff"}
        ),
        key=lambda item: int(
            "".join(character for character in item.stem if character.isdigit())
        ),
    ):
        mask = cv2.imdecode(
            np.fromfile(str(path), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise RuntimeError(f"Unable to decode ground truth: {path}")
        frame_number = int(
            "".join(character for character in path.stem if character.isdigit())
        )
        labels[frame_number] = bool(np.any(mask > 0))
    return labels


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extract the official archive if the dataset is not present."""
    dataset_root = destination / "UCSD_Anomaly_Dataset.v1p2"
    if dataset_root.exists():
        return dataset_root
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing UCSD archive: {archive_path}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination)
    if not dataset_root.exists():
        raise RuntimeError(
            f"Extracted archive does not contain {dataset_root}"
        )
    return dataset_root


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=EXTERNAL_ROOT / "UCSD_Anomaly_Dataset.tar.gz",
    )
    parser.add_argument(
        "--extract-root",
        type=Path,
        default=EXTERNAL_ROOT / "ucsd_extracted",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXTERNAL_ROOT / "UCSDped2-Windows-v1",
    )
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=144)
    args = parser.parse_args()
    dataset_root = extract_archive(args.archive, args.extract_root)
    ped2_root = dataset_root / "UCSDped2"
    train_root = ped2_root / "Train"
    test_root = ped2_root / "Test"
    if not train_root.exists() or not test_root.exists():
        raise RuntimeError(f"Invalid UCSD Ped2 layout in {ped2_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_clips = sorted(
        [path for path in train_root.iterdir() if path.is_dir()],
        key=lambda path: int(path.name.replace("Train", "")),
    )
    test_clips = sorted(
        [
            path
            for path in test_root.iterdir()
            if path.is_dir() and not path.name.endswith("_gt")
        ],
        key=lambda path: int(path.name.replace("Test", "")),
    )

    definitions: list[dict[str, object]] = []
    for clip_index, clip in enumerate(train_clips, start=1):
        if clip_index <= 13:
            role = "normal_train"
        elif clip_index == 14:
            role = "validation"
        else:
            role = "normal_calibration"
        definitions.append(
            {
                "clip": clip,
                "sequence_id": clip.name,
                "role": role,
                "subclass": "normal",
                "label": 0,
                "gt": None,
            }
        )
    for clip in test_clips:
        gt_dir = test_root / f"{clip.name}_gt"
        definitions.append(
            {
                "clip": clip,
                "sequence_id": clip.name,
                "role": "test",
                "subclass": "ped2_test",
                "label": None,
                "gt": read_gt_frames(gt_dir),
            }
        )

    total_windows = 0
    frame_cache: dict[str, list[Path]] = {}
    for definition in definitions:
        frames = list_frames(definition["clip"])
        frame_cache[str(definition["sequence_id"])] = frames
        total_windows += max(0, len(frames) - 10)
    if total_windows <= 0:
        raise RuntimeError("No temporal windows can be built")

    packed_path = args.output_root / "ped2_difference_windows.npy"
    packed = np.lib.format.open_memmap(
        packed_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_windows, 10, args.height, args.width),
    )
    rows: list[dict[str, object]] = []
    packed_index = 0
    for definition in definitions:
        sequence_id = str(definition["sequence_id"])
        frames = frame_cache[sequence_id]
        resized = [
            read_grayscale(path, args.height, args.width)
            for path in frames
        ]
        gt_labels = definition["gt"]
        for start_index in range(len(frames) - 10):
            differences = np.stack(
                [
                    cv2.absdiff(
                        resized[start_index + offset + 1],
                        resized[start_index + offset],
                    )
                    for offset in range(10)
                ]
            )
            packed[packed_index] = differences
            if definition["role"] == "test":
                frame_numbers = list(
                    range(start_index + 1, start_index + 12)
                )
                label = int(
                    any(
                        gt_labels.get(frame_number, False)
                        for frame_number in frame_numbers
                    )
                )
                center_frame_label = int(
                    gt_labels.get(start_index + 6, False)
                )
                subclass = "anomaly" if label else "normal"
            else:
                label = 0
                center_frame_label = 0
                subclass = "normal"
            rows.append(
                {
                    "packed_index": packed_index,
                    "source": "ucsd_ped2",
                    "sequence_id": sequence_id,
                    "role": definition["role"],
                    "subclass": subclass,
                    "transform": "none",
                    "label_abnormal": label,
                    "center_frame_label": center_frame_label,
                    "start_frame": start_index + 1,
                    "center_frame": start_index + 6,
                    "sample_id": f"{sequence_id}_{start_index + 1:04d}",
                }
            )
            packed_index += 1
        print(
            f"prepared {sequence_id}: {len(frames)} frames",
            flush=True,
        )
    packed.flush()
    del packed
    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    role_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["role"])
        role_counts[key] = role_counts.get(key, 0) + 1
    summary = {
        "dataset": "UCSD Ped2",
        "source_url": "http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm",
        "height": args.height,
        "width": args.width,
        "sequence_length": 10,
        "windows": len(rows),
        "roles": role_counts,
        "train_clips": [path.name for path in train_clips],
        "test_clips": [path.name for path in test_clips],
        "packed_array": str(packed_path),
        "manifest": str(manifest_path),
        "protocol_note": (
            "Each window contains ten absolute difference frames. "
            "Test windows are labeled abnormal when any of the eleven "
            "corresponding raw frames intersects a non-empty pixel mask."
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
