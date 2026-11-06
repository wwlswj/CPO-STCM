# SPDX-License-Identifier: AGPL-3.0-only
"""Build normal-only and temporal-stress manifests for CPO-STCM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd().resolve()

from cpo_stcm.paths import load_resolved_config


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def stable_bucket(value: str, seed: int) -> float:
    """Map a sequence ID to a stable value in [0, 1)."""
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write CSV rows with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def transformed_rows(
    row: dict[str, str],
    role: str,
    transforms: list[str],
) -> list[dict[str, object]]:
    """Create normal and transformed evaluation rows for one window."""
    base = {
        "packed_index": int(row["packed_index"]),
        "source": row["source"],
        "sequence_id": row["sequence_id"],
        "batch_id": row["batch_id"],
        "subimage_index": int(row["subimage_index"]),
        "source_split": row["split"],
        "role": role,
        "motion_energy": float(row["motion_energy"]),
        "start_position": int(row["start_position"]),
        "end_position": int(row["end_position"]),
        "sample_relative_path": row["sample_relative_path"],
    }
    result: list[dict[str, object]] = [
        {
            **base,
            "sample_id": f"{row['sample_id']}_original",
            "subclass": row["subtype"],
            "transform": "none",
            "label_abnormal": int(row["subtype"] != "animated"),
        }
    ]
    if row["subtype"] == "animated":
        for transform in transforms:
            result.append(
                {
                    **base,
                    "sample_id": f"{row['sample_id']}_{transform}",
                    "subclass": f"temporal_{transform}",
                    "transform": transform,
                    "label_abnormal": 1,
                }
            )
    return result


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cpo_stcm.yaml",
    )
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    source_rows = read_rows(Path(config["paths"]["base_source_manifest"]))
    seed = int(config["project"]["random_seed"])
    calibration_fraction = float(
        config["data"]["normal_calibration_fraction"]
    )
    transforms = list(config["data"]["temporal_transforms"])
    output_rows: list[dict[str, object]] = []

    normal_train_sequences: set[str] = set()
    normal_calibration_sequences: set[str] = set()
    for row in source_rows:
        if row["split"] == "train" and row["subtype"] == "animated":
            if stable_bucket(row["sequence_id"], seed) < calibration_fraction:
                normal_calibration_sequences.add(row["sequence_id"])
            else:
                normal_train_sequences.add(row["sequence_id"])

    for row in source_rows:
        if (
            row["split"] == "train"
            and row["subtype"] == "animated"
            and row["sequence_id"] in normal_train_sequences
        ):
            output_rows.extend(
                transformed_rows(row, "normal_train", [])
            )
        elif (
            row["split"] == "train"
            and row["subtype"] == "animated"
            and row["sequence_id"] in normal_calibration_sequences
        ):
            output_rows.extend(
                transformed_rows(row, "normal_calibration", [])
            )
        elif row["split"] == "validation":
            output_rows.extend(
                transformed_rows(row, "validation", transforms)
            )
        elif row["split"] == "test":
            output_rows.extend(
                transformed_rows(row, "test", transforms)
            )
        elif row["split"] == "test_domain_B":
            output_rows.extend(
                transformed_rows(row, "domain_test", transforms)
            )

    manifest_path = PROJECT_ROOT / "manifests" / "cpo_stcm_samples.csv"
    write_rows(manifest_path, output_rows)
    summary: dict[str, object] = {
        "samples": len(output_rows),
        "normal_train_sequences": len(normal_train_sequences),
        "normal_calibration_sequences": len(normal_calibration_sequences),
        "roles": {},
        "transforms": {},
    }
    roles: dict[str, dict[str, int]] = {}
    transform_counts: dict[str, int] = {}
    for row in output_rows:
        role = str(row["role"])
        subclass = str(row["subclass"])
        roles.setdefault(role, {})
        roles[role][subclass] = roles[role].get(subclass, 0) + 1
        transform = str(row["transform"])
        transform_counts[transform] = transform_counts.get(transform, 0) + 1
    summary["roles"] = roles
    summary["transforms"] = dict(sorted(transform_counts.items()))
    summary_path = PROJECT_ROOT / "manifests" / "cpo_stcm_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
