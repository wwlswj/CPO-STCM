# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate production-derived natural anomalies without temporal transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path.cwd().resolve()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "joint_evaluation"
            / "charge_cpo_stcm_v3"
            / "calibrated_evidence.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "natural_faults"
            / "charge_v1"
        ),
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20),
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.evidence)
    natural_classes = {"animated", "static_animation", "no_device"}
    test = data[
        (data["role"] == "test")
        & data["subclass"].isin(natural_classes)
    ].copy()
    test["class_name"] = test["subclass"].map(
        {
            "animated": "normal",
            "static_animation": "natural_static_animation",
            "no_device": "natural_device_absent",
        }
    )
    calibration = data[data["role"] == "normal_calibration"]
    methods = {
        "temporal_only": "dynamic_evidence",
        "static_only": "static_evidence",
        "CPO_STCM": "joint_evidence",
    }
    report: dict[str, object] = {
        "alphas": list(args.alphas),
        "test_windows": int(len(test)),
        "normal_windows": int((test["label_abnormal_int"] == 0).sum()),
        "natural_abnormal_windows": int(
            (test["label_abnormal_int"] == 1).sum()
        ),
        "methods": {},
    }
    for name, column in methods.items():
        calibration_scores = calibration[column].to_numpy(dtype=np.float64)
        test_scores = test[column].to_numpy(dtype=np.float64)
        p_values = (
            1.0
            + np.sum(
                calibration_scores[:, None] >= test_scores[None, :],
                axis=0,
            )
        ) / (len(calibration_scores) + 1.0)
        labels = test["label_abnormal_int"].to_numpy(dtype=np.int64)
        method_report: dict[str, object] = {
            "auroc": float(roc_auc_score(labels, test_scores)),
            "auprc": float(
                average_precision_score(labels, test_scores)
            ),
            "operating_points": {},
        }
        for alpha in args.alphas:
            prediction = p_values <= alpha
            class_recall = {}
            for class_name in sorted(test["class_name"].unique()):
                if class_name == "normal":
                    continue
                indices = test["class_name"].to_numpy() == class_name
                class_recall[class_name] = float(
                    np.mean(prediction[indices])
                )
            method_report["operating_points"][f"alpha_{alpha:.2f}"] = {
                "f1": float(
                    f1_score(labels, prediction, zero_division=0)
                ),
                "false_positive_rate": float(
                    np.mean(prediction[labels == 0])
                ),
                "recall": float(np.mean(prediction[labels == 1])),
                "class_recall": class_recall,
            }
        report["methods"][name] = method_report
    test.to_csv(
        args.output / "natural_fault_windows.csv",
        index=False,
        encoding="utf-8",
    )
    (args.output / "natural_fault_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
