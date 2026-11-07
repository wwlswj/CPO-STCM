# SPDX-License-Identifier: AGPL-3.0-only
"""Compare global, source-balanced, and observability-conditional calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from cpo_stcm.conformal import (
    conditional_false_positive_rates,
    mondrian_conformal_p_values,
    observability_stratum,
    source_balanced_conformal_p_values,
)
from cpo_stcm.evaluation import conformal_p_values
from cpo_stcm.paths import load_resolved_config

PROJECT_ROOT = Path.cwd().resolve()


def metric_summary(
    labels: np.ndarray,
    p_values: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Summarize one fixed conformal threshold."""
    prediction = p_values <= alpha
    return {
        "false_positive_rate": float(np.mean(prediction[labels == 0])),
        "recall": float(np.mean(prediction[labels == 1])),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "alarm_rate": float(prediction.mean()),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cpo_stcm.yaml",
    )
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
        "--static-evidence",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "conformal"
            / "charge_conditional_v1"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--minimum-stratum-size", type=int, default=20)
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    if args.static_evidence is None:
        args.static_evidence = Path(config["paths"]["static_evidence_path"])
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(args.evidence)
    static = pd.read_csv(args.static_evidence)
    evidence["base_window_id"] = (
        evidence["sequence_id"].astype(str)
        + "|"
        + evidence["start_position"].astype(int).astype(str)
        + "|"
        + evidence["subimage_index"].astype(int).astype(str)
    )
    static["base_window_id"] = (
        static["sequence_id"].astype(str)
        + "|"
        + static["start_position"].astype(int).astype(str)
        + "|"
        + static["subimage_index"].astype(int).astype(str)
    )
    data = evidence.merge(
        static[["base_window_id", "hard_pass_fraction"]],
        on="base_window_id",
        how="left",
        validate="many_to_one",
    )
    if data["hard_pass_fraction"].isna().any():
        raise RuntimeError("Missing static observability evidence")
    data["stratum"] = data["hard_pass_fraction"].map(
        observability_stratum
    )
    calibration_mask = data["role"] == "normal_calibration"
    calibration = data.loc[calibration_mask]
    results = {}
    for split in ("validation", "test", "domain_test"):
        split_frame = data[data["role"] == split].copy()
        labels = split_frame["label_abnormal_int"].to_numpy(dtype=np.int64)
        global_p = conformal_p_values(
            calibration["joint_evidence"].to_numpy(),
            split_frame["joint_evidence"].to_numpy(),
        )
        source_p = source_balanced_conformal_p_values(
            calibration["joint_evidence"].to_numpy(),
            calibration["sequence_id"].astype(str).tolist(),
            split_frame["joint_evidence"].to_numpy(),
        )
        mondrian_p = mondrian_conformal_p_values(
            calibration["joint_evidence"].to_numpy(),
            calibration["stratum"].to_numpy(dtype=np.int64),
            split_frame["joint_evidence"].to_numpy(),
            split_frame["stratum"].to_numpy(dtype=np.int64),
            minimum_stratum_size=args.minimum_stratum_size,
        )
        variants = {
            "global": global_p,
            "source_balanced": source_p,
            "mondrian": mondrian_p,
        }
        results[split] = {}
        for name, p_values in variants.items():
            results[split][name] = {
                "metrics": metric_summary(labels, p_values, args.alpha),
                "conditional": conditional_false_positive_rates(
                    labels,
                    p_values,
                    split_frame["stratum"].to_numpy(dtype=np.int64),
                    args.alpha,
                ),
            }
        split_frame["global_p"] = global_p
        split_frame["source_balanced_p"] = source_p
        split_frame["mondrian_p"] = mondrian_p
        split_frame.to_csv(
            args.output / f"{split}_pvalues.csv",
            index=False,
            encoding="utf-8",
        )
    report = {
        "alpha": args.alpha,
        "minimum_stratum_size": args.minimum_stratum_size,
        "calibration_windows": int(len(calibration)),
        "calibration_sequence_blocks": int(
            calibration["sequence_id"].nunique()
        ),
        "results": results,
    }
    (args.output / "conditional_conformal_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
