# SPDX-License-Identifier: AGPL-3.0-only
"""Calibrate and evaluate CPO-STCM and evidence-fusion baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path.cwd().resolve()

from cpo_stcm.evaluation import (
    anomaly_type_recall,
    binary_metrics,
    conformal_p_values,
    empirical_two_sided_anomaly_evidence,
    empirical_upper_tail_anomaly_evidence,
    grouped_bootstrap_intervals,
    select_f1_threshold,
)

GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with interpreted numeric fields."""
    return pd.read_csv(path)


def add_base_window_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the spatial base-window identifier used by static evidence."""
    frame = frame.copy()
    frame["base_window_id"] = (
        frame["sequence_id"].astype(str)
        + "|"
        + frame["start_position"].astype(int).astype(str)
        + "|"
        + frame["subimage_index"].astype(int).astype(str)
    )
    return frame


def add_upper_tail_evidence(
    frame: pd.DataFrame,
    calibration_mask: pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    """Add high-is-abnormal evidence for selected upper-tailed features."""
    frame = frame.copy()
    for column in columns:
        frame[f"{column}_evidence"] = empirical_upper_tail_anomaly_evidence(
            frame.loc[calibration_mask, column].to_numpy(dtype=np.float64),
            frame[column].to_numpy(dtype=np.float64),
        )
    return frame


def add_two_sided_evidence(
    frame: pd.DataFrame,
    calibration_mask: pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    """Add high-is-abnormal evidence for features with two-sided shifts."""
    frame = frame.copy()
    for column in columns:
        frame[f"{column}_evidence"] = empirical_two_sided_anomaly_evidence(
            frame.loc[calibration_mask, column].to_numpy(dtype=np.float64),
            frame[column].to_numpy(dtype=np.float64),
        )
    return frame


def add_composite_evidence(
    frame: pd.DataFrame,
    calibration_mask: pd.Series,
    raw_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Calibrate a predefined composite anomaly statistic on normal windows."""
    frame = frame.copy()
    calibration_values = frame.loc[
        calibration_mask,
        raw_column,
    ].to_numpy(dtype=np.float64)
    frame[output_column] = 1.0 - conformal_p_values(
        calibration_values,
        frame[raw_column].to_numpy(dtype=np.float64),
    )
    return frame


def select_grid_weight(
    frame: pd.DataFrame,
    label_column: str,
    left_column: str,
    right_column: str,
) -> tuple[float, list[dict[str, float]]]:
    """Select a two-evidence weight on validation AUROC."""
    from sklearn.metrics import roc_auc_score

    labels = frame[label_column].to_numpy(dtype=np.int64)
    curve: list[dict[str, float]] = []
    best_weight = 0.5
    best_auc = -1.0
    for weight in GRID:
        score = (
            float(weight) * frame[left_column].to_numpy(dtype=np.float64)
            + (1.0 - float(weight))
            * frame[right_column].to_numpy(dtype=np.float64)
        )
        auc = float(roc_auc_score(labels, score))
        curve.append({"weight": float(weight), "validation_auroc": auc})
        if auc > best_auc + 1e-12:
            best_auc = auc
            best_weight = float(weight)
    return best_weight, curve


def select_joint_weight(
    frame: pd.DataFrame,
    label_column: str,
    dynamic_column: str,
    static_column: str,
) -> tuple[float, list[dict[str, float]]]:
    """Select a dynamic/static fusion weight on validation AUROC."""
    return select_grid_weight(
        frame,
        label_column,
        dynamic_column,
        static_column,
    )


def evaluate_scores(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    domain: pd.DataFrame,
    calibration: pd.DataFrame,
    label_column: str,
    validation_score: np.ndarray,
    test_score: np.ndarray,
    domain_score: np.ndarray,
    calibration_score: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Evaluate one score with validation F1 and conformal FPR control."""
    validation_labels = validation[label_column].to_numpy(dtype=np.int64)
    test_labels = test[label_column].to_numpy(dtype=np.int64)
    domain_labels = domain[label_column].to_numpy(dtype=np.int64)
    validation_f1_threshold = select_f1_threshold(
        validation_labels,
        validation_score,
    )
    validation_metrics = binary_metrics(
        validation_labels,
        validation_score,
        validation_f1_threshold,
    )
    test_metrics = binary_metrics(
        test_labels,
        test_score,
        validation_f1_threshold,
    )
    domain_metrics = binary_metrics(
        domain_labels,
        domain_score,
        validation_f1_threshold,
    )
    conformal: dict[str, object] = {}
    for alpha in (0.01, 0.05):
        test_p_value = conformal_p_values(calibration_score, test_score)
        domain_p_value = conformal_p_values(
            calibration_score,
            domain_score,
        )
        test_anomaly_score = 1.0 - test_p_value
        domain_anomaly_score = 1.0 - domain_p_value
        threshold = 1.0 - alpha
        test_result = binary_metrics(
            test_labels,
            test_anomaly_score,
            threshold,
        )
        domain_result = binary_metrics(
            domain_labels,
            domain_anomaly_score,
            threshold,
        )
        test_result["anomaly_type_recall"] = anomaly_type_recall(
            test_labels,
            test_anomaly_score,
            test["subclass"].astype(str).tolist(),
            threshold,
        )
        domain_result["anomaly_type_recall"] = anomaly_type_recall(
            domain_labels,
            domain_anomaly_score,
            domain["subclass"].astype(str).tolist(),
            threshold,
        )
        if abs(alpha - 0.05) < 1e-12:
            test_result["grouped_bootstrap_95ci"] = (
                grouped_bootstrap_intervals(
                    test_labels,
                    test_anomaly_score,
                    test["sequence_id"].astype(str).tolist(),
                    threshold,
                    seed,
                )
            )
        conformal[f"alpha_{alpha:.2f}"] = {
            "test": test_result,
            "domain_test": domain_result,
        }
    return {
        "validation_f1_threshold": validation_f1_threshold,
        "validation": validation_metrics,
        "test_at_validation_f1_threshold": test_metrics,
        "domain_test_at_validation_f1_threshold": domain_metrics,
        "conformal": conformal,
    }


def prepare_dynamic_frame(
    path: Path,
    static_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Merge dynamic scores with static evidence."""
    dynamic = add_base_window_id(load_csv(path))
    static_columns = [
        "base_window_id",
        "layout_anomaly_best",
        "layout_anomaly_mean",
        "layout_anomaly_worst",
        "hard_pass_fraction",
        "missing_slot_rate",
        "persistent_missing_count",
        "persistent_missing_slots",
    ]
    merged = dynamic.merge(
        static_frame[static_columns],
        on="base_window_id",
        how="left",
        validate="many_to_one",
    )
    required_static_columns = [
        "layout_anomaly_best",
        "missing_slot_rate",
        "persistent_missing_count",
    ]
    if merged[required_static_columns].isna().any().any():
        missing = int(merged["layout_anomaly_best"].isna().sum())
        raise RuntimeError(
            f"{missing} dynamic rows have no matching static evidence"
        )
    return merged


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhanced-scores", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="charge")
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_frame = add_base_window_id(
        load_csv(args.static_evidence)
    )
    static_frame = static_frame[
        static_frame["source"].astype(str).str.lower()
        == args.source.lower()
    ]
    reports: dict[str, object] = {}
    weight_curves: dict[str, object] = {}
    evidence_frames: dict[str, pd.DataFrame] = {}
    for variant, path in (
        ("enhanced", args.enhanced_scores),
        ("baseline", args.baseline_scores),
    ):
        frame = prepare_dynamic_frame(path, static_frame)
        calibration_mask = frame["role"] == "normal_calibration"
        validation_mask = frame["role"] == "validation"
        test_mask = frame["role"] == "test"
        domain_mask = frame["role"] == "domain_test"
        if not all(
            mask.any()
            for mask in (
                calibration_mask,
                validation_mask,
                test_mask,
                domain_mask,
            )
        ):
            raise RuntimeError(f"Missing evaluation roles for {variant}")
        frame = add_upper_tail_evidence(
            frame,
            calibration_mask,
            [
                "layout_anomaly_best",
                "layout_anomaly_mean",
                "missing_slot_rate",
                "persistent_missing_count",
            ],
        )
        frame = add_two_sided_evidence(
            frame,
            calibration_mask,
            [
                "reconstruction_mean",
                "reconstruction_topk",
                "order_anomaly",
            ],
        )
        frame["motion_energy_deficit"] = -frame["motion_energy"]
        frame = add_upper_tail_evidence(
            frame,
            calibration_mask,
            ["motion_energy_deficit"],
        )
        frame["dynamic_raw"] = np.maximum(
            frame["reconstruction_topk_evidence"],
            frame["order_anomaly_evidence"],
        )
        frame["static_raw"] = np.maximum(
            frame["layout_anomaly_best_evidence"],
            frame["persistent_missing_count_evidence"],
        )
        frame["joint_raw"] = np.maximum(
            frame["dynamic_raw"],
            frame["static_raw"],
        )
        frame = add_composite_evidence(
            frame,
            calibration_mask,
            "dynamic_raw",
            "dynamic_evidence",
        )
        frame = add_composite_evidence(
            frame,
            calibration_mask,
            "static_raw",
            "static_evidence",
        )
        frame = add_composite_evidence(
            frame,
            calibration_mask,
            "joint_raw",
            "joint_evidence",
        )
        validation = frame.loc[validation_mask].copy()
        frame["fixed_fusion"] = 0.5 * (
            frame["dynamic_evidence"] + frame["static_evidence"]
        )
        logistic_features = [
            "reconstruction_topk_evidence",
            "order_anomaly_evidence",
            "layout_anomaly_best_evidence",
            "persistent_missing_count_evidence",
        ]
        logistic = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=args.seed,
        )
        logistic.fit(
            validation[logistic_features].to_numpy(dtype=np.float64),
            validation["label_abnormal_int"].to_numpy(dtype=np.int64),
        )
        frame["logistic_fusion"] = logistic.decision_function(
            frame[logistic_features].to_numpy(dtype=np.float64)
        )
        score_columns = {
            "motion_energy_rule": "motion_energy_deficit_evidence",
            "reconstruction_mean": "reconstruction_mean_evidence",
            "reconstruction_topk": "reconstruction_topk_evidence",
            "order_only": "order_anomaly_evidence",
            "static_layout_only": "layout_anomaly_best_evidence",
            "static_persistence_only": "persistent_missing_count_evidence",
            "static_only": "static_evidence",
            "dynamic_only": "dynamic_evidence",
            "fixed_weight_fusion": "fixed_fusion",
            "logistic_fusion": "logistic_fusion",
            "CPO_STCM": "joint_evidence",
        }
        calibration = frame.loc[calibration_mask]
        validation = frame.loc[validation_mask]
        test = frame.loc[test_mask]
        domain = frame.loc[domain_mask]
        variant_report: dict[str, object] = {
            "model_variant": variant,
            "fusion_rule": {
                "temporal": "maximum of reconstruction and order evidence",
                "static": "maximum of layout and persistence evidence",
                "joint": "maximum of temporal and static evidence",
            },
            "counts": {
                "calibration": len(calibration),
                "validation": len(validation),
                "test": len(test),
                "domain_test": len(domain),
            },
            "methods": {},
        }
        for method_name, column in score_columns.items():
            variant_report["methods"][method_name] = evaluate_scores(
                validation,
                test,
                domain,
                calibration,
                "label_abnormal_int",
                validation[column].to_numpy(dtype=np.float64),
                test[column].to_numpy(dtype=np.float64),
                domain[column].to_numpy(dtype=np.float64),
                calibration[column].to_numpy(dtype=np.float64),
                args.seed,
            )
        reports[variant] = variant_report
        weight_curves[variant] = {}
        evidence_frames[variant] = frame

    primary = evidence_frames["enhanced"].copy()
    evidence_columns = [
        "sample_id",
        "source",
        "sequence_id",
        "subimage_index",
        "start_position",
        "role",
        "subclass",
        "transform",
        "label_abnormal_int",
        "motion_energy",
        "motion_energy_deficit_evidence",
        "reconstruction_mean_evidence",
        "reconstruction_topk_evidence",
        "order_anomaly_evidence",
        "layout_anomaly_best_evidence",
        "persistent_missing_count_evidence",
        "dynamic_raw",
        "static_raw",
        "joint_raw",
        "dynamic_evidence",
        "static_evidence",
        "joint_evidence",
        "logistic_fusion",
    ]
    primary[evidence_columns].to_csv(
        args.output_dir / "calibrated_evidence.csv",
        index=False,
        encoding="utf-8",
    )
    summary_rows = []
    for variant, variant_report in reports.items():
        for method_name, method_report in variant_report["methods"].items():
            summary_rows.append(
                {
                    "model_variant": variant,
                    "method": method_name,
                    "test_auroc": method_report[
                        "conformal"
                    ]["alpha_0.05"]["test"]["auroc"],
                    "test_auprc": method_report[
                        "conformal"
                    ]["alpha_0.05"]["test"]["auprc"],
                    "test_f1_at_fpr_5pct": method_report[
                        "conformal"
                    ]["alpha_0.05"]["test"]["f1"],
                    "test_fpr_at_fpr_5pct": method_report[
                        "conformal"
                    ]["alpha_0.05"]["test"]["false_positive_rate"],
                    "test_recall_at_fpr_5pct": method_report[
                        "conformal"
                    ]["alpha_0.05"]["test"]["recall"],
                    "domain_test_auroc": method_report[
                        "conformal"
                    ]["alpha_0.05"]["domain_test"]["auroc"],
                    "domain_test_f1_at_fpr_5pct": method_report[
                        "conformal"
                    ]["alpha_0.05"]["domain_test"]["f1"],
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        args.output_dir / "evaluation_summary.csv",
        index=False,
        encoding="utf-8",
    )
    report = {
        "source": args.source,
        "selection_protocol": {
            "branch_fusion": (
                "predefined maximum-statistic fusion before calibration"
            ),
            "evidence_calibration": (
                "normal-calibration empirical tail probabilities"
            ),
            "f1_threshold": "validation only",
            "conformal_fpr_control": [
                0.01,
                0.05,
            ],
            "test_usage": "single frozen evaluation after all selection",
        },
        "reports": reports,
        "weight_curves": weight_curves,
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
