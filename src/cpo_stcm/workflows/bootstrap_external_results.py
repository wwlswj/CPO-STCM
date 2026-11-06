# SPDX-License-Identifier: AGPL-3.0-only
"""Compute source-cluster bootstrap intervals for UCSD Ped2 comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path.cwd().resolve()


def cluster_bootstrap(
    frame: pd.DataFrame,
    score_column: str,
    label_column: str,
    seed: int,
    iterations: int = 1000,
) -> dict[str, object]:
    """Estimate a video-cluster bootstrap interval for frame-level AUROC.

    Args:
        frame: Test split containing one row per centered frame.
        score_column: Continuous anomaly score used for ranking.
        label_column: Binary ground-truth column evaluated by AUROC.
        seed: Reproducibility seed for cluster resampling.
        iterations: Number of bootstrap resamples.

    Returns:
        Point AUROC, percentile interval, and retained bootstrap count.

    Raises:
        ValueError: If the selected labels do not contain both classes.
    """
    groups = frame["sequence_id"].astype(str).unique()
    indices_by_group = {
        group: np.flatnonzero(frame["sequence_id"].astype(str) == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate(
            [indices_by_group[group] for group in sampled_groups]
        )
        sampled = frame.iloc[indices]
        labels = sampled[label_column].to_numpy(dtype=np.int64)
        if len(np.unique(labels)) < 2:
            continue
        values.append(
            float(
                roc_auc_score(
                    labels,
                    sampled[score_column].to_numpy(dtype=np.float64),
                )
            )
        )
    labels = frame[label_column].to_numpy(dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"Label column {label_column!r} must contain both classes."
        )
    point = float(
        roc_auc_score(
            labels,
            frame[score_column].to_numpy(dtype=np.float64),
        )
    )
    return {
        "auroc": point,
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "bootstrap_samples": len(values),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "external"
            / "ucsd_ped2"
            / "bootstrap_results.json"
        ),
    )
    args = parser.parse_args()
    paths = {
        "CPO-STCM baseline reconstruction": (
            PROJECT_ROOT
            / "experiments"
            / "external"
            / "ucsd_ped2"
            / "baseline_v1"
            / "evaluation_upper_v2"
            / "window_scores.csv",
            "reconstruction_topk",
        ),
        "CPO-STCM enhanced reconstruction": (
            PROJECT_ROOT
            / "experiments"
            / "external"
            / "ucsd_ped2"
            / "enhanced_v1"
            / "evaluation_upper_v1"
            / "window_scores.csv",
            "reconstruction_topk",
        ),
        "CPO-STCM enhanced joint": (
            PROJECT_ROOT
            / "experiments"
            / "external"
            / "ucsd_ped2"
            / "enhanced_v1"
            / "evaluation_upper_v1"
            / "window_scores.csv",
            "dynamic_raw",
        ),
        "Conv3D autoencoder": (
            PROJECT_ROOT
            / "experiments"
            / "external"
            / "ucsd_ped2"
            / "conv3d_v1"
            / "evaluation"
            / "window_scores.csv",
            "reconstruction_topk",
        ),
        "Memory autoencoder": (
            PROJECT_ROOT
            / "experiments"
            / "external"
            / "ucsd_ped2"
            / "memae_v1"
            / "evaluation"
            / "window_scores.csv",
            "reconstruction_topk",
        ),
        "Spatiotemporal transformer": (
            PROJECT_ROOT
            / "experiments"
            / "external"
            / "ucsd_ped2"
            / "transformer_v1"
            / "evaluation"
            / "window_scores.csv",
            "reconstruction_topk",
        ),
    }
    output = {}
    for index, (name, (path, column)) in enumerate(paths.items()):
        frame = pd.read_csv(path)
        frame = frame[frame["role"] == "test"].reset_index(drop=True)
        output[name] = cluster_bootstrap(
            frame,
            column,
            "center_frame_label",
            seed=20261103 + index,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
