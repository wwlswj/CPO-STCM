# SPDX-License-Identifier: AGPL-3.0-only
# ruff: noqa: E402
"""Evaluate reimplemented Conv3D and memory autoencoders on UCSD Ped2."""

from __future__ import annotations

import argparse
import json
import os
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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from cpo_stcm.evaluation import (
    conformal_p_values,
    empirical_two_sided_anomaly_evidence,
)
from cpo_stcm.external_baselines import (
    Conv3DAutoencoder,
    MemoryAutoencoder,
    SpatiotemporalTransformerAutoencoder,
)
from cpo_stcm.temporal import TemporalWindowDataset


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets"
            / "external"
            / "UCSDped2-Windows-v1"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    architecture = checkpoint["architecture"]
    if architecture == "conv3d":
        model = Conv3DAutoencoder(
            int(checkpoint["input_height"]),
            int(checkpoint["input_width"]),
        )
    elif architecture == "memae":
        model = MemoryAutoencoder(
            int(checkpoint["input_height"]),
            int(checkpoint["input_width"]),
        )
    else:
        model = SpatiotemporalTransformerAutoencoder(
            int(checkpoint["input_height"]),
            int(checkpoint["input_width"]),
        )
    device = torch.device(f"cuda:{args.device}")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    frame = pd.read_csv(args.dataset_root / "manifest.csv")
    frame = frame[
        frame["role"].isin(
            ["normal_calibration", "validation", "test"]
        )
    ].reset_index(drop=True)
    dataset = TemporalWindowDataset(
        frame.to_dict("records"),
        args.dataset_root / "ped2_difference_windows.npy",
        int(checkpoint["input_height"]),
        int(checkpoint["input_width"]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    mean_errors = []
    topk_errors = []
    with torch.inference_mode():
        for frames, _ in loader:
            frames = frames.to(device, non_blocking=True)
            output = model(frames)
            reconstruction = output[0] if isinstance(output, tuple) else output
            errors = (
                (reconstruction - frames).abs().mean(dim=(2, 3))
            )
            mean_errors.extend(errors.mean(dim=1).cpu().tolist())
            topk_errors.extend(
                torch.topk(errors, k=3, dim=1)
                .values.mean(dim=1)
                .cpu()
                .tolist()
            )
    frame["reconstruction_mean"] = mean_errors
    frame["reconstruction_topk"] = topk_errors
    calibration_mask = frame["role"] == "normal_calibration"
    calibration = frame.loc[calibration_mask]
    frame["reconstruction_evidence"] = (
        empirical_two_sided_anomaly_evidence(
            calibration["reconstruction_topk"].to_numpy(),
            frame["reconstruction_topk"].to_numpy(),
        )
    )
    frame["conformal_p"] = conformal_p_values(
        calibration["reconstruction_topk"].to_numpy(),
        frame["reconstruction_topk"].to_numpy(),
    )
    frame.to_csv(
        args.output_dir / "window_scores.csv",
        index=False,
        encoding="utf-8",
    )
    test = frame[frame["role"] == "test"]
    labels = test["label_abnormal"].to_numpy(dtype=np.int64)
    scores = test["reconstruction_topk"].to_numpy(dtype=np.float64)
    prediction = test["conformal_p"].to_numpy(dtype=np.float64) <= 0.05
    window_metrics = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1_at_fpr_5pct": float(
            f1_score(labels, prediction, zero_division=0)
        ),
        "fpr_at_fpr_5pct": float(
            np.mean(prediction[labels == 0])
        ),
        "recall_at_fpr_5pct": float(
            np.mean(prediction[labels == 1])
        ),
    }
    frame_labels = test["center_frame_label"].to_numpy(dtype=np.int64)
    frame_scores = test["reconstruction_topk"].to_numpy(dtype=np.float64)
    frame_prediction = test["conformal_p"].to_numpy() <= 0.05
    frame_metrics = {
        "auroc": float(roc_auc_score(frame_labels, frame_scores)),
        "auprc": float(
            average_precision_score(frame_labels, frame_scores)
        ),
        "f1_at_fpr_5pct": float(
            f1_score(
                frame_labels,
                frame_prediction,
                zero_division=0,
            )
        ),
        "fpr_at_fpr_5pct": float(
            np.mean(frame_prediction[frame_labels == 0])
        ),
        "recall_at_fpr_5pct": float(
            np.mean(frame_prediction[frame_labels == 1])
        ),
        "scored_frames": int(len(test)),
    }
    report = {
        "dataset": "UCSD Ped2",
        "architecture": architecture,
        "checkpoint": str(args.checkpoint),
        "window_metrics": window_metrics,
        "frame_metrics": frame_metrics,
    }
    (args.output_dir / "external_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
