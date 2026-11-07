# SPDX-License-Identifier: AGPL-3.0-only
# ruff: noqa: E402
"""Evaluate an external UCSD Ped2 temporal model at window and frame level."""

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
    empirical_upper_tail_anomaly_evidence,
)
from cpo_stcm.temporal import LSTMSequenceAutoencoder, TemporalWindowDataset


def build_model(checkpoint: dict, device: torch.device) -> LSTMSequenceAutoencoder:
    """Restore the external temporal model."""
    config = checkpoint["config"]
    model_config = config["model"]
    model = LSTMSequenceAutoencoder(
        sequence_length=10,
        input_height=int(checkpoint["input_height"]),
        input_width=int(checkpoint["input_width"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["lstm_hidden_dim"]),
        decoder_channels=int(model_config["decoder_feature_channels"]),
        decoder_feature_height=int(model_config["decoder_feature_height"]),
        decoder_feature_width=int(model_config["decoder_feature_width"]),
        dropout=float(model_config["dropout"]),
        use_order_head=checkpoint["variant"] == "enhanced",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


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
    parser.add_argument("--batch-size", type=int, default=32)
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
    device = torch.device(f"cuda:{args.device}")
    model = build_model(checkpoint, device)
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
    reconstruction_mean = []
    reconstruction_topk = []
    order_anomaly = []
    with torch.inference_mode():
        for frames, _ in loader:
            frames = frames.to(device, non_blocking=True)
            reconstruction, order_logits = model(frames)
            errors = (
                (reconstruction - frames).abs().mean(dim=(2, 3))
            )
            reconstruction_mean.extend(errors.mean(dim=1).cpu().tolist())
            reconstruction_topk.extend(
                torch.topk(errors, k=3, dim=1)
                .values.mean(dim=1)
                .cpu()
                .tolist()
            )
            if order_logits is None:
                order_anomaly.extend([0.0] * frames.shape[0])
            else:
                order_anomaly.extend(
                    (1.0 - torch.sigmoid(order_logits)).cpu().tolist()
                )
    frame["reconstruction_mean"] = reconstruction_mean
    frame["reconstruction_topk"] = reconstruction_topk
    frame["order_anomaly"] = order_anomaly
    calibration_mask = frame["role"] == "normal_calibration"
    calibration = frame.loc[calibration_mask]
    frame["reconstruction_evidence"] = (
        empirical_upper_tail_anomaly_evidence(
            calibration["reconstruction_topk"].to_numpy(),
            frame["reconstruction_topk"].to_numpy(),
        )
    )
    frame["order_evidence"] = empirical_upper_tail_anomaly_evidence(
        calibration["order_anomaly"].to_numpy(),
        frame["order_anomaly"].to_numpy(),
    )
    if checkpoint["variant"] == "enhanced":
        frame["dynamic_raw"] = np.maximum(
            frame["reconstruction_evidence"],
            frame["order_evidence"],
        )
    else:
        frame["dynamic_raw"] = frame["reconstruction_evidence"]
    calibration = frame.loc[calibration_mask]
    frame["conformal_p"] = conformal_p_values(
        calibration["dynamic_raw"].to_numpy(),
        frame["dynamic_raw"].to_numpy(),
    )
    frame["anomaly_evidence"] = 1.0 - frame["conformal_p"]
    frame.to_csv(
        args.output_dir / "window_scores.csv",
        index=False,
        encoding="utf-8",
    )
    test = frame[frame["role"] == "test"].copy()
    labels = test["label_abnormal"].to_numpy(dtype=np.int64)
    scores = test["anomaly_evidence"].to_numpy(dtype=np.float64)
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
    frame_rows = []
    for sequence_id, group in test.groupby("sequence_id"):
        for row in group.itertuples():
            frame_rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame": int(row.center_frame),
                    "label": int(row.center_frame_label),
                    "score": float(row.anomaly_evidence),
                    "p_value": float(row.conformal_p),
                }
            )
    frame_scores = pd.DataFrame(frame_rows)
    frame_labels = frame_scores["label"].to_numpy(dtype=np.int64)
    frame_prediction = frame_scores["p_value"].to_numpy() <= 0.05
    frame_metrics = {
        "auroc": float(
            roc_auc_score(frame_labels, frame_scores["score"])
        ),
        "auprc": float(
            average_precision_score(
                frame_labels,
                frame_scores["score"],
            )
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
        "scored_frames": int(len(frame_scores)),
    }
    report = {
        "dataset": "UCSD Ped2",
        "checkpoint": str(args.checkpoint),
        "variant": checkpoint["variant"],
        "protocol": (
            "Ten-frame difference windows with stride one. "
            "Window labels use any anomaly in the corresponding eleven "
            "raw frames. Frame-level scores use the window centered on "
            "the frame."
        ),
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
