# SPDX-License-Identifier: AGPL-3.0-only
"""Run a trained temporal autoencoder on calibration and evaluation windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
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
import torch
from torch.utils.data import DataLoader


from cpo_stcm.temporal import LSTMSequenceAutoencoder, TemporalWindowDataset
from cpo_stcm.paths import load_resolved_config


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_config(path: Path) -> dict:
    """Load the project YAML configuration."""
    return load_resolved_config(path)


def file_sha256(path: Path) -> str:
    """Compute a file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint: dict, device: torch.device) -> LSTMSequenceAutoencoder:
    """Restore a model from a training checkpoint."""
    config = checkpoint["config"]
    model_config = config["model"]
    data_config = config["data"]
    model = LSTMSequenceAutoencoder(
        sequence_length=int(data_config["sequence_length"]),
        input_height=int(data_config["input_height"]),
        input_width=int(data_config["input_width"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["lstm_hidden_dim"]),
        decoder_channels=int(model_config["decoder_feature_channels"]),
        decoder_feature_height=int(model_config["decoder_feature_height"]),
        decoder_feature_width=int(model_config["decoder_feature_width"]),
        dropout=float(model_config["dropout"]),
        use_order_head=str(checkpoint["variant"]) == "enhanced",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cpo_stcm.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category", choices=("Charge", "PowerOff"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--store-examples", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(f"cuda:{args.device}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["category"].lower() != args.category.lower():
        raise ValueError(
            "Checkpoint category does not match --category: "
            f"{checkpoint['category']} != {args.category}"
        )
    model = build_model(checkpoint, device)
    rows = [
        row
        for row in read_rows(
            Path(config["paths"]["source_manifest"])
        )
        if row["source"].lower() == args.category.lower()
        and row["role"]
        in {"normal_calibration", "validation", "test", "domain_test"}
    ]
    dataset = TemporalWindowDataset(
        rows,
        Path(config["paths"]["packed_array"]),
        int(config["data"]["input_height"]),
        int(config["data"]["input_width"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    top_k = int(config["scoring"]["top_k_frames"])
    output_rows: list[dict[str, object]] = []
    example_payload: list[dict[str, object]] = []
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (frames, labels) in enumerate(loader):
            frames = frames.to(device, non_blocking=True)
            reconstruction, order_logits = model(frames)
            frame_errors = (
                (reconstruction - frames).abs().mean(dim=(2, 3))
            )
            reconstruction_mean = frame_errors.mean(dim=1)
            reconstruction_topk = torch.topk(
                frame_errors,
                k=min(top_k, frame_errors.shape[1]),
                dim=1,
            ).values.mean(dim=1)
            if order_logits is None:
                order_anomaly = torch.zeros_like(reconstruction_mean)
            else:
                order_anomaly = 1.0 - torch.sigmoid(order_logits)
            for row_index in range(frames.shape[0]):
                row = rows[offset + row_index]
                output_rows.append(
                    {
                        **row,
                        "label_abnormal_int": int(labels[row_index]),
                        "reconstruction_mean": float(
                            reconstruction_mean[row_index].cpu()
                        ),
                        "reconstruction_topk": float(
                            reconstruction_topk[row_index].cpu()
                        ),
                        "order_anomaly": float(
                            order_anomaly[row_index].cpu()
                        ),
                        "frame_reconstruction_errors": json.dumps(
                            [
                                round(float(value), 8)
                                for value in frame_errors[row_index].cpu()
                            ]
                        ),
                    }
                )
                if len(example_payload) < args.store_examples:
                    example_payload.append(
                        {
                            "sample_id": row["sample_id"],
                            "subclass": row["subclass"],
                            "input": frames[row_index, :, None].cpu().numpy(),
                            "reconstruction": reconstruction[
                                row_index, :, None
                            ].cpu().numpy(),
                        }
                    )
            offset += frames.shape[0]
            if batch_index % 20 == 0:
                print(
                    f"scored {offset}/{len(rows)} windows",
                    flush=True,
                )
    if offset != len(rows):
        raise RuntimeError(f"Scored {offset} rows, expected {len(rows)}")
    prediction_path = args.output_dir / "temporal_scores.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)
    if example_payload:
        np.savez_compressed(
            args.output_dir / "temporal_examples.npz",
            sample_ids=np.asarray(
                [item["sample_id"] for item in example_payload]
            ),
            subclasses=np.asarray(
                [item["subclass"] for item in example_payload]
            ),
            inputs=np.stack([item["input"] for item in example_payload]),
            reconstructions=np.stack(
                [item["reconstruction"] for item in example_payload]
            ),
        )
    elapsed = time.perf_counter() - started
    report = {
        "category": args.category,
        "variant": checkpoint["variant"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "rows": len(output_rows),
        "elapsed_seconds": elapsed,
        "milliseconds_per_window": 1000.0 * elapsed / len(output_rows),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    (args.output_dir / "inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
