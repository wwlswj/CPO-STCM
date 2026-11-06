# SPDX-License-Identifier: AGPL-3.0-only
"""Train reimplemented strong normal-only baselines on UCSD Ped2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
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

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader


from cpo_stcm.external_baselines import (
    Conv3DAutoencoder,
    MemoryAutoencoder,
    SpatiotemporalTransformerAutoencoder,
)
from cpo_stcm.temporal import TemporalWindowDataset
from cpo_stcm.workflows.train_temporal_autoencoder import (
    save_training_plot,
    seed_everything,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a manifest."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    memory_attention: torch.Tensor | None,
) -> torch.Tensor:
    """Combine reconstruction, temporal-difference, and memory sparsity."""
    pixel = torch.sqrt((prediction - target) ** 2 + 1e-6).mean()
    temporal = functional.l1_loss(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
    )
    loss = pixel + 0.1 * temporal
    if memory_attention is not None:
        loss = loss + 1e-3 * memory_attention.abs().mean()
    return loss


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("conv3d", "memae", "transformer"),
        required=True,
    )
    parser.add_argument("--run-name", required=True)
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
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    run_root = (
        PROJECT_ROOT
        / "experiments"
        / "external"
        / "ucsd_ped2"
        / args.run_name
    )
    if run_root.exists():
        raise FileExistsError(f"Run directory exists: {run_root}")
    run_root.mkdir(parents=True)
    (run_root / "figures").mkdir()
    seed_everything(20261103)
    device = torch.device(f"cuda:{args.device}")
    rows = read_rows(args.dataset_root / "manifest.csv")
    train_rows = [
        row for row in rows if row["role"] == "normal_train"
    ]
    calibration_rows = [
        row for row in rows if row["role"] == "normal_calibration"
    ]
    packed_path = args.dataset_root / "ped2_difference_windows.npy"
    train_dataset = TemporalWindowDataset(
        train_rows,
        packed_path,
        args.height,
        args.width,
        training_noise=True,
    )
    calibration_dataset = TemporalWindowDataset(
        calibration_rows,
        packed_path,
        args.height,
        args.width,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        shuffle=False,
        **loader_args,
    )
    if args.architecture == "conv3d":
        model = Conv3DAutoencoder(args.height, args.width).to(device)
    elif args.architecture == "memae":
        model = MemoryAutoencoder(args.height, args.width).to(device)
    else:
        model = SpatiotemporalTransformerAutoencoder(
            args.height,
            args.width,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []
    started_at = datetime.now(timezone.utc).isoformat()

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.inference_mode():
            for frames, _ in calibration_loader:
                frames = frames.to(device, non_blocking=True)
                output = model(frames)
                if isinstance(output, tuple):
                    reconstruction, attention = output
                else:
                    reconstruction, attention = output, None
                losses.append(
                    reconstruction_loss(
                        reconstruction,
                        frames,
                        attention,
                    ).item()
                )
        return float(sum(losses) / len(losses))

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for frames, _ in train_loader:
            frames = frames.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                output = model(frames)
                if isinstance(output, tuple):
                    reconstruction, attention = output
                else:
                    reconstruction, attention = output, None
                loss = reconstruction_loss(
                    reconstruction,
                    frames,
                    attention,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        calibration_loss = evaluate()
        scheduler.step(calibration_loss)
        row = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / len(losses)),
            "normal_calibration_loss": calibration_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state_dict": best_state,
                    "architecture": args.architecture,
                    "input_height": args.height,
                    "input_width": args.width,
                    "epoch": best_epoch,
                },
                run_root / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    save_training_plot(history, run_root / "figures" / "training_loss.png")
    with (run_root / "history.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)
    report = {
        "dataset": "UCSD Ped2",
        "architecture": args.architecture,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "train_windows": len(train_dataset),
        "calibration_windows": len(calibration_dataset),
        "history": history,
    }
    (run_root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
