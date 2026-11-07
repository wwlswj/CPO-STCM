# SPDX-License-Identifier: AGPL-3.0-only
# ruff: noqa: E402
"""Train CPO-STCM temporal autoencoders on an external public benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
import yaml
from torch.utils.data import DataLoader

from cpo_stcm.temporal import (
    LSTMSequenceAutoencoder,
    TemporalWindowDataset,
    reconstruction_objective,
)
from cpo_stcm.workflows.train_temporal_autoencoder import (
    evaluate_normal_loss,
    random_observation_mask,
    save_training_plot,
    seed_everything,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    """Read a manifest CSV."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_config(path: Path) -> dict:
    """Read the project configuration."""
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cpo_stcm.yaml",
    )
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
    parser.add_argument(
        "--variant",
        choices=("baseline", "enhanced"),
        required=True,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
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
    figure_root = run_root / "figures"
    figure_root.mkdir()

    config = load_config(args.config)
    seed_everything(int(config["project"]["random_seed"]))
    device = torch.device(f"cuda:{args.device}")
    rows = load_rows(args.dataset_root / "manifest.csv")
    train_rows = [
        row for row in rows if row["role"] == "normal_train"
    ]
    calibration_rows = [
        row for row in rows if row["role"] == "normal_calibration"
    ]
    train_dataset = TemporalWindowDataset(
        train_rows,
        args.dataset_root / "ped2_difference_windows.npy",
        args.height,
        args.width,
        training_noise=True,
    )
    calibration_dataset = TemporalWindowDataset(
        calibration_rows,
        args.dataset_root / "ped2_difference_windows.npy",
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
    model_config = config["model"]
    training_config = config["training"]
    use_order_head = args.variant == "enhanced"
    model = LSTMSequenceAutoencoder(
        sequence_length=10,
        input_height=args.height,
        input_width=args.width,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["lstm_hidden_dim"]),
        decoder_channels=int(model_config["decoder_feature_channels"]),
        decoder_feature_height=int(model_config["decoder_feature_height"]),
        decoder_feature_width=int(model_config["decoder_feature_width"]),
        dropout=float(model_config["dropout"]),
        use_order_head=use_order_head,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_config = training_config["loss"]
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []
    started_at = datetime.now(timezone.utc).isoformat()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses = []
        for frames, _ in train_loader:
            frames = frames.to(device, non_blocking=True)
            if use_order_head:
                observed_mask = random_observation_mask(
                    frames.shape[0],
                    frames.shape[1],
                    float(model_config["mask_probability"]),
                    int(model_config["mask_max_frames"]),
                    device,
                )
            else:
                observed_mask = torch.ones(
                    frames.shape[:2],
                    dtype=torch.float32,
                    device=device,
                )
            visible_frames = frames * observed_mask.unsqueeze(-1).unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                reconstruction, order_logits = model(
                    visible_frames,
                    observed_mask,
                )
                loss = reconstruction_objective(
                    reconstruction,
                    frames,
                    observed_mask,
                    loss_config,
                )
                if use_order_head and order_logits is not None:
                    permutation = torch.randperm(frames.shape[1])
                    negatives = torch.cat(
                        (
                            frames[:, permutation],
                            torch.flip(frames, dims=(1,)),
                        ),
                        dim=0,
                    )
                    negative_states, _ = model.encode(negatives)
                    negative_logits = model.order_head(negative_states)
                    order_loss = functional.binary_cross_entropy_with_logits(
                        torch.cat(
                            (
                                order_logits,
                                negative_logits,
                            )
                        ),
                        torch.cat(
                            (
                                torch.ones_like(order_logits),
                                torch.zeros_like(negative_logits),
                            )
                        ),
                    )
                    loss = loss + float(loss_config["order"]) * order_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        calibration_loss = evaluate_normal_loss(
            model,
            calibration_loader,
            loss_config,
            device,
        )
        scheduler.step(calibration_loss)
        row = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_losses),
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
                    "epoch": best_epoch,
                    "variant": args.variant,
                    "dataset": "ucsd_ped2",
                    "input_height": args.height,
                    "input_width": args.width,
                    "config": config,
                },
                run_root / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    save_training_plot(history, figure_root / "training_loss.png")
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
        "variant": args.variant,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "train_windows": len(train_dataset),
        "calibration_windows": len(calibration_dataset),
        "input_height": args.height,
        "input_width": args.width,
        "history": history,
    }
    (run_root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
