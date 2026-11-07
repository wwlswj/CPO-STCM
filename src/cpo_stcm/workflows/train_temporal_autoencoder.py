# SPDX-License-Identifier: AGPL-3.0-only
# ruff: noqa: E402
"""Train baseline or enhanced normal-only LSTM autoencoders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
(PROJECT_ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)


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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from cpo_stcm.paths import load_resolved_config
from cpo_stcm.temporal import (
    LSTMSequenceAutoencoder,
    TemporalWindowDataset,
    reconstruction_objective,
)


def load_config(path: Path) -> dict:
    """Load the YAML configuration."""
    return load_resolved_config(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV manifest."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def seed_everything(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_observation_mask(
    batch_size: int,
    sequence_length: int,
    probability: float,
    maximum_masked: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a binary observation mask for masked reconstruction."""
    mask = torch.ones(
        (batch_size, sequence_length),
        dtype=torch.float32,
        device=device,
    )
    masked_counts = []
    for _ in range(batch_size):
        if random.random() < probability:
            masked_counts.append(random.randint(1, maximum_masked))
        else:
            masked_counts.append(0)
    for row, masked_count in enumerate(masked_counts):
        if masked_count == 0:
            continue
        indices = random.sample(range(sequence_length), masked_count)
        mask[row, indices] = 0.0
    return mask


def evaluate_normal_loss(
    model: LSTMSequenceAutoencoder,
    loader: DataLoader,
    loss_config: dict[str, float],
    device: torch.device,
) -> float:
    """Evaluate reconstruction loss on normal calibration windows."""
    model.eval()
    losses = []
    with torch.inference_mode():
        for frames, _ in loader:
            frames = frames.to(device, non_blocking=True)
            mask = torch.ones(
                frames.shape[:2],
                device=device,
                dtype=torch.float32,
            )
            reconstruction, _ = model(frames, mask)
            loss = reconstruction_objective(
                reconstruction,
                frames,
                mask,
                loss_config,
            )
            losses.append(loss.item())
    return float(np.mean(losses))


def save_training_plot(
    history: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Save the reconstruction training curve."""
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train")
    plt.plot(
        epochs,
        [row["normal_calibration_loss"] for row in history],
        label="Normal calibration",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cpo_stcm.yaml",
    )
    parser.add_argument("--category", choices=("Charge", "PowerOff"), required=True)
    parser.add_argument(
        "--variant",
        choices=("baseline", "enhanced"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-calibration", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    seed = int(config["project"]["random_seed"])
    seed_everything(seed)
    device = torch.device(f"cuda:{int(training_config['device'])}")

    rows = read_rows(Path(config["paths"]["source_manifest"]))
    category_rows = [
        row
        for row in rows
        if row["source"].lower() == args.category.lower()
    ]
    train_rows = [
        row for row in category_rows
        if row["role"] == "normal_train"
    ]
    calibration_rows = [
        row for row in category_rows
        if row["role"] == "normal_calibration"
    ]
    if args.limit_train is not None:
        train_rows = train_rows[: args.limit_train]
    if args.limit_calibration is not None:
        calibration_rows = calibration_rows[: args.limit_calibration]
    if not train_rows or not calibration_rows:
        raise RuntimeError(
            f"Missing normal train/calibration rows for {args.category}"
        )

    run_name = args.run_name or (
        f"{args.category.lower()}_{args.variant}_v1"
    )
    run_root = PROJECT_ROOT / "experiments" / "temporal_autoencoder" / run_name
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    figure_root = run_root / "figures"
    figure_root.mkdir()

    packed_path = Path(config["paths"]["packed_array"])
    train_dataset = TemporalWindowDataset(
        train_rows,
        packed_path,
        int(data_config["input_height"]),
        int(data_config["input_width"]),
        training_noise=True,
    )
    calibration_dataset = TemporalWindowDataset(
        calibration_rows,
        packed_path,
        int(data_config["input_height"]),
        int(data_config["input_width"]),
    )
    batch_size = int(args.batch_size or training_config["batch_size"])
    loader_args = {
        "batch_size": batch_size,
        "num_workers": int(training_config["num_workers"]),
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
    use_order_head = args.variant == "enhanced"
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
        patience=4,
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_config = training_config["loss"]
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history: list[dict[str, float]] = []
    epochs = int(args.epochs or training_config["epochs"])
    started_at = datetime.now(timezone.utc).isoformat()

    for epoch in range(1, epochs + 1):
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
                    permuted_frames = frames[:, permutation]
                    reversed_frames = torch.flip(frames, dims=(1,))
                    negative_frames = torch.cat(
                        (permuted_frames, reversed_frames),
                        dim=0,
                    )
                    negative_states, _ = model.encode(negative_frames)
                    negative_logits = model.order_head(negative_states)
                    order_targets = torch.ones_like(order_logits)
                    negative_targets = torch.zeros_like(negative_logits)
                    order_loss = functional.binary_cross_entropy_with_logits(
                        torch.cat((order_logits, negative_logits)),
                        torch.cat((order_targets, negative_targets)),
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
            "train_loss": float(np.mean(train_losses)),
            "normal_calibration_loss": calibration_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
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
                    "category": args.category,
                    "config": config,
                },
                run_root / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= int(training_config["patience"]):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No best model state recorded")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": history[-1]["epoch"],
            "variant": args.variant,
            "category": args.category,
            "config": config,
        },
        run_root / "last.pt",
    )
    model.load_state_dict(best_state)
    history_path = run_root / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_training_plot(history, figure_root / "training_loss.png")
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        errors="replace",
    )
    report = {
        "category": args.category,
        "variant": args.variant,
        "run_name": run_name,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_normal_calibration_loss": best_loss,
        "history": history,
        "train_samples": len(train_dataset),
        "calibration_samples": len(calibration_dataset),
        "training_config": {
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "mask_probability": float(model_config["mask_probability"]),
            "mask_max_frames": int(model_config["mask_max_frames"]),
            "loss": loss_config,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "packages": freeze.splitlines(),
        },
    }
    (run_root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
