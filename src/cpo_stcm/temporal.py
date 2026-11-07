# SPDX-License-Identifier: AGPL-3.0-only
"""Normal-only temporal reconstruction models for CPO-STCM."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset

TRANSFORM_PERMUTATIONS = {
    "shuffle_a": (0, 3, 1, 5, 2, 7, 4, 9, 6, 8),
    "shuffle_b": (9, 1, 7, 3, 5, 0, 8, 2, 6, 4),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV manifest."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class TemporalWindowDataset(Dataset):
    """Load ten difference images and apply a temporal transform."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        packed_array_path: Path,
        input_height: int,
        input_width: int,
        training_noise: bool = False,
    ):
        self.rows = rows
        self.packed = np.load(packed_array_path, mmap_mode="r")
        self.input_height = input_height
        self.input_width = input_width
        self.training_noise = training_noise

    def __len__(self) -> int:
        return len(self.rows)

    def _apply_transform(
        self,
        frames: torch.Tensor,
        transform: str,
    ) -> torch.Tensor:
        """Apply a deterministic temporal manipulation."""
        if transform == "none":
            return frames
        if transform == "reverse":
            return torch.flip(frames, dims=(0,))
        if transform in TRANSFORM_PERMUTATIONS:
            return frames[list(TRANSFORM_PERMUTATIONS[transform])]
        if transform == "freeze_first":
            return frames[0:1].repeat(10, 1, 1)
        if transform == "freeze_middle":
            return frames[5:6].repeat(10, 1, 1)
        if transform == "drop_one":
            output = frames.clone()
            output[4] = 0
            return output
        if transform == "drop_three":
            output = frames.clone()
            output[3:6] = 0
            return output
        raise ValueError(f"Unknown temporal transform: {transform}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        raw = np.array(
            self.packed[int(row["packed_index"])],
            copy=True,
        )
        tensor = torch.from_numpy(raw).float().unsqueeze(1) / 255.0
        tensor = functional.interpolate(
            tensor,
            size=(self.input_height, self.input_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        tensor = self._apply_transform(tensor, row["transform"])
        if self.training_noise:
            if random.random() < 0.5:
                tensor = tensor * random.uniform(0.92, 1.08)
            if random.random() < 0.25:
                tensor = (
                    tensor
                    + torch.randn_like(tensor) * random.uniform(0.003, 0.015)
                ).clamp(0.0, 1.0)
        return tensor, int(row["label_abnormal"])


class FrameEncoder(nn.Module):
    """Shared convolutional frame encoder."""

    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32, stride=2),
            self._block(32, 64, stride=2),
            self._block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _block(
        input_channels: int,
        output_channels: int,
        stride: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, sequence, height, width = frames.shape
        flattened = frames.reshape(batch * sequence, 1, height, width)
        encoded = self.features(flattened)
        return encoded.reshape(batch, sequence, -1)


class FrameDecoder(nn.Module):
    """Decode temporal embeddings into difference images."""

    def __init__(
        self,
        embedding_dim: int,
        channels: int,
        feature_height: int,
        feature_width: int,
        output_height: int,
        output_width: int,
    ):
        super().__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_height = output_height
        self.output_width = output_width
        self.project = nn.Linear(
            embedding_dim,
            channels * feature_height * feature_width,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(
                channels // 2,
                channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(
                channels // 4,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = embeddings.shape
        features = self.project(embeddings).reshape(
            batch * sequence,
            -1,
            self.feature_height,
            self.feature_width,
        )
        decoded = self.decoder(features)
        decoded = functional.interpolate(
            decoded,
            size=(self.output_height, self.output_width),
            mode="bilinear",
            align_corners=False,
        )
        return decoded.reshape(batch, sequence, 1, *decoded.shape[-2:])


class TemporalOrderHead(nn.Module):
    """Predict sequence order from local temporal-state transitions."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        intermediate = max(32, hidden_dim // 2)
        self.network = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                intermediate,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                intermediate,
                intermediate,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(intermediate, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return one normal-order logit for each sequence."""
        return self.network(states.transpose(1, 2)).squeeze(1)


class LSTMSequenceAutoencoder(nn.Module):
    """CNN-LSTM sequence autoencoder with an optional order head."""

    def __init__(
        self,
        sequence_length: int,
        input_height: int,
        input_width: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        decoder_channels: int = 128,
        decoder_feature_height: int = 4,
        decoder_feature_width: int = 8,
        dropout: float = 0.1,
        use_order_head: bool = True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.use_order_head = use_order_head
        self.frame_encoder = FrameEncoder(embedding_dim, dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, embedding_dim)
        )
        self.encoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_query = nn.Parameter(
            torch.zeros(1, sequence_length, hidden_dim)
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.frame_decoder = FrameDecoder(
            embedding_dim=embedding_dim,
            channels=decoder_channels,
            feature_height=decoder_feature_height,
            feature_width=decoder_feature_width,
            output_height=input_height,
            output_width=input_width,
        )
        self.order_head = (
            TemporalOrderHead(hidden_dim, dropout)
            if use_order_head
            else None
        )

    def encode(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence and return temporal states and pooled features."""
        encoded = self.frame_encoder(frames)
        encoded = encoded + self.position_embedding[:, : encoded.shape[1]]
        if observed_mask is not None:
            observed = observed_mask.unsqueeze(-1).to(encoded.dtype)
            encoded = (
                encoded * observed
                + self.mask_token.expand_as(encoded) * (1.0 - observed)
            )
        states, (hidden, cell) = self.encoder_lstm(encoded)
        return states, (hidden, cell)

    def forward(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reconstruct a sequence and predict whether its order is normal."""
        states, (hidden, cell) = self.encode(frames, observed_mask)
        queries = self.decoder_query[:, : frames.shape[1]].expand(
            frames.shape[0],
            -1,
            -1,
        )
        decoded_states, _ = self.decoder_lstm(
            queries,
            (hidden, cell),
        )
        embeddings = self.feature_projection(decoded_states)
        reconstruction = self.frame_decoder(embeddings).squeeze(2)
        order_logits = None
        if self.order_head is not None:
            order_logits = self.order_head(states)
        return reconstruction, order_logits


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Compute a differentiable Charbonnier reconstruction loss."""
    return torch.sqrt((prediction - target) ** 2 + epsilon**2).mean()


def _ssim_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute a compact local SSIM map."""
    window_size = 7
    padding = window_size // 2
    mean_x = functional.avg_pool2d(
        prediction,
        window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    mean_y = functional.avg_pool2d(
        target,
        window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    var_x = functional.avg_pool2d(
        prediction**2,
        window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ) - mean_x**2
    var_y = functional.avg_pool2d(
        target**2,
        window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ) - mean_y**2
    covariance = functional.avg_pool2d(
        prediction * target,
        window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ) - mean_x * mean_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (
        (mean_x**2 + mean_y**2 + c1)
        * (var_x + var_y + c2)
    )
    return numerator / denominator.clamp_min(1e-6)


def reconstruction_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    loss_config: dict[str, float],
) -> torch.Tensor:
    """Combine observed, masked, SSIM, and temporal objectives."""
    observed = observed_mask.unsqueeze(-1).unsqueeze(-1).expand_as(
        prediction
    )
    pixel_error = torch.sqrt((prediction - target) ** 2 + 1e-6)
    observed_loss = (
        (pixel_error * observed).sum()
        / observed.sum().clamp_min(1.0)
    )
    masked = 1.0 - observed
    masked_loss = (
        (pixel_error * masked).sum()
        / masked.sum().clamp_min(1.0)
    )
    ssim_loss = 1.0 - _ssim_map(prediction, target).mean()
    temporal_target = target[:, 1:] - target[:, :-1]
    temporal_prediction = prediction[:, 1:] - prediction[:, :-1]
    temporal_loss = functional.l1_loss(
        temporal_prediction,
        temporal_target,
    )
    return (
        float(loss_config["reconstruction"]) * observed_loss
        + float(loss_config["masked_reconstruction"]) * masked_loss
        + float(loss_config["ssim"]) * ssim_loss
        + float(loss_config["temporal"]) * temporal_loss
    )


def temporal_reconstruction_scores(
    model: LSTMSequenceAutoencoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: int = 3,
) -> dict[str, np.ndarray]:
    """Return reconstruction and order evidence for each sample."""
    model.eval()
    sequence_errors = []
    topk_errors = []
    order_anomalies = []
    labels = []
    with torch.inference_mode():
        for frames, targets in loader:
            frames = frames.to(device, non_blocking=True)
            reconstruction, order_logits = model(frames)
            frame_error = (
                (reconstruction - frames).abs().mean(dim=(2, 3))
            )
            topk = torch.topk(
                frame_error,
                k=min(top_k, frame_error.shape[1]),
                dim=1,
            ).values.mean(dim=1)
            sequence_errors.extend(frame_error.mean(dim=1).cpu().tolist())
            topk_errors.extend(topk.cpu().tolist())
            if order_logits is not None:
                order_probability = torch.sigmoid(order_logits)
                order_anomalies.extend(
                    (1.0 - order_probability).cpu().tolist()
                )
            else:
                order_anomalies.extend([0.0] * frames.shape[0])
            labels.extend(targets.tolist())
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "reconstruction_mean": np.asarray(
            sequence_errors,
            dtype=np.float64,
        ),
        "reconstruction_topk": np.asarray(topk_errors, dtype=np.float64),
        "order_anomaly": np.asarray(order_anomalies, dtype=np.float64),
    }
