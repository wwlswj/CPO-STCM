# SPDX-License-Identifier: AGPL-3.0-only
"""Reimplemented normal-only baselines for external video benchmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class Conv3DAutoencoder(nn.Module):
    """A compact spatiotemporal 3D convolutional autoencoder."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.encoder = nn.Sequential(
            self._block(1, 32, stride=(1, 2, 2)),
            self._block(32, 64, stride=(2, 2, 2)),
            self._block(64, 128, stride=(1, 2, 2)),
            nn.AdaptiveAvgPool3d((2, 6, 9)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 6 * 9, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Linear(latent_dim, 128 * 2 * 6 * 9)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                128,
                64,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                output_padding=(0, 1, 1),
            ),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose3d(
                64,
                32,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Reconstruct a ten-frame difference sequence."""
        encoded = self.encoder(frames.unsqueeze(1))
        decoded = self.project(encoded).reshape(-1, 128, 2, 6, 9)
        decoded = self.decoder(decoded)
        decoded = functional.interpolate(
            decoded,
            size=(frames.shape[1], self.input_height, self.input_width),
            mode="trilinear",
            align_corners=False,
        )
        return decoded.squeeze(1)


class MemoryAutoencoder(nn.Module):
    """Frame-wise convolutional autoencoder with a normal memory bank."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        embedding_dim: int = 256,
        memory_slots: int = 64,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.encoder = nn.Sequential(
            self._encoder_block(1, 32),
            nn.MaxPool2d(2),
            self._encoder_block(32, 64),
            nn.MaxPool2d(2),
            self._encoder_block(64, 128),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((6, 9)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 9, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.memory = nn.Parameter(
            torch.randn(memory_slots, embedding_dim) * 0.02
        )
        self.project = nn.Linear(embedding_dim, 128 * 6 * 9)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _encoder_block(
        input_channels: int,
        output_channels: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct all frames and return memory attention."""
        batch, sequence, height, width = frames.shape
        flattened = frames.reshape(batch * sequence, 1, height, width)
        encoded = self.encoder(flattened)
        attention = torch.softmax(
            encoded @ self.memory.t(),
            dim=1,
        )
        memory_embedding = attention @ self.memory
        decoded = self.project(memory_embedding).reshape(
            batch * sequence,
            128,
            6,
            9,
        )
        decoded = self.decoder(decoded)
        decoded = functional.interpolate(
            decoded,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return decoded.reshape(batch, sequence, height, width), attention


class SpatiotemporalTransformerAutoencoder(nn.Module):
    """Masked patch-transformer autoencoder for difference sequences."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        patch_height: int = 24,
        patch_width: int = 24,
        embedding_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        mask_probability: float = 0.25,
    ):
        super().__init__()
        if input_height % patch_height or input_width % patch_width:
            raise ValueError("Input size must be divisible by patch size")
        self.input_height = input_height
        self.input_width = input_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.grid_height = input_height // patch_height
        self.grid_width = input_width // patch_width
        self.sequence_length = 10
        self.mask_probability = mask_probability
        self.patch_embed = nn.Conv2d(
            1,
            embedding_dim,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )
        token_count = (
            self.sequence_length * self.grid_height * self.grid_width
        )
        self.position = nn.Parameter(
            torch.randn(1, token_count, embedding_dim) * 0.02
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output = nn.Linear(
            embedding_dim,
            patch_height * patch_width,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Masked reconstruction of a difference sequence."""
        batch, sequence, height, width = frames.shape
        flattened = frames.reshape(batch * sequence, 1, height, width)
        patches = self.patch_embed(flattened)
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(
            batch,
            sequence,
            self.grid_height * self.grid_width,
            -1,
        ).flatten(1, 2)
        tokens = tokens + self.position
        if self.training and self.mask_probability > 0:
            mask = (
                torch.rand(tokens.shape[:2], device=tokens.device)
                < self.mask_probability
            )
            tokens = torch.where(
                mask.unsqueeze(-1),
                self.mask_token,
                tokens,
            )
        encoded = self.encoder(tokens)
        decoded = self.output(encoded).reshape(
            batch,
            sequence,
            self.grid_height,
            self.grid_width,
            self.patch_height,
            self.patch_width,
        )
        decoded = decoded.permute(0, 1, 2, 4, 3, 5).reshape(
            batch,
            sequence,
            height,
            width,
        )
        return torch.sigmoid(decoded)
