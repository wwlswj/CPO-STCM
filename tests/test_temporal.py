# SPDX-License-Identifier: AGPL-3.0-only
"""Optional PyTorch model tests."""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

if os.environ.get("CPO_STCM_SKIP_ML_TESTS") == "1":
    raise unittest.SkipTest("CPO_STCM_SKIP_ML_TESTS=1")

try:
    import torch

    from cpo_stcm.external_baselines import (
        SpatiotemporalTransformerAutoencoder,
    )
    from cpo_stcm.temporal import TemporalOrderHead, TemporalWindowDataset
except (ImportError, OSError) as exc:
    TORCH_IMPORT_ERROR = exc
    torch = None
    SpatiotemporalTransformerAutoencoder = None
    TemporalOrderHead = None
    TemporalWindowDataset = None
else:
    TORCH_IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"PyTorch unavailable: {TORCH_IMPORT_ERROR}")
class TemporalHeadTests(unittest.TestCase):
    """Test the order head and optional model interfaces."""

    def test_temporal_head_shape(self) -> None:
        head = TemporalOrderHead(hidden_dim=32, dropout=0.0)
        states = torch.randn(4, 10, 32)
        logits = head(states)
        self.assertEqual(tuple(logits.shape), (4,))

    def test_transformer_baseline_shape(self) -> None:
        model = SpatiotemporalTransformerAutoencoder(
            input_height=48,
            input_width=48,
            patch_height=24,
            patch_width=24,
            embedding_dim=32,
            depth=1,
            heads=4,
        )
        output = model(torch.randn(2, 10, 48, 48))
        self.assertEqual(tuple(output.shape), (2, 10, 48, 48))

    def test_memmap_input_has_no_read_only_warning(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        temporary_root = project_root / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=temporary_root,
            ignore_cleanup_errors=True,
        ) as directory:
            packed_path = Path(directory) / "windows.npy"
            np.save(
                packed_path,
                np.full((1, 10, 20, 30), 127, dtype=np.uint8),
            )
            dataset = TemporalWindowDataset(
                [
                    {
                        "packed_index": "0",
                        "transform": "none",
                        "label_abnormal": "0",
                    }
                ],
                packed_path,
                input_height=10,
                input_width=15,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                frames, label = dataset[0]

            self.assertEqual(tuple(frames.shape), (10, 10, 15))
            self.assertTrue(torch.isfinite(frames).all())
            self.assertEqual(label, 0)
            self.assertFalse(
                any(
                    "non-writable" in str(item.message)
                    for item in caught
                )
            )
            del dataset


if __name__ == "__main__":
    unittest.main()
