# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for dataset integrity and safe archive extraction."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from cpo_stcm.data import (
    DatasetError,
    build_packed_array,
    extract_archive,
    sha256_file,
)


class ArchiveTests(unittest.TestCase):
    """Validate minimal dataset integrity guarantees."""

    def _temporary_directory(self):
        """Create a sandbox-safe temporary directory inside the project."""
        root = Path(__file__).resolve().parents[1] / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_sha256_file(self) -> None:
        with self._temporary_directory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"cpo-stcm")
            self.assertEqual(
                sha256_file(path),
                "8f6b6d1c4af103d551b97210705d7b41"
                "21095ae59e6d448808b7be38a69f6b6f",
            )

    def test_extract_archive_creates_expected_directory(self) -> None:
        with self._temporary_directory() as directory:
            root = Path(directory)
            archive = root / "safe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "CPO-LED-Bench-v2.0/README.md",
                    "benchmark",
                )
            extracted = extract_archive(archive, root / "output")
            self.assertTrue((extracted / "README.md").is_file())

    def test_extract_archive_rejects_parent_traversal(self) -> None:
        with self._temporary_directory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "unsafe")
            with self.assertRaises(DatasetError):
                extract_archive(archive, root / "output")
            self.assertFalse((root / "outside.txt").exists())

    def test_build_packed_array_from_indexed_images(self) -> None:
        with self._temporary_directory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            index_rows = ["image_type,base_window_id,frame_index,relative_path"]
            manifest_rows = [
                "sequence_id,start_position,subimage_index,packed_index"
            ]
            expected = np.zeros((2, 2, 3, 4), dtype=np.uint8)
            for packed_index, sequence in enumerate(("A", "B")):
                base_id = f"charge/{sequence}|0|0"
                manifest_rows.append(f"charge/{sequence},0,0,{packed_index}")
                for frame_index in range(1, 3):
                    relative = (
                        f"dynamic/charge/{sequence}/difference_{frame_index:02d}.png"
                    )
                    path = image_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value = packed_index * 20 + frame_index
                    Image.fromarray(
                        np.full((3, 4), value, dtype=np.uint8)
                    ).save(path)
                    index_rows.append(
                        "dynamic_difference,"
                        f"{base_id},{frame_index},{relative}"
                    )
                    expected[packed_index, frame_index - 1] = value
            (image_root / "image_index.csv").write_text(
                "\n".join(index_rows) + "\n",
                encoding="utf-8",
            )
            manifest = root / "samples.csv"
            manifest.write_text(
                "\n".join(manifest_rows) + "\n",
                encoding="utf-8",
            )
            output = root / "packed.npy"
            report = build_packed_array(
                image_root,
                manifest,
                output,
                sequence_length=2,
                expected_height=3,
                expected_width=4,
                compute_checksum=False,
            )
            self.assertEqual(report.windows, 2)
            np.testing.assert_array_equal(np.load(output), expected)


if __name__ == "__main__":
    unittest.main()
