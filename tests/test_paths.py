# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for relocatable software configuration paths."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cpo_stcm.paths import (
    default_config_path,
    load_resolved_config,
    project_root,
)


class ConfigPathTests(unittest.TestCase):
    """Validate path resolution without requiring downloaded data."""

    def test_source_config_paths_are_resolved(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_resolved_config(
            project_root / "configs" / "cpo_stcm.yaml"
        )

        for name, path in config["paths"].items():
            with self.subTest(path=name):
                self.assertIsInstance(path, Path)
                self.assertTrue(path.is_absolute(), path)

        self.assertEqual(
            config["paths"]["project_root"],
            project_root.resolve(),
        )
        self.assertEqual(
            config["paths"]["packed_array"],
            (project_root / "data" / "processed" /
             "dynamic_windows_v1.npy").resolve(),
        )

    def test_relative_paths_resolve_from_config_directory(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        temporary_root = project_root / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            config_dir = Path(directory) / "configs"
            config_dir.mkdir()
            target = Path(directory) / "datasets" / "sample.npy"
            target.parent.mkdir()
            target.touch()
            config_path = config_dir / "experiment.yaml"
            config_path.write_text(
                "paths:\n"
                "  packed_array: ../datasets/sample.npy\n",
                encoding="utf-8",
            )

            config = load_resolved_config(config_path)

            self.assertEqual(config["paths"]["packed_array"], target.resolve())

    def test_project_root_override_and_environment(self) -> None:
        explicit = project_root("relative-root")
        self.assertTrue(explicit.is_absolute())
        self.assertEqual(explicit.name, "relative-root")
        temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            with mock.patch.dict(
                os.environ,
                {"CPO_STCM_PROJECT_ROOT": directory},
            ):
                self.assertEqual(project_root(), Path(directory).resolve())
                self.assertEqual(
                    default_config_path(),
                    Path(directory).resolve() / "configs" / "cpo_stcm.yaml",
                )


if __name__ == "__main__":
    unittest.main()
