# SPDX-License-Identifier: AGPL-3.0-only
"""Relocatable path handling for CPO-STCM experiment configurations."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def project_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the active project root without requiring a source checkout.

    The explicit ``override`` has highest priority, followed by the
    ``CPO_STCM_PROJECT_ROOT`` environment variable and the current working
    directory. A project root is the directory containing ``configs/``,
    ``data/``, ``experiments/`` and ``manifests/`` when the full workflow is
    used.
    """
    selected = override or os.environ.get("CPO_STCM_PROJECT_ROOT") or os.getcwd()
    return Path(selected).expanduser().resolve(strict=False)


def default_config_path(
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the conventional CPO-STCM configuration path."""
    return project_root(root) / "configs" / "cpo_stcm.yaml"


def resolve_config_path(config_dir: Path, value: str | os.PathLike[str]) -> Path:
    """Resolve one configured path relative to its configuration file.

    Environment variables and user-home markers are expanded before relative
    paths are anchored to the configuration directory. The function does not
    require the target to exist, so it can also validate a future output path.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve(strict=False)


def resolve_config_paths(
    config: dict[str, Any],
    config_dir: Path,
) -> dict[str, Any]:
    """Return a config copy whose ``paths`` values are absolute ``Path`` objects."""
    resolved = copy.deepcopy(config)
    configured_paths = resolved.get("paths", {})
    if not isinstance(configured_paths, dict):
        raise TypeError("The config 'paths' section must be a mapping")
    resolved["paths"] = {
        name: resolve_config_path(config_dir, value)
        for name, value in configured_paths.items()
    }
    return resolved


def load_resolved_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve all paths relative to it."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Unable to read YAML configuration: {config_path}"
        ) from exc
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must contain a mapping: {config_path}")
    return resolve_config_paths(config, config_path.parent)


def load_project_config(
    root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Load the project configuration and return its root and resolved values."""
    selected_root = project_root(root)
    selected_config = (
        Path(config_path).expanduser()
        if config_path is not None
        else default_config_path(selected_root)
    )
    if not selected_config.is_absolute():
        selected_config = selected_root / selected_config
    return selected_root, load_resolved_config(selected_config)
