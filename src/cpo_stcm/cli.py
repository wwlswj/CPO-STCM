# SPDX-License-Identifier: AGPL-3.0-only
"""Command-line interface for CPO-STCM."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from cpo_stcm import __version__
from cpo_stcm.data import (
    CPO_LED_ARCHIVE,
    DatasetError,
    build_packed_array,
    extract_archive,
    fetch_cpo_led_bench,
    verify_archive,
)
from cpo_stcm.paths import load_project_config


WORKFLOWS = {
    "bootstrap-external": "cpo_stcm.workflows.bootstrap_external_results",
    "build-manifests": "cpo_stcm.workflows.build_normal_only_manifests",
    "evaluate-conformal": (
        "cpo_stcm.workflows.evaluate_conditional_conformal"
    ),
    "evaluate-deletion": (
        "cpo_stcm.workflows.evaluate_static_deletion_stress"
    ),
    "evaluate-external-baseline": (
        "cpo_stcm.workflows.evaluate_external_baseline"
    ),
    "evaluate-external-ucsd": (
        "cpo_stcm.workflows.evaluate_external_ucsd"
    ),
    "evaluate-joint": "cpo_stcm.workflows.evaluate_joint_model",
    "evaluate-natural": "cpo_stcm.workflows.evaluate_natural_faults",
    "infer-temporal": "cpo_stcm.workflows.run_temporal_inference",
    "precompute-static": "cpo_stcm.workflows.precompute_static_evidence",
    "prepare-ucsd": "cpo_stcm.workflows.prepare_ucsd_ped2",
    "train-external-autoencoder": (
        "cpo_stcm.workflows.train_external_autoencoder"
    ),
    "train-external-baseline": (
        "cpo_stcm.workflows.train_external_baseline"
    ),
    "train-temporal": "cpo_stcm.workflows.train_temporal_autoencoder",
}

OPTIONAL_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "PyYAML",
    "Pillow",
    "opencv-python",
    "torch",
    "pandas",
    "matplotlib",
    "ultralytics",
)


def _package_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing it."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolved_paths(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Serialize configured paths with availability information."""
    result: dict[str, dict[str, Any]] = {}
    for name, path in config.get("paths", {}).items():
        result[name] = {
            "path": str(path),
            "exists": path.exists(),
        }
    return result


def _doctor(args: argparse.Namespace) -> int:
    """Print environment, configuration, and optional-dependency status."""
    root, config = load_project_config(args.project_root, args.config)
    paths = _resolved_paths(config)
    archive_info = paths.get("archive_path")
    report = {
        "status": "ready",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(root),
        "config": str(
            Path(args.config).expanduser()
            if args.config
            else root / "configs" / "cpo_stcm.yaml"
        ),
        "paths": paths,
        "packages": {
            name: _package_version(name) for name in OPTIONAL_PACKAGES
        },
        "dataset": {
            "version": CPO_LED_ARCHIVE.version,
            "archive_url": CPO_LED_ARCHIVE.url,
            "archive_sha256": CPO_LED_ARCHIVE.sha256,
            "archive_present": bool(archive_info and archive_info["exists"]),
        },
        "warnings": [],
    }
    required_packages = ("numpy", "scipy", "scikit-learn", "PyYAML", "Pillow")
    missing = [
        name
        for name in required_packages
        if report["packages"][name] is None
    ]
    if missing:
        report["status"] = "incomplete"
        report["warnings"].append(
            "Missing base dependencies: " + ", ".join(missing)
        )
    if not report["dataset"]["archive_present"]:
        report["warnings"].append(
            "CPO-LED Bench archive is not downloaded; run 'cpo-stcm data fetch'."
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ready" else 1


def _demo(args: argparse.Namespace) -> int:
    """Run a deterministic, dependency-light conformal calibration demo."""
    from cpo_stcm.conformal import mondrian_conformal_p_values

    rng = np.random.default_rng(args.seed)
    calibration = rng.normal(size=args.calibration_size)
    queries = np.asarray(
        [np.quantile(calibration, 0.50), np.quantile(calibration, 0.99), 5.0],
        dtype=np.float64,
    )
    p_values = mondrian_conformal_p_values(
        calibration_scores=calibration,
        calibration_strata=np.zeros(len(calibration), dtype=np.int64),
        test_scores=queries,
        test_strata=np.zeros(len(queries), dtype=np.int64),
        minimum_stratum_size=1,
    )
    output = {
        "seed": args.seed,
        "calibration_size": args.calibration_size,
        "alpha": args.alpha,
        "queries": queries.tolist(),
        "p_values": p_values.tolist(),
        "alarms": (p_values <= args.alpha).astype(int).tolist(),
    }
    print(json.dumps(output, indent=2))
    return 0


def _data_fetch(args: argparse.Namespace) -> int:
    """Download, verify, and extract the public benchmark."""
    root, config = load_project_config(args.project_root, args.config)
    cache_root = Path(args.data_root) if args.data_root else config["paths"][
        "data_cache_root"
    ]
    if not cache_root.is_absolute():
        cache_root = root / cache_root
    archive, extracted = fetch_cpo_led_bench(
        cache_root,
        extract=not args.no_extract,
        force_download=args.force_download,
    )
    print(
        json.dumps(
            {
                "archive": str(archive.resolve()),
                "sha256": CPO_LED_ARCHIVE.sha256,
                "size": archive.stat().st_size,
                "extracted": str(extracted.resolve()) if extracted else None,
            },
            indent=2,
        )
    )
    return 0


def _data_verify(args: argparse.Namespace) -> int:
    """Verify a downloaded CPO-LED archive against pinned size and hash."""
    root, config = load_project_config(args.project_root, args.config)
    archive_path = Path(args.archive) if args.archive else config["paths"][
        "archive_path"
    ]
    if not archive_path.is_absolute():
        archive_path = root / archive_path
    digest = verify_archive(archive_path)
    print(
        json.dumps(
            {
                "archive": str(archive_path.resolve()),
                "size": archive_path.stat().st_size,
                "sha256": digest,
                "status": "verified",
            },
            indent=2,
        )
    )
    return 0


def _data_build_packed(args: argparse.Namespace) -> int:
    """Build the NumPy training array from published benchmark images."""
    root, config = load_project_config(args.project_root, args.config)
    configured = config["paths"]
    dataset_root = Path(args.dataset_root) if args.dataset_root else configured[
        "dataset_root"
    ]
    output_path = Path(args.output) if args.output else configured["packed_array"]
    if not dataset_root.is_absolute():
        dataset_root = root / dataset_root
    if not output_path.is_absolute():
        output_path = root / output_path
    subset_image_root = dataset_root / "images"
    subset_manifest = dataset_root / "manifests" / "cpo_stcm_samples.csv"
    report = build_packed_array(
        subset_image_root,
        subset_manifest,
        output_path,
        sequence_length=int(config["data"]["sequence_length"]),
        expected_height=int(config["data"]["input_height"]),
        expected_width=int(config["data"]["input_width"]),
        max_windows=args.limit,
        progress=lambda done, total: print(
            f"packed {done}/{total} windows",
            file=sys.stderr,
            flush=True,
        ),
        compute_checksum=not args.skip_checksum,
    )
    print(json.dumps(report.__dict__, indent=2))
    return 0


def _workflow(args: argparse.Namespace) -> int:
    """Dispatch a legacy research workflow with unchanged arguments."""
    module = importlib.import_module(WORKFLOWS[args.name])
    main_function: Callable[[], None] | None = getattr(module, "main", None)
    if main_function is None:
        raise RuntimeError(
            f"Workflow module {WORKFLOWS[args.name]} has no main()"
        )
    previous_argv = sys.argv
    try:
        sys.argv = [f"cpo-stcm workflow {args.name}", *args.workflow_args]
        main_function()
    finally:
        sys.argv = previous_argv
    return 0


def _config_show(args: argparse.Namespace) -> int:
    """Print the resolved project configuration."""
    root, config = load_project_config(args.project_root, args.config)
    printable = {
        "project_root": str(root),
        **config,
    }
    printable["paths"] = _resolved_paths(config)
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return 0


def _version(_: argparse.Namespace) -> int:
    """Print the package version."""
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the public command parser."""
    parser = argparse.ArgumentParser(
        prog="cpo-stcm",
        description=(
            "Normal-only LED-animation anomaly detection with "
            "partially observed semantic consistency."
        ),
    )
    parser.add_argument(
        "--project-root",
        help="Project directory containing configs/, data/, and experiments/.",
    )
    parser.add_argument(
        "--config",
        help="Configuration YAML path; defaults to <project-root>/configs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect runtime readiness")
    doctor.set_defaults(handler=_doctor)

    demo = subparsers.add_parser("demo", help="Run an offline calibration demo")
    demo.add_argument("--seed", type=int, default=20261103)
    demo.add_argument("--calibration-size", type=int, default=200)
    demo.add_argument("--alpha", type=float, default=0.05)
    demo.set_defaults(handler=_demo)

    config_parser = subparsers.add_parser(
        "config",
        help="Inspect resolved project configuration",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        required=True,
    )
    config_show = config_subparsers.add_parser("show")
    config_show.set_defaults(handler=_config_show)

    data_parser = subparsers.add_parser("data", help="Manage dataset assets")
    data_subparsers = data_parser.add_subparsers(
        dest="data_command",
        required=True,
    )
    data_fetch = data_subparsers.add_parser("fetch")
    data_fetch.add_argument("--data-root", type=Path)
    data_fetch.add_argument("--no-extract", action="store_true")
    data_fetch.add_argument("--force-download", action="store_true")
    data_fetch.set_defaults(handler=_data_fetch)

    data_verify = data_subparsers.add_parser("verify")
    data_verify.add_argument("--archive", type=Path)
    data_verify.set_defaults(handler=_data_verify)

    data_build = data_subparsers.add_parser("build-packed")
    data_build.add_argument("--dataset-root", type=Path)
    data_build.add_argument("--output", type=Path)
    data_build.add_argument(
        "--limit",
        type=int,
        help="Pack only the first N indexed windows for a smoke test.",
    )
    data_build.add_argument("--skip-checksum", action="store_true")
    data_build.set_defaults(handler=_data_build_packed)

    workflow = subparsers.add_parser(
        "workflow",
        help="Run a training, inference, or evaluation workflow",
    )
    workflow.add_argument("name", choices=sorted(WORKFLOWS))
    workflow.add_argument("workflow_args", nargs=argparse.REMAINDER)
    workflow.set_defaults(handler=_workflow)

    version = subparsers.add_parser("version", help="Print software version")
    version.set_defaults(handler=_version)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (DatasetError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"cpo-stcm: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
