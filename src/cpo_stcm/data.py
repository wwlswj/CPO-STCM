# SPDX-License-Identifier: AGPL-3.0-only
"""Dataset acquisition and packing helpers for CPO-LED Bench.

The public benchmark stores portable PNG files in a GitHub Release archive.
The training workflow consumes a memory-mappable NumPy array, so this module
provides a validated one-time conversion from the published image index to
``dynamic_windows_v1.npy``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CPO_LED_VERSION = "2.0.0"
CPO_LED_RELEASE_TAG = "v2.0.0"
CPO_LED_ARCHIVE_NAME = "CPO-LED-Bench-v2.0-full.zip"
CPO_LED_ARCHIVE_BYTES = 395_563_944
CPO_LED_ARCHIVE_SHA256 = (
    "a462f4c144b4675aaeb7ccdc819066653fd9500615843996f941226c05a3b19d"
)
CPO_LED_REPOSITORY = "https://github.com/wwlswj/CPO-LED-Bench"
CPO_LED_ARCHIVE_URL = (
    f"{CPO_LED_REPOSITORY}/releases/download/"
    f"{CPO_LED_RELEASE_TAG}/{CPO_LED_ARCHIVE_NAME}"
)
CPO_LED_EXTRACTED_DIR = "CPO-LED-Bench-v2.0"


class DatasetError(RuntimeError):
    """Raised when a dataset asset cannot be downloaded or validated."""


@dataclass(frozen=True)
class DatasetArchive:
    """Description of one immutable benchmark archive."""

    version: str
    release_tag: str
    url: str
    filename: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PackedArrayReport:
    """Summary of a completed image-to-NumPy packing operation."""

    output_path: str
    windows: int
    allocated_rows: int
    sequence_length: int
    height: int
    width: int
    bytes: int
    sha256: str


CPO_LED_ARCHIVE = DatasetArchive(
    version=CPO_LED_VERSION,
    release_tag=CPO_LED_RELEASE_TAG,
    url=CPO_LED_ARCHIVE_URL,
    filename=CPO_LED_ARCHIVE_NAME,
    size=CPO_LED_ARCHIVE_BYTES,
    sha256=CPO_LED_ARCHIVE_SHA256,
)


def sha256_file(
    path: str | os.PathLike[str],
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return the SHA-256 digest of a regular file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(
    path: str | os.PathLike[str],
    expected_sha256: str = CPO_LED_ARCHIVE_SHA256,
    expected_size: int | None = CPO_LED_ARCHIVE_BYTES,
) -> str:
    """Verify an archive hash and optional byte size, returning its digest."""
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise DatasetError(f"Dataset archive is not a ZIP file: {archive_path}")
    if expected_size is not None and archive_path.stat().st_size != expected_size:
        actual = archive_path.stat().st_size
        raise DatasetError(
            f"Dataset archive size mismatch: expected {expected_size}, got {actual}"
        )
    digest = sha256_file(archive_path)
    if digest.lower() != expected_sha256.lower():
        raise DatasetError(
            "Dataset archive SHA-256 mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    return digest


def download_archive(
    output_path: str | os.PathLike[str],
    *,
    url: str = CPO_LED_ARCHIVE_URL,
    expected_sha256: str = CPO_LED_ARCHIVE_SHA256,
    expected_size: int | None = CPO_LED_ARCHIVE_BYTES,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download and atomically install the immutable CPO-LED archive.

    A partial file is written beside the destination and renamed only after
    size and SHA-256 validation succeed. An existing valid archive is reused
    unless ``force`` is explicitly requested.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        verify_archive(destination, expected_sha256, expected_size)
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CPO-STCM/1.0 dataset-fetch"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            advertised = response.headers.get("Content-Length")
            if (
                expected_size is not None
                and advertised
                and int(advertised) != expected_size
            ):
                raise DatasetError(
                    "Remote archive size differs from the pinned release: "
                    f"expected {expected_size}, got {advertised}"
                )
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise DatasetError(f"Unable to download {url}: {exc}") from exc

    try:
        verify_archive(temporary, expected_sha256, expected_size)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _safe_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    """Yield archive members after rejecting absolute and escaping paths."""
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise DatasetError(f"Unsafe ZIP member path: {member.filename}")
        yield member


def extract_archive(
    archive_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    """Extract a verified archive without allowing path traversal."""
    archive = Path(archive_path)
    destination = Path(output_dir)
    extracted = destination / CPO_LED_EXTRACTED_DIR
    if extracted.exists() and not force:
        if not extracted.is_dir():
            raise DatasetError(f"Extraction target is not a directory: {extracted}")
        return extracted
    if force and extracted.exists():
        raise DatasetError(
            "Refusing to overwrite an existing extracted dataset. "
            "Remove or relocate it explicitly before retrying."
        )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination, members=_safe_members(bundle))
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"Unable to extract {archive}: {exc}") from exc
    if not extracted.is_dir():
        raise DatasetError(
            f"Archive did not create the expected directory: {extracted}"
        )
    return extracted


def fetch_cpo_led_bench(
    data_root: str | os.PathLike[str],
    *,
    extract: bool = True,
    force_download: bool = False,
) -> tuple[Path, Path | None]:
    """Download, verify, and optionally extract CPO-LED Bench v2.0."""
    root = Path(data_root)
    archive = download_archive(root / CPO_LED_ARCHIVE_NAME, force=force_download)
    extracted = extract_archive(archive, root) if extract else None
    return archive, extracted


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _window_key(row: dict[str, str]) -> str:
    """Return the image-index base-window identifier for a sample row."""
    return (
        f"{row['sequence_id']}|{row['start_position']}|"
        f"{row['subimage_index']}"
    )


def build_packed_array(
    image_root: str | os.PathLike[str],
    source_manifest: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    sequence_length: int = 10,
    expected_height: int = 100,
    expected_width: int = 300,
    max_windows: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    compute_checksum: bool = True,
) -> PackedArrayReport:
    """Pack published dynamic images into the training-time NumPy layout.

    ``image_index.csv`` identifies ten portable PNG difference images for each
    spatial base window. ``cpo_stcm_samples.csv`` supplies the stable
    ``packed_index`` expected by the training manifest. The output is written
    atomically and is suitable for ``mmap_mode="r"`` loading.
    """
    images = Path(image_root)
    index_path = images / "image_index.csv"
    index_rows = _read_csv(index_path)
    sample_rows = _read_csv(Path(source_manifest))

    base_to_index: dict[str, int] = {}
    for row in sample_rows:
        key = _window_key(row)
        packed_index = int(row["packed_index"])
        previous = base_to_index.setdefault(key, packed_index)
        if previous != packed_index:
            raise DatasetError(
                f"Base window {key!r} maps to multiple packed indices"
            )
    if not base_to_index:
        raise DatasetError("The source manifest contains no packed windows")
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        base_to_index = dict(
            sorted(base_to_index.items(), key=lambda item: item[1])[
                :max_windows
            ]
        )

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in index_rows:
        if row.get("image_type") != "dynamic_difference":
            continue
        key = row["base_window_id"]
        if key in base_to_index:
            grouped.setdefault(key, []).append(row)
    missing = sorted(set(base_to_index) - set(grouped))
    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetError(
            f"{len(missing)} manifest windows are absent from image_index.csv: "
            f"{preview}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.part.npy")
    temporary.unlink(missing_ok=True)
    packed_count = max(base_to_index.values()) + 1
    packed = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(
            packed_count,
            sequence_length,
            expected_height,
            expected_width,
        ),
    )
    try:
        for position, (key, packed_index) in enumerate(
            sorted(base_to_index.items(), key=lambda item: item[1]),
            start=1,
        ):
            frames = sorted(
                grouped[key],
                key=lambda row: int(row["frame_index"]),
            )
            if len(frames) != sequence_length:
                raise DatasetError(
                    f"{key} contains {len(frames)} frames; "
                    f"expected {sequence_length}"
                )
            for frame_index, row in enumerate(frames):
                image_path = images / row["relative_path"]
                with Image.open(image_path) as image:
                    frame = np.asarray(image.convert("L"), dtype=np.uint8)
                expected_shape = (expected_height, expected_width)
                if frame.shape != expected_shape:
                    raise DatasetError(
                        f"Unexpected image shape for {image_path}: "
                        f"{frame.shape}, expected {expected_shape}"
                    )
                packed[packed_index, frame_index] = frame
            if progress is not None and (
                position % 250 == 0 or position == len(base_to_index)
            ):
                progress(position, len(base_to_index))
        packed.flush()
        del packed
        packed = None
        temporary.replace(output)
    except Exception:
        if packed is not None:
            del packed
        temporary.unlink(missing_ok=True)
        raise

    report = PackedArrayReport(
        output_path=str(output.resolve()),
        windows=len(base_to_index),
        allocated_rows=packed_count,
        sequence_length=sequence_length,
        height=expected_height,
        width=expected_width,
        bytes=output.stat().st_size,
        sha256=sha256_file(output) if compute_checksum else "NOT_COMPUTED",
    )
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )
    return report
