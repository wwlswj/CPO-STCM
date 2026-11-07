# SPDX-License-Identifier: AGPL-3.0-only
"""Create Zenodo draft records and upload files without publishing them."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

ZENODO_API = "https://zenodo.org/api"
TIMEOUT = (30, 300)


class ZenodoError(RuntimeError):
    """Raised when Zenodo rejects a draft operation."""


def load_metadata(path: Path) -> dict[str, Any]:
    """Load and normalize Zenodo metadata for the legacy deposition API."""
    data = json.loads(path.read_text(encoding="utf-8"))
    related = data.get("related_identifiers", [])
    for item in related:
        item.pop("scheme", None)
        item.setdefault("resource_type", data.get("upload_type", "other"))
    data["related_identifiers"] = related
    data.setdefault("language", "eng")
    data.setdefault("publication_date", "2026-11-07")
    return data


def response_error(response: requests.Response) -> ZenodoError:
    """Return a concise error without exposing authorization headers."""
    try:
        payload = response.json()
        detail = json.dumps(payload, ensure_ascii=False)
    except ValueError:
        detail = response.text[:1000]
    return ZenodoError(
        f"Zenodo API request failed ({response.status_code}): {detail}"
    )


def make_session(token: str) -> requests.Session:
    """Create an authenticated requests session."""
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
    )
    return session


def list_depositions(session: requests.Session) -> list[dict[str, Any]]:
    """List all visible depositions for the authenticated account."""
    response = session.get(
        f"{ZENODO_API}/deposit/depositions",
        params={"size": 100, "sort": "mostrecent"},
        timeout=TIMEOUT,
    )
    if not response.ok:
        raise response_error(response)
    return response.json()


def find_draft(
    depositions: Iterable[dict[str, Any]],
    title: str,
) -> dict[str, Any] | None:
    """Find an unsubmitted deposition with an exact title."""
    for deposition in depositions:
        if (
            deposition.get("title") == title
            and not deposition.get("submitted", False)
        ):
            return deposition
    return None


def create_or_update_draft(
    session: requests.Session,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Create a draft or refresh metadata of an existing draft."""
    title = str(metadata["title"])
    existing = find_draft(list_depositions(session), title)
    if existing is None:
        response = session.post(
            f"{ZENODO_API}/deposit/depositions",
            json={"metadata": metadata},
            timeout=TIMEOUT,
        )
        created = True
    else:
        response = session.put(
            f"{ZENODO_API}/deposit/depositions/{existing['id']}",
            json={"metadata": metadata},
            timeout=TIMEOUT,
        )
        created = False
    if not response.ok:
        raise response_error(response)
    return response.json(), created


def upload_file(
    session: requests.Session,
    deposition: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Upload one file to a draft using the Zenodo bucket endpoint."""
    bucket = deposition.get("links", {}).get("bucket")
    if not bucket:
        raise ZenodoError("Draft does not expose a file bucket")
    url = f"{bucket}/{quote(path.name)}"
    print(f"uploading {path.name} ({path.stat().st_size} bytes)", file=sys.stderr)
    response = None
    for attempt in range(1, 4):
        try:
            with path.open("rb") as stream:
                response = session.put(
                    url,
                    data=stream,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=TIMEOUT,
                )
            if response.ok:
                break
            if response.status_code < 500:
                raise response_error(response)
        except requests.RequestException as exc:
            if attempt == 3:
                raise ZenodoError(
                    f"Upload failed after three attempts: {path.name}: {exc}"
                ) from exc
        if attempt < 3:
            time.sleep(attempt * 2)
    if response is None or not response.ok:
        raise response_error(response) if response is not None else ZenodoError(
            f"Upload failed without a response: {path.name}"
        )
    print(f"uploaded {path.name}", file=sys.stderr)
    return response.json()


def summarize(
    deposition: dict[str, Any],
    files: list[dict[str, Any]],
    *,
    created: bool,
) -> dict[str, Any]:
    """Return non-secret draft identifiers and uploaded-file metadata."""
    metadata = deposition.get("metadata", {})
    reserved = metadata.get("prereserve_doi") or {}
    return {
        "created": created,
        "id": deposition.get("id"),
        "state": deposition.get("state"),
        "submitted": deposition.get("submitted"),
        "html": deposition.get("links", {}).get("html"),
        "doi": reserved.get("doi"),
        "files": [
            {
                "filename": item.get("filename") or item.get("key"),
                "bytes": item.get("filesize") or item.get("size"),
                "checksum": item.get("checksum"),
            }
            for item in files
        ],
    }


def prepare_draft(
    metadata_path: Path,
    files: list[Path],
    workers: int = 1,
) -> dict[str, Any]:
    """Create a reusable draft and upload every requested file."""
    token = os.environ.get("ZENODO_TOKEN", "")
    if not token:
        raise ZenodoError("ZENODO_TOKEN is not set")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing upload files: " + ", ".join(missing))
    session = make_session(token)
    metadata = load_metadata(metadata_path)
    deposition, created = create_or_update_draft(session, metadata)
    detail = session.get(
        f"{ZENODO_API}/deposit/depositions/{deposition['id']}",
        timeout=TIMEOUT,
    )
    if not detail.ok:
        raise response_error(detail)
    detail_json = detail.json()
    existing_files = {
        item.get("filename") or item.get("key")
        for item in detail_json.get("files", [])
    }
    pending = [path for path in files if path.name not in existing_files]
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        for path in pending:
            upload_file(session, deposition, path)
    else:
        def upload_in_session(path: Path) -> dict[str, Any]:
            return upload_file(make_session(token), deposition, path)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(upload_in_session, pending))
    final_detail = session.get(
        f"{ZENODO_API}/deposit/depositions/{deposition['id']}",
        timeout=TIMEOUT,
    )
    if not final_detail.ok:
        raise response_error(final_detail)
    final_json = final_detail.json()
    return summarize(
        final_json,
        final_json.get("files", []),
        created=created,
    )


def status() -> list[dict[str, Any]]:
    """Return non-secret status for all drafts."""
    token = os.environ.get("ZENODO_TOKEN", "")
    if not token:
        raise ZenodoError("ZENODO_TOKEN is not set")
    return [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "state": item.get("state"),
            "submitted": item.get("submitted"),
            "html": item.get("links", {}).get("html"),
        }
        for item in list_depositions(make_session(token))
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--metadata", type=Path, required=True)
    prepare.add_argument("--file", type=Path, action="append", required=True)
    prepare.add_argument("--workers", type=int, default=1)
    subparsers.add_parser("status")
    return parser.parse_args()


def main() -> int:
    """Run the draft preparation command."""
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_draft(args.metadata, args.file, args.workers)
        else:
            result = status()
    except (OSError, ValueError, ZenodoError, requests.RequestException) as exc:
        print(f"zenodo_deposit: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
