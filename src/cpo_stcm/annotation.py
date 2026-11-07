# SPDX-License-Identifier: AGPL-3.0-only
"""Helpers for blinded expert annotation of natural animation order."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

VALID_ORDER_LABELS = frozenset(
    {
        "forward_normal",
        "reverse",
        "shuffle",
        "freeze",
        "frame_drop",
        "other_order_fault",
        "unclear",
    }
)


def _stable_digest(seed: int, value: str) -> str:
    """Return a stable hexadecimal digest for deterministic sampling."""
    return hashlib.sha256(
        f"{seed}|{value}".encode()
    ).hexdigest()


def requires_second_review(
    seed: int,
    blind_id: str,
    fraction: float = 0.20,
) -> bool:
    """Assign second review with an independent salted hash."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Second-review fraction must be in [0, 1]")
    digest = _stable_digest(seed, f"second_review|{blind_id}")
    return int(digest[:8], 16) / 2**32 < fraction


def select_annotation_candidates(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    seed: int,
    random_count: int,
    enriched_count: int,
) -> list[dict[str, Any]]:
    """Select deterministic random-audit and model-enriched review sets.

    Only original animated windows are eligible. Model scores are used solely
    to build the enriched sampling stratum and are not returned in the blinded
    annotation rows.
    """
    if random_count < 0 or enriched_count < 0:
        raise ValueError("Sampling counts must be non-negative")
    eligible = [
        dict(row)
        for row in rows
        if str(row.get("transform")) == "none"
        and str(row.get("subclass")) == "animated"
        and str(row.get("role")) in {"test", "domain_test"}
    ]
    if len({str(row["sample_id"]) for row in eligible}) != len(eligible):
        raise ValueError("Eligible sample IDs must be unique")
    requested = random_count + enriched_count
    if requested > len(eligible):
        raise ValueError(
            f"Requested {requested} review windows but only "
            f"{len(eligible)} are eligible"
        )

    random_order = sorted(
        eligible,
        key=lambda row: _stable_digest(seed, str(row["sample_id"])),
    )
    random_selected = random_order[:random_count]
    random_ids = {
        str(row["sample_id"]) for row in random_selected
    }
    remaining = [
        row
        for row in eligible
        if str(row["sample_id"]) not in random_ids
    ]
    enriched_selected = sorted(
        remaining,
        key=lambda row: (
            -float(scores.get(str(row["sample_id"]), float("-inf"))),
            _stable_digest(seed, str(row["sample_id"])),
        ),
    )[:enriched_count]

    selected: list[dict[str, Any]] = []
    for stratum, group in (
        ("random_audit", random_selected),
        ("model_enriched", enriched_selected),
    ):
        for row in group:
            sample_id = str(row["sample_id"])
            selected.append(
                {
                    **row,
                    "blind_id": _stable_digest(seed, sample_id)[:16],
                    "sampling_stratum": stratum,
                }
            )
    return sorted(selected, key=lambda row: row["blind_id"])
