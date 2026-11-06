# SPDX-License-Identifier: AGPL-3.0-only
"""Capture-profile coverage accounting for calibration planning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def summarize_capture_profiles(
    rows: Sequence[Mapping[str, Any]],
    batch_to_profile: Mapping[str, str],
    default_profile: str,
    minimum_sequences: int,
) -> dict[str, Any]:
    """Summarize unique sequences by profile and evaluation role."""
    windows: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sequences: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    profiles_seen: set[str] = set()
    evaluation_profiles: set[str] = set()
    for row in rows:
        batch_id = str(row["batch_id"])
        profile = str(batch_to_profile.get(batch_id, default_profile))
        role = str(row["role"])
        sequence_id = str(row["sequence_id"])
        profiles_seen.add(profile)
        if role in {"test", "domain_test"}:
            evaluation_profiles.add(profile)
        windows[profile][role] += 1
        sequences[profile][role].add(sequence_id)

    profile_summary: dict[str, Any] = {}
    for profile in sorted(profiles_seen):
        role_summary = {}
        for role in sorted(windows[profile]):
            role_summary[role] = {
                "windows": windows[profile][role],
                "sequences": len(sequences[profile][role]),
            }
        calibration_sequences = len(
            sequences[profile].get("normal_calibration", set())
        )
        profile_summary[profile] = {
            "roles": role_summary,
            "calibration_sequences": calibration_sequences,
            "minimum_sequences": minimum_sequences,
            "calibration_ready": calibration_sequences >= minimum_sequences,
        }
    return {
        "minimum_sequences": minimum_sequences,
        "profiles": profile_summary,
        "unseen_test_profiles": sorted(
            profile
            for profile in evaluation_profiles
            if profile_summary[profile]["calibration_sequences"]
            < minimum_sequences
        ),
    }
