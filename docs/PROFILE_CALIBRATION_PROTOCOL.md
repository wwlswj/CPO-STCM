# Capture-Profile Calibration Protocol

Version: 1.0  
Date: 2026-11-05

## Problem

The frozen mixed-population test uses a nominal 5% alarm budget. The
capture-profile stress subset reaches an empirical false-positive rate of
0.346, which violates the exchangeability assumption required by global and
observability-only conformal calibration.

## Implementation

`cpo_stcm.conformal.profile_mondrian_conformal_p_values` provides a
profile-by-observability calibration hierarchy:

1. exact `capture_profile × observability_stratum` reference;
2. observability-stratum backoff;
3. global reference.

Each level is used only when its normal calibration set meets the configured
minimum size.

## Required Data

- At least 50 independent normal source sequences for every production capture
  profile used in deployment.
- Profile metadata covering camera, lens, exposure, white balance, device
  spacing, and image-resolution changes.
- Normal calibration sequences disjoint from validation and test.
- No test windows used for profile discovery, thresholding, or backoff choice.

## Evaluation

- Nominal budgets: 0.01 and 0.05.
- Primary metric: profile-conditional false-positive rate.
- Guardrails: overall recall, AUPRC, and alarm rate.
- Report profile coverage and fallback rate explicitly.
- If a profile is unseen or below the minimum calibration size, report
  `PROFILE_CALIBRATION_UNAVAILABLE`; backoff results must not be described as
  profile-specific control.

## Prohibited Claims

- No claim of guaranteed control under arbitrary capture shift.
- No use of test-profile normals as calibration data.
- No claim that the existing `domain_test` subset already solves profile
  shift.
