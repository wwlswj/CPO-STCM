# Source provenance

The release candidate was created on 2026-11-07 from the existing research
workspace:

`<research-workspace>/03_Research/POWICM/CPO-STCM`

The source workspace was treated as read-only. The following artifacts were
copied, then locally adapted:

- package modules from `cpo_stcm/`;
- unit tests from `tests/`;
- two calibration and annotation protocol documents from `docs/`;
- the main configuration from `configs/cpo_stcm.yaml`;
- the small calibration file from
  `POWICM/experiments/powicm/oracle_v1/calibration.json`;
- selected workflow scripts from `scripts/`.

Workflow scripts were moved under `src/cpo_stcm/workflows/`, changed to use the
current working directory as their project root, and changed to use package
imports instead of repository-local `sys.path` injection.

The original `CPO-STCM` directory was not modified. Dataset archives, model
weights, experiment outputs, the Pattern Recognition manuscript, and the
literature corpus were excluded from this repository.

A complete file-hash manifest and the final clean-environment run record will
be added to the immutable release capsule before DOI minting.
