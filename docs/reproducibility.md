# Reproducibility

## Current evidence

- The CPO-LED v2.0 image archive is size- and SHA-256-pinned.
- The software and dataset are archived under registered version DOIs.
- Dataset extraction rejects absolute and parent-traversing ZIP members.
- The packed array is generated deterministically from indexed PNG windows.
- Model-free tests and the offline demo do not require network access.
- The historical research workspace records 17 passing unit tests under its
  original environment.

## Verified release gates

- The base suite passes in an independent Python 3.12.10 environment.
- The full CPU ML suite passes with PyTorch 2.15.0+cpu.
- A fresh environment resolved from `uv.lock` reproduces all 24 tests.
- All 5,010 public windows are reconstructed from the image release.
- Ruff, compile checks, dependency checks, and package builds pass.
- GitHub Actions passes Linux Python 3.10-3.12, Windows, and Docker smoke jobs.

## Release gates still open

- Build and record an immutable container digest.
- Compare a frozen quickstart output against an archived expected result.

## Pinned archival identifiers

- Software `1.0.0-rc.1` version DOI:
  [10.5281/zenodo.24031262](https://doi.org/10.5281/zenodo.24031262)
- Software concept DOI:
  [10.5281/zenodo.24031261](https://doi.org/10.5281/zenodo.24031261)
- CPO-LED Bench `2.0.1` version DOI:
  [10.5281/zenodo.24031281](https://doi.org/10.5281/zenodo.24031281)
- CPO-LED Bench concept DOI:
  [10.5281/zenodo.24031280](https://doi.org/10.5281/zenodo.24031280)

## Reproducing a released result

The released workflow is:

```bash
cpo-stcm data fetch
cpo-stcm data verify
cpo-stcm data build-packed
cpo-stcm workflow build-manifests
cpo-stcm workflow train-temporal ...
cpo-stcm workflow infer-temporal ...
cpo-stcm workflow evaluate-joint ...
```

Exact model checkpoints and expected reports will be attached to the software
release capsule rather than committed to the Git history.
