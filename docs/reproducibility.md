# Reproducibility

## Current evidence

- The CPO-LED v2.0 image archive is size- and SHA-256-pinned.
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

## Release gates still open

- Verify the optional PyTorch path through remote Linux and Windows CI.
- Build and record an immutable container digest.
- Compare a frozen quickstart output against an archived expected result.
- Mint a software version DOI and a dataset DOI.

## Reproducing a released result

The intended publication flow is:

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
