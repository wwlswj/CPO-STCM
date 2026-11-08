# CPO-STCM v1.0.0-rc.1

This is the first public-release candidate of the CPO-STCM software
distribution.

## Highlights

- Installable `cpo-stcm` Python package and command-line interface.
- Dependency-light conformal calibration and evaluation utilities.
- Optional PyTorch temporal model and workflow entry points.
- Pinned access to the CPO-LED Bench v2.0.0 data payload.
- Safe image-to-NumPy reconstruction for all 5,010 public windows.
- AGPL-3.0-only software licensing and separate CC-BY-4.0 data licensing.
- Cross-platform `uv.lock` with CPU PyTorch as the reproducible baseline.
- Software Impacts working manuscript with C1-C9 metadata and rendered
  DOCX/PDF.

## Verification

- Independent Python 3.12.10 base suite: 24 tests, 3 optional tests skipped.
- Independent Python 3.12.10 CPU ML suite: 24 tests passed.
- Frozen `uv.lock` environment: 24 tests passed.
- Ruff, compileall, dependency checks, sdist, wheel, and wheel-install smoke
  tests passed.
- GitHub Actions passed Linux Python 3.10/3.11/3.12, Windows Python 3.12, and
  Docker build/run smoke jobs.
- CPO-LED archive SHA-256 verified.
- Full packed-array SHA-256 recorded in `RELEASE_ARTIFACTS.md`.

## Known limitations

- A local Docker engine is unavailable; the ephemeral CI container passed but
  an immutable signed image digest has not yet been recorded.
- At the time of RC tagging, Zenodo DOIs and the Pattern Recognition final
  citation remained pending.
- The Software Impacts draft is not submission-ready until the Pattern
  Recognition acceptance details are available.

## Post-publication status

- Software version DOI: `10.5281/zenodo.24031262`
- Software concept DOI: `10.5281/zenodo.24031261`
- Dataset version DOI: `10.5281/zenodo.24031281`
- Dataset concept DOI: `10.5281/zenodo.24031280`

The DOI fields in the current repository documentation and working manuscript
have been updated. The tag and Release assets remain unchanged.
