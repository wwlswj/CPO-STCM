# Release status

Date: 2026-11-07

Version: `1.0.0-rc.1`

Repository state: public GitHub repository with a published prerelease:

- https://github.com/wwlswj/CPO-STCM
- https://github.com/wwlswj/CPO-STCM/releases/tag/v1.0.0-rc.1

## Passed in the current workspace

- Editable package metadata builds successfully.
- A universal wheel builds and installs into an isolated target directory.
- The wheel-installed package imports without loading PyTorch.
- `python -m cpo_stcm demo` and the installed entry point work.
- `python -m cpo_stcm doctor` resolves all configured paths relative to the
  project.
- `compileall` and `pip check` pass.
- 20 selected tests pass in the current environment.
- The complete CPO-LED archive verifies at 395,563,944 bytes and the pinned
  SHA-256.
- All 5,010 public dynamic windows pack successfully to shape
  `(7322, 10, 100, 300)` in 2,196,600,128 bytes.
- The full packed-array SHA-256 is
  `58e84979e4c10e925c0577e592293730f486108a67c2693882dfa29f3f66e02b`.
- A clean Python 3.12.10 base environment passes 24 tests with three optional
  PyTorch tests skipped.
- A second clean Python 3.12.10 CPU ML environment with PyTorch 2.15.0+cpu
  passes all 24 tests.
- A frozen environment generated from `uv.lock` passes all 24 tests.
- `uv.lock` resolves 83 packages and pins CPU PyTorch for the reproducible
  baseline.
- Ruff, compileall, pip check, sdist, and wheel build gates pass.
- Software and dataset CFF files, Zenodo JSON files, and dataset metadata JSON
  parse successfully.
- The data metadata patch lists dataset authors in the confirmed order Wanli
  Wang, Huafeng Chen, Xian Wei.
- Data pull request https://github.com/wwlswj/CPO-LED-Bench/pull/1 prepares the
  v2.0.1 citation correction while leaving the v2.0.0 tag and archive
  unchanged.
- The Software Impacts working draft contains a 90-word abstract, six keywords,
  C1-C9 metadata, impact and limitations sections, and five reference entries.
- The working manuscript renders cleanly as a two-page A4 DOCX/PDF.
- GitHub Actions passed on Linux Python 3.10, 3.11, and 3.12, Windows Python
  3.12, and an ephemeral Docker build/run smoke test.

## Open release gates

- Mint and record software and dataset DOIs.
- Merge and publish the CPO-LED Bench v2.0.1 metadata correction after DOI
  minting.
- Replace pending DOI and publication-status text in the manuscript after
  acceptance and archival release.
- Perform the final live Guide for Authors and submission-system check.
