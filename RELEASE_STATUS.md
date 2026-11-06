# Release status

Date: 2026-11-07

Version: `1.0.0-rc.1`

Repository state: local Git repository on `main`; no remote configured and no
external publication performed.

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
- Two real dynamic windows pack successfully to shape `(2, 10, 100, 300)`.
- Software and dataset CFF files, Zenodo JSON files, and dataset metadata JSON
  parse successfully.
- The data metadata patch lists dataset authors in the confirmed order Wanli
  Wang, Huafeng Chen, Xian Wei.

## Open release gates

- Run the complete suite in a clean Python 3.12 environment.
- Restore and verify the optional PyTorch path; the current Python 3.14
  environment stalls while loading a torch DLL.
- Run the full 5,010-window packing operation rather than only the two-window
  smoke test.
- Build and verify Docker/OCI images; Docker is not installed on the current
  host.
- Generate a full transitive dependency lock.
- Run the repository CI workflow on Linux and Windows.
- Mint and record software and dataset DOIs.
- Apply and publish the CPO-LED Bench v2.0.1 metadata correction.
- Perform the final Software Impacts template and live Guide check.
