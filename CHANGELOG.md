# Changelog

All notable changes are documented here. The project follows semantic
versioning after the first stable release.

## [1.0.0-rc.1] - 2026-11-07

### Added

- Independent `src/` package and `cpo-stcm` command-line entry point.
- Lazy loading of PyTorch-dependent symbols.
- Configurable project-relative data and experiment paths.
- CPO-LED Bench archive download, size/SHA-256 verification, and safe ZIP
  extraction.
- Conversion of published dynamic PNG windows into a trainable NumPy array.
- Model-free `cpo-stcm demo` and environment `cpo-stcm doctor` commands.
- AGPL-3.0-only licensing, third-party notices, citation metadata, and Zenodo
  metadata.
- Initial unit tests for paths, archives, CLI behavior, and existing algorithm
  primitives.

### Known limitations

- A clean Python 3.12 installation and complete PyTorch test suite remain to be
  verified in CI.
- Full static-detector recomputation requires original raw LED frames; frozen
  public static evidence is supplied through CPO-LED Bench.
- The dependency lock and container digests are release gates that remain open.
