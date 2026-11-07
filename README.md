# CPO-STCM

CPO-STCM is research software for normal-only anomaly detection in LED-screen
animations. It combines temporal reconstruction, partially observed
semantic-slot consistency, and block-aware conformal calibration.

This repository is the independent software distribution intended for the
planned *Software Impacts* Original Software Publication. It is currently a
`1.0.0rc1` release candidate and has not yet been published to GitHub, Zenodo,
or the journal.

## What the software provides

- A Python package with dependency-light conformal calibration utilities.
- An optional CNN-LSTM temporal autoencoder and temporal-order head.
- A static semantic-evidence interface for nine LED display slots.
- A command-line interface for diagnostics, data verification, packing, and
  training/evaluation workflows.
- Validated access to the public CPO-LED Bench v2.0 data payload.
- Reproducibility hooks for locked environments, immutable assets, and future
  Zenodo DOIs.

## Install

Python 3.12 is the primary release target. The current working environment is
also recorded for auditing, but it is not the release baseline.

```bash
uv sync --frozen --extra dev
```

Install the neural-network workflows with:

```bash
uv sync --frozen --extra dev --extra ml
```

The static YOLO evidence workflow is optional and requires:

```bash
uv sync --frozen --extra dev --extra ml --extra static
```

See [docs/install.md](docs/install.md) for the complete dependency matrix.

## First run

Run a model-free conformal calibration example:

```bash
uv run --frozen cpo-stcm demo
```

Inspect the environment and configured paths:

```bash
uv run --frozen cpo-stcm doctor
uv run --frozen cpo-stcm config show
```

## Use CPO-LED Bench

The public archive is pinned to version `2.0.0`, size `395563944` bytes, and
SHA-256:

```text
a462f4c144b4675aaeb7ccdc819066653fd9500615843996f941226c05a3b19d
```

Download, validate, and extract it:

```bash
uv run --frozen cpo-stcm data fetch
uv run --frozen cpo-stcm data verify
```

Build the memory-mappable training array expected by the temporal workflow:

```bash
uv run --frozen cpo-stcm data build-packed
```

The complete workflow is described in [docs/workflows.md](docs/workflows.md).
Raw data, processed arrays, model checkpoints, and experiment outputs are not
tracked by Git.

## Software architecture

```text
CPO-LED Bench images
        |
        +--> packed temporal windows --> CNN-LSTM autoencoder
        |                               + masked reconstruction
        |                               + temporal-order evidence
        |
        +--> static semantic evidence --> partially observed slot model
                                          + weighted isotonic consistency

                 temporal + semantic evidence
                              |
                              v
                 block-aware conformal calibration
                              |
                              v
                    alarm metrics and reports
```

The package does not import PyTorch, OpenCV, or Ultralytics at package import
time. Lightweight path, data-integrity, conformal, and CLI operations therefore
remain available in a base CPU installation.

## Tests

```bash
uv run --frozen --extra dev --extra ml python -m unittest discover -s tests -v
uv run --frozen --extra dev --extra ml ruff check src tests
uv run --frozen --extra dev --extra ml python -m compileall -q src tests
```

Tests requiring PyTorch are skipped when the optional `ml` dependencies are
absent. A clean Python 3.12 CPU environment is the release gate.

## Licensing and citation

Software is licensed under `AGPL-3.0-only`. CPO-LED Bench data remains under
`CC-BY-4.0`; the two licenses apply to different artifacts. Optional
Ultralytics integration remains subject to Ultralytics licensing terms.

Citation metadata is provided in `CITATION.cff`. Until the archival DOI is
minted, cite the versioned source repository rather than an unversioned branch.

## Support

Technical support: `wangwanli@jcut.edu.cn`

First author: Xiaoxue Su, `xiaoxuesu@jcut.edu.cn`
