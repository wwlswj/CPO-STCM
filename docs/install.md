# Installation

## Supported baseline

The primary release and clean-room test target is Python 3.12 on Linux and
Windows. The package metadata allows Python 3.10 and newer, but a version is
not considered release-verified until it passes CI.

## Base installation

The base installation supports configuration, conformal calibration, dataset
verification, data packing, and the offline demo.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
cpo-stcm demo
```

## Neural workflow

The `ml` extra adds PyTorch, OpenCV, pandas, and matplotlib:

```bash
python -m pip install -e ".[ml]"
```

## Static detector workflow

The `static` extra adds Ultralytics:

```bash
python -m pip install -e ".[static]"
```

This extra carries AGPL-3.0 obligations. Review
[licensing.md](licensing.md) before deployment.

## Developer tools

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## Environment diagnosis

```bash
cpo-stcm doctor
cpo-stcm config show
```

`doctor` reports installed optional packages and configured paths but does not
download data or silently import heavy ML dependencies.
