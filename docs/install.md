# Installation

## Supported baseline

The primary release and clean-room test target is Python 3.12 on Linux and
Windows. The package metadata supports Python 3.10 through 3.12.

## Base installation

The base installation supports configuration, conformal calibration, dataset
verification, data packing, and the offline demo.

```bash
uv sync --frozen
uv run --frozen cpo-stcm demo
```

## Neural workflow

The `ml` extra adds PyTorch, OpenCV, pandas, and matplotlib:

```bash
uv sync --frozen --extra dev --extra ml
```

## Static detector workflow

The `static` extra adds Ultralytics:

```bash
uv sync --frozen --extra dev --extra ml --extra static
```

This extra carries AGPL-3.0 obligations. Review
[licensing.md](licensing.md) before deployment.

## Developer tools

```bash
uv sync --frozen --extra dev --extra ml
uv run --frozen --extra dev --extra ml python -m unittest discover -s tests -v
uv run --frozen --extra dev --extra ml ruff check src tests
```

## Environment diagnosis

```bash
uv run --frozen cpo-stcm doctor
uv run --frozen cpo-stcm config show
```

`doctor` reports installed optional packages and configured paths but does not
download data or silently import heavy ML dependencies.
