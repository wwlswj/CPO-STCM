# Architecture

## Package layers

| Layer | Modules | Responsibility |
|---|---|---|
| Core evidence | `conformal.py`, `evaluation.py`, `powicm_core.py`, `static_evidence.py` | Calibration, semantic slots, metrics, and fusion evidence |
| Temporal models | `temporal.py`, `external_baselines.py` | Optional PyTorch models and sequence scoring |
| Data access | `data.py` | Pinned download, verification, safe extraction, image packing |
| Workflows | `workflows/` | Training, inference, stress tests, and evaluation entry points |
| Interface | `cli.py`, `paths.py` | CLI dispatch, configuration, diagnostics |

## Runtime boundary

`cpo_stcm` has a dependency-light import path. Importing the package does not
load PyTorch or Ultralytics. Heavy models are imported only when a corresponding
symbol or workflow is requested.

## Data boundary

The software repository tracks source code, tests, small configuration files,
and documentation. It does not track raw datasets, packed arrays, model
weights, generated manifests, or experiment outputs.

## Reproducibility boundary

A clean installation and passing tests do not by themselves prove experimental
reproducibility. The release gate additionally requires:

1. a locked dependency environment;
2. immutable data and model hashes;
3. a one-command smoke workflow;
4. archived expected outputs;
5. container or environment identity;
6. versioned DOI records.
