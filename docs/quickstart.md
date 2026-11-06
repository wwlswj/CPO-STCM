# Quickstart

## Model-free demo

```bash
python -m pip install -e .
cpo-stcm demo
```

The command constructs a deterministic normal reference sample, computes
split-conformal p-values, and reports alarms at the selected alpha.

## Public benchmark

```bash
cpo-stcm data fetch
cpo-stcm data verify
cpo-stcm data build-packed
```

The default locations are:

- archive: `data/raw/CPO-LED-Bench-v2.0-full.zip`
- extracted data: `data/raw/CPO-LED-Bench-v2.0/`
- packed array: `data/processed/dynamic_windows_v1.npy`

## Train and evaluate

```bash
python -m pip install -e ".[ml]"
cpo-stcm workflow train-temporal --category Charge --variant baseline \
  --run-name charge_baseline_v1
```

The public archive already contains the finalized
`manifests/cpo_stcm_samples.csv`; rebuilding it is only needed when original
base manifests are available.

Use `--help` after a workflow name to see its original research arguments:

```bash
cpo-stcm workflow train-temporal --help
```
