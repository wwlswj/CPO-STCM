# Workflows

All workflows run from the project root by default. Set
`CPO_STCM_PROJECT_ROOT` or pass the global `--project-root` option to use a
different working project.

## Dataset setup

```bash
cpo-stcm data fetch
cpo-stcm data verify
cpo-stcm data build-packed
```

`build-manifests` is an optional research-data workflow. It requires the
original `packed_manifest_curated.csv`, not the finalized public manifest.

## Temporal training

```bash
cpo-stcm workflow train-temporal \
  --category Charge \
  --variant enhanced \
  --run-name charge_enhanced_v2
```

## Temporal inference

```bash
cpo-stcm workflow infer-temporal \
  --checkpoint experiments/temporal_autoencoder/charge_enhanced_v2/best.pt \
  --output-dir experiments/inference/charge_enhanced_v2 \
  --category Charge \
  --device 0
```

## Evaluation

```bash
cpo-stcm workflow evaluate-joint \
  --enhanced-scores experiments/inference/charge_enhanced_v2/temporal_scores.csv \
  --baseline-scores experiments/inference/charge_baseline_v1/temporal_scores.csv \
  --static-evidence data/raw/CPO-LED-Bench-v2.0/evidence/static_evidence.csv \
  --output-dir experiments/joint_evaluation/charge_cpo_stcm_v3
```

Other exposed workflows:

- `evaluate-conformal`
- `evaluate-natural`
- `evaluate-deletion`
- `precompute-static`
- `prepare-ucsd`
- `train-external-autoencoder`
- `evaluate-external-ucsd`
- `train-external-baseline`
- `evaluate-external-baseline`
- `bootstrap-external`

The top-level workflow parser preserves each original script's options. Use
`--help` after a workflow name for the exact interface.
