# Directional-Order Expert Annotation Protocol

Version: 1.0  
Date: 2026-11-05

## Objective

Estimate how often production-derived LED animations contain natural temporal
order faults, and later compare frozen CPO-STCM scores against independent
human adjudication.

The existing reverse, shuffle, freeze, and frame-drop windows are deterministic
stress transformations. They must not be presented as natural-fault evidence.

## Sampling

- Generate the queue with
  `scripts/build_direction_annotation_set.py`.
- Default pilot: 120 randomly audited windows plus 80 model-enriched windows.
- The random-audit stratum supports prevalence estimation; the enriched
  stratum only supports error analysis.
- Selection uses stable hashes and a frozen random seed.
- All 200 selected windows are evaluated with a frozen model checkpoint.
- At least 20% are assigned to two reviewers; disagreements are adjudicated
  by a third reviewer.
- Reviewers do not see model scores, labels, model-enriched status, or the
  other reviewer's response.

## Label Set

| Label | Definition |
|---|---|
| `forward_normal` | The complete animation follows the expected production order without repetition, deletion, or reordering. |
| `reverse` | Most semantic transition events occur in the reverse expected order. |
| `shuffle` | At least one non-adjacent transition group is reordered while the clip otherwise contains valid animation events. |
| `freeze` | A transition frame or short state is repeated long enough to break the expected progression. |
| `frame_drop` | One or more required transition states are absent, producing a discontinuous jump. |
| `other_order_fault` | A defensible temporal-order defect that does not fit the labels above. |
| `unclear` | Visibility, compression, or occlusion prevents a reliable decision. |

Invalid or ambiguous screens are retained with `unclear`; they are not silently
removed. Deletion or exclusion must be pre-specified before score analysis.

## Review Procedure

1. Read the manufacture-defined animation sequence for the product.
2. Open the raw frames from `relative_path`, frames `start_frame` through
   `end_frame`, for the specified `subimage_index`.
3. Judge semantic event order without using object-detector or CPO-STCM output.
4. Record label, confidence in `[0, 1]`, and a short evidence note.
5. Complete an independent second review when requested.
6. Resolve only label disagreements; do not average confidence values.

## Analysis Contract

- Primary outcome: expert-adjudicated natural order-fault prevalence with a
  source-cluster 95% interval.
- Secondary outcome: AUROC and AUPRC of frozen CPO-STCM evidence against
  adjudicated labels.
- Sensitivity analyses: random-audit stratum only; complete-case versus
  `unclear`-retained labels.
- Inter-rater agreement: Cohen's kappa for two-reviewer windows and
  prevalence-adjusted agreement for the full queue where applicable.
- No training, threshold selection, or prompt tuning may use the expert test
  labels.

## Stop Conditions

- Do not report model effectiveness if fewer than 80% of the queue is
  adjudicated.
- Do not estimate prevalence from the model-enriched stratum alone.
- Do not collapse `unclear` into normal.
- Do not call deterministic transformed windows natural anomalies.
