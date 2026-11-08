# Submission checklist

## Open before final submission

- Confirm the Pattern Recognition manuscript is accepted or published and
  replace its working citation with the final journal DOI.
- Confirm the published software `v1.0.0-rc.1` and dataset `v2.0.1` DOI links
  resolve. If a stable `v1.0.0` is released before submission, decide whether
  to update C1, C3, and the software reference to its new version DOI.
- Confirm the reproducibility capsule and available release artifacts resolve;
  add the immutable container digest if one is published before submission.
- Confirm the public GitHub repository, documentation links, and release assets
  resolve from a signed-out browser.
- Recheck the current Software Impacts Guide for Authors and article template
  in the live submission system.
- Confirm all authors approve the final author order, corresponding author,
  affiliations, email addresses, and license statement.
- Render the final DOCX and PDF and inspect every page.
- Confirm body text excludes metadata, tables, figures, and references when
  checking the three-page limit.

## Already verified locally

- Clean Python 3.12.10 base suite: 24 tests, 3 optional PyTorch tests skipped.
- Clean Python 3.12.10 CPU ML suite: 24 tests passed.
- Frozen uv.lock environment: 24 tests passed.
- Ruff, compile checks, build checks, and dependency checks passed.
- Full public image packing: 5,010 windows and 2,196,600,128 bytes.
- Dataset archive hash and extracted shape verified.
- Software DOI `10.5281/zenodo.24031262` and dataset DOI
  `10.5281/zenodo.24031281` are registered and inserted.
