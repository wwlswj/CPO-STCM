# Submission checklist

## Open before final submission

- Confirm the Pattern Recognition manuscript is accepted or published and
  replace its working citation with the final journal DOI.
- Mint the software v1.0.0 version DOI and insert it into C2 and the software
  reference.
- Mint the dataset v2.0.1 DOI and insert it into the dataset reference.
- Replace the reproducibility-capsule pending statement with the final
  permanent link and container digest.
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
