# Software Impacts manuscript

This directory contains the independent Software Impacts Original Software
Publication draft.

- Source of truth: `manuscript.json`
- Bibliography: `references.bib`
- DOCX builder: `build_manuscript.py`
- Submission blockers: `SUBMISSION_CHECKLIST.md`

Rendered working deliverables:

- `CPO-STCM_SoftwareImpacts_Working_Draft.docx`
- `CPO-STCM_SoftwareImpacts_Working_Draft.pdf`

The current render is two A4 pages. The abstract is 90 words, there are six
keywords and nine C1-C9 rows, and the body remains below the template's
three-page limit.

Build after installing `requirements-paper.txt`:

```bash
python build_manuscript.py
```

The draft does not modify or replace the Pattern Recognition manuscript. It
describes the software, reproducibility workflow, impact, limitations, and
enabled publication rather than repeating the research paper's theory and full
experimental evaluation.
