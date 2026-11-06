# Contributing

We welcome reproducibility reports, documentation corrections, bug reports, and
testable code contributions.

1. Open an issue describing the behavior and the smallest reproducible example.
2. Work from a branch and avoid committing data, model weights, or experiment
   outputs.
3. Keep public functions documented and preserve input validation.
4. Run `python -m compileall -q src tests` and
   `python -m unittest discover -s tests -v`.
5. State the exact environment and dataset version used for workflow changes.

By contributing, you agree that your contribution is licensed under
AGPL-3.0-only and that you have the right to submit it.
