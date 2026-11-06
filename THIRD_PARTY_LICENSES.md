# Third-party licenses

CPO-STCM depends on third-party packages that retain their own copyright and
license terms. This file records the direct dependency boundary; it is not a
substitute for the license file included by each distribution.

| Component | Role | Declared license |
|---|---|---|
| NumPy | Numerical arrays | BSD-3-Clause and bundled component licenses |
| SciPy | Assignment and scientific utilities | BSD-style |
| scikit-learn | Evaluation metrics | BSD-3-Clause |
| PyYAML | Configuration parsing | MIT |
| Pillow | PNG decoding for data packing | MIT-CMU |
| PyTorch | Optional temporal models | BSD-style plus bundled component licenses |
| OpenCV Python | Optional image and detector workflow | Apache-2.0 |
| pandas | Optional evaluation tables | BSD-3-Clause |
| matplotlib | Optional diagnostics and figures | PSF-based Matplotlib license |
| Ultralytics | Optional YOLO integration and static model | AGPL-3.0 or Enterprise License |

## Ultralytics notice

The static-evidence branch directly integrates Ultralytics and may use
Ultralytics model weights. Ultralytics states that its open-source software and
models are available under AGPL-3.0, with proprietary integration requiring an
Enterprise License. Users are responsible for assessing whether their use
triggers AGPL source-disclosure or network-use obligations.

The CPO-LED Bench data repository uses CC-BY-4.0 for owner-provided images,
annotations, split definitions, manifests, and derived evidence. Third-party
models and software remain subject to their upstream terms.

For exact dependency versions in a release, use the files under `constraints/`
and the archived release manifest. Upstream canonical license texts should be
included or referenced again when the final release capsule is assembled.
