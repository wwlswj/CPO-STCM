# Limitations

- The method and default geometry target one-row, four-device LED displays and
  ten-frame animation windows.
- The static semantic model expects nine calibrated display slots and a
  compatible object-detection output.
- Capture-profile stress remains difficult: the research manuscript reports
  elevated false-positive rates under an unseen capture profile.
- The public image-only benchmark does not include every original full-size raw
  frame needed to recompute static YOLO detections. Frozen static evidence is
  distributed instead.
- External UCSD Ped2 data remain subject to their original terms and are not
  redistributed by this repository.
- The current release candidate is not validated for safety-critical,
  zero-failure, multi-site, or commercial deployment claims.
- Python 3.14 and CUDA-specific releases are outside the verified 3.10-3.12
  CPU baseline.
- The locked-container and remote-CI gates remain open until release.
