# Third-party source notices

This release preserves original source copyright headers. Files have been trimmed or adapted for the CMPANet SHPCD training and evaluation paths. Shared norm1, strict legacy checkpoint loading and validation are documented in `docs/VERIFICATION.md`.

- The inherited OpenSeg framework code identifies Microsoft Research / openseg-group contributors in its headers. The source project's MIT license is retained as `LICENSE`.
- `lib/models/backbones/head.py` and `lib/models/backbones/vision_transformer/vision_transformer.py` carry ByteDance / iBOT copyright headers. A copy of the Apache License 2.0 from the locally available iBOT source is included at `LICENSES/iBOT-Apache-2.0.txt`. The upstream vision transformer source also credits DINO and timm; those attribution references are retained in the extracted file.
- Installed libraries such as PyTorch, torchvision, timm, OpenCV and scikit-image retain their own licenses; they are dependencies, not bundled binaries.

The inherited root MIT license is not a replacement for third-party file notices. Model weights and datasets are not distributed in this package.
