# Released checkpoint verification

Date: 2026-10-19. Model: `cmpanet_small_head_24_max_performance.pth`.

## Public-entry-point result

The public `test.py` evaluated all **397** real 512×512 samples using the released Small config and the matching epoch8 teacher. The following confusion matrix matches both prior independent evaluations **exactly**:

```text
                         Predicted unchanged   Predicted changed
Ground-truth unchanged          96544638                338945
Ground-truth changed              536252               1953971
```

| Metric | Percent |
|---|---:|
| Precision | 85.2177314825 |
| Recall | 78.4657036739 |
| F1 | 81.7024552287 |
| Change-class IoU | 69.0652163463 |
| OA | 99.1192880345 |
| Two-class mIoU | 84.0834198739 |

Metrics are calculated from global counts, not averaged per-image. Valid pixels: 99,373,806; ignored: 4,697,362; total: 104,071,168; valid change pixels: 2,490,223. Ignore rules match the original loader (all-zero map, all-zero/all-one image). Source label 255 is change before invalid-pixel masking.

Machine-readable values and SHA-256 identities are in [release_metrics.json](release_metrics.json). The manifest hash identifies the local validation list including its path strings, so relocating the same dataset changes that list hash. All 397 per-image confusion matrices were also compared to the independently audited evaluation.

## Fixed inference protocol

- Map ViT-S/16 (384-dimensional features); frozen image ViT-B/16 (768); 24-head CCMAT/TBAM.
- Shared learned `norm1` on each of the four chunks; no whole-feature LayerNorm before splitting.
- Batch 1; FP32; TF32 off; eval/inference mode; no test-time augmentation.
- Original BGR input scaling and image mean/std, unchanged validation resizing.
- Bilinear output interpolation with `align_corners=True`; argmax of two-class logits.
- Strict loading of all **350 decoder state tensors**, and strict image teacher backbone loading.

The historical file contains unused `fuse_convs` modules, restored for strict loading; the surviving forward does not call them. It lacks the newer `chg_head.norm` parameters, so the adapter removes that unused whole-feature norm. No mismatched weights are silently skipped. Newly trained release checkpoints may have a different number of state tensors and still use the same shared-norm1 forward.

The historical source contains a method applying the same norm1 to four branches; the release weight's norm1 is trained while norm2/3/4 remain at initialization. This evidence fixed the normalization choice before the final independent verification. The full 2023 source was not recovered, so this is a documented compatibility reconstruction, not a claim of byte-identical historical source recovery.

## Training implementation and limits

The release Small config uses shared norm1 in both training and inference. Its current loss is OHEM cross entropy plus KL(student map || frozen image teacher), with pixel alignment disabled. Teacher probabilities are detached; KL is averaged over mask-selected patches per image. The mask follows the original label-based patch mask; this release does not revise its semantics. Empty eligible sets produce zero patch loss.

Historical checkpoint metadata: epoch 47, iteration 61,500, saved performance 0.8403511376. That stored value is **not F1**. The saved historical config records `hybrid_loss`, a 10-epoch LR warmup and a warm-start decoder; these differ from the current supplied training recipe. The historical patch-loss formula is not conclusively known. The supplied recipe has not been trained from scratch for 100 epochs to establish equivalence.

The matched teacher is `ssl_vit_base_epoch8.pth`; the paper describes 10 pretraining epochs. The checkpoint is selected using evaluation performance, so reporting the same split is not an independent holdout experiment. A single-checkpoint score is not the paper's multi-run mean and variation.

For this Small architecture, compare primarily with the paper's CMPANet*-S row (F1 81.6 ± 0.15), rather than inferring a Base architecture from similar precision/recall. The separately running Base experiment is not represented by these results.

## Software validation

Validation used Python 3.10, PyTorch 2.3.1+cu118, torchvision 0.18.1 and RTX 4090 on Windows.

- Ten regression tests: shared-norm1 feature transformation and gradients; KL direction, masks, student/teacher gradient behavior and half-precision stability; metric definitions; data masks; scheduler and all-ignore loss behavior.
- FP32 synthetic training smoke: three actual optimizer updates, finite train/validation loss, frozen teacher, evaluation-to-training mode restoration, checkpoint save/load round trip.
- AMP synthetic smoke: eight actual optimizer updates, finite losses, teacher frozen, checkpoint round trip passed; see `smoke_validation.json`.
- Full 397-sample real-data evaluation with the public entry point; exact global and per-image confusion-matrix agreement with the audited reference.

Synthetic smoke tests exercise the trainer without restarting the paused long-running experiment. They are not evidence of 100-epoch training convergence. Linux wrappers are supplied but were not separately executed on Linux.
