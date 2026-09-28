# SAFD implementation and evaluation

Branch: `codex/safd`. This is a PyTorch implementation of stochastic adaptive
Fourier decomposition adapted to RGB image views, not the complete HSI SAFDNet
architecture. No additional package is required. Existing configurations retain
the FFT backend; `bimc_safd.yaml` explicitly selects SAFD.

## What is implemented

The algorithm follows the Szegő dictionary, stochastic maximum selection, and
Takenaka–Malmquist (TM) construction in
[Qian et al., AFD Types Sparse Representations vs. the Karhunen–Loève Expansion](https://arxiv.org/abs/2205.00844),
equations (1), (2), and the stochastic selection principle. The related
[SAFDNet paper](https://doi.org/10.1109/TIP.2024.3357250) motivates the use of
SAFD in image representation; this code does not reproduce its learned filters.

For each image and each spatial axis:

1. Treat RGB scanlines as samples of a stochastic signal. Evenly select up to
   96 scanlines, separately for each image. No class or query labels are used.
2. Form analytic signals with a directly constructed periodic Hilbert matrix.
   There is no `torch.fft` call or cuFFT dependency in the SAFD path.
3. Reserve the constant atom (`a=0`) to preserve channel means. At each subsequent
   step choose a shared pole maximizing `mean_lines |<g, e_a>|²` on a finite
   polar dictionary. Use the generalized backward shift to update `g` and the
   Blaschke product to construct the TM atom. The dictionary includes zero so
   repeated zero poles also recover ordinary Fourier harmonics.
4. Reorthogonalize the real and imaginary atom samples twice. Near-dependent
   directions are zeroed, avoiding arbitrary basis completion on tiny images.
5. Project all full-resolution pixels onto the nested horizontal/vertical spans.
   With `P_low` and `P_mid`, components are `P_low x`, `P_mid x - P_low x`,
   and `x - P_mid x`. They reconstruct the input before view clipping.

This is a finite-grid, greedy, separable SAFD adaptation. Shared atoms are
learned within an image, not over the entire dataset. It is deterministic and
independent of batch composition; the word stochastic refers to the empirical
ensemble of signals, not random view augmentation. Pole selection uses no test
labels or cross-query adaptation. Discrete orthogonalization, a fixed DC atom,
and the 2-D projection are explicit engineering adaptations.

**The three outputs are adaptive approximation scales, not fixed low/middle/high
radial frequency bands.** Energy selection does not guarantee a frequency
ordering. The existing tensor interface and legacy variable names are retained.
The default joint GDA config does not assign frequency-specific text to these
adaptive scales. Existing V2/router/consensus paths also accept the backend, but
their frequency-specific text interpretation should be validated separately.

## Accuracy-oriented starting configuration

`configs/trainers/bimc_safd.yaml` uses frozen CLIP ViT-B/16 and fp32. The original
BiMC scores remain the reference. Joint shared-covariance GDA uses original,
coarse, intermediate, and detail features. Ridge, temperature and mixing strength
are selected using only base training classes by the existing calibration code.
The mixing grid contains zero; a zero result is a fallback, not evidence of an
SAFD improvement. No benchmark result selects these settings.

Initial image-view settings (not benchmark-tuned):

| Setting | Default | Purpose |
|---|---:|---|
| `SAFD.LOW_TERMS` | 8 | Non-DC TM terms in the coarse span |
| `SAFD.MIDDLE_TERMS` | 24 | Non-DC TM terms in the larger span |
| `SAFD.RADII` | `[0, .3, .6, .8]` | Adaptive pole radii; supported range `[0, .95]` |
| `SAFD.ANGLES` | 16 | Candidates on each nonzero-radius circle |
| `SAFD.MAX_SIGNALS` | 96 | Evenly sampled scanlines per image/axis |
| `SAFD.BATCH_SIZE` | 2 | Internal memory bound, independent of dataloader size |
| `SAFD.ORIGINAL_BLEND` | `[.25, .1]` | Original-image fractions in the first two views |
| `HIGH_ENHANCE` | .25 | Detail residual enhancement |
| `SAFD.MAX_DETAIL_RMS` | .04 | Per-image/channel RMS cap in pixel units |

The natural views are `(1-b0)*low+b0*x`, `(1-b1)*mid+b1*x`, and
`x+bounded_detail`. These preserve more CLIP-compatible content than isolated
residuals. `VIEW_MODE=disjoint` gives centered component controls;
`VIEW_MODE=original` bypasses decomposition. `LOW_CUTOFF`, `HIGH_CUTOFF`,
`FFT_DEVICE`, and `FFT_BATCH_SIZE` affect FFT only. SAFD follows the input device.
Projection and selection run in fp32/complex64 even under external autocast.

## Run

Single SAFD experiment (replace the data root):

```bash
python main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc_safd.yaml \
  --opts DATASET.ROOT /absolute/path/to/datasets OUTPUT_DIR outputs/safd_single
```

Matched evaluation on the same supports and calibration episodes:

```bash
python tools/run_safd_experiments.py --seeds 1 2 3 --dry-run \
  --opts DATASET.ROOT /absolute/path/to/datasets
python tools/run_safd_experiments.py --seeds 1 2 3 --execute \
  --opts DATASET.ROOT /absolute/path/to/datasets
```

Core: `baseline`, `zero`, `original_gda`, `fft_gda`, `safd_gda` (15 runs).
`--suite all` adds no-original-blend, no-detail-enhancement, and repeated-original
controls (24 runs). `fft_gda` and `safd_gda` share the classifier, fp32, enhancement
coefficient, calibration budget, and support manifests. They compare the complete
view recipes; only SAFD uses the original blend and detail cap. `safd_no_blend`
helps isolate the blend contribution, but this is not a distortion-matched study.

Use `--variants baseline fft_gda safd_gda --seeds 1` for a smaller first run.
Results include final/average/base/novel/H metrics, paired differences against FFT
and BiMC, support hashes and calibration audits. Keep output roots separate after
changing code/config; the launcher refuses to reuse stale experiment fingerprints.

For a direct replacement in the existing router (a useful performance comparator):

```bash
python main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc_safd_router.yaml \
  --opts DATASET.ROOT /absolute/path/to/datasets OUTPUT_DIR outputs/safd_router
```

Use the same support manifest, seed and fp32 for the paired FFT router. Any change
to decomposition rank or view strength should be selected on base training
validation data, then frozen for benchmark evaluation. The default GDA config is
a testable starting point, not a claim that it beats the existing router.

Any existing FFT configuration can also use SAFD with the single override
`TRAINER.BiMC.FREQUENCY.BACKEND safd`; its downstream classifier remains the same.

## Verification and limits

```bash
python -m unittest discover -s tests -q
```

Tests cover analytic-signal correctness, shared adaptive pole selection on a
rational signal, reconstruction, nested projection energy, constant colors,
batch independence, RGB/dtype handling, CPU/CUDA/autocast agreement, no FFT
execution, config parsing, and all BiMC frequency consumers.

Full CUB-200 data are not present in the local workspace. Numerical tests or small
local-image smoke runs cannot establish benchmark accuracy. Report full paired
multi-seed results before claiming an improvement; include calibration fallbacks
and any base/novel tradeoff.

Local verification on 2026-09-28: all 190 tests passed, including CUDA tests on
GTX 1080 Ti / PyTorch 2.6.0+cu126. After adding the router YAML, its four integration
tests were rerun and passed. `git diff --check` passed.

Real frozen CLIP ViT-B/16 also completed three sessions on a 12-class local
CIFAR PNG subset (12 training images/class, 3 test images/class, 8 base classes,
two 2-class increments). FFT and SAFD used identical support and calibration
hashes. Both selected `alpha=0` and finished at 97.22%, identical to BiMC. A
separate preselected manual `alpha=.1` SAFD smoke run exercised nonzero query
fusion in all sessions and finished at 100% on these 36 test images. This manual
run is an execution check, not a hyperparameter recommendation or benchmark
claim; it does not replace the automatic selection result. Default settings
were not changed in response to these test results.

Local artifacts: `outputs/safd_real_smoke/paired_summary.json`,
`safd_verification.json`, `fft_verification.json`, and
`safd_active_verification.json` in that same directory. These outputs are ignored
by Git; the implementation, tests, configuration and instructions are tracked
source candidates on this branch.
