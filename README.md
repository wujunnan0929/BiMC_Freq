# BiMC

This is the official implementation of paper **Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration (CVPR 2025)**.

## Abstract

Few-shot Class-Incremental Learning (FSCIL) challenges models to adapt to new classes with limited samples, presenting greater difficulties than traditional class-incremental learning. While existing approaches rely heavily on visual models and require additional training during base or incremental phases, we propose a training-free framework that leverages pre-trained visual-language models like CLIP. At the core of our approach is a novel Bi-level Modality Calibration (BiMC) strategy. Our framework initially performs intra-modal calibration, combining LLM-generated fine-grained category descriptions with visual prototypes from the base session to achieve precise classifier estimation. This is further complemented by inter-modal calibration that fuses pre-trained linguistic knowledge with task-specific visual priors to mitigate modality-specific biases. To enhance prediction robustness, we introduce additional metrics and strategies that maximize the utilization of limited data. Extensive experimental results demonstrate that our approach significantly outperforms existing methods.

## Installation

### Dataset

Please follow [CEC](https://github.com/icoz69/CEC-CVPR2021) to download *mini*-ImageNet, CUB-200 and CIFAR-100.

### Requirement

- `torch==1.13.1`
- `torchvision==0.14.1`
- `yacs==0.1.8` 
- `tqdm==4.66.1`
- `ftfy==6.1.1`
- `regex==2023.10.3`
- `scikit-learn==1.3.2`

## Experiments

First, remember to modify the data path `ROOT` in the `dataset` configuration file.

~~~BASH
# CIFAR BIMC
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc.yaml

# CIFAR BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# MiniImagenet BIMC
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc.yaml

# MiniImagenet BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# CUB200 BIMC
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml

# CUB200 BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml
~~~

## Frequency contribution diagnostic

Run the training-free frequency counterfactual diagnostic with:

~~~BASH
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_freq_analysis.yaml
~~~

The diagnostic first estimates three approximately equal-energy radial Fourier
bands from base-session images. It then samples five deterministic,
center-cropped training images per class, removes each band in turn, and measures
the resulting drop in CLIP's correct-class semantic margin. It does not change
or train the BiMC classifier. Per-session and all-class JSON/CSV reports are
written under
`outputs/frequency_analysis/<dataset>_seed<seed>_equal_energy/`. Existing
fixed-band reports are not overwritten.

In `all_classes_frequency_contribution.json`, inspect
`class_effect_eta_squared`, the dominant-band distribution, and
`mean_pairwise_frequency_weight_l1`. Larger values indicate that the measured
frequency contribution varies more strongly between classes. The default
`COMPETITOR_SCOPE: all` uses all dataset class names so results are comparable
between sessions; this is an offline analysis setting, not a deployable FSCIL
protocol. Set it to `accumulated` for a strict session-time diagnostic.

The supplied analysis config uses `FFT_DEVICE: cpu` for compatibility with
PyTorch 1.13.1 CUDA environments that can raise `CUFFT_INTERNAL_ERROR` for
batched 224x224 transforms. Only counterfactual image generation runs on CPU;
CLIP encoding still runs on CUDA. `FFT_DEVICE: auto` first tries the input
device and automatically falls back to CPU after a cuFFT failure.

Run the same analysis on all three benchmarks with:

~~~BASH
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_freq_analysis.yaml
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc_freq_analysis.yaml
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_freq_analysis.yaml
~~~

Use the same configuration and seeds for cross-dataset comparisons. CIFAR100 is
low-resolution and mainly tests coarse structural cues; mini-ImageNet adds more
diverse natural-image spectra, while CUB200 tests whether fine-grained classes
derive more benefit from mid- and high-frequency detail.

For repeated runs, override the YAML seed without duplicating configuration
files, for example:

~~~BASH
for seed in 1 2 3 4 5; do
  python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_freq_analysis.yaml --seed $seed
done
~~~

## Support-to-test frequency predictivity

Experiment B tests whether a class's five-shot support contribution predicts
its actual test-time benefit from retaining the high-frequency band. It uses
only accumulated class names, evaluates the final session, and compares the
full image, high-frequency removal, equal mixing, support-predicted gates, and
descriptive test-label gates. A gate controls each candidate class's logit
column; it does not select one image representation for the whole sample.

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_freq_predictivity.yaml --seed 1
~~~

Reports are written under
`outputs/frequency_predictivity_v3/<dataset>_seed<seed>_equal_energy/`; the
separate directory preserves legacy v2 outputs. The final
session JSON contains accuracy comparisons, Pearson/Spearman support-to-test
correlations, harmful-frequency precision/recall, exact paired McNemar tests,
paired bootstrap confidence intervals, macro/base/novel accuracy, class
improvement/degradation/zeroing counts, and test-only threshold/alpha
sensitivity sweeps. The accompanying `_samples.csv` stores paired predictions
for independent significance checks. With `SAVE_LOGITS=True`, `_logits.pt`
also stores CPU full/removed logits, labels, and support statistics so later
fusion/calibration variants can be evaluated without re-encoding the images.

The v2 hard replacement remains in the report only as a destructive diagnostic.
The primary constrained residual gate uses
`full_weight = 1 - alpha * (1 - support_gate)`; with the default `alpha=0.2`,
every candidate class retains at least 80% of its full-image logit. Alpha must
be selected without test labels (for example on a base-class validation split),
not from the reported test sensitivity sweep.

The `test_label_margin_gate` and `test_label_accuracy_gate` use test labels and
are descriptive leaked references, not oracle upper bounds on global accuracy.
The `test_optimized_coordinate_gate` also uses test labels and performs local
coordinate ascent. It is reported only as a diagnostic ceiling and is neither
deployable nor guaranteed globally optimal.

To test whether the equal-energy high band is too broad, run the fixed radial
band control (`high = [0.35, 1.0]`) with:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_freq_predictivity_fixed.yaml --seed 1
~~~

Fixed-band reports are written under
`outputs/frequency_predictivity_fixed_v3/<dataset>_seed<seed>_fixed/`.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [LibContinual](https://github.com/RL-VIG/LibContinual)
- [LibFewShot](https://github.com/RL-VIG/LibFewShot)


