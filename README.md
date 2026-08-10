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

## Frequency-aware prototype calibration

The frequency experiment is a training-free extension of BiMC. It decomposes
each normalized image into low-, middle-, and high-frequency inputs, builds a
visual prototype for every class and band, and aligns them with three semantic
prompts describing global shape, part structure, and local texture. It also
routes the existing `GPT_PATH` class descriptions into the three bands with
configurable semantic keywords, falling back to all class descriptions when a
band has no keyword match. Semantic calibration is gated by visual-text
agreement and prototype uncertainty. The three bands are then fused with
class-adaptive weights.

~~~BASH
# Replace the dataset config with cub200.yaml or miniimagenet.yaml as needed.
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_frequency.yaml
~~~

Important ablations can be configured in
`configs/trainers/bimc_frequency.yaml`:

- `FREQ_ALPHA: 0.0` disables the frequency prediction branch while retaining
  the original BiMC path.
- `SEMANTIC_WEIGHT: 0.0` keeps frequency visual prototypes but removes their
  semantic calibration.
- `DESCRIPTION_WEIGHT: 0.0` uses only the structured frequency prompts;
  `1.0` uses only frequency-routed GPT descriptions.
- `ADAPTIVE_FUSION: False` uses the fixed `BAND_PRIOR` instead of class-adaptive
  band weights.
- `LOW_CUTOFF` and `HIGH_CUTOFF` control the radial FFT bands.
- `FFT_BATCH_SIZE` only controls internal FFT chunking (not the dataloader
  batch size). Reduce it to `1` if an older CUDA stack reports a cuFFT error.
- `FFT_DEVICE` accepts `auto`, `cuda`, or `cpu`. `auto` falls back to exact CPU
  FFT when cuFFT fails; set it to `cpu` to avoid repeated CUDA attempts on a
  known-incompatible server.

The direct implementation evaluates four frozen CLIP image encodings per
sample (original plus three bands), trading runtime for a clean experimental
isolation of the frequency contribution.

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [LibContinual](https://github.com/RL-VIG/LibContinual)
- [LibFewShot](https://github.com/RL-VIG/LibFewShot)


