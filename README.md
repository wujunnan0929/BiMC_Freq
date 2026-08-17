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

### Frequency V2

V2 replaces isolated residual inputs with three CLIP-compatible views:

- low: the low-pass image;
- middle: low + middle frequencies (the original image without high-frequency
  noise);
- high: the original image with its high-frequency residual enhanced.

It also attenuates semantic correction for uncertain prototypes, disables
frequency-wise novel-to-base calibration, and estimates a class-specific
frequency alpha from visual-semantic agreement, prototype compactness, and
support-set size. `FREQ_ALPHA` is the maximum rather than a fixed coefficient
when `RELIABILITY_ALPHA` is enabled.

#### CUB-200 explicit frequency descriptions

`tools/generate_cub200_frequency_descriptions.py` generates five explicit
low/middle/high candidates per class through an OpenAI-compatible
`/chat/completions` endpoint. It validates visual-only content and JSON shape,
checkpoints after every batch, and resumes from the existing output file.
Run these PowerShell commands from the repository root:

~~~POWERSHELL
# Inspect the exact request without making a network call.
python tools/generate_cub200_frequency_descriptions.py --dry-run --limit 2

# Configure any OpenAI-compatible endpoint.
$env:LLM_API_BASE = "https://your-endpoint.example/v1"
$env:LLM_API_KEY = "your-api-key"
$env:LLM_MODEL = "your-model-name"
$env:LLM_REASONING_EFFORT = "xhigh"

# Optional 10-class smoke generation, followed by a resumable full run.
python tools/generate_cub200_frequency_descriptions.py --limit 10
python tools/generate_cub200_frequency_descriptions.py

# Require all 200 classes and every description to pass schema/content checks.
python tools/generate_cub200_frequency_descriptions.py --validate-only
~~~

If an endpoint does not support `response_format={"type":"json_object"}`, add
`--disable-json-mode`. Delete an invalid partial output or pass `--overwrite`
to regenerate it deliberately.

After generation, enable explicit descriptions and support-set Top-K grounding:

~~~BASH
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_frequency_v2.yaml --opts TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu TRAINER.BiMC.FREQUENCY.USE_EXPLICIT_DESCRIPTIONS True TRAINER.BiMC.FREQUENCY.EXPLICIT_DESCRIPTION_PATH "./description/cub200_frequency_descriptions.json" TRAINER.BiMC.FREQUENCY.DESCRIPTION_TOPK 3 TRAINER.BiMC.FREQUENCY.DESCRIPTION_TEMPERATURE 0.07
~~~

The original flat CUB descriptions remain active in the original BiMC path.
Only the frequency branch switches from keyword routing to the explicit file.
Set `DESCRIPTION_WEIGHT` to `0.0`, `0.5`, or `1.0` for structured-only,
mixed, or explicit-description-only semantic ablations.

#### DeepSeek V4-Pro generator

The separate DeepSeek entry point uses the official OpenAI-format endpoint,
V4-Pro thinking mode, and `reasoning_effort=max` by default. It reads only
`DEEPSEEK_*` variables, so an OpenAI key cannot be sent to DeepSeek by mistake.

Create an ignored local file named `tools/deepseek_config.local.ps1`:

~~~POWERSHELL
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"
$env:DEEPSEEK_API_BASE = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "deepseek-v4-pro"
$env:DEEPSEEK_REASONING_EFFORT = "max"
# Optional robustness overrides; these are already the script defaults.
$env:DEEPSEEK_BATCH_SIZE = "5"
$env:DEEPSEEK_MAX_OUTPUT_TOKENS = "32768"
$env:DEEPSEEK_MAX_RETRIES = "8"
$env:DEEPSEEK_TIMEOUT = "300"
~~~

Load it and generate a smoke batch, then resume the complete dataset:

~~~POWERSHELL
. .\tools\deepseek_config.local.ps1
python tools/generate_cub200_frequency_descriptions_deepseek.py --limit 10
python tools/generate_cub200_frequency_descriptions_deepseek.py
python tools/generate_cub200_frequency_descriptions_deepseek.py --validate-only
~~~

DeepSeek output is kept separate at
`description/cub200_frequency_descriptions_deepseek.json`. To use it in BiMC,
set `EXPLICIT_DESCRIPTION_PATH` to that file. Use `--model
deepseek-v4-flash` for the lower-cost V4 variant, or `--reasoning-effort none`
to disable thinking mode.

~~~BASH
# Main V2 experiment. Use FFT_DEVICE cpu on a server with broken cuFFT.
python main.py \
  --data_cfg ./configs/datasets/cub200.yaml \
  --train_cfg ./configs/trainers/bimc_frequency_v2.yaml \
  --opts TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu

# Three-original-view control; no FFT is executed in this mode.
python main.py \
  --data_cfg ./configs/datasets/cub200.yaml \
  --train_cfg ./configs/trainers/bimc_frequency_v2.yaml \
  --opts TRAINER.BiMC.FREQUENCY.VIEW_MODE original
~~~

Single-view ablations use exact zero priors and a fixed alpha so only the view
changes. Replace `BAND_PRIOR` with `[0.0,1.0,0.0]` or `[0.0,0.0,1.0]` for the
middle-only or high-only run.

~~~BASH
# Low-only example
python main.py \
  --data_cfg ./configs/datasets/cub200.yaml \
  --train_cfg ./configs/trainers/bimc_frequency_v2.yaml \
  --opts \
    TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu \
    TRAINER.BiMC.FREQUENCY.ADAPTIVE_FUSION False \
    TRAINER.BiMC.FREQUENCY.BAND_PRIOR '[1.0,0.0,0.0]' \
    TRAINER.BiMC.FREQUENCY.RELIABILITY_ALPHA False \
    TRAINER.BiMC.FREQUENCY.FREQ_ALPHA 0.10
~~~

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [LibContinual](https://github.com/RL-VIG/LibContinual)
- [LibFewShot](https://github.com/RL-VIG/LibFewShot)


