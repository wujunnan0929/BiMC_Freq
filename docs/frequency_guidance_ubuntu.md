# Frequency Guidance：Ubuntu 执行步骤

实现入口：`tools/run_frequency_guidance_experiments.py`。
默认只预检并打印命令；加 `--execute` 才运行。方法无需梯度训练，参数只从基类训练伪序列选择。
完整定义见 `docs/frequency_guidance_experiment_protocol.md`；本轮不是旧 `safd_dual` 后修正。

## 1. 项目、环境与数据

把本轮源码、配置、描述 JSON 一起同步到服务器。在之前能运行 BiMC 的 Python 环境中执行：

```bash
cd /path/to/BiMC_Freq
export DATA_ROOT=/path/to/datasets
export GPU=0

python -c "import torch, torchvision, yacs, ftfy, regex, PIL, numpy, tqdm; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -q
```

若仅缺少小依赖，可用 `python -m pip install yacs ftfy regex pyyaml tqdm pillow`；
沿用服务器原有匹配的 torch/torchvision。数据根目录沿用旧实验：

```text
datasets/
  CUB_200_2011/              # images/ 和官方三个索引 txt
  cifar-100-python/          # torchvision CIFAR-100 文件
  miniimagenet/images/
  miniimagenet/split/        # train.csv、test.csv
```

CLIP 首次使用会加载/下载 `~/.cache/clip/ViT-B-16.pt`；离线服务器提前放入对应权重。
所有下面的 `--opts` 必须放在命令最后。

## 2. 准备频段描述

先校验已有文件：

```bash
python tools/generate_frequency_descriptions_deepseek.py --dataset cub200 --validate-only
python tools/generate_frequency_descriptions_deepseek.py --dataset cifar100 --validate-only
python tools/generate_frequency_descriptions_deepseek.py --dataset miniimagenet --validate-only
```

只有文件缺失或不完整时才生成；生成会调用 DeepSeek，需在环境中设置 `DEEPSEEK_API_KEY`。
生成器支持从已有输出继续，正常完成后再执行上面的校验命令。

```bash
python tools/generate_frequency_descriptions_deepseek.py --dataset cifar100
python tools/generate_frequency_descriptions_deepseek.py --dataset miniimagenet
# 仅 CUB 描述也缺失时使用：
# python tools/generate_frequency_descriptions_deepseek.py --dataset cub200
```

主实验使用每类每频段前 5 条独立描述；实际启动器还会预检全类别覆盖及组内重复。
G0/G1/G2 不要求频段描述，全部方法都需要各数据集原有的普通 BiMC 描述库。

## 3. 先检查计划、基类校准与尺度响应

```bash
python tools/run_frequency_guidance_experiments.py \
  --dataset cub200 --seeds 1 --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};"
```

只完成 CUB 的基类编码及 20 个伪序列校准，不调用测试集评估：

```bash
python -u main.py \
  --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc_frequency_guidance.yaml \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};" \
  SEED 1 DATASET.SUPPORT_SEED 1 \
  OUTPUT_DIR outputs/frequency_guidance_base_validation/cub200_seed1 \
  TRAINER.BiMC.GUIDANCE.BASE_ONLY True
```

检查输出的 `guidance_calibration.json`：候选网格应为 `[0,.25,.5,.75,1]`，
`data_source` 为 `base_training_only`，A/V 类互斥，审计图像在全部序列中预留，
`selection_status` 为 `base_class_disjoint_validation`。选中 0 是合法回退，不能强行改大。
`metrics.json` 状态应为 `base_validation_completed`，此时没有完整测试结果。

对从不同基类抽取的 8 张训练图像测量模糊、纹理扰动响应及视图失真：

```bash
python tools/audit_frequency_guidance_views.py \
  --dataset cub200 --data-root "$DATA_ROOT" --gpu "$GPU" --samples 8 --backend safd
python tools/audit_frequency_guidance_views.py \
  --dataset cub200 --data-root "$DATA_ROOT" --gpu "$GPU" --samples 8 --backend fft
```

结果位于 `outputs/frequency_guidance_view_audit/`。这是机制诊断，不用于选择 rho；
若响应没有支持低/中/高物理频率解释，应在论文中表述为自适应尺度。

## 4. 三数据集主实验

G0=无视觉引导；G1=原完整 BiMC；G2=视觉尺度；G3=先汇总尺度的图文关系；
G4=尺度内配对图文关系。下面三条命令各运行 15 次，总计 45 次：

```bash
python tools/run_frequency_guidance_experiments.py \
  --dataset cub200 --seeds 1 2 3 --execute \
  --support-root outputs/frequency_guidance_support \
  --output-root outputs/frequency_guidance_cub200 \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};"

python tools/run_frequency_guidance_experiments.py \
  --dataset cifar100 --seeds 1 2 3 --execute \
  --support-root outputs/frequency_guidance_support \
  --output-root outputs/frequency_guidance_cifar100 \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};"

python tools/run_frequency_guidance_experiments.py \
  --dataset miniimagenet --seeds 1 2 3 --execute \
  --support-root outputs/frequency_guidance_support \
  --output-root outputs/frequency_guidance_miniimagenet \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};"
```

论文主结果预先规定五个种子。可以一开始就运行下面的循环；若前三种子已完成，
保持源码、描述文件、路径和其他参数一致，会跳过它们，只补 4/5 并汇总所有五种子：

```bash
set -e
for dataset in cub200 cifar100 miniimagenet; do
  python tools/run_frequency_guidance_experiments.py \
    --dataset "$dataset" --seeds 1 2 3 4 5 --execute \
    --support-root outputs/frequency_guidance_support \
    --output-root "outputs/frequency_guidance_${dataset}" \
    --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};"
done
```

重复同一条命令可跳过已完成的运行；失败或中断的运行会从该运行开头重跑，
不是从会话中间恢复。默认遇错停止，`run.log` 及 summary 中的 `error_tail` 保留原因；
`--keep-going` 可继续后续运行。修改源码/配置/文本后应使用新输出目录，避免混用版本。
默认参数不依赖测试结果选择，不能看到测试成绩后更换 rho 网格或强制非零值。

## 5. 必要消融及 mini-ImageNet 文本预算

CUB、CIFAR 的普通描述均能提供 15 条独立文本。保留同目录的五个主方法，
追加五项控制；已完成的主方法会跳过，summary 保留全部十组：

```bash
for dataset in cub200 cifar100; do
  python tools/run_frequency_guidance_experiments.py \
    --dataset "$dataset" --suite all --seeds 1 2 3 4 5 --execute \
    --variants G0 G1 G2 G3 G4 G4_uniform G4_permuted G4_ordinary G4_repeat G4_augmentation \
    --support-root outputs/frequency_guidance_support \
    --output-root "outputs/frequency_guidance_${dataset}" \
    --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};"
done
```

mini 的现有普通描述最少只有 10 条，所以不能运行默认 15 条的普通文本对照。
先做另外四项控制：

```bash
python tools/run_frequency_guidance_experiments.py \
  --dataset miniimagenet --suite all --seeds 1 2 3 4 5 --execute \
  --variants G0 G1 G2 G3 G4 G4_uniform G4_permuted G4_repeat G4_augmentation \
  --support-root outputs/frequency_guidance_support \
  --output-root outputs/frequency_guidance_miniimagenet \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};"
```

再单独运行匹配预算的频段文本/普通文本比较：两组均为每尺度 3 条，总计 9 条。
这不会替代主表中 5 条/尺度的 G4；因此使用独立输出目录。这里按固定文本顺序
取描述，禁止重复填充或依据图像相似度挑选文本。

```bash
python tools/run_frequency_guidance_experiments.py \
  --dataset miniimagenet --suite all --seeds 1 2 3 4 5 --execute \
  --variants G4 G4_ordinary \
  --support-root outputs/frequency_guidance_support \
  --output-root outputs/frequency_guidance_miniimagenet_text9 \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};" \
  TRAINER.BiMC.GUIDANCE.CANDIDATES_PER_BAND 3
```

扩展消融：`G4_low G4_middle G4_high G4_fft`。例如在 CUB 的完整关键控制之后
追加它们时，使用 `--suite all` 且不传 `--variants`，会检查/运行全部 14 组。
mini 不应直接启动默认的全部 14 组，应分开运行上述匹配文本对照。
单尺度控制仅改变关系融合权重，当前仍提取三个视觉尺度，不能声称编码成本降低。
FFT/SAFD 的失真差异必须结合第 3 节的视图审计解释。

预算：主实验 75 次；关键控制及 mini 的额外匹配参考共 80 次；额外四项消融
三数据集五种子共 60 次。全部展开为 215 次，含额外 mini G4(text9) 五次。

## 6. 读取结果与汇总

每个数据集输出根目录中的 `summary.json` 包含各方法均值、样本标准差、
G4−G0/G1/G2/G3 配对差值、支持清单与校准协议一致性、编码次数和成本。
同一方法/种子的目录包含：

- `metrics.json`：逐会话 Final/Avg/Base/Novel/H、两方向混淆率、资源消耗；
- `guidance_calibration.json`：G2--G4 的候选分数、选择结果和独立审计诊断；
- `guidance_session_*.json`：原始/引导原型、位移角度、基类邻居、关系熵和尺度重叠率；
- `predictions_session_*.npz`：样本 ID、标签和预测，用于跨种子与混淆分析；
- `support.json`、`config.yaml`、`run_spec.json`、`run.log`、`checkpoint.pt`。

只汇总、不执行模型时，将原命令的 `--execute` 改成 `--summarize-only`，其余参数
（包括 `--suite`、`--variants`、`--opts`、输出目录）保持一致。
G0/G1 没有新的 guidance 校准文件。检查 `completed_runs` 是否等于 `planned_runs`，
不要将中断后只完成一部分的方法表当作完整实验。
