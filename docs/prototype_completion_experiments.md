# 支持端联合统计原型补全：首轮实验

代码分支：`codex/frequency-prototype-completion`。本版实现无梯度统计补全及机制对照，尚无新方法的真实数据集性能结论；轻量门控不在本版范围。

## 方法和协议

每张支持图像使用原图与一个辅助视图，默认辅助视图为 FFT 低通图。普通类别文本及固定的尺度描述提供语义锚点；描述不依据当前支持集筛选。先在基类拟合每个视图的低秩子空间、图文平均偏移、语义先验残差协方差 `B` 与类内联合噪声协方差 `W`。后验均值为：

```text
posterior = semantic_prior + B @ solve(B + W/K, observed_mean - semantic_prior)
```

`W` 保留同一张图片在不同视图中的相关性；`B` 估计扣除有限基类样本均值的噪声，协方差进行收缩与正定化。只将原图部分重建回 CLIP 空间，未建模的原图分量保留支持均值。将后验方向与原始支持均值方向的差添加到原 BiMC 视觉校准结果，再按选定 blend 归一化。

基类原型不补全。新类到达时补全一次，历史类别原型保持冻结。新原型仍参与完整 BiMC 的全类别竞争，所以冻结旧原型不保证旧类准确率不下降。原 BiMC 的文本融合、协方差及评分流程保持统一；查询只编码原图，不计算辅助视图。

本版是 **无新增梯度训练、包含基类统计拟合及验证选择**。不能称为不使用数据学习。Gaussian 模型是归一化 CLIP 特征上的近似，不代表物理恢复了没有观察到的属性。

每次运行仅使用原始基类训练集：

1. 按类别分成拟合类 A 和验证类 V。A 预留固定查询，拟合统计完全不读 A 预留查询或 V 图像。
2. 使用同一伪增量协议，比较 blend `0/0.25/0.5/1`，以平均阶段整体准确率选择；相对参考的 Base、Novel 平均下降均须不超过 `0.2pp`，H 作为辅助报告。并列优先较小 blend，允许选零。
3. 选择结束后固定超参数，使用全部原始基类训练图像重新拟合统计，再执行正式 FSCIL。正式增量支持和测试图像不参与选择。

当前默认 A/V 各占基类的一部分，伪任务保持阶段结构，但类别总规模小于正式 benchmark。该设计没有覆盖完整正式类别规模，需检查规模迁移，不能把伪增量收益当正式性能。不同支持种子仍来自同一数据集，不是独立数据集。

## 对照矩阵

| 名称 | 改动 | 检验目的 |
|---|---|---|
| `baseline` | 原完整 BiMC | 原参考 |
| `zero` | 手动 blend=0，跳过自动校准 | 与参考逐分数/逐预测一致 |
| `original` | 仅原图统计补全 | 一般语义收缩的收益 |
| `frequency` | 原图＋FFT 低通 | 频率视图的附加价值 |
| `ordinary_text` | 视觉同上，辅助语义改普通文本 | 特定尺度语义是否必要 |
| `permuted_text` | 文本尺度置换 `[2,0,1]` | 正确对应是否必要 |
| `augmentation` | 同次数的普通辅助增强 | 普通多视图控制 |
| `repeat` | 复制原图视图，跳过重复编码 | 重复证据控制 |
| `block` | 联合 `B/W` 的跨视图块置零 | 跨视图相关性的价值 |

`core` 和 `all` 在首版均指上述九项。每个自动对照独立选择 blend，但使用相同的类别/样本协议、候选集合与保护条件。所有 `VIEW_CONTROL` 共用同一 `AUXILIARY_INDEX`；默认 `1` 为低通/模糊视图，`2` 为低中频/裁剪视图。普通增强是计算次数对照，不声称匹配频率视图的失真强度。

启动器将辅助文本候选预算统一：默认 `structured` 矩阵的尺度模板为 1 条，`ordinary_text` 也固定取 1 条普通描述；`explicit` 矩阵每尺度取前 5 条，`ordinary_text` 同样取前 5 条。均按固定文件顺序聚合，不使用支持集筛选。原图语义锚点仍共用完整原 BiMC 文本定义。直接调用 `main.py` 做手动普通文本控制时，也需设置相同 `CANDIDATES_PER_BAND`，避免候选数量混杂。

“形状/部件/纹理”是语义尺度，不等同于傅里叶频率。若普通文本或错配文本也有相同收益，应收缩关于频率语义的解释。若频率方案不能稳定超过 `original` 和 `augmentation`，先停止扩大频率补全模块。

`block` 禁用跨视图观测传递，但仍使用联合统计的收缩尺度，不要求其数值结果等于独立拟合的 `original`。重复视图在正则化后也不保证与单视图严格等价，因此两者都保留为实测对照。

## Ubuntu 准备与基类验证

使用已有 Python/CLIP 环境及本地数据集，不需新的 LLM 调用，也不需先生成离线特征缓存。先同步该分支代码。以下采用已有服务器路径，数据路径不同只修改 `DATA_ROOT`。

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
git switch codex/frequency-prototype-completion
export DATA_ROOT=../datasets/
export GPU=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export COMPLETION_OUT="outputs/prototype_completion_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
```

本地分支未推送时，服务器须先同步工作区文件；`git switch` 本身不会传输这些文件。保持既有数据划分、描述文件与 CLIP 权重不变。先打印矩阵，确认数据路径与输出路径：

```bash
python tools/run_prototype_completion_experiments.py \
  --dataset cub200 --seeds 1 --variants baseline zero original frequency \
  --output-root "$COMPLETION_OUT/plan" --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "$GPU"
```

首轮可先运行基类开发验证，不加载任何 benchmark 测试图像：

```bash
python -u main.py \
  --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc_prototype_completion.yaml \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "'${GPU}'" \
  SEED 1 DATASET.SUPPORT_SEED 1 OUTPUT_DIR "$COMPLETION_OUT/base_frequency" \
  TRAINER.BiMC.PROTOTYPE_COMPLETION.BASE_ONLY True \
  TRAINER.BiMC.PROTOTYPE_COMPLETION.CALIBRATION_SEED 10102
```

检查 `completion_calibration.json` 的 `candidate_results`、`selected_blend`、A/V 类别、协议哈希与有限统计量，及 `metrics.json` 的 `status=base_validation_completed`。若选择零，应记录为本次未发现可部署补全收益；不要强制非零参数再报告为选中方法。

在 cuFFT 不兼容的服务器，向命令最后的 `--opts` 添加：

```text
TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu
```

这只改变 FFT 计算设备，CLIP 仍使用 GPU。希望整个运行使用 CPU 时另加 `DEVICE.DEVICE_NAME cpu`。命令中的 `--opts` 必须最后出现。

## 三支持种子配对实验

基类检查通过后，先执行四项主控制，共 12 次：

```bash
python tools/run_prototype_completion_experiments.py \
  --dataset cub200 --seeds 1 2 3 \
  --variants baseline zero original frequency \
  --output-root "$COMPLETION_OUT/cub_all" \
  --support-root "$COMPLETION_OUT/support" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "$GPU"
```

补齐机制对照，共 15 次，复用同一支持清单：

```bash
python tools/run_prototype_completion_experiments.py \
  --dataset cub200 --seeds 1 2 3 \
  --variants ordinary_text permuted_text augmentation repeat block \
  --output-root "$COMPLETION_OUT/cub_all" \
  --support-root "$COMPLETION_OUT/support" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "$GPU"
```

两条命令使用同一输出根。完成后按完整九项重新汇总，以包含跨对照配对、参考预测审计及零回归；此命令只读取已完成结果，不启动缺失实验：

```bash
python tools/run_prototype_completion_experiments.py \
  --dataset cub200 --seeds 1 2 3 \
  --output-root "$COMPLETION_OUT/cub_all" \
  --support-root "$COMPLETION_OUT/support" --summarize-only \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "$GPU"
```

每个输出目录含 `run.log`、`run_spec.json`、`run_state.json`、`config.yaml`、`support.json`、`metrics.json`，启用补全时还含 `completion_calibration.json`，并按配置保存逐会话预测。矩阵会拒绝覆盖非受管理的非空目录。源码/配置变化会使运行指纹变化，应选择新的输出根；不建议用 `--rerun` 混合不同实现的结果。

`summary.json` 汇总逐种子指标、相对 baseline/original/augmentation 的配对差值、补全报告、query 辅助编码数以及支持和校准协议审计。基类验证、正式支持构建及查询推理成本应分别报告。基类验证选择之后，才可查看正式测试结果评估性能。

## 固定方法后的五种子三数据集

只有开发验证及机制对照支持继续时，才用固定配置执行下列矩阵；不根据正式测试结果更改 blend 网格或挑选数据集。默认九项 × 五支持种子 × 三数据集为 135 次。

```bash
for DATASET in cub200 cifar100 miniimagenet; do
  python tools/run_prototype_completion_experiments.py \
    --dataset "$DATASET" --seeds 1 2 3 4 5 \
    --output-root "$COMPLETION_OUT/benchmark_$DATASET" \
    --support-root "$COMPLETION_OUT/support" --execute \
    --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "$GPU"
done
```

若保留多个拟合/开发设置，应使用不同输出根，并明确它们是开发探索，不根据测试指标挑最优设置。

## 可选：已有显式频率描述

默认 `structured` 只用现有固定模板。若要评估已有 CUB 描述文件，以独立矩阵运行，所有相应对照保持相同候选数：

```bash
python tools/run_prototype_completion_experiments.py \
  --dataset cub200 --seeds 1 2 3 \
  --semantic-source explicit \
  --description-path description/cub200_frequency_descriptions_deepseek.json \
  --output-root "$COMPLETION_OUT/cub_explicit" \
  --support-root "$COMPLETION_OUT/support" --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "$GPU"
```

确认后将 `--dry-run` 改为 `--execute`。`ordinary_text` 始终使用普通文本，不随该参数变化；`permuted_text` 置换显式描述的尺度。启动器记录描述文件哈希，运行时校验描述内容。

## 实现验证

在已有包含 PyTorch、YACS 等依赖的项目环境中执行：

```bash
python -m unittest discover -s tests -p 'test_*completion*.py' -v
```

这些测试使用合成特征与模拟 Runner，检验统计补全、A/V 隔离、历史冻结、零修正回归、查询无辅助编码及矩阵协议。测试和 dry-run 都不代表真实数据集的性能评估；需执行上述 benchmark 命令才能获得性能结果。
