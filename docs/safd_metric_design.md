# SAFD 尺度响应驱动的基类度量适配：实现与实验步骤

更新：2026-10-22。代码分支：`codex/safd`。

## 1. 当前结论与研究假设

此前 frequency guidance 的主要问题是：频率关系与原视觉关系高度重合，最终原型更新很小，基类伪验证上的微弱改善也没有稳定转移到三个数据集的最终准确率。继续增加 prompt 分支或叠加引导权重，暂时缺少实验证据支持。

本版改为检验一个不同的假设：**SAFD 尺度删除引起的特征响应，可以帮助识别原图特征中随尺度变化的类内方向；在这些方向上学习共享度量，可能改善未见类别的少样本分类。**

这里的“类内方向”不自动等于噪声或无关因素；某些方向也可能对新类别有判别力。是否应增强或抑制，由基类伪增量任务学习，并在类别隔离的验证任务中选择。不能预先宣称准确率提高、因果解耦或达到投稿标准。

主方法允许基类监督训练，**不再称为 training-free**。CLIP 始终冻结，增量阶段全部参数冻结；新类别仍按 FSCIL 协议从支持样本构造统计量。另保留无需梯度的闭式对照。

## 2. 表示与评分

### 2.1 嵌套 SAFD 视图

对同一张基类训练图像，使用同一套 SAFD 分解得到 `I = L + M + H`。缓存三个 CLIP 特征：

```text
x0 = normalize(CLIP(I))
x1 = normalize(CLIP(clamp(L)))
x2 = normalize(CLIP(clamp(L + M)))
u  = concat(x2 - x1, x0 - x2)
```

两个辅助视图不混入原图，不进行高频增强。响应向量保留幅值，不逐响应归一化，以免把极小响应放大成单位长度噪声。SAFD 的粗/中尺度表示是自适应嵌套逼近空间，不应直接等同于固定物理频段；像素截断后的编码差也不是可识别的因果效应。

### 2.2 SAFD 响应解释的类内散度

只用拟合子集，在各类内分别中心化原图特征 `x0` 和响应 `u`，得到 `e`、`v`。对响应作至多 64 维 PCA，得到投影 `R`，计算：

```text
Cee = e.T @ e / df
Cev = e.T @ v / df
Cvv = v.T @ v / df
N = Cev R (R.T Cvv R + ridge_scale I)^(-1) R.T Cev.T
```

`ridge_scale = ridge * max(trace(Cvv) / response_dim, variance_floor)`，其中 `response_dim=2D`，默认 `ridge=0.1`。正则尺度由完整响应空间的平均方差确定。`N` 是半正定的响应可解释类内散度；它不是一个新的因果识别定理。取其前 `r <= 64` 个正特征值对应的正交方向 `U`。

### 2.3 仅训练 r 个系数

```text
a = 0.9 * tanh(theta)
M = I + U diag(a) U.T
```

CLIP 和 `U` 不参加梯度训练，仅学习 `theta`。初始化 `theta=0`，对应严格恒等度量且有非零学习梯度。`M` 的特征值限制在 `[0.1, 1.9]`，保证正定；该限制不保证准确率不下降。零响应或零有效秩时自动退回恒等度量。

对每个已见类别，使用其**原图支持特征的原始均值**再归一化为 `mu`，计算：

```text
s_M(q, mu) = 2 q.T M mu - mu.T M mu
delta = s_M - s_I
log_votes = log(full_BiMC_votes) + 16 * delta
```

输出重新缩放为保持原行总量的正分数。原 BiMC 的视觉/文本校准与 ensemble 构成完整参考分支；度量残差统一作用于基类和所有已到达的新类，不只修改最新类别。增益为零或矩阵严格恒等时直接返回原参考分数，支持精确零对照。

SAFD 只在离线基类特征提取时使用。正式增量运行不对支持图或测试图额外提取 SAFD 特征；测试图只作一次原图 CLIP 编码。

## 3. 训练与选择协议

1. 只缓存原始基类训练图像及普通 BiMC 描述文本。不提取基准测试查询特征，不调用新的 LLM API。
2. 将原始基类划为互不相交的 A（拟合类别）和 V（伪新类别）。为 A 固定保留一组验证查询，在**全部验证序列**中使用同一保留组；这些图像不进入 `U` 拟合或梯度训练。
3. 仅用 A 剩余图像拟合响应散度与 `U`。从同一拟合池生成伪旧/新类训练 episode，各 episode 内支持/查询不相交。
4. 默认预生成 100 个训练 episode，1000 步循环使用。损失为旧类与新类查询交叉熵的等权平均，避免查询较多的组支配优化。
5. 每 100 步用 V 伪增量任务选择 checkpoint，指标为各验证序列/阶段的平均旧新调和准确率 H，同时约束旧类、伪新类的平均准确率相对恒等参考各下降不超过 0.2 个百分点。并列时保留更早步骤，包括步骤 0。
6. 选择结束后，重新初始化，并用**全部原始基类训练图像**重新拟合 `U`，只训练已选定的步数；不再选择 checkpoint。导出冻结矩阵及完整协议记录。
7. 正式 FSCIL 中加载矩阵，所有增量阶段不更新它。支持集合、数据映射、CLIP 权重、普通描述文本、模板、精度和预处理均检查与导出记录兼容。

验证序列共享图像，不能把 20 个序列当成 20 次独立统计重复。0.2pp 保护是基类伪验证上的选择约束，不是正式基准的无退化保证。全基类重新拟合后也不保证复现选择模型的收益，必须通过正式结果检查迁移效果。

## 4. 已实现的公平对照

| 名称 | 拟合方向来源 | 梯度参数 | 检验的问题 |
|---|---|---:|---|
| baseline | 原完整 BiMC | 0 | 是否优于原参考方法 |
| zero | 同一 SAFD 模型，但增益为零 | 增量 0 | 接线是否严格复现 baseline |
| safd | SAFD 响应解释的类内散度 | 至多 64 | 主方法 |
| original | 原图类内散度，同秩同训练器 | 至多 64 | 收益是否仅来自普通度量训练 |
| closed_form | SAFD 响应散度的闭式收缩度量 | 0 | 是否确实需要学习系数 |
| fft | FFT 嵌套视图响应，同训练器 | 至多 64 | 自适应分解是否优于 FFT |
| augmentation | 模糊/中心裁剪响应，同训练器 | 至多 64 | 是否只是额外视图带来的效果 |
| repeat | 原图精确重复，响应为零 | 0（有效秩 0） | 无响应时是否回退 |

闭式对照为 `M = (I + strength * N / max(trace(Cee)/D, floor))^-1`，使用相同 A/V 协议搜索 `[0, 0.1, 1, 10]`，随后全基类固定强度重估。它有监督统计拟合与选择成本，不能称为 data-free。

FFT/普通增强与 SAFD 都编码两张辅助视图，但分解耗时不同；当前只匹配辅助编码次数，**没有匹配像素失真强度**。缓存记录每张图的 pixel RMSE，应报告分布；若 SAFD 有收益，再补失真匹配实验。`original` 可复用 SAFD 缓存中的原图列以减少开发成本，但其算法无需辅助视图，论文计算成本不得强行计入这些未使用视图。

## 5. Ubuntu 上先完成 CUB 筛查

先把当前代码同步到服务器，进入已有、能运行原 BiMC 的 Python 环境。以下项目目录按此前服务器路径填写；若实际路径不同，只修改 `cd`。不要在不同目录下随意切换数据根路径：当前路径型数据的样本身份哈希绑定路径字符串，移动数据后应重新缓存和训练。

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DATA_ROOT=../datasets/
export RUN_ROOT=outputs/safd_metric_v2_20261022
mkdir -p "$RUN_ROOT/cache" "$RUN_ROOT/models"

python tools/cache_safd_base_features.py \
  --dataset cub200 --data-root "$DATA_ROOT" \
  --source safd --gpu 0 --device cuda --batch-size 32 --workers 4 \
  --output "$RUN_ROOT/cache/cub200_safd.npz"

python tools/train_safd_metric.py \
  --cache "$RUN_ROOT/cache/cub200_safd.npz" \
  --output "$RUN_ROOT/models/cub200_safd.npz" \
  --method learned --basis explained --device cpu \
  --steps 1000 --validate-every 100 --episode-pool 100 \
  --rank 64 --ridge 0.1 --lr 0.01 --sequences 20 --seed 1

python tools/train_safd_metric.py \
  --cache "$RUN_ROOT/cache/cub200_safd.npz" \
  --output "$RUN_ROOT/models/cub200_original.npz" \
  --method learned --basis original_within --device cpu \
  --steps 1000 --validate-every 100 --episode-pool 100 \
  --rank 64 --ridge 0.1 --lr 0.01 --sequences 20 --seed 1

python tools/train_safd_metric.py \
  --cache "$RUN_ROOT/cache/cub200_safd.npz" \
  --output "$RUN_ROOT/models/cub200_closed_form.npz" \
  --method closed_form --basis explained --device cpu \
  --strength-grid 0 0.1 1 10 --rank 64 --ridge 0.1 --sequences 20 --seed 1
```

`train_safd_metric.py` 只读取特征缓存，不加载 CLIP。CPU 默认即可；也可使用 `--device cuda`。正式实验不要使用 `--limit-per-class`、`--allow-smoke-cache`，不要删除步骤 0 候选或根据测试结果强制使用非零更新。缓存/模型默认拒绝覆盖；修改方案后使用新的目录名。

先检查 `models/*.training.json` 的 `reference`、`selected_validation`、`selected_step` 和 `gain`；在模型 `*.json` 中，前两项位于 `selection_report` 内。如果仍然选择 0，就记录为“未发现可迁移的改善信号”，不能把关闭适配的结果写成正向创新证据。

先打印配对运行计划：

```bash
python tools/run_safd_metric_experiments.py \
  --dataset cub200 --seeds 1 \
  --model safd="$RUN_ROOT/models/cub200_safd.npz" \
  --model original="$RUN_ROOT/models/cub200_original.npz" \
  --model closed_form="$RUN_ROOT/models/cub200_closed_form.npz" \
  --support-root outputs/safd_metric_support \
  --output-root "$RUN_ROOT/cub_screen" --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;'
```

再运行上述 5 个变体：

```bash
python tools/run_safd_metric_experiments.py \
  --dataset cub200 --seeds 1 \
  --model safd="$RUN_ROOT/models/cub200_safd.npz" \
  --model original="$RUN_ROOT/models/cub200_original.npz" \
  --model closed_form="$RUN_ROOT/models/cub200_closed_form.npz" \
  --support-root outputs/safd_metric_support \
  --output-root "$RUN_ROOT/cub_screen" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;'
```

变体共享支持清单；零对照应与 baseline 逐样本预测一致。单个测试种子仅用于检查运行与复现，不能用来反复选参数。是否扩大实验应优先看预先规定的基类验证信号，而不是挑选有利测试种子。

## 6. FFT 与普通增强对照

完成以下命令后，CUB 总共得到四种训练方向以及闭式对照。各方法保持相同拟合种子、秩、步数和验证预算：

```bash
for SOURCE in fft augmentation; do
  python tools/cache_safd_base_features.py \
    --dataset cub200 --data-root "$DATA_ROOT" \
    --source "$SOURCE" --gpu 0 --device cuda --batch-size 32 --workers 4 \
    --output "$RUN_ROOT/cache/cub200_${SOURCE}.npz"
  python tools/train_safd_metric.py \
    --cache "$RUN_ROOT/cache/cub200_${SOURCE}.npz" \
    --output "$RUN_ROOT/models/cub200_${SOURCE}.npz" \
    --method learned --basis explained --device cpu \
    --steps 1000 --validate-every 100 --episode-pool 100 \
    --rank 64 --ridge 0.1 --lr 0.01 --sequences 20 --seed 1
done

python tools/run_safd_metric_experiments.py \
  --dataset cub200 --seeds 1 2 3 4 5 \
  --model safd="$RUN_ROOT/models/cub200_safd.npz" \
  --model original="$RUN_ROOT/models/cub200_original.npz" \
  --model closed_form="$RUN_ROOT/models/cub200_closed_form.npz" \
  --model fft="$RUN_ROOT/models/cub200_fft.npz" \
  --model augmentation="$RUN_ROOT/models/cub200_augmentation.npz" \
  --support-root outputs/safd_metric_support \
  --output-root "$RUN_ROOT/cub_five_seed" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;'
```

该矩阵为 7 个变体 × 5 个支持种子 = 35 次评估。此处共用训练种子 1 的冻结模型，5 个支持种子只反映支持集抽样变化，不反映训练随机性。若出现稳定增益，应再预先固定 3 个训练种子，保存各自的 A/V 划分、导出模型和配对结果，报告两类变化来源。

## 7. CIFAR-100 与 mini-ImageNet

当基类验证支持继续实验时，下面的循环完成两个数据集的主方法、原图度量与闭式对照，以及各 5 个支持种子的配对测试。FFT/增强对照按第 6 节替换数据集名称执行。

```bash
for DATASET in cifar100 miniimagenet; do
  python tools/cache_safd_base_features.py \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --source safd --gpu 0 --device cuda --batch-size 32 --workers 4 \
    --output "$RUN_ROOT/cache/${DATASET}_safd.npz"

  python tools/train_safd_metric.py \
    --cache "$RUN_ROOT/cache/${DATASET}_safd.npz" \
    --output "$RUN_ROOT/models/${DATASET}_safd.npz" \
    --method learned --basis explained --device cpu \
    --steps 1000 --validate-every 100 --episode-pool 100 \
    --rank 64 --ridge 0.1 --lr 0.01 --sequences 20 --seed 1

  python tools/train_safd_metric.py \
    --cache "$RUN_ROOT/cache/${DATASET}_safd.npz" \
    --output "$RUN_ROOT/models/${DATASET}_original.npz" \
    --method learned --basis original_within --device cpu \
    --steps 1000 --validate-every 100 --episode-pool 100 \
    --rank 64 --ridge 0.1 --lr 0.01 --sequences 20 --seed 1

  python tools/train_safd_metric.py \
    --cache "$RUN_ROOT/cache/${DATASET}_safd.npz" \
    --output "$RUN_ROOT/models/${DATASET}_closed_form.npz" \
    --method closed_form --basis explained --device cpu \
    --strength-grid 0 0.1 1 10 --rank 64 --ridge 0.1 --sequences 20 --seed 1

  python tools/run_safd_metric_experiments.py \
    --dataset "$DATASET" --seeds 1 2 3 4 5 \
    --model safd="$RUN_ROOT/models/${DATASET}_safd.npz" \
    --model original="$RUN_ROOT/models/${DATASET}_original.npz" \
    --model closed_form="$RUN_ROOT/models/${DATASET}_closed_form.npz" \
    --support-root outputs/safd_metric_support \
    --output-root "$RUN_ROOT/${DATASET}_five_seed" --execute \
    --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;'
done
```

## 8. 保存与分析

请保留整个 `RUN_ROOT`，至少包含：

- `cache/*.json`、`cache/*.support.json`：特征来源、分解设置、像素失真、编码次数、源文件/权重/文本哈希与提取耗时。大体积 `.npz` 可以留在服务器，但复现需能恢复它。
- `models/*.npz`、`models/*.json`、`models/*.training.json`：冻结矩阵、拟合成本、完整划分、训练 episode、全部候选和选择结果。
- 每个评估根目录的 `summary.json`，以及各运行的 `metrics.json`、`support.json`、`config.yaml`、`run_spec.json`、日志与 `predictions_session_*.npz`。

主表报告 Session 0、Final、Avg、Novel、H、遗忘及支持种子的均值/标准差。与 baseline、original、FFT 作逐种子配对差值，利用逐样本预测统计被纠正/被破坏样本数。5 个支持种子仍然很少，应同时给出全部种子差值，不只给显著性结论。

计算表必须分开报告：离线缓存、选择与全基类重拟合、正式会话推理。正式评估 `metrics.json` 的 `total_wall_seconds` 不含离线工作；若缓存被多个对照复用，说明共享成本和单方法独立运行成本。运行时保留稠密 D×D 度量矩阵；“只有 64 个学习参数”不等于只存 64 个数。

优先检查这些结果：

1. `zero == baseline`；模型和文本身份检查通过；增量查询辅助编码数为 0。
2. 主方法是否实际选择非零更新，收益是否同时出现在 Novel 与 H，而不是只改善基类。
3. SAFD 是否优于同训练预算的 `original`、`fft`、`augmentation`；否则不能把提升归因于 SAFD。
4. learned 是否优于 closed-form；否则简化为闭式版本可能更有说服力。
5. 秩 16/32/64、删除粗/细响应、失真匹配、3 个训练种子等后续消融，只在预先固定的基类验证上选择，再一次性测试；目前没有将这些扩展全部实现为自动矩阵。

若多数数据集仍选零或与普通度量控制相当，应收缩贡献描述，并重新审视尺度响应是否有稳定的跨类信息，不应继续堆积模块来迎合投稿。当前实现是可检验的新路线，不是已经达到 Neurocomputing / IEEE TAI 录用水平的结论。

## 9. 相关工作定位与实现位置

自适应分解、度量学习、冻结 CLIP 适配本身都不能作为“首次”贡献。SAFD 在少样本/增量学习已有相关工作记录；正式写作需核对全文及与已有工作的关系。可主张的候选贡献应收窄为：**SAFD 尺度响应解释的类内方向、受限的基类 episodic 度量学习，以及不增加增量查询编码的统一残差评分**，并由上述控制证明各部分必要性。

参考原始来源：[AFD/SAFD 表示理论](https://arxiv.org/abs/2205.00844)、[FDR：FSCIL 特征分解与重组](https://openaccess.thecvf.com/content/ICCV2025/html/Xue_Feature_Decomposition-Recomposition_in_Large_Vision-Language_Model_for_Few-Shot_Class-Incremental_Learning_ICCV_2025_paper.html)、[ProKeR：冻结视觉语言模型的核适配](https://openaccess.thecvf.com/content/CVPR2025/html/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.html)。这些文献支持相关工作的存在，不证明本方案有效或有优先权。

- `models/safd_interventions.py`：嵌套视图与响应。
- `models/safd_metric.py`：散度、低秩学习器、闭式控制与评分。
- `tools/cache_safd_base_features.py`：基类特征缓存。
- `engine/safd_metric_training.py`、`tools/train_safd_metric.py`：拟合/验证/重拟合/导出。
- `engine/safd_metric_runtime.py`、`engine/engine.py`：兼容性检查与冻结运行。
- `tools/run_safd_metric_experiments.py`：配对矩阵、零对照与汇总。

## 10. 当前验证范围

本地完整回归测试共 300 项通过，包括非零梯度更新、A/V 类别与查询隔离、闭式不使用 optimizer、真实 Runner 多会话冻结、文本/数据来源检查和精确零对照。文档展开循环后的 21 条 CLI 命令均通过参数解析审查。

另使用真实冻结 CLIP ViT-B/16 和本地 12 类 PNG 子集完成接线验证：8 个基类、每类 12 张支持图，缓存 96 张原图和 192 张辅助视图；基类小规模训练、导出、3 个会话的 baseline/zero/selected 均运行成功，zero 与 baseline 逐样本预测一致，增量查询辅助编码为 0。该小规模验证选择步骤 0 / gain 0，不能据此宣称提升，也不是 CIFAR-100 完整基准结果。

日志与核验记录位于 `outputs/safd_metric_verification_20261022/`。三个完整数据集的新方案准确率仍需在服务器按本文协议运行后确认。
