# Training-free 联合频率 GDA：方法与实验协议

## 为什么继续这个方向

上一轮 CUB 三种子结果：BiMC Final=72.472%，原图共享对角方差=73.404%，
频率收缩方差=73.421%。频率方法相对强原图对照只提高 0.017 个百分点，
Novel 反而降低 0.284 个百分点，尚不能证明频率是主要收益来源。

本轮研究假设是：三个自然频率视角彼此重叠，独立对角评分再平均没有利用
视角间及特征维度间的相关性。将**原图和三个频率视角联合建模**，能否得到
超过原图分类器的额外信息。该假设需要实验验证，不能预先声称准确率提升。

共享协方差 GDA 本身是已有方法，参考
[ICLR 2024 GDA](https://arxiv.org/abs/2402.04087)及其
[作者代码](https://github.com/mrflogs/ICLR24)。本实现不是该论文的逐项复现，
也不把 GDA、闭式分类器或协方差估计本身作为新颖性。

BiMC 已有的原图 Mahalanobis 分支使用整体特征协方差、经校准的原型，并只
参与基类分支融合。本轮辅助分支采用基类的类内残差协方差、原始类别均值，
给全部已见类别统一打分。原图 GDA 对照用于区分这些差异与频率信息的作用。

## 数学与 training-free 边界

每个视角的 CLIP 特征先转 FP32、分别 L2 归一化。联合特征按顺序拼接：

```
z = concat(original, low_pass, low_and_middle, high_enhanced)
P = 4 * 512 = 2048  # CLIP ViT-B/16
mu_c = mean(z_i for i in class c)  # 不重新归一化类别均值
Sigma = sum_c sum_i (z_i - mu_c)(z_i - mu_c)^T / sum_c(n_c - 1)
Sigma_r = Sigma + ridge * max(mean(diag(Sigma)), var_floor) * I
Q = inverse(Sigma_r)
score_c(z) = (z^T Q mu_c - 0.5 mu_c^T Q mu_c) / P
```

同一张图像的视角严格对齐后才计算跨视角协方差。`block_shared` 保留每个视角
内部的完整 512×512 协方差，将视角之间的块置零；`full_shared` 保留所有块。
block 和 full 使用同一正则化定义。`repeat` 用原图特征重复四次作为维度控制，
不额外编码图像。重复视角会造成秩亏，正 ridge 保证求解稳定。

只计算类内散度，不把类间均值差异当作噪声。各类先验相同，不按样本数给基类
加权。新 GDA 模式不使用类别特定协方差、log-determinant 或 `1+1/n_c`；强制
`MEAN_UNCERTAINTY=False`。原图/频率旧对照保留原来的统计定义。

矩阵分解和求逆使用 FP64 Cholesky，最终分类器使用 FP32。
每次类别集合改变时缓存 `Q @ mu` 与偏置，查询批次只做线性打分。
辅助分数经过 temperature softmax 后与完整 BiMC votes 混合：

```
output = (1-alpha)*reference + alpha*sum(reference)*softmax(score/temperature)
```

alpha=0 精确返回参考分数并跳过查询频率编码。所有网络权重冻结，无 optimizer、
backward、可训练 router 或测试时梯度更新；仍使用标签估计统计量和校准超参数。

## 基类选择协议

沿用原实验的基类互斥划分和 support/query 隔离伪增量序列。
先只用 fit 类拟合协方差，再在 validation 类上选择参数；之后固定超参数，
用全部基类训练样本重估一次协方差。真实增量阶段只追加新类均值、方差和计数，
共享协方差与 precision 始终冻结。旧类个体图像特征不跨 session 保留。

- ridge 网格：`[0.01, 0.1, 1.0]`。
- alpha 网格：`[0, 0.05, 0.1, 0.2]`。
- temperature 网格：`[0.1, 0.3, 1.0]`。
- alpha=0 只保留一个候选，其余参数对零融合预测没有影响。
- 同分时依次选择更小 alpha、更大 ridge、更小 temperature。
- `PRIOR_STRENGTH` 在新 GDA 模式中不参与计算，仅为旧日志接口兼容保留。

保持原来的 old/new 平衡准确率目标，避免本轮同时改变多个机制。
它允许 old 提升补偿 new 下降，因此必须查看校准和测试的分组准确率。
校准任务只有 15/20 个候选类，小于 CUB 正式评估的 110–200 类；这项局限尚未
解决。全部基类 refit 也会改变协方差。不能根据测试集更换参数或挑选 seed。

## 对照矩阵

| 变体 | 辅助特征 | 辅助分类统计 |
|---|---|---|
| baseline | 无 | 原 BiMC |
| original_shared | 原图 | 上轮强对照，共享对角预测方差 |
| frequency_shrinkage | 三频率视角 | 上轮频率方案，类别收缩对角预测方差 |
| original_gda | 原图 | 完整共享类内协方差 GDA |
| frequency_gda_block | 原图+三频率视角 | 各视角内部完整协方差，无跨视角相关性 |
| frequency_gda_joint | 原图+三频率视角 | 完整联合协方差 |
| original_diagonal（all） | 原图 | 共享对角方差，关闭均值不确定性修正 |
| original_repeat_gda（all） | 原图重复四份 | 完整联合协方差，无额外图像信息 |

`core` 为 6 组，`all` 为 8 组，全部 training-free。所有同 seed 变体共用支持集、
基类划分与验证序列；各变体分别按同一校准协议选参。它比较的是校准后的完整
系统，不是固定最终温度/正则化系数的纯推理干预。

## 执行命令

在原 Linux 服务器的项目根目录、原 Python 环境执行。DATA_ROOT 必须和上轮
使用相同的路径写法，使支持集目录标识一致。下面显式复用原支持集根目录；
若服务器原支持文件缺失，运行器将按同一 seed 重新生成，需要查看配对审计。
更新所有修改及新增代码后再开始，运行期间不要再改代码或配置。

先进行单种子六组实验：

```bash
DATA_ROOT=/absolute/path/to/datasets

python tools/run_frequency_discriminant_experiments.py \
  --suite core --seeds 1 \
  --output-root outputs/frequency_discriminant_cub \
  --support-root outputs/frequency_uncertainty_cub/support \
  --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID 0 \
         TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu
```

完整三种子扩展对照（24 组，代码配置不变时会跳过已完成的六组）：

```bash
python tools/run_frequency_discriminant_experiments.py \
  --suite all --seeds 1 2 3 \
  --output-root outputs/frequency_discriminant_cub \
  --support-root outputs/frequency_uncertainty_cub/support \
  --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID 0 \
         TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu
```

将 `--execute` 替换为 `--dry-run` 可只检查计划。`--opts` 必须最后。
旧 uncertainty 输出目录不能用于新实验；这次实现和配置有变化，旧 fingerprint
会不同。新目录会重跑 baseline/旧统计对照，以验证兼容性，不覆盖已有结果。

只做基类校准、暂不读取测试集：

```bash
python main.py \
  --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc_frequency_discriminant.yaml \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;' \
         TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu \
         TRAINER.BiMC.UNCERTAINTY.BASE_ONLY True \
         OUTPUT_DIR outputs/discriminant_base_validation
```

## 看什么结果

结果在 `outputs/frequency_discriminant_cub/summary.json`。关注三项配对比较：

1. `original_gda` vs `original_shared`：完整类内协方差有没有价值。
2. `frequency_gda_joint` vs `original_gda`：频率是否提供额外收益。
3. `frequency_gda_joint` vs `frequency_gda_block`：跨视角协方差有没有价值。

扩展对照区分关闭均值不确定性、重复特征维度的影响。必须同时报告 Final、Avg、
Novel、H 和逐 seed 差值，尤其检查新类误伤、评估耗时及内存。单种子用于检查
实现与可行性，不用于宣称提升或 SOTA，也不据此挑选最有利的 seed。

`uncertainty_calibration.json` 保存矩阵形状、迹、对角范围及 SHA256 摘要；
完整最终协方差、precision、类别统计及已构造的线性分类器保存在 `checkpoint.pt`。
校准阶段矩阵的完整数值不写入 JSON，避免每轮产生数百万项 JSON 数组。
checkpoint 是 session 边界快照，不是中途自动续跑功能。

重复四份原图不保证与单份原图结果相同：拼接改变有效正则化及分数尺度，
有限的 ridge/temperature 搜索网格也不完全等价。该对照用于测量这些影响。

## 本地验证

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -q
```

合成验证覆盖类内散度、完整/分块差异、直接 Mahalanobis 公式、均匀类先验、
半精度输入、奇异重复视角、无梯度、基类数据隔离、增量冻结、缓存刷新、
checkpoint、零融合及 BASE_ONLY 不访问测试集。它不能替代服务器上的 CUB 实验。

本次本地验证：全部 175 项测试通过；另外通过了 2048 维相关视角、低秩协方差的
数值检查。当前未运行完整 CUB 新实验，因此没有新的 benchmark 准确率可报告。
