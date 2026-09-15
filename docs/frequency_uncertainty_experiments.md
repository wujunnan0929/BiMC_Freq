# 频带不确定性校准：实现与执行协议

本次实现是一个待验证的 training-free 实验，不预先声称提升准确率。
所有命令从仓库根目录运行，沿用已有 PyTorch、torchvision、YACS 环境。
无需新增频率描述文件或调用 LLM。

## 方法

每张图像的原图或频率视图特征先在 FP32 中归一化，然后按类别估计
原始均值（不再次归一化）、无偏对角方差和样本数。
共享先验为基类的 pooled within-class 方差：

```
v_base[f,d] = sum_c (n_c-1) * v_c[f,d] / sum_c (n_c-1)
rho_c = tau / (tau + n_c - 1)
v_shrunk[c,f,d] = (1-rho_c)*v_c[f,d] + rho_c*v_base[f,d]
v_predictive[c,f,d] = v_shrunk[c,f,d] * (1 + 1/n_c)
```

单样本类别使用完整先验，包括 tau=0 的情况。方差下限默认 1e-6。
`1+1/n` 是均值估计不确定性的 plug-in 近似，不积分协方差不确定性，
不是精确贝叶斯后验；可用 `UNCERTAINTY.MEAN_UNCERTAINTY False` 消融。

每个频带的得分为 `-0.5 * mean_d((q-mu)^2/v_predictive + log(v_predictive))`，
再对频带等权平均。保留 log-variance 项，防止无限放大方差获得虚假优势。
评分使用分块直接差值，避免分配完整 N×C×B×D 张量。

辅助得分经过温度 softmax，与完整 BiMC votes 混合：

```
output = (1-alpha)*reference + alpha*sum(reference)*softmax(auxiliary/T)
```

每行 vote 总量保持不变，alpha=0 精确返回原参考张量，并跳过查询额外编码。
新分支不与旧 FREQUENCY fusion、router、consensus、incremental residual 同时启用。
CLIP 始终冻结，所有新统计和校准均无梯度更新。

## 只用基类选择参数

默认把 100 个 CUB base 类按固定种子分成互斥两组：50 类拟合方差先验，
50 类采样验证伪增量序列。每个序列有 10 个旧类和两批各 5 个新类，
旧类 20-shot、新类 5-shot，每类 5 个独立 query，共 20 个序列。
支持集与查询集不重叠，单个序列中的历史支持集保持固定。

搜索网格：tau=[5,20,100]，temperature=[0.1,0.3,1.0]，alpha=[0,0.05,0.1,0.2]。
选择目标在每个增量阶段对 old/new 准确率各赋一半权重，再对阶段等权平均；
排除 stage 0，new 包括所有已加入的增量类。平局依次选较小 alpha、tau、temperature。
alpha=0 只计一个候选，shared 模式忽略 tau 的无效重复候选。

选择完超参数后，将所有 base 训练图片重新用于估计共享先验，包含原验证组
的支持/查询图片；不使用 benchmark 测试图片或未来类别。随后超参数与先验冻结。
这次 refit 会改变先验数值，日志明确记录 selection prior 和 final prior。
每个真实增量 session 仅追加当前类别均值、方差、计数，保留历史类别统计不变。

## 推荐先运行四组单 seed 验证

Ubuntu/Bash，DATA_ROOT 指向包含 CUB_200_2011 文件夹的目录：

```bash
DATA_ROOT=/absolute/path/to/datasets
OUT=outputs/frequency_uncertainty_cub

python tools/run_frequency_uncertainty_experiments.py \
  --suite core --seeds 1 \
  --variants baseline zero original_shrinkage frequency_shrinkage \
  --output-root "$OUT" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID 0 \
         TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu
```

把 `--execute` 替换为 `--dry-run` 可预览所有命令。
`--opts` 必须最后。FFT_DEVICE=cpu 只将 FFT 放到 CPU，CLIP 仍在 GPU。
如果服务器 cuFFT 正常，可在开始新实验前固定使用 auto，后续保持一致。

## 三个 seed 的完整对照

```bash
python tools/run_frequency_uncertainty_experiments.py \
  --suite core --seeds 1 2 3 \
  --output-root "$OUT" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID 0 \
         TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu
```

| 变体 | 特征 | 协方差处理 |
|---|---|---|
| baseline | 原 BiMC | 新分支关闭 |
| zero | 频率视图 | 手动 alpha=0，检查完全回退 |
| original_shared | 单路原图 | 共享基类方差 |
| original_shrinkage | 单路原图 | 类别方差向基类先验收缩 |
| frequency_shared | 三路频率视图 | 每频带共享基类方差 |
| frequency_shrinkage | 三路频率视图 | 每类别、每频带的收缩方差 |

core 为 6×3=18 个运行。`--suite all` 增加已有 router，共 21 个运行。
原图对照无需三次重复编码。shared 对照默认仍包含同样的 `1+1/n` 修正。
各非零变体用同样的基类划分、验证序列、搜索范围分别选参；比较的是经过
相同协议校准的完整系统，不是固定相同最终 alpha 的纯推理干预。

同 seed 共用支持集 manifest。完成的单 seed 运行可在代码、配置、路径、
覆盖参数均不变时复用；指纹不匹配时使用新的输出目录，不要强行覆盖旧结果。
运行器默认只预览，只有 `--execute` 启动训练/评估进程。

## 只校准基类，不评估测试集

```bash
python main.py \
  --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc_frequency_uncertainty.yaml \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;' \
         TRAINER.BiMC.FREQUENCY.FFT_DEVICE cpu \
         TRAINER.BiMC.UNCERTAINTY.BASE_ONLY True \
         OUTPUT_DIR outputs/uncertainty_base_validation
```

不设置 BASE_ONLY 就是单次完整 FSCIL 运行。BASE_ONLY 仅支持直接 main.py，
矩阵运行器要求完整 session metrics；没有用空测试指标伪造完成状态。
如要对原图单独诊断，再加 `TRAINER.BiMC.UNCERTAINTY.VIEW_CONTROL original`。

## 结果与判据

每个运行输出 `metrics.json`、`uncertainty_calibration.json`、`support.json`、
`config.yaml`、逐 session 预测 NPZ 和 `checkpoint.pt`。
checkpoint 保存先验、冻结参数、各类统计；不保存个体图像特征，也不是自动断点恢复。
矩阵输出根目录下的 `summary.json` 包含配对均值/标准差、参照差值、支持集、
原预测及自动校准协议审计。manual zero 没有验证序列，不参与自动协议一致性比较。

优先看 Final/Average/Novel/H，以及历史增量类的纠错与破坏。
frequency_shrinkage 要同时与 original_shrinkage、frequency_shared 比较，
才能区分额外频率信息与类别方差收缩的作用。若自动选择 alpha=0，应报告
基类校准没有支持增加分支；不能根据测试准确率改选正 alpha。

默认复用 V2 natural 视图：低通、低+中频、原图+高频增强。三路内容重叠，
不应声称是三个独立频段的精确联合密度。该实验先验证现有频率特征是否
能因不确定性建模而改善，没有新增局部表征。验证任务候选类数量仍少于
正式 100–200 类；结果需用配对多 seed 和其他数据集确认。

## 本地逻辑验证

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -q
```

测试覆盖统计公式、单样本、数值稳定性、基类划分、support/query隔离、
零融合、三 session 统计冻结、BASE_ONLY 不访问测试集、矩阵命令与审计。
合成测试不能替代完整 CUB 准确率实验。
