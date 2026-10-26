# 竞争融合的规模与证据诊断：下一轮有限预算实验

日期：2026-10-26。代码已实现，本轮是基类开发诊断，不运行正式 benchmark，不导出可直接用于正式矩阵的模型。

## 1. 要回答的问题

上一轮四个 seed2 学习版本都选择零修正，最高非零 H 也低于参考。训练总类数仅 15–50，而原验证全部为 55–100，不能直接区分类别迁移、任务规模外推、证据尺度和训练目标的影响。

本轮只改变验证规模和输入证据表示。保持原 CLIP、温度、固定融合权重、A/V 划分、训练 S/Q、损失、隐藏维度、学习率、优化步数和 blend 网格。`group/joint` 各使用四种输入，共 8 次小网络训练；所有任务只读取一次已有缓存并共用预计算统计。

## 2. 两个验证 bank

| bank | 默认 CUB 规模 | 作用 |
|---|---|---|
| matched | 旧类 10/20/30，新类 5/10/20，合法且训练实际出现的组合 | 选择 checkpoint/blend，检验同类数范围的迁移 |
| original | 原验证的旧类 50、新类 5–50，总类数 55–100 | 选择结束后观察规模外推 |

默认 20 个序列中，9 种 way 组合均被训练采样时，matched 共 180 个任务；实际以训练 episode 中出现且可从预留 bank 构造的组合为准。original 仍为原协议的 200 个阶段。

matched 从原验证每个序列的同一支持/查询银行取类别子集：旧类使用独立种子固定的嵌套子集，新类保留原到达前缀。新类属于 V，未参与拟合；A 的预留查询仍不进入拟合。保留原验证的每类支持/查询数与增量协方差分组策略，重新计算子任务的视觉校准、协方差及局部类别映射。不使用教师特征。

本轮匹配的是**候选类数范围**。训练旧类支持数量、协方差到达分组等其他差异仍存在，不声称两组任务已完全同分布。matched/original 共享图像与类别，不能当作两套独立统计重复。

两组任务难度不同，不直接比较其绝对 H。比较各自相对同一冻结锚点在该 bank 上的 ΔB/ΔN/ΔH。

## 3. 四种输入，形成两个因素的对照

| representation | 相对类数的表示 | A 拟合池标准化 |
|---|---|---|
| raw | 否，精确保留原输入接口 | 否 |
| standardized | 否 | 是 |
| prior | 是 | 否 |
| prior_standardized | 是 | 是 |

`prior` 是一组明确的证据变换：组概率质量的对数赔率减去 `log(n_old/n_novel)`；组内 top1 改为相对均匀分布的置信度；熵改为相对均匀分布的差异；间隔按组类数缩放；显式的三个类数输入置零。它减少特定的类数先验影响，不声称完全消除了极值统计等所有类数依赖，也不单独归因于一个减法。

这些变换只作用于网络输入，**不改变原专家概率、固定参考分数或其先验**。零输出/零 blend 继续回退到固定参考。

标准化均值和标准差仅由 A 拟合 episode 的查询证据估计，不使用 V 或 original bank 的统计。标准差常量维设为 1，其余下限为 `1e-4`；标准化后截断在 ±5。导出的诊断快照包含这些固定缓冲量，不在查询批次上重估统计。

`group` 与上一轮相同，仅使用查询证据，不读取支持质量；`joint` 与上一轮相同，使用支持质量并调整组内混合。因此四种表示在同一方法内构成严格输入对照；`group/joint` 的差异不只包含一个开关，不据其差值单独归因支持质量。

默认仍为 643 个网络参数，另保存 72 个标准化缓冲标量。无 SAFD、FFT 或新增强干预，不重新编码 CLIP、不生成新 prompt。

## 4. 选择、探针与信息隔离

每个 trial 从相同随机初始权重训练 300 步；100 个固定 episode、S/Q 索引和顺序共用。保持原损失及权重。验证第 1/5/10/20 步、每 50 步和最终步，blend 为 `{.25,.5,1}`，零步/零修正始终保留。

checkpoint/blend **只根据 matched bank** 选择：相对 matched 固定参考，平均 B/N 下降各不超过 0.2pp，最大化 matched 平均 H。零修正始终是该参考的可行回退。报告还提供相对原 BiMC、温度参考的额外保护检查，避免 frozen anchor 在 matched bank 上本身较差时误解收益。

选定同一个模型后才评估 original bank；original 标签、分数不用于本轮 checkpoint 选择。原规模 bank 已在以往开发中使用，仍不是新的独立测试集。本轮不做全基类重拟合。

每个 trial 另拟合一个 37 参数线性组判别探针，输入为同一证据表示，标签仅来自 A 拟合查询的旧/新组身份。100 步、学习率 .01，每个 episode 内两组等权；阈值固定为 logit 0，不在验证上挑阈值。

报告 fitting/matched/original 的逐任务平均 AUROC、逐任务平均 balanced accuracy 及 pooled AUROC。优先看逐任务指标，避免不同任务的组比例造成池化 AUC 假象。AUROC 随机水平为 .5，balanced accuracy 为 50%。线性探针失败不证明所有非线性规则无效；探针较好也不等于完整多类 H 会提高。

## 5. Ubuntu 执行命令

同步新文件：`models/competitive_evidence.py`、`engine/competitive_diagnostics.py`、`tools/diagnose_competitive_evidence.py`、`tools/run_competitive_evidence.sh`，并保留上一轮竞争融合代码。原模型和正式推理路径没有被修改。

使用原 CUB 缓存和上一轮 seed2 的固定锚点。路径来自上一轮服务器日志：

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATASET=cub200
export FIT_SEED=2
export CACHE=outputs/safd_metric_v2_20261022/cache/cub200_safd.npz
export ANCHOR=outputs/competitive_fusion_cub200_20261026/models/cub200_fixed_fit2.npz
export OUT=outputs/competitive_evidence_cub200_fit2_20261026

bash tools/run_competitive_evidence.sh
```

脚本运行 8 个 trial，并自动生成 `${OUT}.tar.gz`。`OUT` 及其同名日志/压缩包必须是新路径；不会覆盖已有结果。

不使用 Bash 包装的等价命令：

```bash
python -u tools/diagnose_competitive_evidence.py \
  --cache "$CACHE" --anchor-model "$ANCHOR" --output-dir "$OUT" \
  --seed 2 --sequences 20 --teacher-shot 15 \
  --methods group joint \
  --representations raw standardized prior prior_standardized \
  --steps 300 --episode-pool 100 --validate-every 50 \
  --probe-steps 100 --device cpu --threads 1
```

固定锚点的缓存 SHA、CLIP 身份、原基类图像 ID、参考设置、拟合种子、原协议 SHA 都会核验。不要改变 seed/sequences/query/teacher-shot 后继续使用旧锚点；这些变化需要对应协议的新锚点。

可用 `--methods group --representations raw standardized` 只跑预先指定的子矩阵进行接线检查，须换新输出路径。正式诊断建议保持完整 2×4 对照，不根据中间分数临时追加超参搜索。

## 6. 输出与判定

- `summary.md`：两个 bank 的固定参考和八个 trial 的增益、保护条件、探针 AUROC。
- `summary.json`：精确总表，包含各 bank 自己的参考指标。
- `banks.json`、`protocol.json`、`matched_protocol.json`：规模覆盖、A/V 与所有子任务的支持/查询索引。
- `fitting_episodes.json`：唯一一份共享训练采样轨迹。
- `trials/*.json`：每个 trial 的候选、训练损失、normalizer、输入漂移、探针及组间/组内错误。
- `models/*.npz/json`：A-only 诊断快照，不是全基类重拟合模型；格式为 `competitive_evidence_diagnostic_v1`，现有正式加载器会拒绝该格式。
- `provenance.json`：代码、缓存及锚点身份。

先看 `selected.matched.delta_vs_fixed_pp` 与 `selected.original.delta_vs_fixed_pp`，并同时看 B/N 下降和 `passes_group_guard`。两个 bank 使用同一 matched 选定模型，不各挑一个最优 checkpoint。

判定路径：

1. matched 有增益、original 无增益：存在规模迁移问题的证据，下一步解决迁移，不直接跑正式矩阵。
2. 标准化或相对类数表示改善 matched/original：定位了值得继续验证的输入问题，尚不构成论文新方法的有效性证明。
3. 探针较好但融合 H 无增益：优先检查学习目标与决策约束；不能只凭二分类 AUC 宣称可恢复原专家并集收益。
4. matched 和探针均无明确改善：结合 matched 参考是否接近满分判断剩余空间；停止当前分数层的无界调参。
5. 两组均有额外增益且通过保护：预先固定配置后，再在已有 seed3/4 对应锚点上重复诊断，之后才考虑全基类重拟合、SAFD 消融和正式测试。

这轮优先回传脚本自动生成的 `${OUT}.tar.gz`。不需要包含大型缓存，不进入上一轮的 40 次 benchmark 矩阵。
