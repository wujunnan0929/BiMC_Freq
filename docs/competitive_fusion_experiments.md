# 竞争感知专家融合：实现、对照与 Ubuntu 实验

日期：2026-10-26。当前已实现代码；完整数据集的性能收益尚待服务器实验验证。

**本轮 CUB 开发结果：当前模型不进入第 7 节的 40 次正式矩阵。** 三个固定筛查相对温度参考仅平均 H +0.1165pp（仅 seed3 有收益）；四个 seed2 学习版本均选 step0/blend0，120 个已评估非零候选全部不可行，且各方法最高非零 H 即使忽略保护条件也低于参考。训练执行完整、SAFD 扰动存在，不能把零回退归因为分支未执行。训练任务总类数 15–50，而验证全部 55–100，需先分开核查类别迁移与任务规模外推，并检验证据尺度/类数先验的影响。当前未证明某一因素是唯一原因。完整审计见 `outputs/competitive_fusion_cub200/analysis_20261026/analysis_zh.md`；下列执行步骤为后续修订模型保留，不代表本轮已取得新性能。

下一轮的 matched-way、A-only 标准化和组判别探针已单独实现，执行 [竞争证据诊断](competitive_evidence_diagnostics.md)，复用本轮固定锚点；本文件原训练与正式评估命令保留作历史及后续参考。

## 1. 从筛查证据到可检验假设

CUB 补充筛查中，校准 BiMC 的开发 H=82.4127。保持相同视觉校准，独立 15-shot 教师提高新类约 4.62pp，同时降低旧类约 4.06pp，H 下降约 0.44pp。融合原型/协方差单专家擅长旧类，但新类明显弱；标签选五专家并集 H=91.5028。这支持检验竞争与互补性，不能推断可部署方法能获得约 9pp 的提升。

本轮假设：旧新类组之间的竞争和组内类别判别需要不同的调整规则。首先建立固定权重强对照，再测试基类训练的轻量条件模块。SAFD 维持辅助角色。原创性需另做文献定位，不能把小网络或一般门控本身当作创新证明。

## 2. 评分与模型

保留冻结 CLIP、原 BiMC 视觉/文本校准、协方差与普通描述。三个专家概率为 `p_f`、`p_cov`、`p_desc`，分支温度先在原始基类开发任务选择。

固定强对照：

```text
s_B = a_B p_f[B] + (1-a_B) p_cov[B]
s_N = (a_N p_f[N] + (1-a_N) p_desc[N]) exp(b_cal + b_delta)
```

独立搜索 `a_B/a_N ∈ {0,.25,.5,.75,.8,1}`、`b_delta ∈ {0,-.2,-.1,.1,.2}`，并显式包含完整校准 BiMC 的恒等配置。CUB 默认 180 个唯一候选，其他原始 alpha 不在网格中时会额外加入恒等候选。此前的 81 组分支温度/偏置对照也保留。

该搜索先选温度，再固定温度搜索融合权重，属于预设预算的顺序选择，不声称所有温度和权重组合的全局最优。

学习模块输出三个有界量：组间新旧赔率修正，以及旧/新组内两个专家混合权重的修正。改变组内混合后重新缩放，使组质量保持固定强对照的值；只有组间输出改变两组的相对质量。概念上对应：

```text
score_B(c) = fixed_mass_B × adapted_conditional_B(c)
score_N(c) = fixed_mass_N × exp(group_delta) × adapted_conditional_N(c)
```

输入为 36 维无类别编号证据：各专家的组质量、组内置信度/熵/间隔、专家分歧、旧新类数，以及两个候选类的支持样本数、集中度和图文相似度。预测接口不接收查询真实标签。支持质量来自允许的支持集，每类只保留 3 个标量。默认隐藏维度 16，架构共 643 个参数；消融开关决定使用哪些输入和输出。

`group` 仅调整组间赔率，不读取支持质量；`joint_evidence` 同时调整组内权重但不读取支持质量；`joint` 使用完整输入；`joint_safd` 只增加训练支持扰动。零输出在冻结推理时返回同一个固定参考对象，零 blend 无条件返回该对象。独立 `zero_full` 运行模式直接返回完整原 BiMC，用于验证接线。

**所有变体的基类会话均保持原 BiMC**，因为此时不存在增量类组。条件模块从首个增量会话开始参与评分。

## 3. 数据隔离与选择

沿用原始基类 A/V 类别分离协议，保留 A 的固定验证查询和 V 的 15 张诊断教师银行，使 seed 2/3/4 可与前次筛查对齐。V 的支持和查询仅用于开发选择，V 教师不进入拟合、校准搜索、固定权重搜索或 checkpoint 评分。

训练 episode 只采样 A 拟合池，支持/查询严格分离，不需要教师中心。旧类数 `(10,20,30)`、新类数 `(5,10,20)` 取可用的合法组合，伪新类支持为 2/5-shot；类数不足时使用明确的小规模回退，不宣称模拟了完整 200 类任务。协方差按原 BiMC 的旧/增量分组统计构造。

损失包含旧新组均衡的完整分类 NLL、组判别 CE、固定参考正确的旧类查询间隔保护及参考分布蒸馏。默认权重分别为 1、0.5、0.5、0.1。默认组赔率修正限制 ±2，组内混合修正幅度 ≤0.5，另通过验证 blend 控制总体幅度。

固定权重对照相对原 BiMC、校准 BiMC 的平均 B/N 下降各不得超过 0.2pp。学习模块还须相对固定强对照满足同一约束，按开发 H 选择步骤与 blend；并列保留更早步骤，允许零步/零修正。保护是平均开发指标约束，不是每类、每阶段或正式测试的无退化保证。

选择结束后，重新初始化网络和优化器，在全部原始基类上按选定步数重拟合，再冻结。此时可使用此前 V 银行中的原始基类图像。不得在正式增量测试后重新选择超参数或拟合种子。

## 4. SAFD 的作用与成本

`joint_safd` 在 25% 的伪新类训练支持集中替换一张样本的特征，使用已有 SAFD 粗/中尺度缓存。查询保持原图。独立辅助 RNG 保持与无辅助控制相同的类别、S/Q 索引及 episode 顺序。协方差、原型和支持质量依据实际扰动后的支持集一致重算。

增量阶段不更新模块权重，不保存旧支持图像或逐样本支持特征，不额外编码 SAFD 查询。只有无辅助 `joint` 有可重复增益后，才比较 SAFD、FFT、普通增强，并报告辅助缓存/离线拟合成本。方法包含基类梯度训练，不能称 training-free；固定网格对照无梯度，但仍使用基类开发标签选择参数。

## 5. Ubuntu：准备与固定融合筛查

同步本轮 `main.py`、`engine/`、`models/`、`utils/`、`tools/`、`configs/` 的变更，使用原 BiMC 环境。已有缓存包含原图与 SAFD 视图，不需重新编码或生成 LLM prompt。

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATASET=cub200
export CACHE=outputs/safd_metric_v2_20261022/cache/cub200_safd.npz
export OUT=outputs/competitive_fusion_cub200_20261026
export DATA_ROOT=../datasets/

ls -lh "$CACHE" "${CACHE%.npz}.json" "${CACHE%.npz}.support.json"
python tools/train_competitive_fusion.py --help
bash tools/run_competitive_fusion.sh screen
```

默认筛查 fitting seed 2/3/4，零梯度步骤，仅选择温度和固定权重并导出误差诊断。每个模型有 `.npz`、`.json`、`.training.json`、`.screening.json` 四个文件。新输出路径不得已存在；重跑需换 `OUT`。脚本任一步失败即停止，日志管道不会掩盖 Python 错误。

重点查看：

- `*.screening.json` 中 `reference`、`calibrated_reference`、`fixed_reference` 的 B/N/H；
- `fixed` 的两个融合权重和偏置，`*.training.json` 中所有候选及保护条件；
- `group_diagnostics` 中旧→新、新→旧及组内错误，并按阶段类数看变化；
- 原分支的旧组质量、置信度在真实旧/新查询上的差异。标签仅用于这些误差统计，不作为网络输入。

即使固定权重没有改善，也可以进行一次受控轻量训练来检验查询条件化的价值；固定权重已有改善时，学习模块必须显示额外收益。

## 6. 四个严格配对的训练对照

预先固定 fitting seed 2，不根据正式测试挑种子。下面复用同一 seed 的固定锚点，核验缓存、协议、CLIP、参考设置和原基类图像身份；不重复搜索温度或固定权重。

```bash
export FIT_SEED=2
bash tools/run_competitive_fusion.sh train
```

相当于对四个控制分别调用：

```bash
python -u tools/train_competitive_fusion.py \
  --cache "$CACHE" \
  --anchor-model "$OUT/models/${DATASET}_fixed_fit${FIT_SEED}.npz" \
  --method joint --auxiliary none \
  --seed "$FIT_SEED" --sequences 20 --teacher-shot 15 \
  --steps 300 --episode-pool 100 --validate-every 50 \
  --device cpu --threads 1 \
  --output "$OUT/models/${DATASET}_joint_fit${FIT_SEED}.npz"
```

脚本还执行 `group --no-support-quality`、`joint --no-support-quality`、`joint --auxiliary safd --auxiliary-probability 0.25` 三个控制。默认验证第 1/5/10/20 步及每 50 步，并遍历 blend `{.25,.5,1}`；零步/零 blend 始终保留。

如 `selected_step=0` 或 `blend=0`，必须记为学习模块没有额外增益；`learned_H_delta_vs_fixed_pp` 在这种情况下为 0。复用锚点时 `fixed_candidate_count=0` 表示本次未重复搜索，来源候选数保存在 training 报告的 `source_candidate_count`。

拟合种子稳定性使用相同协议、预设预算分别运行：

```bash
for FIT_SEED in 3 4; do
  FIT_SEED="$FIT_SEED" bash tools/run_competitive_fusion.sh train
done
```

不改变同一 fitting seed 内的配对条件。不把五个支持采样种子当作五次模型训练。

## 7. 冻结配置后的正式配对测试

2026-10-26 当前开发包的四个 seed2 模型按默认配置等价回退到原 BiMC，本轮不执行本节矩阵；待修订模型通过开发筛查后再使用以下命令。

```bash
export FIT_SEED=2
bash tools/run_competitive_fusion.sh plan
```

默认打印 8 变体 × 5 支持种子 = 40 次评估，不执行。四个控制及四个学习模型：

| 变体 | 目的 |
|---|---|
| baseline | 完整原 BiMC |
| zero_full | 加载模块但直接返回原参考；分数、预测须完全相同 |
| calibrated | 仅基类开发选定的分支温度/偏置 |
| fixed | 独立旧/新组固定融合权重，是学习模块的主要强对照 |
| group | 仅查询证据驱动的组间赔率调整 |
| joint_evidence | 同时调整组内混合，不读取支持质量 |
| joint | 完整条件融合，不使用辅助扰动 |
| joint_safd | 同一条件融合增加 SAFD 支持扰动 |

每个支持种子共享同一支持清单；四个学习模型必须来自同一缓存、拟合种子、协议、温度和固定权重，否则启动器拒绝。完成开发选择后执行：

```bash
bash tools/run_competitive_fusion.sh benchmark
```

仅做首个支持种子的接线测试可设置 `EVAL_SEEDS=1`，但不得根据其正式准确率调参。学习配置冻结后再用同一模型完整跑 1–5；已完成且 fingerprint 一致的运行可复用。

结果在 `$OUT/benchmark_fit2/summary.json`。汇总包含相对原 BiMC、相对 calibrated、相对 fixed 的配对差值；运行原始结果包含旧新组间及组内错误。模型和类统计冻结，原型保持原 BiMC；仅保存每类 3 个支持质量统计。checkpoint 不含历史逐样本支持重放数据。

成本分别记录缓存提取、锚点选择、当前拟合、支持质量准备（`fusion_prepare_seconds`）和查询分类。复用锚点时，单独复现一个方法需计入 `anchor_selection_wall_seconds`，多个配对控制共享锚点的实际开发成本只计一次。总墙钟时间包含初始化与准备；保留张量大小不是峰值显存。

## 8. CIFAR-100、mini-ImageNet 与通用 CLI

先完成 CUB 开发验证和冻结后的测试，再扩展数据集。替换对应完整基类缓存与 `OUT`，各数据集独立拟合，不跨数据集复用模型：

```bash
export DATASET=cifar100
export CACHE=outputs/safd_metric_v2_20261022/cache/cifar100_safd.npz
export OUT=outputs/competitive_fusion_cifar100_20261026
bash tools/run_competitive_fusion.sh screen
FIT_SEED=2 bash tools/run_competitive_fusion.sh train
FIT_SEED=2 bash tools/run_competitive_fusion.sh plan
# 开发选择冻结后：
FIT_SEED=2 bash tools/run_competitive_fusion.sh benchmark
```

mini-ImageNet 使用 `DATASET=miniimagenet`，对应 `miniimagenet_safd.npz` 与独立输出路径。实际缓存名不同则显式设置 `CACHE`。

不使用 Bash 包装也可直接运行：

```bash
python tools/run_competitive_fusion_experiments.py \
  --dataset "$DATASET" --seeds 1 2 3 4 5 \
  --model "joint=$OUT/models/${DATASET}_joint_fit2.npz" \
  --model "joint_safd=$OUT/models/${DATASET}_joint_safd_fit2.npz" \
  --reference-model joint --support-root "$OUT/support" \
  --output-root "$OUT/benchmark_joint_pair_fit2" --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;'
```

上面两模型子矩阵是 6 × 5 次；完整四模型脚本是 8 × 5 次。`--opts` 必须最后，矩阵管理的参考设置、模型路径、开关和 blend 不可覆盖。

FFT/普通增强控制复用 `tools/cache_safd_base_features.py --source fft|augmentation` 的行对齐缓存，再调用 `train_competitive_fusion.py --auxiliary fft|augmentation --aux-cache ... --anchor-model ...`。不同时更改教师/查询/类别采样预算，缓存原图列与来源身份会核验。

## 9. 回传结果与判定

首轮建议回传开发阶段结果，不包含大缓存：

```bash
tar -czf "${OUT}_development.tar.gz" -C "$OUT" models logs
```

重点是三份 fixed screening、四个拟合模型的 JSON/training/screening 报告；NPZ 很小，建议一起保留。正式评估还需保留 summary、各会话 metrics、支持清单、预测与配置。

判断贡献依次比较 calibrated→fixed、fixed→group、group→joint_evidence、joint_evidence→joint、joint→joint_safd。每一步都检查 B/N/H、最终会话、平均会话、遗忘、成本和种子稳定性。某项没有独立收益就收缩论文叙述；不把普通调权、一般门控或增强收益全部归到 SAFD。小缓存或非选定 active probe 的接线测试不能作为论文准确率证据。
