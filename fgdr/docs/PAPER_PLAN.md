# 论文研究与实验设计方案（目标：Neurocomputing 及同档期刊）

> 配套代码在 `fgdr/` 与 `tools/fgdr_*.py`；分析依据见此前两版报告《频率属性的免训练 FSCIL 校准》（v1、v2）。本文只写怎么做成一篇能投的论文：故事、方法、实验、闸门、时间线、写作。

## 0. 现状与总体策略

**能确定的事实**（来自你的实验记录和两篇原文）：

- 现有约 19 条频率路线，在有同结构原图对照的比较里，频率部分的增量都在 −0.77～+0.02pp 之间，落在噪声带内。
- 唯一稳定的大增益是把 BiMC 的会话总体协方差换成池化类内 GDA：CUB Final +1.99pp，3/3 seed 为正。这与频率无关。
- FDR（ICCV 2025）的免训练版本在 5-shot 下相对 TEEN 只高 0.1–0.4pp Aavg，CUB Aavg（74.1）低于 BiMC（75.36）。它的大幅增益在 1-shot 的新类准确率上。FDR 代码里的关键词掩码坐标与联合嵌入空间不一致，还有一步论文没写的文本权重整体校准。

**策略**：一篇 Neurocomputing 档的论文需要一个可重复的正结果，加上让审稿人信服的分析。频率能不能当主角，第 1–2 周就能用闸门判定。所以同时准备两条论文主线，按闸门结果二选一，代码和实验矩阵两条线共用。

| | 主线 A（频率成立） | 主线 B（频率不成立） |
|---|---|---|
| 触发条件 | G0′、G1′ 通过，且 Q1 在伪增量上比 K0 高 ≥0.2pp，同时高于 Q2/Q3/Qp ≥0.1pp | 其余情况 |
| 题目方向 | 频率 grounding 的分解-重组：看尺度，不看词 | shot-aware 统计校准 + 文本引导的分段重组；频率作为受控负结果分析 |
| 方法主体 | 视觉频率 grounding 分段 + 文本选供体 + shot-aware 判别锚点 | 修正坐标的关键词分段 + 文本选供体 + shot-aware 判别锚点 |
| 频率的位置 | 方法核心（第 4 节） | 分析节（第 5.6 节）：为什么 CLIP 嵌入里的频率信息是冗余的 |

两条线的统计锚点和协议贡献完全相同，主线 B 并不需要额外工作量。

## 1. 目标期刊与投稿标准

- 首选 Neurocomputing。同档可备选 Neural Networks、Knowledge-Based Systems、Expert Systems with Applications、Engineering Applications of Artificial Intelligence；Pattern Recognition 要求更高，可作为冲刺。分区以你单位认定的版本为准。
- 这一档期刊对 FSCIL 方法论文的常见期望，也是本方案对齐的标准：
  1. 问题和动机有证据，而不是"引入 X 模块"；
  2. 方法有原理、可解释，不是堆模块；
  3. 三个标准基准上优于或可比于同类免训练方法，并与训练类 CLIP 方法放在同一张表里；
  4. 消融、超参敏感性、效率分析齐全；
  5. 结果多 seed、可复现。
- 本方案额外的加分点：严格的 base-only 选参协议、锁定后只评估一次、配对显著性检验。FSCIL 论文里这样做的不多，可以在贡献里明确写出来。

## 2. 问题陈述与贡献（两条线共用的写法）

**问题**：免训练 CLIP FSCIL 有两类偏差。

- **原型偏差**：5-shot（尤其 1-shot）的新类原型偏离真实类中心。现有重组方法（TEEN、BiMC 的模态内校准、FDR）向相似基类借信息。借的单位要么是整个原型，要么是由文本 grounding 的通道段。
- **统计分支的 shot 偏差**：BiMC† 用会话总体协方差（含类间散布），对所有类用同一个 plug-in 距离，没有考虑新类均值只由 n_c 个样本估计。

**贡献（定稿时按主线取舍）**：

1. **Shot-aware 预测判别锚点**。在 base 会话上估计池化类内协方差，并冻结。新类得分用高斯后验预测分布：方差按 (1+κ/n_c) 膨胀，并带对数行列式项；κ=0 时退化为 plug-in GDA。它以概率混合的方式接入 BiMC† 票数，α=0 时精确回退。这是对 BiMC 统计分支的修正，有清晰的贝叶斯解释，不是又一个混合距离。需要与 HyCal（CVPR 2026，按样本数门控的余弦/马氏混合）、GDA（ICLR 2024）、FeCAM 明确区分。
2. **文本选供体的分段重组**。对 FDR 做受控拆解。发现它的增益主要来自"文本相似度选供体"和"按关键词固定的通道子集"。据此给出修正坐标的分段重组：通道段落在联合嵌入的真实坐标里，供体由 BiMC 校准后的文本原型选择。
3. **（主线 A）视觉频率 grounding**。用逐级高斯模糊下的通道敏感度，给每个类定义尺度分段。它只依赖图像，新类可以由自己的 shot 算出，替代 FDR 的文本 grounding。
   **（主线 B）频率受控分析**。证明在冻结 CLIP 的嵌入里，滤波视图与尺度文本提供的类别信息被原图嵌入吸收。证据包括：信息上界检验 D3、掩码诊断、频率特异性对照。
4. **协议与可复现性**。所有超参只在 base 类的伪增量 episode 上选择，锁定后在测试集上只评估一次。报告 5 个 support seed 的均值±标准差、配对差、McNemar 与 bootstrap 区间。代码与缓存协议一起公开。

## 3. 方法（实现见 `fgdr/`）

记归一化 CLIP 图像特征为 f(x)，BiMC 校准后的文本原型为 t̃_c = unit((1−λ_T) t_c + λ_T d̄_c)。

**3.1 锚点**（`fgdr/anchor.py`）
- W = 池化类内协方差（只用 base），P = (W + r·mean(diag W)·I)^{-1}。
- s_c(x) = −½ (x−μ_c)ᵀP(x−μ_c)/(1+κ/n_c) − (d/2)·log(1+κ/n_c)。
- 最终票数：v′ = (1−α)·v_BiMC† + α·mass(v)·softmax(s/T)。
- 可选：新类 μ_c 用校准后原型 × 原始均值范数（A2c）。

**3.2 分段重组**（`fgdr/recompose.py`）
- 先做整体重组：p̃_n = unit((1−λ)p_n + λ·unit(Σ_b w_nb p_b))，w_nb = softmax_b(τ·cos(t̃_n, t̃_b))，可选 top-r。供体权重用视觉相似度、λ=0.1、τ=16 时，精确等于 BiMC。
- 再对每个段 g 迭代：w^g = softmax_{top r}(τ_s·cos(t̃_n[g], t̃_b[g]))，p̃_n[g] ← (1−α)p̃_n[g] + α Σ_b w^g_b p_b[g]。
- 段的来源：关键词（修正坐标或 FDR 坐标）、频率 grounding、或对照（随机 / 换类 / 非频率扰动）。

**3.3 频率 grounding**（`fgdr/grounding.py`）
- Δ_k(x) = unit(f(G_{σ_{k−1}}x)) − unit(f(G_{σ_k}x))，σ = (0, 1, 2, 4) px。
- S_{c,k} = mean_{x∈c} Δ_k(x) − mean_{b∈base} S_{b,k}，g_{c,k} = TopQ(|S_{c,k}|)。
- 新类直接用自己的 support 计算，不需要任何文本。

**3.4 复杂度**：所有步骤都是闭式运算。每个会话新增的计算是 O(n·S) 次额外前向（S = 模糊级数）和若干 d×d 矩阵运算。效率表需要报告相对 BiMC† 的耗时倍数。

## 4. 实验设计

### 4.1 设定

- **数据集**：CIFAR-100（60 + 8×5-way）、miniImageNet（60 + 8×5-way）、CUB-200（100 + 10×10-way）。
- **shot**：主表 5-shot；补充 1-shot（CUB、mini，对齐 FDR 的设定）与 10-shot（shot 曲线）。
- **support 抽样**：沿用 BiMC 代码的协议（DatasetManager 按 seed 随机抽），seed 取 1–5，报告均值±标准差。正文要写明这与 CEC 固定划分的区别，并说明表中 BiMC 是在同一协议下复现的数字。
- **骨干**：主实验 CLIP ViT-B/16；鲁棒性实验 ViT-B/32、ViT-L/14（CUB + mini）。
- **选参**：只在 base 类伪增量 episode 上进行（划分种子 2/3/4，每个划分 20 条序列）。护栏为相对 BiMC† 的 Base / Novel 降幅都不超过 0.2pp，平局取更简单的配置。锁定后对 3 个数据集 × 5 个 seed 只评估一次。

### 4.2 对比方法

| 类别 | 方法 | 来源 |
|---|---|---|
| 免训练（同一管线复现） | CLIP 零样本、ProtoNet、TEEN、GDA（适配 FSCIL）、FDR（免训练，官方脚本超参）、BiMC、BiMC†、BiMC+FDR | `fgdr/configs/baselines.json` |
| 免训练（引用论文数字） | BiMC/BiMC†（CVPR 2025 正文与补充材料）、FDR（ICCV 2025）；HyCal（CVPR 2026）与 ULA（CVPR 2026 Findings）先确认骨干与基准是否可比，HyCal 主要报告 XD-VSCIL 基准 | 原文 |
| 训练类 CLIP（参考行，单独分组） | CPE-CLIP、PriViLege、LP-DiF、CLIP-M3、FDR+LoRA、SGLC（CVPR 2026） | 原文 |

训练类方法单独成组，标明是否训练，不与免训练方法直接比高低。

### 4.3 主表与补充表

| 编号 | 内容 | 命令 |
|---|---|---|
| T1 | 三个数据集 5-shot：各会话准确率、Avg、Last、PD；Last 会话的 Base / Novel / H | `final` 输出的 `table.md` |
| T2 | 1-shot（CUB、mini）：各会话 Acc 与 NAcc | `final --shot 1` |
| T3 | 组件消融：BiMC† → +A1 → +A2(κ) → +K0 → +K1 →（A: +Q1 / B: 不加）→ 完整 | 每个组件都是锁定后的 family |
| T4 | 频率特异性对照：Q1 / Q2 / Q3 / Qp / K2、单频段 Q1b0–2、Q4 / Q4s | stage1 |
| T5 | 信息上界检验 D3：F0 / F1 / F5 / F6 / F3 / F4 | stage2 |
| T6 | 骨干鲁棒性（B/32、L/14） | 换骨干的缓存 |
| T7 | 效率：每会话耗时、内存，相对 BiMC† 的倍数 | `results.json` 中的 seconds |

### 4.4 分析与图

| 编号 | 内容 | 命令 |
|---|---|---|
| F1 | 动机图：频率文本与视觉 grounding 的对比，以及已有失败证据（引用你的 19 条路线汇总） | 已有记录 |
| F2 | 方法框架图 | 手绘 |
| F3 | 掩码诊断：类特异性（类间 Jaccard）、shot 稳定性，与随机掩码、关键词掩码对比 | `diagnose-masks` |
| F4 | 错误分解：novel→base、base→novel、novel↔novel，三个数据集，BiMC† 与本文方法对比 | `results.json` 中的 errors |
| F5 | 原型质量：校准后新类原型与全量类均值的余弦、自最近邻率、最近邻落在基类的比例 | `diagnose-prototypes` |
| F6 | 超参敏感性（q、r、α、κ、σ 梯度），以伪增量曲线为主；测试集曲线只作透明度补充，并注明"未用于选参" | `dev_rows.json` |
| F7 | shot 曲线（1 / 5 / 10） | `final --shot` |
| F8 | 视图互补性：oracle 增益、double-fault、Q 统计量 | `diagnose-views` |

### 4.5 统计报告规范

- 每个数字都是 5 个 seed 的均值±标准差。相对 BiMC† 的配对差附正向 seed 数（如 5/5）。
- 每个 seed 报告 McNemar 精确检验 p 值与 bootstrap 95% 区间。
- 正文只把"配对差 > 0 且 ≥4/5 seed 为正"的结果称为提升。CUB 最终会话 1 张图约等于 0.017pp，0.1pp 以内的差异一律写成"持平"。

## 5. 闸门与时间线

| 周 | 任务 | 闸门（不过就怎么办） |
|---|---|---|
| 1 | 抽三个数据集的缓存与关键词库；跑 parity、diagnose-masks、diagnose-views | **G0**：BiMC† Last 与 engine 差 ≤0.15pp，不过先修缓存。**G1′**：频率掩码类间 Jaccard 明显 <0.5，且 5-shot 稳定性 ≥0.3，不过就转主线 B |
| 1–2 | dev stage0_anchor 与 stage0_fdr（三个数据集） | **A**：A1/A2 在 CUB 伪增量上相对 BiMC† ≥+0.5pp，且 ≥80% episode 为正。**G0′**：K1 − K0 ≥ +0.2pp，或 1-shot 伪 Novel ≥ +1pp，且 K1 > K2；不过就不做 Q 系列 |
| 2 | dev stage1_frequency、stage2_blocks | 主线 A 的触发条件见第 0 节；**G1（D3）**：F1 − F0 ≥ +0.2pp 且高于 F5、F3、F4 |
| 3 | stage3_compose → 锁定 → 三个数据集 × 5 seed 的 final；1-shot、10-shot | **G3**：CUB Last 相对 BiMC† ≥ +1.0pp 且 5/5 seed 为正；CIFAR/mini 不低于 −0.1pp |
| 4 | 分析 F3–F8；换骨干缓存与 T6；效率 | — |
| 5–6 | 写作（结构见第 6 节）；全文数字从 `results.json` 自动生成表格 | — |
| 7 | 内部审稿：查新（HyCal、ULA、SGLC、QR-Prompt、LDC）、补对比、润色、投稿 | — |

**止损规则**：任何新模块在伪增量上反复只有约 0.02pp 的变化，就停止，不再加组件（沿用你 `performance_first_research_plan.md` 里的规则）。

## 6. 写作结构（Elsevier 单栏约 25–30 页）

1. **引言**：问题、两类偏差、现有方法的不足；写主线 A 时还要说明频率文本为什么失败；贡献列表。
2. **相关工作**：FSCIL；基于 CLIP 的 FSCIL（训练类与免训练类）；原型校准与分布校准（TEEN、Free Lunch、GDA、FeCAM、HyCal、ULA）；视觉中的频率与 CLIP 的形状/纹理偏置。
3. **预备知识与问题分析**：BiMC 回顾；用实测数字说明 BiMC† 统计分支的 shot 偏差与总体协方差问题；FDR 拆解实验（K0/K1/K2 与坐标问题）作为动机。
4. **方法**：锚点、分段重组、（A）频率 grounding、推理与复杂度、选参协议。
5. **实验**：设定、主结果、1-shot、消融、频率对照（A 是验证，B 是分析）、错误分解与原型质量、敏感性与 shot、骨干、效率。
6. **讨论与局限**：CIFAR 原生分辨率低、高频信息物理上缺失；关键词质量依赖描述；免训练方法的上限。
7. **结论**。

图表清单见 4.3、4.4。每张表的数字都由 `table.md` 或 `results.json` 生成，不手抄。

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| 频率增益仍在噪声内 | 走主线 B，频率改写成受控负结果分析；锚点 + 分段重组仍能构成完整方法 |
| 锚点被认为与 GDA / HyCal 重合 | 强调三点：FSCIL 中 base 冻结的池化类内协方差、预测分布的 κ/n_c 推导、与 BiMC 总体协方差的对比实验。GDA_only 已在基线里；HyCal 式按样本数门控目前代码未实现，按其公式补一个对照（约半天工作量） |
| FDR 复现差异被质疑 | 同时报告 FDR 原坐标（K1fdr / FDR_tf，用官方脚本超参）与修正坐标；说明官方仓库无法直接运行的具体位置 |
| 选参被质疑偷看测试集 | 锁文件带校验和与时间戳，开发命令物理上读不到测试特征；附录列出全部网格 |
| CIFAR / mini 无提升 | 预先承诺"不低于 −0.1pp"作为成功标准；用错误分解解释数据集差异（CIFAR 65% 的新类错误错到基类，CUB 只有 9%） |

## 8. 代码与实验的对应

| 实验 | 配置 / 命令 |
|---|---|
| parity / G0 | `fgdr_run.py parity` |
| A1、A2、A2c | `configs/stage0_anchor.json` |
| K0v、K0、K1、K1fdr、K2 | `configs/stage0_fdr.json`（需关键词库） |
| Q1–Q4 | `configs/stage1_frequency.json` |
| F0–F6 | `configs/stage2_blocks.json` |
| 组合 | `configs/stage3_compose.json`（填入前几阶段的选择） |
| 论文基线 | `configs/baselines.json`（锁定时用 `--baselines-config` 加入） |
| 诊断 | `diagnose-masks` / `diagnose-views` / `diagnose-prototypes` |
