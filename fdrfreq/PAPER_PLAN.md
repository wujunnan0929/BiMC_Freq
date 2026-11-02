# 论文研究与实验计划

目标期刊：Neurocomputing（Elsevier）或同档期刊。这里按 Neurocomputing 理解你说的"neuralcomputing"；MIT 的 Neural Computation 是另一本刊，定位和分区都不同。第三方统计站点 journalmetrics.org 列出 Neurocomputing 的 2025 IF 为 6.5、JCR Q1、中科院 2 区。投稿前请以学校认可的分区表为准。

本文件和代码同在 `fdrfreq/` 下：每个实验对应 `configs/paper.json` 中的一个 entry，每张表对应 `python -m fdrfreq.report` 的一种输出。

---

## 0. 结论先行

1. **能写成一篇投稿水平的论文，但主贡献大概率不是"频率"。** 证据最强、最可能复现的增益来自三处：
   - **协方差估计。** 把 BiMC 的会话内总体协方差（含类间散布）换成只在 base 上估计的池化类内协方差。你本地 CUB 上原图 GDA 相对 BiMC 已有 +1.99pp，3 个 seed 全为正。
   - **shot 数修正。** 新类只有 5 个样本，它们的 Mahalanobis 打分带有可解析的 plug-in 偏差，需要修正。
   - **FDR 式重组。** 在 1-shot 下增益最明显。
2. **频率要不要进标题，由开发集闸门决定。** 判断只依据 base 伪增量（第 6 节），不看测试集：
   - 闸门通过，走叙事 A，频率作为视觉分解轴写进方法。
   - 闸门不通过，走叙事 B，频率作为严格对照的分析章节（"频率到底有没有用"），本身也是贡献。
3. **两种叙事共用同一套实验。** 区别只在写法，所以先把所有 dev 实验跑完，再决定论文形态，最后只对锁定配置跑一次测试集。

---

## 1. 期刊定位与审稿预期

Neurocomputing 接收方法类 FSCIL 论文。同类稿件通常要满足以下几点：

| 维度 | 期望 | 本文对应 |
|---|---|---|
| 问题 | 指出现有免训练 CLIP-FSCIL 的具体缺陷 | BiMC 的总体协方差、shot 偏差、新类原型偏差 |
| 方法 | 有原理、可推导，不只是堆模块 | 后验预测 / 去偏推导；FDR 结构的闭式重组 |
| 对比 | 近两年 SOTA | BiMC（CVPR 2025）、FDR（ICCV 2025）、ULA（CVPR 2026 Findings，免训练 logit 调整）、TEEN、FeCAM、HyCal（2026）；训练类方法单列 |
| 数据 | 三个标准基准 + 至少一个扩展设定 | CIFAR-100、miniImageNet、CUB-200；CUB 1-shot；ViT-L/14 泛化 |
| 统计 | 多 seed、显著性 | 5 个 support seed 的 mean±std、配对 bootstrap CI、McNemar |
| 分析 | 为什么有效 | 错误类型分解、敏感性曲线、频率诊断、计算开销 |
| 可复现 | 代码与协议 | 只在 base 上选参、锁定配置、缓存特征逐位复现 |

同档候选期刊有 Knowledge-Based Systems、Neural Networks、Expert Systems with Applications 等，门槛更高的有 Pattern Recognition、IEEE TCSVT 和 TMM。这些刊的分区请自行核对。

---

## 2. 已有证据（决定了论文怎么写）

| 事实 | 来源 |
|---|---|
| 本地 BiMC（CUB Final）为 72.47±0.39 ~ 72.62±0.40；原图 GDA 为 74.49，配对 +1.99pp，3/3 seed 为正 | 你的 `results_20261006.csv` |
| 已有频率分支相对同结构原图版的增量都在 −0.77 ~ +0.02pp | 第一版报告 |
| BiMC 协方差用 `torch.cov(images_features.T)`，即会话内总体协方差 | 官方代码；你的 `models/bimc.py:992` |
| FDR 免训练 5-shot 相对 TEEN：Aavg +0.4 / +0.4 / +0.1（CUB / mini / CIFAR）；CUB 1-shot 最后一个 session 的 NAcc +6.84 | FDR 正文 Table 1–3 |
| FDR 的随机通道分段（54.83 / 32.08）与整体校准（54.48 / 31.56）几乎相同 | FDR Table 4 |
| FDR 代码中的关键词掩码在 value 空间选通道，却用在联合嵌入空间；代码里另有一步论文未写的文本权重整体校准 | FDR 官方仓库 `models/Network.py` |
| BiMC mini 上：Intra-C 在 Inter-C 之上只加 0.28pp Avg；CUB β=0.3，视觉权重 0.7 | BiMC 正文 Table 2 |
| CUB 新类错误中错到基类的仅约 9%（CIFAR 约 65%） | 你的 `performance_first_research_plan.md` |

据此，主要战场在 CUB 和 1-shot，主要机制是统计校准。频率只有在视觉侧测量通过闸门时才进入方法。

---

## 3. 核心问题与两种叙事

**问题陈述（两种叙事共用）。** 在免训练 CLIP-FSCIL 中，基类用几十到几百张图估计，新类只有 K 张，两者在同一个分类器里竞争，由此产生三种偏差：

- **(B1) 协方差错配。** BiMC 用会话内总体协方差，混入了类间散布；增量会话样本太少，协方差估计接近奇异。
- **(B2) 与 shot 数相关的打分偏差。** 用 K 张样本的均值做 plug-in 估计，Mahalanobis 距离的期望会被抬高 tr(PW)/K。新类和基类的 K 相差一到两个数量级，这个偏差是系统性的。
- **(B3) 新类原型偏差。** 少样本均值偏离真实类中心。基类知识可以修正它，但"从哪些维度借"决定了修正是帮忙还是造成新旧类缠绕（FDR 的论点）。

**叙事 A（频率闸门通过）。**
标题方向：*Frequency-Grounded Decomposition–Recomposition with Shot-Aware Discriminants for Training-Free Few-Shot Class-Incremental Learning*。
核心论点：频率作为视觉侧的分解轴，在类特异、shot 稳定的通道子集上做 FDR 式重组。它不依赖 CLIP 对频率词的理解（这一点有 WaffleCLIP 类证据与你的实验支持），并与 shot-aware GDA 互补。

**叙事 B（频率闸门不通过）。**
标题方向：*Shot-Aware Statistical Calibration and Attribute Recomposition for Training-Free CLIP-based Few-Shot Class-Incremental Learning*。
核心论点：(B1)(B2)(B3) 三种偏差都有闭式解；方法统一在"分解–重组 + 后验预测判别"框架下。另设一节 "Does frequency help?"，用 F/Q 系列严格对照给出负结论，并解释原因：CLS 特征已吸收滤波视图的信息；文本端无法 grounding 频率；CUB 的主要错误发生在新类之间。

---

## 4. 方法（统一公式）

**记号。** f(x) 为冻结 CLIP 的单位图像特征；t_c 为模板集成文本；d_c 为描述原型；𝒴₀ 为基类；n_c 为类 c 的样本数。

### 4.1 参考分类器（BiMC / BiMC†）
- 文本：t̃_c = (1−λ_T) t_c + λ_T d_c。
- 视觉：v_c 为单位化的类均值。新类做整体收缩：v_c ← n((1−λ) v_c + λ·n(Σ_b w_b v_b))。
- 融合：μ_c = n(β t̃_c + (1−β) v_c)，p_ref = softmax(f·μ)。BiMC† 再加上基类协方差项和新类描述近邻项。

### 4.2 Shot-aware GDA 锚点（对应 B1、B2）
- **协方差。** W 为 base 训练图的池化类内协方差，用 Ledoit–Wolf 解析式收缩到 (tr W/d) I，得到 P = W_δ⁻¹。增量会话**不再更新** P，新类只追加均值，每个新类开销为 O(d)。
- **去偏（debias）。** 由 E‖x−μ̂_c‖²_P = ‖x−μ_c‖²_P + tr(PW)/n_c，得到
  s_c = μ̂_cᵀPx − ½μ̂_cᵀPμ̂_c + κ·tr(PW)/(2n_c)。
- **后验预测（predictive，更有原理，建议作为正文主形式）。** 均值取平坦先验时，预测分布为 N(μ̂_c, (1+1/n_c)Σ)，于是
  s_c = −½‖x−μ̂_c‖²_P/(1+κ/n_c) − (d/2)·log(1+κ/n_c)。
  κ=1 即标准贝叶斯预测，κ=0 退化为 LDA。
- **融合。** p = (1−ω)·p_ref + ω·softmax(s/(dT))。ω、T、κ 只在 base 伪增量上选。
- **可写进论文的分析点。** 可以给出 κ 修正后基类 / 新类误判率的变化方向。另外，"冻结 base 协方差 + 只追加均值"等价于在 base 空间做白化后的最近类均值，这能解释它为什么不遗忘。

### 4.3 分解–重组（对应 B3，FDR 结构）
- **整体步。** 供体权重改为文本相似度：w = softmax(τ cos(t̃_n, t̃_b))。这是 FDR 代码里实际执行、论文里没写的一步，K0 单独检验它。
- **分段步。** 对每个分段掩码 g，在掩码维上取文本相似度 top-r 的基类，按 FDR 式 (8)(9) 依次更新 p_n[g]。
- **关键词分段。** 关键词由 BiMC 描述自动抽取（不需要 LLM），只由当前协议的 base 类定义词库。grounding 在**最终 CLIP 坐标**中计算，修正了 FDR 代码的坐标不一致。K1_fdrcoord 对照原始坐标。

### 4.4 频率作为视觉分解轴（叙事 A 的核心；叙事 B 中作为对照）
- **视觉频率 grounding。** 用 Gaussian 梯度 σ₀=0<σ₁<σ₂<σ₃ 定义 Δ_k(x) = n(f(G_{σ_{k−1}}x)) − n(f(G_{σ_k}x))。类 c 在频段 k 的掩码为 top-q(|mean Δ_k − 基类公共方向|)。新类用自己的 shot 计算掩码，所有视图都是自然模糊图，不会出现 OOD 的高通图。
- **频带信息块（F 系列）。** 取 ViT 中间层 14×14 patch-token 网格的 DCT 径向频带 RMS 能量，依次做 PCA、对 CLS 岭回归残差化、尺度对齐，再进入联合 GDA。R1 再加频带经验贝叶斯重组。

### 4.5 算法与复杂度
- **零训练。** 全流程不做任何梯度更新。
- **base 阶段。** 一次 d×d 求逆。
- **每个新类。** O(d·K) 计算均值，O(|𝒴₀|·d) 做重组。
- **测试。** 每个查询的计算量与 BiMC† 同量级。频率视图只在 support 上需要（Q 系列）；F 系列在查询上需要一次额外的中间层读取（同一次前向）。

---

## 5. 贡献点草案

**叙事 B**
1. 指出免训练 CLIP-FSCIL 中协方差错配与 shot 相关打分偏差两个被忽视的问题，并给出闭式的 shot-aware 判别器：base 冻结的类内协方差加后验预测修正，附带理论分析。
2. 在统一的分解–重组框架下重新审视 FDR，分离出文本供体权重与关键词分段两个因素，修正 grounding 坐标，并给出它们在 5-shot 与 1-shot 下的作用边界。
3. 在三个基准、5-shot 与 1-shot、两种骨干上系统验证，并用严格对照给出"频率作为属性"的负结论及成因分析。

**叙事 A**：第 3 条换成"提出视觉频率 grounding 的分解–重组，在类特异、shot 稳定的通道子集上迁移基类知识，并以随机 / 换类 / 增强对照证明增益来自频率排序本身"。

---

## 6. 实验设计

### 6.1 设定
- **数据集。** CIFAR-100（60 + 8×5，5-shot）、miniImageNet（60 + 8×5）、CUB-200（100 + 10×10），以及 CUB 1-shot（FDR 的协议）。
- **support。** 5 个 seed（1–5），与 engine 的采样器逐位一致，抽取时会自动核对。如能整理出 CEC/TOPIC 的标准划分 manifest，用 `--standard-manifest` 存为 seed 0 并单独报告，便于和文献数字直接对比。
- **骨干。** 主实验用 ViT-B/16；泛化实验用 ViT-L/14（可选 ViT-B/32）。
- **可选扩展。** ImageNet 子集或另一个细粒度数据集（如 FGVC-Aircraft），需要新写数据加载器，不作为首投的必需项。

### 6.2 指标与统计
- **基本指标。** 逐 session 精度、Avg、Last、PD，以及最后一个 session 的 Base / Novel / HM。CUB 另报 FDR 式 NAcc。
- **统计。** 5 个 seed 报 mean±std。与 anchor 和 BiMC† 做配对比较：bootstrap 95% CI（seed 内对图像重采样）、为正的 seed 数、fixed / broken 计数、每个 seed 的 McNemar 检验。
- **结论门槛。** 一个提升只有满足 CI 不含 0 且至少 4/5 个 seed 为正，才在正文中称为"提升"。

### 6.3 对比方法
- **免训练，在同一缓存上复现。** ZS、Proto、TEEN、BiMC、BiMC†、FDR-TF（发表超参）、LA（常数 logit 调整）。
- **需要补实现。**
  - ULA（CVPR 2026 Findings）：原文 PDF 这次没能读到，请下载后按原文实现；它是 logit 层面的插件，可以叠加在 BiMC† 和本方法之上。
  - HyCal（2026）：按 Mahalanobis–cosine 随样本数加权的形式实现，作为 shot-aware 的另一种做法对照。
  - FeCAM：数字引用 BiMC 论文。
- **训练类方法（单独分组，只引用原文数字）。** CEC、FACT、CPE-CLIP、PriViLege、LP-DiF、FDR（LoRA）、SGLC（CVPR 2026）等，在表中标明 Training-free = No。

### 6.4 表格与图

| 编号 | 内容 | entry |
|---|---|---|
| T1 | 三基准主表（Avg / Last / PD），与免训练及训练类 SOTA 对比 | 全部基线 + Ours |
| T2 | CUB 5-shot 逐 session（含 NAcc） | 同上 |
| T3 | CUB 1-shot 逐 session（含 NAcc），对齐 FDR Table 2 | 同上，`--shot 1` |
| T4 | 组件消融：BiMC† → A1 → A2 → K0 → K1 →（Q1 / F1），三数据集 Last / Base / Novel / HM | A*, K*, Q1, F1 |
| T5 | 分解轴消融（对应 FDR Table 4）：无 / 随机 / 关键词（FDR 坐标）/ 关键词（最终坐标）/ 频率 / 换类频率 / 随机频率。两组数：选参后的（K*、Q*，强度可能被选为 0）和固定 FDR 强度的（AX_*） | K0, K2, K1_fdrcoord, K1, Q1, Q2, Q3；AX_* |
| T6 | "频率到底有没有用"：F1–F8、R1 及其对照，加上 D3 开发集上界 | F*, R1 + diagnostics |
| T7 | shot 修正：debias vs predictive、κ 取值；把增量 shot 换成 1 / 5 / 10 时偏差的变化 | A2 的 select 报告 + `--shot` |
| T8 | 骨干泛化：ViT-B/16 vs ViT-L/14 | 换 `BACKBONE` / `TAG` 重跑 |
| T9 | 计算开销：每个 session 的时间、显存、可训练参数为 0 | evaluate 的 seconds |

图：
- **F1** 框架图。
- **F2** 逐 session 错误类型分解（新→旧、新→新），BiMC† vs Ours。
- **F3** 敏感性曲线（ω、T、κ、α、q、r），直接取自 `select_*.json` 的候选表。这些全是开发集数据，不涉及测试集。
- **F4** 频率掩码诊断：各频段的类间 Jaccard（对照随机值与关键词库）、shot 稳定性、D2 互补性。
- **F5** 新类准确率增益与基类相似度 / 关键词数的关系。
- **F6（可选）** 可靠性图（ECE）。

### 6.5 协议红线（写进论文 Implementation Details）
- 所有超参只在 base 伪增量 episode 上选：A/V 对半划分，3 个划分 seed × 10 个序列；support 与 query 不相交。
- 选参约束：base / novel 精度都不低于 fallback 0.2pp 以上。
- 测试集只评测锁定配置，`evaluate` 会校验 hash 和 bank。
- 最终方法由哪些组件构成，在跑测试集之前依据 `gates.txt` 决定；决定后不得再根据测试结果增删组件。
- BiMC 的 β 在原文中同样是在 base 内部的验证划分上选出的，可以引用这一点说明本文协议与之一致。

### 6.6 计算预算
- **特征抽取。** 每个数据集一次。CUB 约 1.2 万张图 × 6 个视图，加 3 层 DCT，单卡约几分钟到十几分钟。mini 和 CIFAR 视图更多、图更多，DCT 默认只取第 9 层。
- **选参。** 整个矩阵在 CPU 上每个数据集、每个 shot 约 1 小时以内（按 2 核粗估，你的机器应该更快）。
- **评测。** 每个 entry × 5 个 seed 约几分钟。

---

## 7. 闸门与决策树（只看开发集）

`python -m fdrfreq.run gates --dir outputs/fdrfreq/<ds>_<tag>/shot5` 会自动给出下表的判定。

| 闸门 | 判据 | 通过 | 不通过 |
|---|---|---|---|
| G0 锚点 | A2 − BiMC† ≥ +0.5pp（dev） | 锚点进入方法 | 先查 parity，排查流程 |
| G0′ FDR 阳性对照 | K1 − K0 ≥ +0.2pp 且 K1 − K2 ≥ +0.1pp（5-shot），或 1-shot Novel ≥ +1pp | 分段重组进入方法 | 只保留 K0（若 K0 > A2）；Q 系列只作对照 |
| G1′ 频率掩码 | 各频段类间 Jaccard < 0.5，且 shot 稳定性 ≥ 0.3 | 可以做 Q1 | 频率分段在结构上等同随机分段，停止 |
| G1 信息上界 | D3：联合 GDA − CLS GDA ≥ +0.2pp，≥2/3 个划分为正，且比所有对照高 ≥0.1pp | 可以做 F1 / R1 | 停止 F 系列 |
| G2 方法 | Q1 / F1 相对 fallback ≥ +0.3pp，且相对匹配对照 ≥ +0.2pp | 叙事 A | 叙事 B |

最终 "Ours" 的组成：A2 + 所有通过闸门的组件。CIFAR / mini 上不得低于 BiMC† 0.1pp 以上；若频率组件只在 CUB 通过，论文中说明它是细粒度特有的。

---

## 8. 时间表（约 6 周）

| 周 | 任务 | 产出 |
|---|---|---|
| 1 | 跑通 extract（先用 `--limit-per-class 3` 冒烟）；三数据集建 bank；parity 与 engine 对齐（误差 < 0.05pp） | bank、parity_*.json |
| 1–2 | 跑 dev.sh（5-shot 与 1-shot）；读 gates；补实现 ULA、HyCal 两个基线 | gates.txt、选参报告 |
| 2 | **决定论文叙事与 Ours 组成（冻结）** | 一页决策记录 |
| 3 | 跑 final.sh；ViT-L/14 重跑；计时 | T1–T9 原始表 |
| 3–4 | 画图（F2–F5）；错误分析；挑定性样例 | 图 |
| 4–5 | 写作（见第 9 节）；把旧稿中可复用的部分迁移过来 | 初稿 |
| 6 | 自查：数字全部来自 eval_*.json；复现脚本；补充材料 | 投稿包 |

---

## 9. 写作提纲（Neurocomputing 常见篇幅，双栏约 10–14 页）

1. Introduction：三种偏差，配一张"新类被系统性打压 / 抬高"的示意图；列出贡献。
2. Related work：FSCIL（训练类 / 免训练）、CLIP 免训练适配（Tip-Adapter、GDA、ProKeR）、原型校准（TEEN、FDR）、频率视角（只在叙事 A 中展开）。
3. Preliminaries：协议，BiMC 回顾。
4. Method：4.1–4.4，配算法框和复杂度。
5. Experiments：设定、主表、消融、频率研究、敏感性、骨干、开销。
6. Discussion / Limitations：频率的适用边界；对 base 类别分布的依赖；LLM 描述质量的影响。
7. Conclusion。

Elsevier 的投稿常规材料：Highlights（3–5 条，每条不超过 85 个字符）、CRediT 作者贡献、Declaration of competing interest、Data availability（代码链接）。具体要求以投稿时的 Guide for Authors 为准。

---

## 10. 与已有两份草稿的关系

- **FSB-GDA 稿。** 结果表是空的，三组件方法从未真正跑过。可以复用 GDA 部分的推导与写法，方法章节以本计划为准重写。
- **SS-SAFD 稿。** CIFAR / mini 表中有在证据文件里找不到出处的数字（第一版报告已指出），一律作废，以本框架的 eval 输出为准。其中 "+0.017pp 频率增量" 的诚实分析可以并入叙事 B 的频率章节。
- **docs/ 下的诊断文档。** CUB 错误类型、15-shot 教师实验等可以作为分析素材，但数字要在新框架下重算，保证同一协议、同一 bank。

---

## 11. 马上能做的事（按顺序）

1. **建分支，跑冒烟。** 运行 `git switch -c research/fdr-freq-calib`，再运行 `python -m pytest -q tests/test_fdrfreq.py` 和 `python -m fdrfreq.run smoke`。
2. **建冒烟 bank。** `python -m fdrfreq.extract --dataset cub200 --data-root ../datasets --out banks/cub_smoke --limit-per-class 3 --keywords --seeds 1`
3. **建正式 bank，对 parity。** 正式 bank 建好后跑 `python -m fdrfreq.run parity --bank banks/cub200_vitb16 --seeds 1`，与 engine 的 fp32、`USING_ENSEMBLE=True` 结果逐 session 对比。
4. **跑开发阶段。** `bash fdrfreq/scripts/dev.sh cub200 ../datasets 0`，看 `shot5/gates.txt` 与 `shot1/gates.txt`。
5. **决策。** 按第 7 节冻结论文叙事和 Ours 的组成，然后再跑 final.sh。


---

## 12. 决策记录（2026-11-03，仅依据开发集与基线保真度检查）

### 12.1 协议
- **伪增量划分改为 contiguous。** 随机划分会把 CUB 的近缘鸟种拆到两边，使伪增量的 base / novel 偏差变成测试集的 3–5 倍；5-shot 时连方向都反了。contiguous 划分在 7 个固定参考配置上和测试集方向一致，Last 的增量也接近：GDA ω=0.05 为 +1.32 vs +1.00，K0 为 −0.53 vs −0.54。
- **1-shot 的 K0 同样一致**：开发集 +1.63，测试集 +1.62。
- **选参目标定为伪增量最后一个 session 的 HM，不设 base 容差。** 这条规则在看到候选结果之前确定。保真度检查只用固定的基线配置，论文 Implementation Details 中如实写明。

### 12.2 开发集结论（CUB）

| 结论 | 证据 |
|---|---|
| 5-shot：GDA 锚点有效 | A2 − BiMC† = +1.49 |
| 5-shot：频率分段无显著作用 | Q1 − K0 = +0.26，未胜过 Q2 |
| 1-shot：频率 grounding 分段有效 | Q1 − K0 = +1.93，− Q3（随机掩码）= +0.79，− Q2（借别类掩码）= +0.22 |
| 1-shot：频率分段在 base–novel 权衡曲线上占优 | 同等 base 代价下，比随机掩码高 1–2pp novel，比关键词分段高 2–3.5pp novel |
| 机制：1-shot 原型误差集中在对模糊敏感的通道 | 本类频率掩码覆盖 25.4% 的通道，捕获 41.8% 的误差（×1.65）；随机掩码 ×0.99；FDR 式关键词掩码 ×0.88，低于随机 |
| 5-shot 误差仍集中（×1.46），但总量小，所以重组收益小 | 同上 |
| 频率掩码的作用一大半来自"对模糊敏感的通道"这一共性，类特异只贡献一小部分 | 借别类掩码 ×1.32；本类掩码只比借别类的 Q2 高 0.22pp |
| 频率作为额外特征块：负结果 | D3 全部为负，−1.5 ~ −2.7pp |
| 频率作为文本属性：负结果 | 第一版报告 |
| 频率掩码在 shot 之间不稳定 | 1-shot 稳定性 0.12 |
| shot 修正 κ：两个 shot 下均被选为 0 | 记为负结果 |

### 12.3 冻结的方法与论文叙事（叙事 A 的按 shot 版本）
- **Ours = BiMC† + shot-aware GDA 锚点 + 文本供体整体收缩 + 视觉频率 grounding 分段重组。** 各组件强度按 shot 在开发集上选出：5-shot 时锚点开启、分段很弱；1-shot 时锚点被选为关闭、分段开启。对应锁定文件 `shot{1,5}_hm/locked_Q1.json`。
- **核心论点。** 少样本原型的误差集中在随细节变化大的通道上。在这些通道上向基类借信息，优于随机通道和语义关键词通道。频率只作为视觉侧的测量进入方法，不依赖文本，也不作为额外特征。
- **还需补的对照**：Q5_common（基类平均的类无关频率掩码，测"类特异"有多少贡献）和 AX_freq_common。

### 12.4 剩余步骤
1. **CUB 补跑新对照（只用开发集）。**
   ```
   select --entries Q5_common AX_freq_common --split-mode contiguous --objective hm --tolerance 100
   ```
   再跑 `frontier`，把 freq_common 也加进比较。
2. **CIFAR-100 / miniImageNet：建库、parity、probe。** 按同一保真度规则选择划分方式，再按 HM 选参。
3. **补基线**：ULA（按原文实现），HyCal（可选）。
4. **跑测试集，只跑一次**：final 阶段评测下列锁定配置，5 个 seed。
   - 基线：ZS、Proto、TEEN、FDR_tf_published、BiMC、BiMC_ens、LA
   - 本文方法与对照：A1、A2、K0、K1、K1_fdrcoord、K2_random、Q1、Q2_classperm、Q3_random、Q5_common
   - 分解轴研究：AX_*
   - 频率特征块：F1、F3_aug、F4_noise、F5_randband
5. **ViT-L/14 泛化**，以及把 errconc 和 frontier 画成图。
