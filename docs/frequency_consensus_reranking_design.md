# 频率视觉—语义共识重排：一个训练自由的 FSCIL 改进方案

日期：2026-09-07。2026-09-08 更新：已实现首轮验证代码与本地逻辑测试；
尚未运行本方案的完整 benchmark。执行协议见 `frequency_consensus_validation.md`。

## 1. 推荐方案

先用完整原始 BiMC 获得两个最可能的类别，再让低、中、高频的视觉证据与对应
语义证据共同判断这两个类别。只有共同支持的证据才能影响预测。

首版不训练 backbone、字典、router 或新类 code。增量阶段只添加每个新类的三组
视觉原型与语义原型；推理只调一个全局强度 lambda。六个分数尺度是从 base 训练
数据估计并冻结的统计量，不通过梯度优化。

这是一项可检验的改进假设。现有结果不支持承诺一定提升准确率。

## 2. 与现有实现的关系

本地历史 CUB 单 seed 日志中，frequency-off 最终为 72.70%，structured-only
为 72.89%，frequency router 为 73.54%。这些记录提示频率分支值得继续验证；
它们未记录当前统一的 support manifest，不能将差值解释为严格配对收益。
来源分别为 `outputs/cub200_frequency_ablation/frequency_off_seed1.log`、
`outputs/cub200_frequency_ablation/structured_only_seed1.log` 与
`outputs/cub200_router_seed1.log`。

另一项 fixed-v3 辅助分类诊断中，full-image 为 54.9707%，support 预测的类级
hard gate 为 52.5544%；它使用独立的 CLIP margin 分数链，不能与 BiMC 的
72–74% 结果直接比较。该诊断支持检验局部纠错而非按支持集贡献符号硬切频段。
现有能量/贡献离线报告包含不同频段边界，有些竞争类包含未来类别，不能直接用于
FSCIL 在线决策或超参数选择。

当前 `models/frequency.py` 已实现频率视图、视觉原型、语义原型、原型混合与类级
频带权重。当前权重主要依据本类图文对齐程度和视觉紧致度，例如同类图文 cosine
越高，某频带的权重越大。这样的依据并不能直接说明该频带能够区分当前易混的
两个类别。

新方案以 query 与候选类对为单位，分别检查视觉判别和语义判别；不先把它们混成
一个原型。各类采用同一判别规则，不按 base/novel 身份设置额外补偿，不引入随
session 累积的可训练分类偏置。

使用现有 `natural` 视图：low 为低通，middle 为 low+middle，high 为原图加强
高频成分。它们是频率强调视图，并非三个互不重叠的纯频带。形状/结构/纹理描述
与这些视图的对应关系是待验证的语义假设，不是严格的 Fourier 对应关系。

## 3. 原型和语义的构造

对已见类别 c、频带 b，保存归一化视觉原型：

```math
p_c^b = normalize(mean_{x in S_c} normalize(f_b(x))).
```

直接复用 `raw_frequency_mean` 并归一化，避免对同一视觉证据重复做语义混合。
base 使用允许的 base 训练样本，新类使用固定的 5-shot support；完成统计后不保留
逐样本特征。新旧类的原型质量仍可能不同，共识不能完全消除这个差异。

语义原型使用现有离线频率描述，不需要再次生成文本：

```math
t_c^b = normalize(mean_k normalize(g(description_{c,b,k}))).
```

首版等权平均同一频带已有描述。若用视觉原型挑选描述，再把图文意见一致当作
可靠性证据，两路之间的依赖会更强。因此视觉 Top-K grounding 留作消融。
两路本来就共享 CLIP 编码空间，共识不等于统计独立，也不保证正确。

## 4. 推理公式

### 4.1 完整 BiMC 提供候选

令 s 为关闭原有频率融合及低秩残差后的完整 BiMC ensemble 输出。它包含原有
prototype、covariance、description-nearest-neighbor 三部分组合。

输出 s 是正的混合票数，不能把它直接当作 logits 再 softmax。取原始 argmax
为 a，去掉 a 后的 argmax 为 b，固定处理并列的规则。

```math
L_0(x) = log(max(s_a, eps)) - log(max(s_b, eps)) >= 0.
```

对正票数，整体是否归一化不影响该比值。

### 4.2 各频带分别给出视觉和语义的类对证据

归一化 query 的频率特征为 z_b。计算：

```math
d_b^V = z_b^T p_a^b - z_b^T p_b^b,
d_b^T = z_b^T t_a^b - z_b^T t_b^b.
```

正号表示支持 a，负号表示支持 b。这比较的是同一 query 对两类的差异，而非
“该 query 与某个描述是否具有较高绝对相似度”。

视觉 cosine margin 与图文 cosine margin 的分布通常不同。用 base 伪增量校准
episode 中实际 top-2 类对的绝对 margin 中位数，分别估计 s_b^V、s_b^T；每个
统计量设固定数值下限 eps，不引入额外可学习温度。

```math
v_b = tanh(d_b^V / s_b^V),
t_b = tanh(d_b^T / s_b^T).
```

总共 3×2 个尺度。在真实 incremental 阶段保持冻结。若某一尺度持续近零，说明
对应证据可能退化，应停用该路并检查，不能依靠极小分母放大噪声。

### 4.3 共识证据

```math
e_b = sign(v_b) min(|v_b|, |t_b|),  if v_b t_b > 0;
e_b = 0,                         otherwise.
E(x) = (e_low + e_middle + e_high) / 3.
```

两路反对时，该频带不参与改判；一致时取较弱证据，避免单路很强的错误主导结果。
频带没有通过共识时保留它的零贡献，分母仍为 3，不能把仅剩的一个弱频带重新
归一化成全部权重。

首版三频带等权，E 在 [-1,1] 内。不再引入类级可靠性、shot gate 或 router。
按语义类对距离加权可以作为后续消融，但语义距离只是一种可分性启发式，不能
直接解释成正确率或可靠性。

### 4.4 一参数重排

```math
L(x) = L_0(x) + lambda E(x).
prediction = b if L(x) < 0 else a.
```

lambda >= 0 是唯一需要选择的全局超参数。它在 base 验证中确定，之后所有类别和
session 共享。lambda=0 精确回退原始 BiMC。

因为 |E| <= 1，当 L_0 >= lambda 时不可能改判。因此可以先算原图 BiMC，再只给
L_0 < lambda 的 query 编码三种频率视图。相对原图推理，若这部分比例为 rho，
理想的 query 图像编码次数约为 1+3rho；实际速度还取决于小 batch 和 FFT 开销，
support 统计仍需频率编码。

该性质保护当前分数差大于阈值的预测，不能推出高置信预测一定正确，也不能保证
整体零遗忘。随着类别增加，候选对本身仍会变化。

若引擎需要完整分数向量，保持 pair 总票数 M=s_a+s_b，只替换：

```math
s'_a = M sigmoid(L),
s'_b = M (1-sigmoid(L)),
s'_c = s_c,  c not in {a,b}.
```

获胜候选票数至少为 M/2 >= s_b，因此第三类不能因重排进入第一名（并列遵循
固定规则）。不能简单给 a/b 加减 logits 后对所有类别取 argmax，那样可能引入
第三类。lambda=0 或确定不会改判时直接返回原票数，保留精确回退行为。

## 5. 为什么这条路线值得先验证

- 直接复用现有视觉频率特征和文本频率特征，没有新的高维增量优化。
- 同时检验“图像像哪类”和“图像支持哪类描述”，避免把图文空间先混合而无法
  知道哪一路在提供证据。
- 用候选类对 margin 检验区分性，不把频带能量、同类图文相似度或 support
  紧致度等同于有用的分类信息。
- 干预范围明确：仅原始 top-2 且 margin 足够小的样本有机会改判。
- 全局强度固定，没有每 session 新增可训练 code 造成的累计偏置；但原型误差、
  错误文本和跨频带相关性仍然可能导致错误共识。

训练自由地利用视觉 support 与 CLIP 语义的基本路线已有 BiMC 和 Tip-Adapter
等工作支持；此处的“候选类对共识重排”是本项目提出的设计，相关论文不构成
该设计有效性的实验证据。

## 6. 最小实验方案

### 6.1 先判断有没有可纠错空间

在 base 验证中，先报告 BiMC 的 top-1 accuracy A1、top-2 recall R2。
本方案预测始终属于原始 top-2，因此准确率上限不超过 R2，理论最大改善不超过
R2-A1。这个上限只用于诊断，不能把真实标签用于推理或选择候选。

若剩余错误的正确类别大多不在 top-2，先停止这条重排路线的扩展，不堆叠更多
门控。top-3/5 是单独的后续设计，需重新验证其改判范围和尺度。

### 6.2 Base 校准与选择

只使用 base 训练划分。先将用于估计六个尺度的校准数据与用于选择 lambda 的
验证数据分开，优先采用分 base 类的独立划分。所有 episode 的 support/query
严格不重叠，query 不得参与视觉原型或文本选择。

伪增量同时包含 many-shot old 与 5-shot new，并检验多个连续追加阶段。尽可能
覆盖较多候选类；只在 10-way 上验证可能不能代表真实 100–200 类的易混对分布。
若留出类数不足，应明确报告这种覆盖限制，不能用真实增量测试结果弥补。

lambda 候选起点：

```text
0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1
```

现有 BiMC cosine-softmax 票数可能比较平坦，不能直接沿用失败低秩方案的 0.2。
按 base 验证的全部已见类准确率选择，平局取更小 lambda，同时报告 old/new 代价。
若 lambda=0 最好，应保留基线并报告没有找到正向证据。

### 6.3 配对对照

| 变体 | 用途 |
|---|---|
| BiMC | 完整原始参考 |
| 现有 Frequency V2 | 检查新重排相对现有融合的增量价值 |
| 现有 frequency router | 保留历史结果较强的频率方法作为对照 |
| Visual-only rerank | 检查仅频率视觉原型的作用 |
| Semantic-only rerank | 检查频率描述单独作用 |
| Visual+semantic 平均 margin | 检查普通平均是否已经足够 |
| Visual+semantic 共识 margin | 推荐首版 |

所有方案共享支持集、CLIP、描述和评估划分；若各重排变体分别选 lambda，应给
相同 base 验证预算。同时保存同一 lambda 下的消融，区分机制与调参的贡献。

两项必要的后续机制检验：

1. 固定打乱语义频带对应关系，视觉不变。若匹配与打乱效果一样，就缺少“频率
   语义匹配”本身有作用的证据。
2. 三个视觉频带全替换为原图。若结果相同，就可能主要是额外文本集成收益。

### 6.4 指标

正式评估沿用三个配对 seed，记录 Final、Average、Base、Novel、H、Forgetting。
额外记录每个 session 的 top-2 recall、触发频率编码比例、实际改判比例，以及：

- C：原来错误、重排后正确；
- D：原来正确、重排后错误；
- W：两次都错误但预测类别改变。

同一个测试集合上准确率变化严格等于 100(C-D)/N。判断改进看 C 是否持续大于 D；
若只报改判数量或共识率，无法说明有性能收益。分 base/历史增量/当前新增三组
报告上述计数，避免总准确率掩盖旧类代价。

## 7. 预期实现范围

可在 `models/frequency.py` 增加纯张量的 pairwise consensus/reweight 函数，在
`models/bimc.py` 的完整 BiMC ensemble 后增加入口。现有状态已包含
`raw_frequency_mean`；纯文本频带原型从已有 candidates 聚合后保留即可。

需要让“构造频率特征”与“旧频率融合启用”成为可分离设置。首版原始 BiMC 参考
不得已混入 Frequency V2/router 后再次加入同一频率证据。正式代码实现应验证
零强度精确一致、margin 上界、候选集合约束、并列稳定性、support/query 隔离与
六个尺度在 incremental 阶段冻结。

2026-09-08 已实现独立核心 `models/frequency_consensus.py`、base 校准、Runner
接入、关键消融与 Ubuntu 实验入口；完整性能实验仍由远程服务器运行。

## 参考

- BiMC, CVPR 2025：https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Enhancing_Few-Shot_Class-Incremental_Learning_via_Training-Free_Bi-Level_Modality_Calibration_CVPR_2025_paper.html
- Tip-Adapter, ECCV 2022：https://arxiv.org/abs/2207.09519
