# 以频率语义作为属性的原型迁移

本实验借鉴 FDR 的“文本定位通道 → 检索基类属性 → 重组新类原型”，将原来的
shape/part/color/pattern 改为 low/middle/high 三组空间尺度语义。CLIP 冻结，无梯度训练；
可选旧/新类别分数偏置只在原始基类的 A 内部任务上拟合。

参考：[FDR, ICCV 2025 原文](https://openaccess.thecvf.com/content/ICCV2025/papers/Xue_Feature_Decomposition-Recomposition_in_Large_Vision-Language_Model_for_Few-Shot_Class-Incremental_Learning_ICCV_2025_paper.pdf)。

这是 FDR-inspired 实验，不是原论文的严格复现。其分组、跨词汇供体检索、重叠通道平均、
原型归一化和残差更新均沿用本项目的受控设计。FDR 本身也不限定属性必须是部件。

## 1. 实际实现了什么

| 属性 | 希望表达的线索 | 词汇例子 |
|---|---|---|
| low | 整体外观、轮廓、大尺度形态 | silhouette, overall, compact, broad |
| middle | 色块配置、区域对比、较粗结构 | patch, collar, band, contrasting |
| high | 细条纹、羽缘、颗粒和细节 | fine streaks, feather edges, speckled |

这里的分组是**空间尺度文本假设**。条纹可以粗细不同，部件边界也有高频分量，
词汇分组不等于严格的 Fourier 频段。当前版本没有对图像滤波，也没有计算 ViT patch 频谱。
不能把输出通道掩码直接命名为“已验证的高频通道”。

实现流程：

1. 读取现有 `description/cub200_frequency_descriptions_deepseek.json`。
   每类每频段按原顺序最多保留 2 个不同句子；频段内或跨频段重复句不重复计数。
   去掉 flight pattern、飞行行为、饮食、迁徙等非静态外观内容，以及词汇方法无法正确处理的否定句。
   缺少该组内容关键词时保持 unknown，不用类名或通用模板填充。
2. 将完整类别名替换为 `subject`，同时用于通道定位和检索文本编码。
   在实际描述中定位内容词，不把 low/high/frequency 这些标签当作关键词。
   这只去除完整名称，不能保证删除所有潜在的类别身份线索。
3. 用冻结 CLIP 最后一层文本 attention/value 的投影贡献，或关键词删除响应，产生通道分数。
   每组取 top-q（默认 50）通道。最后一层 attention 路径不包括全部残差/MLP，属于定位启发式。
4. **仅将实际成功定位的句子编码结果池化为该组文本**，写入新的频率附属缓存。
   该缓存同时包含 `[C,3,D]` 文本、通道分数、布尔掩码及逐句审计。
   后续频率迁移不再读取旧 shape/part/color/pattern 的文本向量。
5. A 拟合类构成供体库。对于新类每一频段，在该组有效的 A 类中按掩码内文本余弦检索 top-k，
   默认 k=5、softmax 温度系数 16；再借用这些基类的视觉原型通道修正新类 support 均值。

新增入口：`tools/cache_frequency_attribute_grounding.py`。
文本筛选：`utils/frequency_attributes.py`。
已有 `models/attribute_transfer.py` 改为显式可变组数，原四组默认行为保留。
已有实验工具通过 `--attribute-family frequency` 选择三频段模式。

## 2. 原型如何更新

设新类 support 均值为 \(r_c\)，归一化为 \(p_c\)。基类单位视觉原型为 \(p_j\)，
该组文本为 \(t_{c,b}\)，新类掩码为 \(m_{c,b}\)。候选供体仅来自 A，且排除同类别自身。

\[
s_{cj}^{b}=\cos(m_{c,b}\odot t_{c,b},m_{c,b}\odot t_{j,b}),\qquad
w_{cj}^{b}=\operatorname{softmax}_{j\in\operatorname{TopK}(s)}(16s_{cj}^{b})
\]

\[
u_{c,b}=\operatorname{norm}\left(\sum_j w_{cj}^{b}p_j\right),\qquad
\delta_{c,b}=g_{c,b}\,m_{c,b}\odot(u_{c,b}-p_c).
\]

按每个通道上有效且非重复的组数平均这些残差，得到 \(\bar\delta_c\)。未覆盖通道不变：

\[
\widehat r_c=r_c+\|r_c\|\bar\delta_c.
\]

普通 semantic_mask 的 \(g=1\)；semantic_reliable 使用既有启发式门控。
unknown、无供体、无有效掩码或退化方向时不修正。
最终接入相同的 BiMC 参考原型，而不是另换分类器：

\[
p_c^{\rm final}=\operatorname{norm}\{p_c^{\rm BiMC}
 +\lambda[\operatorname{norm}(\widehat r_c)-\operatorname{norm}(r_c)]\}.
\]

\(\lambda=0\) 走精确原始参考路径。供体权重来自文本，实际迁移的是视觉原型分量；
因此已经结合文本和视觉，但还没有引入**物理频率观测**。

本版每个新类使用自己的掩码对目标/供体共同取通道；没有复现 FDR 的共享关键词掩码库和逐关键词递推更新。

## 3. 对照与数据协议

默认 8 种方法 × 2 种偏置方式 × 3 个 A/V 划分种子 = **48 组**：

| 方法标识 | 作用 |
|---|---|
| bimc | 精确原参考；a_inner 为仅分数偏置对照 |
| single_two_text | 保留前轮普通文本 + 结构文本池化基线 |
| frequency_pooled | 同一份筛选后频率文本先跨组池化，再接既有单视觉原型补全 |
| whole_transfer | 同一频率文本检索，但迁移整个原型 |
| random_mask | 同一文本和组数，每组随机同数量通道 |
| shuffled_mask | 同一类内循环打乱有效频段掩码，文本组不变 |
| semantic_mask | 频率语义通道分解与残差重组 |
| semantic_reliable | 上述方法加启发式可靠性门控 |

frequency_pooled 对每个有效组等权池化，再与普通文本等权合并；因此它比较的是同源文本的另一种补全方式，
并非与通道重组完全同构的估计器。single_two_text 保留原来的文本，文本预算也不同，不能用二者之差单独归因。
随机/打乱对照也会改变检索子空间和供体，不能将其差异解释成仅掩码位置的因果效果。

基类训练数据内部拆分 A/V，A 提供供体和统计量，V 用于选择 blend。
只用 A fitting rows 再拆 inner-old/inner-new，拟合分数偏置 b；不使用 V 或预留 A query 拟合 b。
默认 b 网格：`-0.005 -0.002 -0.001 -0.0005 0 0.0005 0.001 0.002 0.005`。
每个方法/λ 分别在相同 A 内部协议下选择 b。保留精确 λ=0,b=0 回退候选。
Base/Novel 各下降不超过 0.2 个百分点的约束仍然生效。

这次额外保存 `inner_scores.npz`：包含每个 λ 的 A 内部旧/新组最大分数、类 ID、样本 ID、
stage 边界及标签，可复算 A 内偏置网格。它不包含完整图像特征。
查询重用导致统计相关；当前输出是开发选择结果，不是正式 FSCIL benchmark 或独立测试性能。

## 4. 执行命令（Linux / Bash，项目根目录）

使用上轮完整视觉缓存；若服务器的实际目录不同，只修改 CACHE。

```bash
export CACHE=outputs/multilayer_completion_20261027/cache/cub200_layers.npz
export STUDY=outputs/frequency_attribute_transfer_20261101
export GROUNDING="$STUDY/cache/frequency_attributes.npz"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u tools/cache_frequency_attribute_grounding.py \
  --cache "$CACHE" \
  --descriptions description/cub200_frequency_descriptions_deepseek.json \
  --output "$GROUNDING" \
  --method attention_value --sentences-per-band 2 --topq 50 \
  --device cuda --gpu 0 --batch-size 128

python -u tools/run_attribute_transfer_experiments.py \
  --attribute-family frequency \
  --cache "$CACHE" --grounding "$GROUNDING" \
  --output-root "$STUDY/development" \
  --fit-seeds 2 3 4 --sequences 20 --inner-sequences 5 \
  --blend-grid 0 0.05 0.1 0.25 0.5 1 \
  --offset-grid -0.005 -0.002 -0.001 -0.0005 0 0.0005 0.001 0.002 0.005 \
  --topk 5 --temperature 16 --device cpu --threads 1 \
  --execute --package "$STUDY/frequency_attribute_analysis.zip"
```

缓存阶段只跑文本编码，不需要重新编码图像；实验阶段可在 CPU 上复用全部缓存。
去掉第二条命令的 `--execute` 可查看 48 组计划，不读缓存、不加载 torch。
匹配输入、配置和源码的输出可续跑；改变源码/配置/文本后必须使用新目录。
ZIP 不覆盖已有文件。若只重打包，使用新的 ZIP 名称：

```bash
python tools/package_attribute_transfer_results.py \
  --root "$STUDY/development" \
  --output "$STUDY/frequency_attribute_analysis_v2.zip"
```

关键词删除对照：将第一条命令改为 `--method deletion`，并使用新的 GROUNDING 与 STUDY 路径，
其余实验设置一致。不要在同一个输出目录混用不同缓存。

## 5. 如何分析，以及物理频率如何接入

先看 semantic_mask 是否稳定优于 frequency_pooled、random_mask、shuffled_mask，
且增益不是仅以更多旧类误判换取新类召回。查看全部非零候选，不能只看回退后的汇总均值。
semantic_reliable 可能只是减小修正幅度，未做等范数门控对照前不声称门控机制有效。
文本筛选器仍无法核实 LLM 描述的物种事实；在生成大量组合实验前检查附属缓存 JSON 中的
`frequency_manifest` 和 `grounding_manifest`。

更强的下一步是增加**独立的视觉频率定位证据**，而不是把通道编号做 FFT：

- 输入图像：在相同裁剪上删除/保留指定空间频段，比较冻结 CLIP 输出通道响应。
  需控制背景、裁剪和滤波伪影；相应响应只能在 A 拟合图像上建立全局定位统计。
- ViT：将 patch token 恢复成 H×W 网格，在空间维度做 FFT/DCT。
  不包含 CLS；不同层不直接等同低/中/高频；不能对无空间顺序的 512 个 CLIP 通道做 FFT 来定义频率。
- 将文本通道分数与 A-only 视觉响应相结合，并比较文本单独、视觉单独、联合、随机频段。
  若融合系数需要选择，使用 A 内部验证；V 仅用于既定开发选择，不用其查询反向拟合映射。

这一增强需要额外的图像或 patch-token 缓存，**尚未在本版实现**。当前实现先检验：
FDR 式属性迁移改用空间尺度语义后，是否优于同源文本池化及随机分解。
