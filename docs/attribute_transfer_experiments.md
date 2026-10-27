# 属性通道迁移实验：实现、协议与运行

本轮实现 FDR 启发的属性定位、基类片段迁移和可拒绝门控，按研究计划先完成阶段 A。LGSP 的局部 CNN 提示池、VPT 和频率训练不在本轮主实验中；它们在属性迁移通过后再单独加入。这里不是 FDR 或 LGSP 的严格复现，也没有执行正式 FSCIL benchmark。

当前分支：`codex/frequency-prototype-completion`。新文件和本地修改需先同步到服务器；仅切换分支不会传输尚未提交／推送的内容。

## 1. 本轮改变了什么

之前的 typed 模型把四组文本都当作同一个低维全局视觉中心的观测。本轮明确限定每组属性可以改变哪些 CLIP 视觉通道，并从拟合基类的对应属性中检索供体。支持样本、供体和文本的作用分别记录，unknown 或无供体时保持原型不变。

两个独立输入：

1. 已有 `multilayer_base_features_v1` 视觉缓存及 `.json/.support.json`，包括最终 CLIP 视觉特征和原有属性文本。直接复用，不重编码图像。
2. 新建的 `attribute_grounding_v1` 文本通道侧车及其 `.json/.support.json`。只运行冻结文本编码器，绑定原视觉缓存、文本来源和 CLIP 权重 SHA。

没有新增梯度优化。原型／噪声估计和 A 内部偏置搜索仍属于统计学习／校准。

## 2. 属性通道定位

默认 `attention_value`：从已有 shape/part/color/pattern 属性句子中匹配明确关键词，排除完整类别名，用 CLIP BPE 定位实际 token。读取最后文本 block 的 EOS attention × token value，通过 attention 输出投影、固定最终 LayerNorm 中心／尺度／权重及 text_projection 映射至最终 CLIP 坐标，聚合关键词贡献，为每类每组选择 top-q 通道。

这是**最后注意力路径的投影贡献启发式**：没有将残差和 MLP 路径归因到关键词，不能当作精确完整特征分解或图像区域定位。默认 `q=50`，每组最多 `min(q,D)` 个通道。没有内容关键词或属性文本时，mask 为空并记录原因。

还提供 `deletion` 对照：删除实际属性关键词，计算最终归一化文本特征的变化，用变化幅度选通道。它同样不需要图像或梯度。两种定位应使用不同侧车及实验输出根。

侧车包含 `channel_score[C,4,D]`、`channel_mask[C,4,D]`、`group_available[C,4]`。JSON 记录所有句子、关键词、字符／BPE 区间、拒绝原因和覆盖率。每组文本池与侧车使用全部真实有效属性句子，不用重复文本填预算，也不按 ordinary 的可用条数截断该实验的属性池。

## 3. 属性供体与原型修正

拟合仅使用 A 类中排除固定查询后的图像，保留 A 类均值、单位原型、各类样本数、属性文本和 pooled within 对角噪声，不保留模型训练样本。

给新类 c 的属性 g，按目标 mask 限制后的属性文本余弦相似度，从具有该组属性的 A 类中选 top-k 供体，softmax 温度默认 16，k 默认 5。原型处于同一冻结 CLIP 空间：

\[
r_{c,g}=\operatorname{norm}\left(\sum_b w_{c,g,b}p_b\right),\qquad
\delta_{c,g}=a_{c,g}m_{c,g}\odot(r_{c,g}-\operatorname{norm}(\bar x_c)).
\]

对每个通道，将不同有效属性的残差按命中次数平均，避免重叠属性重复放大；完全相同的属性证据先合并。输出保持 raw 均值尺度：

\[
p^{comp}_c=\bar x_c+\|\bar x_c\|\delta_c.
\]

然后沿用已有 `blend_completion`，把单位方向差以 λ 加到相同的 BiMC 参考原型上。固定原型在类别到达时完成一次，每类只读取自己的支持图像，不读取查询或未来类支持。

`semantic_reliable` 使用保守门控：正的语义吻合 × 支持均值采样噪声，除以支持噪声、供体分歧、供体均值不确定性、迁移残差能量及数值 epsilon 之和。支持越稳定、供体越矛盾或修正越大，门控越小；它是待验证的经验规则，不是最优 Gaussian 后验。

所有模式自动排除与目标同 ID 的供体；对 A 内部伪新类亦如此。unknown、空 mask、无供体、相消的供体组合或退化输出有显式拒绝路径。

## 4. 实验矩阵

| method | 含义 |
|---|---|
| bimc | 相同 BiMC 参考，λ 只取零 |
| single_two_text | 之前的单视觉＋普通文本和 structured 文本锚点聚合，legacy rank16 |
| whole_transfer | 相同属性文本，整维检索和迁移 |
| random_mask | 与语义 mask 每类每组同通道数的固定随机 mask |
| shuffled_mask | 类内可用组之间循环错配 mask，文本和供体组保持原定义 |
| semantic_mask | 属性语义 mask、固定组内强度 |
| semantic_reliable | 属性语义 mask＋可靠性门控 |

每种分别执行 `__none` 和 `__a_inner`，每个划分 14 项，seed 2/3/4 合计 **42 项**。默认 λ 网格为 `0, .05, .1, .25, .5, 1`。λ 是内部候选，不是额外配置数。

whole/random/shuffled 会同时影响文本供体检索的作用坐标与视觉残差范围，因此这是“定位＋检索”的整体机制对照，不能单独声称收益来自其中某一步。shuffled 对仅一个可用组的类别无法错配，诊断明确标记。随机 mask 对全局类别 ID 和组使用固定种子，不随输入顺序或未来支持改变。

此处 single_two_text 是简单参考，不与四组属性匹配描述条数；属性定位归因应主要看 whole/random/shuffled/semantic 使用相同属性文本池的配对比较。

## 5. A 内部偏置拟合与 V 开发选择

外层沿用先前协议：原始基类分成 A 拟合类和 V 伪增量类；A 的固定查询从所有拟合图像中排除。默认 CUB 是 50/50 类、20 条序列，每个伪增量阶段 5 类、每类 5-shot。

`__a_inner` 再把 A 的拟合行分成 inner-old 和 inner-new，在 A 内生成默认 5 条伪增量序列。inner-old 的固定查询也从供体拟合中排除，inner-new 不参与供体拟合。按外层累计类别到达比例生成较小的内部阶段；CUB 默认是 25/25 类、十个交替 2/3-way 阶段。内部拟合不读取 V 图像／文本，也不读取外层保留的 A 查询。

对每种方法和每个 λ，在 A 内部搜索一个新类 log-score 偏置 b：

\[
\hat y=\arg\max\{\max_{old}\log s_j,\;\max_{new}\log s_j+b\}.
\]

默认 b 网格为 `-.1,-.05,-.02,0,.02,.05,.1`。优先选择满足 Base/Novel 各不降超过 0.2pp 的内部候选，按阶段平均 Overall 最大化；若该 λ 无可行内部候选，记录此事实并保留内部最优 b，交给外层保护约束决定是否回退。相同准确率选绝对值更小的 b。

各方法使用相同内部数据、搜索网格和标量参数预算；**b 对每个方法／λ 分别拟合，并非各方法固定为同一个数值**。与 bimc__a_inner、single_two_text__a_inner 配对，检验补全是否超出偏置校准自身的收益。

外层 V 只选择 λ 对应的候选，不重新搜索 b。每个变体额外保留严格 identity `(λ=0,b=0)` 以保证可以精确回退。注意 `(λ=0,b≠0)` 仍可能改变预测；应看 `selected_candidate_id` 和 `selected_offset`，不能只用 λ 是否为零判断方法是否关闭。

## 6. 运行命令（Ubuntu）

在现有 BiMC/CLIP Python 环境及仓库根目录执行。先同步本地代码。

```bash
export CACHE=outputs/multilayer_completion_20261027/cache/cub200_layers.npz
export STUDY=outputs/attribute_transfer_20261027
export GROUNDING="$STUDY/cache/cub200_attribute_channels.npz"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 只做文本编码，复用既有视觉缓存；不需要重新指定图像数据根目录
python -u tools/cache_attribute_grounding.py \
  --cache "$CACHE" --output "$GROUNDING" \
  --method attention_value --topq 50 \
  --device cuda --gpu 0 --batch-size 128

# 完成42项基类开发实验，结束时自动生成完整结果包
python -u tools/run_attribute_transfer_experiments.py \
  --cache "$CACHE" --grounding "$GROUNDING" \
  --output-root "$STUDY/development" \
  --fit-seeds 2 3 4 --sequences 20 --inner-sequences 5 \
  --topk 5 --temperature 16 --device cpu --threads 1 \
  --execute --package "$STUDY/attribute_transfer_analysis.zip"
```

两个工具均支持 `--dry-run`；运行器默认只打印计划，必须传 `--execute` 才执行，dry-run 不加载缓存或 ML 库。侧车可用 `--device cpu`。

小规模链路检查可以先用新输出根执行 `--fit-seeds 2 --sequences 2 --inner-sequences 1`，随后完整实验应使用新的 `development` 根，不能把不同协议的结果混在一起。

若没有旧视觉缓存，先按 [多层实验文档](multilayer_completion_experiments.md) 提取一次。原 `.npz`、`.json`、`.support.json` 三件套必须完整；新 grounding 三件套也需保留。权重、描述来源或缓存身份不匹配会拒绝使用。

可选 deletion 定位对照，用另一缓存和结果根运行相同矩阵：

```bash
python -u tools/cache_attribute_grounding.py \
  --cache "$CACHE" --output "$STUDY/cache/cub200_channels_deletion.npz" \
  --method deletion --topq 50 --device cuda --gpu 0 --batch-size 128

python -u tools/run_attribute_transfer_experiments.py \
  --cache "$CACHE" --grounding "$STUDY/cache/cub200_channels_deletion.npz" \
  --output-root "$STUDY/deletion_development" \
  --fit-seeds 2 3 4 --sequences 20 --inner-sequences 5 \
  --device cpu --threads 1 --execute \
  --package "$STUDY/deletion_analysis.zip"
```

## 7. 打包和分析内容

自动打包后提供 `attribute_transfer_analysis.zip` 即可继续分析。已完成实验也可单独打包：

```bash
python tools/package_attribute_transfer_results.py \
  --root "$STUDY/development" \
  --output "$STUDY/attribute_transfer_analysis.zip"
```

已有 ZIP 永不覆盖；自动打包成功后不要再向同一路径重复打包。中断时重跑相同输入，已完成项跳过，缺少必要诊断的受管理项重新计算。源码、缓存、划分或参数变化后使用新结果根。

包内包括：

- 42 项完整计划、状态、汇总 CSV、配对比较、来源与源码快照。
- 每项的全部候选校准、阶段 Overall/Base/Novel/H，以及忽略另一组竞争时的组内准确率。
- 所有候选的改对／改错、旧→新／新→旧和组内错误。
- 候选的两个组内获胜类别与 float64 log-score、b、λ、预测、样本／序列／阶段身份；可精确重放任意已存候选偏置决策。
- 每类每组的 mask、供体 IDs、权重、相似度、噪声／分歧、门控和拒绝原因。
- **所有 λ 的教师几何诊断**，不再仅保存被选中强度。教师在选择完成后读取，与同一序列支持／查询互斥，但未跨序列全局预留；不是独立正式测试或性能上界。
- A 内部完整协议、偏置搜索记录、grounding 文本／关键词来源与覆盖率，以及每个文件 SHA256。

不放图像、CLIP 权重、完整视觉／grounding NPZ 缓存；不跟随 symlink 或读取结果根外的文件。缺少计划或必要结果时默认拒绝认证为完整包；失败诊断可用新的 ZIP 路径加 `--allow-incomplete`，manifest 会标明不完整。

## 8. 同时修正的 typed 协方差问题

`models/multievidence_completion.py` 升级到 v2：对每个测量块独立建立对角块收缩目标；对跨块协方差仅作统一衰减以保证 PSD，不再用异构块的全局平均方差改变既有最终层／文本噪声块。新增数值测试验证共享块不变、缺失文本、完全重复证据与极端尺度 PSD。

本轮属性迁移不依赖该 Gaussian 模型，因此这项修复不会被算作属性迁移的贡献。旧 78 项结果保持原样；如要重跑修复后的 typed 实验，使用新的输出根与旧多层运行命令。旧源码结果不能断点接到新源码运行上。

这轮的继续条件仍是：语义 mask 超过随机／打乱；可靠性门控能够减少有害迁移；与相同偏置拟合预算的简单参考相比仍有配对收益；正式测试标签不用于任何拟合或选择。
