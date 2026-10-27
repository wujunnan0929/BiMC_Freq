# 多层与多属性原型补全：基类开发实验与分析包

本轮实现 repeat 拆解、ViT 中间层筛查和多属性对照。所有拟合、模型选择和诊断只使用原始基类训练图像；不运行正式 FSCIL benchmark，不使用真实增量类或 benchmark 测试标签。没有新增梯度优化，但包含 PCA、闭式回归、协方差拟合和基类验证选择。

设计背景见 [研究方案](multilayer_multiattribute_completion_plan.md)。当前分支为 `codex/frequency-prototype-completion`。后续查询端是否利用中层表示由独立的简单分数融合对照检查；中间层不默认与文本空间对齐。

## 1. 实现范围

- 支持一次原图前向，提取最终全局 CLIP 表示和 block 3/6/9 的原生 CLS。默认最终输出保持原行为；不将中间层套上最终投影后直接当语言对齐特征。
- 复用缓存完成所有类别划分及对照，不为每个配置重新加载图像或编码 CLIP。
- 四类文本为 shape、part、color、pattern，每类最多两条；ordinary 控制最多八条。默认从已有普通类别描述保守筛选明确可视句子，不调用新的 LLM。
- unknown 使用显式 mask，不复制同一条句子填满预算，不因属性不全删除类别。缓存元数据中的 `attribute_manifest` 保存所用句子、来源和覆盖率。默认已有 CUB 描述的属性覆盖可能不完整；这些规则只是文本整理启发式，不能把它当准确的人工属性标注。
- 不把四份语义先验当四倍视觉样本；新的异维测量模型区分视觉采样噪声和不随 shot 缩小的映射／语义残差。
- 新程序只导出开发结果与诊断，不导出全基类部署模型，也不声称已获得真实测试提升。patch 空间频率和学习门控不在本轮范围。

## 2. 一次提取基类缓存

在已有 BiMC Python/CLIP 环境中执行。先将本地分支实现同步到服务器；切换分支本身不会传输尚未提交或推送的文件。

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
export DATA_ROOT=../datasets/
export GPU=0
export STUDY=outputs/multilayer_completion_20261027
export CACHE="$STUDY/cache/cub200_layers.npz"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python tools/cache_multilayer_base_features.py \
  --dataset cub200 --data-root "$DATA_ROOT" \
  --layers 3 6 9 --gpu "$GPU" --device cuda \
  --batch-size 32 --workers 4 \
  --output "$CACHE" --dry-run
```

确认路径后，执行同一命令并去掉 `--dry-run`：

```bash
python -u tools/cache_multilayer_base_features.py \
  --dataset cub200 --data-root "$DATA_ROOT" \
  --layers 3 6 9 --gpu "$GPU" --device cuda \
  --batch-size 32 --workers 4 \
  --output "$CACHE"
```

输出为 `.npz`、同 stem 的 `.json` 与 `.support.json`。后两个文件认证缓存及原始基类训练样本身份；三件套需保留。不会请求 benchmark 测试 loader。

如需人工检查过的属性描述，可增加 `--attribute-descriptions FILE.json`。文件按类别名组织，组键为 `shape/part/color/pattern`，每组为字符串列表，缺失可为空列表。程序仍记录文本来源、有效 mask 和覆盖率。普通描述与人工整理版本应使用不同缓存和输出根，不能混合后比较。

`--limit-per-class` 仅用于编码烟雾测试，会标记为不完整缓存，正式开发选择拒绝使用。CPU 编码使用 `--device cpu`；统计评估本身可在 CPU 上进行。

## 3. 先运行 repeat 与中层筛查

运行器默认只打印计划，不加载缓存，不导入 PyTorch/NumPy，也不创建结果目录：

```bash
python tools/run_multilayer_completion_experiments.py \
  --cache "$CACHE" --output-root "$STUDY/development" \
  --fit-seeds 2 3 4 --stages repeat layers \
  --device cpu --threads 1 --sequences 20 --dry-run
```

执行基类开发并自动打包第一阶段结果：

```bash
python -u tools/run_multilayer_completion_experiments.py \
  --cache "$CACHE" --output-root "$STUDY/development" \
  --fit-seeds 2 3 4 --stages repeat layers \
  --device cpu --threads 1 --sequences 20 \
  --execute --package "$STUDY/repeat_layers_analysis.zip"
```

这里的 `fit-seeds` 控制原始基类内的类别划分，不是正式 benchmark 的支持种子。每个划分中的对照复用同一支持／查询协议、评分规则和候选 blend。默认网格为 `0/0.25/0.5/1`；选择以平均阶段整体准确率为主，同时满足平均 Base 与 Novel 相对对应参考下降不超过 `0.2pp`。允许选零。

repeat 阶段包含原图 rank 16/32、复制视觉＋相同文本、复制视觉＋第二份属性文本、单视觉＋两份文本，以及双块低 rank 和新测量模型的单观测／去重对照。旧联合先验与新测量模型属于不同参数化；解释特定因素时应使用相同模型族、文本和容量的配对参考。

layers 阶段包含 typed 最终层参考、容量参考、三个单中层候选、简单多层类别分数融合，以及同温度的最终层原生专家融合对照。各中层先保持原生视觉空间，按基类数据拟合低维统计，不假设其已经图文对齐。

先查看 `analysis.md`、`results.csv` 和 `summary.json` 的 `layer_selection`。自动选择只读取基类验证指标；若中层未超过简单参考，不应把被选层当作已验证的新信息，需报告回退或未获得增益。

## 4. 多属性对照

有同一结果根中匹配的中层选择记录后，在相同缓存与划分上继续：

```bash
python -u tools/run_multilayer_completion_experiments.py \
  --cache "$CACHE" --output-root "$STUDY/development" \
  --fit-seeds 2 3 4 --stages attributes \
  --device cpu --threads 1 --sequences 20 --execute
```

最终层／最终层＋选中中层分别比较：普通文本、属性文本统一聚合、相同属性文本按语义分组、相同文本随机分组，并保留容量控制。随机分组与语义分组匹配预声明的维度／rank 上限；未知属性会使实际有效秩、可拟合组数及 H/Q 容量不同，必须同时检查报告中的 actual rank、coverage 和完整属性类别的查询子集指标，不能在存在 unknown 时声称容量严格相等。不得挑最差随机分组当参考。

T0 与属性方案按每个类别可用的文本 slot 数匹配，取普通可视文本与属性文本有效数的较小值；不足八条时不补造文本。T1/T2/T3 使用完全相同的有效属性句子池和 mask，缺失的组在该类别的预测中省略。T2/T3 的比较才能检验语义分组；属性文本优于普通文本，仍可能只是文本内容更好。unknown 覆盖率应同时检查，缺失属性不应被误记为高置信观测。

也可预先指定 `--selected-layer 6`，但这是人工预声明选择，报告会标记 manual；不能根据正式测试排名设置它。manual 与自动选层属于不同输入策略，切换策略或改变人工选定层须使用新输出根；同一输出根的后续命令保持相同 `--selected-layer` 参数。没有匹配的基类层选择或手动声明时，不允许属性阶段偷偷从测试结果选层。

若希望一次执行全部三阶段，使用：

```bash
python -u tools/run_multilayer_completion_experiments.py \
  --cache "$CACHE" --output-root "$STUDY/development" \
  --fit-seeds 2 3 4 --stages repeat layers attributes \
  --device cpu --threads 1 --sequences 20 \
  --execute --package "$STUDY/all_analysis.zip"
```

首版配置数以 dry-run 的 engine 清单为准：每个划分 repeat 9 项、layers 7 项、attributes 10 项，三个划分合计 78 个基类配置评估。这些配置复用一次 CLIP 缓存，不是 78 次正式训练或 benchmark。

## 5. 打包需要分析的结果

补齐属性阶段后，单独导出新的完整分析包：

```bash
python tools/package_multilayer_completion_results.py \
  --root "$STUDY/development" \
  --output "$STUDY/all_analysis.zip"
```

将 `all_analysis.zip` 提供给后续分析即可。它包含：

- 根级 `summary.json`、完整预期配置 `plan.json`、`results.csv`、`analysis.md` 和输入来源记录。中断时尚未启动的配置仍计入计划，并标记 pending；不会只用已经完成的子集冒充完整矩阵。
- 每项配置、运行指纹、状态、完整类别／样本协议、候选校准与阶段指标。
- 小型逐查询预测 NPZ、纠错／破坏、类别竞争、配对比较及属性组诊断；包含实际维度／秩、覆盖率和完整属性类别子集指标。
- 选择完成后的独立教师原型／间隔诊断。每个序列的教师图像与该序列的支持、查询互斥，教师未全局预留，同一图像可能在另一序列出现；这些诊断不参与候选选择，也不是正式性能上界。
- 缓存元数据中的文本来源与覆盖率、CLIP／代码／配置哈希，以及可用的源码快照。
- `bundle_manifest.json`：每个包内文件的相对路径、字节数、SHA256，完整性审计及明确的排除清单。

包中不放 CLIP 权重、checkpoint、完整图像特征缓存、图片、API key 或本地私有配置。不跟随 symlink，不根据结果文件中的绝对路径读取根目录之外的文件。文件名不在分析 allowlist 的内容会被排除。

打包只读取结果，不重新拟合或评估。若缺少配置、协议、指标或预测等必要文件，默认拒绝将其认证为完整结果。需要诊断失败运行时可显式导出部分包：

```bash
python tools/package_multilayer_completion_results.py \
  --root "$STUDY/development" \
  --output "$STUDY/incomplete_analysis.zip" --allow-incomplete
```

部分包的 manifest 会标记 `complete=false` 并列出缺失／错误，不能把它作为完整比较结果。

已有 zip 永不覆盖；第一次只跑 repeat/layers 时用 `repeat_layers_analysis.zip`，最终用不同的 `all_analysis.zip`。实验根默认允许匹配指纹的断点复用：已完成配置直接跳过，同一输入下中断的受管理配置重新计算并替换其已知结果文件；源码、缓存、配置或协议改变后应使用新的结果根。`--no-resume` 不授权覆盖已有文件。

## 6. 首先检查的结果

1. 配对协议、缓存身份和零修正一致性是否正常；所有拟合是否仅来自 A 类且排除了固定 A 查询。
2. repeat 的收益能否由单视觉＋额外文本或更大原图子空间解释。
3. 新中层是否超过最终层、同容量参考和去重表示；简单分数融合是否比最终层补全更有效。
4. 相同属性文本的语义分组是否超过统一聚合及同容量随机分组。
5. 整体准确率、Base/Novel、H、改对与改错是否同时支持继续；不能仅据中心误差下降扩大模型。

这些是原始基类上的开发选择证据。只有通过上述机制比较后，才固定方法进入正式 FSCIL；当前工具不会生成正式 benchmark 的 Final 或宣称投稿性能。
