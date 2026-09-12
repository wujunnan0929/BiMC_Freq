# 频率共识验证实验：实现与 Ubuntu 执行

> 2026-09-13 更新：完整 core 结果后的校准消融、固定强度对照及新执行命令见
> [后续实验说明](frequency_consensus_followup.md)。原默认校准规则保持不变；代码更新后
> 不应在旧输出目录直接续跑，应使用新目录保留实验版本。

日期：2026-09-08。分支：`codex/frequency-consensus-validation`。
本轮实现可行性与机制对照，不代表已经取得 benchmark 性能提升。

## 本轮要回答的问题

1. 完整原 BiMC 的 top-2 中，是否存在足够的可纠错空间？
2. 视觉—语义共识是否优于单路、普通平均和已有 Frequency V2/router？
3. 正确频带对应是否优于错配；收益是否只是额外文本或普通多视图集成？
4. base 校准的规则在新类加入后是否仍有效，是否以损害旧类为代价？

主方法冻结 CLIP、保留完整 BiMC 候选分数，仅追加每类三个原始视觉均值和三个
纯文本原型。显式频带描述等权平均，不进行视觉 Top-K 选择。新类不反向传播。
共识只重排原始 top-2；尺度来自 base，唯一选择的决策强度为 lambda。

## 实验矩阵

| 变体 | 功能 | 分组 |
|---|---|---|
| baseline | 完整原 BiMC，无旧频率融合和残差 | core |
| frequency_v2 | 现有 Frequency V2，同一文本输入 | core |
| router | 现有 base 训练的 frequency router，同一文本输入 | core |
| zero | 构造并校准新统计，但 lambda 固定为 0 | core |
| visual | 仅视觉 margin | core |
| semantic | 仅语义 margin | core |
| average | 两路有界 margin 直接平均 | core |
| consensus | 同号时保留较弱证据，三频带固定平均 | core |
| shuffle_120 | 语义对应循环改为 [1,2,0] | all |
| shuffle_201 | 语义对应循环改为 [2,0,1] | all |
| original_views | 三个视图均为原图，复用原图编码 | all |
| ordinary_views | 水平翻转、约 90% 中心裁剪缩放、灰度 | all |

core 为 8×3=24 个运行；all 总共 12×3=36 个运行，包含 core。
先 core 后 all，保持命令参数和输出目录一致，已完成且指纹一致的 core 自动跳过。
新 YAML 统一使用 **fp32**，避免极小票数差被 fp16 舍入主导；baseline/V2/router
也使用相同精度。不能直接与历史 fp16、不同支持集的结果相减作为配对收益。

普通多视图不是能量/失真匹配的频率移除实验。本轮没有实现移除响应矩阵、
学习频带对应或因果推断，不应以这些对照宣称频率的因果作用。

## Base 校准协议

- 只读取当前 base 训练缓存，实际增量类和 benchmark test 数据不进入校准。
- base 类按种子分为 50% scale-fit 类、50% lambda-validation 类，严格不交叉。
- 每组默认 20 条伪增量序列：10 个 old 类，每类 20-shot；随后两阶段各追加
  5 个新类，每类 5-shot。每类 query 为 5 张，support/query 无放回且不重叠。
- 每个阶段候选数为 10、15、20。阶段间保留原有 support，协方差按各伪 session
  的类别数加权，完整复用 BiMC prototype/covariance/description 分数链。
- 六个尺度是 fit 类实际 top-2 margin 的绝对值中位数；低于 1e-4 的来源停用，
  不用极小分母放大噪声。validation 类只负责选择 lambda。
- 候选为 `0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1`，按全部验证
  query 出现次数的准确率选择，平局取更小值。0 最优时诚实回退。
- 不同重排变体共享类划分、序列、支持/查询索引与搜索预算，分别校准自己的
  尺度和 lambda。旧 V2 使用原配置，router 使用原 400-episode 训练方案；
  两者并不具有与新方法相同的训练计算预算，需报告实际成本。
- `consensus_calibration.json` 保存序列索引、协议哈希、六尺度、有效来源、
  所有候选 lambda 的准确率/纠错/破坏，以及最终选择。

限制：伪序列最多 20 个候选，低于 CUB 正式 100–200 类；这里检验迁移，而非
假设两者等价。同一 query 会在后续阶段再次出现，不能把出现次数当独立样本
计算置信区间。正式多 seed 的配对差值是主要随机性证据。

## Ubuntu 上运行

先将本分支的全部代码、配置和已有 `description/` 同步到服务器。在服务器的
仓库根目录执行。沿用已有 PyTorch/CUDA 环境；本次没有新增第三方包依赖。
需要已有的 `torch torchvision numpy pillow tqdm yacs ftfy regex` 和 CLIP 权重。
CLIP 默认缓存为 `~/.cache/clip/ViT-B-16.pt`，缺失时沿用项目原有下载流程。
数据根目录的布局仍由 `datasets/cub200.py` 决定，不改变数据集加载协议。

先做不下载 CLIP、不读取完整数据集的代码测试：

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -q
```

设置数据路径，然后预览命令；`--opts` 必须放在最后：

```bash
DATA_ROOT=/absolute/path/to/datasets
OUT=outputs/frequency_consensus_cub

python tools/run_frequency_consensus_experiments.py \
  --suite core --seeds 1 2 3 --output-root "$OUT" --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

运行第一阶段（24 个实验，逐个运行，不抢占多张 GPU）：

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite core --seeds 1 2 3 --output-root "$OUT" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

若想先检查服务器环境，可在相同命令中增加 `--variants baseline zero consensus`
和使用 `--seeds 1`；之后移除限制重跑即可复用已完成结果。这是完整单 seed
benchmark，不是已经缩小图像数量的 smoke test。

补全机制对照（总计 36 个，已有 24 个会跳过）：

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite all --seeds 1 2 3 --output-root "$OUT" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

只重新汇总，不启动模型：

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite all --seeds 1 2 3 --output-root "$OUT" --summarize-only \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

参数、代码、配置、描述文件或支持集发生变化，旧运行不会被视为当前完成结果。
此时优先换输出目录，不要使用 `--rerun` 混淆实验版本。中断的单个运行会从头
开始；当前 checkpoint 是 session 边界快照，不支持从中间 session 恢复推理。
同一个输出目录不要同时启动多个 launcher。

## 同一强度的机制对照

默认各变体分别选 lambda，检验各自相同校准预算下的效果。要排除强度选择因素，
可从主方法 **base 校准文件**读取该 seed 的 lambda，在独立输出目录固定它：

```bash
LAMBDA=$(python -c 'import json; print(json.load(open("outputs/frequency_consensus_cub/cub200/consensus/seed_1/consensus_calibration.json"))["selected_lambda"])')

python tools/run_frequency_consensus_experiments.py \
  --suite all --seeds 1 --fixed-lambda "$LAMBDA" \
  --variants baseline consensus average shuffle_120 shuffle_201 original_views ordinary_views \
  --output-root outputs/frequency_consensus_cub_fixed_seed1 --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

seed 2/3 分别读取自己的 base 选择。这里只固定决策强度，各变体仍用相同 fit
划分重估自身 margin 尺度。若主方法选择 0，固定强度对照全部回退，不能据此
推断匹配与错配等价。不要读取 test 最优结果来设置该值。

## 输出及判读

- `summary.json`：各方法 Final/Average/Base/Novel/H/Forgetting 均值与样本标准差；
  与 baseline、router、average、consensus 的配对 Final 差值统计。n 不足时
  不填造标准差。尚未完成或源代码已变化的运行不会纳入汇总。
- 每个运行目录：`config.yaml`、`support.json`、`run_spec.json`、`run_state.json`、
  `run.log`、`metrics.json`，以及新方法的 `consensus_calibration.json`。
- `metrics.json`：逐 session 的纠错 C、破坏 D、错误间改判 W、Top-2 recall、
  改判/触发比例，分别列 all/base/历史增量/当前新类；另列 base→所有增量类混淆。
- `predictions_session_XX.npz`：测试 sample_id、标签、原预测、第二候选、最终
  预测和候选票数；新方法还保存 eligibility、是否额外编码及 evidence。
  未计算 evidence 用 NaN 表示，只存于 NPZ，不当作“证据为零”。不保存图像特征。
- `pairing_audit` 核对支持集与校准序列哈希、测试样本顺序、标签及预测一致性。
  zero 相对 baseline 的 `output_prediction_mismatches` 应为 0；各重排方法的
  `reference_prediction_mismatches` 应为 0。若不为 0，应先排查协议/数值问题。
- 统计额外图像编码次数、保留张量字节、fit/eval 时间；eval 时间包含数据加载，
  不是已完成 warm-up 的纯模型延迟测量。新增梯度参数为 0 不代表推理无开销。

同一测试集上 `Delta Acc = 100(C-D)/N`。正式判断同时看逐 seed 配对收益、
旧类代价、相对 router/average 的收益和机制消融；不按单个 test 结果反复换
lambda 网格、band cutoffs 或文本。若进行探索性调整，应单列开发实验，冻结
最终方案后做新的确认实验。

## 后续扩展

本轮默认 CUB-200 与已有显式描述。CIFAR-100/miniImageNet 可更换 `--data-cfg`，
但不能继续误用 CUB 描述文件：需要匹配的数据集描述，或显式设置
`TRAINER.BiMC.FREQUENCY.USE_EXPLICIT_DESCRIPTIONS False`，后者退回已有关键词
路由语义，属于不同文本设定，必须标注。修改 shot 时应同时设置
`DATASET.NUM_INC_SHOT` 和 `TRAINER.BiMC.CONSENSUS.SHOT`。

第四数据集、第二骨干全面实验、频率移除响应、能量/失真匹配干预与更强理论
不是本轮代码已经完成的内容。先用上述实验决定当前研究假设是否值得推进。
