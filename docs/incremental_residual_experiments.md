# FSCIL 增量低秩残差：实验协议与运行说明

## 研究问题与实现边界

本实验检验：冻结 CLIP 和 base 阶段确定的共享字典后，仅使用当前 session
的 few-shot support 学习新类系数，能否改善新旧类别的共同分类。
残差输入为原图 CLIP 特征；参考预测可使用 BiMC ensemble，或已训练并冻结的
frequency router。当前实现没有更新 CLIP 内部投影或注意力层，也没有实现每个
频带各自的低秩字典。

对于特征维度 `d`、字典秩 `r` 和当前新增类别数 `k`，冻结字典
`U ∈ R^(d×r)`，仅更新当前类别的 `A_new ∈ R^(r×k)`。
历史类别系数与字典在真实增量过程中均冻结。残差作用于参考预测的对数分数，
由 `MAX_DELTA` 限制修正幅度。旧类保护使用保存的视觉均值锚点；这些均值不是
旧类完整数据分布，因此不能声称冻结参数就保证零遗忘。

需要回答的四个问题：

1. 相对于参考算法，少量增量参数是否提高所有已见类别的准确率？
2. 收益是否来自共享字典，而不仅是增加可训练参数？
3. 旧类保护是否降低旧样本被当前新类抢占的比例，代价是什么？
4. 元训练字典能否优于随机/SVD 字典，以及不受低秩约束的完整维度修正？

本文给出可执行协议，不包含真实数据集上的新结果。

## 固定的数据与调参规则

- 沿用数据集配置的类别顺序、base/new 类数、shot 数和测试划分。默认 CUB-200
  为 100 个 base 类、每次新增 10 类、每类 5-shot；其他数据集以各自 YAML 为准。
- 每个 seed 的所有变体共享一个 `DATASET.SUPPORT_MANIFEST`；默认 seed 为
  `1, 2, 3`，模型随机种子与 `DATASET.SUPPORT_SEED` 均设为该 seed。
  统计 loader 和训练 loader 也必须使用同一批样本，不能各自重抽。
- 共享字典训练只访问 base 训练数据。SVD 使用互不重叠的少样本 support
  与 reference；meta 内层使用 support，外层使用独立 query。
- 字典秩、学习率、训练步数、正则系数、分数修正上限只允许用 base 类的验证
  episode 确定。`META_VAL_FRACTION` 留出验证类；检查点和验证记录应保留。
  不能用真实 incremental test accuracy 选择超参、训练轮数或实验变体。
- 默认实验配置启用 `RESERVE_BASE_VALIDATION=True`，frequency router 若启用，
  也只用字典训练侧的 base 类学习；验证类不参与 router 拟合。该选项对整个矩阵
  生效，包括 baseline。关闭它仍不使用增量测试数据，但验证类不再独立于 router
  训练。启用后的 router baseline 应重新测量，不能直接套用旧实验结果。
- 若先在一个数据集的 base 验证任务上选择超参，再迁移到其他数据集，应明确报告
  “跨数据集固定超参”；若每个数据集单独用 base 验证，应明确说明。
- 增量训练仅使用当前 support 与旧类均值锚点；session 完成适配后会释放逐样本
  图像特征缓存，只保留类别/协方差统计和残差参数。它属于统计记忆协议，不能
  将“没有逐样本 feature replay”表述成“没有任何旧类记忆”。
- 文本描述、CLIP 权重、图像增强、参考分类器、类别顺序、支持集和评估方式保持
  一致。任何额外 feature replay、多个增强视图、模型选择步骤都应单列实验。

## 主实验矩阵

工具默认只输出命令，不启动训练。`core` 每个 seed 有 5 个变体，3 个 seed
共 15 次运行；`extended` 每个 seed 有 7 个变体，共 21 次运行。
可以用 `--variants` 先选少量变体检查集成。

| 变体 | 字典/适配方式 | 主要比较目的 |
|---|---|---|
| `baseline` | 关闭残差，保留同一参考算法 | 建立参考性能 |
| `zero_residual` | meta 字典与适配流程启用，`GAIN=0` | 验证残差关闭时预测回到参考算法 |
| `random` | 固定随机正交字典 | 判断低维参数化本身的作用 |
| `residual_svd` | base 的 support/reference 原型残差做 SVD | 判断数据驱动子空间的作用 |
| `meta` | SVD 初始化，加 base episodic 元训练 | 检验适配后 query 监督的收益 |
| `meta_no_old_loss` | meta，元训练及真实增量的旧类保护权重均为 0 | 测量移除旧类保护的整体影响 |
| `full_rank` | `DICTIONARY=identity`，完整维度类系数 | 检验低秩约束是否有益 |

最后两个变体只在 `extended` 中出现。`full_rank` 的有效秩自动为特征维度，
不受工具 `--rank` 限制；仍使用相同的学习率、步数、范数正则和融合上限作为
第一轮对照。若额外调整其学习率，必须使用相同 base 验证预算并单独说明。

默认起点是 `r=8`、`MAX_DELTA=0.2`、`GAIN=1.0`、
`OLD_LOSS_WEIGHT=1.0`。这些是待验证的起点，不能表述为已经优化的最佳值。
默认配置的系数优化器为 SGD、学习率 0.1，与 meta 内层的优化器和学习率一致；
meta 内层默认只展开 5 步，真实适配与验证默认运行 100 步，因此仍需检查这种
步数差异是否影响迁移。可以用 `OPTIMIZER=adam` 做额外对照，但 meta 内层仍为 SGD，
不能称为完全匹配的元训练。旧类保护实际采用 softplus margin；`OLD_MARGIN`
作用于除以 `TEMPERATURE` 后的分数，而非原始对数分数。
每个增量 session 的系数参数量为 `r × k`；例如 `d=512, r=8, k=10`
时为 80 个，而完整维度对照为 5120 个。应同时报告共享字典和累积历史系数的
总存储开销，不能只报当前 80 个参数。

`zero_residual` 应逐 session 与 `baseline` 得到相同的分类结果，而不是只看
最终平均值“接近”。命令工具会汇总准确率差，但汇总值相同不等于逐样本预测已被
证明完全相同；零增益恒等性的模块测试和确定性评估仍然必要。
若存在差异，应先排查 support、参考模型训练随机性和分数转换，不能继续解释收益。
零增益时，增量 `fit_session` 会校验输入并登记新类，但跳过无效的梯度优化，
报告实际 `steps=0`、保留零系数。因此它是预测回退/流程验证，不是等训练计算预算对照。

## 运行

从仓库根目录运行。先安装原项目依赖、准备本地数据集、CLIP 权重与文本描述，
修改 dataset YAML 的 `ROOT`，或通过末尾 `--opts DATASET.ROOT ...` 覆盖。
本工具不安装依赖、不下载数据；`main.py` 的 CLIP 加载器在本地没有权重时可能
尝试下载，请提前准备权重或使用已有缓存。

先只用 base 训练划分做字典学习与独立验证，不读取 benchmark 测试图像：

```powershell
python main.py --data-cfg configs/datasets/cub200.yaml --train-cfg configs/trainers/bimc_incremental_residual.yaml --opts TRAINER.BiMC.RESIDUAL.BASE_ONLY True TRAINER.BiMC.RESIDUAL.DICTIONARY meta OUTPUT_DIR outputs/residual_base_validation
```

验证准确率保存在 `metrics.json` 的 `dictionary.validation` 中；比较不同设置时
固定 seed 和支持集，并使用独立输出目录。该模式状态为 `base_validation_completed`，
不会产生测试 session 指标，不能作为主矩阵的完成结果。选定配置后关闭 `BASE_ONLY`
再执行完整实验。默认仍保留验证类划分，不自动用验证类重新训练字典。

先检查 2 个变体、1 个 seed 的完整命令，确认路径与参数：

```powershell
python tools/run_incremental_residual_experiments.py --data-cfg configs/datasets/cub200.yaml --seeds 1 --variants baseline zero_residual --dry-run
```

执行小规模集成检查：

```powershell
python tools/run_incremental_residual_experiments.py --data-cfg configs/datasets/cub200.yaml --seeds 1 --variants baseline zero_residual --output-root outputs/residual_smoke --execute --opts TRAINER.BiMC.RESIDUAL.META_STEPS 2 TRAINER.BiMC.RESIDUAL.TRAIN_STEPS 2
```

这仍会完整编码/评估真实数据，不是几秒钟的单元测试。它只缩短优化步骤，
不能将结果当作主实验。若没有真实数据，可先运行文末的合成/工具测试。

运行默认主实验：

```powershell
python tools/run_incremental_residual_experiments.py --data-cfg configs/datasets/cub200.yaml --suite core --seeds 1 2 3 --output-root outputs/residual_cub_core --execute
```

继续做扩展对照，复用同一个 output root、seed 和 support manifest：

```powershell
python tools/run_incremental_residual_experiments.py --data-cfg configs/datasets/cub200.yaml --suite extended --seeds 1 2 3 --output-root outputs/residual_cub_core --execute
```

确认代码与配置未变时，工具会跳过已有的 15 个完整结果，只执行新增的 6 次。
更换数据集只需更换 `--data-cfg` 并选择独立的输出根目录。

参考分支切换为 frequency router 时，必须对整个矩阵统一切换，不能只对 meta 切换：

```powershell
python tools/run_incremental_residual_experiments.py --data-cfg configs/datasets/cub200.yaml --train-cfg configs/trainers/bimc_frequency_router.yaml --suite core --seeds 1 2 3 --output-root outputs/residual_cub_router --dry-run --opts TRAINER.BiMC.RESIDUAL.RESERVE_BASE_VALIDATION True TRAINER.BiMC.RESIDUAL.LR 0.1
```

工具会显式覆盖残差开关和矩阵变量，因此可以直接使用 router 配置。
router 与残差字典的元训练均只发生在 base 阶段，真实增量不再更新共享模块。
这组实验衡量原图残差对频率参考分支的额外价值，不等同于已经实现频率残差字典。

`--opts` 必须放在最后，接受 YACS 的 `KEY VALUE` 对；seed、输出路径、支持集路径
和矩阵变量由工具管理，重复覆盖会报错。使用 `--rank`、`--max-delta`、`--gain`
和 `--old-loss-weight` 修改对应矩阵变量。

## 恢复、失败与产物

每次训练以参数数组调用 `subprocess.run`，不经过 shell；变体顺序执行，避免并发
争抢 GPU。日志位于独立 run 目录。输出结构为：

```text
<output-root>/
  support/<dataset-config-stem>_<dataset-spec-hash>/seed_1.json
  <dataset-config-stem>/meta/seed_1/
    run_spec.json
    run_state.json
    run.log
    metrics.json
    support.json
    config.yaml
    checkpoint.pt
  summary.json
```

`metrics.json` 由主训练流程保存；`support.json` 是该 run 的支持集记录。
`checkpoint.pt` 保存最近一次 session 边界的字典、系数、router 与统计量，
不包含逐样本图像特征或预训练 CLIP 权重。它用于检查与后续加载扩展，当前未提供
自动恢复训练接口；中断后的命令重试会重新运行该变体，并非从检查点续训。
工具在每次运行后写入或更新汇总。只有满足以下条件才会跳过一次已有运行：

- `run_state.json` 标记完成，且命令、配置文件、相关 Python 源码指纹一致；
- `metrics.json` 明确 `status="completed"`，包含有效的 sessions 与完整 summary；
- 本次使用的共享 support manifest 仍存在，且内容哈希与完成时一致。

代码/配置改变后，原目录不会被当成同一个实验静默复用。请使用新 output root，
或显式加 `--rerun`。重跑会保留旧指标为带时间戳的 `metrics.previous.*.json`，
然后重写运行记录和日志；需要完整保存旧检查点/日志时应使用新输出目录。

训练返回非零退出码、缺少指标文件、缺少共享 support manifest、或指标格式不合法
都算失败；工具默认停止，整体返回非零。`--keep-going` 会尝试后续运行，但最终仍
返回非零。目录存在或有旧日志本身不代表实验已经完成。中断后可用同一命令重试。

指纹保护针对代码/配置和支持集文件，不会完整校验数据集图像、CLIP 缓存或外部
文本描述内容。正式实验还应记录这些资源的版本或哈希；不能在重复运行间替换它们。
支持集 manifest v2 另外校验训练样本的有序身份：数组数据包含 shape/dtype/内容哈希，
路径数据包含有序路径哈希；后者不包含路径指向文件的完整内容。旧版 v1 manifest
不满足该校验，需在独立实验目录重新生成，不要静默混用。

## 指标与判读

所有准确率/比例单位为百分数，差值单位为百分点。`metrics.json` 的 `summary`
包含以下字段；session 级指标保存在 `sessions` 中。

| 字段 | 定义 |
|---|---|
| `final_accuracy` | 最后一个 session，全部已见类别的准确率 |
| `average_accuracy` | 从 base session 到最后 session 的准确率算术平均 |
| `base_accuracy` | 最后 session，在最初 base 类测试样本上的准确率 |
| `novel_accuracy` | 最后 session，在所有增量类测试样本上的准确率 |
| `harmonic_accuracy` | base/novel 两组准确率的调和平均 |
| `old_to_new_rate` | 旧类测试样本被预测为当前 session 新类的比例 |
| `forgetting` | 对历史 session 类别组，历史峰值准确率减当前准确率的平均；不含当前新组 |

`old_to_new_rate` 的分母是历史所有已见类的测试样本，不只是 base 类。
`novel_accuracy` 则覆盖所有增量类，两者集合定义不同。
base session 没有新类竞争，相关指标记为 `null`，不能补成 0 再求均值。
冻结字典/旧系数仅保护参数，不保证 `forgetting=0`。
历史峰值仅取当前 session 之前的结果；如果旧类当前准确率提高，遗忘值允许为负，
不会截断为 0。

汇总文件提供每个变体的 mean、样本标准差 `std`、有效 seed 数 `n`，以及相同
seed 对 baseline 的配对差 `paired_delta_vs_baseline`。只有一个有效值时 `std`
为 `null`；缺失值不会被填零。若存在失败，应报告实际完成的 seed 数，不能把
部分成功结果当作完整的 3-seed 实验。`summary.json` 只覆盖本次请求并尝试的矩阵，
不是 output root 中所有历史实验的扫描结果。

除了上述指标，还应保留 session 曲线和主流程输出的运行时间、头部参数量、
实际持有张量字节数。`retained_tensor_bytes` 统计被持有张量的逻辑字节数，
不包括共享 CLIP 和临时计算缓存，不等于 GPU 峰值显存。当前系数库按总类别容量
预分配，存储量应包含尚未使用的零列/行。共享字典 base 训练耗时与每次增量适配
耗时应分开比较；当前 `fit_seconds` 还包含当前 session 的特征编码/统计构造，
不是纯粹优化器耗时。

只有“新类上升、旧类代价可接受、总体/调和平均改善，且配对 seed 趋势一致”时，
才有证据支持机制有效。若 random 与 meta 相当，应报告缺乏元训练收益的证据；
若 full_rank 更好，应检查低秩子空间是否限制了新类，而不是只挑选有利指标。

## 后续敏感性实验

主矩阵通过后，在 base 验证 episode 上分别选择：

- 秩 `r ∈ {4, 8, 16, 32}`；
- 融合上限 `MAX_DELTA ∈ {0.1, 0.2, 0.4}`；
- 旧类保护权重 `OLD_LOSS_WEIGHT ∈ {0.1, 1.0, 10.0}`，关闭保护由扩展对照覆盖；
- support 重采样 seed 的稳定性，以及字典训练/每次增量训练步数。

这些是候选范围，不建议默认做全部笛卡尔积。先控制其他参数单因素比较，
用独立输出根目录保存配置，再冻结最终设置执行真实增量评估。
下一阶段再考虑分频带字典、可靠性门控或投影层 LoRA；它们应是独立模块与实验。

## 本地验证

工具测试仅依赖 Python 标准库，不加载 CLIP、不访问网络、不启动真实训练：

```powershell
python -m unittest discover -s tests -p test_incremental_residual_experiments.py -v
```

它检查支持集配对、独立输出目录、禁止 seed 覆盖、无写入 dry-run、完成状态校验、
错误返回、失败重跑不遗留成功状态，以及带缺失值的配对汇总。
残差模块和支持集协议的合成测试需要另行运行；它们通过也不能替代真实数据集验证。

在已安装项目依赖的环境中运行完整测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖零增益恒等、旧系数与字典冻结、真正的 meta 外层梯度、support/query 隔离、
支持集复现与样本顺序校验、指标定义，以及无需下载 CLIP 的多 session 集成。
