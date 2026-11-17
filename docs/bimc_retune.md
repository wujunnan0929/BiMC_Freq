# BiMC 基线手动与自动重新调参

`retune` 分支提供单独的配置入口，适用于原始 BiMC 以及 BiMC†（ensemble）。这是手动指定一组超参数后执行评估的入口，不自动搜索、不训练 CLIP。初始配置沿用已发表设置，并非新找到的最优参数。

自动搜索入口是 `tools/tune_bimc.py`，用法见下文“自动调参”。它复用缓存特征和基类伪增量验证流程，输出手动入口可直接读取的最优配置。

## 参数在哪里修改

根据数据集编辑一个文件即可：

- `configs/retune/cub200.yaml`
- `configs/retune/cifar100.yaml`
- `configs/retune/miniimagenet.yaml`

| 配置键 | 含义 | 取值 |
|---|---|---|
| `DATASET.BETA` | 图文融合的**文本权重**，图像权重为 `1-BETA` | 浮点数 `[0,1]` |
| `TRAINER.BiMC.LAMBDA_T` | 文本校准中类别描述的权重，普通 prompt 权重为 `1-LAMBDA_T` | 浮点数 `[0,1]` |
| `TRAINER.BiMC.LAMBDA_I` | 新类视觉原型向基类 donor 靠拢的强度 | 浮点数 `[0,1]`；`0.0` 关闭该修正 |
| `TRAINER.BiMC.TAU` | 基类 donor 权重 `softmax(TAU * cosine)` 的乘数 | 正整数；越大越集中，不是除数形式的温度 |
| `TRAINER.BiMC.USING_ENSEMBLE` | `True` 使用 BiMC†，`False` 使用普通 BiMC | 布尔值 |
| `DATASET.ENSEMBLE_ALPHA` | ensemble 中主图文原型分支的权重，余下权重给协方差/描述近邻分支 | 浮点数 `[0,1]`；只在 ensemble 开启时影响输出 |
| `TRAINER.BiMC.GAMMA_BASE / GAMMA_INC` | 原实现协方差估计中的收缩系数 | 非负浮点数 |
| `TRAINER.BiMC.VISION_CALIBRATION` | 是否校准新类视觉原型 | 布尔值 |
| `TRAINER.BiMC.TEXT_CALIBRATION` | 是否混入类别描述；关闭时有效 `LAMBDA_T=0` | 布尔值 |
| `DATASET.NUM_INC_SHOT` | 每个新增类别的支持样本数 | 正整数，如 `1`、`5` |
| `SEED` | 本次运行的随机种子 | 非负整数；建议比较时用 `1,2,3,4,5`，`0` 沿用原实现随机模式 |
| `DATASET.SUPPORT_SEED` | 单独控制支持集抽样 | `-1` 跟随 `SEED`；固定为相同非负值可保持同一支持集 |
| `OUTPUT_DIR` | 实验保存目录 | 空字符串自动生成含配置哈希的路径；也可指定新目录 |

数据目录仍在 `configs/datasets/<dataset>.yaml` 的 `DATASET.ROOT`；描述文件在 `DATASET.GPT_PATH`。CLIP 型号、精度、GPU 和 batch size 继承 `configs/trainers/bimc_ensemble.yaml`，也可以写进 retune 文件或通过 `--opts` 覆盖。例如 `MODEL.BACKBONE.NAME ViT-L/14`。

配置优先级从低到高：**代码默认值 → 数据集 YAML → trainer YAML → retune YAML → `--opts`**。retune 文件中有数据集名称检查，避免误把 CUB-200 的参数配置直接用于 CIFAR-100。布尔值写 `True/False`；浮点配置请写 `0.0/1.0`，遵循现有 YACS 类型检查。

## 如何运行

先激活安装了原项目依赖的环境，从项目根目录执行。下面的单行命令适用于 PowerShell 和 Bash。

仅预览最终生效的配置，不读取数据、不加载 CLIP、不创建实验目录：

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc_ensemble.yaml --retune_cfg configs/retune/cub200.yaml --dry-run
```

编辑配置后正式执行，去掉 `--dry-run` 即可：

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc_ensemble.yaml --retune_cfg configs/retune/cub200.yaml
```

临时覆盖参数，无须修改 YAML。下面使用 CUB-200、1-shot、`beta=0.6, lambda_T=0.7, lambda_I=0.2, tau=32`；它是一组可运行的参数示例，不承诺重新运行的精度：

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc_ensemble.yaml --retune_cfg configs/retune/cub200.yaml --opts DATASET.NUM_INC_SHOT 1 DATASET.BETA 0.6 TRAINER.BiMC.LAMBDA_T 0.7 TRAINER.BiMC.LAMBDA_I 0.2 TRAINER.BiMC.TAU 32
```

`--opts` 必须放最后。需要先预览时，把 `--dry-run` 放到 `--opts` 前。

普通 BiMC 使用 `--opts TRAINER.BiMC.USING_ENSEMBLE False`。换数据集时同时替换 `--data_cfg` 和 `--retune_cfg` 的文件名。多种子运行可以修改 `SEED`，例如：

```bash
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/bimc_ensemble.yaml --retune_cfg configs/retune/cifar100.yaml --opts SEED 2 DATASET.NUM_INC_SHOT 1 DATASET.BETA 0.9
```

默认目录为 `outputs/retune/<dataset>/<method>_shot<K>_seed<S>_<hash>/`。哈希由最终合并配置计算，改变超参数、骨干或种子会生成另一目录。目录中沿用 Runner 的输出：

- `config.yaml`：实际使用的完整冻结配置。
- `support.json`：每个 session 的支持集样本，便于配对复现实验。
- `metrics.json`：各 session 指标及 final/average/base/novel/HM 等汇总；完成时 `status=completed`。
- `checkpoint.pt` 及已有评估输出：按原 Runner 的开关保存。

手动调参模式会拒绝覆盖已有非空目录。如果需要有意重跑同一配置，用 `--opts OUTPUT_DIR outputs/retune/cub200/repeat_02` 指定另一个目录。

## 代码执行逻辑

1. `main.py:setup_cfg()` 读取数据和 trainer 配置，随后读 `--retune_cfg`，最后应用命令行覆盖。
2. `utils/bimc_retune.py` 校验参数范围与数据集匹配，确认 frequency、residual 等实验分支未启用，并在未手填目录时生成默认目录。配置随后冻结。
3. `--dry-run` 打印配置后退出，因此机器没有 GPU 或模型权重也可检查配置（仍需要 YACS）。正式运行将同一配置交给已有 `engine.engine.Runner`。
4. `models/bimc.py:soft_calibration()` 用 `LAMBDA_I/TAU` 构造并融合基类 donor；原有原型统计流程只在校准开关开启时使用这一步。
5. `reference_scores_from_features()` 使用 `LAMBDA_T` 混合文本，使用 `BETA` 融合文本和视觉原型；ensemble 开启时再由 `ENSEMBLE_ALPHA` 混合协方差/描述近邻分支。

本次没有改写 BiMC 的数学计算，也没有引入 DCR。使用 `--retune_cfg` 时进行额外的基线配置检查；不传该参数的现有实验保持原配置流程。

## 与论文选参协议的关系

正式运行上面的 `main.py` 命令会按现有 benchmark 流程评测测试集，**它本身不执行 base-only 验证选参**。论文中要报告重新调参结果时，应先在基类伪增量验证集上确定参数，再固定参数评估测试集。仓库已有 `fdrfreq.run select` 和 `fdrfreq/configs/paper.json` 中的 `BiMC_rt` 提供基于缓存特征的验证搜索；本次手动入口不改动该协议，也不把反复测试的结果自动认定为验证集最优配置。

## 自动调参

### 1. 准备特征库（已有则直接复用）

搜索读取现有 `fdrfreq_bank_v1` 特征库，在 CPU 上复用图像和文本特征，不为每个候选重新跑 CLIP。每个数据集、骨干各提取一次。示例：

```bash
python -m fdrfreq.extract --dataset cub200 --data-root ../datasets --out banks/cub200_vitb16 --backbone ViT-B/16 --sigmas --aug --dct-layers --seeds 1 2 3 4 5
```

这一步需要原项目的 PyTorch/CLIP 环境、数据和描述文件；只提原图及文本，不提频率视图、关键词或 DCT。提取程序编码 train/test 图像以供后续固定评估使用；**自动搜索仅在原始基类训练图像上构造验证任务，不计算 benchmark 测试指标**。特征库可以供不同 shot 数复用。当前电脑未发现 `banks/`，完整真实数据搜索需要先生成或放入对应特征库。

### 2. 设置搜索范围

编辑 `configs/retune/search.yaml`。其中 `grid` 的键与手动参数名相同：

```yaml
objective: hm
split_mode: contiguous
split_seeds: [2, 3, 4]
sequences: 10
grid:
  DATASET.BETA: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  TRAINER.BiMC.LAMBDA_T: [0.3, 0.5, 0.7]
  TRAINER.BiMC.LAMBDA_I: [0.0, 0.05, 0.1, 0.2, 0.3]
  TRAINER.BiMC.TAU: [8, 16, 32]
```

- 默认枚举 **270 组**参数，额外比较 `configs/retune/<dataset>.yaml` 中的初始配置，避免在验证集上选到比初始配置更差的点。得分相同时保留初始配置；其他并列候选沿用网格顺序。
- 固定某个参数：将其写成单值列表，如 `TRAINER.BiMC.LAMBDA_I: [0.1]`；或者从 `grid` 删除该键，让它继承手动配置或 `--opts`。
- 同一参数不能同时出现在 `grid` 和 `--opts`，避免固定值与搜索范围冲突。
- 还支持搜索 `DATASET.ENSEMBLE_ALPHA`、`TRAINER.BiMC.GAMMA_BASE` 和 `GAMMA_INC`；这三个搜索要求 `USING_ENSEMBLE=True`。
- `objective: hm` 最大化最后一个伪 session 的基类/新类准确率调和均值；也可设 `last`（最后一个伪 session 的总准确率）或 `avg`（伪 sessions 平均准确率）。所有目标只基于验证任务。
- `split_mode: contiguous` 使用一个固定的基类类别划分，三个 seed 改变图像划分；`random` 会随 seed 改变伪基类/伪新类划分。每次用半数原始基类作伪基类，其余作伪新类；伪基类每类约 20% 图像留作 query，伪新类 query 与 support 分离并按类限额。
- `split_seeds × sequences` 决定验证任务数，默认 30。可先减小网格和任务数检查流程，再在正式实验前固定完整搜索范围和协议。

### 3. 搜索与导出

预览搜索规模，无须特征库，也不会创建文件：

```bash
python tools/tune_bimc.py --dataset cub200 --bank banks/cub200_vitb16 --shot 1 --dry-run
```

执行自动搜索：

```bash
python tools/tune_bimc.py --dataset cub200 --bank banks/cub200_vitb16 --shot 1 --out outputs/retune/cub200_auto_1shot
```

切换普通 BiMC：追加 `--opts TRAINER.BiMC.USING_ENSEMBLE False`。切换 ViT-L/14：使用对应特征库，并追加 `--opts MODEL.BACKBONE.NAME ViT-L/14`。参数搜索使用 fp32 提取的固定 CLIP 特征，导出配置也设置 `PREC: fp32`。程序检查特征库的数据集、骨干、类别划分和描述文件哈希（库有记录时），拒绝误用其他数据或子集库。

不写 `--out` 时，搜索目录自动生成为 `outputs/retune/search/<dataset>/shot<K>_<hash>/`。完成后包含：

| 文件 | 内容 |
|---|---|
| `best.yaml` | 完整的最优配置；可交给手动入口运行，独立生成评估目录 |
| `search_spec.json` | 实际固定配置、搜索范围和验证协议 |
| `search_report.json` | 每个候选的验证指标、初始配置指标、最优参数及选择依据 |
| `locked_BiMC_auto.json` | 与已有 fdrfreq 工具兼容的锁定配置与哈希 |

输出目录必须为空，已完成或中断的搜索结果不会被静默覆盖；如需重跑，指定新目录。`search_spec.json` 记录搜索开始，只有成功写出 `best.yaml` 才表示选择完成。

### 4. 用选好的参数固定评估

```bash
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc_ensemble.yaml --retune_cfg outputs/retune/cub200_auto_1shot/best.yaml --dry-run
```

核对后去掉 `--dry-run`。可用 `--opts SEED 2` 等运行不同支持集种子。`best.yaml` 包含完整配置，因此优先于 trainer 的默认参数；额外的 `--opts` 仍拥有最高优先级。

自动入口的代码链路是：**合并手动初始配置 → 读取搜索范围 → 同一批基类伪增量任务评估所有候选 → 按选定目标取最好值 → 导出可运行 YAML 和锁定结果**。搜索结果是所设网格和验证协议内的最优选择，不代表任意参数空间或测试集上的全局最优。本次实现已通过合成特征库的端到端选择、无测试样本访问及导出配置重载验证；尚未运行真实数据集的 270 组搜索。
