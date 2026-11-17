# BiMC 基线手动重新调参

`retune` 分支提供单独的配置入口，适用于原始 BiMC 以及 BiMC†（ensemble）。这是手动指定一组超参数后执行评估的入口，不自动搜索、不训练 CLIP。初始配置沿用已发表设置，并非新找到的最优参数。

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
