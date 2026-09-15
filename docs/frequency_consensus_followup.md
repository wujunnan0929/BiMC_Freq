# 完整 CUB core 结果后的验证实验

日期：2026-09-13。沿用分支 `codex/frequency-consensus-validation`。
这是看到原 core 结果后的探索性修订，不是独立确认实验，也不承诺提升性能。

## 目标与预先约定

原 24/24 结果：consensus 相对 baseline 的 Final 为 −0.086 pp，Base 为
+1.117 pp，Novel 为 −1.263 pp；router 的 Final 为 +0.575 pp。
本轮先检验校准目标，不改变 backbone、频率视图、描述、证据公式或 lambda 网格。
只用 base 训练类的校准数据选 lambda，benchmark 测试标签仅用于事后评估。

主修订预先指定为 `consensus_balanced`，`consensus_safe` 是保守约束消融。
不能运行后按 test 最优结果改称另一个变体为预先指定主方法。报告三个 seed 的
Final/Average/Base/Novel/H 配对差值、逐阶段 C/D、触发比例和计算量。
若 Novel 和 H 仍受损，或仅靠 lambda=0 消除损害，应报告失败/回退，不能视为提升。
新 seed 仅评估随机性，仍使用相同测试图像；之后还需其他数据集的冻结方案验证。

## A：只改变 base 校准目标（7 × 3 = 21 个运行）

| 变体 | 选择规则 | 用途 |
|---|---|---|
| baseline | 原 BiMC | 当前代码下的配对参照 |
| router | 原 frequency router | 当前最强性能参照 |
| zero | lambda=0 | 原预测一致性检查 |
| consensus | micro_all | 保留旧规则，包含 stage 0 |
| consensus_inc_micro | micro_incremental | 只排除 stage 0，仍按 query 数加权 |
| consensus_balanced | balanced_incremental | 排除 stage 0，阶段内旧/新各一半，阶段间等权 |
| consensus_safe | balanced_incremental + guard=0 | 每个增量阶段旧/新验证准确率均不低于 lambda=0 |

设 A_old(t, lambda) 和 A_new(t, lambda) 是按阶段汇总全部 validation episodes
后得到的准确率；new 包含该阶段所有已见伪增量类，另记录历史增量/当前新类。

```
J_balanced(lambda) = mean_{t=1..STAGES} [ (A_old(t,lambda) + A_new(t,lambda)) / 2 ]
```

所有规则平局选择更小 lambda，0 始终在网格中。safe 的约束逐阶段检查，不是
先把阶段混合后检查；它仅约束 base 验证数据，不保证正式测试或每个类别无损。
固定 lambda 模式不执行自动选择，约束结果仅作为诊断记录。

旧规则默认伪候选数 10/15/20，old query 占全部出现次数的 66.7%；去掉 stage 0
后仍为 57.1%。均衡目标进一步去除组别数量权重。六个尺度的拟合继续包含 stage 0，
fit/validation 类划分、支持/查询索引及搜索预算保持一致，因此 A 只消融 lambda 选择。

所有校准运行都记录各规则的 base-only 候选选择 `selection_audit`，可直接看到
规则是否选到不同 lambda。若 lambda 相同，应先验证预测相同，而非声称机制不同。

## Ubuntu 环境与命令

同步本次修改的代码、配置、新增工具和 `utils/consensus_selection.py`，沿用服务器
已有 PyTorch/CUDA 环境，无新增第三方依赖。只同步旧脚本、不同步新 utils 会报错。
下面在仓库根目录执行；`--opts` 必须最后。DATA_ROOT 指向含 CUB_200_2011 的目录。

```bash
DATA_ROOT=/absolute/path/to/datasets
OUT=outputs/frequency_consensus_cub_calibration_v2
FIXED=outputs/frequency_consensus_cub_fixed_v2

test -f "$DATA_ROOT/CUB_200_2011/images.txt"
OMP_NUM_THREADS=1 python -m unittest discover -s tests -q
```

可以先分析旧完整日志，不加载模型/图像，也不会重新运行旧实验。旧 schema 缺少
逐阶段统计，无法从它精确重算新增目标；诊断工具会标明这一限制。

```bash
python tools/analyze_consensus_calibration.py \
  --input-root outputs/frequency_consensus_cub \
  --output outputs/frequency_consensus_cub/calibration_diagnostics.json
```

若仅有 summary.json 而没有每个运行的 consensus_calibration.json，跳过这条诊断
命令；它不是 A 的先决条件。

先预览 A，再执行（两个命令的参数一致，仅切换运行开关）：

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite calibration --seeds 1 2 3 --output-root "$OUT" --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'

python tools/run_frequency_consensus_experiments.py \
  --suite calibration --seeds 1 2 3 --output-root "$OUT" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'

python tools/analyze_consensus_calibration.py \
  --input-root "$OUT" --output "$OUT/calibration_diagnostics.json"
```

想先检查环境，可在 A 的执行命令加 `--variants baseline zero consensus_balanced`
并改为 `--seeds 1`，之后执行完整 A 命令可复用完成结果。这仍是单 seed 完整 benchmark。

**不要继续写入原 outputs/frequency_consensus_cub。** 本次代码/配置指纹已经改变。
新目录重新运行必要参照，原 24 个历史运行不删除、不覆盖。A 内各方法共享支持集。
如需跨新目录共用支持集，可用 `--support-root "$OUT/support"`。若指定原目录的
support 根目录，需要原 dataset 配置哈希和 DATASET 参数也相同才能命中原 manifest。
同一随机种子本身不等于完成了跨版本的配对审计。

中断后直接重跑相同命令；只有指纹和支持集一致的完成运行才跳过。不用 `--rerun`
覆盖旧实验。不要同时启动两个 launcher 写同一输出目录。

## B：相同 lambda 下比较证据（6 × 3 = 18 个运行）

A 完成后，直接读取预先指定的 `consensus_balanced` 的 **每个 seed 的 base
校准 lambda**，固定给 visual/semantic/average/consensus。baseline/zero 提供
同目录配对检查。不同 seed 可以有不同 lambda，同一 seed 的非零对照共用强度。
各变体仍从同一 fit 数据拟合尺度；频率视图及语义对应相同的变体应得到相同尺度。

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite core --seeds 1 2 3 \
  --variants baseline zero visual semantic average consensus \
  --lambda-from-root "$OUT" --lambda-from-variant consensus_balanced \
  --support-root "$OUT/support" --output-root "$FIXED" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

自动读取会校验 base-only 来源、seed、auto_calibrate、代码/配置哈希和用户覆盖参数，
把源文件哈希和 lambda 写入 run_spec。**A/B 之间不要改代码、配置、描述或命令中的
数据/训练参数**。若必须改变，使用新版本 A，不绕过校验，也不复制伪造 run_spec。
不读取 benchmark test accuracy 来选择强度。可以把 `--execute` 换成 `--dry-run`
预览，但源校准文件必须已存在。

如果三个 seed 都选择 0，先报告均衡规则回退；B/C 的改判机制会退化，应暂停这些
昂贵对照并诊断 base 证据，不从 test 网格搜索一个正数强行运行。部分 seed 为 0
时保留并如实报告零值，不剔除它们美化均值。

## C：补全机制对照（在 B 上追加 12 个运行）

在 A/B 显示值得继续研究的证据后，保持相同固定 lambda，追加频带错配与普通视图。
下面总计划 30 个运行，与 B 相同的 18 个会跳过；输出完整包含 B 的新 summary。

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite all --seeds 1 2 3 \
  --variants baseline zero visual semantic average consensus \
             shuffle_120 shuffle_201 original_views ordinary_views \
  --lambda-from-root "$OUT" --lambda-from-variant consensus_balanced \
  --support-root "$OUT/support" --output-root "$FIXED" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

这些对照检验匹配和额外视图的作用，不是能量/失真匹配的因果频率干预，不能过度解释。

## 可选：扩大伪候选数，单独改变协议

若 A 显示 base 与正式阶段仍有迁移落差，可在独立目录把 10/15/20 扩大到
20/30/40，保持阶段类别比例不变。每个 fit/validation split 有 50 个 base 类，
40 个候选可行。这仍远小于正式 100–200 类，并不能完全消除候选数分布差异。
此处同时改变伪支持集规模、协方差和尺度，属于协议敏感性实验，不是纯 lambda 消融。

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite calibration --seeds 1 2 3 \
  --variants baseline zero consensus consensus_inc_micro consensus_balanced \
  --support-root "$OUT/support" \
  --output-root outputs/frequency_consensus_cub_candidates40 --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0' \
         TRAINER.BiMC.CONSENSUS.OLD_WAY 20 TRAINER.BiMC.CONSENSUS.NEW_WAY 10
```

这是额外 15 个运行，先完成 A 后决定是否需要，不与原协议混合汇总。

## 输出检查与返回文件

- `summary.json`：原准确率指标、与 baseline/router/consensus/consensus_balanced 的
  配对指标差值；现在保留候选 lambda 的逐阶段校准统计，不再只保留选中值。
- `consensus_calibration.json` schema 2：每个 lambda 的各阶段 all/old/new/
  historical_incremental/current_new 的分母、准确率、C/D/W、Top-2 recall、触发率；
  选择得分、约束违反组别及各规则的 base-only 选择。
- `metrics.json`：全 session 曲线；新增历史增量/当前新类误判到 base 的比例，及
  “原本正确、改判后错误”的目标组别计数，区分增量→base 与增量→其他增量。
- `pairing_audit`：zero 相对 baseline 的输出差异应为 0；重排方法原预测差异应为 0；
  核对实际检查的 session 数、样本/标签对齐及同视图同语义对应下的尺度一致性。
- `run_spec.json`：固定 lambda 的来源和哈希。不要修改实验完成后的配置/来源文件。

只重新汇总 A（不会启动模型）：

```bash
python tools/run_frequency_consensus_experiments.py \
  --suite calibration --seeds 1 2 3 --output-root "$OUT" --summarize-only \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0'
```

把 A 的 summary.json 和 calibration_diagnostics.json 返回即可先做完整校准分析。
若要定位逐 session 的损害，另返回四个 consensus 变体各 seed 的 metrics.json；
需要样本级解释再同步 predictions_session_XX.npz，无需传图像特征。

本地测试只验证规则、协议与合成数据集成，不替代 Ubuntu/CUDA 上的正式结果。
