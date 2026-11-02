# FGDR：免训练频率 grounding 的分解-重组校准（实验框架）

这套代码只新增文件，不改动仓库里任何已有文件。它把所有校准规则写成冻结 CLIP 特征上的闭式 NumPy 计算：在 GPU 机器上抽一次特征缓存，之后所有网格搜索、选参和最终评估都在 CPU 上跑，几秒到几十分钟一轮。

论文定位、贡献点、实验矩阵、闸门和时间线见 `fgdr/docs/PAPER_PLAN.md`。

## 0. 建分支（在仓库根目录，PowerShell 或 bash 都一样）

这些文件已经写进你的工作区，但还是未跟踪状态。当前分支是 `codex/frequency-prototype-completion`。

```bash
git status                                   # 先确认当前分支的改动已提交或 stash
git switch -c claude/fgdr-calibration        # 未跟踪的新文件会跟着进入新分支
git add fgdr tools/fgdr_cache_features.py tools/fgdr_keywords.py tools/fgdr_run.py \
        tests/fgdr_synth.py tests/test_fgdr_core.py tests/test_fgdr_pipeline.py tests/test_fgdr_cli.py
git commit -m "Add FGDR training-free calibration framework"
python -m pytest -q tests/test_fgdr_core.py tests/test_fgdr_pipeline.py tests/test_fgdr_cli.py
```

核心代码只依赖 numpy；两个抽特征脚本依赖你现有的 torch / CLIP / DatasetManager 环境。

## 1. 抽特征缓存（GPU，每个数据集一次）

```bash
# 原图 + 3 级高斯模糊 + 3 个非频率对照视图 + 第 8/12 层 patch-token DCT 频带能量
python tools/fgdr_cache_features.py --dataset cub200 --data-root ../datasets \
    --output outputs/fgdr/cache/cub200 --seeds 1 2 3 4 5 --shots 5 1 10
python tools/fgdr_cache_features.py --dataset miniimagenet --data-root ../datasets \
    --output outputs/fgdr/cache/miniimagenet --seeds 1 2 3 4 5 --shots 5 1 10
python tools/fgdr_cache_features.py --dataset cifar100 --data-root ../datasets \
    --output outputs/fgdr/cache/cifar100 --seeds 1 2 3 4 5 --shots 5 1 10

# FDR 式关键词库（两种坐标：final = 正确投影到联合空间；fdr = FDR 官方代码的 value 空间下标）
python tools/fgdr_keywords.py --dataset cub200 --data-root ../datasets \
    --output outputs/fgdr/cache/cub200_keywords --m 8 --q 37
```

- 缓存把训练集和测试集分成两个文件。`dev`、`diagnose-*` 只读训练文件。测试文件只有拿着已锁定、带校验和的选择文件（`lock`）才能读。
- support 抽样直接从 `DatasetManager` 读取，每个 (shot, seed) 一份，与 engine 完全一致。元数据里的 `support_replica_match` 应全为 true。
- 先用 `--limit-per-class 3` 做一次冒烟测试，确认能跑通后再跑全量。
- 换骨干：加 `--backbone ViT-B/32` 或 `--backbone ViT-L/14`，输出到另一个 stem。

## 2. 闸门 G0：复现 BiMC（必须先过）

```bash
python tools/fgdr_run.py parity --cache outputs/fgdr/cache/cub200 --out outputs/fgdr/parity/cub200 \
    --seeds 1 2 3 4 5 --expect 72.62      # 你本地 fp32+ensemble 的 5-seed G1 数字
```

`BiMC_dagger` 的 Last 与 engine 的差超过 0.15pp 时，命令返回非零，此时先查缓存，不要比较任何方法。parity 只评估两个固定参考，不涉及任何选参。

## 3. 只用 base 数据的诊断（几分钟）

```bash
python tools/fgdr_run.py diagnose-masks --cache outputs/fgdr/cache/cub200 --out outputs/fgdr/diag/cub200 \
    --q 37 50 --shots 1 5 --keywords outputs/fgdr/cache/cub200_keywords      # 闸门 G1'
python tools/fgdr_run.py diagnose-views --cache outputs/fgdr/cache/cub200 --out outputs/fgdr/diag/cub200   # D1/D2
```

## 4. 伪增量开发与选参（只用 base 训练图）

```bash
python tools/fgdr_run.py dev --cache outputs/fgdr/cache/cub200 --config fgdr/configs/stage0_anchor.json \
    --out outputs/fgdr/dev/cub200/stage0_anchor
python tools/fgdr_run.py dev --cache outputs/fgdr/cache/cub200 --config fgdr/configs/stage0_fdr.json \
    --keywords outputs/fgdr/cache/cub200_keywords --out outputs/fgdr/dev/cub200/stage0_fdr     # 闸门 G0'
# 只有 G0' 与 G1' 都通过才跑：
python tools/fgdr_run.py dev --cache outputs/fgdr/cache/cub200 --config fgdr/configs/stage1_frequency.json \
    --keywords outputs/fgdr/cache/cub200_keywords --out outputs/fgdr/dev/cub200/stage1_frequency
python tools/fgdr_run.py dev --cache outputs/fgdr/cache/cub200 --config fgdr/configs/stage2_blocks.json \
    --out outputs/fgdr/dev/cub200/stage2_blocks                                                  # 闸门 G1 (D3)
```

- 默认伪协议：CUB 用 50 个 A 类拟合，50 个 V 类分成 5 个 10-way 会话；CIFAR/mini 用 30 + 6×5-way，每类最多 100 张 query。划分种子 2/3/4，每个划分 20 条序列。
- 每个 family 的最优配置写进 `selection_draft.json`，护栏是相对 BiMC† 的 Base / Novel 降幅都不超过 0.2pp。`paired_last_diff` 给出逐 episode 的配对差、标准差和正负计数。
- 网格里每个强度都把 0（或恒等值）放在第一个，平局时自动选更简单的配置。

## 5. 锁定，然后在测试集上只跑一次

```bash
python tools/fgdr_run.py lock --draft outputs/fgdr/dev/cub200/stage0_fdr/selection_draft.json \
    --families K0 K1 K1fdr K2 --baselines-config fgdr/configs/baselines.json \
    --out outputs/fgdr/locks/cub200_stage0_fdr.json
python tools/fgdr_run.py final --cache outputs/fgdr/cache/cub200 --lock outputs/fgdr/locks/cub200_stage0_fdr.json \
    --keywords outputs/fgdr/cache/cub200_keywords --out outputs/fgdr/final/cub200_stage0_fdr --seeds 1 2 3 4 5
python tools/fgdr_run.py compare --results outputs/fgdr/final/cub200_stage0_fdr/results.json --a BiMC_dagger --b K1
python tools/fgdr_run.py diagnose-prototypes --cache outputs/fgdr/cache/cub200 \
    --lock outputs/fgdr/locks/cub200_stage0_fdr.json --keywords outputs/fgdr/cache/cub200_keywords \
    --out outputs/fgdr/diag/cub200_protos
```

`final` 输出 `results.json`（逐 seed、逐会话指标，以及相对参考的配对 C/D、McNemar p、bootstrap 区间）、`predictions.npz` 和可直接贴进论文的 `table.md`。

## 6. 方法与配置对照

| family | 含义 | 配置文件 |
|---|---|---|
| BiMC / BiMC_dagger | 官方 BiMC / BiMC†，与 `models/bimc.py` 逐式一致 | 内置 |
| A1 / A2 / A2c | 池化类内 GDA 混入 BiMC† 票数；A2 加 shot-aware 预测分布（κ/n_c 方差膨胀）；A2c 用校准后的新类原型作 GDA 均值 | stage0_anchor |
| K0v / K0 | 整体重组；供体权重分别来自视觉 / 文本相似度（FDR 代码中论文没写的第一步） | stage0_fdr |
| K1 / K1fdr / K2 | K0 + 关键词分段（修正坐标 / FDR 原坐标 / 每个关键词固定的随机掩码） | stage0_fdr |
| Q1 / Q2 / Q3 / Qp | 频率 grounding 分段 / 换成别类掩码 / 随机掩码 / 非频率扰动掩码 | stage1_frequency |
| Q1b0–2 | 单个频段（0 = 最细） | stage1_frequency |
| Q4 / Q4s | 关键词分段的强度由细节能量决定 / 打乱后的对照 | stage1_frequency |
| F0–F6 | 联合 GDA 的信息上界检验（DCT 频带、随机系数组、模糊视图、非频率视图、纯噪声） | stage2_blocks |
| CLIP_ZS, ProtoNet, TEEN, GDA_only, FDR_tf, BiMC_FDR | 不调参的论文基线 | baselines |

## 7. 测试

`tests/test_fgdr_*.py` 用合成特征覆盖以下几点：
- 等价性：BiMC 公式逐式核对；视觉整体重组等于 BiMC soft calibration；α=0 / κ=0 精确回退；缓存路径与非缓存路径预测相同。
- 协议：伪 episode 只用 base 类，support 与 query 不重叠；没有锁文件不能读测试集。
- 端到端 CLI：parity → dev → lock → final → compare → 各项诊断。
