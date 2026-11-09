# fdrfreq：缓存特征上的免训练 FSCIL 校准框架

这个目录独立于 `engine/` 主流程，只读取项目里现有的 CLIP、数据集和 BiMC 文本代码，不修改任何已有文件。论文计划见同目录的 `PAPER_PLAN.md`。

## 为什么用缓存特征

所有方法都是冻结 CLIP 之上的闭式计算。每个数据集只做一次 GPU 前向，把需要的特征写进一个 bank 目录。之后的选参、消融和评测都是纯 numpy，在 CPU 上就能跑，结果可逐位复现。

- **单元测试与 parity。** `tests/test_fdrfreq.py` 对比了缓存版 BiMC / BiMC† 与一个逐循环的独立重实现，两者逐 session 完全一致。真实数据上要再用 `parity` 命令和 engine 的 fp32 结果对一遍。
- **运行时间。** 在一个 CUB 规模的合成 bank 上（200 类、512 维），2 核 CPU 跑 15 个配置 × 30 个伪增量 episode 约 40 秒。

## 两阶段流程（务必按顺序）

```bash
# 阶段 1：只用 base 训练图像（parity 除外，它只复现已发表基线）
bash fdrfreq/scripts/dev.sh cub200 ../datasets 0
#   -> banks/cub200_vitb16/                     特征库
#   -> outputs/fdrfreq/cub200_vitb16/parity_shot5.json
#   -> outputs/fdrfreq/cub200_vitb16/shot{5,1}/diagnostics_*.json, select_*.json, locked_*.json, gates.txt

# 阅读 gates.txt，在碰测试集之前确定最终方法由哪些组件构成（见 PAPER_PLAN.md 第 6 节）

# 阶段 2：只对锁定配置跑一次测试集
bash fdrfreq/scripts/final.sh cub200 "ZS ZS_desc Proto TEEN FDR_tf_published BiMC BiMC_ens LA A1 A2 K0 K1 K1_fdrcoord K2_random Q1 Q2_classperm Q3_random F1 F3_aug F4_noise F5_randband"
#   -> shot{5,1}/eval_*.json, pred_*.npz, tables.md, tables.tex
```

Windows 下用 `fdrfreq\scripts\dev.ps1` 和 `final.ps1`，参数相同。

**最终方法（PAPER_PLAN §12.9）。** 在 dev_core.sh 跑完的数据集上，再运行 `bash fdrfreq/scripts/dev_m4.sh <dataset>`：选 M4 一族（对照、消融、插件）→ gates → `summary_m4.txt`。冻结的最终方法 M4a（§12.11）用 `dev_m4a.sh` 选参、`final_m4a.sh` 测试（已于 2026-11-04 各运行一次，§12.13）。

**当前协议（PAPER_PLAN §12.8）。** 新数据集用 `bash fdrfreq/scripts/dev_core.sh <dataset> <data-root> <gpu>`：建 bank → parity → 在 contiguous 伪增量划分上以 HM 为目标选论文的核心 entry（输出到 `shot{1,5}_hm/`）→ gates → errconc。`EXTRA=1` 时再跑频率分析的 entry（Q1/Q2/Q3、F1/F3/F4/F5）。

**Neurocomputing 版的新实验（PAPER_PLAN §13.8–13.9，规则在运行前已写定）。** 都在已有 bank 上运行；顺序是先开发集、发回检查，再测试：

```bash
# 1. ViT-B/16 开发集（只需 CPU；三个数据集各一次）
bash fdrfreq/scripts/dev_extra.sh cub200       # E1–E4、E10 选参，E7 开发集标准误，E8 误差 profile，E9 预算预测，E11 耗时
bash fdrfreq/scripts/dev_shots.sh cub200       # E5：K = 2、3、10 的选参
# 2. ViT-L/14（GPU 提特征，之后只需 CPU）
bash fdrfreq/scripts/dev_backbone.sh cub200 ../datasets 0   # 提特征 → parity（mini 上有闸门）→ 选参 → profile → kpredict
# 3. 检查通过后：每个配置只测试一次（evaluate-once 拒绝重复评估，并写 TEST_LOG.jsonl）
bash fdrfreq/scripts/final_extra.sh cub200
bash fdrfreq/scripts/final_shots.sh cub200
bash fdrfreq/scripts/final_backbone.sh cub200
# 4. 三个数据集都测完后：主比较族的 Holm 校正
bash fdrfreq/scripts/holm.sh                   # ViT-L/14：TAG=vitl14 bash fdrfreq/scripts/holm.sh
```

`select` 不会重新锁定已经有测试结果（`eval_<entry>_shot<K>.json`）的条目，所以误重跑 dev_* 脚本不会改动测过的配置。

## 命令

| 命令 | 作用 | 会读测试集吗 |
|---|---|---|
| `python -m fdrfreq.extract ...` | 建 bank：原图、Gaussian 模糊梯度（σ=1,2,4）、flip/gray 对照，patch-token DCT 频带能量，BiMC 文本，关键词 grounding，各 seed 的 support 下标 | 编码，但不评测 |
| `python -m fdrfreq.run parity` | 缓存版 BiMC / BiMC† 的逐 session 精度，用来和 engine 结果对齐 | 是（只用于基线） |
| `python -m fdrfreq.run diagnose` | 频率掩码诊断、D0 top-k recall、D2 互补性、D3 信息上界 | 否 |
| `python -m fdrfreq.run select` | 在伪增量 episode 上选参并锁定配置（`locked_*.json`） | 否 |
| `python -m fdrfreq.run probe` | 协议保真度：固定参考配置在两种伪增量划分（random / contiguous）上的 base / novel 变化方向；加 `--with-test` 时并列给出基准上的同一组固定配置 | 仅 `--with-test` 时 |
| `python -m fdrfreq.run frontier` | 以某个锁定配置为起点，按分解轴扫强度 α 与通道预算 q，输出 ΔLast / ΔBase / ΔNovel / HM 与实际改动的通道数（base–novel 权衡曲线）；`--axes prior:var prior:random …` 配合 `--k` 扫按基类统计量排序的通道集合（`prior:ebmask` 时 `--k` 为阈值 c） | 否 |
| `python -m fdrfreq.run errconc` | 原型误差集中度：K-shot 原型误差落在频率掩码、借来的掩码、类无关掩码（方差 / 低方差 / 幅值 / Fisher）、随机掩码、关键词掩码中的占比；同尺寸掩码的 Jaccard 重叠；捕获率–通道预算曲线（含 oracle 上界） | 否 |
| `python -m fdrfreq.run summary` | 各 entry 选出的开发集指标（Last / B / N / HM / 改动通道数 / 三个划分 / 选出的超参） | 否 |
| `python -m fdrfreq.run gates` | 根据 select / diagnose 输出给出各闸门的 PASS / FAIL | 否 |
| `python -m fdrfreq.run evaluate` | 只接受 `locked_*.json`；文件被改过（hash 不符）或来自其他 bank 都会拒绝 | 是 |
| `python -m fdrfreq.run evaluate-once` | §13 起所有新测试都走它：条目已有测试结果就拒绝（`--skip-existing` 改为跳过），每次评估向 `TEST_LOG.jsonl` 追加名称、配置 hash、锁定与评估时间、种子 | 是 |
| `python -m fdrfreq.run profile` | E8：固定预算（默认 10% / 25%）下各通道排序（方差、低方差、幅值、Fisher、APE、随机、oracle、关键词）捕获的原型误差，到 k=1.0 的捕获曲线，基类与新类方差排序的 Spearman / Jaccard，Fisher 与方差排序的相关，class-episode 数；默认排除 support 图像 | 否 |
| `python -m fdrfreq.run kpredict` | E9：Corollary 1 给出的 k*(K)（单一 t² 与逐通道 t² 两种），`--locked` 时 r、τ 取自锁定配置 | 否 |
| `python -m fdrfreq.run devse` | E7：锁定配置在开发 episode 上的均值 ± 标准误，以及相对 `--ref` 的配对差 | 否 |
| `python -m fdrfreq.run timing` | E11：每个新类的重组耗时（ms） | 否 |
| `python -m fdrfreq.errtypes` | 已存测试预测的误差类型（新→新、新→基、基→新、基→基，同 / 跨 session），不做新的测试评估 | 读 pred_*.npz |
| `python -m fdrfreq.report` | 主表、与 anchor 的配对比较（bootstrap CI、seed 计数、fixed/broken、McNemar）、逐 session 表；`--hier-json` 写出每对比较的两种 bootstrap（种子内；先抽种子再抽图像）、种子级均值 ± 标准差和 p 值；`--holm FILES --family A:B ...` 对整个族做 Holm 校正 | 读 evaluate 输出 |
| `python -m fdrfreq.run smoke` | 合成 bank 上的端到端 CPU 冒烟测试 | — |

## 配置矩阵 `configs/paper.json`

每个 entry 对应论文表中的一行。`chain` 表示以同一输出目录下另一个 entry 的锁定配置为起点，不带 `grid` 的 entry 是匹配对照，直接复用它所对照方法的锁定超参。选参目标是伪增量最后一个 session 的精度，约束为 base / novel 精度都不比 fallback（该组件关闭时的配置）低 0.2pp 以上。grid 从弱到强排列，打平时保留较弱的设置。当前协议用 `--objective hm --tolerance 100`（即不加 base / novel 约束，按伪增量最后一个 session 的 HM 选参）。`include_fallback: true` 把关闭该组件的配置也作为候选；`fallback: "chain"` 表示关闭点就是 chain 起点的锁定配置本身。

| 组 | entry | 内容 |
|---|---|---|
| 基线 | ZS, ZS_desc, Proto, TEEN, FDR_tf_published, BiMC, BiMC_ens, LA | 零样本、原型、TEEN、FDR 免训练复现（发表超参）、BiMC、BiMC†、常数 logit 调整 |
| 锚点 | A1, A2 | 池化类内协方差 GDA（只在 base 估计并冻结）与 BiMC† 融合；A2 加入 shot 数修正（debias / predictive 两种形式） |
| FDR 阳性对照 | K0, K1, K1_fdrcoord, K2_random | K0：文本供体权重；K1：关键词分段（最终 CLIP 坐标）；K1_fdrcoord：FDR 原始坐标；K2：随机分段 |
| 视觉频率 grounding | Q1, Q2_classperm, Q3_random, Q5_common, Q4, Q4_shuffled | Q1：模糊梯度通道掩码分段；Q2 / Q3 / Q5：换类掩码 / 随机掩码 / 基类平均的类无关掩码对照；Q4：用频率设置关键词段的强度，Q4_shuffled 为打乱对照 |
| 频带信息 | F1–F8, R1 | F1：patch-DCT 频带块进入联合 GDA；F2：块对角；F3：增强视图；F4：噪声；F5：随机频带；F6：像素 DoG；F7 / F8：换层；R1：频带经验贝叶斯重组 |
| 分解轴研究 | AX_* | 在 K0 之上以 FDR 发表强度（α=0.2、q=37、r=29、τ=22）固定比较六种分解轴，不经过选参，只用于分解轴表（T5） |
| 最终方法（§12.11） | M4a 及其对照 M4a_* 、插件 P4a_* | 见下文"方差引导分段重组" |
| §13 新增 | DCR_ens, S1_soft, M4a_ape, M4a_apelow, M4a_k01, M4a_k02, ULA_fb, BiMC_rt, BiMC_rt_D | DCR_ens：不带 anchor 的 BiMC† + DCR（主表 "Ours" 行，E4）；S1_soft：只用 σ̄² 的软收缩（E1）；M4a_ape / M4a_apelow：按 APE 准则替换最看重 / 最不看重的通道（E2）；M4a_k01 / k02：1-shot 固定预算 0.1 / 0.2（E3，事后分析）；ULA_fb：带关闭选项的 ULA（E10a）；BiMC_rt：重调的 BiMC†，BiMC_rt_D：其上的 DCR（E10b，开发集增益 ≥ 0.3 才跑） |

## 代码结构

```
fdrfreq/
  bank.py       特征库读写（.npy 目录 + meta.json，mmap 读取）
  protocol.py   基准 session、engine 采样器复刻、base-only 伪增量 episode
  pipeline.py   统一管线：BiMC 参考、重组、锚点、融合（强度为 0 时精确关闭）
  gda.py        池化类内 LDA，含 debias / predictive 两种 shot 修正
  recompose.py  整体收缩与分段重组（TEEN / BiMC / FDR 形式）
  keywords.py   不依赖 LLM 的关键词抽取、关键词库（只由当前协议的 base 类定义）
  freqmask.py   视觉频率 grounding 掩码与诊断
  dctband.py    patch-grid DCT 频带、额外特征块构造
  select.py     网格选参、约束、锁定
  diagnose.py   D0–D3 与掩码诊断
  metrics.py    FSCIL 指标、McNemar、配对 bootstrap、两层 bootstrap、Holm、fixed/broken
  report.py     论文表格（Markdown / LaTeX）、配对统计、Holm 表
  profile.py    误差 profile（E8）、方差迁移、Corollary 1 预算预测（E9）、耗时（E11）
  errtypes.py   测试预测的误差类型分解
  extract.py    唯一依赖 torch 的模块
  synthetic.py  合成 bank（测试与冒烟用）
```

`select` 和 `diagnose` 支持 `--split-mode random|contiguous`；`select` 支持 `--tolerance`（统一覆盖 base / novel 容差）和 `--objective last|hm|avg`（选参目标）。

## 已知限制

- **extract.py。** ViT-B/16 的三个 bank 已在 GPU 上建好并通过 parity。ViT-L/14 是第一次运行：dev_backbone.sh 只提原图特征（不提 blur、DCT、增强视图和关键词），在 miniImageNet 上把 BiMC / BiMC† 的 5-shot parity 与已发表值（94.46 / 94.46）比较，相差超过 0.5 就停下（`PARITY_OK=1` 只在查明原因后使用）。脚本在第一个 batch 上的自检仍然有效：自定义前向必须与 `encode_image` 一致。
- **support 来源。** support 下标直接取自各 seed 的 `DatasetManager`，并核对复刻采样器与之是否一致，结果记录在 `meta.json` 的 `support_replica_matches_engine`。标准 CEC 划分可通过 `--standard-manifest` 存为 seed 0。
- **FDR 原始坐标掩码。** 只在文本宽度等于嵌入维度时可用（ViT-B/16、ViT-L/14 满足）。

### 方差引导分段重组（M1 / M2 / M3，§12.7–12.8）
M4a 是冻结的最终方法（§12.11）：A2 锚点 + 无整体收缩 + 用视觉相似度选出的基类供体（`scope=global`、`sim=visual`）替换方差最大的 k 个通道，α ∈ {0.7, 1}；M4a_* 为匹配预算对照与消融，P4a_Proto / P4a_BiMC 为插件实验。M4 为其上一版（文本相似度供体、α=1），M3 更早（供体在通道内排序）。donor 是 top-r 个单位基类原型的 softmax 加权平均，本身不归一化，只有重组后的向量做归一化。`k` 与 `r` 取浮点 ≤ 1 时分别表示维度比例与基类比例（1.0 = 全部）。

`segments.type = "prior"`：所有新类共用一个只由基类数据决定的通道集合。
- `rank`：`var`（合并类内方差 top-k，本文方法）、`lowvar`、`mag`、`fisher`、`random`、`freq`、`ebratio`、`ebmask`；§13 新增 `ape` / `ape_low`（APE 官方代码的免训练准则 −0.7·S_j + 0.3·V_j：S_j 为不同类样本在通道 j 上乘积的均值，样本是基类文本嵌入与基类训练图像；V_j 为基类文本权重在通道 j 上的方差。取准则最高 / 最低的 k 个通道）和 `softvar`（不选通道，而是逐通道权重 w_j = σ̄_j² / (σ̄_j² + c·median σ̄²)，只支持 `scope=global`）。
- `k`：通道预算，<1 时为维度比例。
- `scope`：`segment`（供体在通道内排序，默认）或 `global`。
- `ebmask` 用 `c` 和 `s2="empirical"`：当实测 K-shot 误差 s_j²(K) > c·t_j² 时借用通道 j。
`segments.type = "eb"` 的 `s2` 可取 `analytic`（σ²/(K g²)）或 `empirical`（在基类上抽 K 张图实测）。
