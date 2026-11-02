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

## 命令

| 命令 | 作用 | 会读测试集吗 |
|---|---|---|
| `python -m fdrfreq.extract ...` | 建 bank：原图、Gaussian 模糊梯度（σ=1,2,4）、flip/gray 对照，patch-token DCT 频带能量，BiMC 文本，关键词 grounding，各 seed 的 support 下标 | 编码，但不评测 |
| `python -m fdrfreq.run parity` | 缓存版 BiMC / BiMC† 的逐 session 精度，用来和 engine 结果对齐 | 是（只用于基线） |
| `python -m fdrfreq.run diagnose` | 频率掩码诊断、D0 top-k recall、D2 互补性、D3 信息上界 | 否 |
| `python -m fdrfreq.run select` | 在伪增量 episode 上选参并锁定配置（`locked_*.json`） | 否 |
| `python -m fdrfreq.run probe` | 协议保真度：固定参考配置在两种伪增量划分（random / contiguous）上的 base / novel 变化方向；加 `--with-test` 时并列给出基准上的同一组固定配置 | 仅 `--with-test` 时 |
| `python -m fdrfreq.run frontier` | 以某个锁定配置为起点，按分解轴扫强度 α 与通道预算 q，输出 ΔLast / ΔBase / ΔNovel / HM 与实际改动的通道数（base–novel 权衡曲线） | 否 |
| `python -m fdrfreq.run errconc` | 原型误差集中度：K-shot 原型误差落在频率掩码、借来的掩码、类无关掩码、随机掩码、关键词掩码中的占比 | 否 |
| `python -m fdrfreq.run gates` | 根据 select / diagnose 输出给出各闸门的 PASS / FAIL | 否 |
| `python -m fdrfreq.run evaluate` | 只接受 `locked_*.json`；文件被改过（hash 不符）或来自其他 bank 都会拒绝 | 是 |
| `python -m fdrfreq.report` | 主表、与 anchor 的配对比较（bootstrap CI、seed 计数、fixed/broken、McNemar）、逐 session 表 | 读 evaluate 输出 |
| `python -m fdrfreq.run smoke` | 合成 bank 上的端到端 CPU 冒烟测试 | — |

## 配置矩阵 `configs/paper.json`

每个 entry 对应论文表中的一行。`chain` 表示以同一输出目录下另一个 entry 的锁定配置为起点，不带 `grid` 的 entry 是匹配对照，直接复用它所对照方法的锁定超参。选参目标是伪增量最后一个 session 的精度，约束为 base / novel 精度都不比 fallback（该组件关闭时的配置）低 0.2pp 以上。grid 从弱到强排列，打平时保留较弱的设置。

| 组 | entry | 内容 |
|---|---|---|
| 基线 | ZS, ZS_desc, Proto, TEEN, FDR_tf_published, BiMC, BiMC_ens, LA | 零样本、原型、TEEN、FDR 免训练复现（发表超参）、BiMC、BiMC†、常数 logit 调整 |
| 锚点 | A1, A2 | 池化类内协方差 GDA（只在 base 估计并冻结）与 BiMC† 融合；A2 加入 shot 数修正（debias / predictive 两种形式） |
| FDR 阳性对照 | K0, K1, K1_fdrcoord, K2_random | K0：文本供体权重；K1：关键词分段（最终 CLIP 坐标）；K1_fdrcoord：FDR 原始坐标；K2：随机分段 |
| 视觉频率 grounding | Q1, Q2_classperm, Q3_random, Q5_common, Q4, Q4_shuffled | Q1：模糊梯度通道掩码分段；Q2 / Q3 / Q5：换类掩码 / 随机掩码 / 基类平均的类无关掩码对照；Q4：用频率设置关键词段的强度，Q4_shuffled 为打乱对照 |
| 频带信息 | F1–F8, R1 | F1：patch-DCT 频带块进入联合 GDA；F2：块对角；F3：增强视图；F4：噪声；F5：随机频带；F6：像素 DoG；F7 / F8：换层；R1：频带经验贝叶斯重组 |
| 分解轴研究 | AX_* | 在 K0 之上以 FDR 发表强度（α=0.2、q=37、r=29、τ=22）固定比较六种分解轴，不经过选参，只用于分解轴表（T5） |

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
  metrics.py    FSCIL 指标、McNemar、配对 bootstrap、fixed/broken
  report.py     论文表格（Markdown / LaTeX）
  extract.py    唯一依赖 torch 的模块
  synthetic.py  合成 bank（测试与冒烟用）
```

`select` 和 `diagnose` 支持 `--split-mode random|contiguous`；`select` 支持 `--tolerance`（统一覆盖 base / novel 容差）和 `--objective last|hm|avg`（选参目标）。

## 已知限制

- **extract.py 尚未在 GPU 上跑过。** 本次开发环境装不了 torch。脚本在第一个 batch 上做了两项自检，任一项不一致都会直接报错：自定义前向必须与 `encode_image` 一致；torch 版 DCT 频带能量必须与 numpy 参考实现一致。建议先加 `--limit-per-class 3` 跑一个冒烟 bank。
- **support 来源。** support 下标直接取自各 seed 的 `DatasetManager`，并核对复刻采样器与之是否一致，结果记录在 `meta.json` 的 `support_replica_matches_engine`。标准 CEC 划分可通过 `--standard-manifest` 存为 seed 0。
- **FDR 原始坐标掩码。** 只在文本宽度等于嵌入维度时可用（ViT-B/16、ViT-L/14 满足）。
