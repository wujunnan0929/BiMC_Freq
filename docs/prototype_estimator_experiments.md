# 性能优先 V3：诊断、条件原型估计与冻结评估

更新：2026-10-26。复用已提取的 SAFD 基类缓存，主方法无需频率查询分支。

## 1. 实现范围

已实现：

- **P0**：完整 BiMC 参考、81 组分支温度/旧新偏置候选、仅组偏置控制、等描述数量控制。
- **P1/P2**：独立较多样本教师原型、逐查询专家正确性并集。这两项是使用额外信息的诊断，不是可部署方法，不参与模型选择。
- **筛查补充**：支持/教师样本银行 × 开启/关闭原 BiMC 视觉校准的四组对照；各单专家、相对校准参考的修正/破坏计数及旧/新类专家互补性。
- **P3**：同参数预算的 `center` 原型回归、`boundary` 边界监督条件估计器，以及 `boundary + SAFD` 训练辅助。
- **P4**：可选 FFT/普通增强辅助，要求缓存行与普通原图严格对齐。
- 正式运行时加载冻结模型，只在每个新类到达时估计一次原型。旧原型与模型参数保持冻结；不保存历史逐样本支持特征；查询只编码原图。

默认隐藏宽度 32 时模型有 1794 个梯度参数。只在基类训练，不应称主方法 training-free。校准对照没有梯度训练，但使用基类验证选择参数。

## 2. 协议与评分

将原始基类分成 A/V 类别，固定保留 A 的验证查询；选择阶段只用 A 的拟合池做梯度训练。V 的支持/查询用于验证选择，V 的固定教师银行只供诊断，不参与选择阶段的网络拟合或 checkpoint 选择。

CUB 常见每类 30 张：A 保留验证查询 5 张，剩余 25 张可采样 `S=5、T=15、Q=5`。训练 S/T/Q 互不重叠。也会采样 2-shot 任务；旧类支持最多 20 张。旧新类数从可用的 `(10,20,30) × (5,10)` 合法组合采样；类数不足时不会把合法 5-way 新类任务挤成 1-way。

本轮全局预留 V 教师 15 张，改变了相对旧 metric 实验的查询/支持抽样。应比较本轮同一协议产生的参考，不直接沿用旧的 83.8182 H。

`center` 只优化独立教师中心目标，不读取竞争类几何；`boundary` 加入独立查询上的教师类间间隔监督与旧类正确间隔保护。修正在原型空间完成，不加任意尺度的 logit 残差。估计器使用标量几何，规则与特征坐标旋转、支持排列及类别重新排序无关。

验证指标来自完整 BiMC 评分。温度和组偏置先在基类验证选择；选择模型 checkpoint/blend 时，同时满足相对原 BiMC、相对校准参考的平均旧/新类下降各不超过 0.2pp。并列保留更早步骤及较小改变量，包括零步/零修正。

选择结束后，重新初始化模型和优化器，在全部原始基类上按已选步数重拟合，再导出冻结模型。这时可重新使用 V 及其教师银行中的原始基类图像。改变 V 教师特征不会影响此前的模型选择（有回归测试），但可能改变后续全基类重拟合参数。

`boundary + SAFD` 只在 25% 的伪新类训练支持集中替换一张图的特征，使用已缓存的粗/中尺度视图；教师与查询保持原图。辅助采样使用独立 RNG，与无辅助控制的 S/T/Q、类别和 episode 顺序保持匹配。

## 3. Ubuntu 准备

同步当前代码中的 `models/`、`engine/`、`tools/`、`main.py` 和 `configs/` 新改动，继续使用原 BiMC 环境。下面使用已知服务器项目路径。

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CACHE=outputs/safd_metric_v2_20261022/cache/cub200_safd.npz
export V3=outputs/prototype_estimator_v3_20261025
export DATA_ROOT=../datasets/
mkdir -p "$V3/logs" "$V3/models"

ls -lh "$CACHE" "${CACHE%.npz}.json" "${CACHE%.npz}.support.json"
python tools/screen_prototype_space.py --help
python tools/train_prototype_estimator.py --help
```

缓存三件套必须保留。无需重新提取 CUB 特征，也不调用新的 LLM。正式训练不接受不完整缓存。各筛查输出目录、模型 NPZ/JSON/training.json 必须是新路径；重跑应更换 `V3`，避免覆盖已有证据。

三个主训练对照复用已有 SAFD 缓存以节省开发工作；`center`/`boundary --auxiliary none` 只使用其中的原图列。报告成本时，区分共享缓存的实际提取成本和不使用辅助视图的方法所必需的独立运行成本。

## 4. 先跑性能空间筛查

```bash
for FIT_SEED in 2 3 4; do
  python -u tools/screen_prototype_space.py \
    --cache "$CACHE" --seed "$FIT_SEED" \
    --sequences 20 --teacher-shot 15 --device cpu --threads 1 \
    --output-dir "$V3/screen_seed${FIT_SEED}" \
    2>&1 | tee "$V3/logs/screen_seed${FIT_SEED}.log"
done
```

先查看各 `summary.json`：

- `baseline` / `calibrated_reference`：校准本身带来多少改善。
- `group_bias_only`：一般旧新类偏置控制。
- `equal_description_count_control`：仅 max-description 分支统一数量；语义均值原型仍保持原定义。
- `teacher_prototype_diagnostic.metrics`：旧版本保留字段，使用未经视觉校准的教师原型；与开启视觉校准的参考比较时同时改变样本银行和校准机制，不能单独归因为样本数改善。
- `expert_union_diagnostic.metrics`：已有专家是否有互补空间。
- `prototype_factorial_diagnostics`：在固定分支温度、组偏置、旧原型、协方差和文本下，分别隔离视觉校准与教师银行变化。`support_guided` 是校准参考；`teacher_guided` 对更多教师样本应用完全相同的视觉校准。
- `expert_profile_diagnostics`：完整原 BiMC、校准 BiMC、融合原型、协方差、max-description 的单专家性能，以及每对专家的旧/新类正确性交集。每个单专家均在全部已见类上预测。

教师和专家并集明确 `deployable=false, selection_candidate=false`，不能填成正式方法结果。新补充的所有对照都不自动参与模型选择。先比较相同视觉校准下的教师收益、关闭视觉校准的支持集控制、最强单专家与并集的差距，再决定是否执行 P3。教师银行独立于支持集，并非嵌套增加图像，其结果也不是严格性能上界。

旧 `baseline` 等字段的修正/破坏计数继续以原 BiMC 为参考；新四组对照和单专家表的计数以校准 BiMC 为参考，具体对照差值中的计数以该对照声明的 `control` 为参考。计数单位是可重复的 stage-query 观测，不是去重图像。

## 5. 三个配对训练对照

**2026-10-26：收到三份 CUB 筛查后，建议先执行第 9 节补充诊断，再决定是否投入本节训练与第 6 节正式矩阵。现有教师比较受视觉校准混杂影响，专家并集也缺少单专家对照。**

首轮可先令 `FIT_SEED=2`；确认流程及验证信号后，再对 3/4 执行相同命令。不要根据正式测试集挑拟合种子。

```bash
FIT_SEED=2

python -u tools/train_prototype_estimator.py \
  --cache "$CACHE" --method center --auxiliary none \
  --seed "$FIT_SEED" --sequences 20 --teacher-shot 15 \
  --steps 300 --episode-pool 100 --validate-every 50 \
  --device cpu --threads 1 \
  --output "$V3/models/cub200_center_fit${FIT_SEED}.npz" \
  2>&1 | tee "$V3/logs/center_fit${FIT_SEED}.log"

python -u tools/train_prototype_estimator.py \
  --cache "$CACHE" --method boundary --auxiliary none \
  --seed "$FIT_SEED" --sequences 20 --teacher-shot 15 \
  --steps 300 --episode-pool 100 --validate-every 50 \
  --device cpu --threads 1 \
  --output "$V3/models/cub200_boundary_fit${FIT_SEED}.npz" \
  2>&1 | tee "$V3/logs/boundary_fit${FIT_SEED}.log"

python -u tools/train_prototype_estimator.py \
  --cache "$CACHE" --method boundary --auxiliary safd \
  --auxiliary-probability 0.25 \
  --seed "$FIT_SEED" --sequences 20 --teacher-shot 15 \
  --steps 300 --episode-pool 100 --validate-every 50 \
  --device cpu --threads 1 \
  --output "$V3/models/cub200_boundary_safd_fit${FIT_SEED}.npz" \
  2>&1 | tee "$V3/logs/boundary_safd_fit${FIT_SEED}.log"
```

默认除每 50 步外，还验证第 1/5/10/20 步。模型 `*.training.json` 保存参考、校准候选、所有 checkpoint/blend、拟合划分、S/T/Q 索引、辅助替换和重拟合轨迹。

重点比较 **相对 calibrated_reference 的 Novel/H**。如果最佳步骤为 0、blend 为 0，应记录为没有估计器增益；即便完整方法优于 BiMC，也可能完全来自校准。

如需快速接线测试，可以减少 steps/episode-pool/sequences；改变查询数应传 `--query`，它会同步训练与验证查询配置。小规模配置不能与默认正式候选混为同一实验。

## 6. 正式 CUB 配对测试

先打印计划，默认不执行。三个模型必须来自同一拟合种子、缓存、A/V 协议和校准；启动器会拒绝混用不同拟合种子。

```bash
FIT_SEED=2
python tools/run_prototype_estimator_experiments.py \
  --dataset cub200 --seeds 1 2 3 4 5 \
  --model center="$V3/models/cub200_center_fit${FIT_SEED}.npz" \
  --model boundary="$V3/models/cub200_boundary_fit${FIT_SEED}.npz" \
  --model boundary_safd="$V3/models/cub200_boundary_safd_fit${FIT_SEED}.npz" \
  --reference-model boundary \
  --support-root outputs/prototype_estimator_support \
  --output-root "$V3/benchmark_fit${FIT_SEED}" --dry-run \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;'
```

固定模型与超参数后执行：

```bash
python tools/run_prototype_estimator_experiments.py \
  --dataset cub200 --seeds 1 2 3 4 5 \
  --model center="$V3/models/cub200_center_fit${FIT_SEED}.npz" \
  --model boundary="$V3/models/cub200_boundary_fit${FIT_SEED}.npz" \
  --model boundary_safd="$V3/models/cub200_boundary_safd_fit${FIT_SEED}.npz" \
  --reference-model boundary \
  --support-root outputs/prototype_estimator_support \
  --output-root "$V3/benchmark_fit${FIT_SEED}" --execute \
  --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID '0;'
```

共有 6 变体 × 5 支持种子 = 30 次正式评估：

| 名称 | 含义 |
|---|---|
| baseline | 完整原 BiMC |
| zero_full | 模块加载但 blend=0 且关闭校准，必须精确复现原 BiMC |
| calibrated | blend=0，仅使用已选温度/组偏置，是主方法的强参考 |
| center | 普通教师中心回归 |
| boundary | 边界监督条件估计器 |
| boundary_safd | 同一估计器加入 SAFD 支持扰动 |

首个支持种子接线评估可把 `--seeds` 改为 `1`，但不得据正式测试结果继续调参。不同拟合种子的矩阵使用不同 `benchmark_fit${FIT_SEED}` 目录。五个支持种子不能替代训练随机性分析；两种随机性应分开报告。

summary 同时保存相对 baseline 和相对 calibrated 的配对差值。正式推理时间不包含离线缓存/拟合成本；模型 sidecar 和结果报告另记录这些成本。`retained_tensor_bytes` 包括估计器参数与额外原型等保留张量，但不是峰值内存。

## 7. FFT/普通增强辅助控制

只有 `boundary` 有增益后，再检验 SAFD 是否有独立作用。额外缓存必须使用相同数据路径和缓存采样种子，行顺序/CLIP/文本/预处理会核验。

```bash
mkdir -p "$V3/cache"
for SOURCE in fft augmentation; do
  python tools/cache_safd_base_features.py \
    --dataset cub200 --data-root "$DATA_ROOT" --seed 1 \
    --source "$SOURCE" --gpu 0 --device cuda --batch-size 32 --workers 4 \
    --output "$V3/cache/cub200_${SOURCE}.npz"
  python tools/train_prototype_estimator.py \
    --cache "$CACHE" --aux-cache "$V3/cache/cub200_${SOURCE}.npz" \
    --method boundary --auxiliary "$SOURCE" --auxiliary-probability 0.25 \
    --seed "$FIT_SEED" --sequences 20 --teacher-shot 15 --steps 300 \
    --device cpu --threads 1 \
    --output "$V3/models/cub200_boundary_${SOURCE}_fit${FIT_SEED}.npz"
done
```

两个来源都编码两张辅助视图，但不是失真强度匹配。正式比较可向启动器额外添加 `--model boundary_fft=...` 和 `--model boundary_augmentation=...`。

## 8. 其他数据集与结果回传

CIFAR-100/mini-ImageNet 使用同样三个 CLI。先选各自完整基类缓存，如没有缓存则用 `tools/cache_safd_base_features.py --dataset cifar100` 或 `--dataset miniimagenet` 提取；随后替换 `CACHE`、模型名称和评估 `--dataset`，保持各数据集独立拟合/选择。不把 CUB 模型用于另一个数据集。

请先提供筛查目录的 `summary.json`、`calibration.json`，以及训练模型的 `*.json` / `*.training.json`。可打包全部开发结果而不包含大缓存：

```bash
tar -czf "$V3/cub_development_results.tar.gz" \
  -C "$V3" screen_seed2 screen_seed3 screen_seed4 models logs
```

模型 NPZ 本身体积较小，建议一起保存。正式评估还应保留 `summary.json` 和各运行的配置、支持清单、日志、metrics、predictions。

新方案尚无完整 benchmark 结果。小样本真实 CLIP 测试只验证缓存、训练、导出和多会话接线，不能作为准确率提升证据。

## 9. CUB 三份筛查后的补充诊断

`summary2/3/4.json` 的均值：原 BiMC H=82.2538，校准参考 H=82.4127，未经视觉校准的 15-shot 教师 H=82.8530。教师相对校准参考新类 +3.9322pp、旧类 -2.1700pp、H +0.4404pp，三个划分均超出旧类允许下降 0.2pp 的保护范围。专家正确性并集 H=91.4928，但使用查询标签选专家，不能视为可部署的 +9.0801pp 改善。等描述数控制 H 变化平均 -0.0035pp。

因此先复用相同缓存和划分参数补齐诊断，暂不扩大原型网络训练。同步 `engine/prototype_evaluation.py` 和 `tools/screen_prototype_space.py` 后运行：

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CACHE=outputs/safd_metric_v2_20261022/cache/cub200_safd.npz
DETAILS=outputs/prototype_estimator_screen_details_20261026
mkdir -p "$DETAILS/logs"

for S in 2 3 4; do
  python -u tools/screen_prototype_space.py \
    --cache "$CACHE" --seed "$S" --sequences 20 --teacher-shot 15 \
    --device cpu --threads 1 --output-dir "$DETAILS/seed${S}" \
    2>&1 | tee "$DETAILS/logs/seed${S}.log"
done

tar -czf "${DETAILS}.tar.gz" -C outputs "${DETAILS##*/}"
```

新目录各自保存 `summary.json`、`prototype_controls.json`、`expert_profiles.json`、`calibration.json`、`group_bias_control.json`、`protocol.json`、`provenance.json`。只重放缓存，不进行梯度训练、CLIP/SAFD 重新编码或正式测试。旧三份文件未记录 seed，以上 2/3/4 延续此前运行计划；应使用原运行的 `provenance.json` 核实。缓存与参数相同时 `protocol_sha256` 应与对应旧文件一致；不一致时不能当作严格配对复现。

阅读顺序与后续分支：

1. `prototype_factorial_diagnostics.contrasts.disable_guidance_support`：若仅关闭校准就能稳定取得收益，先把它作为简单控制，不将其归功于原型估计器。
2. `teacher_vs_support_guided` / `teacher_vs_support_raw`：在相同校准状态下看教师改善是否保留，及是否仍明显伤害旧类。仅新类改善不足以支持整套 FSCIL 方法。
3. `expert_profile_diagnostics.experts`：看固定单专家是否已经接近标签并集，尤其关注协方差专家在新类上独立使用的结果。若单专家足够强，优先验证固定融合/分组权重，作为后续可学习模块的强对照。
4. 单专家都不足但旧/新类互补性稳定时，再考虑只在基类拟合的置信度/竞争类条件融合。预测规则不能读取查询真实类别，也不能用正式测试集挑阈值。
5. 在无辅助主方法有稳定收益之后，再比较 SAFD、FFT、普通增强辅助；当前筛查未检验 SAFD 的独立贡献。

三份摘要属于 CUB 基类开发筛查，不是正式增量测试。200 个 stage 共享图像和类别，不作 200 次独立统计重复，不据此计算显著性或承诺期刊录用。
