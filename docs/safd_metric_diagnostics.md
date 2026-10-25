# CUB SAFD metric V2：服务器诊断命令

使用新增的 `tools/diagnose_safd_metric.py`。先将该脚本同步至服务器，并使用原来运行 BiMC 的 Python 环境。保留原报告与缓存，输出使用新目录。

这些命令只读取基类缓存，不加载 CLIP，不重新做图像编码，不读取正式测试集，也不导出新分类模型。学习模式会重新执行原始 A 类拟合轨迹；所有增益扫描只作用于诊断评分，不会分别用各增益重新训练。诊断结果不能作为调过参数后的独立测试结果。

## 1. 检查文件与环境

在 Bash 中执行。`set -euo pipefail` 让任一命令或管道失败时停止。

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RUN_ROOT=outputs/safd_metric_v2_20261022
export DIAG_ROOT="$RUN_ROOT/diagnostics_cub_20261025"
mkdir -p "$DIAG_ROOT/logs"

ls -lh "$RUN_ROOT/cache/cub200_safd.npz" \
       "$RUN_ROOT/cache/cub200_safd.json" \
       "$RUN_ROOT/cache/cub200_safd.support.json" \
       "$RUN_ROOT/models/cub200_closed_form.training.json" \
       "$RUN_ROOT/models/cub200_safd.training.json" \
       "$RUN_ROOT/models/cub200_original.training.json"

python tools/diagnose_safd_metric.py --help
```

每个 `--output-dir` 必须不存在。若重跑，修改 `DIAG_ROOT` 后缀；不要覆盖失败记录。三份缓存文件必须一起保留，缓存读取器会检查各自校验和。`.training.json` 是训练协议报告，不能用模型 `.json` 代替。

## 2. 优先重放闭式退化

原报告中非零强度均为 0%。该命令复现这些强度，同时检查更小强度和评分增益。共 6 个强度 × 8 个增益，复用相同 200 个伪增量阶段，不会重复拟合 48 次。

```bash
python -u tools/diagnose_safd_metric.py \
  --cache "$RUN_ROOT/cache/cub200_safd.npz" \
  --training-report "$RUN_ROOT/models/cub200_closed_form.training.json" \
  --mode closed_form \
  --strengths 0 0.001 0.01 0.1 1 10 \
  --gains 0 0.01 0.03 0.1 0.3 1 3 16 \
  --device cpu --threads 1 --save-score-probe \
  --output-dir "$DIAG_ROOT/closed_form" \
  2>&1 | tee "$DIAG_ROOT/logs/closed_form.log"
```

## 3. SAFD 学习轨迹与早期更新

保留原始训练 gain=16、优化器和 episode 顺序。只改变诊断时的评分 gain，观察同一组已学系数是否因残差过强而退化。没有在这些结果上重新选择或导出模型。

```bash
python -u tools/diagnose_safd_metric.py \
  --cache "$RUN_ROOT/cache/cub200_safd.npz" \
  --training-report "$RUN_ROOT/models/cub200_safd.training.json" \
  --mode learned \
  --checkpoints 0 1 5 10 20 50 100 1000 \
  --gains 0 0.03 0.1 0.3 1 16 \
  --device cpu --threads 1 --save-score-probe \
  --output-dir "$DIAG_ROOT/safd_learned" \
  2>&1 | tee "$DIAG_ROOT/logs/safd_learned.log"
```

## 4. 原图度量对照

复用同一缓存，但按原图度量训练报告使用 `original_within` 基，帮助区分一般度量融合问题与 SAFD 特有的问题。

```bash
python -u tools/diagnose_safd_metric.py \
  --cache "$RUN_ROOT/cache/cub200_safd.npz" \
  --training-report "$RUN_ROOT/models/cub200_original.training.json" \
  --mode learned \
  --checkpoints 0 1 5 10 20 50 100 1000 \
  --gains 0 0.03 0.1 0.3 1 16 \
  --device cpu --threads 1 --save-score-probe \
  --output-dir "$DIAG_ROOT/original_learned" \
  2>&1 | tee "$DIAG_ROOT/logs/original_learned.log"
```

## 5. 应检查与提供的结果

每个诊断目录包含：

- `checks.json`：缓存行顺序、A/V 协议与全局/局部标签一致性，以及参考指标重放；参考指标误差超过默认 0.01pp 会停止扫描。
- `summary.json`：全部候选的 Base/Novel/H、纠正/破坏数量、增益扫描、原报告 checkpoint 的复现误差。
- `matrix_diagnostics.json`：矩阵特征值范围、正定性、条件数、偏离恒等的程度。
- `stage_details.json`：每阶段的分组准确率、预测变化、旧新类误判计数。
- `training_curve.json`：学习模式的全训练 episode 平均 CE、分组准确率和系数大小；闭式模式为空列表。
- `score_probe.npz`：首阶段伪旧类/伪新类各最多 16 个查询的原参考分数、残差、标签/样本身份、原型与矩阵快照。
- `provenance.json`：输入与源码哈希、参数、Python/PyTorch/NumPy 版本。
- 若失败，包含 `failure.json`；保留它与日志，不要通过放宽检查掩盖不一致。

先检查 `summary.json` 的 `all_compared_checkpoints_within_tolerance`。使用上述命令应有可比较的原报告候选；若为 false，先定位缓存、代码、设备或版本的复现差异，不直接解释增益曲线。

`fraction_pair_change_exceeds_margin` 表示原第一/第二名之间的残差差值幅度超过原 margin 的比例；`fraction_reference_pair_reversed` 表示该对次序反转比例。其他类别也可能成为第一名，最终变化仍以 `changed/corrected/damaged` 为准。所有数量是跨阶段重复出现的查询观测，不是独立图像数或独立统计重复。

判断顺序：矩阵与分数是否有效 → 原 gain=16 是否复现退化 → 小 gain 是否减少破坏 → SAFD 与原图度量是否同样退化 → 训练平均 CE 改善是否伴随跨类验证变差。即使小 gain 恢复准确率，也不能直接认定有 SAFD 增益，应与 gain=0 的同一验证参考比较。

完成后打包全部诊断目录，文件体积远小于原始缓存：

```bash
tar -czf "${DIAG_ROOT}.tar.gz" \
  -C "$RUN_ROOT" "${DIAG_ROOT##*/}"
```

提供该压缩包即可继续分析。无需再次传大体积特征缓存，除非重放检查本身失败且诊断数据不足。
