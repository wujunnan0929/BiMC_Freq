# 现有结果打包、离线诊断与有限基类验证

日期：2026-10-21。优先级：先使用已有运行诊断，再决定是否执行补充验证。无需重新运行已完成的 75 次主实验。

## 1. 第一轮需要哪些文件

最小集合为 9 个运行目录：

| 数据集 | 支持集种子 | 方法 |
|---|---|---|
| CUB-200 | 1 | G1、G3、G4 |
| CIFAR-100 | 3 | G1、G3、G4 |
| mini-ImageNet | 3 | G1、G3、G4 |

每个目录保留以下已有文件：

- `metrics.json`、`config.yaml`、`run_spec.json`、`run_state.json`、`support.json`、`run.log`；
- G3/G4 的 `guidance_calibration.json`、全部 `guidance_session_*.json`；
- 三个方法的全部 `predictions_session_*.npz`。

G1 没有 guidance 文件是正常情况。第一轮不用传 `checkpoint.pt`、CLIP 权重、数据集图像或 API key。
若后续需要精确核对 G1→G4 的原型角度，再补对应 checkpoint；JSON 中的 `movement_degrees` 是原始原型→当前引导原型，不能当成 G1→G4。

这几个种子用于诊断，不替代正式五种子的总体结论。若方便，可额外提供 CUB seed1 的 G0 目录，分析原引导的 Base/Novel 交换。

## 2. 立即可执行：打包现有目录

已有 summary 记录的服务器项目路径如下。若项目已经移动，只修改第一行。这个命令不要求同步新脚本。

```bash
cd /home/user/junnan/FSCIL/BiMC_Freq

tar --exclude='checkpoint.pt' --exclude='*.tmp' \
  -czf outputs/frequency_guidance_diagnostics_20261021.tar.gz \
  outputs/frequency_guidance_cub200/cub200/{G1,G3,G4}/seed_1 \
  outputs/frequency_guidance_cifar100/cifar100/{G1,G3,G4}/seed_3 \
  outputs/frequency_guidance_miniimagenet/miniimagenet/{G1,G3,G4}/seed_3

ls -lh outputs/frequency_guidance_diagnostics_20261021.tar.gz
```

使用 Bash 执行以支持花括号展开。若 tar 提示目录不存在，先检查实际目录，不要补跑：

```bash
find outputs -type f -name run_spec.json -path '*frequency_guid*' | sort
```

如本地文件只保留 summary，但服务器仍有完整输出，应在服务器执行打包。
上述目录是代码的标准运行目录，含配置、日志和诊断；如自己向其中放入了其他文件，可用下一节的精确白名单打包。

## 3. 同步新工具后：离线诊断并精确打包

将 `tools/diagnose_frequency_guidance.py` 同步到服务器同路径；在原来的 Python 环境运行，只需要 NumPy，不加载 CLIP、不占用 GPU，也不启动训练。

```bash
python tools/diagnose_frequency_guidance.py \
  --root outputs/frequency_guidance_cub200 \
  --root outputs/frequency_guidance_cifar100 \
  --root outputs/frequency_guidance_miniimagenet \
  --case cub200:1 --case cifar100:3 --case miniimagenet:3 \
  --variants G1 G3 G4 \
  --output outputs/frequency_guidance_diagnosis_20261021 \
  --archive outputs/frequency_guidance_diagnosis_bundle_20261021.tar.gz
```

该工具按 `run_spec.json` 元数据识别运行，保留源结果，只写新的报告和压缩包。
缺文件、重复运行身份、预测样本/标签不匹配等问题会明确报告并返回非零退出码；不会以部分文件假装完成诊断。
即使不完整，也保留可用报告/文件的压缩包，方便提供给分析者。

优先查看报告中的：

1. 运行完整性与同种子支持清单是否一致；
2. G4−G1、G3−G1、G4−G3 各会话预测变化、改对数、改错数，以及 Base/Novel 分组差值；
3. G3/G4 的原型差异、尺度邻居重叠和图文关系相似度；
4. 每个 rho 候选相对 rho=0 的基类验证差值、约束可行性和最终选择。

计数指标只支持定位现象，不能单凭它们断言原因。准确率相同不意味着预测相同。

## 4. 有需要时才执行：CUB seed1 的有限基类控制

这一步需要 GPU，但仅调用原始基类训练图像和伪增量验证；不评估正式测试集。
当前矩阵启动器不能用 `--opts ...GUIDANCE.BASE_ONLY True`，因为该键由矩阵管理、共享完成检查也要求正式测试完成；以下使用 `main.py` 直接调用。

保持旧实验的 Python 环境、数据根路径、CLIP 权重与描述文件。先从旧配置确认 `DATASET.ROOT`，再设置 `DATA_ROOT`：

```bash
sed -n '1,80p' outputs/frequency_guidance_cub200/cub200/G4/seed_1/config.yaml

export DATA_ROOT=/absolute/path/to/the/same/datasets
export GPU=0
export BASE_OUT="outputs/frequency_guidance_base_controls_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"

run_base_control() {
  local name="$1"
  shift
  local out="$BASE_OUT/cub200/$name/seed_1"
  if [ -e "$out" ]; then
    echo "Output already exists; choose a fresh BASE_OUT: $out" >&2
    return 1
  fi
  mkdir -p "$out"
  python -u main.py \
    --data_cfg configs/datasets/cub200.yaml \
    --train_cfg configs/trainers/bimc_frequency_guidance.yaml \
    --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "${GPU};" \
    SEED 1 DATASET.SUPPORT_SEED 1 \
    DATASET.SUPPORT_MANIFEST outputs/frequency_guidance_cub200/cub200/G4/seed_1/support.json \
    OUTPUT_DIR "$out" \
    TRAINER.BiMC.GUIDANCE.CALIBRATION_SEED 10102 \
    TRAINER.BiMC.GUIDANCE.BASE_ONLY True \
    TRAINER.BiMC.RESIDUAL.SAVE_CHECKPOINT False \
    "$@" > "$out/run.log" 2>&1
}

set -e
run_base_control G4
run_base_control G4_permuted TRAINER.BiMC.GUIDANCE.SEMANTIC_PERMUTATION '[2,0,1]'
run_base_control G4_augmentation TRAINER.BiMC.GUIDANCE.VIEW_CONTROL augmentation
run_base_control G4_ordinary TRAINER.BiMC.GUIDANCE.SEMANTIC_SOURCE ordinary
# 可选的第五项：
# run_base_control G4_uniform TRAINER.BiMC.GUIDANCE.SEMANTIC_SOURCE uniform
```

共 4 次基类验证（可额外加入 uniform），保持 20 个伪序列和相同校准种子。使用新目录，避免直接 main.py 覆盖旧结果。
`support.json` 记录了样本身份、类别顺序和支持集索引；更换数据根路径可能导致清单不匹配，不能绕过校验。
这是固定五种子矩阵前的诊断性验证，不能据其结果强制选非零 rho。

检查这些输出应是 `metrics.json: status=base_validation_completed`，并有 `guidance_calibration.json`；没有正式测试准确率和预测文件。
无需保存 checkpoint，可节约磁盘。直接 main.py 运行没有矩阵 `run_spec.json`，请按下列命令单独读取候选表：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['BASE_OUT'])
for path in sorted(root.glob('cub200/*/seed_1/guidance_calibration.json')):
    report = json.loads(path.read_text())
    rows = report['candidate_results']
    ref = next(row for row in rows if row['rho'] == 0)
    metrics = json.loads((path.parent / 'metrics.json').read_text())
    print('\n', path.parent.parent.name,
          'status=', metrics['status'], 'selected_rho=', report['selected_rho'],
          'protocol=', report['protocol_sha256'])
    for row in rows:
        print('rho=', row['rho'],
              'dBase=', round(row['mean_base_accuracy']-ref['mean_base_accuracy'], 4),
              'dNovel=', round(row['mean_novel_accuracy']-ref['mean_novel_accuracy'], 4),
              'dH=', round(row['mean_harmonic_accuracy']-ref['mean_harmonic_accuracy'], 4),
              'feasible=', row['feasible'])
PY
```

只比较 protocol 相同的控制。若扩展到 mini-ImageNet 的普通文本控制，两组必须同时设置
`TRAINER.BiMC.GUIDANCE.CANDIDATES_PER_BAND 3`（9 对 9 条文本），并使用独立目录；不能与默认 G4 的 15 条文本直接归因比较。

## 5. 视图响应诊断（可选）

只使用基类训练图像；该结果检验尺度响应，不用于选 rho：

```bash
python tools/audit_frequency_guidance_views.py \
  --dataset cub200 --data-root "$DATA_ROOT" --gpu "$GPU" \
  --samples 8 --seed 1 --backend safd

python tools/audit_frequency_guidance_views.py \
  --dataset cub200 --data-root "$DATA_ROOT" --gpu "$GPU" \
  --samples 8 --seed 1 --backend fft
```

输出在 `outputs/frequency_guidance_view_audit/`。模糊/棋盘纹理响应只是受控诊断，不能单独证明三种文本与物理频带一一对应。

## 6. 后续决策

- 尺度关系同质、置换不影响：先改表示或停止频率路线；
- 关系不同但决策几乎不变：检查分类度量、原型更新和集成稀释；
- 改对与改错抵消：研究有害先验风险，并与普通置信度控制比较；
- 校准微小收益不迁移：检查估计稳定性和任务匹配。

通过机制检查和便宜控制之后，再启动三数据集五种子主比较。当前先提供第 2 或第 3 节的一份压缩包即可。
