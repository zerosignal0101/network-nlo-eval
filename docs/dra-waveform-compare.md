# DRA 波形参考与解析模型对照

日期：2026-11-04

> **2026-11-05 修正**：物理侧 `DistributedRamanFiber` 曾把频变拉曼增益
> 逐样本乘在时域（而非频谱），给前向泵波形引入随记录长度变化的虚假
> 调制。本文所有**前向泵**的旧数值（3/5/7ch 的 +0.79/+0.96/+1.00 dB、
> 全 WDM 的 +4.8/+13.8 dB）都基于该实现，已被修正后的数据取代；
> nopump 逐位不变，backward 仅在全 WDM 宽带下有小幅修正。修正后的
> 全 WDM 结果见 §13.6 与 `docs/dra-open-issues.md` §5。

## 1. 数据来源

`physical-nlo-eval` 的 `DistributedRamanFiber`：

- 独立 NumPy SSFM，逐 SSFM 步施加泵浦增益；
- 未耗尽泵浦，Gaussian 拉曼谱；
- 3/5/7 信道、100 km、50 GBd DP-QPSK、1 mW/偏振（0 dBm/pol）；
- 前向/后向 1450 nm、0.1/0.2W；
- 数据（旧、500 m）：`data/dra_waveform/dra_reference.{json,npz}`；
- 数据（收敛、62.5 m）：`data/dra_waveform_converged/`，详见 §13。

> **历史数据提示**：`data/dra_waveform/` 的 5/7ch 参考固定 500 m SSFM
> 步长，未收敛（5ch 1.53 dB、7ch 4.01 dB 伪影）。本文 §3–§12 中所有
> 500 m 的 5/7ch 数字只作历史/趋势参考，不能用于校准。收敛后的完整表
> 见 §13。

统一后的 NLI 定义：

```text
P_signal = FFT 带内 |NL|² / N²
a        = argmin ||NL - a*LIN||²  (单复标量)
P_NLI    = FFT 带内 |NL - a*LIN|² / N²
SNR_NLI  = 10 log10(P_signal / P_NLI)
```

分析带宽固定 50 GHz；`a` 只吸收公共增益/相位，不吸收带内频率相关失真。
原始未拟合差值 `P_raw` 仍保留，但不用作 NLI。

## 2. 对照方法

`worktree` 的 `scripts/dra_waveform_compare.py` 用与波形相同的 3/5/7 信道、
100 km、1 mW/偏振（与波形 0 dBm/pol 对齐），在 50 GHz 占用栅格上运行：

- `hybrid_raman_paper`（paper）
- `buglia2024_ra_gn`
- `pcfm_h3_raman`
- `numerical_dra`
- `pcfm` + lumped

比较中心信道的 `SNR_NLI` 和泵浦引起的 SNR 变化。脚本从参考 JSON 的
`pump_model` 自动选择解析侧 `pump_depletion`（`undepleted` → `False`，
`depleted` → `True`），hybrid/H3 会透传该口径；其余模型没有该开关，见 §13。

## 3. 结果（500 m 旧参考，历史/未收敛）

> 本节数字来自 `data/dra_waveform/` 的 500 m 参考。3ch 在 500 m 已收敛
> （伪影 0.07 dB），5/7ch 未收敛（1.53/4.01 dB）。**5/7ch 仅作历史对照，
> 不进入校准域。** 收敛参考见 §13。

### 3.1 绝对 SNR

3 信道中心信道：

| 配置 | 波形 SNR_NLI | hybrid | H3 | numerical | lumped |
|---|---:|---:|---:|---:|---:|
| 无泵 | 36.78 | 36.33 | 36.65 | 36.43 | 36.15 |
| 后向 0.2W | 36.51 | 36.16 | 36.40 | 36.16 | 36.15 |
| 前向 0.2W | 21.75 | 23.80 | 24.11 | 23.73 | 36.15 |

5 信道中心信道（**历史/未收敛，500 m**）：

| 配置 | 波形 SNR_NLI | hybrid | H3 | numerical | lumped |
|---|---:|---:|---:|---:|---:|
| 无泵 | 34.79 | 35.71 | 35.96 | 35.80 | 35.54 |
| 后向 0.2W | 34.51 | 35.54 | 35.72 | 35.56 | 35.54 |
| 前向 0.2W | 19.64 | 23.49 | 23.72 | 23.43 | 35.54 |

7 信道（**历史/未收敛，500 m**；压力测试，不纳入校准）：

| 配置 | 波形 | hybrid |
|---|---:|---:|
| 无泵 | 32.27 | 35.33 |
| 后向 0.2W | 31.95 | 35.17 |
| 前向 0.2W | 17.64 | 23.39 |

旧 500 m：3/5 信道在**无泵和后向泵**下只差 0.4–1.1 dB；前向泵差
2.1–3.9 dB；7 信道差 3–6 dB。
绝对差不是 Raman 方向错误，而是对照口径不同：

- 波形 NLI 是 **50 GHz 理想矩形带内、单复标量公共增益拟合后的残差**；
  `nli_analysis_bandwidth_hz=50e9`、`nli_definition` 已写入 JSON；
- 解析 NLI 是 GN 功率，信道是 50 GHz 矩形谱；
- 波形是 3/5/7 信道、RRC 55 GHz、有限样本；
- 波形未加 ASE，解析只取 NLI，但两者的接收机/带宽归一化仍不同。

因此当前不能把 3–6 dB 当作模型误差直接校准。

### 3.2 泵浦趋势

更重要的是相对变化：

| 数据集 | 配置 | 波形 ΔSNR | hybrid ΔSNR | H3 ΔSNR | lumped ΔSNR |
|---|---|---:|---:|---:|---:|
| 3ch | 后向 0.2W | -0.27 | -0.17 | -0.25 | 0.00 |
| 3ch | 前向 0.2W | -15.03 | -12.53 | -12.54 | 0.00 |
| 5ch | 后向 0.2W | -0.28 | -0.17 | -0.24 | 0.00 |
| 5ch | 前向 0.2W | -15.15 | -12.22 | -12.24 | 0.00 |

旧 500 m 校准域（**历史/已废弃**）：

| 数据集 | 泵方向 | 解析 vs 波形 | 状态 |
|---|---|---:|---|
| 3ch | 无泵 | -0.45 dB | 500 m 已收敛 |
| 3ch | 后向 | -0.36 dB | 500 m 已收敛 |
| 3ch | 前向 | +2.06 dB | 仅趋势 |
| 5ch | 无泵 | +0.91 dB | 含 1.53 dB 步长伪影 |
| 5ch | 后向 | +1.03 dB | 含步长伪影 |
| 5ch | 前向 | +3.85 dB | 含步长伪影，仅趋势 |
| 7ch | 无/后向 | +3.2 dB | 含 4.01 dB 步长伪影 |
| 7ch | 前向 | +5.75 dB | 含步长伪影，仅趋势 |

这张表只记录历史结论，不再作为校准依据；收敛后的校准域见 §13。

仍成立的趋势结论：

- 波形、hybrid、H3 都显示**前向泵大幅恶化 NLI，后向泵影响很小**；
- lumped 对前向泵完全无响应，趋势错 15 dB；
- H3 与 hybrid 的趋势一致，H3 略乐观约 0.1–0.6 dB；
- 旧 500 m 的绝对偏差需要在步长收敛和 NLI 提取/带宽口径统一后再校准。

## 4. 这轮新增的参考价值

1. 第一次有独立 SSFM DRA 波形数据，而不是只用数值 GN 互比。
2. 验证了前向/后向 DRA 的物理趋势，尤其是 lumped 的结构性失败。
3. 暴露了当前波形 NLI 提取与解析 GN 口径不可直接对齐，后续需要统一接收机/NLI 定义。

## 5. 限制

- 3/5/7 信道，不是 57/152 信道全 WDM；7ch 未校准；
- 泵浦未耗尽，没有泵浦-泵浦耦合；
- 波形数据无 ASE，SNR 是 NLI-limited；
- NLI 用单复标量拟合，可能移除部分与信号相关的 NLI；
- 解析模型用矩形 50 GHz，波形是 RRC 55 GHz；
- 不能把绝对 3–6 dB 差直接归因于某个模型。

## 6. 复现

旧 500 m 参考（保留原行为）：

```bash
PYTHONPATH=src python scripts/dra_waveform_compare.py
```

结果：`results/raman/waveform_compare.json`。

收敛 62.5 m 参考（`pump_depletion` 由参考 JSON 的 `pump_model` 自动选择）：

```bash
PYTHONPATH=src python scripts/dra_waveform_compare.py \
  --data-dir data/dra_waveform_converged \
  --reference dra_reference.json \
  --output results/raman/waveform_compare_converged.json

PYTHONPATH=src python scripts/dra_waveform_compare.py \
  --data-dir data/dra_waveform_converged \
  --reference dra_reference_depleted.json \
  --output results/raman/waveform_compare_converged_depleted.json
```

`--data-dir` 默认 `data/dra_waveform`，`--reference` 默认
`dra_reference.json`，`--output` 缺省时沿用旧路径（非默认数据目录会写到
`results/raman/waveform_compare_<dir>.json`，避免覆盖旧结果）。

需要重跑 WP1 的未耗尽对照 pass 时，加 `--control-output`：

```bash
PYTHONPATH=src python scripts/dra_waveform_compare.py \
  --control-output results/raman/waveform_compare_undepleted.json
```

它只对 hybrid/H3 重跑相反的 `pump_depletion` 口径，并写入
`closure_db = depleted_delta - undepleted_delta`。

## 7. 前向泵功率与步长证据

3ch 前向泵：

| 泵功率 | 波形 | hybrid | 差 |
|---|---:|---:|---:|
| 0.1W | 30.03 | 30.58 | +0.54 |
| 0.2W | 21.75 | 23.80 | +2.06 |

误差在 0.1W 与 0.2W 之间变号并放大，不能用一个固定 dB 修正。

3ch 前向 0.2W 的 SSFM 步长收敛（物理波形）：

| 步长 | 波形 SNR |
|---|---:|
| 500m | 21.747 |
| 250m | 21.697 |
| 125m | 21.676 |
| 62.5m | 21.664 |

步长收敛到 <0.1 dB，因此偏差不是 SSFM 离散误差。

5ch/7ch 的前向偏差继续随信道数增长：

| 数据集 | 0.1W | 0.2W |
|---|---:|---:|
| 3ch | +0.54 | +2.06 |
| 5ch | +1.83 | +3.85 |
| 7ch | +3.70 | +5.75 |

这指向前向泵的跨信道/高增益项，而不是 NLI 定义或功率口径。

## 8. 前向泵的 SPM/XPM 拆分

用同一 100 km、1 mW/偏振、1450 nm 0.2W 前向泵做 1ch 与 3ch 探针：

| 场景 | 波形 | hybrid | 差 |
|---|---:|---:|---:|
| 1ch 无泵 | 38.14 | 37.85 | -0.29 |
| 1ch 前向 | 23.32 | 24.65 | +1.33 |
| 3ch 无泵 | 36.78 | 36.33 | -0.45 |
| 3ch 前向 | 21.75 | 23.80 | +2.06 |

拆分：

- 1ch 前向差 1.33 dB，主要是**泵浦下的 SPM**；
- 3ch 前向再增加 0.73 dB，来自**跨信道项**；
- 3ch 无泵的 XPM 反而只增加 0.16 dB 偏差，说明问题集中在前向泵与 SPM/增益剖面的耦合，不是普通 XPM。

因此下一步应优先校准前向泵的 SPM 增益剖面，而不是继续调总 NLI 系数。

## 9. 前向泵与 SPM 增益增强

单信道 0.2W 前向泵：

| 量 | 波形 | hybrid |
|---|---:|---:|
| 无泵 SNR | 38.14 | 37.85 |
| 前向 SNR | 23.32 | 24.65 |
| SNR 下降 | 14.82 dB | 13.20 dB |

物理波形的 NLI 对前向增益更敏感，解析模型少约 1.6 dB。

同一前向泵下切换 `hybrid_raman_paper` 的 NLI 形式：

| 形式 | 1ch SNR | 3ch SNR |
|---|---:|---:|
| `paper` | 24.84 | 23.80 |
| `integral` | 24.77 | 23.75 |
| `pcfm` | 24.84 | 23.80 |

三种形式几乎一样，说明偏差不是 Eq. (6)-(9) 的闭式离散误差，而是
**剖面/增益权重**本身：解析 ODE 对前向泵下的 SPM 噪声增强仍然偏乐观。

## 10. 前向泵 ODE 步数收敛

1ch 前向 0.2W，`hybrid_raman_paper` 的 `spp_steps`：

| spp_steps | SNR |
|---:|---:|
| 12 | 24.877 |
| 32 | 24.836 |
| 64 | 24.835 |
| 128 | 24.835 |
| 256 | 24.836 |

32 步以后收敛到 <0.01 dB，说明偏差不是 ODE 分辨率或 Eq. (10) 拟合采样不足。
剩余问题是剖面/增益权重本身，需改模型项，不是调步数。

## 11. 校准状态与阻塞（收敛参考重写）

以下为 62.5 m 收敛参考下的校准域（表格与口径见 §13）：

已接受：

- **3/5/7 信道、无泵**：hybrid −0.38/−0.62/−0.95 dB，
  H3 −0.06/−0.37/−0.73 dB，均在 ±1.0 dB 内；
- **3/5/7 信道、后向泵（0.1/0.2 W）**：hybrid −0.36…−0.71 dB，
  H3 −0.05…−0.69 dB，均在 ±1.0 dB 内；
- **3/5 信道、前向泵、耗尽参考**：hybrid +0.79/+0.96 dB（0.2 W），
  H3 +1.09/+1.19 dB；hybrid 在 ±1.0 dB 内，H3 因默认 EGN 略乐观
  0.2–0.3 dB；
- 1 mW/偏振、50 GHz 分析带。

未校准：

- **7 信道前向泵**：目前只有未耗尽参考，同口径残差 hybrid +2.86 dB；
  待 WP5a 后续生成耗尽 7ch 数据后再判断，当前只作趋势；
- **前向泵 H3**：1.1–1.2 dB，略超 ±1.0 dB；属 EGN 的已知乐观偏差，
  需要时可用 `use_egn=False` 复核。

前向泵偏差在排除以下项后仍存在：

- 功率口径；
- NLI 分析带宽与投影；
- SSFM 步长（62.5 m 已收敛）；
- ODE 步数；
- Eq. (6)-(9) 的 paper/integral/pcfm 形式。

收敛且口径对齐后，前向残差从旧 500 m 的 +2.1/+3.9 dB 降到
耗尽参考下的 +0.8/+1.0 dB（hybrid，3/5ch）；剩余部分主要来自
前向泵 SPM 增益增强项，不是常数校准或参数扫描能修复的。

## 12. 7 信道状态更新（2026-11-04）

`docs/dra-7ch-audit.md` 的复核发现：7ch 无泵 3.06 dB 缺口来自 500 m SSFM
步长未收敛，不是 GN 适用域。同一 `input_7ch` 从 500 m 加密到 15.625 m 后，
中心信道 32.263 → 36.270 dB，与 hybrid 的差从 3.06 降到 0.94 dB；
3ch 只变 0.08 dB，5ch 变 1.53 dB。

因此：

- 本文第 3 节的 500 m 7ch 数字（32.27、3.2 dB 等）只代表未收敛参考，
  不可用于校准或模型误差归因；5ch 的 500 m 数字同理；
- WP5a 已生成 62.5 m 收敛参考，§13 给出重算结果：7ch 无泵
  hybrid/H3 残差 −0.95/−0.73 dB，后向 0.2W −0.71/−0.56 dB；
- 7ch 无泵/后向因此重新进入**校准候选**（±1.0 dB）；
- 7ch 前向仍只有未耗尽参考（同口径残差 +2.86 dB），且 WP5a 尚未生成
  耗尽 7ch 数据，继续只作趋势，不进入校准域。

## 13. 收敛参考（62.5 m）

### 13.1 数据与口径

WP5a 在 `physical-nlo-eval` 重新生成了 62.5 m 收敛参考，本仓副本：

| 文件 | 内容 | `pump_model` |
|---|---|---|
| `data/dra_waveform_converged/dra_reference.{json,npz}` | 3/5/7ch、5 配置 | `undepleted` |
| `data/dra_waveform_converged/dra_reference_depleted.{json,npz}` | 3/5ch + 1ch + 功率扫描 | `depleted` |
| `data/dra_waveform_converged/step_convergence.json` | 500→31.25 m 步长探针 | — |

旧 `data/dra_waveform/` 保留不动，作为历史/未收敛参考。所有收敛数据集的
`step_converged` 为 `true`，`coarse_step_m = 62.5 m`。

`scripts/dra_waveform_compare.py` 从参考 JSON 顶层读取 `pump_model`，自动
选择解析侧口径：

- `undepleted` → `pump_depletion=False`（hybrid/H3）；
- `depleted` → `pump_depletion=True`（hybrid/H3）。

`numerical_dra`、`buglia2024_ra_gn` 和 lumped `pcfm` 没有该开关，行内
`pump_depletion_applied=false`；在未耗尽参考下它们的泵浦口径仍是耗尽的，
表里用 `†` 标注，**不能**与未耗尽波形做同口径残差。

### 13.2 未耗尽参考（62.5 m）

绝对 `SNR_NLI`（中心信道，dB）：

| 数据集 | 配置 | 波形 | hybrid | H3 | numerical† | lumped |
|---|---|---:|---:|---:|---:|---:|
| 3ch | 无泵 | 36.708 | 36.328 | 36.650 | 36.426 | 36.166 |
| 3ch | 前向 0.1W | 29.953 | 30.429 | 30.781 | 30.653 | 36.166 |
| 3ch | 前向 0.2W | 21.664 | 23.258 | 23.546 | 23.732 | 36.166 |
| 3ch | 后向 0.1W | 36.604 | 36.245 | 36.547 | 36.319 | 36.166 |
| 3ch | 后向 0.2W | 36.442 | 36.151 | 36.393 | 36.164 | 36.166 |
| 5ch | 无泵 | 36.331 | 35.706 | 35.962 | 35.800 | 35.557 |
| 5ch | 前向 0.1W | 29.229 | 29.821 | 30.098 | 30.133 | 35.557 |
| 5ch | 前向 0.2W | 20.273 | 22.607 | 22.818 | 23.429 | 35.557 |
| 5ch | 后向 0.1W | 36.216 | 35.627 | 35.864 | 35.701 | 35.557 |
| 5ch | 后向 0.2W | 36.021 | 35.537 | 35.716 | 35.558 | 35.557 |
| 7ch | 无泵 | 36.274 | 35.326 | 35.549 | 35.417 | 35.186 |
| 7ch | 前向 0.1W | 28.975 | 29.436 | 29.675 | 29.836 | 35.186 |
| 7ch | 前向 0.2W | 19.321 | 22.185 | 22.354 | 23.334 | 35.186 |
| 7ch | 后向 0.1W | 36.146 | 35.248 | 35.453 | 35.323 | 35.186 |
| 7ch | 后向 0.2W | 35.868 | 35.161 | 35.308 | 35.187 | 35.186 |

同口径残差（解析 − 波形，dB）：

| 数据集 | 无泵 | 后向 0.1W | 后向 0.2W | 前向 0.1W | 前向 0.2W |
|---|---:|---:|---:|---:|---:|
| 3ch hybrid | −0.380 | −0.359 | −0.290 | +0.476 | +1.594 |
| 3ch H3 | −0.058 | −0.056 | −0.048 | +0.828 | +1.882 |
| 5ch hybrid | −0.624 | −0.589 | −0.484 | +0.592 | +2.333 |
| 5ch H3 | −0.368 | −0.352 | −0.305 | +0.869 | +2.545 |
| 7ch hybrid | −0.948 | −0.897 | −0.707 | +0.462 | +2.864 |
| 7ch H3 | −0.725 | −0.693 | −0.560 | +0.701 | +3.034 |

注意：前向列的 +1.59/+2.33/+2.86 dB 是**未耗尽参考 vs 未耗尽解析**的
同口径残差，不是模型误差被耗尽口径放大后的值。它与耗尽参考下的残差
（§13.3）要分开看。

### 13.3 耗尽参考（62.5 m）

3/5/7ch 绝对 `SNR_NLI`（dB）：

| 数据集 | 配置 | 波形 | hybrid | H3 | hybrid 差 | H3 差 |
|---|---|---:|---:|---:|---:|---:|
| 3ch | 无泵 | 36.708 | 36.328 | 36.650 | −0.380 | −0.058 |
| 3ch | 前向 0.1W | 30.299 | 30.575 | 30.911 | +0.277 | +0.613 |
| 3ch | 前向 0.2W | 23.012 | 23.803 | 24.105 | +0.791 | +1.093 |
| 3ch | 后向 0.1W | 36.608 | 36.247 | 36.549 | −0.361 | −0.059 |
| 3ch | 后向 0.2W | 36.454 | 36.156 | 36.398 | −0.298 | −0.057 |
| 5ch | 无泵 | 36.331 | 35.706 | 35.962 | −0.624 | −0.368 |
| 5ch | 前向 0.1W | 29.846 | 30.061 | 30.313 | +0.216 | +0.467 |
| 5ch | 前向 0.2W | 22.531 | 23.491 | 23.724 | +0.960 | +1.193 |
| 5ch | 后向 0.1W | 36.223 | 35.630 | 35.867 | −0.593 | −0.356 |
| 5ch | 后向 0.2W | 36.043 | 35.545 | 35.724 | −0.499 | −0.320 |
| 7ch | 无泵 | 36.274 | 35.326 | 35.549 | −0.948 | −0.725 |
| 7ch | 前向 0.1W | 29.845 | 29.771 | 29.972 | −0.075 | +0.126 |
| 7ch | 前向 0.2W | 22.384 | 23.387 | 23.583 | +1.003 | +1.199 |
| 7ch | 后向 0.1W | 36.155 | 35.252 | 35.457 | −0.903 | −0.698 |
| 7ch | 后向 0.2W | 35.903 | 35.171 | 35.319 | −0.731 | −0.584 |

1ch 与 3ch 发射功率扫描（同一耗尽参考）：

| 数据集 | 配置 | 波形 | hybrid | H3 | hybrid 差 | H3 差 |
|---|---|---:|---:|---:|---:|---:|
| 1ch | 无泵 | 38.043 | 37.869 | 38.461 | −0.174 | +0.417 |
| 1ch | 前向 0.2W | 23.704 | 24.836 | 25.382 | +1.132 | +1.678 |
| 1ch | 后向 0.2W | 37.774 | 37.664 | 38.158 | −0.110 | +0.384 |
| 3ch 0.5 mW | 无泵 | 42.721 | 42.349 | 42.671 | −0.372 | −0.050 |
| 3ch 0.5 mW | 前向 0.2W | 28.374 | 29.561 | 29.857 | +1.187 | +1.484 |
| 3ch 2.0 mW | 无泵 | 30.704 | 30.308 | 30.591 | −0.397 | −0.113 |
| 3ch 2.0 mW | 前向 0.2W | 18.089 | 18.256 | 18.624 | +0.167 | +0.536 |

（3ch 1.0 mW 与上表 3ch 同一行，不再重复。）

注：H3 列是旧默认 `egn=True` 的结果。默认改为 `egn="auto"`（前向泵关闭
EGN）后的前向值见 §13.4 与 `docs/h3-forward-audit.md` §11。

### 13.4 校准域（收敛参考）

| 数据集 | 泵方向 | hybrid 残差 | H3 残差 | 状态 |
|---|---|---:|---:|---|
| 3ch | 无泵 | −0.38 dB | −0.06 dB | 校准域 |
| 5ch | 无泵 | −0.62 dB | −0.37 dB | 校准域 |
| 7ch | 无泵 | −0.95 dB | −0.73 dB | 校准候选 |
| 3ch | 后向 0.1/0.2W | −0.36/−0.29 dB | −0.06/−0.05 dB | 校准域 |
| 5ch | 后向 0.1/0.2W | −0.59/−0.48 dB | −0.35/−0.31 dB | 校准域 |
| 7ch | 后向 0.1/0.2W | −0.90/−0.71 dB | −0.69/−0.56 dB | 校准候选 |
| 3ch | 前向 0.2W（耗尽） | +0.79 dB | +1.09 dB | hybrid 校准域；H3 略超 |
| 5ch | 前向 0.2W（耗尽） | +0.96 dB | +1.19 dB | hybrid 校准域；H3 略超 |
| 7ch | 前向 0.1W（耗尽） | −0.08 dB | +0.13 dB | 校准域 |
| 7ch | 前向 0.2W（耗尽） | +1.00 dB | +1.20 dB | hybrid 校准域；H3 略超 |

结论：

- 3/5/7ch 的无泵与后向泵残差全部在 **±1.0 dB** 内；7ch 从旧 500 m 的
  +3.06 dB 修正到 −0.95 dB（hybrid）；
- 前向泵在耗尽参考下，3/5/7ch hybrid 为 +0.79/+0.96/+1.00 dB，进入 ±1.0 dB；
  7ch 前向 0.1W 只有 −0.08 dB。H3 的 +1.09/+1.19/+1.20 dB 含旧默认 EGN；
  默认改为 `egn="auto"`（前向泵关闭 EGN）后实测为 +0.60/+0.78/+0.83 dB，
  也进入 ±1.0 dB，见 `docs/h3-forward-audit.md` §11；
- 7ch 前向的 +2.86 dB（未耗尽参考）是口径错配造成的旧数字；加入 7ch 耗尽
  数据后，同口径残差降到 +1.00 dB（hybrid），与 3/5ch 一致；
- 单次 256-symbol 实现的统计误差约 0.3–0.5 dB（见
  `docs/dra-7ch-audit.md` §4），上述 ±1.0 dB 校准带里有约一半是统计预算。

### 13.5 复现与产物

```bash
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python scripts/dra_waveform_compare.py \
  --data-dir data/dra_waveform_converged \
  --reference dra_reference.json \
  --output results/raman/waveform_compare_converged.json

PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python scripts/dra_waveform_compare.py \
  --data-dir data/dra_waveform_converged \
  --reference dra_reference_depleted.json \
  --output results/raman/waveform_compare_converged_depleted.json
```

产物：

- `results/raman/waveform_compare_converged.json`（未耗尽，75 行）；
- `results/raman/waveform_compare_converged_depleted.json`（耗尽，120 行，
  含 7ch）。

脚本对 `data/dra_waveform_converged/` 强制要求每个数据集
`step_converged: true` 且 `coarse_step_m <= 62.5`；不满足会直接退出。

### 13.6 开放问题与全 WDM 对照

残差的功率依赖、1ch 前向、seed 统计预算和偏振耗尽驱动的
factor=1/2 复核见
`docs/dra-open-issues.md`（产物 `results/raman/launch_power_analysis.json`、
`one_ch_forward.json`、`seed_ensemble.json`）。57/152ch 全 WDM 波形参考
已到位：`data/dra_waveform_wide/{c57,c57_seed20261105,cl152}.{json,npz}`，
用 `scripts/dra_wideband_compare.py` 读取并以 `SpectrumGrid`、64 GBd、
B = 符号率、`pump_depletion=True` 跑 hybrid/H3/numerical/lumped 四模型；
152ch 用 `--skip-numerical` 加速。结果：

- `results/raman/wideband_compare_c57.json`（57ch，seed 20261104）
- `results/raman/wideband_compare_c57_seed20261105.json`（57ch，第二 seed）
- `results/raman/wideband_compare_cl152.json`（152ch）

要点见 `docs/dra-open-issues.md` §5：修正物理侧增益施加后，57ch
nopump 两 seed 均值 −0.52 dB、backward **−0.50 dB**、forward
**−3.21 dB**（解析保守；1×；2× 实测为 −3.65 dB）；152ch 修正后
1× 为 nopump +0.29、backward −1.30、forward +1.52 dB。2× 128-symbol
nopump 扩到 4 seed 后为 33.502/34.247/23.178/31.422 dB，均值
**30.587 ± 5.082 dB**（中位数 32.462）；对解析 32.765 dB 的残差按
均值 +2.18、按中位数 +0.30 dB，被 5 dB 的 seed 散布主导。旧的
2-seed 33.874 dB / −1.11 dB 不稳健，不能引用。2× 泵浦实测
factor=1 残差：c57 bwd −1.17、fwd −3.65；cl152 bwd −2.92、fwd
+1.26 dB（泵浦 fwd 的 1×→2× 偏差只有 +0.6…+0.8 dB，旧 alias-free
外推 −2.7/+0.1 作废）。旧的“forward 因周期记录伪影
不可用”结论作废：那是把频变拉曼增益逐样本乘在时域造成的虚假调制
（详见 `docs/dra-forward-audit.md` 末节）。1× 采样率会高估 NLI
0.4–1.4 dB（2× 已足够）；64-symbol 记录的 seed 散布可达 5.6 dB
（cl152，4 seeds），绝对 NLI 必须用多 seed 集成，
64-symbol 只用于同 seed 的步长趋势。色散斜率必须按波形 β₃ 反解
（cl152 的 β₃=1e-4 ps³/km → S≈−0.0219），不能沿用默认 S=0.06。

### 13.7 偏振驱动开关：drive/state 分离（2026-11-05）

物理 `DistributedRamanFiber` 用双偏振总功率驱动泵浦耗尽。解析
`RamanPaperSpanSolver` 新增 `signal_polarizations`（默认 1）：state 仍是
每偏振功率，耦合项的信号驱动 `drive_i = signal_polarizations * P_s,i`，
接收波自己的功率只乘一次。`1` 是历史每偏振口径且逐位保持旧结果，
`2` 是物理总功率口径。该开关通过 `DRASpanKernelBase` 暴露到
`hybrid_raman_paper` / `pcfm_h3_raman`，两个对照脚本新增
`--signal-polarizations {1,2}`（默认 1）；`numerical_dra` 使用独立
`solve_ivp` 求解器，本次未接入。

62.5 m 收敛耗尽参考、前向/后向 0.2 W（残差 = 解析 − 波形，hybrid）：

| 数据集 | 配置 | 波形 | 1× | 2× |
|---|---|---:|---:|---:|
| 3ch | fwd | 23.265 | +0.538 | +1.012 |
| 5ch | fwd | 23.239 | +0.252 | +0.964 |
| 7ch | fwd | 24.399 | −1.012 | −0.111 |
| 3ch | bwd | 36.452 | −0.296 | −0.292 |
| 5ch | bwd | 36.046 | −0.501 | −0.494 |
| 7ch | bwd | 35.992 | −0.821 | −0.811 |

无泵配置在中心信道逐位不变；关闭 `include_isrs` 时严格逐位不变。
全 WDM 的 c57/cl152 factor=2 结果与完整表见 `docs/dra-open-issues.md`
§1.1：57ch 前向 −3.21 → −1.24 dB，后向 −0.50 → −0.43 dB；
cl152 六泵后向 −1.30 → −0.13 dB，前向 +1.52 → +2.30 dB（1× 混叠
约 1.4 dB 未处理）。factor=2 不是通用修正，默认仍为 1。

产物：`results/raman/waveform_compare_converged_depleted_pol2.json`、
`wideband_compare_c57_pol2.json`、`wideband_compare_c57_seed20261105_pol2.json`、
`wideband_compare_cl152_pol2.json`。

> 注：仓库里的 `results/raman/waveform_compare_converged_depleted.json`
> 仍是 3/5/7ch 数据重生成前的数值（例如 3ch fwd 波形 23.012 dB）。
> §13.7 的 1×/2× 数字是对重生成后的 `data/dra_waveform_converged`
> 实测的（3ch fwd 波形 23.265 dB）。两者口径相同，只是数据版本不同；
> 本次没有覆盖旧文件。
