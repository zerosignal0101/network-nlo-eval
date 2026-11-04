# 前向泵口径审计：泵浦耗尽 vs 未耗尽（WP1）

日期：2026-11-04
分支：`feat/dra-undepleted`（从 `multi-model-dev` `c69ba98` 创建）

## 1. 结论

物理波形参考 `physical-nlo-eval` 的 `DistributedRamanFiber` 采用**未耗尽泵浦**：
泵浦沿 z 解析衰减，不被信号耗尽。解析侧 `RamanPaperSpanSolver` 解的是耦合
信号/泵浦 ODE，泵浦会被信号耗尽。这个口径差异在前向泵下系统性地降低了解析
模型的 NLI。

本轮给 `RamanPaperSpanSolver` 增加了 `pump_depletion` 开关，并打通到 hybrid/H3
的核与模型。结果是可复现的：

- 前向 0.2W 的闭合量为 **0.55–1.23 dB**，随信道数/泵浦功率增长；
- 前向 0.1W 的闭合量为 **0.13–0.33 dB**；
- 后向泵的闭合量为 **0.001–0.011 dB**（不敏感，符合物理预期）；
- 未耗尽口径下前向残差仍有 **+0.4 ~ +4.7 dB**（3/5/7ch，0.1/0.2W），
  剩余差异不是泵浦耗尽一项能解释的；
- `numerical_dra` 没有该开关，但其默认残差与 hybrid 只差 0.1–0.2 dB；
  因此剩余前向残差是**所有 GN 模型与 SSFM 波形之间的共同差异**，不是
  `pcfm_h3_raman` 或 H3 核独有的问题。

WP5 之后，用 62.5 m 收敛的**耗尽**参考复测：3/5/7ch 前向 0.2W 的 hybrid
残差为 **+0.79/+0.96/+1.00 dB**，H3 为 +1.09/+1.19/+1.20 dB。剩余约 1 dB
是强前向增益下单信道 SPM 的 GN 适用域与默认 EGN，不是泵浦耗尽口径；
完整复核见 §8。旧表里的 +1.5–4.5 dB 同时混入了 500 m 波形步长伪影和
未耗尽/耗尽口径错配。

`pump_depletion=False` 是**对照口径**，不是更物理的默认。真实 DRA 在高信号
功率下会发生泵浦耗尽；默认仍为 `True`，所有既有结果保持不变。

## 2. 实现

### 2.1 求解器开关

`RamanPaperSpanSolver.__init__(..., pump_depletion: bool = True)`：

- `True`：现有耦合 Eq. (2) 行为，逐位不变；
- `False`：构建耦合矩阵后把泵浦行（含泵浦-泵浦耦合）清零。泵浦仍留在状态
  向量里，只按 `dP_p/dz = -alpha_p P_p` 演化；信号行不变，仍从泵浦获得增益。
  反向泵的预设剖面本来就是 `P0 exp(-alpha_p (L-z))`，未耗尽时不再做不动点
  迭代（泵浦不依赖信号），直接使用该解析剖面。

### 2.2 打通路径

- `DRASpanKernelBase` / `HybridRamanPaperSpanKernel`
- `HybridRamanPcfmSpanKernel`（`nli_form="pcfm"` 旧路径）
- `HybridRamanPaperModel`
- `PCFH3RamanSpanKernel`
- `PCFH3RamanModel`
- `HYBRID_RAMAN_PAPER_MODEL_INFO.options` 与 `H3_RAMAN_MODEL_INFO.options`
  各增加一项 `pump_depletion` 说明。

无泵路径不进入 Raman 分支，`pump_depletion=False` 也保持与 H3 PCFM 逐位一致。

## 3. 结果

数据来源：`scripts/dra_waveform_compare.py`，3/5/7 信道、100 km、50 GBd、
1 mW/偏振、1450 nm 泵。`waveform_compare.json` 的默认数值未变，只新增
`pump_depletion: true` 字段；对照 pass 写入
`results/raman/waveform_compare_undepleted.json`。

### 3.1 绝对 SNR_NLI（中心信道，dB）

| 数据集 | 配置 | 波形 | hybrid 耗尽 | hybrid 未耗尽 | H3 耗尽 | H3 未耗尽 |
|---|---|---:|---:|---:|---:|---:|
| 3ch | 前向 0.1W | 30.034 | 30.575 | 30.429 | 30.911 | 30.781 |
| 3ch | 前向 0.2W | 21.747 | 23.803 | 23.258 | 24.105 | 23.546 |
| 5ch | 前向 0.1W | 28.230 | 30.061 | 29.820 | 30.313 | 30.098 |
| 5ch | 前向 0.2W | 19.644 | 23.491 | 22.606 | 23.724 | 22.818 |
| 7ch | 前向 0.1W | 26.068 | 29.771 | 29.437 | 29.972 | 29.676 |
| 7ch | 前向 0.2W | 17.636 | 23.387 | 22.185 | 23.583 | 22.354 |

### 3.2 误差与闭合量（dB）

误差 = 解析 SNR - 波形 SNR；closure = 耗尽误差 - 未耗尽误差，正值表示
未耗尽口径更接近波形。

| 数据集 | 配置 | hybrid 耗尽 | hybrid 未耗尽 | closure | H3 耗尽 | H3 未耗尽 | closure |
|---|---|---:|---:|---:|---:|---:|---:|
| 3ch | 前向 0.1W | +0.541 | +0.395 | +0.146 | +0.877 | +0.747 | +0.130 |
| 3ch | 前向 0.2W | +2.056 | +1.511 | +0.545 | +2.358 | +1.799 | +0.559 |
| 5ch | 前向 0.1W | +1.831 | +1.590 | +0.241 | +2.082 | +1.868 | +0.215 |
| 5ch | 前向 0.2W | +3.848 | +2.963 | +0.885 | +4.081 | +3.174 | +0.906 |
| 7ch | 前向 0.1W | +3.702 | +3.368 | +0.334 | +3.903 | +3.607 | +0.296 |
| 7ch | 前向 0.2W | +5.751 | +4.549 | +1.202 | +5.947 | +4.718 | +1.229 |

闭合量在 0.2W 达到 0.55–1.23 dB，且随泵浦功率和信道数单调增长，与
"泵浦耗尽随信号总功率增长"的预期一致。

### 3.3 后向泵

| 数据集 | 配置 | H3 耗尽误差 | H3 未耗尽误差 | closure |
|---|---|---:|---:|---:|
| 3ch | 后向 0.1W | -0.126 | -0.127 | +0.001 |
| 3ch | 后向 0.2W | -0.117 | -0.121 | +0.004 |
| 5ch | 后向 0.1W | +1.176 | +1.174 | +0.003 |
| 5ch | 后向 0.2W | +1.213 | +1.205 | +0.007 |
| 7ch | 后向 0.1W | +3.300 | +3.296 | +0.004 |
| 7ch | 后向 0.2W | +3.367 | +3.356 | +0.011 |

后向泵的闭合量在 0.01 dB 量级，说明这个开关只影响前向泵的口径，不会借
口径调整后向结果。

### 3.4 无泵

无泵时 hybrid 的 `pump_depletion=False` 不改变任何输出；H3 的
`pump_depletion=False` 与 H3 PCFM 逐位一致（测试
`test_no_pump_with_undepleted_option_is_bit_identical_to_h3_pcfm`）。

## 4. 剩余残差归属

未耗尽口径只解释了一部分前向偏差。剩余残差（未耗尽解析 - 波形）：

| 数据集 | 配置 | hybrid | H3 |
|---|---|---:|---:|
| 3ch | 前向 0.1W | +0.395 | +0.747 |
| 3ch | 前向 0.2W | +1.511 | +1.799 |
| 5ch | 前向 0.1W | +1.590 | +1.868 |
| 5ch | 前向 0.2W | +2.963 | +3.174 |
| 7ch | 前向 0.1W | +3.368 | +3.607 |
| 7ch | 前向 0.2W | +4.549 | +4.718 |

同一 pass 中 `numerical_dra`（默认耗尽口径）的误差：

| 数据集 | 配置 | numerical_dra |
|---|---|---:|
| 3ch | 前向 0.1W | +0.620 |
| 3ch | 前向 0.2W | +1.985 |
| 5ch | 前向 0.1W | +1.903 |
| 5ch | 前向 0.2W | +3.785 |
| 7ch | 前向 0.1W | +3.768 |
| 7ch | 前向 0.2W | +5.698 |

`numerical_dra` 与 hybrid 在每个场景只差 0.1–0.2 dB，且同样采用耦合 ODE
的耗尽口径。即使扣掉本轮量化的 0.55–1.23 dB，剩余 1.5–4.5 dB 仍会同时出现在
hybrid、H3 和 numerical 上。因此：

1. 剩余前向残差不是 H3 核或三次多项式拟合的专属问题；
2. 它也不是 hybrid Eq. (6)-(9) 闭式特有的，因为用 Eq. (6)-(8) 数值积分的
   `numerical_dra` 给出同样的量；
3. 它更像 GN 模型与 SSFM 波形在**强前向增益下的 NLI 生成/传播权重**差异，
   下一轮应在物理侧和 GN-vs-SSFM 层面拆解，而不是继续调解析模型系数。

## 5. 接口与复现

`results/raman/waveform_compare_undepleted.json` 每行包含：

- `pump_depletion`：对照口径（本文生成的文件为 `false`）；
- `snr_delta_db`：对照口径解析 SNR - 波形 SNR；
- `nli_ratio_db`：解析 NLI / 波形 NLI；
- `depleted_snr_delta_db`：同配置的默认耗尽误差；
- `undepleted_snr_delta_db`：同配置的未耗尽误差；
- `closure_db`：`depleted_snr_delta_db - undepleted_snr_delta_db`。

```bash
cd /home/zsig/Documents/gits/wt/dra-rebaseline
# 旧 500 m 参考的未耗尽对照 pass
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python scripts/dra_waveform_compare.py \
  --control-output results/raman/waveform_compare_undepleted.json
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python -m pytest \
  tests/test_hybrid_raman.py tests/test_pcfm_h3_raman.py -q
```

注意：本文 §3 的旧 pass 使用 500 m 参考；收敛参考的复核见 §8。

## 6. 限制与负结果

- `pump_depletion=False` 不是更物理的默认；它只用于和未耗尽波形对齐。
- 0.1W 的闭合量只有 0.13–0.33 dB，不能解释 0.1W 下的全部偏差。
- `numerical_dra` 目前没有 `pump_depletion` 开关，因此本轮的未耗尽 pass
  只覆盖 hybrid/H3；"残差共同存在" 的判断基于默认口径下的 numerical 对照。
- 7ch 无泵本身仍有约 3 dB 缺口（见 WP4），本文件不把它归因于泵浦耗尽。
- 未耗尽后前向残差仍有 1.5–4.5 dB，当前基准仍不能声称覆盖前向 DRA。

## 7. 下一步

1. **WP3（物理侧）**：给 `DistributedRamanFiber` 增加耦合泵耗尽
   `pump_model="depleted"`，生成耗尽波形参考。若"泵浦耗尽"假设成立，
   耗尽波形应把解析-波形闭合量从 0.55–1.23 dB 中拿回来，而不是继续放大。
2. **GN-vs-SSFM 审计**：在耗尽口径对齐后，针对前向泵的 SPM/XPM 增益权重
   逐项对照 PCFM Eq. (30)/(A.7) 与 Buglia Eq. (6)-(8) 的 `rho(z)` 积分，
   判断剩余残差来自 NLI 生成后的 `G(z->L)` 传播权重、剖面拟合，还是
   SSFM 与 GN 的适用域差异。
3. **WP4**：7ch 无泵的 3 dB 缺口与泵浦项分开处理，不混入本轮的闭合量。

## 8. 收敛参考复核（WP5b）

本文 §3 的闭合量 0.55–1.23 dB 是**旧 500 m 参考**下的值。WP4/WP5a 发现
500 m 的 5ch/7ch 波形未收敛（5ch 1.53 dB、7ch 4.01 dB 伪影），因此旧表里
的 5ch/7ch 数字同时混入了“泵浦耗尽口径差”和“波形步长误差”。用
`data/dra_waveform_converged/` 的 62.5 m 参考重算：

### 8.1 同口径残差（前向 0.2W，dB）

| 数据集 | 参考 | 波形 | hybrid | H3 | hybrid−wave | H3−wave |
|---|---|---|---:|---:|---:|---:|
| 3ch | 未耗尽 | 21.664 | 23.258 | 23.546 | +1.594 | +1.882 |
| 3ch | 耗尽 | 23.012 | 23.803 | 24.105 | +0.791 | +1.093 |
| 5ch | 未耗尽 | 20.273 | 22.607 | 22.818 | +2.333 | +2.545 |
| 5ch | 耗尽 | 22.531 | 23.491 | 23.724 | +0.960 | +1.193 |
| 7ch | 未耗尽 | 19.321 | 22.185 | 22.354 | +2.864 | +3.034 |
| 7ch | 耗尽 | 22.384 | 23.387 | 23.583 | +1.003 | +1.199 |

“未耗尽参考 vs 未耗尽解析”和“耗尽参考 vs 耗尽解析”都是同口径对照；
不能拿未耗尽解析去减耗尽波形。

### 8.2 闭合量重算

解析侧的口径差（同一波形下，耗尽解析 − 未耗尽解析）仍为：

| 数据集 | hybrid | H3 |
|---|---:|---:|
| 3ch | +0.545 | +0.559 |
| 5ch | +0.884 | +0.906 |

这两个值与本文 §3.2 的 500 m 结果一致，因为它是纯解析量，不依赖波形。
变化的是**残差**：

- 3ch：旧 500 m 残差 +2.056 → 收敛未耗尽 +1.594 → 收敛耗尽 +0.791。
  3ch 的 500 m 参考本来已收敛，变化主要来自把波形从“未耗尽”换成
  “耗尽”物理参考。
- 5ch：旧 500 m 残差 +3.848 → 收敛未耗尽 +2.333 → 收敛耗尽 +0.960。
  5ch 同时受旧参考 1.53 dB 步长伪影和未耗尽口径影响，因此变化显著大于
  3ch。
- 7ch：旧 500 m 残差 +5.751 → 收敛未耗尽 +2.864（步长修正 1.67 dB）→
  收敛耗尽 +1.003（口径对齐再修正 1.86 dB）。加入 7ch 耗尽数据后，
  +2.864 dB 不应再被当作模型误差。

### 8.3 结论修正

- 旧“前向仍有 1.5–4.5 dB 残差”的判断基于未耗尽波形参考；在耗尽参考下，
  3/5ch hybrid 的残差是 **+0.79/+0.96 dB**，已进入 ±1.0 dB；
- H3 的 +1.09/+1.19 dB 比 hybrid 多约 0.2–0.3 dB，来自 WP2 已定位的
  默认 EGN 乐观偏差，不是新的泵浦口径问题；
- 1ch 耗尽参考前向 0.2W：hybrid +1.13 dB、H3 +1.68 dB，与“单信道 SPM
  在强前向增益下仍偏乐观”一致；
- 7ch 前向耗尽参考 0.2W：hybrid +1.00 dB、H3 +1.20 dB；0.1W hybrid −0.08 dB。
  7ch 进入前向校准候选，但 H3 仍受默认 EGN 影响略超 ±1.1 dB。
- **WP6（2026-11-04）**：H3 默认改为 `egn="auto"`（前向泵关闭 EGN，无泵/
  后向保持开启）后，前向 0.2W 的 H3 残差实测降到 3ch +0.60、5ch +0.78、
  7ch +0.83 dB（1ch +0.95 dB），全部进入 ±1.0 dB；无泵/后向与旧默认逐位
  一致。三档表与逐位一致检查见 `docs/h3-forward-audit.md` §11 与
  `results/raman/h3_egn_auto.json`。

产物：`results/raman/waveform_compare_converged.json`、
`results/raman/waveform_compare_converged_depleted.json`；口径与完整表见
`docs/dra-waveform-compare.md` §13。

统计与功率依赖：单次 256-symbol 实现的 1σ 统计预算为 0.3–0.6 dB
（4-seed，见 `docs/dra-open-issues.md` §3），占 ±1.0 dB 带的一半量级；
前向残差还随发射功率变化（3ch 0.5/1/2 mW：hybrid +1.19/+0.79/+0.17 dB），
因此 §8 的 +0.79/+0.96/+1.00 dB 只在 1 mW/pol、前向 0.2 W 下成立，
不能外推到其它功率。
