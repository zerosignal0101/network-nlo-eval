# DRA 校准开放问题收口

日期：2026-11-05
工作树：`feat/dra-wideband-compare`（从 `multi-model-dev` `c871d11`）
脚本：`scripts/dra_open_issues.py`、`scripts/dra_wideband_compare.py`

> **2026-11-05 更新**：偏振驱动的正确实现（drive/state 分离）与 factor=2
> 对照在 `feat/dra-poldrive`（从 `multi-model-dev` `cc0ab23`）完成；
> §1.1、§5、§6 已加入新数字。旧的 `_paper_span_center(drive_factor=2)`
> 只保留为历史只读探针。
>
> **2026-11-06 光子因子修正**：本 worktree `feat/dra-photon-factor`（基于
> `multi-model-dev` `e1ee4af`）修正了 `RamanPaperSpanSolver` 负耦合的频率
> 因子。§8 给出论文口径、光子数守恒判据、修正前后 c57/cl152 残差；§1.1/§5
> 的旧数字仍是修正前口径，不能与 §8 的 after 列混用。

本文收口 HANDOFF v2 §6 的三个开放问题：发射功率依赖、1ch 前向、
单次实现的统计预算；同时给出偏振耗尽驱动（drive/state 分离，factor=1/2）
与 152ch 解析一致性。
所有数字都由本仓脚本在当前工作树生成，产物在 `results/raman/`。

口径统一为：

- 残差 = 解析 `SNR_NLI` − 波形 `SNR_NLI`（中心信道，dB），正数表示解析乐观；
- 波形参考是 62.5 m 收敛耗尽参考
  （`data/dra_waveform_converged/dra_reference_depleted.json`）；
- hybrid = `hybrid_raman_paper(nli_form="paper")`，H3 =
  `pcfm_h3_raman(egn="auto")`，数值 = `numerical_dra`；
- 有泵配置统一 `pump_depletion=True`；偏振驱动有两套口径：factor=1
  （`signal_polarizations=1`，历史每偏振功率驱动，所有旧结果默认口径）
  与 factor=2（`signal_polarizations=2`，双偏振总功率驱动泵浦耗尽）。
  本文 §1.1 给出两套数字，不要混用。

## 1. 发射功率扫描（3ch，前向 0.2 W）

数据来自 `dra_reference_depleted.json` 的 `3ch_0p5mw/1p0mw/2p0mw`。
`scripts/dra_open_issues.py` 用同一中心信道口径重跑解析模型，结果在
`results/raman/launch_power_analysis.json`。

| 发射功率 | 波形 | hybrid | H3 | numerical | hybrid 差 | H3 差 | numerical 差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 mW/pol | 28.374 | 29.561 | 29.366 | 29.477 | +1.187 | +0.992 | +1.103 |
| 1.0 mW/pol | 23.012 | 23.803 | 23.615 | 23.732 | +0.791 | +0.602 | +0.720 |
| 2.0 mW/pol | 18.089 | 18.256 | 18.135 | 18.208 | +0.167 | +0.046 | +0.119 |

残差随发射功率单调下降，0.5→2 mW 下降约 1.0 dB。也就是说，
"前向 0.2 W 下 +0.8…+1.2 dB"不是一个固定偏差，而是功率相关的斜率差。

NLI 的 log-log 拟合指数（中心信道 NLI 对每偏振发射功率）：

| 对象 | NLI 指数 |
|---|---:|
| 波形 | 2.461 |
| hybrid | 2.878 |
| H3 | 2.865 |
| numerical | 2.872 |

解析三者彼此只差 0.013，说明这是 hybrid/H3/numerical 共有的口径差，
不是 H3 的 EGN 项（`egn="auto"` 在前向泵下已关闭）。波形指数 2.46 比
解析低约 0.42，与"波形的前向拉曼增益在高功率下被信号耗尽得更明显、
解析 ODE 的耗尽响应偏弱"一致；这是解释，不是已分离的结论。

注意指数差不能直接换算成残差斜率：解析 SNR 用入纤功率作参考，波形
SNR 用接收功率，后者还含随功率变化的净增益。实测残差漂移约
−1.6…−1.7 dB/decade（hybrid/H3/numerical 一致），应以这个实测值为准。

### 1.1 偏振耗尽驱动：整体乘 2 的旧负结果与 drive/state 分离复核

**历史探针（只读，口径已被下面的新实现取代）**：假设“解析泵浦耗尽只用了
单偏振信号功率，应该乘 2”，`scripts/dra_open_issues.py::_paper_span_center`
把 `solver.solve(drive_factor * power)` 整体乘 2。这个探针同时放大了状态
功率和驱动功率，不是严格的驱动口径；旧表如下（使用含时域增益 bug 的
波形，仅作历史记录）：

| 数据集 | 驱动 | 波形 | 解析 | 残差 | 相对 1× 变化 |
|---|---|---:|---:|---:|---:|
| 3ch | 1× | 23.012 | 23.803 | +0.791 | — |
| 3ch | 2× | 23.012 | 24.277 | +1.264 | +0.474 |
| 5ch | 1× | 22.531 | 23.491 | +0.960 | — |
| 5ch | 2× | 22.531 | 24.203 | +1.672 | +0.712 |

当时的结论“2× 让残差增大，所以不是偏振口径 bug”只在 3/5ch、且波形含
时域增益 bug 的条件下成立。物理侧修正增益施加后，1× 残差变为 3ch
**+0.538**、5ch **+0.252** dB，7ch **−1.012** dB、57ch **−3.21** dB；
残差随信道数由正变负，“固定 2×”不是全信道数通用的修正。

**新实现（本分支 `feat/dra-poldrive`）**：`RamanPaperSpanSolver` 增加
`signal_polarizations`（默认 1）：

* 状态仍是每偏振功率 `P_s,i`，泵浦功率不变；
* 耦合项用驱动功率 `C @ drive`，信号槽 `drive_i = signal_polarizations *
  P_s,i`，泵浦槽不变；
* 信号行 `dP_s,i/dz = P_s,i (C @ drive)_i`，泵浦行
  `dP_p/dz = P_p (C @ drive)_p`。接收波自己的每偏振功率只乘一次，
  因此不会像 `solve(2 * P)` 那样把状态本身也放大；
* `signal_polarizations=1` 与旧路径逐位一致（测试
  `test_signal_polarizations_one_is_bit_identical_to_the_default`）。
  该开关已打通到 `hybrid_raman_paper`、`pcfm_h3_raman` 与
  `DRASpanKernelBase`；`numerical_dra` 使用独立的 `solve_ivp` 求解器，
  本次未接入，其 factor=2 行仍是 factor=1 数值。

`dra_waveform_compare.py` / `dra_wideband_compare.py` 新增
`--signal-polarizations {1,2}`（默认 1）。factor=2 产物：
`results/raman/waveform_compare_converged_depleted_pol2.json`、
`wideband_compare_c57_pol2.json`、`wideband_compare_c57_seed20261105_pol2.json`、
`wideband_compare_cl152_pol2.json`。

**62.5 m 收敛耗尽参考（中心信道，残差 = 解析 − 波形）**：

| 数据集 | 配置 | 波形 | hybrid 1× | hybrid 2× | H3 1× | H3 2× |
|---|---|---:|---:|---:|---:|---:|
| 3ch | fwd 0.2W | 23.265 | +0.538 | **+1.012** | +0.350 | +0.833 |
| 5ch | fwd 0.2W | 23.239 | +0.252 | **+0.964** | +0.073 | +0.793 |
| 7ch | fwd 0.2W | 24.399 | −1.012 | **−0.111** | −1.186 | −0.279 |
| 3ch | bwd 0.2W | 36.452 | −0.296 | −0.292 | −0.055 | −0.050 |
| 5ch | bwd 0.2W | 36.046 | −0.501 | −0.494 | −0.322 | −0.314 |
| 7ch | bwd 0.2W | 35.992 | −0.821 | −0.811 | −0.673 | −0.663 |

上表的波形/解析值来自重生成后的 `data/dra_waveform_converged`；仓库里的
`results/raman/waveform_compare_converged_depleted.json` 仍是重生成前的
3/5/7ch 数值（3ch fwd 波形 23.012 dB），本次没有覆盖它。

无泵配置的 1× 与 2× 在中心信道逐位相同（变化 0.000 dB）；关闭
`include_isrs` 时 ODE 只剩衰减，两口径严格逐位一致。开启 ISRS 时，2×
只把信道-信道耦合的源功率加倍（任务约定的“信号-信号源功率也用总功率”），
在 9ch 探针上相对功率变化 <1e-4，属于小量。

**c57（256 symbols，两个 seed 的残差均值）**：

| 配置 | hybrid 1× | hybrid 2× | H3 1× | H3 2× |
|---|---:|---:|---:|---:|
| nopump | −0.515 | −0.515 | −0.497 | −0.497 |
| bwd 0.2W | −0.496 | −0.429 | −0.542 | −0.472 |
| fwd 0.2W | −3.210 | **−1.243** | −3.405 | **−1.499** |

逐 seed 的 fwd hybrid 残差：seed 20261104 从 −4.062 到 −2.095，seed
20261105 从 −2.358 到 −0.391。方向与预期一致（57ch 向 0 靠拢，实测
−3.21 → −1.24，接近预期的约 −1.4 dB）。

**cl152（128 symbols，1× 采样）**：

| 配置 | hybrid 1× | hybrid 2× | H3 1× | H3 2× |
|---|---:|---:|---:|---:|
| nopump | +0.292 | +0.290 | +0.348 | +0.348 |
| bwd_sixpumps | −1.302 | **−0.130** | −1.864 | −0.370 |
| fwd 0.2W | +1.523 | **+2.299** | +1.350 | +2.120 |

结论：

1. **3/5ch 前向**按预期从 +0.54/+0.25 移到 +1.01/+0.96；**7ch** 从
   −1.01 移到 −0.11，基本回到 0。**57ch** 从 −3.21 移到 −1.24，是本次
   修正最明显的改善。
2. **后向单泵**（3/5/7ch、c57）变化只有 0.003–0.07 dB，符合物理预期；
   但 **cl152 六泵后向**不是小量：−1.302 → −0.130（+1.17 dB）。该配置
   的六台泵总功率 1.756 W、152ch 信号总功率更大，泵浦被信号显著耗尽，
   所以“双偏振总功率”驱动会明显改变泵浦剖面。这是数据集依赖，不是
   实现错误；它说明后向 DRA 的耗尽口径必须和泵浦数量/总功率一起看。
3. **cl152 前向**从 +1.523 移到 +2.299，方向与 3/5ch 相同（更乐观）。
   该数据是 1× 采样，按 §5.3 的约 1.4 dB 混叠偏差外推，alias-free 残差
   约 +0.9 dB。本次**没有**处理 1× 混叠；混叠与外推结论仍需 2× 数据
   确认。
4. 因此不能把 factor=2 设为默认：它修正了物理总功率口径，但固定 2× 在
   3/5ch 和 cl152 前向会把残差推得更正；剩余项仍包含 GN 适用域、
   信号-信号 ISRS、采样混叠和统计预算。默认保持 1 以保持所有既有结果
   逐位不变，factor=2 作为显式对照开关。

## 2. 1ch 前向

1ch 是纯 SPM 探针，用来判断前向 +1.1 dB 是固定 GN 偏差还是功率相关项。
波形由 `physics_nlo_eval.DistributedRamanFiber(pump_model="depleted")`
生成：1ch、100 km、50 GBd、100 GHz 采样率、62.5 m、1450 nm 0.2 W 前向、
256 symbols、seed 20261104；NLI 用与参考相同的 50 GHz 单复标量投影。
结果在 `results/raman/one_ch_forward.json`。

自检：1 mW/pol 的 1ch 波形与参考逐位一致（无泵 38.043291985，
前向 23.703940949，差 0.0 dB），说明 TX/传播/NLI 口径复现无误。

| 发射功率 | 配置 | 波形 | hybrid | H3 | hybrid 差 | H3 差 |
|---:|---|---:|---:|---:|---:|---:|
| 0.25 mW | 无泵 | 50.061 | 49.910 | 50.502 | −0.151 | +0.441 |
| 0.5 mW | 无泵 | 44.048 | 43.890 | 44.481 | −0.159 | +0.433 |
| 1.0 mW | 无泵 | 38.043 | 37.869 | 38.461 | −0.174 | +0.417 |
| 2.0 mW | 无泵 | 32.054 | 31.848 | 32.407 | −0.206 | +0.354 |
| 0.25 mW | 前向 0.2 W | 35.288 | 36.738 | 36.547 | +1.450 | +1.259 |
| 0.5 mW | 前向 0.2 W | 29.423 | 30.764 | 30.575 | +1.341 | +1.152 |
| 1.0 mW | 前向 0.2 W | 23.704 | 24.836 | 24.650 | +1.132 | +0.946 |
| 2.0 mW | 前向 0.2 W | 18.249 | 18.994 | 18.854 | +0.745 | +0.605 |

结论：

- 无泵残差几乎不随功率变化（hybrid −0.15…−0.21，H3 +0.35…+0.44），
  是固定 GN 偏差，与文献一致；
- 前向残差从 0.25 mW 的 +1.45 单调降到 2 mW 的 +0.75，**不是固定 GN
  偏差**；方向和 3ch 扫描相同（功率越高残差越小）；
- 前向 NLI 指数：波形 2.804，hybrid 2.965，H3 2.960；无泵三者为
  2.994/3.000/3.003。前向的斜率差 +0.16 与残差随功率下降的方向一致；
- 单次 256-symbol 实现的统计误差约 0.3–0.6 dB（§3），但四个功率点上的单调
  趋势（0.7 dB）大于统计噪声，因此可以排除"纯统计项"；剩余项与
  前向耗尽/增益剖面的功率响应一致，不能仅凭这组数据区分"耗尽模型"
  和"GN 适用域边界"；
- 实用结论：1ch 前向在 1 mW/pol 的 +1.13/+0.95 dB 应标注为 GN 适用域
  边界，而不是固定偏差；低功率（≤0.5 mW）会更差，高功率（2 mW）回到
  ±0.75 dB。

## 3. seed 集成与统计预算

对 3ch（500 m；其 62.5 m 步长伪影只有 0.07 dB）和 5/7ch（62.5 m）
各跑 4 seeds × {无泵, 前向 0.2 W, 后向 0.2 W}：256 symbols、1 mW/pol、
100 GHz 栅格、耗尽泵浦。结果在 `results/raman/seed_ensemble.json`。

| 数据集 | 步长 | 配置 | mean (dB) | std (dB) | sem (dB) | 参考单次 − 集成均值 |
|---|---:|---|---:|---:|---:|---:|
| 3ch | 500 m | 无泵 | 37.040 | 0.353 | 0.176 | −0.331 |
| 3ch | 500 m | 前向 | 23.365 | 0.502 | 0.251 | −0.352 |
| 3ch | 500 m | 后向 | 36.745 | 0.331 | 0.166 | −0.290 |
| 5ch | 62.5 m | 无泵 | 36.437 | 0.509 | 0.254 | −0.107 |
| 5ch | 62.5 m | 前向 | 22.760 | 0.278 | 0.139 | −0.229 |
| 5ch | 62.5 m | 后向 | 36.184 | 0.466 | 0.233 | −0.141 |
| 7ch | 62.5 m | 无泵 | 36.142 | 0.627 | 0.313 | +0.132 |
| 7ch | 62.5 m | 前向 | 22.577 | 0.515 | 0.257 | −0.193 |
| 7ch | 62.5 m | 后向 | 35.889 | 0.610 | 0.305 | +0.013 |

说明：

- 3ch 的 std 为 0.33–0.50 dB，与之前 5-seed 的 0.31–0.45 dB 一致；
  5/7ch 为 0.28–0.63 dB。4 个 seed 的 std 自身有约 40% 的不确定度，
  这些数只用于量级判断；
- 参考里的单次实现与 4-seed 均值相差 0.01–0.35 dB；3ch 的
  −0.29…−0.35 dB 里还含 500 m 相对 62.5 m 的 +0.07…+0.08 dB
  步长伪影（seed 20261104 在 500 m 与 62.5 m 的差），不是统计量；
- 因此 ±1.0 dB 校准带里，单次实现的 1σ 统计预算约 0.3–0.6 dB，
  量级上约占一半。任何小于 ~0.3 dB 的模型差异都不应据单次实现下结论；
- 5ch/7ch 在 62.5 m 的 seed 20261104 与参考逐位一致（差 0.0），
  说明传播与 NLI 口径没有漂移。

## 4. 152ch 解析一致性与 runtime

背景：WP7 的 57/152ch 波形参考尚未到位，本轮**没有**全 WDM 波形数字，
本节只验证解析模型在 152ch 规模上互相一致、并测 runtime。配置与
`dra_wideband_compare.py` 的默认一致：`SpectrumGrid(152, 75e9, 190.15e12)`、
64 GBd、B = 符号率 64 GHz、D=17、S=0.06、α=0.162 平坦、A_eff=80、
n2=2.6e-20、无泵、1 mW/pol、单跨、EDFA NF 5.5 dB。WP7 的 fiber 若只给
`beta2_ps2_per_km=-21.7`，脚本会换算成 D≈17.014；本表用 D=17.0，
两者中心信道差 0.003 dB。

| 模型 | 中心信道 (dB) | 全带均值 (dB) | 最小值 (dB) | 最大值 (dB) | runtime |
|---|---:|---:|---:|---:|---:|
| hybrid | 33.225 | 33.424 | 32.743 | 35.356 | 166.2 s |
| H3 | 33.303 | 33.444 | 33.077 | 35.254 | <0.1 s |
| numerical | 33.220 | 33.419 | 32.743 | 35.349 | 85.5 s |
| pcfm | 33.219 | 33.341 | 33.001 | 35.034 | <0.1 s |

- 中心信道四者极差 0.083 dB（33.219–33.303），全带均值极差 0.103 dB；
  与之前一次性探针的 0.086 dB 一致（绝对值整体低约 0.03–0.04 dB，
  来源未逐项核对；极差本身不受影响）；
- 边缘信道差异更大：H3 的最小值比 hybrid 高 0.33 dB，pcfm 最大值
  比 hybrid 低 0.32 dB；全 WDM 对照不能只看中心信道；
- runtime：152ch、单配置、单跨，hybrid 首次 166 s、numerical 85.5 s。
  三个配置的 c57/cl152 全套对照预计 hybrid 约 8 min、numerical 约
  4.5 min（152ch）；`dra_wideband_compare.py --skip-numerical` 可用于
  快速迭代。57ch 未单独计时；
- 结论：全 WDM 解析对照可行；瓶颈是 hybrid/numerical，不是 H3/pcfm。

上表可用 `dra_wideband_compare.evaluate_analytic` 直接复现（无需 WP7 波形）：

```bash
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python - <<'PY'
import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.network.elements import FiberSpanConfig

from dra_wideband_compare import evaluate_analytic

fiber = FiberSpanConfig(
    length_km=100.0, attenuation_db_km_ref=0.162, dispersion_parameter_d=17.0,
    dispersion_slope_s=0.06, effective_area_um2_ref=80.0, nonlinear_index_n2=2.6e-20,
)
grid = SpectrumGrid(152, 75e9, 190.15e12)
for model in ("hybrid_raman_paper", "pcfm_h3_raman", "numerical_dra", "pcfm"):
    sig, nli = evaluate_analytic(model, grid, fiber, [], symbol_rate_hz=64e9, launch_power_w=1e-3)
    print(model, 10 * np.log10(sig[76] / nli[76]))
PY
```


## 5. 全 WDM 57/152ch 波形对照（WP7 数据）

### 5.1 数据与口径

WP7 数据复制到 `data/dra_waveform_wide/`：

| 文件 | 计划 | symbols | seed | 配置 |
| --- | --- | ---: | ---: | --- |
| `c57.{json,npz}` | 57ch | 256 | 20261104 | `nopump`、`fwd_0p20w`、`bwd_0p20w` |
| `c57_seed20261105.{json,npz}` | 57ch | 256 | 20261105 | 同上 |
| `cl152.{json,npz}` | 152ch | 128 | 20261104 | `nopump`、`bwd_sixpumps`、`fwd_0p20w` |

`dra_wideband_compare.py` 用 `SpectrumGrid(n, 75e9, center)`、64 GBd、
`channel_bandwidth_source="symbol_rate"`（B=64 GHz）、`pump_depletion=True`。
两个必须注意的口径：

1. **色散斜率按 β₃ 反解**。波形用 β₃=1e-4 ps³/km，按 network 的
   `_calc_beta3_from_s_d` 反解等价于 S≈−0.0219 ps/(nm²·km)。脚本优先
   从 JSON 的 `beta3_ps3_per_km` 反解；用默认 S=0.06（β₃≈0.133）
   会把 cl152 中心 SNR 抬高 0.46 dB（32.765 → 33.225）。
2. **波形 NLI 是接收端残差**：每信道 64 GHz 带内、单复标量公共增益
   拟合后的 `NL−a·LIN` 功率；解析侧是 TX 平面 GN NLI。只比较
   SNR/残差趋势，不比较绝对 NLI。

### 5.2 c57（57ch，256 symbols，两个 seed）

残差 = 解析 − 波形（dB，正值表示解析更乐观）；中心信道：

| 配置 | 波形 seed 20261104 | 波形 seed 20261105 | 两 seed 均值 | 解析 hybrid | 残差 seed1 | 残差 seed2 | 均值残差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nopump` | 34.588 | 32.709 | 33.649 | 33.133 | −1.455 | +0.424 | **−0.516** |
| `bwd_0p20w` | 34.373 | 32.479 | 33.426 | 32.930 | −1.443 | +0.451 | **−0.496** |
| `fwd_0p20w` | 28.732 | 27.028 | 27.880 | 24.670 | −4.062 | −2.358 | **−3.210** |

> **2026-11-05 增益施加修正**：旧表（bwd 32.221/30.683、fwd 20.131/19.638）
> 使用把频变拉曼增益逐样本乘在时域的波形；57ch 全带宽内增益谱变化约
> 5 dB，时间域调制在强前向增益下被放大，造成随记录长度的虚假 NLI。
> 修正后 nopump 逐位不变，bwd 抬高约 +2.0 dB，fwd 抬高约 +8 dB。
> 旧的“forward 是周期记录伪影、不可用”结论**作废**；前向现在可用于
> 绝对对照，但残差为负。物理侧细节见 `docs/dra-waveform.md` §3/§12。

H3 auto 与 hybrid 趋势一致（均值残差：nopump −0.50、bwd −0.54、
fwd −3.41 dB）；numerical 也一致（−0.52、−0.58、−3.27）。

**factor=2（双偏振总功率驱动）对照**（同样的两 seed 波形，残差均值）：

| 配置 | hybrid 1× | hybrid 2× | H3 1× | H3 2× |
|---|---:|---:|---:|---:|
| `nopump` | −0.515 | −0.515 | −0.497 | −0.497 |
| `bwd_0p20w` | −0.496 | −0.429 | −0.542 | −0.472 |
| `fwd_0p20w` | −3.210 | **−1.243** | −3.405 | **−1.499** |

逐 seed 的前向 hybrid 残差：seed 20261104 从 −4.062 到 −2.095，seed
20261105 从 −2.358 到 −0.391；后向只变化 +0.07 dB，无泵不变。也就是说，
57ch 前向的 −3.21 dB 里约 2.0 dB 可以由“信号驱动少乘了一个偏振”
解释，剩余约 −1.2 dB 仍留给 GN 适用域/统计项。详见 §1.1。

- **nopump**：两 seed 均值与解析差 **−0.52 dB**，落在 ±1 dB；但两个
  256-symbol seed 相差 1.88 dB，单次实现不能用于 0.1 dB 级比较。
- **backward**：均值 **−0.50 dB**，落在 ±1 dB；修正前为 +1.48 dB。
- **forward**：两 seed 均值 27.880 dB，对解析 24.670 dB 的残差为
  **−3.21 dB**（解析保守），seed 散布 1.70 dB。这与 3ch (+0.54)、
  5ch (+0.25)、7ch (−1.01) 构成随信道数单调下降的序列：总功率越高，
  解析模型越保守。最可能的口径项是泵浦耗尽的偏振驱动（波形用双偏振
  总功率，解析 ODE 用每偏振功率）；但固定的 2× 驱动在 3/5ch 会把残差
  推得更正，说明还叠加了强前向增益下的 GN 适用域偏差。**这是当前最大的
  开放问题**，详见 §1.1 的复核。注意这些是 1× 数据；实测 2× 同 seed
  的 fwd 偏差只有 +0.35…+0.53 dB（bwd +0.5…+0.83 dB），所以
  alias-free 的 forward 残差是 **−3.65 dB**（两 seed 均值，单 seed
  −4.59/−2.71）、backward **−1.17 dB**（单 seed −2.28/−0.06）。
  再用 factor=2（总功率驱动）的解析值组合 2× 波形，得到 backward
  **−1.10**、forward **−1.68 dB**（H3 −1.15/−1.94）；剩余与
  1–2 dB 的 seed 散布同量级。
- nopump 的记录长度/seed 序列（64/128/256/512 symbols，两个 256 seed）
  为 32.370/32.825/34.588/32.709/33.722 dB，均值 **33.24 dB、标准差
  0.90 dB**，没有单调趋势；这是统计散布，不是系统偏差。
- **混叠**：1× 采样率会高估 NLI。c57 64-symbol 0.488 m 从 32.370
  (1×) 升到 32.783 (2×)，+0.41 dB；256-symbol 从 34.588 升到
  35.476，+0.89 dB。2× 数据更接近连续谱物理。

### 5.3 cl152（152ch）

主数据（128 symbols、0.244 m、1×、seed 20261104）的逐模型对照：

| 配置 | 波形中心 | hybrid | 残差 | H3 auto | 残差 | pcfm | 残差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nopump` | 32.473 | 32.765 | **+0.292** | 32.821 | **+0.348** | 32.737 | +0.272 |
| `bwd_sixpumps` | 32.010 | 30.708 | **−1.302** | 30.146 | **−1.864** | 33.746 | +0.735 |
| `fwd_0p20w` | 28.376 | 29.900 | **+1.523** | 29.726 | **+1.350** | 32.745 | +4.369 |

> **2026-11-05 增益修正后重生成**：旧值 bwd 29.720 dB / fwd 16.146 dB
> （残差 +0.99/+0.43 和 +13.75/+13.58）作废。修正后 nopump 不变，
> bwd 抬高 2.29 dB、fwd 抬高 12.23 dB。1× 主数据仍含约 1.4 dB 的
> **同 seed** 混叠偏差（1× 高估 NLI）；bwd/fwd 的 2× 数据尚未生成。
>
> **统计修正**：2× 128-symbol nopump 扩到 4 seed 后为 33.502 / 34.247 /
> 23.178 / 31.422 dB，均值 **30.587 ± 5.082 dB**、中位数 32.462 dB。
> 之前的 2-seed 33.874 ± 0.53 dB 只用了两个偏高 seed，**不能当作收敛
> 基准**；对解析 32.765 dB 的残差按均值 +2.18 dB、按中位数 +0.30 dB，
> 但被 5 dB 的 seed 散布主导。旧的 −1.11 dB alias-free 残差同样不可
> 引用。cl152 的绝对 NLI 需要更多 seed 或更长记录才能收敛。

- `nopump` 与 `fwd_0p20w` 的 hybrid/H3 残差在 **±1.5 dB** 内（1×）；
  修正后的前向从 +13.75 dB 回到 +1.52 dB，已恢复可用。
- `bwd_sixpumps` 的 hybrid 残差 −1.30 dB、H3 −1.86 dB，H3 已超出
  ±1.5 dB；叠加 ~1.4 dB 同 seed 混叠修正后更负。这是新的开放问题
  （六泵后向的增益/耗尽剖面），pcfm 的 +0.74 dB 仍是 lumped Raman
  口径的离群值。

**factor=2（双偏振总功率驱动）对照**（同一 1× 主数据）：

| 配置 | 波形中心 | hybrid 1× | hybrid 2× | H3 1× | H3 2× |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nopump` | 32.473 | +0.292 | +0.290 | +0.348 | +0.348 |
| `bwd_sixpumps` | 32.010 | −1.302 | **−0.130** | −1.864 | **−0.370** |
| `fwd_0p20w` | 28.376 | +1.523 | **+2.299** | +1.350 | +2.120 |

- 六泵后向是本次最反直觉的一项：factor=2 把 hybrid 残差从 −1.30 拉到
  −0.13 dB（H3 从 −1.86 到 −0.37）。该配置六台泵总功率 1.756 W，
  152ch 信号总功率更大，泵浦被信号显著耗尽，所以“信号驱动是否含双偏振”
  会明显改变泵浦剖面；这不是后向泵都应该不敏感的例外，而是功率/泵数
  依赖。
- 前向残差从 +1.52 移到 +2.30 dB，方向与 3/5ch 一致。1× 混叠（约
  1.4 dB，本次未处理）按 §5.3 外推后 alias-free 约 +0.9 dB。

步长与混叠（nopump，64 symbols，中心信道，**单 seed 20261104**）：

| 采样率 | 0.977 m | 0.488 m | 0.244 m | 0.122 m |
| --- | ---: | ---: | ---: | ---: |
| 1× (11.52 THz) | 30.120 | 33.216 | 34.226 | 34.679 |
| 2× (23.04 THz) | 31.434 | 34.481 | 35.635 | — |
| 3× (34.56 THz) | 31.432 | — | — | — |

- 2× 与 3× 在 0.977 m 只差 **0.003 dB**（同一 seed），说明 2× 在固定
  输入下已消除混叠；1×→2× 偏差 +1.31/+1.27/+1.41 dB，1× 系统性
  高估 NLI 约 1.1–1.4 dB。
- **64-symbol 的 seed 散布极大**：cl152 nopump 2×、0.244 m 的 4 个
  seed 为 35.635 / 26.216 / 24.446 / 33.962 dB，均值
  **30.06 ± 5.56 dB**（极差 11.19 dB）。35.635 是 4 个中的最大值；
  之前由它做的三点外推 36.05 dB 不能引用。
- **128-symbol、2×、0.244 m 的 4 个 seed** 为 33.502 / 34.247 /
  23.178 / 31.422 dB，均值 **30.587 ± 5.082 dB**、中位数 32.462 dB。
  之前的 2-seed 33.874 ± 0.53 dB 只取了两个偏高 seed，**不稳健**；
  同 seed 的 1×→2× 偏差 +1.3…+1.4 dB 仍然成立，但绝对 NLI 被
  ~5 dB 的 seed 散布主导。
- 对解析 32.765 dB 的残差：4-seed 均值 **+2.18 dB**、中位数
  **+0.30 dB**。旧的 −1.11 dB alias-free 残差同样不可引用；cl152
  的绝对 NLI 需要更多 seed 或更长记录才能收敛。
- **2× 泵浦实测（2026-11-05，每配置 2 seed）**：同 seed 的 1×→2×
  偏差为 bwd +1.00/+1.27 dB、fwd +0.78/+0.60 dB，**明显小于 nopump
  的 +1.3…+1.4 dB**，不能套用 nopump 的混叠修正。2× factor=1 残差
  均值：bwd **−2.92 dB**（单 seed −2.30/−3.54，超出 ±1.5）、fwd
  **+1.26 dB**（单 seed +0.74/+1.77，均值在 ±1.5 内、seed 20261105
  略超）。旧的 alias-free 外推 −2.7/+0.1 dB 作废。
- **派生 factor=2 × 实测 2× 波形**（解析值与采样率无关）：
  bwd **−1.75 dB**（H3 −1.99）、fwd **+2.03 dB**（H3 +1.85）。
  backward 改善、forward 变正，剩余 0.3–0.5 dB 超出 ±1.5 的部分与
  1–2 dB 的泵浦 seed 散布同量级；cl152 nopump 的 5 dB 统计散布
  仍未解决，不能把这些值当收敛基准。
- **调制口径（收敛控制）**：2×、4 seed 的 Gaussian−QPSK NLI 比为
  cl152 **+1.47 ± 5.70 dB**（中位数 +2.42）、c57 **+4.70 ± 3.09 dB**
  （中位数 +3.97）；配对 NLI 均值之比 4.28–4.29 dB。旧 1×/64-symbol
  单次探针的 **7.0 dB 是重尾中的一个高实现**，不是口径常数。最佳
  中央估计约 **+4 dB**，带 3–6 dB 实现不确定度，与四阶矩预期同量级。
  这是 nopump 口径，不能外推到泵浦。

### 5.4 结论

1. **57ch**：修正增益施加后，nopump 两 seed 均值 −0.52 dB、
   backward −0.50 dB，都在 ±1 dB；forward 均值 **−3.21 dB**（解析保守），
   seed 散布 1.70 dB，且残差随信道数变负（3ch +0.54 → 57ch −3.21 dB）。
   factor=2（双偏振总功率驱动）把 forward 均值拉到 **−1.24 dB**
   （H3 −1.50 dB），backward 只变 +0.07 dB，nopump 不变。
2. **152ch**：修正后 1× 主数据 nopump +0.29/+0.35 dB（hybrid/H3）、
   backward **−1.30/−1.86 dB**、forward **+1.52/+1.35 dB**；
   factor=2 下 backward 变为 **−0.13/−0.37 dB**、forward 变为
   **+2.30/+2.12 dB**，nopump 不变。2× 128-symbol nopump 扩到 4 seed
   后为 **30.587 ± 5.082 dB**（中位数 32.462）；对解析 32.765 dB 的
   残差按均值 +2.18、按中位数 +0.30 dB，被 5 dB 的 seed 散布主导。
   旧的 2-seed 33.874 dB / −1.11 dB 不稳健，不能引用。2× 泵浦实测
   factor=1 残差为 bwd **−2.92 dB**（超出 ±1.5）、fwd **+1.26 dB**
   （均值在带内）；派生 factor=2 × 2× 为 bwd **−1.75**、fwd **+2.03**。
   泵浦 fwd 的同 seed 1×→2× 偏差只有 +0.6…+0.8 dB，旧外推 −2.7/+0.1
   作废。绝对残差引多 seed 分布，64-symbol 单次值不可用。
3. **混叠**：1× 采样率系统性高估 NLI（c57 +0.4…0.9 dB，cl152
   +1.1…1.4 dB）；2× 在固定输入下已足够（3× 与 2× 差 0.003 dB）。
   后续全 WDM 参考应默认 2×。
4. **前向泵全 WDM 已恢复可用**：旧的记录长度发散是物理侧时域增益
   施加 bug，不是周期记录伪影。修正后 c57 forward 64→1024 symbols
   的频谱 SNR 为 26.93/26.91/28.73/27.95/27.41 dB，无单调下降
   （EVM 同步）；256-symbol 主参考 28.732/27.028 dB。cl152 forward
   从 16.15 dB 修正到 28.376 dB（1×，128 symbols），hybrid 残差
   +1.52 dB。旧 `c57_forward_record_evm` 与 `c57_forward_long_record_*`
   已标记 superseded。前向残差随信道数变负的问题现在大部分由
   双偏振总功率驱动解释（57ch −3.21 → −1.24 dB），见 §1.1；剩余
   −1.2 dB 仍未归因。
5. **统计预算**：c57 nopump 5 次测量标准差 0.90 dB；cl152 64-symbol
   4-seed 标准差 **5.56 dB**、128-symbol 2-seed 标准差 **0.53 dB**。
   绝对 NLI 必须用 128-symbol（或更长）多 seed 集成；64-symbol 只能
   用于同一 seed 的步长趋势。

## 6. 对校准结论的影响

1. 前向泵的 +0.8…+1.2 dB 残差应表述为"1 mW/pol、0.2 W 前向"下的值，
   并注明随发射功率变化（0.5 mW 更差、2 mW 更好）。不要把它外推成
   任意功率下的固定偏差；
2. 偏振驱动有两套口径，不能混用：factor=1（`signal_polarizations=1`，
   历史每偏振驱动）保持所有旧结果；factor=2（双偏振总功率驱动）是物理
   波形口径，57ch 前向改善约 2.0 dB、3/5ch 与 cl152 前向反而更乐观、
   7ch 回到约 0。固定 factor=2 不是通用修正，**默认仍为 1**；
3. 单次实现 0.3–0.6 dB 的统计预算意味着 ±1.0 dB 校准带里约一半是统计量，
   0.1–0.2 dB 级的模型排序需要多 seed 或更长符号记录；
4. 1ch 前向是 GN 适用域边界，不是新的实现 bug；
5. 57/152ch 的波形残差已由 `scripts/dra_wideband_compare.py` 跑出：
   nopump/backward 在 ±1.5 dB 内（cl152 hybrid/H3 在 ±1 dB 内），
   forward 在修正物理侧增益 bug 后可用；全 WDM 单次实现的统计预算是
   2–3 dB。详见 §5。

## 7. 复现

```bash
cd /home/zsig/Documents/gits/wt/dra-wideband-compare
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python scripts/dra_open_issues.py
# 只跑其中一段：
#   --skip-one-ch --skip-ensemble
#   --skip-launch --skip-ensemble
#   --skip-launch --skip-one-ch

PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python -m pytest \
  tests/test_dra_open_issues.py tests/test_dra_wideband_compare.py -q
```

产物：

- `results/raman/launch_power_analysis.json`
- `results/raman/one_ch_forward.json`
- `results/raman/seed_ensemble.json`

全 WDM 数据到位后：

```bash
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python scripts/dra_wideband_compare.py \
  --reference data/dra_waveform_wide/c57.json \
  --output results/raman/wideband_compare_c57.json
```

偏振驱动 factor=2（新产物，不覆盖 factor=1）：

```bash
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python scripts/dra_waveform_compare.py \
  --data-dir data/dra_waveform_converged --reference dra_reference_depleted.json \
  --signal-polarizations 2 \
  --output results/raman/waveform_compare_converged_depleted_pol2.json

for ref in c57 c57_seed20261105 cl152; do
  extra=""
  [ "$ref" = "cl152" ] && extra="--skip-numerical"
  PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python scripts/dra_wideband_compare.py \
    --reference "$ref.json" --signal-polarizations 2 $extra \
    --output "results/raman/wideband_compare_${ref}_pol2.json"
done
```

注：`numerical_dra` 使用独立的 `solve_ivp` 求解器，本次没有接入
`signal_polarizations`；上述 wideband 文件里它的行仍是 factor=1 数值。

限制：本文的统计量都基于 256-symbol、单/四个 seed；耗尽模型仍用中心
频率的带内总功率驱动泵浦，未建模泵浦-泵浦拉曼转移；1ch 的 TX 复用了
3/5/7ch 的 50 GBd/100 GHz 口径，不是 WP7 的 64 GBd 全 WDM 口径。

## 8. 光子因子审计（2026-11-06）

### 8.1 论文口径

Buglia 2024 Eq. (2)（学位论文 Eq. (2.39)）对接收波 `i` 写为

```text
∂P_i/∂z = -Σ_{k>i} (f_k/f_i) g_i P_k P_i
          -Σ_{p: f_i>f_p} (f_p/f_i) g_i P_p P_i
          +Σ_{k<i} g_i P_k P_i
          +Σ_{p: f_i<f_p} g_i P_p P_i - α_i P_i ,
```

信道按频率降序编号。负耦合的频率因子是 `f_source/f_receiver`。
`RamanPaperSpanSolver._build_coupling_matrix` 的旧实现逐位采用这个印刷式，
所以旧代码**忠实于论文**；但印刷式与标准光子数守恒方程互为倒数：

```text
C[j,k]/f_j + C[k,j]/f_k = 0    ⇔    d/dz Σ_j P_j/f_j = 0 .
```

若 `f_k > f_j`，标准形式是 `C[j,k] = +g`、`C[k,j] = -(f_k/f_j) g`，
即负耦合因子为 `f_receiver/f_source`。物理 `DistributedRamanFiber` 的
`-(f_p/f_s) γ P_p P_s` 和 Semrau 2019 JLT Eq. (6) 的 `-(f_i/f_k)` 都是
这个口径。结论是：**代码忠实实现了论文印刷式，但论文印刷式的负因子与
标准光子数守恒差 `(f_low/f_high)^2`；本次改代码，不改物理侧。**

修正位置：`src/network_nlo_eval/models/raman_ode.py::_build_coupling_matrix`，
负耦合由 `f_source/f_receiver` 改为 `f_receiver/f_source`。增益谱、`A_eff`
归一、偏振驱动、RK4/不动点结构未改。

### 8.2 违反量

用统一 `A_eff` 隔离频率因子；`旧相对违反 = 1 - (f_low/f_high)^2`：

| 计划 | 代表对 | `f_high/f_low` | 旧负因子 | 新负因子 | 旧相对违反 | 新相对违反 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| c57 | 1450 nm 泵 – 中心信号 | 206.753/193.750 THz | 0.9371 | 1.0671 | 0.1218 | <1e-15 |
| c57 | 信号带边 | 195.850/191.650 THz | 0.9786 | 1.0219 | 0.0424 | <1e-15 |
| cl152 | 1450 nm 泵 – 中心信号 | 206.753/190.113 THz | 0.9195 | 1.0875 | 0.1545 | <1e-15 |
| cl152 | 信号带边 | 195.813/184.488 THz | 0.9422 | 1.0614 | 0.1123 | <1e-15 |
| cl152 | 六泵相邻泵对 | 210.086–223.726 THz | 0.9530–0.9853 | 1.0149–1.0493 | 0.0254–0.1010 | <1e-15 |

实际求解器仍按接收波 `A_eff` 归一；逐波 `A_eff` 不同时，即使频率因子修正，
`C[j,k]/f_j + C[k,j]/f_k` 仍会残留 `O(ΔA_eff/A_eff)`。这不是频率因子，
本次按要求未改 `A_eff`。c57 fwd 的泵浦行聚合旧/新耦合比为 **0.878**
（旧少耗尽 12.2%）。

### 8.3 修正前后残差

解析模型 `hybrid_raman_paper`（`pump_depletion=True`，`nli_form="paper"`）。
残差 = 解析 − 波形，中心信道。1× 波形：c57 为 seed 20261104/20261105
均值，cl152 为 2× 诊断文件里 factor=1 的两 seed 均值；2× 波形：泵浦配置为
同两 seed 的 factor=2 均值，c57 nopump 为单 seed，cl152 nopump 为 2-seed
128-symbol 集成（该配置的 seed 散布达 5 dB，绝对值不作校准）。解析值
与采样率无关，因此 1×/2× 残差差只来自波形参考。

| 计划 | 配置 | f | 解析 SNR 前 → 后 | 1× 残差 前 → 后 | 2× 残差 前 → 后 |
| --- | --- | ---: | ---: | ---: | ---: |
| c57 | `bwd_0p20w` | 1 | 32.930 → 32.942 (+0.012) | -0.496 → -0.484 (+0.012) | -1.167 → -1.156 (+0.012) |
| c57 | `bwd_0p20w` | 2 | 32.997 → 33.010 (+0.013) | -0.429 → -0.416 (+0.013) | -1.100 → -1.087 (+0.013) |
| c57 | `fwd_0p20w` | 1 | 24.670 → 25.031 (+0.361) | -3.210 → -2.849 (+0.361) | -3.650 → -3.289 (+0.361) |
| c57 | `fwd_0p20w` | 2 | 26.637 → 27.011 (+0.375) | -1.243 → -0.869 (+0.375) | -1.684 → -1.309 (+0.375) |
| c57 | `nopump` | 1 | 33.133 → 33.133 (+0.000) | -0.515 → -0.515 (+0.000) | -2.343 → -2.343 (+0.000) |
| c57 | `nopump` | 2 | 33.133 → 33.133 (+0.000) | -0.515 → -0.515 (+0.000) | -2.343 → -2.343 (+0.000) |
| cl152 | `bwd_sixpumps` | 1 | 30.708 → 31.454 (+0.746) | -1.787 → -1.041 (+0.746) | -2.921 → -2.175 (+0.746) |
| cl152 | `bwd_sixpumps` | 2 | 31.880 → 32.204 (+0.324) | -0.615 → -0.291 (+0.324) | -1.748 → -1.425 (+0.324) |
| cl152 | `fwd_0p20w` | 1 | 29.900 → 30.055 (+0.156) | +1.946 → +2.102 (+0.156) | +1.254 → +1.410 (+0.156) |
| cl152 | `fwd_0p20w` | 2 | 30.676 → 30.825 (+0.149) | +2.722 → +2.871 (+0.149) | +2.030 → +2.179 (+0.149) |
| cl152 | `nopump` | 1 | 32.765 → 32.766 (+0.002) | +0.292 → +0.293 (+0.002) | -1.110 → -1.108 (+0.002) |
| cl152 | `nopump` | 2 | 32.763 → 32.766 (+0.003) | +0.290 → +0.292 (+0.003) | -1.112 → -1.109 (+0.003) |

产物：`results/raman/photon_factor_audit.json`（含每条的前后解析值、
波形值和两套残差）。

### 8.4 解读

1. **nopump**：c57 变化 <0.0005 dB；cl152 +0.002/+0.003 dB。这是信号-信号
   ISRS 负耦合的修正，量级可以忽略。`include_isrs` 默认开启，所以无泵配置
   并非严格逐位不变，但变化远小于统计预算。
2. **c57 fwd**：解析 SNR +0.361 dB（f=1）/+0.375 dB（f=2），残差从
   **−3.210 → −2.849**、**−1.243 → −0.869**（1×）；2× 同样改善 0.361/0.375 dB。
   方向正确：旧解析少耗尽泵浦，信号增益偏高、NLI 偏大、SNR 偏低；修正后
   泵浦输出 5.153e-5 → 4.090e-5 W（−20.6%），中心信号输出
   9.861e-5 → 9.158e-5 W（−7.1%），SNR_NLI 向波形靠拢。剩余 −2.85/−0.87 dB
   仍主要由偏振驱动（factor=2 可再收起约 2.0 dB）和 GN 适用域解释。
3. **c57 bwd**：+0.012/+0.013 dB，可忽略；后向泵的耗尽本来就弱。
4. **cl152 fwd**：+0.156 dB（f=1）/+0.149 dB（f=2），残差从
   **+1.946 → +2.102**、**+2.722 → +2.871**（1×）。该配置残差本来就是正的，
   更多泵浦耗尽会进一步降低信号/NLI、抬高 SNR_NLI，因此**恶化是正确物理
   方向**，不能为改善残差回退修正。1× 绝对值受 128-symbol 统计影响，
   不要按 0.15 dB 排序模型。
5. **cl152 `bwd_sixpumps`**：+0.746 dB（f=1）/+0.324 dB（f=2），残差从
   **−1.787 → −1.041**、**−0.615 → −0.291**（1×），2× 从 **−2.921 → −2.175**、
   **−1.748 → −1.425**。这是本次最大的变化，且是物理量：六泵的直接 ODE
   泵浦输出下降 18–33%，剖面最大相对变化 33.8%。但 physical 参考目前
   **不含泵浦-泵浦转移**，解析侧的泵浦-泵浦负耦合却被本次修正同步放大
   （六泵相邻对旧违反 2.5–10.1%）；因此这一行的改善同时包含“光子因子修正”
   和“物理侧缺项”的相互作用，**不能当作最终校准值**。待物理侧补上泵浦-泵浦
   后重跑，才能判断该配置的真实残差。
6. 由于 `hybrid_raman_paper` 与 `pcfm_h3_raman` 共用 `RamanPaperSpanSolver`，
   两者的泵浦剖面会按同一方向变化；H3 的 NLI 核不同，本文的残差表只列
   hybrid，未把 H3 的位移当作同一数值。`numerical_dra` 与 Semrau
   ISRS-GN 核仍用印刷式负因子，本次未改。若后续要统一，应单独回归，
   不应把本次的 hybrid 数字直接套到 numerical 行。

### 8.5 测试与复现

- 新增 `tests/test_hybrid_raman.py::test_coupling_matrix_conserves_photon_number`
  （随机频率、统一 `A_eff`，相对违反 <1e-12）和
  `::test_two_wave_power_transfer_matches_the_frequency_ratio`
  （两波 `ΔP_p/ΔP_s ≈ -f_p/f_s`，rel <1e-4）。
- 更新 `tests/test_dra_open_issues.py::test_doubled_pump_drive_moves_hybrid_away`：
  旧断言 0.47 dB 编码了非守恒口径，修正后为 **0.528 dB**，断言改为
  `0.53 ± 0.05`。
- `PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python -m pytest -q`
  全量 670 passed；改动文件 ruff 通过。
- 复现审计：解析侧用 `scripts/dra_wideband_compare.py::evaluate_analytic`
  对 c57/cl152 的 `nopump`/`fwd_0p20w`/`bwd_*` 跑
  `signal_polarizations=1,2`；波形侧只读
  `data/dra_waveform_wide/*.json` 与 physical 仓
  `examples/dra_reference_wide/diagnostics/*_2x_spectral.json`。未重生成
  physical npz。

## 9. 强增益倾斜下的 NLI 复核（2026-11-07）

### 9.1 口径与产物

physical 侧只读调用 `physical-nlo-eval` 的 `DistributedRamanFiber`，参数与
`examples/dra_reference_wide/diagnostics/profile_snr_spectral_pump_pump.json`
一致：cl152 `bwd_sixpumps`、100 km、`pump_model="depleted"`、
`depletion_model="spectral"`、`spectral_bins=256`、`pump_pump_raman=True`、
`depleted_max_iterations=32`、`coarse_step_size_m=0.48828125`。输入波形取
`examples/dra_reference_wide/cl152.npz::input_cl152`（2×23040）。非线性只跑
一次 SSFM；线性参考复用
`examples/diagnose_dra_profile.py::_linear_forward` 的频域精确解，泵浦剖面
与 SSFM 同源；逐信道 NLI 复用
`examples/generate_dra_reference.py::band_nli_metrics`（64 GHz 符号率带内，
单个复标量拟合公共增益/相位后取残差）。`nli_power_raw_w` 含公共增益/相位，
不作为 NLI 口径。

解析侧用同一 `SpectrumGrid(152, 75e9, 190.15e12)`、64 GBd、
`channel_bandwidth_source="symbol_rate"`、同一泵浦、`pump_depletion=True`、
`raman_ase=False`；逐信道取 `spm_noise_w + xpm_noise_w`。`numerical_dra`
不支持 `signal_polarizations=2`，只有 f=1。`hybrid_raman_paper` 与
`pcfm_h3_raman` 的 ODE 在 `pump_depletion=True` 时耦合矩阵包含泵浦-泵浦项
（`include_isrs=False` 只关信号-信号，不影响泵浦行），因此解析剖面
**已经包含泵浦-泵浦转移**；这与 physical 的 `pump_pump_raman=True` 是本次
可比的前提。

驱动/总功率口径已对齐：输入总功率 0.304191 W；SSFM 总输出 0.445825 W，
线性参考 0.445871 W，相对差 **1.03e-4**。泵浦-泵浦净转移 −0.07217 W，
信号拉曼增益 0.54843 W；中心/低端/高端开关增益分别为
**9.83 / 9.70 / 25.24 dB**。因此本节讨论的是 NLI 口径差，不是泵浦耗尽
驱动功率错配。

产物：

- `results/raman/strong_tilt_nli.json`：152ch 逐信道物理/解析结果；
- `results/raman/strong_tilt_nli_step.json`：步长收敛探针汇总；
- `results/raman/strong_tilt_nli_8ch.json`：8ch 强功率倾斜 + 带边消融。

### 9.2 逐信道残差（128 symbols，0.488 m）

残差 = 解析 − 波形；正值表示解析更乐观。代表信道（完整 152 行在 JSON）：

| ch | 频偏 GHz | 波形 SNR | hybrid f1 | H3 f1 | numerical f1 | hybrid 残差 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | −5662.5 | 22.217 | 32.767 | 32.575 | 31.568 | **+10.550** |
| 17 | −4387.5 | 28.953 | 31.091 | 30.850 | 29.617 | +2.138 |
| 34 | −3112.5 | 28.798 | 31.121 | 30.862 | 29.771 | +2.322 |
| 50 | −1912.5 | 28.971 | 31.509 | 31.236 | 30.400 | +2.538 |
| 67 | −637.5 | 28.231 | 31.557 | 31.242 | 30.431 | +3.325 |
| 75 | −37.5 | 28.583 | 31.469 | 31.127 | 30.279 | +2.886 |
| 76 | +37.5 | 28.191 | 31.454 | 31.107 | 30.254 | +3.263 |
| 77 | +112.5 | 28.425 | 31.437 | 31.087 | 30.228 | +3.012 |
| 84 | +637.5 | 27.708 | 31.296 | 30.915 | 30.005 | +3.588 |
| 101 | +1912.5 | 27.836 | 30.640 | 30.138 | 29.052 | +2.803 |
| 117 | +3112.5 | 27.559 | 28.912 | 28.092 | 26.748 | +1.353 |
| 134 | +4387.5 | 23.381 | 25.585 | 24.042 | 22.073 | +2.204 |
| 151 | +5662.5 | 21.895 | 26.870 | 25.273 | 22.616 | +4.975 |

全带统计：

| 模型 | 中心残差 | 均值 | rms | min | max (ch) | 低端 ch0 | 高端 ch151 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid f1 | +3.263 | +2.991 | 3.196 | +1.207 | +10.550 (0) | +10.550 | +4.975 |
| hybrid f2 | +4.013 | +4.413 | 4.665 | +2.664 | +10.705 (0) | +10.705 | +8.888 |
| H3 f1 | +2.917 | +2.399 | 2.696 | +0.232 | +10.359 (0) | +10.359 | +3.379 |
| H3 f2 | +3.867 | +4.068 | 4.264 | +2.368 | +10.679 (0) | +10.679 | +7.826 |
| numerical f1 | +2.064 | +1.160 | 1.846 | −1.550 | +9.351 (0) | +9.351 | +0.722 |

三点观察：

1. **残差不是单纯随增益倾斜单调增加。** 中心附近约 +2.9~+3.3 dB，高端
   （净增益 +9 dB）反而在 +1.4~+3.6 dB，最大残差集中在低端 ch0。
2. **低端 ch0 是记录边缘伪影，不是增益效应。** ch0 的开关增益 9.70 dB，
   与中心 9.83 dB 几乎相同，但波形 NLI 是中心的 3.8 倍；残差从 ch0 的
   +10.55 dB 单调衰减到 ch11 的 +2.30 dB、ch17 的 +2.14 dB。记录采样率
   11.52 THz 对 11.325 THz 占用带宽只有 **97.5 GHz** 保护带；pp 把高端
   抬到 25.24 dB 后，高端产生的带外 FWM/展宽最可能折叠/泄漏到低端
   （需 2× 采样率对照确认）。这个残差在同一 seed、同一线性参考下对所有
   模型都在 +9~+10.5 dB，不能用 GN 核差异解释。
3. **模型排序稳定：numerical < H3 < hybrid。** 高端 ch151 的残差分别为
   +0.72 / +3.38 / +4.98 dB；中心为 +2.06 / +2.92 / +3.26 dB。数值 GN
   用 `solve_ivp` 的精确剖面，H3 用 PCFM 多项式剖面，hybrid 用 5 参数
   Buglia 拟合；在 15 dB 增益倾斜下，**剖面表示**比 GN XPM 公式本身更
   决定残差。f=2 在所有模型上都让残差增大 0.75~1.5 dB，双偏振驱动不是
   修正方向。

### 9.3 步长收敛：0.488 m 的 pp 波形不是收敛的 NLI 参考

同一输入波形、只改步长（96 symbols，17280 samples）：

| 步长 m | 波形中心 SNR | hybrid f1 中心残差 | hybrid f1 均值 | H3 f1 中心残差 | H3 f1 均值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.48828125 | 28.487 | +2.967 | +3.058 | +2.621 | +2.466 |
| 0.244140625 | 29.581 | **+1.873** | **+2.153** | **+1.527** | **+1.561** |

48 symbols 的 pp/no-pp 对照（同一输入，只改步长）：

| 配置 | 步长 m | 波形中心 SNR | hybrid f1 中心残差 | hybrid f1 均值 |
| --- | ---: | ---: | ---: | ---: |
| pp on | 0.48828125 | 24.764 | +6.689 | +5.663 |
| pp on | 0.244140625 | 25.996 | +5.458 | +5.191 |
| pp off | 0.48828125 | 29.708 | +1.746 | +0.905 |
| pp off | 0.244140625 | 30.443 | +1.011 | +0.235 |

结论：

- 步长减半，pp 的中心残差下降约 **1.0~1.1 dB**（96sym）/ **1.2 dB**
  （48sym）；no-pp 只下降 0.74 dB。pp 把高端功率抬高 ~15 dB，非线性相位
  更大，因此对步长更敏感。`cl152.json` 自身的 `step_converged=false` 和
  0.976→0.488 m 仍变化 3.1 dB 的 nopump 记录也指向同一问题。
- 0.244 m 仍未收敛；按每减半步长 ~0.5~1 dB 的单调趋势外推，0.488→收敛
  的中心残差大约还要再收 1~2 dB。**这不是实测到 0.061 m 的定量修正**，
  但足以说明当前 3.26 dB 中约一半是 SSFM 步长误差，而不是 GN 核误差。

### 9.4 8ch 受控强倾斜与带边消融

为隔离“强倾斜下 GN 的 XPM 项是否失效”，构造 8ch、75 GHz 间隔、64 GBd、
128 symbols 的受控算例：光纤/六泵与 cl152 相同，步长 0.122 m（比全带更细），
发射功率为 `[0,0,0,0,+6,+9,+12,+15] dBm/pol`。`edge_off` 把后四个强信道
降到 −60 dBm。ch3 为弱侧探针（−37.5 GHz）。

| case | 物理 ch3 SNR | hybrid f1 | H3 f1 | numerical f1 | hybrid 均值 | numerical 均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edge_on | 20.330 | 17.390 (−2.938) | 17.070 (−3.265) | 16.850 (−3.478) | −1.934 | −2.419 |
| edge_off | 32.527 | 35.070 (+2.547) | 35.080 (+2.557) | 33.490 (+0.959) | +2.119 | +0.922 |

括号内为解析 − 波形。**残差符号随负载翻转**：强干扰存在时 GN 反而更悲观
（−2.9~−3.5 dB），只有 4 个弱信道时 GN 更乐观（+2.5 dB hybrid/H3，
+1.0 dB numerical）。因此不能用“GN 在强倾斜下低估 XPM”解释 152ch 的
正残差；152ch 的正残差更像**密集多信道 FWM + 记录边缘 + 步长**的组合。

限制：`edge_off` 的输入功率从 0.1268 W 降到 0.0080 W，泵浦耗尽变弱
（净增益 −2.4 → +0.6 dB），所以这是“信号内容消融”，不是冻结泵浦剖面的
严格隔离；用它判断符号方向是可靠的，不能用它量化 FWM 的精确占比。

### 9.5 主因判断

按证据强度排序：

1. **physical 波形侧：0.488 m SSFM 未收敛（主因之一）。** pp 使高端增益
   达 25.24 dB，非线性相位远超 nopump；0.488→0.244 已实测收掉约 1.0 dB
   中心残差，继续减半预计还有 1~2 dB。当前 3.26 dB 不能作为 GN 模型误差
   的校准值。
2. **physical 波形侧：低端记录边缘伪影。** ch0 残差 +10.55 dB，ch0 的
   开关增益与中心相同；残差在 11 个信道内衰减到 +2.3 dB。最可能的原因是
   97.5 GHz 保护带不足，pp 的高端带外产物折叠/泄漏到低端。这是
   `band_nli_metrics` 在边缘信道上的口径问题，不是 NLI 物理；要定量确认
   需做 2× 采样率（23.04 THz）的同 seed 对照，本次未做，因此这是
   高置信度推断而不是已隔离的定论。
3. **解析侧：强倾斜下 hybrid 的 5 参数剖面拟合最弱。** 同一 GN 核下，
   numerical（精确剖面）在高端比 hybrid 低 4.3 dB 残差、在中心低 1.2 dB；
   H3 居中。若必须在强倾斜/pp 口径下出解析数，优先 H3 或 numerical，
   或改进 hybrid 的剖面拟合，而不是改 XPM 系数。
4. **剩余 ~1~2 dB 中心残差：密集 FWM 候选。** 三种解析模型都只含 SPM+XPM；
   152ch、75 GHz 密集梳状谱在 15 dB 倾斜下会产生大量非简并 FWM。8ch 对照
   的符号翻转说明这不是通用 XPM 失效，但还没有做“只保留 FWM 项”的专门
   消融，因此列为候选而非定论。
5. **统计：** 128 symbols、单 seed；cl152 已知 4-seed QPSK 散布 5.08 dB。
   0.1 dB 级的模型排序没有统计意义。

### 9.6 对 `pump_pump_raman` 的结论

- pp 的物理效应是真实且能量守恒的：信号拉曼增益 0.54843 W，
  泵浦-泵浦净转移 −0.07217 W，总输出与线性参考对齐到 1e-4；pp 使中心
  SNR 从 31.45 dB 降到 28.19 dB（−3.26 dB），不能忽略。
- 但**当前 0.488 m / 128 symbols / 单 seed 的 pp 波形不能作为 NLI 验证
  参考**：它同时有步长未收敛、低端边缘伪影和统计散布三个问题。用它去
  调解析模型会把数值伪影拟合进 GN。
- 建议：**生产/参考生成暂不启用 `pump_pump_raman=True`**，直到：
  1. 步长 ≤0.122 m，并在 0.061 m 做一次收敛确认；
  2. 采样率 ≥2×（23.04 THz）或至少 ≥500 GHz 保护带，消除低端折叠；
  3. ≥4 seeds 或更长记录，把 5 dB 级散布压到 <0.5 dB；
  4. 用收敛后的波形重跑本节逐信道对照；按当前每减半步长的收敛量粗估，
     中心残差可能落在 ~1.2~1.5 dB，但这不是实测值；届时再决定是否需要
     给解析侧补 FWM。
- 如果业务必须先看 pp 的影响，解析侧用 `pcfm_h3_raman` 或
  `numerical_dra`，并在报告中标注 +1~+3 dB 的强倾斜不确定度；不要用
  hybrid 的 31.45 dB 去对 pp 波形的 28.19 dB 下“模型高估 3.2 dB”的结论。

### 9.7 复现与测试

```bash
# 152ch 主对照（首次约 6 min SSFM；缓存后重跑只跑解析）
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python \
  scripts/dra_strong_tilt_nli.py --output results/raman/strong_tilt_nli.json

# 8ch 强倾斜 + 带边消融
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python \
  scripts/dra_strong_tilt_nli.py --ablation-only --ablation-step-m 0.1220703125 \
  --ablation-output results/raman/strong_tilt_nli_8ch.json

# 步长探针（同一符号数、只改步长）
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python \
  scripts/dra_strong_tilt_nli.py --probe-symbols 96 --step-m 0.48828125 \
  --models hybrid_raman_paper,pcfm_h3_raman --output /tmp/step96_0p488.json
PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python \
  scripts/dra_strong_tilt_nli.py --probe-symbols 96 --step-m 0.244140625 \
  --models hybrid_raman_paper,pcfm_h3_raman --output /tmp/step96_0p244.json
```

- 新增 `tests/test_dra_strong_tilt_nli.py`（残差统计、代表信道、4ch 解析
  自检、CLI 默认值），7 passed。
- 改动文件 `scripts/dra_strong_tilt_nli.py`、`tests/test_dra_strong_tilt_nli.py`
  通过 `ruff check` 与 `ruff format --check`。
- 未修改 physical 仓、未重生成 physical npz、未改 `data/` 参考数据；
  physical 只读路径通过 `sys.path` 注入。
