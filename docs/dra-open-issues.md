# DRA 校准开放问题收口

日期：2026-11-05
工作树：`feat/dra-wideband-compare`（从 `multi-model-dev` `c871d11`）
脚本：`scripts/dra_open_issues.py`、`scripts/dra_wideband_compare.py`

本文收口 HANDOFF v2 §6 的三个开放问题：发射功率依赖、1ch 前向、
单次实现的统计预算；同时给出 2× 偏振耗尽驱动负结果与 152ch 解析一致性。
所有数字都由本仓脚本在当前工作树生成，产物在 `results/raman/`。

口径统一为：

- 残差 = 解析 `SNR_NLI` − 波形 `SNR_NLI`（中心信道，dB），正数表示解析乐观；
- 波形参考是 62.5 m 收敛耗尽参考
  （`data/dra_waveform_converged/dra_reference_depleted.json`）；
- hybrid = `hybrid_raman_paper(nli_form="paper")`，H3 =
  `pcfm_h3_raman(egn="auto")`，数值 = `numerical_dra`；
- 有泵配置统一 `pump_depletion=True`；波形与解析都按每偏振功率驱动。

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

### 1.1 2× 偏振耗尽驱动：负结果

假设"解析泵浦耗尽只用了单偏振信号功率，应该乘 2"：只把耦合 ODE 的
信号驱动乘 2，NLI 仍按标称入纤功率计算（`_paper_span_center` 的
`drive_factor=2`），结果如下。

| 数据集 | 驱动 | 波形 | 解析 | 残差 | 相对 1× 变化 |
|---|---|---:|---:|---:|---:|
| 3ch | 1× | 23.012 | 23.803 | +0.791 | — |
| 3ch | 2× | 23.012 | 24.277 | +1.264 | +0.474 |
| 5ch | 1× | 22.531 | 23.491 | +0.960 | — |
| 5ch | 2× | 22.531 | 24.203 | +1.672 | +0.712 |

如果 2× 偏振口径是 bug，加倍驱动应该让残差靠近 0；实际残差反而增大
0.47/0.71 dB。方向相反，**这不是 2× 偏振口径的 bug**，负结果保留。
加倍驱动后 NLI 下降 0.47/0.71 dB，与"泵浦被更强耗尽、增益下降"一致，
也再次说明前向残差主要来自耗尽/增益剖面的功率响应，而不是驱动系数漏乘。

> **2026-11-05 复核**：上表使用的波形含时域增益施加 bug。物理侧修正后
> 用同一 62.5 m 数据重跑，1× 残差变为 3ch **+0.538**、5ch **+0.252** dB；
> 2× 驱动为 3ch **+1.012**、5ch **+0.964** dB。3/5ch 的 1× 残差已明显
> 缩小，但 2× 仍把它们推得更正。与此同时 7ch 修正后为 **−1.01 dB**、
> 57ch 为 **−3.21 dB**：残差随信道数变负，说明高负载下解析模型确实
> 保守。固定的 2× 驱动不是全信道数通用的修正；需要把“双偏振总功率
> 耗尽”与“强前向增益 GN 偏差”分开。旧“负结果”只对 3/5ch、且只对
> 含 bug 的波形成立，现已重开。

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

- **nopump**：两 seed 均值与解析差 **−0.52 dB**，落在 ±1 dB；但两个
  256-symbol seed 相差 1.88 dB，单次实现不能用于 0.1 dB 级比较。
- **backward**：均值 **−0.50 dB**，落在 ±1 dB；修正前为 +1.48 dB。
- **forward**：两 seed 均值 27.880 dB，对解析 24.670 dB 的残差为
  **−3.21 dB**（解析保守），seed 散布 1.70 dB。这与 3ch (+0.54)、
  5ch (+0.25)、7ch (−1.01) 构成随信道数单调下降的序列：总功率越高，
  解析模型越保守。最可能的口径项是泵浦耗尽的偏振驱动（波形用双偏振
  总功率，解析 ODE 用每偏振功率）；但固定的 2× 驱动在 3/5ch 会把残差
  推得更正，说明还叠加了强前向增益下的 GN 适用域偏差。**这是当前最大的
  开放问题**，详见 §1.1 的复核。
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
> 混叠偏差（波形的 alias-free SNR 更高），因此 alias-free 残差大致为
> nopump −1.11、bwd ≈−2.7、fwd ≈+0.1 dB。bwd/fwd 的 2× 数据尚未生成，
> 上述外推只用于方向判断。

- `nopump` 与 `fwd_0p20w` 的 hybrid/H3 残差在 **±1.5 dB** 内（1×）；
  修正后的前向从 +13.75 dB 回到 +1.52 dB，已恢复可用。
- `bwd_sixpumps` 的 hybrid 残差 −1.30 dB、H3 −1.86 dB，H3 已超出
  ±1.5 dB；叠加 ~1.4 dB 混叠修正后更负。这是新的开放问题（六泵后向
  的增益/耗尽剖面），pcfm 的 +0.74 dB 仍是 lumped Raman 口径的离群值。

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
- **128-symbol、2×、0.244 m 的两个 seed** 为 33.502 / 34.247 dB，
  均值 **33.874 ± 0.53 dB**；1× 128-symbol 主数据 32.473 dB 加
  ~1.3 dB 混叠 ≈ 33.8 dB，两条独立路径一致。
- 因此 cl152 的 alias-free 中心是 **~33.8–33.9 dB**。2× 128-symbol
  两 seed 均值 33.874 对解析 32.765 给出残差 **−1.11 dB**；1× 主数据
  +0.29 dB 在混叠校正后也约 −1.0 dB。解析略保守，但在 ±1.5 dB 内，
  **之前“3 dB 系统偏差”的说法不成立**。
- **调制口径对照**（不变）：同 1×、0.244 m、64 symbols，DP-QPSK
  34.226 dB vs 复高斯 27.220 dB（NLI 高 7.0 dB）。GN 的 Gaussian
  假设是一阶口径项；但 152ch 的绝对残差实测仅 ~1 dB，要拆解调制项
  需要收敛的高斯控制 + 多 seed。

### 5.4 结论

1. **57ch**：修正增益施加后，nopump 两 seed 均值 −0.52 dB、
   backward −0.50 dB，都在 ±1 dB；forward 均值 **−3.21 dB**（解析保守），
   seed 散布 1.70 dB，且残差随信道数变负（3ch +0.54 → 57ch −3.21 dB）。
2. **152ch**：修正后 1× 主数据 nopump +0.29/+0.35 dB（hybrid/H3）、
   backward **−1.30/−1.86 dB**、forward **+1.52/+1.35 dB**；
   2× 128-symbol nopump 两 seed 均值 33.874 dB 对解析 32.765 dB 给出
   **−1.11 dB**（hybrid，alias-free）。1× 数据含约 1.4 dB 混叠偏差，
   alias-free 的 backward/forward 残差约为 −2.7/+0.1 dB（外推，待 2×
   数据确认）。绝对残差引 128-symbol 多 seed 值；64-symbol 单次值不可用。
3. **混叠**：1× 采样率系统性高估 NLI（c57 +0.4…0.9 dB，cl152
   +1.1…1.4 dB）；2× 在固定输入下已足够（3× 与 2× 差 0.003 dB）。
   后续全 WDM 参考应默认 2×。
4. **前向泵全 WDM 已恢复可用**：旧的记录长度发散是物理侧时域增益
   施加 bug，不是周期记录伪影。修正后 c57 forward 64→1024 symbols
   的频谱 SNR 为 26.93/26.91/28.73/27.95/27.41 dB，无单调下降
   （EVM 同步）；256-symbol 主参考 28.732/27.028 dB。cl152 forward
   从 16.15 dB 修正到 28.376 dB（1×，128 symbols），hybrid 残差
   +1.52 dB。旧 `c57_forward_record_evm` 与 `c57_forward_long_record_*`
   已标记 superseded。前向的新问题是解析残差随信道数变负
   （57ch −3.21 dB），需要把双偏振总功率耗尽与强前向增益 GN 偏差
   分开，见 §1.1。
5. **统计预算**：c57 nopump 5 次测量标准差 0.90 dB；cl152 64-symbol
   4-seed 标准差 **5.56 dB**、128-symbol 2-seed 标准差 **0.53 dB**。
   绝对 NLI 必须用 128-symbol（或更长）多 seed 集成；64-symbol 只能
   用于同一 seed 的步长趋势。

## 6. 对校准结论的影响

1. 前向泵的 +0.8…+1.2 dB 残差应表述为"1 mW/pol、0.2 W 前向"下的值，
   并注明随发射功率变化（0.5 mW 更差、2 mW 更好）。不要把它外推成
   任意功率下的固定偏差；
2. 2× 偏振驱动负结果支持现有每偏振功率口径，不需要改驱动系数；
3. 单次实现 0.3–0.6 dB 的统计预算意味着 ±1.0 dB 校准带里约一半是统计量，
   0.1–0.2 dB 级的模型排序需要多 seed 或更长符号记录；
4. 1ch 前向是 GN 适用域边界，不是新的实现 bug；
5. 57/152ch 的波形残差已由 `scripts/dra_wideband_compare.py` 跑出：
   nopump/backward 在 ±1.5 dB 内（cl152 hybrid/H3 在 ±1 dB 内），
   forward 因周期记录伪影不可用；全 WDM 单次实现的统计预算是 2–3 dB。
   详见 §5。

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

限制：本文的统计量都基于 256-symbol、单/四个 seed；耗尽模型仍用中心
频率的带内总功率驱动泵浦，未建模泵浦-泵浦拉曼转移；1ch 的 TX 复用了
3/5/7ch 的 50 GBd/100 GHz 口径，不是 WP7 的 64 GBd 全 WDM 口径。
