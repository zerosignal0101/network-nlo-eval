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


## 5. 对校准结论的影响

1. 前向泵的 +0.8…+1.2 dB 残差应表述为"1 mW/pol、0.2 W 前向"下的值，
   并注明随发射功率变化（0.5 mW 更差、2 mW 更好）。不要把它外推成
   任意功率下的固定偏差；
2. 2× 偏振驱动负结果支持现有每偏振功率口径，不需要改驱动系数；
3. 单次实现 0.3–0.6 dB 的统计预算意味着 ±1.0 dB 校准带里约一半是统计量，
   0.1–0.2 dB 级的模型排序需要多 seed 或更长符号记录；
4. 1ch 前向是 GN 适用域边界，不是新的实现 bug；
5. 57/152ch 的波形残差等 WP7 数据到位后由
   `scripts/dra_wideband_compare.py` 跑出；本轮不伪造全 WDM 波形数字。

## 6. 复现

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
