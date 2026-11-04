# DRA benchmark：独立数值参考

日期：2026-11-04
分支：`feat/dra-bench`

## 1. 参考模型

`numerical_dra`：

- 信号/泵浦/ASE coupled ODE 用 `scipy.solve_ivp` DOP853，`rtol=1e-10`、`atol=1e-15`；
- 不复用 `RamanPaperSpanSolver` 的固定步长 RK4；
- NLI 用 Buglia 2024 Eq. (6)-(8) 的双重数值积分，剖面取 ODE 原始 `rho(z,f)`；
- 不调用 Eq. (9)/(11)/(12) 的任何闭式；
- 跨间按 `epsilon=0` 功率相加，因此与 Buglia/hybrid 默认口径一致，不包含 Zhang 的相干累加。

频率方向对 XPM 相干峰做 `tan` 变换；纵向用复合 Simpson。限制：

- 信道内剖面按中心频率插值，实测对 NLI <0.1 dB；
- Raman 谱是 Gaussian/CZ 近似；
- 泵浦 ASE 未单独求解，信号 ASE 完整；
- 零色散点附近 `tan` 尺度截断到信道半带宽。

## 2. 结果

`results/raman/dra_benchmark.json`。误差 = 模型 SNR - numerical_dra，逐信道。

| 场景 | 模型 | 平均误差 (dB) | 最大绝对误差 (dB) | 耗时 |
|---|---|---:|---:|---:|
| c_57ch 10 跨，无泵 | `hybrid_raman_paper` | +0.001 | 0.011 | 7.8s |
| | `buglia2024_ra_gn` | +0.001 | 0.011 | 7.8s |
| | `pcfm`（H3） | -0.454 | 0.562 | 78ms |
| c_57ch 10 跨，后向 0.2W | `hybrid_raman_paper` | +0.054 | 0.070 | 4.7s |
| | `buglia2024_ra_gn` | +0.054 | 0.070 | 4.7s |
| | `zhang2024_dra_egn` | +0.469 | 0.801 | 2.4s |
| | `pcfm` + lumped | -0.295 | 0.615 | 55ms |
| c_57ch 1 跨，后向 | `hybrid_raman_paper` | +0.054 | 0.070 | 2.6s |
| | `buglia2024_ra_gn` | +0.054 | 0.070 | 2.6s |
| | `zhang2024_dra_egn` | -0.052 | 0.070 | 0.5s |
| | `pcfm` + lumped | +0.173 | 0.293 | 3ms |
| c_57ch 1 跨，前向 | `hybrid_raman_paper` | +0.035 | 0.041 | 15.1s |
| | `buglia2024_ra_gn` | +0.035 | 0.041 | 15.2s |
| | `pcfm` + lumped | **+6.012** | **6.831** | 3ms |
| scaling_40ch 10 跨，后向 | `hybrid_raman_paper` | +0.043 | 0.051 | 1.6s |
| | `buglia2024_ra_gn` | +0.043 | 0.051 | 1.6s |
| | `zhang2024_dra_egn` | -0.037 | 0.044 | 0.3s |
| | `pcfm` + lumped | +0.343 | 0.432 | 3ms |

## 3. 结论

1. 独立参考与 Buglia/hybrid 完整模型在 C 波段 **0.07 dB 内**，说明两篇论文实现和数值参考一致。
2. `pcfm` + lumped 在前向泵下错 **6 dB**：它完全不改跨内 NLI，只乘跨末增益；不能用于前向 DRA。
3. 后向泵下 lumped 误差约 0.3–0.6 dB，也大于完整模型。
4. Zhang 在 1 跨后向误差 0.05 dB，但 10 跨误差 0.47 dB，主要来自标定/相干口径；不能当独立真值。
5. `pcfm_h3_raman` 剖面感知模型不在本分支；应在合并后用同一 reference 复测。

## 4. 复现

```bash
PYTHONPATH=src pytest tests/test_numerical_dra.py -q
PYTHONPATH=src python scripts/dra_benchmark.py
```
