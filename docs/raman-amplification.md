# 拉曼放大支持：集总等效与论文 hybrid 对照模型

日期：2026-11-03
分支：`feat/hybrid-full`（基于 `70efe1d`）

本文档区分两个层次，避免把近似当成论文模型：

1. **集总等效 DRA**（`models/raman.py`，注册名 `raman_lumped`）：泵浦未耗尽、
   跨末乘增益、跨内 NLI 不变。用于快速链路预算与泵浦初值筛选。
2. **论文 hybrid 模型**（`models/hybrid_raman.py`，注册名 `hybrid_raman_paper`）：
   论文 Eq. (2)/(4) 的 coupled signal/pump/ASE ODE，NLI 用论文 Eq. (6)-(9)
   的 SPM/XPM 闭式（默认 `nli_form="paper"`）。

## 1. 现状

| 路径 | 拉曼泵浦支持 | 说明 |
|---|---|---|
| 波形物理 `physics_nlo_eval.RamanPump` | **有** | Gaussian 增益 + ASE，FW/BW，任意泵；本身也是**未耗尽集总**块，不是论文 coupled ODE |
| `FiberSpanConfig.raman_pumps` | **配置有** | 解析 lumped / paper 模型都可读取 |
| PCFM/H3 解析 | **原无** | 只做信道间 ISRS；`Raman pumping is not modelled` |
| `raman_lumped` | **有** | 跨末乘 on-off 增益，Raman ASE 在 EDFA 前注入 |
| `pcfm_h3_raman` | **有（剖面感知）** | H3 PCFM SCI/XCI 核 + 粗 coupled ODE 剖面/ASE；无泵时逐位回到 H3 PCFM |
| `hybrid_raman_paper` | **有（完整 NLI）** | Eq. (2)/(4) ODE + Eq. (10) 拟合 + Eq. (6)-(9) 闭式；`nli_form="pcfm"` 保留旧近似 |
| cGN | 部分 | `k_bar` 只是泵浦功率占比上限，不是谱分辨拉曼放大 |
| D'Amico2022 | 无 | 明确不做 DRA |

## 2. 集总等效组件

`src/network_nlo_eval/models/raman.py`

- `RamanGainModel`
  - Gaussian（默认，峰 13.2 THz，σ=2.12 THz）或 CZ 线性截断谱；
  - 任意 FW/BW 泵；
  - 未耗尽泵浦有效长度；
  - 逐信道 `A_eff` 与泵浦/信号光子能量因子；
  - Raman ASE 沿跨一维积分。
- `RamanAmplifier`
  - `gain/gain_db/ase_w/apply`；
  - 集总在跨段末端，随后由 EDFA 补偿剩余损耗。

PCFM opt-in：

```python
model = create_model(
    "pcfm", grid, fiber,
    raman_model="lumped",          # 或 raman_pumps=[...] 自动启用
    raman_pumps=[RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",
        attenuation_db_per_km=0.25,
    )],
    raman_gain_model="gaussian",   # 或 "linear"
    raman_ase=True,
)
```

默认关闭；无泵/关闭时逐位保持旧 PCFM 结果。

## 3. 论文 hybrid 模型（Eq. (2)-(10)）

`src/network_nlo_eval/models/hybrid_raman.py`

```python
model = create_model(
    "hybrid_raman_paper", grid, fiber,
    raman_pumps=[RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",
        attenuation_db_per_km=0.25,
    )],
    include_isrs=True,
    fixed_point_iterations=24,
    nli_form="paper",              # "paper"（默认）/ "integral" / "pcfm"
    spm_coherence_exponent=0.0,    # Eq. (6) 的 epsilon
    spp_steps=32,
)
```

### 3.1 公式映射（逐项）

| 论文 | 内容 | 代码 |
|---|---|---|
| Eq. (2) | 信号/泵浦耦合功率方程 | `raman_ode.RamanPaperSpanSolver` |
| Eq. (3) | LA 极限（无泵） | 同一求解器，`pumps=[]` |
| Eq. (4) | 信号 ASE 耦合方程 | 同一求解器 |
| Eq. (6) | `eta_n = sum_j (P_ij/P_i)^2 [SPM_j n^eps + XPM_j]` | `nli_closed_form` |
| Eq. (7) | `eta_XPM = sum_k eta_XPM^(k)` | `nli_closed_form` 内的干扰信道求和 |
| Eq. (8) | 单干扰信道 XPM 闭式（含 `Upsilon`、`l1+l2` 和式、纵向积分） | `buglia2024_closed.eta_xpm_closed` |
| Eq. (9) | SPM 闭式（`asinh` + 两个 `ln` 修正） | `buglia2024_closed.eta_spm_closed` |
| Eq. (10) | 5 参数剖面 `rho(z, f_i)` | `buglia2024_closed.fit_buglia_profile_5param` |
| Eq. (6)-(8) | 数值积分基准 | `buglia2024_closed.nli_integral`，`nli_form="integral"` |

论文 Sec. III-D 明说 Eq. (8)/(9) 的 SPM/XPM 半解析式来自文献 [46]
（Buglia 2024 JLT, RA-GN 的 Eq. (11)/(12)）。因此本模型与
`buglia2024_ra_gn` 共用同一套闭式实现；两者在相同参数下逐信道结果相同
（`tests/test_hybrid_raman.py::test_paper_no_pump_long_link_difference_is_a_kernel_effect`），
差别只在默认选项与 legacy `pcfm` 口径。

`Upsilon` 按 PDF 原文：`Upsilon_i = T_i (-T_f,i/T_i)^{l1} (T_b,i/T_i)^{l2}`，
`T_f,i = -P_f C_f,i (f_i - f_hat)/alpha_f,i`，
`T_i = 1 + T_f,i - T_b,i e^{-alpha_b,i L}`，
`alpha_l,i = alpha_i + l1 alpha_f,i - l2 alpha_b,i`，
`kappa_f,i = e^{-(alpha_i + l1 alpha_f,i)L}`，`kappa_b,i = e^{-l2 alpha_b,i L}`。
代码里用拟合得到的 `c_f = C_f,i P_f`、`c_b = C_b,i P_b`，与原文等价。

### 3.2 已复现（数值可验证）

- **Eq. (2)/(4)**：任意数量 FW/BW 泵；耦合矩阵同时包含泵浦-信号、
  泵浦-泵浦、信道间 ISRS（`include_isrs=False` 可关）与泵浦耗尽。
  BW 泵用“正向信号 + 反向泵浦”的不动点迭代；FW 泵与信号同向积分。
- **Eq. (10)**：逐信道非线性最小二乘拟合 5 个系数
  `(alpha_i, C_f,i, C_b,i, alpha_f,i, alpha_b,i)`；`f_hat` 取泵浦功率加权
  平均频率（无泵时取信道平均）。
- **Eq. (8)/(9)**：逐项实现，含 `l1+l2<=1` 的三项和、`Upsilon`、`atan`/
  `asinh`/`log` 与跨长项；`nli_form="paper"` 是默认路径。
- **Eq. (6)/(7)**：跨间按 `[P_ij/P_i]^2 [SPM_j n^eps + XPM_j]` 的功率口径；
  核内不返回复振幅（`emits_nli_amplitudes=False`），**不使用** PCFM/Zhang 的
  相干契约。`epsilon` 论文未印出（见下），默认 0。
- **每跨初值与跨间 ASE**：跨末 EDFA 后的 ASE 作为下一跨光纤入口初值；交给
  EDFA 的是本跨**新产生**的拉曼 ASE，避免与路由层重复计数。
- **闭式 vs 积分**：`c_57ch` 1 跨、`scaling_40ch` 1/10 跨上，
  `nli_form="paper"` 与 `nli_form="integral"` 的平均 SNR 差 **<0.02 dB**
  （`tests/test_hybrid_raman.py::test_paper_closed_form_matches_integral_baseline`
  要求逐信道 <0.1 dB）。
- **无泵退化**：`scaling_40ch` 单跨无泵时 `paper` 与 PCFM 的平均 SNR 差
  **0.030 dB**（<0.1 dB）。`c_57ch` 10 跨的 0.456 dB 不是 Raman 残项：同一
  链路下 `paper` 与 `buglia2024_ra_gn` 逐位一致，差异来自 Eq. (8)/(9) 闭式
  与 PCFM `K_SCI/K_XCI` 核本身（NLI 比 0.864，10 跨累积）。

### 3.3 仍是近似（明确边界）

1. **`epsilon` 未印出**。论文 Eq. (6) 的相干因子处是缺字符（"is the coherent
   factor [28, Eq. (22)]"），默认 `spm_coherence_exponent=0`（功率相加，
   与仓库其余模型一致）；`=1` 可复现全相干 SPM。
2. **Eq. (8)/(9) 印刷版符号**。Eq. (9) 最后一项在 MinerU 提取里漏了加号，
   按 Appendix D Eq. (56) 补齐；Eq. (8)/(9) 第三项按 Eq. (31)-(33) 推导取
   负号（印刷版为正号），与 Eq. (6)-(8) 数值积分对照验证。详见
   `buglia2024_closed.py` 模块文档。
3. **`T_b,i` 未印出**。Eq. (9) 的系数块只印了 `T_f,i` 与 `T_i`；`T_b,i` 按
   对称性取 `-P_b C_b,i (f_i - f_hat)/alpha_b,i`，与 Eq. (10) 的
   `L_tilde_eff` 一致。拟合时直接解 `c_f = C_f,i P_f`、`c_b = C_b,i P_b`。
4. **拉曼增益谱** 用 `RamanGainModel` 的 Gaussian/CZ 近似，不是论文 Fig. 4 的
   实测二氧化硅谱（数值在 FigShare 仓库）。
5. **泵浦 ASE** 没有单独求解；信号 ASE 的耦合是完整的。泵浦 ASE 对信号 ASE
   的贡献远小于泵浦-信号自发散射。
6. **泵浦-信号 XPM** 按论文 Eq. (7) 后的说明略去（泵浦远离 WDM 谱）。
7. 不包含 droop 的显式控制、动态增益控制、非均匀跨长拟合。
8. `physics_nlo_eval.RamanPump` 只作为增益形状对照；它本身也是未耗尽集总块，
   不能替代论文 coupled ODE。
9. EDFA ASE 用路由层既有 `NF h f B (G-1)` 口径；它与论文 Eq. (5) 的
   `2(G-1)n_sp h f B` 在高增益下近似等价，但不是逐项相同。Raman ASE 与
   EDFA ASE 在结果里是分开累积的。

### 3.4 旧口径（回归对照）

`nli_form="pcfm"` 保留升级前的近似：Eq. (2)/(4) 的 ODE 剖面 + PCFM 的
`K_SCI/K_XCI` 闭式核（`HybridRamanPcfmSpanKernel`）。它不是 Eq. (6)-(9)，
只用于回答“旧近似偏多少”：

| `c_57ch` 10 跨 | 平均 SNR (dB) | 平均 NLI (W) |
|---|---:|---:|
| `nli_form="paper"` | 17.569 | 2.587e-5 |
| `nli_form="pcfm"` | 16.969 | 3.042e-5 |

旧口径的 NLI 高约 18%，SNR 低 0.60 dB。无泵时它是 PCFM 基线的逐位复制
（回归守卫测试），因此不能把这条路径当作论文实现。

## 4. 为什么 lumped 不能替代论文模型

集总等效做了三个会系统性改变结论的假设：

1. **泵浦未耗尽**：增益按 `P_p * L_eff` 线性外推。泵浦功率较高或信道数较多时，
   这会把 on-off 增益估高。
2. **增益集总在跨末**：跨内信号功率剖面不随拉曼增益变化，因此 NLI 完全不变。
   真实 FW 泵会抬高跨内功率、增加 NLI；BW 泵则把增益放在跨末，ASE 更高。
3. **ASE 用末端 `G-1` 近似**：不求解 Eq. (4) 的沿跨耦合，也不含跨间 ASE 初值
   对后续跨的影响。

对照基准（`c_57ch`、10×100 km、1450 nm / 0.2 W 后向泵）：

| 模型 | 平均 SNR (dB) | 平均 ASE (W) | 平均 NLI (W) |
|---|---:|---:|---:|
| `pcfm`（纯 EDFA） | 16.291 | 1.216e-5 | 2.911e-5 |
| `raman_lumped` | 17.221 | 4.194e-6 | 2.911e-5 |
| `hybrid_raman_paper`（paper） | 17.569 | 4.922e-6 | 2.587e-5 |

- lumped 的 NLI 与纯 EDFA **逐位相同**，因为它不改跨内剖面；论文 Eq. (6)-(9)
  的 NLI 比纯 EDFA 低约 11%（BW 泵把增益放在跨末，跨内功率反而略低）。
- 论文模型的平均 SNR 比 lumped 高 **0.35 dB**：ASE 略高（4.92e-6 vs 4.19e-6），
  但 NLI 更低，净收益为正。
- 论文模型的 ASE 比 lumped 高约 17%，因为它求解 Eq. (4) 的沿跨自发散射，
  而不是末端 `G-1` 近似。
- 同一泵浦下，论文 ODE 的 on-off 增益平均 **8.53 dB**，lumped 为 **10.25 dB**，
  物理 `RamanPump`（未耗尽）为 **9.60 dB**。泵浦耗尽把增益压低了约 1.7 dB。
- 小信号极限下，论文 ODE 的未耗尽增益与物理 `RamanPump` 逐信道差 **<0.1 dB**
  （`tests/test_hybrid_raman.py`），说明泵浦-信号耦合的方向与谱形实现正确；
  差异主要来自工作功率下的泵浦耗尽。

其余算例（`results/raman/paper_comparison.json`，平均 SNR dB）：

| 算例 | 纯 EDFA | lumped | paper | 旧 pcfm 口径 |
|---|---:|---:|---:|---:|
| `c_57ch` 1 跨 | 26.665 | 27.688 | 27.569 | 27.436 |
| `c_57ch` 10 跨 | 16.291 | 17.221 | 17.569 | 16.969 |
| `scaling_40ch` 1 跨 | 27.837 | 30.827 | 30.527 | 30.437 |
| `scaling_40ch` 10 跨 | 17.675 | 20.511 | 20.527 | 20.123 |

## 4.5 剖面感知快速模型 `pcfm_h3_raman`

### 定位

`pcfm_h3_raman` 是 `pcfm` + `raman_lumped` 的升级路径，不是第二份论文实现：

* **保留** H3 PCFM 的 SCI/XCI 闭式核、`edge_fix`、`cache_spans`、EGN 与
  `spp_steps="auto"`；
* **替换** 跨内归一化剖面：由泵浦/信号耦合的粗 coupled ODE 给出
  `p_i(z)=P_i(z)/P_i(0)`，送入同一套多项式拟合 Eq. (23) 与岛核 Eq. (24)/(30)，
  不再把 Raman 增益只在跨末乘一次；
* **同时** 用同一份 ODE 给出新产生的 Raman ASE，交给 EDFA 前的路由层；
* **无泵时** 完全走父类 `PCFMSpanKernel`，与 H3 PCFM 逐位一致。

明确边界：剖面/ASE 的 ODE 用 12 步粗步长（完整 paper 用 32 步），输入 ASE
初值取零，Raman 谱仍是 Gaussian/CZ 近似，泵浦 ASE 与泵浦-信号 XPM 未建模。
因此它是**工程近似**，不是 Buglia Eq. (6)-(9) 的完整实现。

```python
model = create_model(
    "pcfm_h3_raman", grid, fiber,
    raman_pumps=[RamanPumpConfig(
        wavelength_nm=1450.0, power_w=0.2,
        direction="backward", attenuation_db_per_km=0.25,
    )],
    raman_steps=12,          # 粗 ODE 步数
    raman_iterations=24,     # BW 泵不动点上限（收敛即提前退出）
    span_coherence="pcfm",   # 默认 H3 相干口径；"incoherent" 为 paper 匹配口径
)
```

### 误差来源（先量化，再实现）

`c_57ch`（57 信道、10×100 km、1450 nm / 0.2 W 后向泵，**H3 配置**
`poly_degree=3, egn=True`）上，lumped 与 paper 的平均 SNR 差为
**−0.191 dB**；1 跨为 **+0.214 dB**。把误差拆开
（`results/raman/h3_raman.json` 的 `error_sources`）：

| 量 | 1 跨 | 10 跨 |
|---|---:|---:|
| lumped − paper 平均 SNR (dB) | +0.214 | −0.191 |
| lumped/paper NLI 比 (dB) | −0.134 | +0.333 |
| lumped/paper ASE 比 (dB) | −0.695 | −0.695 |
| 只把 NLI 换成 paper 后的平均 SNR (dB) | 27.678 | 17.678 |
| 只把 ASE 换成 paper 后的平均 SNR (dB) | 27.675 | 17.280 |

结论：

1. **ASE 是 lumped 的系统性偏差**。lumped 的末端 `G−1` 近似比 paper 的
   Eq. (4) 沿跨 ASE 低 0.70 dB；10 跨累积后，仅 ASE 一项就让 lumped 的 SNR
   偏高约 0.10 dB。
2. **NLI 的方向会随跨数翻转**。1 跨时 paper 的 NLI 比 lumped 低 0.13 dB；
   10 跨时 paper 的 NLI 反而比 lumped 高 0.33 dB。后者不是剖面效应，而是
   paper 默认 `epsilon=0` 的**功率相加**口径：paper 的 10 跨 NLI 恰好是
   1 跨的 10 倍，而 PCFM/数值参考的可见度加权相干和是 11.32 倍
   （无泵 10 跨/1 跨 NLI 比：PCFM 11.32，数值参考 11.32，paper 10.0）。
3. **只看剖面**：把同一个 PCFM 核的剖面换成 paper ODE 剖面
   （`hybrid_raman_paper(nli_form="pcfm")`），10 跨 NLI 比 lumped 高
   0.36 dB；paper 自己的 NLI 相对其无泵基线只高 0.12 dB。两者相差约
   0.7 dB，主要是 PCFM 闭式核与 Buglia Eq. (8)/(9) 的核间差，加上跨间口径。
   新模型复现的是**Raman 引起的剖面/ASE 变化**，不是这 0.7 dB 的核间基线。

因此，评估新模型时同时报告两个量：

* **绝对 SNR 误差**（对 paper）：包含 paper 的 `epsilon=0` 口径；
* **Raman 效应误差**：各自减去自己的无泵基线，隔离剖面/ASE 的贡献。

### 精度分布与耗时

`results/raman/h3_raman.json` 的完整逐信道分布；下表是平均值与 p95/最大
绝对误差（逐信道，dB）：

| 算例 | 模型/口径 | 平均 SNR (dB) | 平均误差 vs paper | p95 abs | max abs | 热调用 (ms) |
|---|---|---:|---:|---:|---:|---:|
| `c_57ch` 1 跨 | `pcfm_h3_raman` | 27.528 | −0.041 | 0.065 | 0.066 | 14.9 |
| `c_57ch` 10 跨 | `pcfm_h3_raman`（默认相干） | 17.123 | −0.446 | 0.462 | 0.463 | 15.5 |
| `c_57ch` 10 跨 | `pcfm_h3_raman`（`incoherent`） | 17.528 | −0.041 | 0.065 | 0.066 | 15.7 |
| `scaling_40ch` 1 跨 | `pcfm_h3_raman` | 30.541 | +0.014 | 0.040 | 0.070 | 12.7 |
| `scaling_40ch` 10 跨 | `pcfm_h3_raman`（默认相干） | 20.269 | −0.258 | 0.274 | 0.274 | 13.9 |
| `scaling_40ch` 10 跨 | `pcfm_h3_raman`（`incoherent`） | 20.541 | +0.014 | 0.040 | 0.070 | 13.4 |
| `cl_40ch` 10 跨（6 泵） | `pcfm_h3_raman`（默认相干） | 14.972 | +0.095 | 1.491 | 1.760 | 19.9 |
| `cl_40ch` 10 跨（6 泵） | `raman_lumped` | 15.763 | +0.886 | 2.209 | 2.506 | 3.5 |

* **C 波段 1 跨**：默认口径已在 0.15 dB 内；逐信道 p95/max 也只有 0.065 dB。
* **C 波段 10 跨**：默认相干口径的绝对误差 0.446 dB 来自 paper 的
  `epsilon=0` 口径，不是 Raman 剖面；把 `span_coherence="incoherent"` 后
  绝对误差 0.041 dB、p95 0.065 dB，满足 <0.15 dB。
* **C+L 6 泵**：默认相干口径平均误差 0.095 dB，逐信道 p95 1.49 dB；
  lumped 是 0.886 dB / p95 2.21 dB。新模型在均值与分布上都优于 lumped，
  但 paper 的 C+L 逐信道形状本身与新模型差异较大，不能用单一均值掩盖。
* **速度**：C 波段 10 跨热调用 15.5 ms，C+L 10 跨 19.9 ms，均 <50 ms 目标；
  首个跨承担粗 ODE 求解，后续透明跨命中缓存。lumped 的 3 ms 仍是更快的
  预算工具，但它没有跨内剖面。

### 前向泵：剖面感知最直接的证据

1 跨、0.2 W 1450 nm、`c_57ch`：

| 方向 | 无泵 NLI (W) | lumped NLI (W) | 新模型 NLI (W) | paper NLI (W) | lumped SNR (dB) | 新模型 SNR (dB) | paper SNR (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 后向 | 2.509e-6 | 2.509e-6 | 2.622e-6 | 2.587e-6 | 27.783 | 27.528 | 27.569 |
| 前向 | 2.509e-6 | 2.509e-6 | 1.010e-5 | 1.040e-5 | 28.241 | 22.276 | 22.159 |

lumped 在两种方向下 NLI 逐位不变；前向泵把增益放在跨首，真实 NLI 增加
约 6 dB，lumped 的 SNR 因此高估约 6 dB。新模型跟随 paper，1 跨误差
+0.117 dB。这是“剖面送入 NLI”而不是“跨末乘增益”的核心区别。

### 限制

* 粗 ODE 的 12 步、零输入 ASE 初值是**工程近似**；精度随 `raman_steps`
  可调，但不要把它当作完整 paper ODE。
* 默认 `span_coherence="pcfm"` 与 H3/数值参考一致；paper 的 `epsilon=0`
  未印出，`"incoherent"` 只是 paper 匹配口径，不是物理上更优的选择。
* 与 lumped 一样，Raman 谱是 Gaussian/CZ 近似，未做具体光纤标定；泵浦 ASE、
  泵浦-信号 XPM、droop 控制未建模。
* 不支持跨内集总损耗（模型层没有该入口）；kernel 层的 `lumped_loss` 与
  Raman 粗 ODE 不应同时使用。

### 复现

```bash
PYTHONPATH=src pytest tests/test_pcfm_h3_raman.py -q
PYTHONPATH=src python scripts/h3_raman_comparison.py
```

结果：`results/raman/h3_raman.json`。

## 5. 三篇 DRA 论文总表

文献库 `~/Downloads/强化学习融合 QoT 信息/QoT 文献包 2026-10-20` 里，单模
分布式拉曼放大的 QoT 模型三篇：

| 文献 | 注册模型 | 实现程度 |
|---|---|---|
| Buglia 2024, JLT, `10.1109/jlt.2023.3315127` | `buglia2024_ra_gn` | Eq. (2) 五参数剖面、Eq. (9)/(10) 链路函数、Eq. (11)/(12) SPM/XPM 闭式；Eq. (6)-(8) 为积分基准 |
| Buglia 2024, JLT, `10.1109/JLT.2024.3439999` | `hybrid_raman_paper` | Eq. (2)/(4) coupled ODE + Eq. (10) 拟合 + Eq. (6)-(9) 闭式（默认）；`pcfm` 口径为旧近似 |
| Zhang 2024, OECC, `10.1109/OECC54135.2024.10975528` | `zhang2024_dra_egn` | Eq. (1)/(2)/(4) 相干 GN；Sec. II-C 标定按本仓数据拟合；论文未印出的 `eta_XPM_corr` 未实现 |

`ecz` / `semrau2019` 是 ISRS 对照，不是 DRA，不能混入上表。
`buglia2024_ra_gn` 与 `hybrid_raman_paper` 共享同一套 Eq. (8)/(9) 闭式
（hybrid 论文引用 [46]），相同参数下数值一致；两者的差别是 hybrid 模型多一条
legacy `pcfm` 口径与不同的默认选项。

总表 `results/raman/full_paper_comparison.json`（单泵 1450 nm / 0.2 W 后向；
C+L 算例用 6 个 E 波段泵）：

| 算例 | 模型 | 平均 SNR (dB) | 平均 ASE (W) | 平均 NLI (W) |
|---|---|---:|---:|---:|
| `c_57ch` 1 跨 | `pcfm` | 26.665 | 1.216e-6 | 2.572e-6 |
| | `raman_lumped` | 27.688 | 4.194e-7 | 2.572e-6 |
| | `hybrid_raman_paper`（paper） | 27.569 | 4.922e-7 | 2.587e-6 |
| | `hybrid_raman_paper`（pcfm） | 27.436 | 4.922e-7 | 2.683e-6 |
| | `buglia2024_ra_gn` | 27.569 | 4.922e-7 | 2.587e-6 |
| | `zhang2024_dra_egn` | 27.464 | 4.922e-7 | 2.663e-6 |
| | `ecz` | 26.759 | 1.216e-6 | 2.492e-6 |
| `c_57ch` 10 跨 | `pcfm` | 16.291 | 1.216e-5 | 2.911e-5 |
| | `hybrid_raman_paper`（paper） | 17.569 | 4.922e-6 | 2.587e-5 |
| | `buglia2024_ra_gn` | 17.569 | 4.922e-6 | 2.587e-5 |
| | `zhang2024_dra_egn` | 17.984 | 4.922e-6 | 2.310e-5 |
| `scaling_40ch` 10 跨 | `pcfm` | 17.675 | 1.199e-5 | 5.026e-6 |
| | `hybrid_raman_paper`（paper） | 20.527 | 4.319e-6 | 4.507e-6 |
| | `zhang2024_dra_egn` | 20.786 | 4.319e-6 | 3.997e-6 |
| `cl_40ch` 10 跨 | `pcfm` | 15.115 | 1.196e-5 | 6.516e-5 |
| | `hybrid_raman_paper`（paper） | 14.877 | 2.823e-6 | 7.888e-5 |
| | `hybrid_raman_paper`（pcfm） | 14.075 | 2.823e-6 | 9.561e-5 |
| | `zhang2024_dra_egn` | 14.790 | 2.823e-6 | 8.113e-5 |

C+L 算例里 6 个泵抬高了跨内功率，NLI 增加足以抵消 ASE 的收益，hybrid 与
Zhang 的 SNR 都低于纯 EDFA；这是该多泵配置在 40 信道 C+L 上的结果，不是模型
符号错误（`pcfm` 口径给出更低的 14.075 dB）。

### 运行

```bash
PYTHONPATH=src pytest tests/test_hybrid_raman.py tests/test_dra_paper_models.py -q
PYTHONPATH=src python scripts/full_paper_comparison.py
```

## 6. 限制与后续

- 拉曼谱未标定到具体光纤；默认系数与物理 `RamanPump` 的 Gaussian 近似对齐，
  不是实测 G.652.D 参数。
- BW 泵不动点迭代在高损耗/强耗尽下可能需要更多次数；模型会返回
  `converged=False`，但当前公开接口只暴露路径结果，不暴露该标志。
- 未实现泵浦 ASE、泵浦-信号 XPM、droop 控制；跨长非均匀时 Eq. (10) 的拟合
  逐跨进行，但 `epsilon` 口径只按总跨数缩放。
- 模型内部的跨间 ASE 状态不经过 ROADM/booster；基准路径无 ROADM。带 ROADM 的
  链路仍可用，但 ODE 的 ASE 初值只跟踪跨末 EDFA 输出。
- 物理波形模型只做了增益形状对照；没有把波形仿真当作在线替代。

## 7. 复现命令

```bash
PYTHONPATH=src pytest tests/test_raman.py tests/test_hybrid_raman.py tests/test_dra_paper_models.py -q
PYTHONPATH=src pytest tests/test_pcfm.py -q
PYTHONPATH=src python scripts/raman_paper_comparison.py
PYTHONPATH=src python scripts/dra_paper_comparison.py
PYTHONPATH=src python scripts/full_paper_comparison.py
```

结果：`results/raman/pump_sweep.json`、`results/raman/paper_comparison.json`、
`results/raman/paper_controls.json`、`results/raman/full_paper_comparison.json`。
