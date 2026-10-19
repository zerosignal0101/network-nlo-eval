# WP-O2 报告：ode-dev 按 P1/P2 实现，逐项对照 `wp/fg-uwb-fid`

> 本文件由 `tools/ode_report.py` 从 `results/*.json` **渲染**生成，请勿手改数字。
> 判据 J1–J4 派发前写死，未改。

**唯一结论先行**：`ode-dev` 在 6 个 `legacy_*` 场景上与 `ecz-dev` **逐位相同**（6/6 行一致；5 个有 SNR 的场景 Δ = 0.000000000 dB，第 6 个两端都拒绝）。
这不是巧合：**逐条读论文后，`ecz-dev` 的物理本就符合 P1**，因此没有任何一条 `fix_to_paper` 需要落地（裁定数 `fix_to_paper = 0`）。移植进来的只有**工程能力**。

## 0. 本轮实际执行的判据

| 判据 | 内容 | 本轮怎么执行的 |
|---|---|---|
| **J1** | 每条物理公式必须有 P1/P2 公式号 | `tools/ode_formula_map.py --check` 逐条在 `papers/*.md` 里 grep 锚点，grep 不到即失败 |
| **J2** | 论文提及且不一致 ⇒ 改成论文；论文未提及 ⇒ 保留 ecz | 见 §2 裁定表；「论文未提及」一律记 `keep_ecz_no_paper`，**不作为移植许可证** |
| **J3** | fu4 不是权威 | 只有与论文一致才可作移植来源 |
| **J4** | 差异①与 P1 (34) 冲突，不得移植 | 记入 `rejected` 并引公式号；守卫 2 断言 |

## 1. 逐项裁定总表（brief §9 首段要求）

| 项 | fu4 怎么做 | P1/P2 怎么说 | 裁定 | 理由（含公式号） |
|---|---|---|---|---|
| `ase_gain_source`<br>差异①：ASE 增益因子 | 把 numba 内核里 G_i = p_in/p_out_fiber（跨段入/出功率比）的 ASE 接上，并删掉 Python 侧的 EDFA 增益块（isrs_gn.py 收下 s2_ase）。 | **P1 Eq. (34)** / P2 Eq. (5) | **`reject`** | J4 明令禁止移植。本 WP 独立复核后**同意**该裁决：ecz 的 gain 取自 EDFA 配置增益，正是 (34) 的 G；fu4 换成跨段损耗比后，即使 EDFA 配的是 17 dB 它也照用 16 dB，等于用「损耗补偿量」冒充「放大器增益」。另外它让 ASE 对 target_gain_db / gain_ripple_db 完全不敏感——而 (34) 明确要求 G 是**每一级**放大器+DGE 的增益。 |
| `alpha_polynomial`<br>差异②：损耗多项式的开关 | 删掉 use_attenuation_polynomial 开关，**永远**走二次多项式 α(λ)。 | **P1 §II（(1) 之后的无编号式）** | **`keep_ecz_no_paper`** | 两种口径 P1 都写了（§II-A 的 ᾱ 与 (20) 的 α_i=α(f_i)），所以**论文没有要求删掉开关**，把「必须无条件用多项式」当作 fix_to_paper 是误判。ecz 的多项式系数与 λ_0 与论文逐字相同（elements.py:26-29），开关默认 False 是场景取值。J2 的方向：论文未裁定 ⇒ 保留 ecz 实现。 |
| `nli_model`<br>差异③：NLI 模型换成 Eq.21 张量版 + 峰度修正 | 把 SPM/XPM 换成 2×2 双重求和的 Upsilon/T_i=1+t_tilde 展开（docstring 自称「基于 Eq. 21」），phi_i 取 −4π²(β2+2πβ3 f)（**负号**且前因子 −4π² 而非 (3/2)π²），并加入 (5/6)·φ_kurtosis 的格式修正项。 | **P1 Eq. (29) 与 (30)** / P2 §III.D (8)/(9) 与末段 LA 说明 | **`reject`** | 两重理由：(a) J3——fu4 不是权威；(b) J1——它自称的 Eq. 21 在 P1/P2 均不存在，而 phi 的符号与归一化与 P1 (29)/(30) 冲突。brief §10 已预先把「fu4 的 NLI 数值拟合」列为 reject 候选；本 WP 读论文后**确认**该候选成立。ecz 侧的 (29)/(30) 实现是逐项对得上的（见 results/ode_dev_formula_map.json 的p1_eq29_spm / p1_eq30_xpm），故保留 ecz。 |
| `flexgrid`<br>灵活栅格（FlexibleGrid + ConnectionSpec） | 新增 core/flexgrid.py + network/slots.py，支持逐连接中心频率与逐槽占用带。 | 论文未提及 | **`accept`** | 纯工程能力，与物理公式无关，**论文未提及**，不受 J1 约束。移植后 legacy 模式（connections=None）数值**逐位不变**（实测 Δ=0，见 diff_vs_ecz.json）。 |
| `ode_accel`<br>ode_accel 加速档位 | 三档 off / cached_g / matvec：缓存拉曼 G 矩阵、NLI 张量化、matvec 去分支。 | 论文未提及 | **`accept`** | **论文未提及**加速档位。按 brief §7 测试 4 的口径「只测可达，不测物理」，交付**档位接口 + 逐位相等的 memoization 档**。刻意不做浮点求和重排：matvec/cached_g 需改写 numba_kernels.py 里被冻结的内核，按 brief §8 属「先停下来报告」的范围。 |
| `raman_spectrum`<br>拉曼增益谱档位 legacy / tabular | 两档拉曼谱：legacy 合成折线（13.2 THz 爬升、15.6 THz 归零）与tabular 熔融石英谱（Lin & Agrawal 闭式）。 | **P1 Eq. (18) + §II.B 三角近似** | **`reject`** | 两档都**不是** P1 的三角近似：(a) legacy 折线是 fg-ode-fast 那条数值线的合成曲线，fu4 自己注明它是「FG-4a 的数值金标准」依赖的曲线；(b) tabular 明确「会改变结果」——那是物理改进，不是数值重排。J3 下 fu4 无权威性，故**两档都不移植**。本分支按 P1 (18) 的三角近似走，已由 p1_eq18_raman_profile 锁定。 |
| `phi_kurtosis`<br>峰度修正 phi_kurtosis | 给 MultiBandISRSGN 与 NLI 加 phi_kurtosis（0.0 高斯 / −0.6190 64QAM）。 | **P1 Eq. (29)/(30)（无峰度项）** / P2 §III.D 引言 | **`reject`** | J1：找不到公式号 ⇒ 不许保留。本分支保持无该参数（ecz 原状），NLI 严格按 P1 (29)/(30) 的高斯模型。与 PLAN §0 的裁决一致：phi_kurtosis 不是 +0.77 dB 的原因，两边都是默认 0。 |
| `raman_pumps`<br>拉曼泵浦 | 同样通过 raman_pumps → k_bar 进入。 | **P1 Eq. (22)** | **`keep_ecz_no_paper`** | **能力已在 ecz-dev 存在**，无需移植。论文给了 (1−k̄) 缩放，但**没有**规定 k̄ 如何由泵浦功率导出；ecz 取 min(ΣP_pump, 1) 是论文未提及的工程取值，保留。 |
| `arbitrary_power_input`<br>任意功率输入 / 自动均衡 | 始终用 p_in/p_out_fiber 推出增益。 | **P2 Eq. (5)** | **`keep_ecz_no_paper`** | **能力已在 ecz-dev 存在**，无需移植。ecz 的两条分支都合规：给了 target_gain_db 用定值（(34) 的 G）；给 None 时用自动均衡比，此时该比**就是**放大器增益（P2 (5) 的定义）。fu4 的问题不在于用了这个比值，而在于**无视这个区分、永远用它**，从而在配了定值增益时违反 (34)——即 ase_gain_source。 |
| `qot_checker_refactor`<br>qot_checker 的函数化重构 | 把 ecz 的 inline ROADM/booster 逻辑抽成 _roadm_terms() 等函数。 | **P2 Eq. (1)** | **`keep_ecz_no_paper`** | PLAN §0 已实测：两树 SNR 累加主循环逻辑几乎逐字相同，「_roadm_terms 与 inline 写法的**语义差异是零**」，且该路径解释不了 0.77 dB。纯重构，无物理收益 ⇒ 不移植。 |

裁定统计：accept 2，fix_to_paper 0，keep_ecz_no_paper 4，reject 4。

## 2. 公式 → 论文 → 代码 映射表（brief §5 步骤 1，本 WP 的地基）

共 18 条物理公式，其中 15 条符合、1 条违反、2 条是 ODE 侧基准（本闭式分支不实现）。

| 公式 | 论文 | 原式（摘要） | 本仓代码位置 | 符合 | 裁定 |
|---|---|---|---|---|---|
| `p1_alpha_polynomial` | **P1 §II, (1) 之后的无编号式** | alpha[dB/km] ~ alpha2*(lambda-lambda0)^2 + alpha1*(lambda-lambda0) + alpha0, lambda0=1550 nm; alpha2=3.7685e-6, alpha1=-7.3764e-5, alpha0=0.162 | `elements.py:26`<br>`elements.py:29`<br>`numba_kernels.py:91`<br>`numba_kernels.py:92`<br>`isrs_gn.py:529` | ✅ | `keep_ecz` |
| `p1_eq18_raman_profile` | **P1 Eq. (18)** | r(f) = case1: P_t*f ; case2: 0 ; case3: (P_t/B_t)(f^2/2 - f*f_m + (f_M^2 - df_co^2)/2) ; case4: (P_t/B_t)(f_M*f - f^2/2 - (f_m^2 - df_co^2)/2) | `numba_kernels.py:213`<br>`numba_kernels.py:215`<br>`numba_kernels.py:218`<br>`numba_kernels.py:221` | ✅ | `keep_ecz` |
| `p1_eq20_span_power` | **P1 Eq. (20)** | P_i(z) ~ P e^{−C_r L_eff(z) r(f_i) − alpha_i z} P_t / sum_j P e^{−C_r L_eff(z) r(f_j)}, L_eff(z)=(1−e^{−alpha_i z})/alpha_i, alpha_i=alpha(f_i) | `numba_kernels.py:253`<br>`numba_kernels.py:261`<br>`numba_kernels.py:263`<br>`numba_kernels.py:264` | ✅ | `keep_ecz` |
| `p1_eq22_preemphasis` | **P1 Eq. (22)** | C_r L_eff(L) r(f) -> (1 − k_bar) C_r L_eff(L) r(f) | `isrs_gn.py:390`<br>`numba_kernels.py:259` | ✅ | `keep_ecz` |
| `p1_eq23_nli_total` | **P1 Eq. (23)** | sigma^2_NLI(f_i) = (16/27) B_i int int gamma^2 |eta|^2 G(0,f_i+f_1) G(0,f_i+f_1+f_2) G(0,f_i+f_2) df_1 df_2, with B_i the channel bandwidth | `isrs_gn.py:881`<br>`numba_kernels.py:459`<br>`numba_kernels_p2.py:1126`<br>`numba_kernels.py:504`<br>`numba_kernels_p2.py:1139` | ✅ | `keep_ecz` |
| `p1_eq29_spm` | **P1 Eq. (29)** | (4/9)(P_i^3/B_i^2) N^{1+eps} (gamma_ii^2 pi/(phi_i 3 alpha_i^2)) e^{2 C_r k_bar L_eff(L) r(f_i)} * [(T_i−alpha_i^2)/alpha_i asinh(phi_i B_i^2/(pi alpha_i)) + (4 alpha_i^2−T_i)/(2 alpha_i) asinh(phi_i B_i^2/(2 pi alpha_i))] | `numba_kernels.py:319`<br>`numba_kernels.py:326`<br>`numba_kernels.py:327`<br>`numba_kernels_p2.py:1126`<br>`numba_kernels.py:329`<br>`numba_kernels.py:333`<br>`numba_kernels.py:334`<br>`numba_kernels.py:336`<br>`numba_kernels.py:455`<br>`numba_kernels.py:459`<br>`numba_kernels_p2.py:871` | ✅ | `keep_ecz` |
| `p1_eq30_xpm` | **P1 Eq. (30)** | (32/27) P_i N sum_l (P_l^2/B_l)(gamma_il^2/(phi_il 3 alpha_l^2)) e^{2 C_r k_bar L_eff(L) r(f_l)} * [(T_l−alpha_l^2)/alpha_l atan(phi_il B_i/alpha_l) + (4alpha_l^2−T_l)/(2alpha_l) atan(phi_il B_i/(2 alpha_l))] | `numba_kernels.py:381`<br>`numba_kernels.py:387`<br>`numba_kernels.py:389`<br>`numba_kernels.py:390`<br>`numba_kernels.py:392`<br>`numba_kernels.py:504`<br>`numba_kernels_p2.py:1140`<br>`numba_kernels.py:482`<br>`numba_kernels_p2.py:1038`<br>`numba_kernels.py:486`<br>`numba_kernels.py:491`<br>`numba_kernels_p2.py:890`<br>`numba_kernels.py:492`<br>`numba_kernels_p2.py:891` | ✅ | `keep_ecz` |
| `p1_eq31_t_factor` | **P1 Eq. (31)** | T_i = (2 alpha_i − C_r r(f_i))^2 | `numba_kernels.py:641` | ✅ | `keep_ecz` |
| `p1_eq32_gamma` | **P1 Eq. (32)** | gamma_{i,l} = (2 pi/lambda_i) * (2 n2 / (A_eff(lambda_i) + A_eff(lambda_l))) | `numba_kernels.py:133` | ✅ | `keep_ecz` |
| `p1_eq33_snr` | **P1 Eq. (33)** | SNR(f_i) = P / (sigma^2_ASE(f_i) + sigma^2_NLI(f_i)) | `isrs_gn.py:1446`<br>`isrs_gn.py:1449` | ✅ | `keep_ecz` |
| `p1_eq34_ase` | **P1 Eq. (34)** | sigma^2_ASE(f_i) = h f_i F(f_i) B_i G(f_i) N, G(f_i),F(f_i) = gain & NF of **each line amplifier + DGE** | `isrs_gn.py:898`<br>`isrs_gn.py:903`<br>`isrs_gn.py:907`<br>`isrs_gn.py:908` | ✅ | `keep_ecz` |
| `p1_eq34_ase_dead_kernel` | **P1 Eq. (34)** | sigma^2_ASE(f_i) = h f_i F(f_i) B_i G(f_i) N | `numba_kernels.py:575`<br>`numba_kernels.py:578`<br>`isrs_gn.py:865` | ⚠️ 违反（死代码） | `dead_code_violates_eq34` |
| `p1_nf_per_band` | **P1 §IV 正文（(34) 之后）** | per-band noise figure = 6 dB (E), 7 dB (S), 5.5 dB (C), 6 dB (L) | `numba_kernels.py:528`<br>`numba_kernels.py:529`<br>`numba_kernels.py:531`<br>`numba_kernels.py:533` | ✅ | `keep_ecz` |
| `p1_epsilon_coherent` | **P1 Eq. (29) 的 N^{1+eps}（§III 末段解释）** | N^{1+eps}，eps 刻画 SPM 沿跨段的**相干**累积（XPM 为非相干，故用 N） | `numba_kernels.py:328`<br>`numba_kernels.py:291` | ✅ | `keep_ecz` |
| `p2_eq05_lumped_ase` | **P2 Eq. (5)** | P'_ASE,i = 2(G_i − 1) n_sp,i h f_i B_i, n_sp,i ≈ (G_i NF − 1)/(2(G_i−1)), G_i = P_i(L)/P_i(0) 为**放大器**在信道 i 上的增益 | `isrs_gn.py:896`<br>`isrs_gn.py:908` | ✅ | `keep_ecz` |
| `p2_eq01_snr_inv` | **P2 Eq. (1)** | SNR_i^-1 ≈ SNR_TRX,i^-1 + SNR_ASE,i^-1 + SNR_NLI,i^-1 = (P_i / (kappa_i P_i + P_ASE,i'' + eta_n(f_i) P_i^3))^-1 | `isrs_gn.py:1446` | ✅ | `keep_ecz` |
| `p2_eq10_rho_semi_analytical` | **P2 Eq. (10)** | rho(z,f_i) = e^{−alpha_i z}[1 − (C_f,i P_f L_eff + C_b,i P_b L_tilde_eff)(f_i − f_hat)] | （本分支不实现） | —（基准） | `ode_side_reference_only` |
| `p2_eq02_eq04_ra_ode` | **P2 Eq. (2) 与 Eq. (4)** | ± dP_i/dz = −sum_k (f_k/f_i) g_i(|df|)(P_k+P_ASE,k) P_i − … + … − alpha_i P_i | （本分支不实现） | —（基准） | `out_of_scope_this_branch` |

### 2.1 ⚠️ 一个必须记录的隐患：死代码里藏着违反 P1 (34) 的写法

`**若被启用即违反 P1 (34)**：G_i 取的是跨段入/出功率比（损耗补偿比），不是「线路放大器增益」。当前 isrs_gn.py:180 用 `_` 丢弃该返回值，真正生效的是 p1_eq34_ase。**故本分支数值合规**。wp/fg-uwb-fid 恰恰把它接上了（那边 isrs_gn 收下 s2_ase 并删掉 EDFA 增益块），这就是 WP-O1 差异①。按 J4 不得移植。WP-Q1 只把这里的 B_i 参数化为逐信道数组；G_i 的口径**未改动**（不在 WP-Q1 范围）。`

**当前 `ode-dev` 的数值是合规的**（live ASE 走 `isrs_gn.py:214`，用 EDFA 配置增益）。但 `wp/fg-uwb-fid` 恰恰把这个函数接上了——这正是 WP-O1 的差异①。按 brief §8，本 WP **没有**动它：它属于冻结内核，且删除死代码不改变物理。

## 3. 三树逐场景 Δ（brief §9 第 1、2 条）

### 3.1 `ode-dev` vs `ecz-dev`

| 场景 | ode-dev (dB) | ecz-dev (dB) | Δ (dB) | 逐位相同 |
|---|---:|---:|---:|:---:|
| `legacy_two_hop_equalize_on` | +25.265176304718 | +25.265176304718 | +0.000000000 | ✅ |
| `legacy_three_hop_equalize_on` | +23.391590837855 | +23.391590837855 | +0.000000000 | ✅ |
| `legacy_two_hop_equalize_off` | +22.239480441560 | +22.239480441560 | +0.000000000 | ✅ |
| `legacy_express_attenuation` | +25.227016934880 | +25.227016934880 | +0.000000000 | ✅ |
| `legacy_high_filtering` | +24.255573729782 | +24.255573729782 | +0.000000000 | ✅ |
| `legacy_blocked_channel` | 拒绝 | 拒绝 | — | ✅ |

逐位相同 **6/6 行**（其中 5 行有可比的 SNR 且 Δ = 0；第 6 行 `legacy_blocked_channel` 两端都拒绝，无 SNR 可比）。这正是 brief §6 的**关键判据**：按 P1 实现的 `ode-dev` 就应当贴近 `ecz-dev`，实测差为 0。

### 3.2 `ode-dev` vs `wp/fg-uwb-fid`

| 场景 | ode-dev (dB) | fu4 (dB) | Δ = ode−fu4 (dB) | fu4−ecz（PLAN §0 已测） |
|---|---:|---:|---:|---:|
| `legacy_two_hop_equalize_on` | +25.265176304718 | +26.034730429573 | -0.769554125 | +0.769554 |
| `legacy_three_hop_equalize_on` | +23.391590837855 | +24.152139265826 | -0.760548428 | +0.760548 |
| `legacy_two_hop_equalize_off` | +22.239480441560 | +23.029699790920 | -0.790219349 | +0.790219 |
| `legacy_express_attenuation` | +25.227016934880 | +25.989211917342 | -0.762194982 | +0.762195 |
| `legacy_high_filtering` | +24.255573729782 | +25.025781931578 | -0.770208202 | +0.770208 |
| `legacy_blocked_channel` | 拒绝 | 拒绝 | — | — |

⇒ `ode-dev` 与 fu4 的差距 = PLAN §0 实测的 +0.77 dB，因为 **`ode-dev` 与 `ecz-dev` 逐位相同**。这从数值侧再次确认：那 0.77 dB 是 fu4 偏离论文的结果，不是 `ecz-dev` 的误差。

## 4. 能力清单（brief §9 第 3 条）

| 能力 | 可用 | 落点 | 对 legacy SNR 的影响 |
|---|:---:|---|:---:|
| 灵活栅格 `flexgrid` | ✅ | `core/flexgrid.py`（343 行）+ `network/slots.py`（253 行），`MultiBandISRSGN(connections=…)` | **0**（legacy 走 `connections=None`，逐位相同） |
| `ode_accel` 档位 | ✅ | `ODE_ACCEL_MODES = ("off", "cached")`，默认 `"off"` | **0**（纯 memoization，命中返回同一 ndarray） |
| 拉曼泵浦 | ✅（原有） | `FiberSpanConfig.raman_pumps` → `isrs_gn.py:85` `k_bar` → P1 (22) 的 `(1−k̄)` | **0**（legacy 不配泵浦） |
| 任意功率输入 | ✅（原有） | `EDFAConfig.target_gain_db=None` → `isrs_gn.py:280-281` 自动均衡 | **0**（legacy 用定值增益） |

⚠️ `ode_accel` **刻意只有两档**，且都是逐位安全的。fu4 的 `matvec` / `cached_g` 需要改写 `numba_kernels.py` 里被冻结的内核，按 brief §8 属「先停下来报告」的范围——见 §6。

## 5. 没做什么 / 没跑完什么

| 项 | 状态 | 原因 |
|---|---|---|
| 三处差异的**逐项 dB 归因复算** | **未跑** | 那是 WP-O1 的交付物。本 WP 引用 O1 结论，brief §3 明示「直接引用，不必重做」 |
| `fu4` 的 `matvec` / `cached_g` / `tabular` 加速档 | **未跑，未移植** | 需改冻结内核（brief §8）；且其 NLI 基础是被 reject 的 Eq.21 模型 |
| `flex_*` 场景三树对照 | **未跑** | `ecz-dev` 无 `core/flexgrid`（PLAN §0 已确认），无从对照 |
| 逐连接**异构**符号速率 | **未跑（显式拒绝）** | 冻结 njit 核 `symbol_rate_hz` 是标量参数；改成数组 = 改冻结内核。已 `raise ValueError`，不静默取代表值 |
| P2 (8)/(9) 的完整拉曼放大 NLI | **未跑** | 那是 P2 的 RA 线；本分支是 P1 闭式线 |
| 删除 `numba_kernels.py:529-560` 死代码 | **未做** | 属冻结内核（brief §8）；且不改物理 |

## 6. 需要编排方知道的边界（brief §8「先停下来报告」）

**没有**需要立即叫停的 `fix_to_paper`——因为逐条比对后 `ecz-dev` 已符合 P1，`fix_to_paper` 计数为 0。但有三处**主动不移植**，理由如下，供编排方裁决是否要另开 WP：

1. **`numba_kernels.py:529-560` 的死代码 `_calculate_ase_noise_variance`** 用 `G_i = p_in/p_out`，违反 P1 (34)。它当前**不生效**（`isrs_gn.py:180` 用 `_` 丢弃返回值）。清理它要动冻结内核，且会波及上游 ODE 线的数值金标准——建议留待专门 WP。
2. **逐连接异构 Rs** 需要把冻结核的 `symbol_rate_hz` 改成数组。本 WP 选择显式 `ValueError` 而非静默降级，能力缺口是公开可见的。
3. **fu4 的 `matvec`/`cached_g`** 若要移植，前置条件是先解决 ③（其加速建立在那套 NLI 上）。只抄加速而不抄物理，会得到一个既不对应 P1 也不对应 P2 的四不像。

## 附录 A：公式锚点索引（J1 可 grep 证据）

下表列出每条物理公式与每条裁定所引用的论文锚点，`tests/test_ode_dev.py::test_guard6_*` 会逐条在 `papers/*.md` 里 grep。

| 来源 | 条目 | 论文 | 定位 | 锚点字符串 |
|---|---|---|---|---|
| formula_map | `p1_alpha_polynomial` | P1 | §II, (1) 之后的无编号式 | `3.7685\times10^{-6}` |
| formula_map | `p1_eq18_raman_profile` | P1 | Eq. (18) | `P_{\mathrm{t}} f}{}` |
| formula_map | `p1_eq20_span_power` | P1 | Eq. (20) | `P_{i}(z)\simeq P` |
| formula_map | `p1_eq22_preemphasis` | P1 | Eq. (22) | `\tag{22}` |
| formula_map | `p1_eq23_nli_total` | P1 | Eq. (23) | `with $B_{i}$ the channel bandwidth` |
| formula_map | `p1_eq29_spm` | P1 | Eq. (29) | `]_{(29)}` |
| formula_map | `p1_eq30_xpm` | P1 | Eq. (30) | `]_{(30)}` |
| formula_map | `p1_eq31_t_factor` | P1 | Eq. (31) | `\tag{31}` |
| formula_map | `p1_eq32_gamma` | P1 | Eq. (32) | `\tag{32}` |
| formula_map | `p1_eq33_snr` | P1 | Eq. (33) | `\tag{33}` |
| formula_map | `p1_eq34_ase` | P1 | Eq. (34) | `\tag{34}` |
| formula_map | `p1_eq34_ase_dead_kernel` | P1 | Eq. (34) | `gain and noise figure of each line amplifier + DGE` |
| formula_map | `p1_nf_per_band` | P1 | §IV 正文（(34) 之后） | `per - band noise figure` |
| formula_map | `p1_epsilon_coherent` | P1 | Eq. (29) 的 N^{1+eps}（§III 末段解释） | `coherent accumulation` |
| formula_map | `p2_eq05_lumped_ase` | P2 | Eq. (5) | `P_{\text{ASE},i}^{\prime } = 2(G_{i}-1)` |
| formula_map | `p2_eq01_snr_inv` | P2 | Eq. (1) | `P_{\text{ASE,i}}^{\prime \prime } + \eta _{n}(f_{i})P_{i}^{3}}` |
| formula_map | `p2_eq10_rho_semi_analytical` | P2 | Eq. (10) | `C_{f,i} P_{f}L_\text{eff}` |
| formula_map | `p2_eq02_eq04_ra_ode` | P2 | Eq. (2) 与 Eq. (4) | `\pm \frac{\partial P_{i}}{\partial z}` |
| verdict | `ase_gain_source` | P1 | Eq. (34) | `\tag{34}` |
| verdict | `ase_gain_source` | P2 | Eq. (5) | `P_{\text{ASE},i}^{\prime } = 2(G_{i}-1)` |
| verdict | `alpha_polynomial` | P1 | §II（(1) 之后的无编号式） | `3.7685\times10^{-6}` |
| verdict | `nli_model` | P1 | Eq. (29) 与 (30) | `]_{(29)}` |
| verdict | `nli_model` | P2 | §III.D (8)/(9) 与末段 LA 说明 | `\Upsilon _{i}` |
| verdict | `raman_spectrum` | P1 | Eq. (18) + §II.B 三角近似 | `P_{\mathrm{t}} f}{}` |
| verdict | `phi_kurtosis` | P1 | Eq. (29)/(30)（无峰度项） | `]_{(30)}` |
| verdict | `phi_kurtosis` | P2 | §III.D 引言 | `\Upsilon _{i}` |
| verdict | `raman_pumps` | P1 | Eq. (22) | `\tag{22}` |
| verdict | `arbitrary_power_input` | P2 | Eq. (5) | `P_{\text{ASE},i}^{\prime } = 2(G_{i}-1)` |
| verdict | `qot_checker_refactor` | P2 | Eq. (1) | `P_{\text{ASE,i}}^{\prime \prime } + \eta _{n}(f_{i})P_{i}^{3}}` |

## 附录 B：论文与复算入口

```bash
export PYTHONPATH=$PWD/src
PY=~/miniconda3/envs/rl/bin/python
$PY tools/ode_formula_map.py --write --check   # J1：公式锚点逐条 grep
$PY tools/ode_verdicts.py     --write --check   # J1/J4：裁定与拒绝项
$PY tools/ode_compare.py      --write --check   # 三树 SNR 逐位复算
$PY tools/ode_report.py                          # 重渲染本文档
$PY -m pytest tests/ -q                          # 既有 6 条 + 新增 6 条
```
