"""WP-O2 步骤 1：论文 -> 公式清单（`results/ode_dev_formula_map.json`）。

这是本 WP 的**地基**。每条 `ode-dev` 的物理公式都必须挂上一条论文定位（P1 或 P2
的公式号 / 章节号 + 一段能在 `papers/*.md` 里 grep 到的锚点字符串）。

用法::

    PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python tools/ode_formula_map.py --write
    PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python tools/ode_formula_map.py --check

``--check`` 逐条复算两件事（J1 守卫）：

1. 每条 ``class == "physics"`` 的条目都有非空的 ``paper.anchor``，且该字符串在
   ``paper.file`` 指定的论文文件里 **grep 得到**（公式号不许是编的）。
2. 每条 ``code`` 锚点的 ``file:line`` 真实存在，且该行确实含有 ``contains``
   子串（防止行号漂移后静默指向别的东西）。

``class == "engineering"`` 的条目是工程能力（J2：论文未提及 ⇒ 保留 ecz-dev），
不强制公式号，但必须写明 ``paper_note`` 为什么论文没有覆盖它。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPERS = REPO / "papers"
OUT = REPO / "results" / "ode_dev_formula_map.json"

P1 = "papers/A_Generalized_Raman_Scattering_Model_for_Real-Time_SNR_Estimation_of_Multi-Band_Systems_2036278533910728704.md"
P2 = "papers/Ultra-Wideband_Modeling_of_Optical_Fibre_Nonlinearity_in_Hybrid-Amplified_Links.md"

K = "src/network_nlo_eval/models/numba_kernels.py"
G = "src/network_nlo_eval/models/isrs_gn.py"
E = "src/network_nlo_eval/network/elements.py"

# ---------------------------------------------------------------------------
# 物理公式清单
# ---------------------------------------------------------------------------
FORMULAS: list[dict] = [
    # ---- P1 §II：损耗曲线 -------------------------------------------------
    {
        "id": "p1_alpha_polynomial",
        "quantity": "波长相关光纤损耗系数 alpha(f_i) [Np/m]",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "§II, (1) 之后的无编号式",
            "section": "§II POWER EVOLUTION WITH SRS",
            "anchor": "3.7685\\times10^{-6}",
            "file": P1,
        },
        "paper_formula": "alpha[dB/km] ~ alpha2*(lambda-lambda0)^2 + alpha1*(lambda-lambda0) + alpha0, "
                         "lambda0=1550 nm; alpha2=3.7685e-6, alpha1=-7.3764e-5, alpha0=0.162",
        "code": [
            {"file": E, "line": 26, "contains": "attenuation_alpha2: float = Field(3.7685e-6"},
            {"file": E, "line": 29, "contains": "reference_wavelength_nm: float = Field(1550.0"},
            {"file": K, "line": 78, "contains": "alpha_dbkm = alpha_2 * (lambda_nm - lambda_ref_nm) ** 2"},
            {"file": K, "line": 79, "contains": "return alpha_dbkm / (NP_TO_DB * 1000)"},
            {"file": G, "line": 424, "contains": "self.alpha_power_npm_channels = _calc_fiber_attenuation_npm("},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "系数与 lambda0 与论文**逐字相同**。ecz-dev 用 use_attenuation_polynomial 开关暴露它；"
                 "P1 §II-A 同时允许频率无关的 alpha-bar，§II-B (20) 允许 alpha_i=alpha(f_i)，"
                 "论文没有要求删掉开关 -> 见 rejected/alpha_polynomial。",
    },
    # ---- P1 (18)：拉曼整形剖面 r(f) --------------------------------------
    {
        "id": "p1_eq18_raman_profile",
        "quantity": "Raman shaping profile r(f)（四种 case 的分段闭式）",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (18)",
            "section": "§II.B",
            "anchor": "P_{\\mathrm{t}} f}{}",
            "file": P1,
        },
        "paper_formula": "r(f) = case1: P_t*f ; case2: 0 ; "
                         "case3: (P_t/B_t)(f^2/2 - f*f_m + (f_M^2 - df_co^2)/2) ; "
                         "case4: (P_t/B_t)(f_M*f - f^2/2 - (f_m^2 - df_co^2)/2)",
        "code": [
            {"file": K, "line": 213, "contains": "r_f[i] = p_total_w * fi"},
            {"file": K, "line": 215, "contains": "r_f[i] = 0.0"},
            {"file": K, "line": 218, "contains": "0.5 * (f_M_hz**2 - delta_f_co_hz**2)"},
            {"file": K, "line": 221, "contains": "0.5 * (f_m_hz**2 - delta_f_co_hz**2)"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "四个 case 与 (18) 逐项对齐（含 case2 = 0）。case 判定条件见 :206-209。",
    },
    # ---- P1 (20)：闭式信道功率演化 ----------------------------------------
    {
        "id": "p1_eq20_span_power",
        "quantity": "单跨段出纤信道功率 P_i(z)（SRS 倾斜 + 光纤损耗）",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (20)",
            "section": "§II.B",
            "anchor": "P_{i}(z)\\simeq P",
            "file": P1,
        },
        "paper_formula": "P_i(z) ~ P e^{−C_r L_eff(z) r(f_i) − alpha_i z} P_t / sum_j P e^{−C_r L_eff(z) r(f_j)}, "
                         "L_eff(z)=(1−e^{−alpha_i z})/alpha_i, alpha_i=alpha(f_i)",
        "code": [
            {"file": K, "line": 253, "contains": "L_eff_array = (1 - np.exp(-alpha_power_npm * length_m)) / alpha_power_npm"},
            {"file": K, "line": 261, "contains": "p_out_w[i] = ("},
            {"file": K, "line": 263, "contains": "* np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array[i] * r_f[i] - alpha_power_npm[i] * length_m)"},
            {"file": K, "line": 264, "contains": "* p_total_in"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "分子/分母结构与 (20) 一致；L_eff 用**逐信道** alpha_i（:253 数组逐元素），"
                 "正是 (20) 脚注要求的 alpha_i=alpha(f_i)。",
    },
    # ---- P1 (22)：预加重 --------------------------------------------------
    {
        "id": "p1_eq22_preemphasis",
        "quantity": "预加重因子 k_bar 对 SRS 项的 (1−k_bar) 缩放",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (22)",
            "section": "§II.C",
            "anchor": "\\tag{22}",
            "file": P1,
        },
        "paper_formula": "C_r L_eff(L) r(f) -> (1 − k_bar) C_r L_eff(L) r(f)",
        "code": [
            {"file": G, "line": 282, "contains": "self.k_bar = min(sum(pump.power_w for pump in ref_fiber_config.raman_pumps), 1.0)"},
            {"file": K, "line": 259, "contains": "np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array[i] * r_f)"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "(22) 只说缩放系数是 (1−k_bar)，未规定 k_bar 如何由泵浦功率得到；"
                 "ecz 取 min(sum(P_pump), 1) 是**论文未提及**的工程取值 -> engineering 备注见 k_bar 映射。",
    },
    # ---- P1 (23)：NLI 总方差（逐信道 B_i 的总纲）--------------------------
    {
        "id": "p1_eq23_nli_total",
        "quantity": "NLI 总方差 sigma^2_NLI(f_i) 的前因子 B_i（(23) 总纲式）",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (23)",
            "section": "§III CLOSED-FORM GN MODEL",
            "anchor": "with $B_{i}$ the channel bandwidth",
            "file": P1,
        },
        "paper_formula": "sigma^2_NLI(f_i) = (16/27) B_i int int gamma^2 |eta|^2 "
                         "G(0,f_i+f_1) G(0,f_i+f_1+f_2) G(0,f_i+f_2) df_1 df_2, "
                         "with B_i the channel bandwidth",
        "code": [
            {"file": G, "line": 725, "contains": "symbol_rate_hz=self.channel_bandwidth_hz_channels,"},
            {"file": K, "line": 1126, "contains": "spm_pre = ((16.0 / 27.0) * (gamma_ii_sq * p_cube / (symbol_rate_hz**2))) * (big_t * big_t)"},
            {"file": K, "line": 1139, "contains": "pre = ((gamma_sq * p_sq) * p_in_w[:, None]) / symbol_rate_hz"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "(23) 是**积分**总纲；本仓按 P1 §III 用它的闭式解 (29)/(30) 求值，"
                 "故 B_i 没有单独一行代码，而是经 (29) 的 b_i_hz 与 (30) 的 b_l_hz / b_i_hz "
                 "进入。原文明确『with B_i the channel bandwidth』⇒ B_i 是**逐信道**量。"
                 "WP-Q1 据此把此前全局标量的 B 逐信道化；这是 J2『论文提及且与本仓不一致 "
                 "⇒ 改成论文』的直接授权点。",
    },
    # ---- P1 (29)：SPM 闭式 ------------------------------------------------
    {
        "id": "p1_eq29_spm",
        "quantity": "SPM NLI 方差 sigma^2_SPM(f_i)",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (29)",
            "section": "§III CLOSED-FORM GN MODEL",
            "anchor": "]_{(29)}",
            "file": P1,
        },
        "paper_formula": "(4/9)(P_i^3/B_i^2) N^{1+eps} (gamma_ii^2 pi/(phi_i 3 alpha_i^2)) "
                         "e^{2 C_r k_bar L_eff(L) r(f_i)} * [(T_i−alpha_i^2)/alpha_i asinh(phi_i B_i^2/(pi alpha_i)) "
                         "+ (4 alpha_i^2−T_i)/(2 alpha_i) asinh(phi_i B_i^2/(2 pi alpha_i))]",
        "code": [
            {"file": K, "line": 319, "contains": "term_exp = np.exp(2 * cr_w_m_hz * k_bar * l_eff_i_m * r_fi)"},
            {"file": K, "line": 326, "contains": "(4 / 9)"},
            {"file": K, "line": 1126, "contains": "spm_pre = ((16.0 / 27.0) * (gamma_ii_sq * p_cube / (symbol_rate_hz**2))) * (big_t * big_t)"},
            {"file": K, "line": 329, "contains": "(gamma_ii**2 * PI) / (phi_i * 3 * alpha_i_npm**2)"},
            {"file": K, "line": 333, "contains": "asinh_1 = np.arcsinh((phi_i * b_i_hz**2) / (PI * alpha_i_npm))"},
            {"file": K, "line": 334, "contains": "asinh_2 = np.arcsinh((phi_i * b_i_hz**2) / (2 * PI * alpha_i_npm))"},
            {"file": K, "line": 336, "contains": "term_bracket = ((t_i - alpha_i_npm**2) / alpha_i_npm) * asinh_1"},
            {"file": K, "line": 455, "contains": "phi_i = 1.5 * PI**2 * (beta_2_s2_m + 2 * PI * f_rel_hz[i] * beta_3_s3_m)"},
            {"file": K, "line": 871, "contains": "p_in_w[i], symbol_rate_hz[i], n_span, gamma_ii, phi_i, alpha_power_npm[i], t_i_tilde, L_m, epsilon_n_span=0.05"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "4/9 前因子、P^3/B^2、N^{1+eps}、gamma^2 pi/(phi 3 alpha^2)、exp(2 C_r k_bar L_eff r)、"
                 "两个 asinh 的自变量与系数 (T_i−alpha^2)/alpha 与 (4alpha^2−T_i)/(2alpha) **逐项对应**。"
                 "phi_i 取 P1 §III 的 phi_i=(3/2)pi^2(beta2+2pi f_i beta3)。"
                 "WP-Q1：B_i 由全局标量 rs_hz 改为逐信道 b_i_hz（调用点 b_hz_channels[i]），"
                 "均匀取值下与旧的标量是同一个 float64 ⇒ legacy 逐位不变。",
    },
    # ---- P1 (30)：XPM 闭式 ------------------------------------------------
    {
        "id": "p1_eq30_xpm",
        "quantity": "XPM NLI 方差 sigma^2_XPM(f_i)（对干扰信道求和）",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (30)",
            "section": "§III",
            "anchor": "]_{(30)}",
            "file": P1,
        },
        "paper_formula": "(32/27) P_i N sum_l (P_l^2/B_l)(gamma_il^2/(phi_il 3 alpha_l^2)) e^{2 C_r k_bar L_eff(L) r(f_l)} "
                         "* [(T_l−alpha_l^2)/alpha_l atan(phi_il B_i/alpha_l) + (4alpha_l^2−T_l)/(2alpha_l) atan(phi_il B_i/(2 alpha_l))]",
        "code": [
            {"file": K, "line": 381, "contains": "term_exp = np.exp(2 * cr_w_m_hz * k_bar * l_eff_l_m * r_fl)"},
            {"file": K, "line": 387, "contains": "pre_factor = (p_l_w**2 / b_l_hz) * (gamma_il**2 / (phi_il * 3 * alpha_l_npm**2)) * term_exp"},
            {"file": K, "line": 389, "contains": "atan_1 = np.arctan((phi_il * b_i_hz) / alpha_l_npm)"},
            {"file": K, "line": 390, "contains": "atan_2 = np.arctan((phi_il * b_i_hz) / (2 * alpha_l_npm))"},
            {"file": K, "line": 392, "contains": "term_bracket = ((t_l - alpha_l_npm**2) / alpha_l_npm) * atan_1"},
            {"file": K, "line": 1140, "contains": "pre = ((32.0 / 27.0) * pre) * (big_t * big_t)"},
            {"file": K, "line": 1038, "contains": "phi_il = delta_f_hz * phi_base"},
            {"file": K, "line": 486, "contains": "(beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[l_idx] + f_rel_hz[i]))"},
            {"file": K, "line": 890, "contains": "rs_l_hz=symbol_rate_hz[l_idx],"},
            {"file": K, "line": 891, "contains": "rs_i_hz=symbol_rate_hz[i],"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "(32/27) P_i N 的外层因子、逐对项求和与 phi_il 取值均逐项对应 P1 (30)。"
                 "WP-Q1：B_ℓ 与 B_i **分开**取——求和项 (P_ℓ²/B_ℓ) 用干扰信道 ℓ 自己的带宽，"
                 "atan(φ_{i,ℓ} B_i/α_ℓ) 用 COI 的带宽。WP-Q1 之前二者是同一个全局标量"
                 "（rs_hz 同时传给 rs_l_hz 与 rs_i_hz），故多槽宽连接进不了 QoT。",
    },
    # ---- P1 (31)：T_i -----------------------------------------------------
    {
        "id": "p1_eq31_t_factor",
        "quantity": "SRS 修正的有效衰减 T_i",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (31)",
            "section": "§III",
            "anchor": "\\tag{31}",
            "file": P1,
        },
        "paper_formula": "T_i = (2 alpha_i − C_r r(f_i))^2",
        "code": [
            {"file": K, "line": 641, "contains": "t_factors = (2.0 * alpha_power_npm - cr_w_m_hz * r_f_values) ** 2"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "逐字对应。该 T_i 同时喂给 (29) 与 (30) 的方括号项。",
    },
    # ---- P1 (32)：gamma ---------------------------------------------------
    {
        "id": "p1_eq32_gamma",
        "quantity": "非线性系数 gamma_{i,l}（波长依赖 A_eff）",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (32)",
            "section": "§III",
            "anchor": "\\tag{32}",
            "file": P1,
        },
        "paper_formula": "gamma_{i,l} = (2 pi/lambda_i) * (2 n2 / (A_eff(lambda_i) + A_eff(lambda_l)))",
        "code": [
            {"file": K, "line": 120, "contains": "return (2 * PI / lambda_i_m) * (2 * n2 / (a_eff_i_m2 + a_eff_l_m2))"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "SPM(l=i) 与 XPM(l!=i) 共用同一个函数（:442 / :464）。",
    },
    # ---- P1 (33)：SNR -----------------------------------------------------
    {
        "id": "p1_eq33_snr",
        "quantity": "信道 SNR = P/(sigma^2_ASE + sigma^2_NLI)",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (33)",
            "section": "§IV SNR ESTIMATION",
            "anchor": "\\tag{33}",
            "file": P1,
        },
        "paper_formula": "SNR(f_i) = P / (sigma^2_ASE(f_i) + sigma^2_NLI(f_i))",
        "code": [
            {"file": G, "line": 1030, "contains": "total_noise = total_spm + total_xpm + total_ase"},
            {"file": G, "line": 1033, "contains": "snr[active] = 10 * np.log10(power[active] / total_noise[active])"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "ASE 与 NLI 作为统计独立可加噪声进入分母，与 (33) 一致。",
    },
    # ---- P1 (34)：ASE —— 本 WP 的决定性一条 -------------------------------
    {
        "id": "p1_eq34_ase",
        "quantity": "ASE 噪声方差 sigma^2_ASE(f_i) = h f_i F(f_i) B_i G(f_i) N",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (34)",
            "section": "§IV",
            "anchor": "\\tag{34}",
            "file": P1,
        },
        "paper_formula": "sigma^2_ASE(f_i) = h f_i F(f_i) B_i G(f_i) N, "
                         "G(f_i),F(f_i) = gain & NF of **each line amplifier + DGE**",
        "code": [
            {"file": G, "line": 435, "contains": "gain = np.full_like(power_in_w, 10 ** (edfa_config.target_gain_db / 10))"},
            {"file": G, "line": 440, "contains": "gain *= 10 ** (ripple / 10)"},
            {"file": G, "line": 444, "contains": "s2_ase = current_nf_lin * 6.62607015e-34 * self.f_abs_hz * self.channel_bandwidth_hz_channels * np.maximum("},
            {"file": G, "line": 445, "contains": "gain - 1, 0"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "**G = EDFA 配置增益**（target_gain_db × gain_ripple_db），F = NF，B = 逐信道带宽 B_i，"
                 "逐跨段在 evaluate_path 里累加（N）。符合 (34) 原文「line amplifier + DGE 的增益」。"
                 "WP-Q1：B 由标量 symbol_rate_hz 改为逐信道 channel_bandwidth_hz_channels"
                 "（P2 称 B_i 为 \"bandwidth of the COI\"）；均匀取值下与旧标量是同一个 float64 ⇒ legacy 逐位不变。"
                 "这就是 J4 禁止移植的那一项：ecz 侧合规。",
    },
    {
        "id": "p1_eq34_ase_dead_kernel",
        "quantity": "[死代码] numba 内核里的 ASE，G_i = p_in/p_out_fiber",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (34)",
            "section": "§IV",
            "anchor": "gain and noise figure of each line amplifier + DGE",
            "file": P1,
        },
        "paper_formula": "sigma^2_ASE(f_i) = h f_i F(f_i) B_i G(f_i) N",
        "code": [
            {"file": K, "line": 1326, "contains": "G_i = p_in_w[i] / p_out_w[i]"},
            {"file": K, "line": 1329, "contains": "n_span * nf_lin[i] * H_PLANCK * f_abs_hz[i] * max((G_i - 1.0), 0.0) * symbol_rate_hz[i]"},
            {"file": G, "line": 707, "contains": "p_out_fiber, s2_spm, s2_xpm, s2_ase = _calc_span_noise_and_power_jit("},
        ],
        "conforms": False,
        "verdict": "dead_code_violates_eq34",
        "notes": "**若被启用即违反 P1 (34)**：G_i 取的是跨段入/出功率比（损耗补偿比），"
                 "不是「线路放大器增益」。当前 isrs_gn.py:180 用 `_` 丢弃该返回值，"
                 "真正生效的是 p1_eq34_ase。**故本分支数值合规**。"
                 "wp/fg-uwb-fid 恰恰把它接上了（那边 isrs_gn 收下 s2_ase 并删掉 EDFA 增益块），"
                 "这就是 WP-O1 差异①。按 J4 不得移植。"
                 "WP-Q1 只把这里的 B_i 参数化为逐信道数组；G_i 的口径**未改动**（不在 WP-Q1 范围）。",
    },
    # ---- P1 §IV：分频段 NF ------------------------------------------------
    {
        "id": "p1_nf_per_band",
        "quantity": "分频段放大器噪声系数 E/S/C/L = 6/7/5.5/6 dB",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "§IV 正文（(34) 之后）",
            "section": "§IV",
            "anchor": "per - band noise figure",
            "file": P1,
        },
        "paper_formula": "per-band noise figure = 6 dB (E), 7 dB (S), 5.5 dB (C), 6 dB (L)",
        "code": [
            {"file": K, "line": 1289, "contains": "if wavel_nm < 1460:  # E-band"},
            {"file": K, "line": 529, "contains": "nf_lin[i] = _db_to_lin(6.0)"},
            {"file": K, "line": 1292, "contains": "nf_lin[i] = _db_to_lin(7.0)"},
            {"file": K, "line": 1294, "contains": "nf_lin[i] = _db_to_lin(5.5)"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "分档阈值 1460/1530/1565 nm 与四个 NF 值与论文逐字一致（论文只给了 NF 值，没给分档波长；"
                 "分档波长是工程取值，见 engineering 备注）。用户配置 noise_figure_db 时会覆盖。",
    },
    # ---- P1 §III：相干累积因子 eps ----------------------------------------
    {
        "id": "p1_epsilon_coherent",
        "quantity": "SPM 的相干累积因子 eps（N^{1+eps}）",
        "class": "physics",
        "paper": {
            "id": "P1",
            "locator": "Eq. (29) 的 N^{1+eps}（§III 末段解释）",
            "section": "§III",
            "anchor": "coherent accumulation",
            "file": P1,
        },
        "paper_formula": "N^{1+eps}，eps 刻画 SPM 沿跨段的**相干**累积（XPM 为非相干，故用 N）",
        "code": [
            {"file": K, "line": 760, "contains": "num = 2.0 * PI * k[l_ch] * k[lp] * (float(n_span) ** (1.0 + epsilon_n_span))"},
            {"file": K, "line": 740, "contains": "epsilon_n_span: float = 0.05"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "**公式结构符合**（N^{1+eps} 且 XPM 用 N，:493）。但论文**没有给出 eps 的数值**；"
                 "ecz 取 0.05 是论文未提及的标定量，不属于「与论文冲突」。",
    },
    # ---- P2 (5)：分段放大器 ASE -------------------------------------------
    {
        "id": "p2_eq05_lumped_ase",
        "quantity": "集总放大器单跨段 ASE（(G−1) 形式 + 任意增益定义）",
        "class": "physics",
        "paper": {
            "id": "P2",
            "locator": "Eq. (5)",
            "section": "§III.C ASE Noise",
            "anchor": "P_{\\text{ASE},i}^{\\prime } = 2(G_{i}-1)",
            "file": P2,
        },
        "paper_formula": "P'_ASE,i = 2(G_i − 1) n_sp,i h f_i B_i, n_sp,i ≈ (G_i NF − 1)/(2(G_i−1)), "
                         "G_i = P_i(L)/P_i(0) 为**放大器**在信道 i 上的增益",
        "code": [
            {"file": G, "line": 433, "contains": "gain = np.divide(power_in_w, p_out_fiber, out=np.ones_like(power_in_w), where=p_out_fiber > 0)"},
            {"file": G, "line": 445, "contains": "gain - 1, 0"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "P2 给出 (G−1) 的严格形式，ecz 的 max(gain−1,0) 与之对齐。"
                 "P2 还把 G_i 定义为放大器增益 = 出入功率比 —— 当 target_gain_db=None 时"
                 "（自动均衡跨段损耗）**这正是放大器增益**，故 isrs_gn.py:205 合规；"
                 "当 target_gain_db 给了定值时该比值不再等于放大器增益，此时必须用定值（见 p1_eq34_ase）。"
                 "fu4 无视这个区分、永远用 p_in/p_out，那才是违反 (34)/(5) 的地方。",
    },
    # ---- P2 (1)：SNR 倒数相加 ---------------------------------------------
    {
        "id": "p2_eq01_snr_inv",
        "quantity": "SNR^-1 = SNR_TRX^-1 + SNR_ASE^-1 + SNR_NLI^-1（各噪声源可加）",
        "class": "physics",
        "paper": {
            "id": "P2",
            "locator": "Eq. (1)",
            "section": "§III.A The Received SNR",
            "anchor": "P_{\\text{ASE,i}}^{\\prime \\prime } + \\eta _{n}(f_{i})P_{i}^{3}}",
            "file": P2,
        },
        "paper_formula": "SNR_i^-1 ≈ SNR_TRX,i^-1 + SNR_ASE,i^-1 + SNR_NLI,i^-1 "
                         "= (P_i / (kappa_i P_i + P_ASE,i'' + eta_n(f_i) P_i^3))^-1",
        "code": [
            {"file": G, "line": 1030, "contains": "total_noise = total_spm + total_xpm + total_ase"},
        ],
        "conforms": True,
        "verdict": "keep_ecz",
        "notes": "P2 (1) 允许把 TRX 实现代价作为独立可加项；ecz 把这条实现成"
                 "rwa/qot_checker.py 里的 booster_nf 指数项与 filtering_penalty。",
    },
    # ---- P2 (10)：rho 半解析式（ODE 线基准，本分支不实现） -----------------
    {
        "id": "p2_eq10_rho_semi_analytical",
        "quantity": "[ODE 线基准] rho(z,f_i) 半解析拟合式",
        "class": "physics",
        "paper": {
            "id": "P2",
            "locator": "Eq. (10)",
            "section": "§III.D",
            "anchor": "C_{f,i} P_{f}L_\\text{eff}",
            "file": P2,
        },
        "paper_formula": "rho(z,f_i) = e^{−alpha_i z}[1 − (C_f,i P_f L_eff + C_b,i P_b L_tilde_eff)(f_i − f_hat)]",
        "code": [],
        "conforms": None,
        "verdict": "ode_side_reference_only",
        "notes": "本仓是**闭式** P1 线，不用 ODE/半解析传播，故不实现。登记它是为了固定"
                 "「闭式 vs ODE」的对照基准；ode_accel 只是缓存档位，不改传播物理。",
    },
    # ---- P2 (2)/(4)：RA ODE ------------------------------------------------
    {
        "id": "p2_eq02_eq04_ra_ode",
        "quantity": "[ODE 线] 拉曼放大下信号/ASE 的耦合偏微分方程",
        "class": "physics",
        "paper": {
            "id": "P2",
            "locator": "Eq. (2) 与 Eq. (4)",
            "section": "§III.B / §III.C",
            "anchor": "\\pm \\frac{\\partial P_{i}}{\\partial z}",
            "file": P2,
        },
        "paper_formula": "± dP_i/dz = −sum_k (f_k/f_i) g_i(|df|)(P_k+P_ASE,k) P_i − … + … − alpha_i P_i",
        "code": [],
        "conforms": None,
        "verdict": "out_of_scope_this_branch",
        "notes": "本仓走 P1 闭式线，不做沿距离的 Raman ODE 积分。拉曼泵浦在本仓以 P1 (22) 的"
                 "k_bar 预加重项进入（见 p1_eq22_preemphasis），不是 P2 的分布式泵浦 ODE。",
    },
]

# ---------------------------------------------------------------------------
# 工程能力（J2：论文未提及 ⇒ 保留 ecz-dev）
# ---------------------------------------------------------------------------
ENGINEERING: list[dict] = [
    {
        "id": "eng_span_noise_accumulation",
        "quantity": "逐跨段噪声累积（按信号传递比缩放已有噪声 + 累加本跨段噪声）",
        "class": "engineering",
        "code": [{"file": G, "line": 996, "contains": "total_spm *= transfer"}],
        "verdict": "keep_ecz",
        "paper_note": "P1 用 (24) 的跨段求和与 (34) 的 ×N 表达多跨段；(29)(30) 用 N^{1+eps} / N。"
                      "「已有噪声按信号传递比缩放」这一递推写法是 P1 未规定的实现细节，"
                      "保留 ecz 实现。",
    },
    {
        "id": "eng_booster_ase",
        "quantity": "ROADM 均衡 booster 引入的 ASE",
        "class": "engineering",
        "code": [{"file": G, "line": 763, "contains": ") * np.asarray(self.channel_bandwidth_hz_channels, dtype=np.float64) * (terms.loss - 1.0)"}],
        "verdict": "keep_ecz",
        "paper_note": "P1 (34) 的 G/F 是「每级线路放大器 + DGE」，ROADM booster 属线路放大器，"
                      "故沿用同一 (34) 形式是合理的；论文未单独给出 booster 的 NF/增益口径，"
                      "保留 ecz 实现。",
    },
    {
        "id": "eng_filtering_penalty",
        "quantity": "级联滤波统计模型 SNR 惩罚（只乘噪声）",
        "class": "engineering",
        "code": [{"file": G, "line": 1023, "contains": "total_spm *= terms.filtering_penalty"}],
        "verdict": "keep_ecz",
        "paper_note": "P2 (1) 允许把实现代价作为独立可加项（SNR_TRX 项）；P1 没有这一项。"
                      "论文未提及具体口径，保留 ecz 实现。",
    },
    {
        "id": "eng_nf_band_edges",
        "quantity": "NF 分档波长阈值 1460/1530/1565 nm",
        "class": "engineering",
        "code": [{"file": K, "line": 1289, "contains": "wavel_nm < 1460"}],
        "verdict": "keep_ecz",
        "paper_note": "P1 §IV 只给了四个频段的 NF **值**（6/7/5.5/6 dB），没有给分档波长。"
                      "阈值属工程取值，保留 ecz 实现。",
    },
    {
        "id": "eng_kurtosis_eps_value",
        "quantity": "eps = 0.05 的标定值",
        "class": "engineering",
        "code": [{"file": K, "line": 740, "contains": "epsilon_n_span: float = 0.05"}],
        "verdict": "keep_ecz",
        "paper_note": "P1 (29) 引入 eps 但未给数值。标定值非公式，保留 ecz 实现。",
    },
    {
        "id": "eng_numeric_guards",
        "quantity": "phi 钳位到 ±EPSILON、方差 max(·,0)、edfa=None 时 ASE 置零",
        "class": "engineering",
        "code": [
            {"file": K, "line": 1041, "contains": "abs_phi_base = max(abs(phi_base), _EPSILON)"},
            {"file": K, "line": 1127, "contains": "sigma_spm_2 = np.where(p_in_w > 0, np.maximum(spm_pre * spm_sum, 0.0), 0.0)"},
            {"file": G, "line": 430, "contains": "s2_ase = np.zeros_like(power_in_w)"},
        ],
        "verdict": "keep_ecz",
        "paper_note": "纯数值防护，论文无对应式。",
    },
    {
        "id": "eng_blocked_channels",
        "quantity": "被阻断信道清零",
        "class": "engineering",
        "code": [{"file": G, "line": 1028, "contains": "power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0"}],
        "verdict": "keep_ecz",
        "paper_note": "网络层语义，论文无对应式。",
    },
    {
        "id": "eng_delta_f_co_value",
        "quantity": "三角近似的截断频率 df_co = 15 THz",
        "class": "engineering",
        "code": [{"file": G, "line": 283, "contains": "self.delta_f_co_hz = 15e12"}],
        "verdict": "keep_ecz",
        "paper_note": "P1 §II-B 引入 df_co 作为拟合参数但**未给数值**；§II-B 正文只说"
                      "「约 15 THz 之外拉曼增益迅速归零」。保留 ecz 取值。",
    },
]


def build() -> dict:
    return {
        "wp": "O2",
        "branch": "ode-dev",
        "generated_by": "tools/ode_formula_map.py",
        "papers": {
            "P1": {"file": P1, "role": "闭式 ECZ/GN（ecz-dev 的理论出处）"},
            "P2": {"file": P2, "role": "数值传播（上游 origin/main 的真 ODE 线）"},
        },
        "judgement": {
            "J1": "每条物理公式必须有 P1/P2 的公式号支撑",
            "J2": "论文提及且与 ecz 不一致 ⇒ 改成论文；论文未提及 ⇒ 保留 ecz",
            "J3": "wp/fg-uwb-fid 不是权威",
            "J4": "fu4 的 ASE 增益因子 G_fiber=p_in/p_out 与 P1 (34) 冲突 ⇒ 不得移植",
        },
        "physics": FORMULAS,
        "engineering": ENGINEERING,
        "summary": {
            "physics_total": len(FORMULAS),
            "physics_conforms": sum(1 for f in FORMULAS if f["conforms"] is True),
            "physics_violates": sum(1 for f in FORMULAS if f["conforms"] is False),
            "physics_reference_only": sum(1 for f in FORMULAS if f["conforms"] is None),
            "engineering_total": len(ENGINEERING),
        },
    }


def check(doc: dict) -> list[str]:
    """逐条复算 J1 守卫。返回错误列表（空 = 通过）。"""
    errs: list[str] = []
    cache: dict[str, str] = {}
    for item in FORMULAS + ENGINEERING:
        iid = item["id"]
        if item["class"] == "physics":
            paper = item.get("paper") or {}
            anchor = paper.get("anchor")
            if not anchor:
                errs.append(f"[J1] {iid}: physics 条目缺少 paper.anchor")
                continue
            if not paper.get("locator"):
                errs.append(f"[J1] {iid}: physics 条目缺少 paper.locator")
            pf = paper.get("file")
            if not pf:
                errs.append(f"[J1] {iid}: physics 条目缺少 paper.file")
                continue
            if pf not in cache:
                cache[pf] = (REPO / pf).read_text(encoding="utf-8")
            if anchor not in cache[pf]:
                errs.append(f"[J1] {iid}: anchor {anchor!r} 在 {pf} 里 grep 不到（公式号可能是编的）")
        else:
            if not item.get("paper_note"):
                errs.append(f"[J2] {iid}: engineering 条目必须写明 paper_note（为什么论文未提及）")

        for anchor in item.get("code", []):
            path = REPO / anchor["file"]
            if not path.is_file():
                errs.append(f"[J1] {iid}: 代码锚点文件不存在 {anchor['file']}")
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            ln = anchor["line"]
            if not (1 <= ln <= len(lines)):
                errs.append(f"[J1] {iid}: {anchor['file']}:{ln} 超出文件行数 {len(lines)}")
                continue
            if anchor["contains"] not in lines[ln - 1]:
                errs.append(
                    f"[J1] {iid}: {anchor['file']}:{ln} 不含 {anchor['contains']!r}"
                    f"（实际是 {lines[ln - 1].strip()!r}）"
                )
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写出 results/ode_dev_formula_map.json")
    ap.add_argument("--check", action="store_true", help="逐条复算 J1 守卫")
    args = ap.parse_args()

    doc = build()
    errs = check(doc)

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}  ({len(FORMULAS)} physics + {len(ENGINEERING)} engineering)")

    if args.check:
        if errs:
            print(f"FAIL: {len(errs)} 条")
            for e in errs:
                print("  " + e)
            return 1
        print(f"OK: {len(FORMULAS)} 条物理公式全部有可 grep 的论文锚点；{len(ENGINEERING)} 条工程能力均有 paper_note")
    if not args.write and not args.check:
        print(json.dumps(doc["summary"], indent=2, ensure_ascii=False))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
