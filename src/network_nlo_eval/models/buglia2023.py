r"""Buglia 2023 闭式 ISRS-GN 模型：任意损耗、任意纤长。

文献
----
H. Buglia, M. Jarmolovičius, A. Vasylchenkova, E. Sillekens, L. Galdino, R. I. Killey,
P. Bayvel, "A Closed-Form Expression for the Gaussian Noise Model in the Presence of
Inter-Channel Stimulated Raman Scattering Extended for Arbitrary Loss and Fibre Length",
J. Lightwave Technol. vol. 41, no. 11, pp. 3577-3588, Jun. 2023,
doi:10.1109/JLT.2023.3256185.

模型要点
--------
相对 Semrau 2019 的唯一结构性改动在论文 Eq. (30)：链路函数里

.. math::

    \frac{1 - e^{-(\alpha_{l,i} - j\phi)L}}{-\alpha_{l,i} + j\phi}
    \;\approx\; \frac{\kappa_{l,i}}{-\tilde{\alpha}_{l,i} + j\phi}

取代了 Semrau 的 Eq. (42)（要求 :math:`\alpha_{l,i} L \to \infty`）。新近似用
:math:`\tilde{\alpha}_{l,i}` (Eq. (15)) 与 :math:`\kappa_{l,i}` (Eq. (16)) 在
:math:`\phi = 0` 处做一阶 Taylor 匹配，因此**短跨与极低损耗也成立**（论文
Appendix D 给出适用条件 :math:`\alpha_{l,i} L \gg 0`）。

本模块实现的闭式链条：

* Eq. (13)：功率剖面 :math:`\rho(z,f_i) = e^{-\alpha_i z}[1 - P_{tot} C_{r,i} f_i L_{eff}(z)]`；
* Eq. (14)：链路函数闭式；
* Eq. (15)/(16)：:math:`\tilde{\alpha}_{l,i}` 与 :math:`\kappa_{l,i}`；
* Eq. (17)：单对干扰信道的 XPM 贡献；
* Eq. (18)：SPM 贡献。

相位失配按论文取值：:math:`\phi_i = -4\pi^2(\beta_2 + 2\pi\beta_3 f_i)`、
:math:`\phi_{i,k} = -4\pi^2(f_k-f_i)[\beta_2 + \pi\beta_3(f_i+f_k)]`。

本模型只做 GN。论文 Eq. (19)/(20) 的调制格式修正项未实现（见下）。

实现取舍（与论文的偏差）
------------------------
1. **剖面参数不做逐链路拟合。** 论文 Eq. (13) 里的 :math:`\alpha_i`、
   :math:`\bar\alpha_i`、:math:`C_{r,i}` 是为复现数值拉曼剖面而**拟合**出来的
   逐信道参数，论文自己也指出"每换一条链路就要重新拟合"。本实现改用可解析、
   逐信道确定的取值：:math:`\alpha_i` 取该波长的光纤功率衰减（支持衰减多项式），
   :math:`\bar\alpha_i = \alpha_i`（Semrau 的默认），:math:`C_{r,i}` 取该光纤的
   拉曼增益谱斜率常数。这样"任意损耗 / 任意纤长"的推广完全来自 Eq. (30) 的链路
   函数近似本身，不依赖任何迭代优化。
2. **出纤功率用冻结核的拉曼闭式剖面**（``models.numba_kernels`` 的 ECZ 三角增益
   近似），而不是论文的半解析式 (13)。理由：(13) 是一阶展开，强拉曼下括号可能
   变负；ECZ 闭式数值稳健，且与本仓库的 ``ecz`` 基线口径一致，便于横向比较。
   NLI 一律走论文闭式。
3. **未实现 Eq. (19)/(20) 的调制格式修正项**（论文 Eq. (21) 里的 :math:`\Phi`
   修正）。本模型是 GN 模型，修正项需要超额峰度这一外部参数；论文 Eq. (20) 与
   Eq. (21) 的前因子记账本身也不自洽（``2\pi`` 与 ``5/6``），需要先厘清。
4. **相干因子 :math:`n^\epsilon`** 按论文 Eq. (3) 施加在 SPM 上，但冻结的
   :mod:`network_nlo_eval.models.route` 逐跨累加噪声时固定传 ``n=1``，所以多跨
   场景下该因子实际不生效（``ecz`` 基线同样如此）。
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.core.constants import PI
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, SpanFiberOutput, register_model
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_gamma_coeff_scalar,
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
)
from network_nlo_eval.models.route import RouteModel
from network_nlo_eval.network.elements import FiberSpanConfig

__all__ = [
    "BUGLIA2023_MODEL_INFO",
    "Buglia2023Model",
    "Buglia2023SpanKernel",
    "alpha_tilde_kappa",
]

#: 拉曼增益谱斜率 ``C_r`` [1/(W·m·Hz)]。
#:
#: 与冻结核 :mod:`network_nlo_eval.models.numba_kernels` 的 ``_SRS_CR`` 同一口径：
#: 该系数乘以 ``P_tot * f`` (W·Hz) 与 ``L_eff`` (m) 得到无量纲的拉曼耗尽指数，
#: 与论文 Eq. (13) 的 ``P_tot C_{r,i} f_i L_eff(z)`` 一致。
_SRS_CR = 0.028 / 1e3 / 1e12

#: GN 相干因子指数 :math:`\epsilon`（Poggiolini 2012, Eq. (22)）。
_COHERENCE_EPSILON = 0.05

#: 相位失配的夹取下界，避免 :math:`1/\phi` 在自配对处发散。
_EPS_PHI = 1e-35

#: SRS 增益谱的截断频率 (Hz)。
_DELTA_F_CO_HZ = 15e12

BUGLIA2023_MODEL_INFO = ModelInfo(
    name="buglia2023",
    summary="Buglia 2023 closed-form ISRS-GN for arbitrary loss and fibre length.",
    family="gn",
    reference="Buglia et al., 'A Closed-Form Expression for the Gaussian Noise Model in the "
    "Presence of Inter-Channel Stimulated Raman Scattering Extended for Arbitrary Loss and "
    "Fibre Length', JLT 2023, doi:10.1109/JLT.2023.3256185",
    options=(
        ("egn", "Not supported: the paper derives the GN model only."),
    ),
)


def alpha_tilde_kappa(alpha_l: float, length_m: float) -> tuple[float, float]:
    r"""论文 Eq. (15)/(16)：一阶 Taylor 匹配出来的有效衰减与幅度系数。

    .. math::

        \tilde{\alpha}_{l,i} = \frac{\alpha_{l,i}(1 - e^{-\alpha_{l,i} L})}
        {1 - e^{-\alpha_{l,i} L} - \alpha_{l,i} L e^{-\alpha_{l,i} L}},\qquad
        \kappa_{l,i} = \frac{\tilde{\alpha}_{l,i}(1 - e^{-\alpha_{l,i} L})}{\alpha_{l,i}}

    极限行为：:math:`\alpha_{l,i} L \to \infty` 时 :math:`\tilde{\alpha} \to \alpha`、
    :math:`\kappa \to 1`（即退回 Semrau 的 Eq. (42)）；:math:`\alpha_{l,i} L \to 0` 时
    :math:`\tilde{\alpha} \to 2/L`、:math:`\kappa \to 2`，链路函数退化为零损耗的
    :math:`L^2`。两种极限都用级数展开求值，避免 ``1 - (1+x) e^{-x}`` 的相消误差。

    Args:
        alpha_l: :math:`\alpha_{l,i} = \alpha_i + l\tilde{\alpha}_i` (Np/m)。
        length_m: 跨段长度 (m)。
    Returns:
        ``(tilde_alpha, kappa)``，分别是有效衰减 (Np/m) 与无量纲幅度系数。
    """
    x = alpha_l * length_m
    if x < 1e-3:
        # 级数：1 - e^{-x} = x - x^2/2 + x^3/6 - x^4/24 ...
        em1 = x * (1.0 - x / 2.0 + x * x / 6.0 - x * x * x / 24.0)
        # 级数：1 - (1+x) e^{-x} = x^2/2 - x^3/3 + x^4/8 - x^5/30 ...
        denom = x * x * (0.5 - x / 3.0 + x * x / 8.0 - x * x * x / 30.0)
    else:
        exp_neg = np.exp(-x)
        em1 = 1.0 - exp_neg
        denom = 1.0 - (1.0 + x) * exp_neg
    tilde_alpha = alpha_l * em1 / denom
    kappa = em1 * em1 / denom
    return float(tilde_alpha), float(kappa)


class Buglia2023SpanKernel:
    """Buglia 2023 单跨物理核。

    只做单跨：出纤功率用冻结核的拉曼闭式功率剖面，NLI 用论文 Eq. (17)/(18) 的闭式。
    放大器、ROADM 与噪声累积在 :mod:`network_nlo_eval.models.route` 里统一处理。

    Attributes
    ----------
        n_channels: 信道数 N。
    """

    def __init__(
        self,
        *,
        f_rel_hz: NDArrayFloat,
        lambdas_m: NDArrayFloat,
        a_eff_m2: NDArrayFloat,
        alpha_npm: NDArrayFloat,
        beta2_s2_m: float,
        beta3_s3_m: float,
        n2_m2_w: float,
        b_hz_channels: NDArrayFloat,
        c_r_w_m_hz: float = _SRS_CR,
        symbol_rate_hz: float = 50e9,
    ) -> None:
        r"""构造单跨核。

        Args:
            f_rel_hz: ``(N,)`` 相对频率 (Hz)，以波段中心为原点。
            lambdas_m: ``(N,)`` 绝对波长 (m)。
            a_eff_m2: ``(N,)`` 有效模面积 (m^2)。
            alpha_npm: ``(N,)`` 逐信道功率衰减系数 (Np/m)。
            beta2_s2_m: :math:`\beta_2` (s^2/m)。
            beta3_s3_m: :math:`\beta_3` (s^3/m)。
            n2_m2_w: 非线性折射率 (m^2/W)。
            b_hz_channels: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
            c_r_w_m_hz: 拉曼增益谱斜率 ``C_r`` [1/(W·m·Hz)]。
            symbol_rate_hz: 符号率 (Hz)，只用于确定 SRS 频带边界。
        """
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.lambdas_m = np.asarray(lambdas_m, dtype=np.float64)
        self.a_eff_m2 = np.asarray(a_eff_m2, dtype=np.float64)
        self.alpha_npm = np.asarray(alpha_npm, dtype=np.float64)
        self.beta2_s2_m = float(beta2_s2_m)
        self.beta3_s3_m = float(beta3_s3_m)
        self.n2_m2_w = float(n2_m2_w)
        self.b_hz_channels = np.asarray(b_hz_channels, dtype=np.float64)
        self.c_r_w_m_hz = float(c_r_w_m_hz)
        self.symbol_rate_hz = float(symbol_rate_hz)

        self.n_channels = int(self.f_rel_hz.shape[0])

        # Eq. (17)/(18) 的 SPM 相位失配：phi_i = -4 pi^2 (beta_2 + 2 pi beta_3 f_i)。
        self.phi_i = -4.0 * PI**2 * (self.beta2_s2_m + 2.0 * PI * self.beta3_s3_m * self.f_rel_hz)
        self.gamma_ii = np.array(
            [
                _calc_gamma_coeff_scalar(
                    self.lambdas_m[i], self.lambdas_m[i], self.n2_m2_w, self.a_eff_m2[i], self.a_eff_m2[i]
                )
                for i in range(self.n_channels)
            ],
            dtype=np.float64,
        )

        # (N, N) 常量按跨长缓存：只依赖信道几何、频带与跨长，不随功率变。
        self._pair_cache: dict[float, tuple[NDArrayFloat, ...]] = {}

    # --- 内部常量 ---------------------------------------------------------
    def _profile_coefficients(self, length_m: float) -> tuple[NDArrayFloat, ...]:
        r"""给定跨长，算出 Eq. (15)/(16) 的逐信道量与 Eq. (17) 的 ``(N, N)`` 配对矩阵。

        返回 ``(m0, m1, a0, k0, a1, k1)``：

        * ``m0[i, k]``、``m1[i, k]``：Eq. (17) 的两项配对权重，含 :math:`\gamma_{ik}^2`；
        * ``a0/k0``、``a1/k1``：逐信道的 :math:`\tilde{\alpha}_{l,k}` 与 :math:`\kappa_{l,k}`。

        Args:
            length_m: 跨段长度 (m)。
        Returns:
            上述六个数组。
        """
        cached = self._pair_cache.get(length_m)
        if cached is not None:
            return cached

        n = self.n_channels
        alpha = self.alpha_npm
        a0 = np.empty(n, dtype=np.float64)
        k0 = np.empty(n, dtype=np.float64)
        a1 = np.empty(n, dtype=np.float64)
        k1 = np.empty(n, dtype=np.float64)
        for i in range(n):
            a0[i], k0[i] = alpha_tilde_kappa(alpha[i], length_m)
            # alpha_{1,i} = alpha_i + 1 * bar_alpha_i，bar_alpha_i = alpha_i。
            a1[i], k1[i] = alpha_tilde_kappa(2.0 * alpha[i], length_m)

        # phi_{i,k} = -4 pi^2 (f_k - f_i) [beta_2 + pi beta_3 (f_i + f_k)]
        freq = self.f_rel_hz
        delta = freq[None, :] - freq[:, None]
        total = freq[:, None] + freq[None, :]
        phi_ik = -4.0 * PI**2 * delta * (self.beta2_s2_m + PI * self.beta3_s3_m * total)
        phi_safe = np.where(np.abs(phi_ik) < _EPS_PHI, np.where(phi_ik >= 0.0, _EPS_PHI, -_EPS_PHI), phi_ik)

        # gamma_{ik}（同信道时为 gamma_ii）
        gamma_ik = (2.0 * PI / self.lambdas_m)[:, None] * (
            2.0 * self.n2_m2_w / (self.a_eff_m2[:, None] + self.a_eff_m2[None, :])
        )
        gamma_sq = gamma_ik**2

        band = self.b_hz_channels[:, None]
        atan0 = np.arctan(phi_safe * band / (2.0 * a0[None, :]))
        atan1 = np.arctan(phi_safe * band / (2.0 * a1[None, :]))
        # 因子 4 来自 (l, l') 求和按 (l <-> l') 对称折叠：原始求和是
        # sum_{l,l'} 2*kappa_l*kappa_l'*(A_l + A_l')/(phi*(a_l + a_l'))，
        # 对称化后等于 4/phi * sum_{l,l'} kappa_l*kappa_l'*A_l/(a_l + a_l')。
        m0 = 4.0 * gamma_sq * atan0 / phi_safe
        m1 = 4.0 * gamma_sq * atan1 / phi_safe
        # 自配对不进 XPM 求和（论文 Eq. (4) 的 k != i）。
        np.fill_diagonal(m0, 0.0)
        np.fill_diagonal(m1, 0.0)

        result = (m0, m1, a0, k0, a1, k1)
        self._pair_cache[length_m] = result
        return result

    # --- 单跨核 -----------------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """计算一个跨段的出纤功率与 NLI 噪声。

        Args:
            power_in_w: ``(N,)`` 跨段入口每信道功率 (W)。0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 有效跨段数，SPM 的相干因子用（论文 Eq. (3)）。
        Returns:
            出纤功率与 SPM/XPM 噪声，均为 ``(N,)`` 且不与输入共享内存。
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        length_m = float(span_length_m)
        n_spans = max(int(n_effective_spans), 1)

        p_total = float(np.sum(power))

        # --- 出纤功率：拉曼闭式剖面（与冻结核同一套 SRS 三角增益近似）---
        f_m_hz = self.f_rel_hz[0] - self.symbol_rate_hz / 2.0
        f_M_hz = self.f_rel_hz[-1] + self.symbol_rate_hz / 2.0
        r_f = _calc_raman_profile_jit(self.f_rel_hz, f_m_hz, f_M_hz, _DELTA_F_CO_HZ, p_total)
        power_out = _calc_span_out_power_jit(power, self.f_rel_hz, self.alpha_npm, length_m, r_f, 0.0)

        # --- Eq. (13) 的剖面参数：T_i = 1 + T_tilde_i, T_tilde_i = -P_tot C_r f_i / bar_alpha_i ---
        m0, m1, a0, k0, a1, k1 = self._profile_coefficients(length_m)
        t_tilde = -p_total * self.c_r_w_m_hz * self.f_rel_hz / self.alpha_npm
        t_factor = 1.0 + t_tilde
        # T_i <= 0 意味着一阶拉曼展开已经失效（强拉曼）；夹到极小正值避免除零。
        t_factor = np.where(np.abs(t_factor) < 1e-12, 1e-12, t_factor)
        weight = -t_tilde / t_factor

        # Eq. (17)/(18) 的 (l, l') 求和按对称性折叠成两项，见模块文档。
        sum_a0_a1 = a0 + a1
        c0 = k0 * k0 / (2.0 * a0) + weight * k0 * k1 / sum_a0_a1
        c1 = weight * k0 * k1 / sum_a0_a1 + weight * weight * k1 * k1 / (2.0 * a1)

        # --- Eq. (18)：SPM ---
        phi_safe = np.where(np.abs(self.phi_i) < _EPS_PHI, np.where(self.phi_i >= 0.0, _EPS_PHI, -_EPS_PHI), self.phi_i)
        band_sq = self.b_hz_channels**2
        asinh0 = np.arcsinh(3.0 * phi_safe * band_sq / (8.0 * PI * a0))
        asinh1 = np.arcsinh(3.0 * phi_safe * band_sq / (8.0 * PI * a1))
        eta_spm = (
            (16.0 / 27.0)
            * (self.gamma_ii**2 / band_sq)
            * t_factor**2
            * (4.0 * PI / phi_safe)
            * (asinh0 * c0 + asinh1 * c1)
        )
        spm_noise = eta_spm * power**3 * n_spans**_COHERENCE_EPSILON

        # --- Eq. (17)：XPM ---
        u0 = (power**2 / self.b_hz_channels) * t_factor**2 * c0
        u1 = (power**2 / self.b_hz_channels) * t_factor**2 * c1
        xpm_noise = (32.0 / 27.0) * power * (m0 @ u0 + m1 @ u1)

        spm_noise = np.where(np.isfinite(spm_noise), spm_noise, 0.0)
        xpm_noise = np.where(np.isfinite(xpm_noise), xpm_noise, 0.0)
        return SpanFiberOutput(power_out, np.maximum(spm_noise, 0.0), np.maximum(xpm_noise, 0.0))


class Buglia2023Model(RouteModel):
    """Buglia 2023 闭式 ISRS-GN 模型评估器（GN only）。

    在冻结契约 :class:`network_nlo_eval.models.route.RouteModel` 之上只实现单跨核：
    逐信道常数在模型层准备好，逐跨核按 :class:`FiberSpanConfig` 记忆化。
    """

    info = BUGLIA2023_MODEL_INFO

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        **route_options,
    ) -> None:
        """初始化模型。

        Args:
            grid: 频谱网格（遗留 ``SpectrumGrid`` 或 ``FlexibleGrid``）。
            ref_fiber_config: 参考光纤配置，提供色散、非线性与有效模面积。
            symbol_rate_hz: 符号率 (Hz)。
            egn: 论文只给出 GN 模型，``True`` 直接报错，不静默降级。
            **route_options: 透传给 :class:`RouteModel` 的选项。
        Raises:
            ValueError: ``egn=True``。
        """
        if egn:
            raise ValueError(
                "buglia2023 implements the GN model only: Buglia et al. (JLT 2023, "
                "doi:10.1109/JLT.2023.3256185) derive the closed-form expression for the "
                "Gaussian-noise model with ISRS; no EGN variant is published. "
                "Use egn=False."
            )
        super().__init__(grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options)

        self.ref_fiber_config = ref_fiber_config
        self.n2_m2_w = float(ref_fiber_config.nonlinear_index_n2)
        self.a_eff_m2 = _calc_aeff_dynamic(
            self.lambdas_m,
            ref_fiber_config.effective_area_um2_ref,
            ref_fiber_config.aeff_slope_um2_nm,
            ref_fiber_config.reference_wavelength_nm,
        )
        ref_lambda_m = 2.99792458e8 / self.band_center_hz
        self.beta2_s2_m = _calc_beta2_from_d(ref_fiber_config.dispersion_parameter_d, ref_lambda_m)
        self.beta3_s3_m = _calc_beta3_from_s_d(
            ref_fiber_config.dispersion_slope_s,
            ref_fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> Buglia2023SpanKernel:
        """按光纤参数构造单跨核。

        Args:
            fiber_config: 该跨段的光纤配置。
        Returns:
            与该光纤配置匹配的单跨核。
        """
        if fiber_config.use_attenuation_polynomial:
            alpha_npm = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            alpha_db_km = fiber_config.attenuation_db_km_ref
            alpha_npm = np.full(self.n_nodes, alpha_db_km * (np.log(10) / 10.0) / 1e3, dtype=np.float64)
        return Buglia2023SpanKernel(
            f_rel_hz=self.f_rel_hz,
            lambdas_m=self.lambdas_m,
            a_eff_m2=self.a_eff_m2,
            alpha_npm=alpha_npm,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            n2_m2_w=self.n2_m2_w,
            b_hz_channels=self.channel_bandwidth_hz_channels,
            c_r_w_m_hz=_SRS_CR,
            symbol_rate_hz=self.symbol_rate_hz,
        )


register_model(BUGLIA2023_MODEL_INFO, Buglia2023Model)
