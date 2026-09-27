"""多波段广义高斯噪声 (Generalized Gaussian Noise, GGN) 模型评估器。

ISRS-GN (Inter-channel Stimulated Raman Scattering - Generalized Noise)
"""

from dataclasses import dataclass, field

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_nf_lin_jit,
    _calc_span_noise_and_power_jit,
    _db_to_lin,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)]


@dataclass
class MultiSpanOpticalPath:
    """定义一条光路，包含多个跨段及其对应的配置。"""

    fiber_configs: list[FiberSpanConfig]
    edfa_configs: list[EDFAConfig]
    roadm_configs: list[ROADMConfig] | None = None
    # 实际路径中的每个 link_key, 方便关联到 network_state
    link_keys: list[tuple[int, int]] = field(default_factory=list)
    node_path: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证所有逐跳配置长度一致."""
        count = len(self.fiber_configs)
        if count == 0 or len(self.edfa_configs) != count:
            raise ValueError("Optical path requires matching non-empty fiber and EDFA config lists.")
        if self.roadm_configs is not None and len(self.roadm_configs) != count:
            raise ValueError("ROADM config count must match the number of route links.")
        if self.link_keys and len(self.link_keys) != count:
            raise ValueError("Link key count must match the number of route links.")


@dataclass(frozen=True)
class PathQoTResult:
    """统计物理层沿路径计算的逐信道结果."""

    signal_power_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    ase_noise_w: NDArrayFloat
    snr_db: NDArrayFloat
    span_count: int
    roadm_count: int


class MultiBandISRSGN:
    """多波段 ISRS-GN 模型评估器。

    用于快速计算多波段光网络中单跨段或多跨段的传输质量 (QoT)。
    """

    def __init__(
        self,
        grid: SpectrumGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
    ):
        """初始化 ISRS-GN 评估器。

        Args:
            grid: 频谱网格配置。
            ref_fiber_config: 参考光纤配置，用于计算各种波长相关参数。
            symbol_rate_hz: 系统的符号速率 (Hz)。
        """
        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.cr_w_m_hz = _SRS_CR  # 拉曼增益系数
        self.k_bar = min(sum(pump.power_w for pump in ref_fiber_config.raman_pumps), 1.0)
        self.delta_f_co_hz = 15e12  # 石英光纤受激拉曼散射（SRS）增益谱的截断频率
        self.n2_m2_w = ref_fiber_config.nonlinear_index_n2

        self._precompute_channel_dependent_properties()
        self._fiber_evaluator_cache: dict[str, MultiBandISRSGN] = {}

    def _precompute_channel_dependent_properties(self) -> None:
        """预计算所有信道通用的波长依赖物理属性，如波长、有效模面积等。

        这些属性在整个仿真中通常不变。
        """
        self.lambdas_m = self.grid.wavelengths  # (m)
        self.f_abs_hz = self.grid.frequencies  # (Hz)

        # 衰减系数 (Np/m)
        if self.ref_fiber_config.use_attenuation_polynomial:
            self.alpha_power_npm_channels = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                self.ref_fiber_config.reference_wavelength_nm,
                self.ref_fiber_config.attenuation_alpha2,
                self.ref_fiber_config.attenuation_alpha1,
                self.ref_fiber_config.attenuation_alpha0,
            )
        else:
            alpha = self.ref_fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            self.alpha_power_npm_channels = np.full(self.grid.num_channels, alpha)

        # 有效模面积 (m^2)
        self.a_eff_m2_channels = _calc_aeff_dynamic(
            self.lambdas_m,
            self.ref_fiber_config.effective_area_um2_ref,
            self.ref_fiber_config.aeff_slope_um2_nm,
            self.ref_fiber_config.reference_wavelength_nm,
        )

        # 噪声系数 (线性)
        self.nf_lin_channels = _calc_nf_lin_jit(self.lambdas_m)

        # beta2 和 beta3 (s^2/m, s^3/m)
        # 简化：在每个跨段的中心波长处计算 beta2 和 beta3，或使用参考波长
        # 这里使用参考波长处的 D 和 S 来计算
        ref_lambda_m = C_LIGHT / self.grid.center_frequency_hz
        self.beta2_s2_m = _calc_beta2_from_d(self.ref_fiber_config.dispersion_parameter_d, ref_lambda_m)
        self.beta3_s3_m = _calc_beta3_from_s_d(
            self.ref_fiber_config.dispersion_slope_s,
            self.ref_fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

        # 相对频率 (Hz)，用于 NLI/SRS 计算
        self.f_rel_hz = self.f_abs_hz - self.grid.center_frequency_hz
        self.f_m_hz = self.f_rel_hz[0] - self.symbol_rate_hz / 2
        self.f_M_hz = self.f_rel_hz[-1] + self.symbol_rate_hz / 2

    def _calc_span_noise_and_power(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """计算单跨段的功率演化、NLI 噪声和 ASE 噪声。

        Args:
            power_in_w: 各信道入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            edfa_config: 跨段后的 EDFA 配置。
            roadm_config: (可选) 跨段后的 ROADM 配置。
            n_effective_spans: 有效跨段数，用于 NLI 累积效应，对于单跨段通常为 1。

        Returns
        -------
            Tuple 包含:
            - power_out_w: 经历损耗、SRS 和放大后的出纤功率。
            - sigma2_spm_w: 产生的 SPM 噪声方差 (W)。
            - sigma2_xpm_w: 产生的 XPM 噪声方差 (W)。
            - sigma2_ase_w: 产生的 ASE 噪声方差 (W)。
        """
        # 1. 处理放大器噪声系数 (NF)
        # 如果 edfa_config 提供了全频段的 NF 数组则使用之，否则使用预计算的默认值
        if edfa_config is not None:
            if edfa_config.noise_figure_db is not None:
                noise_figure_db = np.asarray(edfa_config.noise_figure_db, dtype=float)
                if noise_figure_db.ndim == 0:
                    current_nf_lin = np.full_like(self.nf_lin_channels, 10 ** (float(noise_figure_db) / 10))
                else:
                    if noise_figure_db.shape != (self.grid.num_channels,):
                        raise ValueError("EDFA noise-figure array must contain one value per channel.")
                    current_nf_lin = 10 ** (noise_figure_db / 10)
        else:
            current_nf_lin = np.ones_like(self.nf_lin_channels)

        # 2. 调用 JIT 内核 (热路径)
        p_out_fiber, s2_spm, s2_xpm, _ = _calc_span_noise_and_power_jit(
            power_in_w=power_in_w,
            span_length_m=span_length_m,
            f_rel_hz=self.f_rel_hz,
            f_abs_hz=self.f_abs_hz,
            lambdas_m=self.lambdas_m,
            alpha_power_npm=self.alpha_power_npm_channels,
            a_eff_m2=self.a_eff_m2_channels,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            n2_m2_w=self.n2_m2_w,
            nf_lin_channels=current_nf_lin,
            symbol_rate_hz=self.symbol_rate_hz,
            n_effective_spans=n_effective_spans,
            cr_w_m_hz=self.cr_w_m_hz,
            k_bar=self.k_bar,
            delta_f_co_hz=self.delta_f_co_hz,
        )

        # 3. 显式应用 EDFA。旧实现返回未放大的出纤功率，且 edfa=None 仍产生 ASE。
        if edfa_config is None:
            p_out = p_out_fiber
            s2_ase = np.zeros_like(power_in_w)
        else:
            if edfa_config.target_gain_db is None:
                gain = np.divide(power_in_w, p_out_fiber, out=np.ones_like(power_in_w), where=p_out_fiber > 0)
            else:
                gain = np.full_like(power_in_w, 10 ** (edfa_config.target_gain_db / 10))
            if edfa_config.gain_ripple_db is not None:
                ripple = np.asarray(edfa_config.gain_ripple_db, dtype=float)
                if ripple.shape != (self.grid.num_channels,):
                    raise ValueError("EDFA gain-ripple array must contain one value per channel.")
                gain *= 10 ** (ripple / 10)
            p_out = p_out_fiber * gain
            s2_ase = current_nf_lin * 6.62607015e-34 * self.f_abs_hz * self.symbol_rate_hz * np.maximum(gain - 1, 0)
            s2_ase[power_in_w <= 0] = 0

        # 4. ROADM 损耗同等衰减信号和已生成噪声。
        if roadm_config:
            loss_lin = _db_to_lin(roadm_config.path_loss_db())
            p_out /= loss_lin
            s2_spm /= loss_lin
            s2_xpm /= loss_lin
            s2_ase /= loss_lin

        return p_out, s2_spm, s2_xpm, s2_ase

    def for_fiber(self, fiber_config: FiberSpanConfig) -> "MultiBandISRSGN":
        """返回匹配链路光纤参数的缓存评估器."""
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = MultiBandISRSGN(self.grid, fiber_config, self.symbol_rate_hz)
        return self._fiber_evaluator_cache[key]

    # Backward-compatible internal alias for pre-integration callers.
    _for_fiber = for_fiber

    def evaluate_path(self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat) -> PathQoTResult:
        """按拓扑顺序评估光纤、跨段 EDFA、ROADM 和 booster."""
        power = np.asarray(power_in_w, dtype=np.float64).copy()
        if power.shape != (self.grid.num_channels,) or np.any(power < 0):
            raise ValueError("power_in_w must be a non-negative vector with one value per channel.")
        total_spm = np.zeros_like(power)
        total_xpm = np.zeros_like(power)
        total_ase = np.zeros_like(power)
        span_count = 0
        roadm_configs = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)

        for hop_index, (fiber, edfa, roadm) in enumerate(
            zip(optical_path.fiber_configs, optical_path.edfa_configs, roadm_configs, strict=True)
        ):
            evaluator = self.for_fiber(fiber)
            remaining_km = fiber.length_km
            while remaining_km > 1e-12:
                span_km = min(remaining_km, fiber.max_span_length_km)
                previous_power = power
                power, spm, xpm, ase = evaluator._calc_span_noise_and_power(
                    power, span_km * 1e3, edfa, n_effective_spans=1
                )
                transfer = np.divide(power, previous_power, out=np.zeros_like(power), where=previous_power > 0)
                total_spm *= transfer
                total_xpm *= transfer
                total_ase *= transfer
                total_spm += spm
                total_xpm += xpm
                total_ase += ase
                span_count += 1
                remaining_km -= span_km

            if roadm is None:
                continue
            blocked = [index for index in roadm.blocked_channels if index < self.grid.num_channels]
            operation = "drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"
            loss_db = roadm.path_loss_db(operation)
            loss = 10 ** (loss_db / 10)
            power /= loss
            total_spm /= loss
            total_xpm /= loss
            total_ase /= loss
            if roadm.equalize_output_power:
                gain = loss
                power *= gain
                total_spm *= gain
                total_xpm *= gain
                total_ase *= gain
                nf = 10 ** (roadm.booster_noise_figure_db / 10)
                booster_ase = nf * 6.62607015e-34 * self.f_abs_hz * self.symbol_rate_hz * (gain - 1)
                booster_ase[power <= 0] = 0
                total_ase += booster_ase
            penalty = 10 ** (roadm.filtering_penalty_db / 10)
            total_spm *= penalty
            total_xpm *= penalty
            total_ase *= penalty
            if blocked:
                power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0

        total_noise = total_spm + total_xpm + total_ase
        snr = np.full_like(power, -np.inf)
        active = (power > 0) & (total_noise > 0)
        snr[active] = 10 * np.log10(power[active] / total_noise[active])
        snr[(power > 0) & (total_noise == 0)] = np.inf
        return PathQoTResult(
            power, total_spm, total_xpm, total_ase, snr, span_count, sum(roadm is not None for roadm in roadm_configs)
        )
