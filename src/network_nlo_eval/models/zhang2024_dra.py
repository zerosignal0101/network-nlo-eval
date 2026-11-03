"""Zhang 2024 后向 DRA 相干 EGN 论文模型（``zhang2024_dra_egn``）。

Reference
---------
Y. Zhang, Q. Guo, L. Gan, L. Yi, Q. Qiu, W. Hu, Z. Wang, Q. Zhuge,
"Accurate Closed-Form Coherent EGN Model Supporting Distributed Raman
Amplification", OECC 2024, doi:10.1109/OECC54135.2024.10975528.

实现范围
--------
* **Eq. (1)**：后向 DRA 的三指数剖面，``(T, A, D, d)`` 逐信道拟合到耦合拉曼
  ODE 的解（论文 Sec. II-A 的流程）。
* **Eq. (2)**：纵向积分的闭式。论文印刷式只保留了积分上限、丢掉了
  ``e^{-alpha L}`` 等下限量；本实现把三项的公共相位 ``e^{j phi L}`` 统一补全，
  否则前向项与后向项落在不同相位参考上，Eq. (4) 的跨间相干无从定义
  （详见 :mod:`network_nlo_eval.models.zhang2024_coherent`）。
* **Eq. (3)-(5)**：``eta = eta_SPM + eta_XPM``；跨间相位放在**频率积分内部**，
  同跨与跨间项（``P_FmP_Fn``、``P_FmP_Bn``、``P_BmP_Bn``）全部保留——论文
  只是说后几项较小不纳入 CFM，本实现不丢。
* **Sec. II-C**：``P.P.^EGN/P.P.^CFM = prod 1/(a_i exp(b_i Param_i^{c_i}+d_i))``
  的标定流程；系数由本仓在 Table I 参数范围内拟合，见
  ``models/data/zhang2024_calibration.json``。

未复现的部分（论文没有印出，无法从 PDF 恢复）
----------------------------------------------
1. Eq. (5) 每个 ``P_{F(B)m(n)}`` 的解析式（Sec. II-B 明说因篇幅不给）；
2. 标定系数 ``a_i, b_i, c_i, d_i``、``Param_i`` 清单与 512 组数据；
3. ``eta_XPM^corr``：Eq. (3) 的调制格式修正项，论文只引了 Carena [12]，
   没有给出表达式。本模型因此是**相干 GN + 本仓标定**，不是论文的完整 EGN。
   调用方可用 ``modulation_format_correction=False``（默认）显式记录这一点。

相对路由层"每跨一个标量相位"的差别
------------------------------------
本模型在 ``evaluate_path`` 里对整条链路做一次 NLI 计算：跨间相位
``exp(j phi z_m)`` 在频率积分内部逐点相乘，因此同跨项与跨间项的相干是精确的，
而不是把一个等效相位斜坡塞给每跨的标量振幅。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, PathQoTResult, SpanContext, register_model
from network_nlo_eval.models.dra_common import (
    DEFAULT_DRA_NLI_QUADRATURE,
    DEFAULT_DRA_STEPS,
    DRARamanModelBase,
    DRASpanKernelBase,
)
from network_nlo_eval.models.raman import C_LIGHT
from network_nlo_eval.models.zhang2024_coherent import (
    CalibrationSet,
    CalibrationTerm,
    ZhangCoherentSpan,
    ZhangSpanProfile,
    cfm_and_egn_terms,
    fit_zhang_coherent_profile,
)
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "ZHANG2024_DRA_MODEL_INFO",
    "Zhang2024DRAModel",
    "Zhang2024DRASpanKernel",
    "default_calibration",
    "load_calibration",
]

#: 本仓拟合的标定系数文件（相对本模块）。
_CALIBRATION_FILE = Path(__file__).with_name("data") / "zhang2024_calibration.json"


def load_calibration(path: Path | None = None) -> dict[str, CalibrationSet]:
    """读取标定系数 JSON。

    Args:
        path: 文件路径；``None`` 时用仓库内的
            ``models/data/zhang2024_calibration.json``。

    Returns
    -------
        项名 → :class:`CalibrationSet`；文件不存在时返回空字典（等价于不标定）。
    """
    target = _CALIBRATION_FILE if path is None else Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    result: dict[str, CalibrationSet] = {}
    for term, entries in payload.get("terms", {}).items():
        result[term] = CalibrationSet(
            terms=tuple(
                CalibrationTerm(name=item["name"], parameters=tuple(item["parameters"]))  # type: ignore[arg-type]
                for item in entries
            )
        )
    return result


def default_calibration() -> dict[str, CalibrationSet]:
    """仓库内置标定（由 ``scripts/fit_zhang2024_calibration.py`` 生成）。"""
    return load_calibration()


class Zhang2024DRASpanKernel(DRASpanKernelBase):
    """Zhang 2024 的单跨核：只负责 ODE 剖面与功率，NLI 由链路层统一算。

    NLI 必须放到链路层，因为跨间相位在频率积分内部；单跨核返回零 NLI，
    真正的 NLI 在 :meth:`Zhang2024DRAModel.evaluate_path` 里一次算完。
    """

    profile_kind = "zhang_eq1"
    emits_nli_amplitudes = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.last_profile: ZhangSpanProfile | None = None
        self.gamma_per_w_m = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (self.n2_m2_w / self.a_eff_m2)

    def _fit_profile(self, profile_norm: NDArrayFloat, span_length_m: float) -> ZhangSpanProfile:
        """拟合 Eq. (1) 的 ``(T, A, D, d)``。"""
        return fit_zhang_coherent_profile(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_npm,
            self._n_steps,
        )

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: ZhangSpanProfile,  # type: ignore[override]
        span_context: SpanContext | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """单跨核不产 NLI：链路层会覆盖（见类文档）。"""
        del power_internal, span_context
        self.last_profile = profile
        return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)


ZHANG2024_DRA_MODEL_INFO = ModelInfo(
    name="zhang2024_dra_egn",
    summary=(
        "Zhang 2024 backward-DRA coherent EGN: Eq. (1) profile, Eq. (2) closed-form "
        "longitudinal integral, Eq. (4) span-coherent NLI with the inter-span phase "
        "inside the frequency integral, plus the Sec. II-C per-term calibration "
        "fitted in this repository. eta_XPM_corr is not printed in the paper and is "
        "not implemented."
    ),
    family="egn+raman+coherent",
    reference="Zhang et al., OECC 2024, doi:10.1109/OECC54135.2024.10975528",
    options=(
        ("raman_pumps", "Backward RamanPumpConfig sequence; omitted uses FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear' (CZ)."),
        ("raman_ase", "Expose the newly generated Raman ASE before the EDFA. Default True."),
        ("raman_temperature_k", "Fibre temperature of the spontaneous Raman factor. Default 300 K."),
        ("raman_peak_gain_coefficient_m_per_w", "Peak local Raman gain rate. Default 6e-14."),
        ("include_isrs", "Include channel-channel ISRS in the coupled ODE. Default True."),
        ("fixed_point_iterations", "Maximum backward-pump fixed-point iterations. Default 24."),
        ("n_steps", "RK4 steps of the ODE and of the Eq. (1) fit. Default 32."),
        ("n_quadrature", "Gauss-Legendre nodes of the XPM frequency integral. Default 24."),
        ("n_spm_quadrature", "Gauss-Legendre nodes per SPM frequency direction. Default 24."),
        ("calibration", "Override the per-term calibration dict; None uses the built-in JSON."),
        ("use_calibration", "Apply the Sec. II-C calibration factors. Default True."),
        ("modulation_format_correction", "Reserved for eta_XPM_corr; the paper does not print it."),
    ),
)


class Zhang2024DRAModel(DRARamanModelBase):
    """Zhang 2024 后向 DRA 相干 EGN 模型。

    只有后向泵（论文 Sec. II-A）。

    Args:
        grid: 频谱栅格。
        ref_fiber_config: 参考光纤。
        symbol_rate_hz: 符号率 (Hz)。
        connections: 可选逐连接栅格。
        raman_pumps: 每跨的后向泵浦；``None`` 用光纤配置里的泵浦。
        raman_gain_model: ``"gaussian"`` 或 ``"linear"``。
        raman_ase: 是否在 EDFA 前计入拉曼 ASE。
        raman_temperature_k: 光纤温度 (K)。
        raman_peak_gain_coefficient_m_per_w: 峰值拉曼增益率。
        include_isrs: 是否计入信道-信道 ISRS。
        fixed_point_iterations: 后向泵不定点迭代上限。
        n_steps: ODE 的 RK4 步数。
        n_quadrature: XPM 频率积分节点数。
        n_spm_quadrature: SPM 每个频率方向的节点数。
        calibration: 覆盖内置标定系数；``None`` 用仓库 JSON。
        use_calibration: 是否施加标定因子。
        modulation_format_correction: 论文未给出 ``eta_XPM^corr``，置 True 会报错。
        **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。

    Raises
    ------
        ValueError: 传入前向泵，或请求了论文未给出的调制格式修正项。
    """

    info = ZHANG2024_DRA_MODEL_INFO
    _kernel_class = Zhang2024DRASpanKernel
    supports_span_context = False

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections=None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str = "gaussian",
        raman_ase: bool = True,
        raman_temperature_k: float = 300.0,
        raman_peak_gain_coefficient_m_per_w: float = 6e-14,
        include_isrs: bool = True,
        fixed_point_iterations: int = 24,
        n_steps: int = DEFAULT_DRA_STEPS,
        n_quadrature: int = DEFAULT_DRA_NLI_QUADRATURE,
        n_spm_quadrature: int = 24,
        calibration: dict[str, CalibrationSet] | None = None,
        use_calibration: bool = True,
        modulation_format_correction: bool = False,
        **route_options: object,
    ) -> None:
        if modulation_format_correction:
            raise ValueError(
                "eta_XPM_corr is not printed in Zhang 2024 (Eq. (3) only cites Carena [12]); "
                "this model cannot implement it faithfully."
            )
        if raman_pumps is not None:
            self._validate_backward(raman_pumps)
        if n_spm_quadrature < 8:
            raise ValueError("n_spm_quadrature must be >= 8.")
        self._n_spm_quadrature = int(n_spm_quadrature)
        self._calibration = dict(default_calibration() if calibration is None else calibration)
        self._use_calibration = bool(use_calibration)
        self._span_records: list[tuple[ZhangSpanProfile, NDArrayFloat, float, float]] = []
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            raman_pumps=raman_pumps,
            raman_gain_model=raman_gain_model,
            raman_ase=raman_ase,
            raman_temperature_k=raman_temperature_k,
            raman_peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            include_isrs=include_isrs,
            fixed_point_iterations=fixed_point_iterations,
            n_steps=n_steps,
            n_quadrature=n_quadrature,
            **route_options,
        )

    @staticmethod
    def _validate_backward(pumps: Sequence[RamanPumpConfig]) -> None:
        if any(str(pump.direction) != "backward" for pump in pumps):
            raise ValueError("zhang2024_dra_egn supports backward Raman pumps only.")

    def _resolved_pumps(self, fiber_config: FiberSpanConfig) -> list[RamanPumpConfig]:
        pumps = super()._resolved_pumps(fiber_config)
        self._validate_backward(pumps)
        return pumps

    def span_step(
        self,
        fiber_config,
        edfa_config,
        power_in_w,
        span_length_m,
        span_context: SpanContext | None = None,
    ):
        """单跨：记录剖面与入纤功率，供链路层做相干求和。"""
        element = super().span_step(fiber_config, edfa_config, power_in_w, span_length_m, span_context)
        kernel = self.kernel_for(fiber_config)
        profile = getattr(kernel, "last_profile", None)
        if profile is not None:
            transfer = float(
                np.mean(np.divide(element.power_out_w, power_in_w, out=np.ones_like(power_in_w), where=power_in_w > 0))
            )
            self._span_records.append(
                (profile, np.array(power_in_w, dtype=np.float64, copy=True), float(span_length_m), transfer)
            )
        return element

    def evaluate_path(self, optical_path, power_in_w: NDArrayFloat) -> PathQoTResult:
        """整条链路的相干 NLI 一次算完，再替换路由层给出的功率口径结果。

        步骤：

        1. ``super().evaluate_path`` 只用来算功率、ASE 与 ROADM 语义；单跨核返回
           零 NLI（见 :class:`Zhang2024DRASpanKernel`）。
        2. 用 ``span_step`` 记录的（剖面、入纤功率、起点）构造
           :class:`ZhangCoherentSpan` 列表，一次性算 Eq. (4) 的相干 NLI。
        3. 同跨项用 :func:`single_span_eta_closed` 的闭式（CFM），跨间项用积分
           （EGN − 同跨）。两者相加再乘 Sec. II-C 的标定因子。
        """
        self._span_records = []
        result = super().evaluate_path(optical_path, power_in_w)
        if not self._span_records:
            return result

        kernel = self.kernel_for(optical_path.fiber_configs[0])
        order = kernel.order
        inverse = kernel.inverse_order
        bandwidths = self.channel_bandwidth_hz_channels[order]
        gamma = kernel.gamma_per_w_m
        beta2 = kernel.beta2
        beta3 = kernel.beta3
        reference_hz = kernel.reference_frequency_hz

        spans = []
        start = 0.0
        for profile, launch, length, _transfer in self._span_records:
            spans.append(ZhangCoherentSpan(profile, np.asarray(launch, dtype=np.float64)[order], start))
            start += length
        reference = spans[0].launch_power_w

        eta_spm_cfm, eta_xpm_cfm, eta_spm_all, eta_xpm_all = cfm_and_egn_terms(
            spans,
            reference,
            bandwidths,
            gamma,
            beta2,
            beta3,
            reference_hz,
            n_frequency=max(16, int(self._n_quadrature) * 2),
            n_spm_frequency=max(8, self._n_spm_quadrature),
        )
        del eta_spm_all, eta_xpm_all

        if self._use_calibration and self._calibration:
            params = {
                "launch_power_dbm": 10.0 * np.log10(max(float(np.mean(reference[reference > 0])), 1e-30) * 1e3),
                "spans": float(len(spans)),
            }
            term_spm = self._calibration.get("spm")
            term_xpm = self._calibration.get("xpm")
            if term_spm is not None:
                eta_spm_cfm = eta_spm_cfm * term_spm.apply(params)
            if term_xpm is not None:
                eta_xpm_cfm = eta_xpm_cfm * term_xpm.apply(params)

        spm = eta_spm_cfm * reference**3
        xpm = eta_xpm_cfm * reference**3
        spm_model = np.ascontiguousarray(spm[inverse])
        xpm_model = np.ascontiguousarray(xpm[inverse])

        signal = np.asarray(result.signal_power_w, dtype=np.float64)
        total_noise = spm_model + xpm_model + np.asarray(result.ase_noise_w, dtype=np.float64)
        snr = np.full_like(signal, -np.inf)
        active = (signal > 0) & (total_noise > 0)
        snr[active] = 10.0 * np.log10(signal[active] / total_noise[active])
        snr[(signal > 0) & (total_noise == 0)] = np.inf
        return PathQoTResult(
            signal_power_w=result.signal_power_w,
            spm_noise_w=spm_model,
            xpm_noise_w=xpm_model,
            ase_noise_w=result.ase_noise_w,
            snr_db=snr,
            span_count=result.span_count,
            roadm_count=result.roadm_count,
        )


register_model(ZHANG2024_DRA_MODEL_INFO, Zhang2024DRAModel)
