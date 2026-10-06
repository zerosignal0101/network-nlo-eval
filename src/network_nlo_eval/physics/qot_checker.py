"""Topology-aware waveform QoT evaluation using ``physics-nlo-eval``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat, NodeID
from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
from network_nlo_eval.models.numba_kernels import _calc_beta2_from_d, _calc_beta3_from_s_d, _w_to_dbm
from network_nlo_eval.physics.metrics import compute_snr_from_evm

if TYPE_CHECKING:
    from network_nlo_eval.network.topology import NetworkTopology


def _load_physical_layer() -> dict[str, Any]:
    """Import the optional numerical simulator only when waveform QoT is requested."""
    try:
        from physics_nlo_eval.channel.fiber import EDFA, FiberChannel, IdealAmplifier
        from physics_nlo_eval.channel.network import ROADM, PumpSpec, RamanPump
        from physics_nlo_eval.channel.wdm import ChannelExtractor, ChannelSpec, WDMDemux, WDMMux, WDMPlan
        from physics_nlo_eval.dsp.carrier import PhaseNoiseCompensation
        from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp, TimingRecovery
        from physics_nlo_eval.metrics.ber_evm import BERCalculator, Demapper
        from physics_nlo_eval.tx.mapper import Mapper
        from physics_nlo_eval.tx.optics import IQModulator
        from physics_nlo_eval.tx.rrc import RRCFilter
    except ImportError as exc:
        raise ImportError(
            "Waveform comparison requires physics-nlo-eval installed in the same Python environment."
        ) from exc
    # 显式注册表：键即波形引擎与本工程之间的契约，避免依赖 locals() 的隐式查找。
    return {
        "BERCalculator": BERCalculator,
        "ChannelExtractor": ChannelExtractor,
        "ChannelSpec": ChannelSpec,
        "ChromaticDispersionComp": ChromaticDispersionComp,
        "Demapper": Demapper,
        "EDFA": EDFA,
        "FiberChannel": FiberChannel,
        "IQModulator": IQModulator,
        "IdealAmplifier": IdealAmplifier,
        "Mapper": Mapper,
        "PhaseNoiseCompensation": PhaseNoiseCompensation,
        "PumpSpec": PumpSpec,
        "ROADM": ROADM,
        "RRCFilter": RRCFilter,
        "RamanPump": RamanPump,
        "TimingRecovery": TimingRecovery,
        "WDMDemux": WDMDemux,
        "WDMMux": WDMMux,
        "WDMPlan": WDMPlan,
    }


class PyNLOQoTEvaluator:
    """Numerically evaluate the same ordered route used by the statistical model.

    The historical name is retained for API compatibility. The physical
    package runs the Manakov solver through PyNLO, which requires a CuPy
    compute context for dual-polarization WDM propagation.
    """

    def __init__(
        self,
        spectrum_grid: SpectrumGrid,
        modulation: str = "DP-QPSK",
        symbol_rate_hz: float = 50e9,
        sps: int = 4,
        rrc_roll_off: float = 0.1,
        seed: int = 42,
        compute_ctx_name: str = "cupy",
        pilot_info: dict[str, Any] | None = None,
    ) -> None:
        """初始化拓扑波形 QoT 评估器。

        ``compute_ctx_name`` 默认为 ``"cupy"``：Manakov 求解器在 MKL FFT
        后端上会被 PyNLO 直接拒绝，CPU 回退对本评估器不可用。
        """
        if sps <= 0 or symbol_rate_hz <= 0:
            raise ValueError("Symbol rate and samples per symbol must be positive.")
        self.physical = _load_physical_layer()
        self.spectrum_grid = spectrum_grid
        self.modulation = modulation
        self.symbol_rate_hz = symbol_rate_hz
        self.sps = sps
        self.rrc_roll_off = rrc_roll_off
        self.seed = seed
        self.compute_ctx_name = compute_ctx_name
        self.pilot_info = pilot_info

        self.mapper = self.physical["Mapper"](
            modulation=modulation, power_norm=True, pilot_info=pilot_info, sym_rate=symbol_rate_hz
        )
        self.tx_rrc = self.physical["RRCFilter"](sps=sps, roll_off=rrc_roll_off, is_matched_filter=False)
        self.rx_rrc = self.physical["RRCFilter"](sps=sps, roll_off=rrc_roll_off, is_matched_filter=True)
        self.demapper = self.physical["Demapper"](modulation=modulation, power_norm=True)
        self.frequencies_hz = spectrum_grid.frequencies
        self.center_freq_hz = spectrum_grid.center_frequency_hz
        self.channels_spec = [
            self.physical["ChannelSpec"](index, frequency - self.center_freq_hz)
            for index, frequency in enumerate(self.frequencies_hz)
        ]
        self.wdm_plan = self.physical["WDMPlan"](self.center_freq_hz, self.channels_spec)
        occupied_width = (spectrum_grid.num_channels - 1) * spectrum_grid.channel_spacing_hz
        self._mux_target_sr = max(symbol_rate_hz * sps, 1.25 * (occupied_width + symbol_rate_hz * (1 + rrc_roll_off)))

    def create_wdm_signal(
        self, launch_power_profile_w: NDArrayFloat, num_symbols: int = 1024
    ) -> tuple[Any, list[NDArrayBool]]:
        """创建与统计模型功率参考面一致的双偏振 WDM 波形."""
        powers = np.asarray(launch_power_profile_w, dtype=float)
        if powers.shape != (self.spectrum_grid.num_channels,) or np.any(powers < 0):
            raise ValueError("Launch power must contain one non-negative value per channel.")
        rng = np.random.default_rng(self.seed)
        channel_signals: list[Any] = []
        all_bits: list[NDArrayBool] = []
        for channel_idx in range(self.spectrum_grid.num_channels):
            total_bits = num_symbols * self.mapper.bits_per_symbol * self.mapper.n_pol
            bits = rng.integers(0, 2, size=total_bits, dtype=np.int8).astype(bool)
            all_bits.append(bits)
            electrical = self.tx_rrc.forward(self.mapper.forward(bits))
            # Network launch power is total DP power; IQModulator expects per-pol power.
            per_pol_power = powers[channel_idx] / self.mapper.n_pol
            optical = self.physical["IQModulator"](
                launch_power_dbm_per_pol=_w_to_dbm(per_pol_power),
                wavelength_nm=C_LIGHT / self.frequencies_hz[channel_idx] * 1e9,
            ).forward(electrical)
            channel_signals.append(optical)
        mux = self.physical["WDMMux"](self.wdm_plan, target_sample_rate_hz=self._mux_target_sr)
        return mux.combine(channel_signals), all_bits

    @staticmethod
    def _fiber_dispersion(fiber: Any, center_frequency_hz: float) -> tuple[float, float]:
        """将网络 D/S 参数转换为物理求解器的 beta2/beta3 单位."""
        wavelength_m = C_LIGHT / center_frequency_hz
        beta2 = _calc_beta2_from_d(fiber.dispersion_parameter_d, wavelength_m) * 1e27
        beta3 = _calc_beta3_from_s_d(fiber.dispersion_slope_s, fiber.dispersion_parameter_d, wavelength_m) * 1e39
        return float(beta2), float(beta3)

    @staticmethod
    def _edfa_values(edfa: Any) -> tuple[float | None, float]:
        """返回物理 EDFA 支持的标量增益/NF，拒绝静默降级."""
        noise_figure = np.asarray(edfa.noise_figure_db)
        if noise_figure.ndim != 0 or edfa.gain_ripple_db is not None:
            raise ValueError(
                "Waveform comparison currently requires scalar EDFA noise figure and no gain-ripple array."
            )
        return edfa.target_gain_db, float(noise_figure)

    def propagate_path(self, wdm_signal: Any, optical_path: MultiSpanOpticalPath) -> Any:
        """按路径顺序应用物理光纤、放大器、拉曼泵和 ROADM."""
        current = wdm_signal
        roadms = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
        channel_info = current.ctx.meta["wdm_channels_info"]
        span_sequence = 0
        active_channels = np.flatnonzero(
            [entry.get("original_ctx").meta.get("tx_power_per_pol_w", 0) > 0 for entry in channel_info]
        ).tolist()
        for hop_index, (fiber, edfa, roadm) in enumerate(
            zip(optical_path.fiber_configs, optical_path.edfa_configs, roadms, strict=True)
        ):
            if fiber.use_attenuation_polynomial:
                raise ValueError(
                    "Waveform comparison currently requires constant attenuation_db_km_ref per fiber link."
                )
            beta2, beta3 = self._fiber_dispersion(fiber, current.ctx.center_freq)
            target_gain_db, noise_figure_db = self._edfa_values(edfa)
            remaining_km = fiber.length_km
            while remaining_km > 1e-12:
                span_km = min(remaining_km, fiber.max_span_length_km)
                channel = self.physical["FiberChannel"](
                    length_m=span_km * 1e3,
                    alpha_db_per_km=fiber.attenuation_db_km_ref,
                    beta2_ps2_per_km=beta2,
                    beta3_ps3_per_km=beta3,
                    coarse_step_size_m=min(2000.0, span_km * 1e3),
                    n2_si=fiber.nonlinear_index_n2,
                    a_eff_um2=fiber.effective_area_um2_ref,
                    slope_a_eff_um2_per_nm=fiber.aeff_slope_um2_nm,
                    ref_lambda_nm=fiber.reference_wavelength_nm,
                    compute_ctx_name=self.compute_ctx_name,
                )
                current = channel.forward(current)
                if fiber.raman_pumps:
                    pumps = [self.physical["PumpSpec"](**pump.model_dump()) for pump in fiber.raman_pumps]
                    current = self.physical["RamanPump"](pumps, span_km, seed=self.seed + span_sequence).forward(
                        current
                    )
                if target_gain_db is None:
                    current = self.physical["IdealAmplifier"](nf_db=noise_figure_db, default_power_dbm=0.0).forward(
                        current
                    )
                else:
                    current = self.physical["EDFA"](
                        gain_db=target_gain_db,
                        nf_db=noise_figure_db,
                        seed=self.seed + span_sequence,
                    ).forward(current)
                remaining_km -= span_km
                span_sequence += 1

            if roadm is None:
                continue
            allowed = [index for index in active_channels if index not in roadm.blocked_channels]
            current = self.physical["ROADM"](
                pass_channels=allowed,
                passband_hz=roadm.passband_hz or self.spectrum_grid.channel_spacing_hz,
                insertion_loss_db=roadm.path_loss_db(
                    "drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"
                ),
                filter_order=roadm.filter_order,
            ).forward(current)
            if roadm.equalize_output_power:
                current = self.physical["EDFA"](
                    gain_db=roadm.path_loss_db(
                        "drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"
                    ),
                    nf_db=roadm.booster_noise_figure_db,
                    seed=self.seed + hop_index + 1,
                ).forward(current)
        return current

    def _receiver_metrics(
        self,
        rx_wdm: Any,
        tx_bits: list[NDArrayBool],
        optical_path: MultiSpanOpticalPath,
        active_channels: NDArrayBool,
    ) -> dict[str, NDArrayFloat]:
        """补偿路径色散并计算每个活动信道的 BER/EVM/SNR."""
        compensated = rx_wdm
        for fiber in optical_path.fiber_configs:
            beta2, beta3 = self._fiber_dispersion(fiber, compensated.ctx.center_freq)
            if beta2 == 0 and beta3 == 0:
                continue
            compensated = self.physical["ChromaticDispersionComp"](
                fiber_length_km=fiber.length_km,
                beta2_ps2_per_km=beta2,
                beta3_ps3_per_km=beta3,
            ).forward(compensated)
        demuxed = self.physical["WDMDemux"](filter_type="ideal_rectangular").forward(compensated)
        ber = np.full(self.spectrum_grid.num_channels, np.nan)
        evm = np.full(self.spectrum_grid.num_channels, np.nan)
        snr = np.full(self.spectrum_grid.num_channels, np.nan)
        calculator = self.physical["BERCalculator"]()
        for index in range(self.spectrum_grid.num_channels):
            if not active_channels[index]:
                continue
            channel = self.physical["ChannelExtractor"](channel_idx=index).forward(demuxed)
            channel = self.rx_rrc.forward(channel)
            channel = self.physical["TimingRecovery"](sps_in=channel.ctx.sps).forward(channel)
            method = "pilot_assisted" if self.pilot_info is not None else "BPS"
            kwargs = {"method": method, "constellation": self.demapper.constellation}
            if method == "BPS":
                kwargs.update({"n_test_phases": 256, "block_size": 256})
            channel = self.physical["PhaseNoiseCompensation"](**kwargs).forward(channel)
            received_bits = self.demapper.forward(channel)
            evm[index] = calculator.calculate_evm(channel, self.demapper.constellation, tx_bits[index]) * 0.01
            ber[index] = calculator.calculate_ber(received_bits, tx_bits[index])
            snr[index] = compute_snr_from_evm(evm[index])
        return {"ber": ber, "evm": evm, "snr": snr}

    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        launch_power_profile_w: NDArrayFloat,
        num_symbols: int = 1024,
    ) -> tuple[Any, dict[str, NDArrayFloat]]:
        """生成波形并沿给定拓扑路径完成端到端 QoT 评估."""
        signal, bits = self.create_wdm_signal(launch_power_profile_w, num_symbols=num_symbols)
        received = self.propagate_path(signal, optical_path)
        active = np.asarray(launch_power_profile_w) > 0
        for roadm in optical_path.roadm_configs or []:
            active[[index for index in roadm.blocked_channels if index < len(active)]] = False
        return received, self._receiver_metrics(received, bits, optical_path, active)

    def evaluate_topology_path(
        self,
        topology: NetworkTopology,
        path: list[NodeID],
        launch_power_profile_w: NDArrayFloat,
        num_symbols: int = 1024,
    ) -> tuple[Any, dict[str, NDArrayFloat]]:
        """从拓扑节点路径构建共享物理路径并评估."""
        return self.evaluate_path(topology.build_optical_path(path), launch_power_profile_w, num_symbols)

    def evaluate_channel_evm_and_ber(
        self,
        launch_power_profile_w: NDArrayFloat,
        fiber_length_km: float = 100.0,
        num_spans: int = 1,
        amplifier_nf_db: float = 5.5,
    ) -> tuple[Any, dict[str, NDArrayFloat]]:
        """兼容旧 API 的均匀单链路评估入口."""
        from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

        path = MultiSpanOpticalPath(
            fiber_configs=[FiberSpanConfig(length_km=fiber_length_km) for _ in range(num_spans)],
            edfa_configs=[EDFAConfig(noise_figure_db=amplifier_nf_db) for _ in range(num_spans)],
        )
        return self.evaluate_path(path, launch_power_profile_w)


WaveformQoTEvaluator = PyNLOQoTEvaluator
