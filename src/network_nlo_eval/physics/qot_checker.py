"""传输质量 (Quality of Transmission, QoT) 评估器 - 基于 PyNLO 物理层仿真。

使用 PyNLO 的 Manakov 方程求解器进行准确的波传播仿真，
计算链路的 EVM/BER。
"""

import numpy as np
from physics_nlo_eval.channel.fiber import FiberChannel
from physics_nlo_eval.channel.wdm import ChannelExtractor, ChannelSpec, WDMDemux, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.dsp.carrier import PhaseNoiseCompensation
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp, TimingRecovery
from physics_nlo_eval.metrics.ber_evm import BERCalculator, Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter
from pynlo.utility import chi3

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat
from network_nlo_eval.models.numba_kernels import _w_to_dbm
from network_nlo_eval.physics.metrics import compute_snr_from_evm


def calculate_grid_dt(n_points: int, bandwidth_hz: float) -> float:
    """计算 PyNLO Grid 的时间步长 dt。

    PyNLO Grid.create uses:
        dv = bandwidth / (n - 1)
        dt = 1.0 / (n * dv)
    """
    dv = bandwidth_hz / (n_points - 1)
    dt = 1.0 / (n_points * dv)
    return dt


class PyNLOQoTEvaluator:
    """基于 PyNLO 物理层仿真的 QoT 评估器。

    使用 Manakov 方程求解器进行准确的波传播仿真，计算链路的 EVM/BER。
    """

    def __init__(
        self,
        spectrum_grid: SpectrumGrid,
        modulation: str = "DP-QPSK",
        symbol_rate_hz: float = 50e9,
        sps: int = 4,
        rrc_roll_off: float = 0.1,
        seed: int = 42,
        compute_ctx_name: str = "numpy",
    ):
        """初始化 QoT 验证器。

        Parameters
        ----------
        spectrum_grid : SpectrumGrid
            频谱网格配置。
        modulation : str
            调制格式。
        symbol_rate_hz : float
            符号速率 (Hz)。
        sps : int
            每符号采样点数。
        rrc_roll_off : float
            RRC 滤波器滚降系数。
        seed : int
            随机种子。
        compute_ctx_name : str
            计算上下文名称 ('numpy' 或 'cupy')。
        """
        self.spectrum_grid = spectrum_grid
        self.modulation = modulation
        self.sps = sps
        self.rrc_roll_off = rrc_roll_off
        self.seed = seed
        self.compute_ctx_name = compute_ctx_name

        # TX components
        self.mapper = Mapper(modulation=modulation, power_norm=True, sym_rate=symbol_rate_hz)
        self.tx_rrc = RRCFilter(sps=sps, roll_off=rrc_roll_off, is_matched_filter=False)
        self.rx_rrc = RRCFilter(sps=sps, roll_off=rrc_roll_off, is_matched_filter=True)
        self.demapper = Demapper(modulation=modulation, power_norm=True)

        # Channel Spec
        self.frequencies_hz = self.spectrum_grid.frequencies
        self.center_freq_hz = self.spectrum_grid.center_frequency_hz
        self.channels_spec: list[ChannelSpec] = []
        for channel_idx in range(self.spectrum_grid.num_channels):
            self.channels_spec.append(ChannelSpec(channel_idx, self.frequencies_hz[channel_idx] - self.center_freq_hz))

        # WDM Plan
        self.wdm_plan = WDMPlan(self.center_freq_hz, self.channels_spec)

    def create_wdm_signal(
        self,
        launch_power_profile_w: NDArrayFloat,
        num_symbols: int = 768,
    ) -> tuple[Signal, list[NDArrayBool]]:
        """创建 WDM 信号。

        Parameters
        ----------
        launch_power_profile_w : NDArrayFloat
            每个通道的入纤功率 (w/pol)。
        num_symbols : int
            每个通道的符号数量。

        Returns
        -------
        tuple[Signal, list[NDArrayBool]]
            (合并的 WDM 信号, 各通道原始比特列表)。
        """
        np.random.seed(self.seed)
        channels_signal: list[Signal] = []
        all_bits = []

        for ch_spec in self.channels_spec:
            bits_per_symbol = self.mapper.bits_per_symbol
            n_pol = self.mapper.n_pol
            total_bits = num_symbols * bits_per_symbol * n_pol

            tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
            all_bits.append(tx_bits)

            sig = self.mapper.forward(tx_bits)
            sig_rrc = self.tx_rrc.forward(sig)

            iq_mod = IQModulator(launch_power_dbm_per_pol=_w_to_dbm(launch_power_profile_w[ch_spec.channel_id]))
            sig_optical = iq_mod.forward(sig_rrc)

            channels_signal.append(sig_optical)

        # WDM mux
        mux = WDMMux(wdm_plan=self.wdm_plan, target_sample_rate_hz=128.0 * 1e12)
        wdm_signal = mux.combine(channels_signal)

        return wdm_signal, all_bits

    def run_through_fiber(
        self,
        wdm_signal: Signal,
        fiber_length_km: float,
    ) -> Signal:
        """通过光纤传播。"""
        beta2_ps2_per_km: float = -20.0
        beta3_ps3_per_km: float = 0.1e-3
        n2_si: float = 2.6e-20
        r_w = [0.18, 12.2e-15, 32e-15]
        n_points = wdm_signal.num_samples
        bandwidth = wdm_signal.ctx.sample_rate
        dt = calculate_grid_dt(n_points, bandwidth)
        _rv_grid, raman = chi3.raman(n_points, dt, r_weights=r_w)
        fiber = FiberChannel(
            length_m=fiber_length_km * 1000,
            alpha_db_per_km=0.2,
            beta2_ps2_per_km=beta2_ps2_per_km,
            beta3_ps3_per_km=beta3_ps3_per_km,
            coarse_step_size_m=2000.0,
            n2_si=n2_si,
            a_eff_um2=80.0,
            slope_a_eff_um2_per_nm=0.05,
            ref_lambda_nm=1550.0,
            d_pmd_ps_per_sqrt_km=0.0,
            raman_r3=raman,
            compute_ctx_name=self.compute_ctx_name,
        )
        return fiber.forward(wdm_signal)

    def evaluate_channel_evm_and_ber(
        self,
        launch_power_profile_w: NDArrayFloat,
        fiber_length_km: float = 100.0,
    ) -> tuple[Signal, dict[str, NDArrayFloat] | None]:
        """评估单个通道的 EVM 和 BER。"""
        wdm_signal, tx_bits = self.create_wdm_signal(
            launch_power_profile_w=launch_power_profile_w,
        )

        rx_wdm = self.run_through_fiber(wdm_signal, fiber_length_km)

        # CD compensation
        cdc = ChromaticDispersionComp(
            fiber_length_km=fiber_length_km,
            beta2_ps2_per_km=-20.0,
            beta3_ps3_per_km=0.1e-3,
        )
        rx_cdc = cdc.forward(rx_wdm)

        # Demux
        demux = WDMDemux(filter_type="fir")
        demuxed = demux.forward(rx_cdc)

        timing_recovery = TimingRecovery(sps_in=demuxed.ctx.sps)

        # BPS 补偿
        ideal_constellation = self.demapper.constellation
        bps = PhaseNoiseCompensation(
            method="BPS",
            n_test_phases=256,
            block_size=256,
            constellation=ideal_constellation,
        )

        ber_calc = BERCalculator()
        ber_list = []
        evm_list = []
        snr_list = []

        for i, _ch_spec in enumerate(self.wdm_plan.channels):
            extractor = ChannelExtractor(channel_idx=i)
            rx_ch = extractor.forward(demuxed)
            rx_filt = self.rx_rrc.forward(rx_ch)
            rx_sync = timing_recovery.forward(rx_filt)

            rx_comp = bps.forward(rx_sync)

            rx_bits = self.demapper.forward(rx_comp)

            evm_percentage = ber_calc.calculate_evm(rx_comp, ideal_constellation, tx_bits[i])
            evm_val = evm_percentage * 0.01

            ber = ber_calc.calculate_ber(rx_bits, tx_bits[i])
            ber_list.append(ber)
            evm_list.append(evm_val)
            snr_list.append(compute_snr_from_evm(evm_val))

        ber_array = np.asarray(ber_list)
        evm_array = np.asarray(evm_list)
        snr_array = np.asarray(snr_list)

        return rx_wdm, dict(ber=ber_array, evm=evm_array, snr=snr_array)
