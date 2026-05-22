"""传输质量 (Quality of Transmission, QoT) 评估器 - 基于 PyNLO 物理层仿真。

使用 PyNLO 的 Manakov 方程求解器进行准确的波传播仿真，
计算链路的 EVM/BER。
"""


import numpy as np
from physics_nlo_eval.channel.fiber import FiberChannel, IdealAmplifier
from physics_nlo_eval.channel.wdm import ChannelExtractor, ChannelSpec, WDMDemux, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal, SignalContext
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
        pilot_info: dict | None = None,
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
        pilot_info : dict | None
            导频配置，例如 {"symbols": np.array([1+1j])/np.sqrt(2), "period": 32}。
            启用后接收端自动使用 pilot_aided 相位恢复替代 BPS。
        """
        self.spectrum_grid = spectrum_grid
        self.modulation = modulation
        self.sps = sps
        self.rrc_roll_off = rrc_roll_off
        self.seed = seed
        self.compute_ctx_name = compute_ctx_name
        self.pilot_info = pilot_info

        # TX components
        self.symbol_rate_hz = symbol_rate_hz
        self._mux_target_sr = 128.0 * 1e12  # WDM 合波目标采样率
        self.mapper = Mapper(modulation=modulation, power_norm=True,
                             pilot_info=pilot_info, sym_rate=symbol_rate_hz)
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
        num_symbols: int = 1024,
    ) -> tuple[Signal, list[NDArrayBool]]:
        """创建 WDM 信号。"""
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

        mux = WDMMux(wdm_plan=self.wdm_plan, target_sample_rate_hz=self._mux_target_sr)
        wdm_signal = mux.combine(channels_signal)

        return wdm_signal, all_bits

    def run_through_fiber(
        self,
        wdm_signal: Signal,
        fiber_length_km: float,
        num_spans: int = 1,
        amplifier_nf_db: float = 5.5,
    ) -> Signal:
        """通过光纤传播，支持多跨段。

        Parameters
        ----------
        wdm_signal : Signal
            输入的 WDM 信号。
        fiber_length_km : float
            每个跨段的光纤长度 (km)。
        num_spans : int
            跨段数量。默认为 1 (单跨段)。
        amplifier_nf_db : float
            放大器噪声系数 (dB)。默认为 5.5 dB。

        Returns
        -------
        Signal
            经过多跨段传播后的输出信号。
        """
        beta2_ps2_per_km: float = -20.0
        beta3_ps3_per_km: float = 0.1e-3
        n2_si: float = 2.6e-20
        alpha_db_per_km: float = 0.2
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

        # 初始信号
        current_signal = wdm_signal

        amplifier = IdealAmplifier(nf_db=amplifier_nf_db)

        for _span_idx in range(num_spans):
            # 1. 光纤传播
            current_signal = fiber.forward(current_signal)

            # 2. 频域逐信道放大 (IdealAmplifier 自动检测宽带 WDM 信号,
            #    使用 wdm_channels_info 元数据进行频域增益, 避免 demux/remux 相位破坏)
            current_signal = amplifier.forward(current_signal)

        return current_signal

    def evaluate_channel_evm_and_ber(
        self,
        launch_power_profile_w: NDArrayFloat,
        fiber_length_km: float = 100.0,
        num_spans: int = 1,
        amplifier_nf_db: float = 5.5,
    ) -> tuple[Signal, dict[str, NDArrayFloat] | None]:
        """评估单个通道的 EVM 和 BER。"""
        wdm_signal, tx_bits = self.create_wdm_signal(
            launch_power_profile_w=launch_power_profile_w,
        )

        rx_wdm = self.run_through_fiber(wdm_signal, fiber_length_km, num_spans, amplifier_nf_db)

        # 宽带 CD 补偿 (先补偿再分波，保证所有通道的载波偏移相位和群时延正确)
        total_length_km = fiber_length_km * num_spans
        cdc = ChromaticDispersionComp(
            fiber_length_km=total_length_km,
            beta2_ps2_per_km=-20.0,
            beta3_ps3_per_km=0.1e-3,
        )
        rx_cdc = cdc.forward(rx_wdm)

        # 分波 (使用 ideal_rectangular 避免 FIR 群时延)
        demux = WDMDemux(filter_type="ideal_rectangular")
        demuxed = demux.forward(rx_cdc)

        timing_recovery = TimingRecovery(sps_in=demuxed.ctx.sps)

        ber_calc = BERCalculator()
        ber_list = []
        evm_list = []
        snr_list = []

        for i, _ch_spec in enumerate(self.wdm_plan.channels):
            extractor = ChannelExtractor(channel_idx=i)
            rx_ch = extractor.forward(demuxed)
            rx_filt = self.rx_rrc.forward(rx_ch)

            # TimingRecovery + 相位恢复 (BPS 或 pilot_aided)
            rx_sync = timing_recovery.forward(rx_filt)

            if self.pilot_info is not None:
                phase_comp = PhaseNoiseCompensation(
                    method="pilot_assisted", constellation=self.demapper.constellation,
                )
            else:
                phase_comp = PhaseNoiseCompensation(
                    method="BPS", n_test_phases=256,
                    block_size=256, constellation=self.demapper.constellation,
                )
            rx_comp = phase_comp.forward(rx_sync)

            rx_bits = self.demapper.forward(rx_comp)

            evm_percentage = ber_calc.calculate_evm(rx_comp, self.demapper.constellation, tx_bits[i])
            evm_val = evm_percentage * 0.01

            ber = ber_calc.calculate_ber(rx_bits, tx_bits[i])
            ber_list.append(ber)
            evm_list.append(evm_val)
            snr_list.append(compute_snr_from_evm(evm_val))

        ber_array = np.asarray(ber_list)
        evm_array = np.asarray(evm_list)
        snr_array = np.asarray(snr_list)

        return rx_wdm, dict(ber=ber_array, evm=evm_array, snr=snr_array)
