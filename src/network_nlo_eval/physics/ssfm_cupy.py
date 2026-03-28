"""基于 CuPy 的广义 Manakov 方程求解器 (SSFM)。

用于物理层高精度仿真，作为快速解析模型 (ISRS-GN) 的验证基准。
"""

from dataclasses import dataclass

import numpy as np
import tqdm

from network_nlo_eval.core.constants import C_LIGHT, PI
from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
from network_nlo_eval.network.elements import FiberSpanConfig

try:
    import cupy as cp

    _USE_GPU = True
except ImportError:
    cp = np  # Fallback to numpy if cupy not available
    _USE_GPU = False


@dataclass
class SSFMInputSignal:
    """SSFM 仿真输入信号的详细配置。"""

    num_channels: int
    channel_spacing_ghz: float
    center_frequency_thz: float
    baud_rate_gbps: float
    power_dbm_per_channel: float
    samples_per_symbol: int  # 每个符号的采样点数，影响 dt

    def __post_init__(self):
        """后处理输入配置。"""
        self.center_frequency_hz = self.center_frequency_thz * 1e12
        self.channel_spacing_hz = self.channel_spacing_ghz * 1e9
        self.baud_rate_hz = self.baud_rate_gbps * 1e9
        self.symbol_duration_s = 1.0 / self.baud_rate_hz
        self.dt = self.symbol_duration_s / self.samples_per_symbol
        self.fs = 1.0 / self.dt

        # 估算总采样点数 (需要足够长以包含所有信道)
        # 简化：假设总采样点数是 2^N 且足够覆盖带宽
        num_symbols = 2**10  # 比如 1024 个符号
        self.N_samples = int(num_symbols * self.samples_per_symbol)
        # 确保 N_samples 是 2 的幂次，以便 FFT 高效
        self.N_samples = 2 ** int(np.ceil(np.log2(self.N_samples)))

        self.time_window = self.N_samples * self.dt
        if (
            self.N_samples
            < self.num_channels * (self.channel_spacing_hz / self.baud_rate_hz) * self.samples_per_symbol * 2
        ):
            print(
                f"Warning: N_samples ({self.N_samples}) might be too small for {self.num_channels} channels. "
                f"Consider increasing `num_symbols` or `samples_per_symbol`."
            )


class SSFMDriver:
    """广义 Manakov 方程求解器 (Split-Step Fourier Method) Driver。

    用于高精度模拟光信号在光纤中的传播。
    支持 CuPy GPU 加速。
    """

    def __init__(
        self,
        input_signal_config: SSFMInputSignal,
        fiber_config: FiberSpanConfig,
        num_spans: int,
        span_length_m: float,
        dt_adaptive_min_m: float = 10.0,  # 最小自适应步长 (m)
        dt_adaptive_max_m: float = 1000.0,  # 最大自适应步长 (m)
        tolerance: float = 1e-3,  # 自适应步长误差容忍度
        pmd_coefficient_ps_sqrt_km: float = 0.05,  # PMD 系数 [ps/sqrt(km)]
        pmd_scattering_length_m: float = 2000.0,  # PMD 散射段长度 (m)
        use_gpu: bool = _USE_GPU,
    ):
        self.xp = cp if use_gpu else np
        self.input_signal_config = input_signal_config
        self.fiber_config = fiber_config
        self.num_spans = num_spans
        self.span_length_m = span_length_m
        self.total_length_m = num_spans * span_length_m

        self.dt_adaptive_min_m = dt_adaptive_min_m
        self.dt_adaptive_max_m = dt_adaptive_max_m
        self.tolerance = tolerance

        self.pmd_coefficient_ps_sqrt_km = pmd_coefficient_ps_sqrt_km
        self.pmd_scattering_length_m = pmd_scattering_length_m
        self.use_gpu = use_gpu

        self._precompute_operators()

    def _precompute_operators(self) -> None:
        """预计算用于 SSFM 的线性算子、非线性系数等。

        包括色散、衰减、拉曼响应、Manakov 因子等。
        """
        N = self.input_signal_config.N_samples
        dt = self.input_signal_config.dt

        self.time = self.xp.linspace(-N * dt / 2, N * dt / 2, N, endpoint=False)
        self.freq = self.xp.fft.fftfreq(N, d=dt)
        self.omega = 2 * PI * self.freq

        # 绝对频率和波长
        f_abs = self.input_signal_config.center_frequency_hz + self.freq
        lambdas_grid = C_LIGHT / f_abs

        # 动态衰减 (alpha)
        lambda_nm = lambdas_grid * 1e9
        alpha_dbkm = (
            self.fiber_config.attenuation_alpha2 * (lambda_nm - self.fiber_config.reference_wavelength_nm) ** 2
            + self.fiber_config.attenuation_alpha1 * (lambda_nm - self.fiber_config.reference_wavelength_nm)
            + self.fiber_config.attenuation_alpha0
        )
        self.alpha_power = alpha_dbkm / (10 * np.log10(np.exp(1)) * 1000)  # Convert dB/km to Np/m

        # 色散 (beta2, beta3)
        # D [ps/(nm*km)] -> beta2 [s^2/m] = -D * lambda^2 / (2*pi*c)
        # S [ps/(nm^2*km)] -> beta3 [s^3/m] = (lambda^2/(2*pi*c)^2) * (lambda^2*S + 2*lambda*D)
        # NOTE: 这里简化为在中心频率附近恒定，实际应随波长变化
        ref_lambda_m = self.input_signal_config.center_frequency_hz / C_LIGHT
        self.beta2_coeff = (
            -self.fiber_config.dispersion_parameter_d * 1e-12 * ref_lambda_m**2 / (2 * PI * C_LIGHT)
        )  # s^2/m
        self.beta3_coeff = (ref_lambda_m**2 / (2 * PI * C_LIGHT) ** 2) * (
            ref_lambda_m**2 * self.fiber_config.dispersion_slope_s * 1e-27
            + 2 * ref_lambda_m * self.fiber_config.dispersion_parameter_d * 1e-12
        )  # s^3/m

        self.linear_operator_D = (
            -(self.alpha_power / 2)
            + 1j * (self.beta2_coeff / 2) * (self.omega**2)
            - 1j * (self.beta3_coeff / 6) * (self.omega**3)
        )

        # 动态 A_eff 和 gamma (Manakov 因子)
        A_eff_array = (
            self.fiber_config.effective_area_um2_ref * 1e-12
            + self.fiber_config.aeff_slope_um2_nm
            * 1e-12
            / 1e-9
            * (lambdas_grid - self.fiber_config.reference_wavelength_nm * 1e-9)
        )
        manakov_factor = 8.0 / 9.0
        self.gamma_effective_array = (
            manakov_factor * (2 * PI * f_abs * self.fiber_config.nonlinear_index_n2) / (C_LIGHT * A_eff_array)
        )

        # 拉曼响应 (简化，来自用户原始代码)
        fr = 0.18  # Fraction of Raman response
        tau1 = 12.2e-15
        tau2 = 32.0e-15
        t_raman = self.xp.linspace(0, N * dt, N, endpoint=False)
        hr_t = (tau1**2 + tau2**2) / (tau1 * tau2**2) * self.xp.exp(-t_raman / tau2) * self.xp.sin(t_raman / tau1)
        hr_t[t_raman < 0] = 0
        self.hr_omega = self.xp.conj(self.xp.fft.fft(hr_t)) * dt  # 频域拉曼响应
        self.ONE_MINUS_FR = 1 - fr

        # PMD 算子
        delta_tau = self.pmd_coefficient_ps_sqrt_km * self.xp.sqrt(self.pmd_scattering_length_m / 1000.0) * 1e-12
        self.pmd_dgd_op_x = self.xp.exp(-1j * self.omega * (delta_tau / 2))
        self.pmd_dgd_op_y = self.xp.exp(1j * self.omega * (delta_tau / 2))

    def _manakov_nonlinear_term(self, A_omega_vec_current: NDArrayComplex) -> NDArrayComplex:
        """计算 Manakov 方程的非线性项。"""
        A_t_vec = self.xp.fft.ifft(A_omega_vec_current, axis=1)
        intensity_t = self.xp.abs(A_t_vec[0]) ** 2 + self.xp.abs(A_t_vec[1]) ** 2
        intensity_omega = self.xp.fft.fft(intensity_t)
        conv_t = self.xp.fft.ifft(intensity_omega * self.hr_omega)
        nonlinear_response_t = self.ONE_MINUS_FR * intensity_t + self.xp.real(
            conv_t
        )  # 拉曼响应可能是复数，但强度应为实数

        product_t = A_t_vec * nonlinear_response_t
        product_omega = self.xp.fft.fft(product_t, axis=1)
        term = 1j * self.gamma_effective_array * product_omega
        return term

    def _rk4ip_core_step(self, A_omega_in: NDArrayComplex, dz_step: float) -> NDArrayComplex:
        """RK4IP 单步 - GPU 优化版本"""
        h_op = self.xp.exp(self.linear_operator_D * (dz_step / 2.0))

        # RK4 步骤
        A_I = h_op * A_omega_in
        k1 = h_op * self._manakov_nonlinear_term(A_omega_in) * dz_step
        k2 = self._manakov_nonlinear_term(A_I + k1 / 2.0) * dz_step
        k3 = self._manakov_nonlinear_term(A_I + k2 / 2.0) * dz_step
        A_tmp = h_op * (A_I + k3)
        k4 = self._manakov_nonlinear_term(A_tmp) * dz_step

        # 组合结果 (注意 h_op 的位置)
        A_out = h_op * (A_I + k1 / 6.0 + k2 / 3.0) + k3 / 3.0 + k4 / 6.0  # Correction: h_op only applies to A_I once
        # A_out = h_op * (A_I + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0) # Original user code might be simpler
        return A_out

    def _generate_random_unitary_matrix(self) -> NDArrayComplex:
        """生成随机 2x2 幺正矩阵."""
        theta = self.xp.arcsin(self.xp.sqrt(self.xp.random.rand()))
        phi = 2 * PI * self.xp.random.rand()
        psi = 2 * PI * self.xp.random.rand()

        r00 = self.xp.cos(theta) * self.xp.exp(1j * phi)
        r01 = self.xp.sin(theta) * self.xp.exp(1j * psi)
        r10 = -self.xp.sin(theta) * self.xp.exp(-1j * psi)
        r11 = self.xp.cos(theta) * self.xp.exp(-1j * phi)

        return self.xp.array([[r00, r01], [r10, r11]], dtype=self.xp.complex128)

    def _apply_coarse_step_pmd(self, A_omega_vec_in: NDArrayComplex) -> NDArrayComplex:
        """应用粗步长 PMD (随机旋转 + DGD)."""
        R = self._generate_random_unitary_matrix()
        A_rot = self.xp.zeros_like(A_omega_vec_in)
        A_rot[0] = R[0, 0] * A_omega_vec_in[0] + R[0, 1] * A_omega_vec_in[1]
        A_rot[1] = R[1, 0] * A_omega_vec_in[0] + R[1, 1] * A_omega_vec_in[1]

        Ax_out = A_rot[0] * self.pmd_dgd_op_x
        Ay_out = A_rot[1] * self.pmd_dgd_op_y
        return self.xp.vstack([Ax_out, Ay_out])

    def run_simulation(self) -> tuple[NDArrayComplex, list[NDArrayFloat]]:
        """运行 SSFM 仿真，返回最终的信号场和沿途记录的功率谱。

        Returns
        -------
            Tuple[NDArrayComplex, List[NDArrayFloat]]:
            - 最终的信号场 (频域)
            - 沿途记录的功率谱历史 (CPU 上)
        """
        # 信号初始化 (这里可以放入更复杂的 WDM 信号生成逻辑)
        # 为简化，这里直接使用用户提供的 CW 填充随机相位的方法
        Ax_t = self.xp.zeros(self.input_signal_config.N_samples, dtype=self.xp.complex128)
        Ay_t = self.xp.zeros(self.input_signal_config.N_samples, dtype=self.xp.complex128)

        P_per_channel_W = 10 ** ((self.input_signal_config.power_dbm_per_channel - 30) / 10)

        for _i, f_ch in enumerate(self.input_signal_config.spectrum_grid.frequencies):
            freq_local = self.xp.fft.fftfreq(self.input_signal_config.N_samples, d=self.input_signal_config.dt)
            mask = self.xp.abs(freq_local - (f_ch - self.input_signal_config.center_frequency_hz)) < (
                self.input_signal_config.baud_rate_hz / 2
            )

            noise_spec = self.xp.exp(1j * self.xp.random.uniform(0, 2 * PI, self.input_signal_config.N_samples)) * mask
            ch_t = self.xp.fft.ifft(noise_spec)

            current_p = self.xp.mean(self.xp.abs(ch_t) ** 2)
            ch_t = ch_t * self.xp.sqrt(P_per_channel_W / (current_p + 1e-12))

            u1, u2 = self.xp.random.rand(2)
            theta = self.xp.arcsin(self.xp.sqrt(u1))
            phi = 2 * PI * u2
            Ax_t += ch_t * self.xp.cos(theta)
            Ay_t += ch_t * self.xp.exp(1j * phi) * self.xp.sin(theta)

        A_t_vec = self.xp.vstack([Ax_t, Ay_t])
        A_omega_vec = self.xp.fft.fft(A_t_vec, axis=1)

        history_power_spectrum: list[NDArrayFloat] = []  # 存储功率谱历史 (在 CPU 上)

        z = 0.0
        dz_adaptive = self.dt_adaptive_max_m

        # 记录初始状态
        if self.use_gpu:
            initial_psd = self.xp.asnumpy(self.xp.sum(self.xp.abs(A_omega_vec) ** 2, axis=0))
        else:
            initial_psd = self.xp.sum(self.xp.abs(A_omega_vec) ** 2, axis=0)
        history_power_spectrum.append(initial_psd)

        pmd_scatter_points = self.xp.arange(
            self.pmd_scattering_length_m,
            self.total_length_m + self.pmd_scattering_length_m / 100,
            self.pmd_scattering_length_m,
        )
        pmd_scatter_points = pmd_scatter_points[pmd_scatter_points <= self.total_length_m]
        if pmd_scatter_points.size > 0 and pmd_scatter_points[-1] < self.total_length_m:
            pmd_scatter_points = self.xp.append(pmd_scatter_points, self.total_length_m)

        current_pmd_target_idx = 0

        with tqdm(total=self.total_length_m, desc="SSFM Simulation") as pbar:
            while z < self.total_length_m:
                target_z_for_pmd = self.total_length_m
                if current_pmd_target_idx < len(pmd_scatter_points):
                    target_z_for_pmd = pmd_scatter_points[current_pmd_target_idx]

                while z < target_z_for_pmd - 1e-9:
                    current_dz = min(dz_adaptive, target_z_for_pmd - z)

                    # 自适应步长控制 (RK4IP)
                    A_single = self._rk4ip_core_step(A_omega_vec, current_dz)
                    A_mid = self._rk4ip_core_step(A_omega_vec, current_dz / 2.0)
                    A_double = self._rk4ip_core_step(A_mid, current_dz / 2.0)

                    error_norm = self.xp.linalg.norm(A_double - A_single) / self.xp.linalg.norm(A_double)

                    if error_norm <= self.tolerance or current_dz <= self.dt_adaptive_min_m:
                        z += current_dz
                        A_omega_vec = A_double
                        pbar.update(current_dz)  # 更新进度条

                        if error_norm < self.tolerance / 2.0:
                            dz_adaptive = min(dz_adaptive * 1.25, self.dt_adaptive_max_m)
                    else:
                        dz_adaptive = max(dz_adaptive * 0.5, self.dt_adaptive_min_m)

                # 抵达 PMD 散射点
                if z < self.total_length_m - 1e-9:
                    A_omega_vec = self._apply_coarse_step_pmd(A_omega_vec)
                current_pmd_target_idx += 1

                # 记录状态 (可以在每个 span 结束时记录，或更频繁)
                if self.use_gpu:
                    current_psd = self.xp.asnumpy(self.xp.sum(self.xp.abs(A_omega_vec) ** 2, axis=0))
                else:
                    current_psd = self.xp.sum(self.xp.abs(A_omega_vec) ** 2, axis=0)
                history_power_spectrum.append(current_psd)

        return A_omega_vec, history_power_spectrum
