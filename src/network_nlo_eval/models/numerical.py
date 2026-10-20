"""Numerical ISRS-GN reference model.

This is the ground-truth model of the multi-model comparison: the numerical
counterpart of the closed-form GN models, implemented directly from

    C. Semrau, R. Killey, P. Bayvel,
    "The Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman
    Scattering", JLT 2018, doi:10.1109/jlt.2018.2830973

Two things are done numerically, where the closed-form models use algebraic
approximations:

* the inter-channel SRS power profile is obtained by integrating the coupled
  ordinary differential equations of Eq. (6) of the paper over the span, with a
  Raman gain spectrum that rolls off beyond the Stokes shift instead of the
  triangular approximation;
* the GN non-linear interference is obtained by evaluating the double integral
  of Eq. (4) of the paper with a composite quadrature over the frequency
  variables.

The frequency integral of Eq. (4) runs over the offsets of the two input waves
inside their own bands, ``f1 = f_i + x`` and ``f2 = f_l + y``; the generated
third wave then sits at ``f3 = f_l + x + y``, so the piecewise constant
third-wave PSD is split exactly at the band edges.  The remaining integral over
the intra-band offset uses a graded (cubic) grid that clusters the nodes
towards the band centre, where the phase mismatch of the self-interference
vanishes.  The squared transfer function of the span is the Cauchy-like kernel
of the phase mismatch; because the intra-band mismatch oscillates with a period
of ~156 MHz, far below the channel bandwidth, the kernel is evaluated with its
band average, which reproduces a brute-force evaluation of Eq. (4) to better
than 0.5 % and reproduces the closed-form GN limit exactly in the coherent
(zero dispersion) limit.

The model is deliberately slow rather than clever: its value is traceability,
not speed.
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, SpanFiberOutput, register_model
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
)
from network_nlo_eval.models.numerical_kernels import _gn_nli_jit, _solve_isrs_profile_jit
from network_nlo_eval.models.route import RouteModel
from network_nlo_eval.network.elements import FiberSpanConfig

__all__ = [
    "NUMERICAL_MODEL_INFO",
    "NumericalISRSGN",
    "NumericalISRSGNKernel",
]

#: Default number of distance steps of the power-profile integration.
#:
#: The coupled ODEs of Eq. (6) are integrated with a fixed-step RK4 scheme.
#: The Raman interaction rate over a 100 km span is of the order of
#: ``C_r * P_tot * B``, i.e. a few tens of e-foldings, so the step has to stay
#: well below the stability limit of RK4; 64 steps is already converged to
#: three digits (see the report).
DEFAULT_ODE_STEPS = 64

#: Default number of quadrature points per integration sub-interval of the
#: frequency integral of Eq. (4).
#:
#: The frequency integral runs over the offset inside the band of the channel of
#: interest.  Its integrand has a narrow coherent peak at the band centre, so
#: the quadrature uses a graded (cubic) grid whose cells cluster towards the
#: centre; 64 cells per half band put the converged value within 0.02 dB of the
#: result obtained with 128 cells (see the report).
DEFAULT_QUADRATURE_POINTS = 64

NUMERICAL_MODEL_INFO = ModelInfo(
    name="numerical",
    summary="Numerical ISRS-GN reference: ODE power profile plus double-integral NLI.",
    family="gn",
    reference="Semrau, Killey, Bayvel, JLT 2018, doi:10.1109/jlt.2018.2830973",
    options=(
        ("quadrature_points", "Frequency quadrature points per integration sub-interval."),
        ("ode_steps", "RK4 steps of the coupled ISRS power-profile integration."),
    ),
)


class NumericalISRSGNKernel:
    """Single-span kernel of the numerical ISRS-GN model.

    The kernel owns everything that only depends on the fibre and on the
    channel plan: the per-channel attenuation, effective area and dispersion,
    the ordering of the channels (the paper indexes the channels from the
    highest centre frequency) and the quadrature grids.

    Args:
        f_abs_hz: ``(N,)`` channel centre frequencies (Hz).
        b_hz: ``(N,)`` channel bandwidths (Hz).
        fiber_config: Fibre parameters of the span.
        quadrature_points: Frequency quadrature points per sub-interval.
        ode_steps: RK4 steps of the power-profile integration.
    """

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        ode_steps: int = DEFAULT_ODE_STEPS,
    ) -> None:
        """Prepare the per-channel constants of the span.

        Raises
        ------
            ValueError: The quadrature or integration resolution is not
                positive.
        """
        if quadrature_points < 1:
            raise ValueError(f"quadrature_points must be >= 1, got {quadrature_points}.")
        if ode_steps < 1:
            raise ValueError(f"ode_steps must be >= 1, got {ode_steps}.")

        self.quadrature_points = int(quadrature_points)
        self.ode_steps = int(ode_steps)
        self.n_nodes = int(f_abs_hz.shape[0])

        f_abs = np.asarray(f_abs_hz, dtype=np.float64)
        bandwidths = np.asarray(b_hz, dtype=np.float64)
        # The paper numbers the channels from the highest centre frequency, so
        # the internal order is descending; the model order is restored on the
        # way out.
        self.order = np.argsort(-f_abs, kind="stable")
        self.f_abs_hz = np.ascontiguousarray(f_abs[self.order])
        self.b_hz = np.ascontiguousarray(bandwidths[self.order])
        self.lambdas_m = 2.99792458e8 / self.f_abs_hz

        if fiber_config.use_attenuation_polynomial:
            self.alpha_npm = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            alpha = fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            self.alpha_npm = np.full(self.n_nodes, alpha, dtype=np.float64)

        self.a_eff_m2 = _calc_aeff_dynamic(
            self.lambdas_m,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        # Nonlinearity coefficient of every channel, 1/(W m).  The GN model of
        # Eq. (4) carries gamma^2 outside the frequency integral, so a single
        # coefficient per channel is enough.
        self.gamma_w_m = (2.0 * np.pi / self.lambdas_m) * (
            fiber_config.nonlinear_index_n2 / self.a_eff_m2
        )
        # The GN phase of Eq. (4) carries the frequency dependence through
        # beta3*(f1+f2); beta2 and beta3 are taken at the reference wavelength.
        lambda_ref_m = fiber_config.reference_wavelength_nm * 1e-9
        self.beta2 = float(_calc_beta2_from_d(fiber_config.dispersion_parameter_d, lambda_ref_m))
        self.beta3 = float(
            _calc_beta3_from_s_d(
                fiber_config.dispersion_slope_s,
                fiber_config.dispersion_parameter_d,
                lambda_ref_m,
            )
        )
        # The Taylor expansion of the propagation constant is centred on the
        # reference frequency; the third-order dispersion term of the GN phase
        # is expressed relative to it.
        self.f_ref_hz = 2.99792458e8 / lambda_ref_m
        self.n2_m2_w = float(fiber_config.nonlinear_index_n2)

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """Evaluate one span.

        Args:
            power_in_w: ``(N,)`` launch power per channel (W), in model order.
            span_length_m: Span length (m).
            n_effective_spans: Number of identical spans entering the
                phased-array term of Eq. (5) of the paper.

        Returns
        -------
            Output powers and the self- (SCI) and cross- (XCI) interference
            noise powers, all in model order.
        """
        power = np.ascontiguousarray(np.asarray(power_in_w, dtype=np.float64)[self.order])
        dz = span_length_m / self.ode_steps
        z_mid = (np.arange(self.ode_steps, dtype=np.float64) + 0.5) * dz

        profile, power_out = _solve_isrs_profile_jit(
            self.f_abs_hz,
            power,
            self.alpha_npm,
            span_length_m,
            self.ode_steps,
        )
        # Split the profile into the fiber attenuation and the SRS-only gain.
        srs_gain = profile * np.exp(self.alpha_npm[None, :] * z_mid[:, None])

        spm = np.zeros(self.n_nodes, dtype=np.float64)
        xpm = np.zeros(self.n_nodes, dtype=np.float64)
        _gn_nli_jit(
            self.f_abs_hz,
            self.b_hz,
            power,
            self.gamma_w_m,
            self.alpha_npm,
            srs_gain,
            self.ode_steps,
            dz,
            int(n_effective_spans),
            float(span_length_m),
            self.beta2,
            self.beta3,
            self.f_ref_hz,
            self.quadrature_points,
            spm,
            xpm,
        )

        inverse = np.empty(self.n_nodes, dtype=np.int64)
        inverse[self.order] = np.arange(self.n_nodes)
        return SpanFiberOutput(
            power_out_w=np.ascontiguousarray(power_out[inverse]),
            spm_noise_w=np.ascontiguousarray(spm[inverse]),
            xpm_noise_w=np.ascontiguousarray(xpm[inverse]),
        )


class NumericalISRSGN(RouteModel):
    """Numerical ISRS-GN model: ODE power profile plus double-integral NLI.

    Args:
        grid: Spectrum grid (legacy ``SpectrumGrid`` or ``FlexibleGrid``).
        ref_fiber_config: Reference fibre configuration.
        symbol_rate_hz: Symbol rate (Hz).
        egn: Not supported; the numerical EGN correction is out of scope.
        quadrature_points: Frequency quadrature points per sub-interval.
        ode_steps: RK4 steps of the power-profile integration.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.
    """

    info = NUMERICAL_MODEL_INFO

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        ode_steps: int = DEFAULT_ODE_STEPS,
        **route_options,
    ) -> None:
        """Initialize the model.

        Raises
        ------
            ValueError: ``egn=True`` was requested, or a resolution setting
                is not positive.
        """
        if egn:
            raise ValueError(
                "The numerical ISRS-GN model does not support egn=True: the numerical EGN "
                "correction is out of scope for this work package. Use egn=False."
            )
        if quadrature_points < 1:
            raise ValueError(f"quadrature_points must be >= 1, got {quadrature_points}.")
        if ode_steps < 1:
            raise ValueError(f"ode_steps must be >= 1, got {ode_steps}.")
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.egn = False
        self.quadrature_points = int(quadrature_points)
        self.ode_steps = int(ode_steps)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> NumericalISRSGNKernel:
        """Build the single-span kernel of this fibre."""
        return NumericalISRSGNKernel(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            quadrature_points=self.quadrature_points,
            ode_steps=self.ode_steps,
        )


register_model(NUMERICAL_MODEL_INFO, NumericalISRSGN)
