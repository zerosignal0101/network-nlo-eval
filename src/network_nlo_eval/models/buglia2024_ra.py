"""Buglia 2024 RA-GN paper control (``buglia2024_ra_gn``).

Reference
---------
H. Buglia, M. Jarmolovičius, L. Galdino, R. I. Killey, P. Bayvel,
"A Closed-Form Expression for the Gaussian Noise Model in the Presence of
Raman Amplification", J. Lightwave Technol. 42(2), 636-648, 2024,
doi:10.1109/jlt.2023.3315127.

What is implemented
-------------------
* **Eq. (1)**: the coupled signal/pump power equations.  They are reused from
  :class:`~network_nlo_eval.models.hybrid_raman.RamanPaperSpanSolver`, which
  integrates any number of forward/backward pumps with pump depletion,
  pump-pump and channel-channel ISRS.  Frequency-dependent fibre attenuation,
  effective area, dispersion and per-pump attenuation are taken from the
  fibre and pump configuration.
* **Eq. (2)**: the five-coefficient semi-analytical normalised profile
  ``rho(z, f)``.  The coefficients are fitted to the Eq. (1) numerical
  solution by nonlinear least squares, exactly the workflow of the paper
  (Sec. II-A).  ``C_f P_f`` and ``C_b P_b`` are fitted per channel; the two
  effective pump attenuations are fixed to the power-weighted pump
  attenuation because the three-exponential form is otherwise degenerate.
* **Eq. (3)-(8)**: the NLI coefficient and the SPM/XPM link functions.  The
  longitudinal integral of Eq. (8) is analytic for the fitted
  three-exponential profile; the in-band frequency integrals of Eq. (6) are
  evaluated by a quadrature that resolves the narrow XPM coherence peak.
* **Eq. (4)**: per-span NLI accumulation.  The route layer sums the spans
  incoherently, which is Eq. (4) with ``epsilon = 1``; the paper's ``n**epsilon``
  exponent is not specified in the extracted text.

What is not reproduced
----------------------
* The printed closed forms of Eq. (9), (11) and (12) are **not** copied term
  by term.  The extracted equations are partly garbled, and their stated
  accuracy relies on five fitted coefficients that the paper does not
  publish.  This module evaluates the exact GN double integral that those
  closed forms approximate, using the fitted Eq. (2) profile.  The SPM part
  is about 0.5 dB below the repository's PCFM reference on the C-band
  benchmark; the XPM part matches it.
* The Raman gain spectrum is the repository Gaussian/CZ approximation, not
  the measured silica spectrum of the paper's Fig. 1.
* The paper models no ASE; this control still uses the repository EDFA and
  the hybrid solver's Raman ASE, so the reported SNR includes noise that the
  paper's ``SNR_NLI`` does not.
* Pump-induced XPM is not included (the paper also neglects it, but notes it
  can be added as extra interferers).

The model is therefore labelled ``paper-inspired / reproduced subset`` and
must not be cited as a full reproduction of Buglia 2024.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, register_model
from network_nlo_eval.models.dra_common import (
    DEFAULT_DRA_NLI_QUADRATURE,
    DEFAULT_DRA_STEPS,
    BugliaProfile,
    DRARamanModelBase,
    DRASpanKernelBase,
    fit_buglia_profile,
)
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "BUGLIA2024_RA_MODEL_INFO",
    "Buglia2024RAModel",
    "Buglia2024RASpanKernel",
]


class Buglia2024RASpanKernel(DRASpanKernelBase):
    """Single-span kernel of the Buglia 2024 RA-GN control."""

    profile_kind = "buglia_eq2"

    def _fit_profile(self, profile_norm: NDArrayFloat, span_length_m: float) -> BugliaProfile:
        """Fit Eq. (2) to the coupled-ODE profile."""
        pumps = list(self._pumps)
        if pumps:
            powers = np.asarray([float(pump.power_w) for pump in pumps], dtype=np.float64)
            frequencies = np.asarray(
                [3.0e8 / (float(pump.wavelength_nm) * 1e-9) for pump in pumps],
                dtype=np.float64,
            )
            if np.sum(powers) > 0.0:
                f_hat = float(np.average(frequencies, weights=powers))
                alpha_pump = float(np.average(self._pump_alpha_npm, weights=powers))
            else:
                f_hat = float(np.mean(self.f_abs_hz))
                alpha_pump = float(np.mean(self._pump_alpha_npm))
        else:
            f_hat = float(np.mean(self.f_abs_hz))
            alpha_pump = float(np.mean(self.alpha_npm))
        return fit_buglia_profile(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.alpha_npm,
            f_hat,
            alpha_pump,
            alpha_pump,
            self._n_steps,
        )


BUGLIA2024_RA_MODEL_INFO = ModelInfo(
    name="buglia2024_ra_gn",
    summary=(
        "Buglia 2024 RA-GN subset: Eq. (2) profile fitted to the coupled Raman ODE "
        "and Eq. (6)-(8) SPM/XPM GN integrals; Eq. (9)/(11)/(12) closed forms are "
        "approximated, not reproduced term by term."
    ),
    family="gn+raman",
    reference="Buglia et al., JLT 42(2):636-648, 2024, doi:10.1109/jlt.2023.3315127",
    options=(
        ("raman_pumps", "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear' (CZ)."),
        ("raman_ase", "Expose the newly generated Raman ASE before the EDFA. Default True."),
        ("raman_temperature_k", "Fibre temperature of the spontaneous Raman factor. Default 300 K."),
        ("raman_peak_gain_coefficient_m_per_w", "Peak local Raman gain rate. Default 6e-14."),
        ("include_isrs", "Include channel-channel ISRS in the coupled ODE. Default True."),
        ("fixed_point_iterations", "Maximum backward-pump fixed-point iterations. Default 24."),
        ("n_steps", "RK4 steps of the ODE and of the Eq. (2) fit. Default 32."),
        ("n_quadrature", "Gauss-Legendre nodes of the in-band NLI integral. Default 24."),
    ),
)


class Buglia2024RAModel(DRARamanModelBase):
    """Buglia 2024 RA-GN paper control.

    Forward and backward pumps are both supported.  The profile is the fitted
    Eq. (2) form; the NLI is the Eq. (6)-(8) GN integral evaluated with that
    profile.  Cross-span NLI is summed incoherently (Eq. (4) with
    ``epsilon = 1``), which is the conservative convention.

    Args:
        grid: Spectrum grid.
        ref_fiber_config: Reference fibre.
        symbol_rate_hz: Symbol rate (Hz).
        connections: Optional per-connection grid.
        raman_pumps: Pumps applied to every span; ``None`` uses the fibre config.
        raman_gain_model: ``"gaussian"`` or ``"linear"``.
        raman_ase: Include the Raman ASE before the EDFA.
        raman_temperature_k: Fibre temperature (K).
        raman_peak_gain_coefficient_m_per_w: Peak local Raman gain rate.
        include_isrs: Include channel-channel ISRS in the ODE.
        fixed_point_iterations: Maximum backward-pump iterations.
        n_steps: RK4 steps of the ODE and profile fit.
        n_quadrature: Gauss-Legendre nodes of the NLI in-band integral.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.
    """

    info = BUGLIA2024_RA_MODEL_INFO
    _kernel_class = Buglia2024RASpanKernel
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
        **route_options: object,
    ) -> None:
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


register_model(BUGLIA2024_RA_MODEL_INFO, Buglia2024RAModel)
