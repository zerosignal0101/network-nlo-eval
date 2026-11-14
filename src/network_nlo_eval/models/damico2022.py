"""D'Amico 2022 scalable, disaggregated GGN approximation.

The model separates single-channel (SC/SPM) and cross-channel (XC/XPM)
noise, using frequency-dependent dispersion and the segmented approximation
in Eq. (28)-(29). Four-wave mixing is omitted. The SC term uses a standard
closed-form GN estimator rather than a triple GGN integral. The SRS front end
uses the ECZ triangular approximation rather than the paper's coupled SRS
solver; Raman amplification and DRA pumps are not modeled.

The paper's cross-span SC coherence coefficient has no published closed form
in the paper. This implementation constructs a phase-mismatch visibility
envelope and converts its total power factor to an equivalent scalar phase
ramp. The ramp reproduces the total factor for identical spans but compresses
pairwise and channel-specific phase information; prefix factors need not match
the pairwise model. XC uses a ramp reproducing incoherent total accumulation.

References
----------
A. D'Amico et al., "Scalable and Disaggregated GGN Approximation Applied to a
C+L+S Optical Network", J. Lightwave Technol. 40, 3499-3511 (2022),
doi:10.1109/JLT.2022.3162134."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
    SpanKernel,
    register_model,
)
from network_nlo_eval.models.isrs_gn import _SRS_CR
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_fiber_attenuation_npm,
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
)
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "DAMICO2022_MODEL_INFO",
    "DEFAULT_COHERENCE_POINTS",
    "NOMINAL_BAND_EDGES_HZ",
    "SPM_WEIGHT",
    "XPM_WEIGHT",
    "Damico2022Model",
    "Damico2022SpanKernel",
]

SPM_WEIGHT = 16.0 / 27.0

XPM_WEIGHT = 2.0 * 16.0 / 27.0

DELTA_F_CO_HZ = 15e12

NOMINAL_BAND_EDGES_HZ: tuple[tuple[float, float], ...] = (
    (184.4e12, 191.6e12),  # L
    (191.6e12, 195.9e12),  # C
    (195.9e12, 205.4e12),  # S
    (205.4e12, 220.4e12),  # E
)

XC_AGGREGATIONS = ("exact", "centroid")

XC_AGGREGATION_DEFAULT = "exact"


def _band_group_masks(
    f_abs_hz: NDArrayFloat,
    band_edges_hz: tuple[tuple[float, float], ...],
    n_bands: int | None,
) -> list[NDArrayFloat]:
    """Group channels by band and return nonempty Boolean masks.

    Parameters
    ----------
    f_abs_hz : numpy.ndarray
        Channel center frequencies in Hz.
    band_edges_hz : tuple
        Nominal band edges, used when ``n_bands`` is None.
    n_bands : int or None
        Number of equally sized groups of sorted channels.

    Returns
    -------
    list of numpy.ndarray
        Boolean masks, including a group for channels outside nominal bands."""
    masks: list[NDArrayFloat] = []
    if n_bands is None:
        for low, high in band_edges_hz:
            mask = (f_abs_hz >= low) & (f_abs_hz < high)
            if np.any(mask):
                masks.append(mask)
        covered = np.zeros_like(f_abs_hz, dtype=bool)
        for mask in masks:
            covered |= mask
        if np.any(~covered):
            masks.append(~covered)
        return masks
    if n_bands < 1:
        msg = f"n_bands must be >= 1, got {n_bands}."
        raise ValueError(msg)
    order = np.argsort(f_abs_hz)
    chunks = np.array_split(order, n_bands)
    for chunk in chunks:
        if chunk.size == 0:
            continue
        mask = np.zeros_like(f_abs_hz, dtype=bool)
        mask[chunk] = True
        masks.append(mask)
    return masks


def _coherence_envelope(
    beta2_abs_mean: float,
    rs_hz: float,
    span_length_m: float,
    n_spans: int,
    n_points: int,
) -> float:
    """Calculate the constructed cross-span SC coherence coefficient.

    For span separation d, the visibility is the absolute mean cosine of
    ``2*pi**2*|beta2|*nu**2*d*L`` over the channel bandwidth. The coefficient
    is ``(2/N) * sum((N-d)*visibility(d))``. It ranges from zero for incoherent
    accumulation to N-1 for full coherence. This is an implementation model,
    not a closed form supplied by the paper.

    Parameters
    ----------
    beta2_abs_mean : float
        Mean absolute dispersion in s²/m.
    rs_hz : float
        Channel symbol rate in Hz.
    span_length_m : float
        Span length in meters.
    n_spans : int
        Number of spans.
    n_points : int
        Number of midpoint quadrature points.

    Returns
    -------
    float
        Nonnegative dimensionless coherence coefficient."""
    if n_spans <= 1:
        return 0.0
    if rs_hz <= 0.0 or span_length_m <= 0.0:
        return 0.0
    step = rs_hz / n_points
    nu = (np.arange(n_points) + 0.5) * step - rs_hz / 2.0
    nu2 = nu * nu
    total = 0.0
    for delta in range(1, n_spans):
        phase = 2.0 * np.pi**2 * beta2_abs_mean * nu2 * delta * span_length_m
        kappa = float(np.abs(np.mean(np.cos(phase))))
        total += (n_spans - delta) * kappa
    return 2.0 * total / n_spans


DEFAULT_COHERENCE_POINTS = 64

_RAMP_BISECTION_STEPS = 60


def _dirichlet_factor(n_spans: int, psi: float) -> float:
    """Return the coherent power factor for equal amplitudes with phase step ``psi``.

    The factor is ``[sin(N*psi/2)/sin(psi/2)]**2``, with limit N² at zero."""
    n = int(n_spans)
    if n <= 1:
        return 1.0
    denominator = np.sin(0.5 * psi)
    if abs(denominator) < 1e-12:
        return float(n) ** 2
    return float(np.sin(0.5 * n * psi) / denominator) ** 2


def _equivalent_phase_ramp(n_spans: int, target: float) -> float:
    """Invert the Dirichlet factor to obtain an equivalent phase step in radians.

    The target is clipped to [0, N²]. The ramp matches the total factor, but
    cannot retain the original pairwise phase structure."""
    n = int(n_spans)
    if n <= 1:
        return 0.0
    goal = float(np.clip(target, 0.0, float(n) ** 2))
    low = 0.0
    high = 2.0 * np.pi / n
    for _ in range(_RAMP_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if _dirichlet_factor(n, mid) > goal:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


class Damico2022SpanKernel:
    """Single-span SC/XC kernel with frequency-dependent dispersion.

    Channel properties and the XC pair matrix are precomputed. Exact XC uses
    pairwise summation; centroid aggregation approximates each band by a single
    interferer and can incur large errors near band edges.

    Attributes
    ----------
    alpha_npm : numpy.ndarray
        Channel power attenuation in Np/m.
    beta2_s2_m : numpy.ndarray
        Frequency-dependent dispersion in s²/m.
    gamma_w_m : numpy.ndarray
        Channel nonlinear coefficients in 1/(W m).
    mx : numpy.ndarray
        Pairwise XC kernel with zero diagonal."""

    def __init__(
        self,
        *,
        fiber_config: FiberSpanConfig,
        lambdas_m: NDArrayFloat,
        f_abs_hz: NDArrayFloat,
        f_rel_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        symbol_rate_hz: float,
        srs_cr_w_m_hz: float,
        n_z_steps: int,
        xc_aggregation: str,
        band_edges_hz: tuple[tuple[float, float], ...],
        n_bands: int | None,
        coherence_points: int,
        c_inf_enabled: bool,
        use_dispersion_slope: bool,
    ) -> None:
        """Prepare channel properties and pairwise matrices for the fiber.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber attenuation, dispersion, and mode area parameters.
        lambdas_m, f_abs_hz, f_rel_hz, b_hz : numpy.ndarray
            Wavelengths, absolute and relative frequencies, and channel bandwidths.
        symbol_rate_hz : float
            Symbol rate used for SRS band edges.
        srs_cr_w_m_hz : float
            Raman gain slope in 1/(W m Hz).
        n_z_steps : int
            Number of fiber segments.
        xc_aggregation : {"exact", "centroid"}
            Pairwise or band centroid XC aggregation.
        band_edges_hz : tuple
            Nominal frequency band edges in Hz.
        n_bands : int or None
            Number of channel groups, or None for nominal bands.
        coherence_points : int
            Number of visibility quadrature points.
        c_inf_enabled : bool
            Enable the SC coherence envelope.
        use_dispersion_slope : bool
            Use the configured wavelength-dependent dispersion profile."""
        if xc_aggregation not in XC_AGGREGATIONS:
            msg = f"xc_aggregation must be one of {XC_AGGREGATIONS}, got {xc_aggregation!r}."
            raise ValueError(msg)
        if n_z_steps < 2:
            msg = f"n_z_steps must be >= 2, got {n_z_steps}."
            raise ValueError(msg)
        if coherence_points < 8:
            msg = f"coherence_points must be >= 8, got {coherence_points}."
            raise ValueError(msg)

        self.fiber_config = fiber_config
        self.f_abs_hz = np.asarray(f_abs_hz, dtype=np.float64)
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.b_hz = np.asarray(b_hz, dtype=np.float64)
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)
        self.n_z_steps = int(n_z_steps)
        self.xc_aggregation = xc_aggregation
        self.coherence_points = int(coherence_points)
        self.c_inf_enabled = bool(c_inf_enabled)
        self.use_dispersion_slope = bool(use_dispersion_slope)

        lambdas = np.asarray(lambdas_m, dtype=np.float64)
        if self.fiber_config.use_attenuation_polynomial:
            alpha_npm = _calc_fiber_attenuation_npm(
                lambdas,
                self.fiber_config.reference_wavelength_nm,
                self.fiber_config.attenuation_alpha2,
                self.fiber_config.attenuation_alpha1,
                self.fiber_config.attenuation_alpha0,
            )
        else:
            alpha = self.fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            alpha_npm = np.full(lambdas.shape, alpha, dtype=np.float64)
        a_eff_m2 = _calc_aeff_dynamic(
            lambdas,
            self.fiber_config.effective_area_um2_ref,
            self.fiber_config.aeff_slope_um2_nm,
            self.fiber_config.reference_wavelength_nm,
        )

        if use_dispersion_slope:
            d_ps_nm_km = (
                self.fiber_config.dispersion_parameter_d
                + self.fiber_config.dispersion_slope_s
                * (lambdas * 1e9 - self.fiber_config.reference_wavelength_nm)
            )
        else:
            d_ps_nm_km = np.full(
                lambdas.shape,
                self.fiber_config.dispersion_parameter_d,
                dtype=np.float64,
            )
        beta2 = np.array(
            [
                _calc_beta2_from_d(float(d), lam)
                for d, lam in zip(d_ps_nm_km, lambdas, strict=True)
            ],
            dtype=np.float64,
        )
        n2 = self.fiber_config.nonlinear_index_n2
        gamma = 2.0 * np.pi * n2 / (lambdas * a_eff_m2)

        self.alpha_npm = alpha_npm
        self.a_eff_m2 = a_eff_m2
        self.beta2_s2_m = beta2
        self.gamma_w_m = gamma

        beta2_avg = 0.5 * (beta2[:, None] + beta2[None, :])
        df = self.f_abs_hz[:, None] - self.f_abs_hz[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = 1.0 / (4.0 * np.pi * np.abs(beta2_avg * df))
        np.fill_diagonal(mx, 0.0)
        self.mx = mx
        self.beta2_avg = beta2_avg

        self.group_masks = _band_group_masks(self.f_abs_hz, band_edges_hz, n_bands)
        self.group_ids = np.zeros(self.f_abs_hz.shape[0], dtype=np.int64)
        for group_index, mask in enumerate(self.group_masks):
            self.group_ids[mask] = group_index

        self._coherence_cache: dict[
            tuple[int, float, float, float], tuple[float, float, float, float, float]
        ] = {}

    def _gamma_per_channel(self) -> NDArrayFloat:
        """Return the channel nonlinear coefficients in 1/(W m), as in Eq. (5)."""
        return self.gamma_w_m

    def _srs_profile(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Compute output power, Raman tilt, and normalized span power profiles.

        The ECZ triangular Raman approximation supplies the output power. The
        analytical profile is normalized at the input and matches the output endpoint.

        Returns
        -------
        tuple of numpy.ndarray
            Output powers, tilt coefficients, and channel-by-position profiles."""
        total_power = float(np.sum(power_in_w))
        f_m_hz = float(self.f_rel_hz[0] - self.symbol_rate_hz / 2.0)
        f_M_hz = float(self.f_rel_hz[-1] + self.symbol_rate_hz / 2.0)
        r_f = _calc_raman_profile_jit(
            self.f_rel_hz, f_m_hz, f_M_hz, DELTA_F_CO_HZ, total_power
        )
        p_out = _calc_span_out_power_jit(
            power_in_w, self.f_rel_hz, self.alpha_npm, span_length_m, r_f, 0.0
        )

        steps = self.n_z_steps
        z = np.linspace(0.0, span_length_m, steps + 1)
        leff_z = (1.0 - np.exp(-self.alpha_npm[:, None] * z[None, :])) / self.alpha_npm[
            :, None
        ]
        leff_end = leff_z[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio2 = np.where(power_in_w > 0.0, (p_out / power_in_w) ** 2, 0.0)
        ratio2 = np.nan_to_num(ratio2, nan=0.0, posinf=0.0, neginf=0.0)
        exponent = self.alpha_npm[:, None] * (
            span_length_m - z[None, :]
        ) + self.srs_cr_w_m_hz * r_f[:, None] * (leff_end[:, None] - leff_z)
        exponent = np.clip(exponent, -700.0, 700.0)
        rho2 = ratio2[:, None] * np.exp(exponent)
        # Eq. (29) requires input-normalized profiles; absolute power enters separately.
        head = rho2[:, :1]
        rho2 = np.where(head > 0.0, rho2 / np.where(head > 0.0, head, 1.0), rho2)
        return p_out, r_f, rho2

    def _psi_approx(self, rho2: NDArrayFloat, z: NDArrayFloat) -> NDArrayFloat:
        """Compute the segmented Eq. (28)-(29) XC integration factor.

        Each segment approximates the normalized power profile exponentially. The
        returned factor is the square of the sum of the segment effective lengths.

        Parameters
        ----------
        rho2 : numpy.ndarray
            Normalized channel-by-position power profile.
        z : numpy.ndarray
            Positions along the span in meters.

        Returns
        -------
        numpy.ndarray
            Integration factors for each channel."""
        dz = float(z[1] - z[0])
        safe = np.where(rho2 > 0.0, rho2, 1.0)
        log_rho2 = np.log(safe)
        # The logarithmic slope is negative for attenuation.
        pump_alpha = (log_rho2[:, 1:] - log_rho2[:, :-1]) / dz
        step_drop = rho2[:, 1:] - rho2[:, :-1]
        alive = (rho2[:, 1:] > 0.0) & (rho2[:, :-1] > 0.0)
        pump_alpha = np.where(alive, pump_alpha, -1.0)
        leff = np.abs(step_drop / np.sqrt(np.abs(pump_alpha))) * np.sign(pump_alpha)
        leff = np.where(alive, leff, 0.0)
        total = np.sum(leff, axis=1)
        total = np.where(np.isfinite(total), total, 0.0)
        return np.square(total)

    def _sc_noise(
        self,
        power_in_w: NDArrayFloat,
        rho2: NDArrayFloat,
        z: NDArrayFloat,
        r_f: NDArrayFloat,
    ) -> NDArrayFloat:
        """Calculate single-span SC noise in watts using the GN closed form.

        The profile integral gives effective length; the inverse local attenuation
        including Raman tilt gives asymptotic length. The GN estimator remains finite
        on the diagonal where the XC expression is singular."""
        dz = float(z[1] - z[0])
        l_eff = np.sum(0.5 * (rho2[:, :-1] + rho2[:, 1:]) * dz, axis=1)
        alpha_tilde = self.alpha_npm + self.srs_cr_w_m_hz * r_f
        alpha_tilde = np.where(alpha_tilde > 0.0, alpha_tilde, np.inf)
        l_asym = 1.0 / alpha_tilde
        beta2_abs = np.abs(self.beta2_s2_m)
        rs = self.b_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            asinh_arg = np.pi**2 * l_asym * beta2_abs * rs * rs / 2.0
            psi = (
                np.arcsinh(asinh_arg)
                * l_eff
                * l_eff
                / (2.0 * np.pi * beta2_abs * l_asym)
            )
        psi = np.where(np.isfinite(psi), psi, 0.0)
        gamma = self._gamma_per_channel()
        noise = SPM_WEIGHT * gamma * gamma * psi * power_in_w**3 / (rs * rs)
        return np.where(power_in_w > 0.0, np.maximum(noise, 0.0), 0.0)

    def _xc_noise(
        self,
        power_in_w: NDArrayFloat,
        z_int: NDArrayFloat,
    ) -> NDArrayFloat:
        """Calculate single-span XC noise in watts using Eq. (25) and (29).

        Exact aggregation sums all channel pairs and is independent of grouping.
        Centroid aggregation replaces each group with a representative interferer;
        its approximation error can be large near band edges."""
        gamma = self._gamma_per_channel()
        rs = self.b_hz
        # a_j = P_j²/Rs_j, w_j = z_int_j · a_j.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(rs > 0.0, power_in_w * power_in_w / rs, 0.0)
        w = z_int * a

        if self.xc_aggregation == "exact":
            weighted = self.mx @ w
        else:
            # Remove the channel under test from its own centroid group.
            weighted = np.zeros_like(w)
            for mask in self.group_masks:
                sel = np.zeros_like(w)
                sel[mask] = w[mask]
                strength = float(np.sum(sel))
                if strength <= 0.0:
                    continue
                moment = float(np.sum(sel * self.f_abs_hz))
                own = mask & (power_in_w > 0.0)
                own_w = np.where(own, w, 0.0)
                own_strength = float(np.sum(own_w))
                own_moment = float(np.sum(own_w * self.f_abs_hz))
                group_strength = np.full(
                    self.f_abs_hz.shape, strength, dtype=np.float64
                )
                centroid_f = np.full(
                    self.f_abs_hz.shape, moment / strength, dtype=np.float64
                )
                if own_strength > 0.0 and own_strength < strength:
                    usable = own & (own_strength < strength)
                    group_strength = np.where(usable, strength - own_strength, strength)
                    centroid_f = np.where(
                        usable,
                        (moment - own_moment) / np.where(usable, group_strength, 1.0),
                        centroid_f,
                    )
                beta2_centroid = np.interp(centroid_f, self.f_abs_hz, self.beta2_s2_m)
                beta2_avg = 0.5 * (self.beta2_s2_m + beta2_centroid)
                df = self.f_abs_hz - centroid_f
                with np.errstate(divide="ignore", invalid="ignore"):
                    kernel = 1.0 / (4.0 * np.pi * np.abs(beta2_avg * df))
                kernel = np.where(np.isfinite(kernel), kernel, 0.0)
                weighted += kernel * group_strength

        noise = XPM_WEIGHT * gamma * gamma * power_in_w * weighted
        noise = np.where(np.isfinite(noise), noise, 0.0)
        return np.where(power_in_w > 0.0, np.maximum(noise, 0.0), 0.0)

    def _band_coherence(
        self,
        power: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> tuple[float, float, float, float, float]:
        """Return the band-averaged envelope, power factors, and phase steps.

        The tuple contains ``(c_inf, factor_spm, factor_xpm, psi_spm, psi_xpm)``.
        The SC factor is N*(1+c_inf), while the XC factor is N. Phase ramps match
        these totals for equal amplitudes but compress pairwise phase information."""
        n = max(1, int(n_spans))
        if n <= 1:
            return 0.0, 1.0, 1.0, 0.0, 0.0
        active = power > 0.0
        if np.any(active):
            beta2_mean = float(np.mean(np.abs(self.beta2_s2_m[active])))
            rs_mean = float(np.mean(self.b_hz[active]))
        else:
            beta2_mean = float(np.mean(np.abs(self.beta2_s2_m)))
            rs_mean = float(np.mean(self.b_hz))
        key = (n, float(span_length_m), beta2_mean, rs_mean)
        cached = self._coherence_cache.get(key)
        if cached is not None:
            return cached
        if self.c_inf_enabled:
            c_inf = _coherence_envelope(
                beta2_mean, rs_mean, span_length_m, n, self.coherence_points
            )
        else:
            c_inf = 0.0
        factor_spm = float(n) * (1.0 + c_inf)
        factor_xpm = float(n)
        psi_spm = _equivalent_phase_ramp(n, factor_spm)
        psi_xpm = _equivalent_phase_ramp(n, factor_xpm)
        cached = (float(c_inf), factor_spm, factor_xpm, psi_spm, psi_xpm)
        self._coherence_cache[key] = cached
        return cached

    def evaluate_components(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> dict[str, NDArrayFloat | float]:
        """Calculate single-span powers and coherence diagnostics.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Channel launch powers in watts; zero indicates an inactive channel.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Link span count for the coherence calculation.

        Returns
        -------
        dict
            Output power, SC/XC single-span noise, coherence coefficient, factors,
            phase steps, and link-level SC/XC totals. ``spm_w`` and ``xpm_w`` are
            single-span values; ``link_spm_w`` and ``link_xpm_w`` include factors."""
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            msg = f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            raise ValueError(msg)
        if span_length_m <= 0.0:
            msg = f"span_length_m must be positive, got {span_length_m}."
            raise ValueError(msg)
        n_spans = max(1, int(n_effective_spans))

        p_out, r_f, rho2 = self._srs_profile(power, span_length_m)
        z = np.linspace(0.0, span_length_m, self.n_z_steps + 1)
        z_int = self._psi_approx(rho2, z)
        sc = self._sc_noise(power, rho2, z, r_f)
        xc = self._xc_noise(power, z_int)

        c_inf, factor_spm, factor_xpm, psi_spm, psi_xpm = self._band_coherence(
            power, span_length_m, n_spans
        )

        return {
            "power_out_w": p_out,
            "sc_w": sc,
            "xc_w": xc,
            "c_inf": float(c_inf),
            "spm_w": sc,
            "xpm_w": xc,
            "factor_spm": factor_spm,
            "factor_xpm": factor_xpm,
            "psi_spm": psi_spm,
            "psi_xpm": psi_xpm,
            "link_spm_w": factor_spm * sc,
            "link_xpm_w": factor_xpm * xc,
        }

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Calculate fiber output and NLI for a span or link diagnostic.

        With no span context, return link-level powers for ``n_effective_spans``.
        With a single-span context, return single-span powers without amplitudes.
        With a multispan context, return single-span powers and complex amplitudes
        whose phase steps reproduce the SC and XC coherence factors for equal spans.
        The amplitudes are referenced to the fiber output of the current span.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Channel launch powers in watts.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Link span count for diagnostics without a context.
        span_context : SpanContext or None, optional
            Position of the current span within the link.

        Returns
        -------
        SpanFiberOutput
            Fiber output powers and, for multispan contexts, NLI amplitudes."""
        if span_context is None:
            parts = self.evaluate_components(
                power_in_w, span_length_m, max(1, int(n_effective_spans))
            )
            return SpanFiberOutput(
                power_out_w=np.asarray(parts["power_out_w"], dtype=np.float64),
                spm_noise_w=np.asarray(parts["link_spm_w"], dtype=np.float64),
                xpm_noise_w=np.asarray(parts["link_xpm_w"], dtype=np.float64),
            )

        parts = self.evaluate_components(
            power_in_w, span_length_m, int(span_context.total_spans)
        )
        sc = np.asarray(parts["sc_w"], dtype=np.float64)
        xc = np.asarray(parts["xc_w"], dtype=np.float64)
        power_out = np.asarray(parts["power_out_w"], dtype=np.float64)

        if int(span_context.total_spans) <= 1:
            return SpanFiberOutput(
                power_out_w=power_out, spm_noise_w=sc, xpm_noise_w=xc
            )

        phase_scale = span_context.cumulative_length_m / float(
            span_context.span_length_m
        )
        psi_spm = float(parts["psi_spm"])
        psi_xpm = float(parts["psi_xpm"])
        amplitude_spm = np.sqrt(sc) * np.exp(1j * (psi_spm * phase_scale))
        amplitude_xpm = np.sqrt(xc) * np.exp(1j * (psi_xpm * phase_scale))
        return SpanFiberOutput(
            power_out_w=power_out,
            spm_noise_w=sc,
            xpm_noise_w=xc,
            nli_amplitude_spm=amplitude_spm,
            nli_amplitude_xpm=amplitude_xpm,
        )


DAMICO2022_MODEL_INFO = ModelInfo(
    name="damico2022",
    summary="D'Amico 2022 disaggregated GGN approximation: SC/XC split with C_inf coherent SC envelope.",
    family="ggn",
    reference="A. D'Amico et al., 'Scalable and Disaggregated GGN Approximation Applied to a "
    "C+L+S Optical Network', JLT 2022, doi:10.1109/JLT.2022.3162134",
    options=(
        ("srs_cr_w_m_hz", "Raman gain slope C_r (1/(W*m*Hz)); 0 disables ISRS."),
        ("n_z_steps", "Number of fiber steps for the Eq. (28)/(29) segmented sum."),
        ("xc_aggregation", "'exact' | 'centroid'; per-band XC aggregation mode."),
        (
            "n_bands",
            "Equal-width band groups for the XC sum; None uses nominal band edges.",
        ),
        ("c_inf_enabled", "Enable the cross-span coherent SC envelope C_inf."),
        (
            "coherence_points",
            (
                "Quadrature points of the C_inf visibility integral; larger values converge "
                "the envelope for strongly dispersive spans (the default 64 is enough for SMF)."
            ),
        ),
    ),
)


class Damico2022Model(RouteModel):
    """Route model for the D'Amico 2022 GGN approximation.

    The route layer handles span accumulation, amplifiers, and SNR. The kernel
    provides single-span noise and equivalent cross-span coherence phases.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Channel grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber parameters.
    symbol_rate_hz : float, optional
        Symbol rate in Hz.
    srs_cr_w_m_hz : float, optional
        Raman gain slope; zero disables ISRS.
    n_z_steps : int, optional
        Number of fiber segments.
    xc_aggregation : {"exact", "centroid"}, optional
        XC aggregation method.
    n_bands : int or None, optional
        Channel group count, or None for nominal L/C/S/E boundaries.
    c_inf_enabled : bool, optional
        Enable the SC coherence envelope.
    use_dispersion_slope : bool, optional
        Use the configured wavelength-dependent dispersion profile.
    coherence_points : int, optional
        Number of visibility quadrature points.
    **route_options
        Options passed to the route model."""

    info = DAMICO2022_MODEL_INFO

    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        *,
        srs_cr_w_m_hz: float = _SRS_CR,
        n_z_steps: int = 64,
        xc_aggregation: str = XC_AGGREGATION_DEFAULT,
        n_bands: int | None = None,
        c_inf_enabled: bool = True,
        use_dispersion_slope: bool = True,
        coherence_points: int = DEFAULT_COHERENCE_POINTS,
        **route_options,
    ) -> None:
        """Initialize the route and model-specific options."""
        super().__init__(
            grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options
        )
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)
        self.n_z_steps = int(n_z_steps)
        self.xc_aggregation = xc_aggregation
        self.n_bands = n_bands
        self.c_inf_enabled = bool(c_inf_enabled)
        self.use_dispersion_slope = bool(use_dispersion_slope)
        if coherence_points < 8:
            msg = f"coherence_points must be >= 8, got {coherence_points}."
            raise ValueError(msg)
        self.coherence_points = int(coherence_points)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Build a span kernel for the given fiber configuration."""
        return Damico2022SpanKernel(
            fiber_config=fiber_config,
            lambdas_m=self.lambdas_m,
            f_abs_hz=self.f_abs_hz,
            f_rel_hz=self.f_rel_hz,
            b_hz=self.channel_bandwidth_hz_channels,
            symbol_rate_hz=self.symbol_rate_hz,
            srs_cr_w_m_hz=self.srs_cr_w_m_hz,
            n_z_steps=self.n_z_steps,
            xc_aggregation=self.xc_aggregation,
            band_edges_hz=NOMINAL_BAND_EDGES_HZ,
            n_bands=self.n_bands,
            coherence_points=self.coherence_points,
            c_inf_enabled=self.c_inf_enabled,
            use_dispersion_slope=self.use_dispersion_slope,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """Evaluate the span and apply amplifier and ASE effects.

        The fiber output carries physical single-span powers and, for multispan
        links, complex amplitudes referenced to the fiber output. The route layer
        handles their propagation and coherent accumulation."""
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate(
            power_in_w, span_length_m, 1, span_context=span_context
        )
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        plain = SpanFiberOutput(
            power_out_w=fiber_output.power_out_w,
            spm_noise_w=fiber_output.spm_noise_w,
            xpm_noise_w=fiber_output.xpm_noise_w,
        )
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=plain,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )
        if fiber_output.nli_amplitude_spm is None:
            return element
        return replace(
            element,
            nli_amplitude_spm=fiber_output.nli_amplitude_spm,
            nli_amplitude_xpm=fiber_output.nli_amplitude_xpm,
        )


register_model(DAMICO2022_MODEL_INFO, Damico2022Model)
