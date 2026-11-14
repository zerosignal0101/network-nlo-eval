"""Network QoT validity policies, reports, and uncertainty integration.

Validity integration is off by default. Explicit flag, reject, degrade and
soft policies respectively audit, reject invalid channels, subtract the
recommended margin, or subtract only the tilt-derived soft margin. Enabled
policies prevent NaN or negative-infinite SNR from passing a threshold check;
positive infinity remains a valid zero-noise result.

Dense-grid overlap, under-resolved Raman integration and unmodeled FWM are
independent flags by default. Each may be explicitly upgraded to rejection
without changing physical SNR. Soft margin is separate from hard validity and
from the uncertainty margin. Link reports own immutable copies of their arrays;
path reports combine validity across links and add margins in dB.

The bounded LRU cache keys reports by link, exact occupied-power fingerprint
and span length. This prevents reuse after an occupancy or power change; the
provider fingerprints power in the model's convention after conversion.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from network_nlo_eval.models.base import MultiSpanOpticalPath, QoTModel
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)
from network_nlo_eval.models.uncertainty import (
    H3ErrorBudget,
    H3UncertaintyConfig,
    estimate_h3_error_budget,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.types import LinkKey, NDArrayBool, NDArrayFloat

__all__ = [
    "TILT_SOFT_MARGIN_FULL_DB",
    "TILT_SOFT_MARGIN_FULL_DB_RECOMMENDED",
    "ChannelValidity",
    "DenseGridAction",
    "FwmAction",
    "H3LinkValidityProvider",
    "InvalidChannelPolicy",
    "LinkValidity",
    "LinkValidityCache",
    "LinkValidityPolicy",
    "LinkValidityProvider",
    "RamanResolutionAction",
    "link_optical_path",
    "occupancy_fingerprint",
    "tilt_soft_margin_db",
]


#: Full-scale additivity of the informational tilt soft margin (dB).
#:
#: The informational score decays from 1 at
#: :data:`~network_nlo_eval.models.pcfm.TILT_INFORMATIONAL_KNEE_DB` (10 dB) to 0
#: at :data:`~network_nlo_eval.models.pcfm.TILT_INFORMATIONAL_ZERO_DB` (30 dB).
#: A full scale of 1 dB puts the hard-flag boundary (15 dB) at
#: ``(1 - 0.84375) * 1 = 0.15625 dB``, which is the order of the ~0.15 dB
#: representation residual measured at the C+L six-pump benchmark.  The scale
#: is deliberately small: it is a soft correction and must never compete with
#: the hard flag, which is what actually gates validity.
TILT_SOFT_MARGIN_FULL_DB: float = 1.0

#: Recommended full scale of the informational tilt soft margin (dB).
#:
#: The fifth-round accuracy matrix measured the H3 residual at the hard-flag
#: boundary: 0.244 dB at 10 dB tilt, 0.302 dB at 15 dB and 0.302 dB at 20 dB.
#: At 15 dB the informational curve has lost ``1 - 0.84375 = 0.15625`` of its
#: confidence, so the default scale produces only 0.156 dB there -- about half
#: the measured residual.  Covering it requires
#: ``0.302 / 0.15625 = 1.9328`` dB; 1.933 dB is the first three-decimal value
#: that is not below the measurement.  The value is a documented
#: recommendation, not the default: :class:`LinkValidityPolicy` still starts
#: at :data:`TILT_SOFT_MARGIN_FULL_DB`, so no existing number moves.
TILT_SOFT_MARGIN_FULL_DB_RECOMMENDED: float = 1.933


def tilt_soft_margin_db(
    confidence: NDArrayFloat | float,
    *,
    full_scale_db: float = TILT_SOFT_MARGIN_FULL_DB,
) -> NDArrayFloat:
    r"""Map informational tilt confidence to an additive soft margin.

    The formula is linear in the *lost* confidence:

    .. math::

        m_{soft} = m_{full} \cdot \left(1 - \mathrm{clip}(c, 0, 1)\right).

    It is zero while the tilt is at or below the informational knee (``c = 1``),
    increases monotonically as the tilt grows, and reaches ``full_scale_db`` at
    the zero of the informational curve.  A non-finite confidence is treated as
    zero confidence (the worst case), so a missing or broken score can only
    increase the margin, never silently remove it.

    The mapping is intentionally not a validity rule: it only produces a
    number.  A caller opts in to subtracting it with
    :class:`LinkValidityPolicy` ``mode="soft"``.

    Parameters
    ----------
    confidence : ndarray | float
        Informational tilt confidence, scalar or ``(N,)`` array.
    full_scale_db : float
        Margin at ``confidence == 0`` (dB); default
        :data:`TILT_SOFT_MARGIN_FULL_DB`.

    Returns
    -------
    ndarray
        Non-negative soft margin (dB), same shape as ``confidence``; a scalar
        input returns a 0-d array.

    Raises
    ------
    ValueError
        ``full_scale_db`` is negative or not finite.
    """
    scale = float(full_scale_db)
    if not np.isfinite(scale) or scale < 0.0:
        msg = f"full_scale_db must be finite and >= 0, got {full_scale_db!r}."
        raise ValueError(msg)
    values = np.asarray(confidence, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    return scale * (1.0 - np.clip(values, 0.0, 1.0))


class InvalidChannelPolicy(StrEnum):
    """Policy for invalid channels and uncertainty margins.

    Attributes
    ----------
    OFF : str
        Disable validity integration and return the original SNR.
    FLAG : str
        Audit validity without changing usable SNR.
    REJECT : str
        Replace invalid-channel SNR with negative infinity.
    DEGRADE : str
        Reject invalid channels and subtract the recommended margin from valid ones.
    SOFT : str
        Reject invalid channels and subtract only the tilt-derived soft margin.
    """

    OFF = "off"
    FLAG = "flag"
    REJECT = "reject"
    DEGRADE = "degrade"
    SOFT = "soft"


class DenseGridAction(StrEnum):
    """Choose whether dense-grid overlap is flagged or rejected.

    Rejection changes availability only; the selected invalid-channel policy
    then determines its effect on SNR. Flag is the default.
    """

    FLAG = "flag"
    REJECT = "reject"


class RamanResolutionAction(StrEnum):
    """Choose whether under-resolved Raman integration is flagged or rejected.

    Rejection changes availability only; the selected invalid-channel policy
    then determines its effect on SNR. Flag is the default.
    """

    FLAG = "flag"
    REJECT = "reject"


class FwmAction(StrEnum):
    """Choose whether unmodeled FWM is flagged or rejected.

    FWM can be required near zero dispersion or an out-of-band dispersion anchor.
    The flag records a model applicability boundary. Rejection changes availability
    only and the invalid-channel policy determines its effect on SNR.
    """

    FLAG = "flag"
    REJECT = "reject"


#: Accepted values of :attr:`LinkValidity.fwm_validity_mode`.
_FWM_VALIDITY_MODES: tuple[str, ...] = tuple(action.value for action in FwmAction)


@dataclass(frozen=True)
class LinkValidityPolicy:
    """Explicit policy for applying network-level validity.

    Parameters
    ----------
    mode : {"off", "flag", "reject", "degrade", "soft"}
        Policy mode; the default is off.
    soft_margin_full_scale_db : float
        Full-scale tilt margin in dB. Stored margins use the default scale and are
        proportionally rescaled only in soft mode.
    dense_grid_action, raman_resolution_action, fwm_action : {"flag", "reject"}
        Independent actions for dense overlap, under-resolution and unmodeled FWM.
        Each defaults to flag and only affects availability when mode is enabled.

    Raises
    ------
    ValueError
        If a policy, action or margin scale is invalid.
    """

    mode: InvalidChannelPolicy | str = InvalidChannelPolicy.OFF
    soft_margin_full_scale_db: float = TILT_SOFT_MARGIN_FULL_DB
    dense_grid_action: DenseGridAction | str = DenseGridAction.FLAG
    raman_resolution_action: RamanResolutionAction | str = RamanResolutionAction.FLAG
    fwm_action: FwmAction | str = FwmAction.FLAG

    def __post_init__(self) -> None:
        """Normalize policy enums and validate the full-scale margin.

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        try:
            normalized = InvalidChannelPolicy(self.mode)
        except ValueError:
            allowed = [policy.value for policy in InvalidChannelPolicy]
            msg = f"unknown validity policy {self.mode!r}; expected one of {allowed}."
            raise ValueError(msg) from None
        object.__setattr__(self, "mode", normalized)
        try:
            dense_action = DenseGridAction(self.dense_grid_action)
        except ValueError:
            allowed = [action.value for action in DenseGridAction]
            msg = f"unknown dense grid action {self.dense_grid_action!r}; expected one of {allowed}."
            raise ValueError(msg) from None
        object.__setattr__(self, "dense_grid_action", dense_action)
        try:
            raman_action = RamanResolutionAction(self.raman_resolution_action)
        except ValueError:
            allowed = [action.value for action in RamanResolutionAction]
            msg = (
                f"unknown Raman resolution action {self.raman_resolution_action!r}; "
                f"expected one of {allowed}."
            )
            raise ValueError(msg) from None
        object.__setattr__(self, "raman_resolution_action", raman_action)
        try:
            fwm_action = FwmAction(self.fwm_action)
        except ValueError:
            allowed = [action.value for action in FwmAction]
            msg = f"unknown FWM action {self.fwm_action!r}; expected one of {allowed}."
            raise ValueError(msg) from None
        object.__setattr__(self, "fwm_action", fwm_action)
        scale = float(self.soft_margin_full_scale_db)
        if not np.isfinite(scale) or scale < 0.0:
            msg = f"soft_margin_full_scale_db must be finite and >= 0, got {self.soft_margin_full_scale_db!r}."
            raise ValueError(msg)
        object.__setattr__(self, "soft_margin_full_scale_db", scale)

    @property
    def enabled(self) -> bool:
        """Return whether the policy mode is enabled."""
        return self.mode is not InvalidChannelPolicy.OFF

    @property
    def rejects_dense_grid(self) -> bool:
        """Return whether enabled policy rejects dense-grid overlap."""
        return self.enabled and self.dense_grid_action is DenseGridAction.REJECT

    @property
    def rejects_raman_resolution(self) -> bool:
        """Return whether enabled policy rejects Raman under-resolution."""
        return (
            self.enabled
            and self.raman_resolution_action is RamanResolutionAction.REJECT
        )

    @property
    def rejects_fwm(self) -> bool:
        """Return whether enabled policy rejects unmodeled FWM."""
        return self.enabled and self.fwm_action is FwmAction.REJECT

    @staticmethod
    def _flag_mask(
        flag: bool | NDArrayBool, shape: tuple[int, ...], name: str
    ) -> NDArrayBool:
        """Normalize a scalar or per-channel flag to a broadcastable boolean mask.

        Parameters
        ----------
        flag : bool | ndarray
            Scalar or per-channel flag.
        shape : tuple[int, ...]
            Target channel-array shape.
        name : str
            Flag name for error messages.

        Returns
        -------
        ndarray
            Broadcastable boolean mask.

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        mask = np.asarray(flag, dtype=bool)
        if mask.ndim > 1 or mask.shape not in ((), shape):
            msg = f"{name} must be a scalar or have shape {shape}, got {mask.shape}."
            raise ValueError(msg)
        return mask

    def effective_valid(
        self,
        valid: NDArrayBool,
        dense_grid_overlap: bool | NDArrayBool = False,
        *,
        raman_auto_under_resolved: bool | NDArrayBool = False,
        fwm_required: bool | NDArrayBool = False,
    ) -> NDArrayBool:
        """Apply enabled rejection actions to the validity mask.

        Off mode and flag actions preserve the original mask. Dense-grid, Raman and
        FWM rejection actions combine independently without changing physical SNR.

        Parameters
        ----------
        valid : ndarray
            Per-channel validity mask.
        dense_grid_overlap : bool | ndarray
            Dense-grid overlap flag, scalar or per-channel.
        raman_auto_under_resolved : bool | ndarray
            Raman under-resolution flag, scalar or per-channel.
        fwm_required : bool | ndarray
            Unmodeled-FWM flag, scalar or per-channel.

        Returns
        -------
        ndarray
            Validity mask for policy decisions.
        """
        values = np.asarray(valid, dtype=bool)
        if not (
            self.rejects_dense_grid or self.rejects_raman_resolution or self.rejects_fwm
        ):
            return values
        effective = values
        if self.rejects_dense_grid:
            dense = self._flag_mask(
                dense_grid_overlap, values.shape, "dense_grid_overlap"
            )
            if bool(np.any(dense)):
                effective = effective & ~dense
        if self.rejects_raman_resolution:
            under = self._flag_mask(
                raman_auto_under_resolved, values.shape, "raman_auto_under_resolved"
            )
            if bool(np.any(under)):
                effective = effective & ~under
        if self.rejects_fwm:
            fwm = self._flag_mask(fwm_required, values.shape, "fwm_required")
            if bool(np.any(fwm)):
                effective = effective & ~fwm
        return effective

    def scale_soft_margin(self, soft_margin_db: NDArrayFloat) -> NDArrayFloat:
        """Scale stored tilt margins to the policy full-scale value.

        Parameters
        ----------
        soft_margin_db : ndarray
            Per-channel tilt soft margin (dB).

        Returns
        -------
        ndarray
            Scaled soft margin (dB).
        """
        values = np.asarray(soft_margin_db, dtype=np.float64)
        factor = float(self.soft_margin_full_scale_db) / TILT_SOFT_MARGIN_FULL_DB
        if factor == 1.0:
            return values
        return values * factor

    def apply(
        self,
        snr_db: NDArrayFloat,
        valid: NDArrayBool,
        margin_db: NDArrayFloat,
        soft_margin_db: NDArrayFloat | None = None,
        *,
        dense_grid_overlap: bool | NDArrayBool = False,
        raman_auto_under_resolved: bool | NDArrayBool = False,
        fwm_required: bool | NDArrayBool = False,
    ) -> NDArrayFloat:
        """Apply this policy to physical SNR for threshold comparisons.

        Off returns the input unchanged. Enabled modes prevent NaN and negative
        infinity from passing; flag audits only, reject drops invalid channels,
        degrade subtracts the recommended margin, and soft subtracts the scaled tilt
        margin.

        Parameters
        ----------
        snr_db : ndarray
            Physical SNR per channel (dB).
        valid : ndarray
            Per-channel validity mask.
        margin_db : ndarray
            Recommended margin per channel (dB).
        soft_margin_db : ndarray | None
            Per-channel tilt soft margin (dB).
        dense_grid_overlap : bool | ndarray
            Dense-grid overlap flag, scalar or per-channel.
        raman_auto_under_resolved : bool | ndarray
            Raman under-resolution flag, scalar or per-channel.
        fwm_required : bool | ndarray
            Unmodeled-FWM flag, scalar or per-channel.

        Returns
        -------
        ndarray
            Policy-adjusted SNR (dB).
        """
        values = np.asarray(snr_db, dtype=np.float64)
        mode = InvalidChannelPolicy(self.mode)
        if mode is InvalidChannelPolicy.OFF:
            return values
        return _effective_snr(
            values,
            self.effective_valid(
                valid,
                dense_grid_overlap,
                raman_auto_under_resolved=raman_auto_under_resolved,
                fwm_required=fwm_required,
            ),
            np.asarray(margin_db, dtype=np.float64),
            mode,
            None if soft_margin_db is None else self.scale_soft_margin(soft_margin_db),
        )


def occupancy_fingerprint(power_profile_w: NDArrayFloat, span_length_m: float) -> str:
    """Hash the exact power-vector bit patterns and span length.

    The 16-byte BLAKE2b digest distinguishes signed zero and any change in
    occupancy or power, preventing reports from being reused for another spectrum.

    Parameters
    ----------
    power_profile_w : ndarray
        Per-channel launch power (W).
    span_length_m : float
        Span length (m).

    Returns
    -------
    str
        Hexadecimal 16-byte BLAKE2b digest.

    Raises
    ------
    ValueError
        If an input shape or configuration value is invalid.
    """
    values = np.asarray(power_profile_w, dtype=np.float64)
    if values.ndim != 1:
        msg = f"power_profile_w must be one-dimensional, got shape {values.shape}."
        raise ValueError(msg)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(np.float64(span_length_m).tobytes())
    return digest.hexdigest()


def _effective_snr(
    snr_db: NDArrayFloat,
    valid: NDArrayBool,
    margin_db: NDArrayFloat,
    mode: InvalidChannelPolicy,
    soft_margin_db: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Apply an enabled policy to SNR, treating NaN and negative infinity as invalid.

    Parameters
    ----------
    snr_db : ndarray
        Physical SNR per channel (dB).
    valid : ndarray
        Per-channel validity mask.
    margin_db : ndarray
        Recommended margin per channel (dB).
    mode : InvalidChannelPolicy
        Invalid-channel policy.
    soft_margin_db : ndarray | None
        Per-channel tilt soft margin (dB).

    Returns
    -------
    ndarray
        Policy-adjusted SNR (dB).
    """
    values = np.asarray(snr_db, dtype=np.float64)
    if mode is InvalidChannelPolicy.OFF:
        return values
    # Positive infinity represents zero noise; reject only NaN and negative infinity.
    usable = ~np.isnan(values) & ~np.isneginf(values)
    if mode is InvalidChannelPolicy.FLAG:
        # Audit mode preserves usable SNR and prevents NaN from passing a threshold.
        return np.where(usable, values, -np.inf)
    accept = np.asarray(valid, dtype=bool) & usable
    if mode is InvalidChannelPolicy.REJECT:
        return np.where(accept, values, -np.inf)
    if mode is InvalidChannelPolicy.SOFT:
        soft = (
            np.zeros(values.shape, dtype=np.float64)
            if soft_margin_db is None
            else soft_margin_db
        )
        # Invalid soft margins are treated as zero to avoid increasing SNR.
        soft = np.where(np.isfinite(soft) & (soft >= 0.0), soft, 0.0)
        return np.where(accept, values - soft, -np.inf)
    penalty = np.asarray(margin_db, dtype=np.float64)
    penalty = np.where(accept & np.isfinite(penalty), penalty, 0.0)
    return np.where(accept, values - penalty, -np.inf)


@dataclass(frozen=True)
class LinkValidity:
    """Immutable channel validity for one link and occupied spectrum.

    Arrays are copied, validated, and frozen in model channel order. A flag records
    an applicability concern and does not by itself imply invalidity. Optional
    informational tilt confidence defaults to one and soft margin defaults to zero.
    Dense-grid, Raman resolution and FWM flags change availability only when the
    corresponding action is explicitly set to reject.

    Attributes
    ----------
    valid, active, flagged : ndarray
        Per-channel availability, occupancy and applicability flags.
    confidence : ndarray
        Evidence-strength score in [0, 1].
    margin_db : ndarray
        Recommended margin in dB; infinity indicates invalidity.
    tilt_confidence_informational : ndarray
        Informational tilt score; does not change validity or confidence.
    soft_margin_db : ndarray
        Tilt-derived margin in dB, applied only by soft policy.
    model : str
        Model producing the report.
    span_count : int
        Number of physical spans.
    fingerprint : str
        Occupied-spectrum fingerprint.
    link_key : LinkKey or None
        Link identifier.
    dense_grid_overlap, raman_auto_under_resolved, fwm_required : bool
        Independent applicability flags.
    fwm_reason : str
        Plus-separated reasons for the FWM flag.
    fwm_validity_mode : str
        FWM validity mode used by the budget.
    """

    valid: NDArrayBool
    active: NDArrayBool
    flagged: NDArrayBool
    confidence: NDArrayFloat
    margin_db: NDArrayFloat
    model: str
    span_count: int = 0
    fingerprint: str = ""
    link_key: LinkKey | None = None
    tilt_confidence_informational: NDArrayFloat | float | None = None
    soft_margin_db: NDArrayFloat | float | None = None
    dense_grid_overlap: bool = False
    raman_auto_under_resolved: bool = False
    fwm_required: bool = False
    fwm_reason: str = ""
    fwm_validity_mode: str = "flag"

    def __post_init__(self) -> None:
        """Copy and freeze channel arrays, validate values, and broadcast optional scalars.

        Missing informational confidence defaults to one and soft margin to zero.

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        arrays: dict[str, Any] = {}
        for name, dtype in (
            ("valid", np.bool_),
            ("active", np.bool_),
            ("flagged", np.bool_),
            ("confidence", np.float64),
            ("margin_db", np.float64),
        ):
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.ndim != 1:
                msg = f"{name} must be one-dimensional, got shape {value.shape}."
                raise ValueError(msg)
            value.setflags(write=False)
            arrays[name] = value
            object.__setattr__(self, name, value)

        shapes = {name: value.shape for name, value in arrays.items()}
        if len(set(shapes.values())) != 1:
            msg = f"validity arrays must share one shape, got {shapes}."
            raise ValueError(msg)
        # Broadcast optional fields after validating the required arrays.
        num_channels = int(arrays["valid"].shape[0])
        for name, default in (
            ("tilt_confidence_informational", 1.0),
            ("soft_margin_db", 0.0),
        ):
            raw = getattr(self, name)
            if raw is None:
                value = np.full(num_channels, default, dtype=np.float64)
            else:
                value = np.asarray(raw, dtype=np.float64)
                if value.ndim == 0:
                    value = np.full(num_channels, float(value), dtype=np.float64)
                elif value.shape != (num_channels,):
                    msg = (
                        f"{name} must have shape ({num_channels},), got {value.shape}."
                    )
                    raise ValueError(msg)
                value = np.array(value, dtype=np.float64, copy=True)
            value.setflags(write=False)
            arrays[name] = value
            object.__setattr__(self, name, value)

        if np.any(np.isnan(arrays["margin_db"])) or np.any(arrays["margin_db"] < 0.0):
            msg = "margin_db must be non-negative and never NaN."
            raise ValueError(msg)
        confidence = arrays["confidence"]
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            msg = "confidence must lie in [0, 1]."
            raise ValueError(msg)
        informational = arrays["tilt_confidence_informational"]
        if np.any(~np.isfinite(informational)) or np.any(
            (informational < 0.0) | (informational > 1.0)
        ):
            msg = "tilt_confidence_informational must be finite and lie in [0, 1]."
            raise ValueError(msg)
        soft = arrays["soft_margin_db"]
        if np.any(~np.isfinite(soft)) or np.any(soft < 0.0):
            msg = "soft_margin_db must be finite and non-negative."
            raise ValueError(msg)
        if int(self.span_count) < 0:
            msg = f"span_count must be >= 0, got {self.span_count}."
            raise ValueError(msg)
        object.__setattr__(
            self, "raman_auto_under_resolved", bool(self.raman_auto_under_resolved)
        )
        object.__setattr__(self, "fwm_required", bool(self.fwm_required))
        object.__setattr__(self, "fwm_reason", str(self.fwm_reason))
        if self.fwm_validity_mode not in _FWM_VALIDITY_MODES:
            msg = f"fwm_validity_mode must be one of {_FWM_VALIDITY_MODES}, got {self.fwm_validity_mode!r}."
            raise ValueError(msg)
        object.__setattr__(self, "fwm_validity_mode", str(self.fwm_validity_mode))

    @property
    def num_channels(self) -> int:
        """Return the number of channels.

        Returns
        -------
        int
            Channel count.
        """
        return int(self.valid.shape[0])

    def channel_valid(self, channel_idx: int) -> bool:
        """Return hard validity for one channel.

        Parameters
        ----------
        channel_idx : int
            Channel index.

        Returns
        -------
        bool
            Whether the channel is valid.
        """
        return bool(self.valid[channel_idx])

    def margin_at(self, channel_idx: int) -> float:
        """Return one channel recommended margin (dB).

        Parameters
        ----------
        channel_idx : int
            Channel index.

        Returns
        -------
        float
            Recommended margin (dB).
        """
        return float(self.margin_db[channel_idx])

    def soft_margin_at(self, channel_idx: int) -> float:
        """Return one channel tilt soft margin (dB).

        Parameters
        ----------
        channel_idx : int
            Channel index.

        Returns
        -------
        float
            Soft margin (dB).
        """
        return float(self.soft_margin_db[channel_idx])

    def effective_valid(self, policy: LinkValidityPolicy | None = None) -> NDArrayBool:
        """Return link availability under the selected policy.

        Parameters
        ----------
        policy : LinkValidityPolicy | None
            Validity policy; None selects the default off policy.

        Returns
        -------
        ndarray
            Validity mask for policy decisions.
        """
        return (policy or LinkValidityPolicy()).effective_valid(
            self.valid,
            self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )

    def adjusted_snr_db(
        self,
        snr_db: NDArrayFloat,
        policy: LinkValidityPolicy | None = None,
    ) -> NDArrayFloat:
        """Apply a policy to this link SNR vector.

        Parameters
        ----------
        snr_db : ndarray
            Physical SNR per channel (dB).
        policy : LinkValidityPolicy | None
            Validity policy; None selects the default off policy.

        Returns
        -------
        ndarray
            Policy-adjusted SNR (dB).

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        values = np.asarray(snr_db, dtype=np.float64)
        if values.shape != self.valid.shape:
            msg = f"snr_db shape {values.shape} does not match the report {self.valid.shape}."
            raise ValueError(msg)
        return (policy or LinkValidityPolicy()).apply(
            values,
            self.valid,
            self.margin_db,
            self.soft_margin_db,
            dense_grid_overlap=self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )

    @classmethod
    def from_budget(
        cls,
        budget: H3ErrorBudget,
        *,
        link_key: LinkKey | None = None,
        fingerprint: str = "",
        span_count: int | None = None,
    ) -> LinkValidity:
        """Construct a link report from the matching evaluated path budget.

        Validity additionally requires active channels and finite recommended margin.
        The provider supplies a fingerprint of the model-coordinate power spectrum.

        Parameters
        ----------
        budget : H3ErrorBudget
            Error budget computed for this link and spectrum.
        link_key : LinkKey | None
            Link identifier.
        fingerprint : str
            Fingerprint of the model-coordinate spectrum.
        span_count : int | None
            Physical span count; None uses the budget count.

        Returns
        -------
        LinkValidity
            Immutable link validity report.
        """
        report = budget.validity
        margin = np.asarray(budget.margin_db, dtype=np.float64)
        valid = (
            np.asarray(report.valid, dtype=bool)
            & np.asarray(report.active, dtype=bool)
            & np.isfinite(margin)
        )
        # Broadcast path tilt confidence; soft margin applies to active channels only.
        informational = float(report.tilt_confidence_informational)
        active = np.asarray(report.active, dtype=bool)
        soft = np.where(active, tilt_soft_margin_db(informational), 0.0)
        return cls(
            valid=valid,
            active=active,
            flagged=np.asarray(report.flagged, dtype=bool),
            confidence=np.asarray(report.confidence, dtype=np.float64),
            margin_db=margin,
            tilt_confidence_informational=np.full(
                active.shape, informational, dtype=np.float64
            ),
            soft_margin_db=soft,
            model=str(budget.model),
            span_count=int(report.span_count if span_count is None else span_count),
            fingerprint=fingerprint,
            link_key=link_key,
            dense_grid_overlap=bool(report.dense_grid_overlap),
            raman_auto_under_resolved=bool(report.raman_auto_under_resolved),
            fwm_required=bool(report.fwm_required),
            fwm_reason=str(report.fwm_reason),
            fwm_validity_mode=str(report.fwm_validity_mode),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (non-finite margins as ``None``)."""
        return {
            "model": self.model,
            "link_key": list(self.link_key) if self.link_key is not None else None,
            "fingerprint": self.fingerprint,
            "span_count": int(self.span_count),
            "valid": [bool(value) for value in self.valid],
            "active": [bool(value) for value in self.active],
            "flagged": [bool(value) for value in self.flagged],
            "confidence": [float(value) for value in self.confidence],
            "margin_db": [
                float(value) if np.isfinite(value) else None for value in self.margin_db
            ],
            "tilt_confidence_informational": [
                float(value) for value in self.tilt_confidence_informational
            ],
            "soft_margin_db": [float(value) for value in self.soft_margin_db],
            "dense_grid_overlap": bool(self.dense_grid_overlap),
            "raman_auto_under_resolved": bool(self.raman_auto_under_resolved),
            "fwm_required": bool(self.fwm_required),
            "fwm_reason": str(self.fwm_reason),
            "fwm_validity_mode": str(self.fwm_validity_mode),
        }


@dataclass(frozen=True)
class ChannelValidity:
    """Aggregate per-channel validity across a path.

    Validity is the intersection of link validity, and valid-link margins are
    summed in dB. This is a conservative additive approximation, not a power-domain
    noise conversion. Soft margins are summed separately. Dense-grid, Raman and
    FWM flags are ORed across links; FWM reasons are deduplicated and reject is
    the more conservative FWM mode.

    Attributes
    ----------
    valid : ndarray
        Per-channel path validity.
    margin_db, soft_margin_db : ndarray
        Summed recommended and tilt-derived margins in dB.
    dense_grid_overlap, raman_auto_under_resolved, fwm_required : bool
        Applicability flags from any link.
    fwm_reason : str
        Deduplicated, plus-separated FWM reasons.
    fwm_validity_mode : str
        Reject if any link uses reject, otherwise flag.
    """

    valid: NDArrayBool
    margin_db: NDArrayFloat
    soft_margin_db: NDArrayFloat | None = None
    dense_grid_overlap: bool = False
    raman_auto_under_resolved: bool = False
    fwm_required: bool = False
    fwm_reason: str = ""
    fwm_validity_mode: str = "flag"

    def __post_init__(self) -> None:
        """Copy, validate and freeze path validity and margin arrays.

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        valid = np.array(self.valid, dtype=bool, copy=True)
        margin = np.array(self.margin_db, dtype=np.float64, copy=True)
        if valid.ndim != 1 or valid.shape != margin.shape:
            msg = f"validity arrays must be one-dimensional and equal, got {valid.shape} and {margin.shape}."
            raise ValueError(msg)
        if np.any(np.isnan(margin)) or np.any(margin < 0.0):
            msg = "margin_db must be non-negative and never NaN."
            raise ValueError(msg)
        num_channels = int(valid.shape[0])
        raw_soft = self.soft_margin_db
        if raw_soft is None:
            soft = np.zeros(num_channels, dtype=np.float64)
        else:
            soft = np.asarray(raw_soft, dtype=np.float64)
            if soft.ndim == 0:
                soft = np.full(num_channels, float(soft), dtype=np.float64)
            elif soft.shape != (num_channels,):
                msg = f"soft_margin_db must have shape ({num_channels},), got {soft.shape}."
                raise ValueError(msg)
            soft = np.array(soft, dtype=np.float64, copy=True)
        if np.any(~np.isfinite(soft)) or np.any(soft < 0.0):
            msg = "soft_margin_db must be finite and non-negative."
            raise ValueError(msg)
        valid.setflags(write=False)
        margin.setflags(write=False)
        soft.setflags(write=False)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "margin_db", margin)
        object.__setattr__(self, "soft_margin_db", soft)
        object.__setattr__(self, "dense_grid_overlap", bool(self.dense_grid_overlap))
        object.__setattr__(
            self, "raman_auto_under_resolved", bool(self.raman_auto_under_resolved)
        )
        object.__setattr__(self, "fwm_required", bool(self.fwm_required))
        object.__setattr__(self, "fwm_reason", str(self.fwm_reason))
        if self.fwm_validity_mode not in _FWM_VALIDITY_MODES:
            msg = f"fwm_validity_mode must be one of {_FWM_VALIDITY_MODES}, got {self.fwm_validity_mode!r}."
            raise ValueError(msg)
        object.__setattr__(self, "fwm_validity_mode", str(self.fwm_validity_mode))

    @classmethod
    def combine(
        cls, links: Sequence[LinkValidity], num_channels: int
    ) -> ChannelValidity:
        """Combine link reports by intersecting validity and summing valid-link margins.

        Flags are ORed; FWM reasons are deduplicated and reject is the conservative
        combined FWM mode. An empty link list yields all-valid zero-margin channels.

        Parameters
        ----------
        links : Sequence[LinkValidity]
            Link reports in path order.
        num_channels : int
            Number of channels.

        Returns
        -------
        ChannelValidity
            Aggregated path validity.

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        valid = np.ones(num_channels, dtype=bool)
        margin = np.zeros(num_channels, dtype=np.float64)
        soft = np.zeros(num_channels, dtype=np.float64)
        dense = False
        raman_under = False
        fwm_required = False
        fwm_reasons: list[str] = []
        fwm_reject = False
        for link in links:
            if link.num_channels != num_channels:
                msg = f"link {link.link_key} has {link.num_channels} channels, expected {num_channels}."
                raise ValueError(msg)
            valid &= link.valid
            margin += np.where(
                link.valid,
                np.where(np.isfinite(link.margin_db), link.margin_db, 0.0),
                0.0,
            )
            soft += np.where(link.valid, link.soft_margin_db, 0.0)
            dense = dense or bool(link.dense_grid_overlap)
            raman_under = raman_under or bool(link.raman_auto_under_resolved)
            fwm_required = fwm_required or bool(link.fwm_required)
            reason = str(link.fwm_reason or "")
            for token in reason.split("+") if reason else ():
                if token and token not in fwm_reasons:
                    fwm_reasons.append(token)
            fwm_reject = fwm_reject or link.fwm_validity_mode == FwmAction.REJECT
        return cls(
            valid=valid,
            margin_db=margin,
            soft_margin_db=soft,
            dense_grid_overlap=dense,
            raman_auto_under_resolved=raman_under,
            fwm_required=fwm_required,
            fwm_reason="+".join(fwm_reasons),
            fwm_validity_mode=FwmAction.REJECT if fwm_reject else FwmAction.FLAG,
        )

    def adjusted_snr_db(
        self,
        snr_db: NDArrayFloat,
        policy: LinkValidityPolicy | None = None,
    ) -> NDArrayFloat:
        """Apply a policy to the path SNR vector.

        Parameters
        ----------
        snr_db : ndarray
            Physical SNR per channel (dB).
        policy : LinkValidityPolicy | None
            Validity policy; None selects the default off policy.

        Returns
        -------
        ndarray
            Policy-adjusted SNR (dB).

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        values = np.asarray(snr_db, dtype=np.float64)
        if values.shape != self.valid.shape:
            msg = f"snr_db shape {values.shape} does not match the verdict {self.valid.shape}."
            raise ValueError(msg)
        return (policy or LinkValidityPolicy()).apply(
            values,
            self.valid,
            self.margin_db,
            self.soft_margin_db,
            dense_grid_overlap=self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )

    def effective_valid(self, policy: LinkValidityPolicy | None = None) -> NDArrayBool:
        """Return path availability under the selected policy.

        Parameters
        ----------
        policy : LinkValidityPolicy | None
            Validity policy; None selects the default off policy.

        Returns
        -------
        ndarray
            Validity mask for policy decisions.
        """
        return (policy or LinkValidityPolicy()).effective_valid(
            self.valid,
            self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )


class LinkValidityProvider(Protocol):
    """A callable that produces the validity report of one link.

    The signature mirrors the information :class:`QoTValidator` already has in
    ``_calc_link_noise``; a provider must be a pure function of its arguments
    plus its own caches.
    """

    def __call__(
        self,
        link_key: LinkKey,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_profile_w: NDArrayFloat,
        span_length_m: float,
    ) -> LinkValidity:
        """Return the validity report of one link at one occupied spectrum."""
        ...


@dataclass
class LinkValidityCache:
    """Bounded LRU cache for link validity reports.

    Keys contain the link, exact spectrum fingerprint and span length. A link key
    alone is insufficient because pump mode, bandwidth and tilt affect margins.

    Attributes
    ----------
    max_entries : int
        Maximum cache size; least-recently-used entries are evicted.
    hits, misses : int
        Lookup counters.
    """

    max_entries: int = 64
    hits: int = 0
    misses: int = 0
    _entries: OrderedDict[tuple[LinkKey, str, int], LinkValidity] = field(
        default_factory=OrderedDict, repr=False
    )

    def __post_init__(self) -> None:
        """Reject a non-positive capacity.

        Raises
        ------
        ValueError
            ``max_entries < 1``。
        """
        if self.max_entries < 1:
            msg = f"max_entries must be >= 1, got {self.max_entries}."
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of cached links."""
        return len(self._entries)

    @staticmethod
    def key(
        link_key: LinkKey, power_profile_w: NDArrayFloat, span_length_m: float
    ) -> tuple[LinkKey, str, int]:
        """Build the link, spectrum fingerprint and rounded span-length cache key.

        Parameters
        ----------
        link_key : LinkKey
            Link identifier.
        power_profile_w : ndarray
            Per-channel launch power (W).
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[LinkKey, str, int]
            Link key, spectrum fingerprint and span length in micrometers.
        """
        return (
            link_key,
            occupancy_fingerprint(power_profile_w, span_length_m),
            round(span_length_m * 1e6),
        )

    def get(
        self, link_key: LinkKey, power_profile_w: NDArrayFloat, span_length_m: float
    ) -> LinkValidity | None:
        """Look up a link state and update hit or miss counters.

        Parameters
        ----------
        link_key : LinkKey
            Link identifier.
        power_profile_w : ndarray
            Per-channel launch power (W).
        span_length_m : float
            Span length (m).

        Returns
        -------
        LinkValidity | None
            Cached report, or None on a miss.
        """
        key = self.key(link_key, power_profile_w, span_length_m)
        cached = self._entries.get(key)
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return cached

    def put(
        self,
        link_key: LinkKey,
        power_profile_w: NDArrayFloat,
        span_length_m: float,
        validity: LinkValidity,
    ) -> None:
        """Store a report and evict least-recently-used entries when full.

        Parameters
        ----------
        link_key : LinkKey
            Link identifier.
        power_profile_w : ndarray
            Per-channel launch power (W).
        span_length_m : float
            Span length (m).
        validity : LinkValidity
            Link report to cache.
        """
        key = self.key(link_key, power_profile_w, span_length_m)
        self._entries[key] = validity
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def invalidate(self, link_key: LinkKey) -> int:
        """Remove all cached spectra for one link.

        Parameters
        ----------
        link_key : LinkKey
            Link identifier.

        Returns
        -------
        int
            Number of entries removed.
        """
        keys = [key for key in self._entries if key[0] == link_key]
        for key in keys:
            del self._entries[key]
        return len(keys)

    def clear(self) -> None:
        """Drop all entries and reset the counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0


def link_optical_path(
    fiber_config: FiberSpanConfig,
    edfa_config: EDFAConfig | None,
    span_length_m: float,
) -> MultiSpanOpticalPath:
    """Build the one-link optical path used by the provider.

    The route splits the full link into spans and a remainder using the supplied
    span length. A default EDFA is used when none is provided.

    Parameters
    ----------
    fiber_config : FiberSpanConfig
        Fiber configuration for the link.
    edfa_config : EDFAConfig | None
        Span EDFA configuration, or None for the default.
    span_length_m : float
        Span length (m).

    Returns
    -------
    MultiSpanOpticalPath
        One-link optical path.

    Raises
    ------
    ValueError
        If an input shape or configuration value is invalid.
    """
    span_km = float(span_length_m) / 1e3
    if span_km <= 0.0:
        msg = f"span_length_m must be positive, got {span_length_m}."
        raise ValueError(msg)
    if float(fiber_config.length_km) <= 0.0:
        msg = f"fiber length must be positive, got {fiber_config.length_km}."
        raise ValueError(msg)
    fiber = fiber_config.model_copy(update={"max_span_length_km": span_km})
    default_edfa = EDFAConfig(
        target_gain_db=None, noise_figure_db=5.0, gain_ripple_db=None
    )
    return MultiSpanOpticalPath([fiber], [edfa_config or default_edfa])


class H3LinkValidityProvider:
    """Evaluate a link with PCFM/H3 and produce a cached validity report.

    The provider evaluates the physical path and error budget without changing its
    point SNR. Input power is converted once from the declared input convention to
    the model launch convention; the converted spectrum is used as the cache key.

    Parameters
    ----------
    model : QoTModel
        PCFM/H3 model exposing ``evaluate_path``, ``kernel_for`` and ``n_nodes``.
    input_power_convention : str, optional
        Input power convention, ``total`` or ``per_pol``; defaults to total.
    config : H3UncertaintyConfig or None, optional
        Error-budget options.
    cache : LinkValidityCache or None, optional
        Report cache; a new cache is created when omitted.

    Raises
    ------
    ValueError
        If the input power convention is invalid.
    """

    def __init__(
        self,
        model: QoTModel,
        *,
        input_power_convention: str = "total",
        config: H3UncertaintyConfig | None = None,
        cache: LinkValidityCache | None = None,
    ) -> None:
        self.model = model
        self.input_power_convention = resolve_launch_power_convention(
            input_power_convention
        )
        self.config = config or H3UncertaintyConfig()
        self.cache = cache if cache is not None else LinkValidityCache()
        self._warned_convention_mismatch = False

    @property
    def model_power_convention(self) -> str:
        """Return the model launch convention, defaulting to total.

        Returns
        -------
        str
            Model power convention.
        """
        return str(getattr(self.model, "launch_power_convention", "total"))

    def _warn_once_if_mismatched(self) -> None:
        """Warn once when input and model power conventions differ."""
        if self._warned_convention_mismatch:
            return
        source = self.input_power_convention
        target = self.model_power_convention
        if source == target:
            return
        factor = 0.5 if target == "per_pol" else 2.0
        direction = "total -> per_pol" if target == "per_pol" else "per_pol -> total"
        warnings.warn(
            f"h3 validity provider: caller input_power_convention={source!r} differs from the model "
            f"launch_power_convention={target!r}; the launch power is scaled by {factor:g} "
            f"({direction}) before the H3 evaluation and used as the cache key.",
            UserWarning,
            stacklevel=2,
        )
        self._warned_convention_mismatch = True

    def __call__(
        self,
        link_key: LinkKey,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_profile_w: NDArrayFloat,
        span_length_m: float,
    ) -> LinkValidity:
        """Return the cached or newly evaluated report for one occupied link.

        Parameters
        ----------
        link_key : LinkKey
            Link identifier.
        fiber_config : FiberSpanConfig
            Fiber configuration for the link.
        edfa_config : EDFAConfig | None
            Span EDFA configuration, or None for the default.
        power_profile_w : ndarray
            Per-channel launch power (W).
        span_length_m : float
            Span length (m).

        Returns
        -------
        LinkValidity
            Link validity report.

        Raises
        ------
        ValueError
            If an input shape or configuration value is invalid.
        """
        power = np.asarray(power_profile_w, dtype=np.float64)
        if power.shape != (self.model.n_nodes,):
            msg = f"power_profile_w must have shape ({self.model.n_nodes},), got {power.shape}."
            raise ValueError(msg)
        self._warn_once_if_mismatched()
        power_model = to_convention(
            power, self.input_power_convention, self.model_power_convention
        )
        cached = self.cache.get(link_key, power_model, span_length_m)
        if cached is not None:
            return cached

        path = link_optical_path(fiber_config, edfa_config, span_length_m)
        result = self.model.evaluate_path(path, power_model)
        budget = estimate_h3_error_budget(
            self.model, path, power_model, result, config=self.config
        )
        validity = LinkValidity.from_budget(
            budget,
            link_key=link_key,
            fingerprint=occupancy_fingerprint(power_model, span_length_m),
            span_count=result.span_count,
        )
        self.cache.put(link_key, power_model, span_length_m, validity)
        return validity
