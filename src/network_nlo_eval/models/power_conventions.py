"""Define and convert launch-power polarization conventions.

The network reference plane uses total dual-polarization power per channel.
Models may instead use per-polarization power, half the physical total.
Signal and noise must use the same convention; SNR is independent of the
choice. Convert once at the model boundary and do not reuse a noise cache
whose known convention differs from that of the evaluator.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

__all__ = [
    "LAUNCH_POWER_CONVENTIONS",
    "ase_polarization_scale",
    "launch_power_scale",
    "resolve_launch_power_convention",
    "to_convention",
]

#: Supported launch-power conventions.
#:
#: ``"total"`` is dual-polarization power per channel (the historical default).
#: ``"per_pol"`` is per-polarization power; total power is twice this value.
LAUNCH_POWER_CONVENTIONS: tuple[str, ...] = ("total", "per_pol")


def resolve_launch_power_convention(
    launch_power_convention: str,
    signal_polarizations: int | None = None,
) -> str:
    """Validate a launch-power convention and optional polarization count.

    Parameters
    ----------
    launch_power_convention : str
        Either ``"total"`` for dual-polarization power or ``"per_pol"`` for
        power per polarization.
    signal_polarizations : int or None, optional
        Number of polarizations represented by the ODE driving power. When
        supplied with ``"per_pol"``, this must be 2. ``None`` skips this check.

    Returns
    -------
    str
        Validated convention.

    Raises
    ------
    ValueError
        If the convention is unknown or per-polarization power is paired with
        a polarization count other than 2.

    Notes
    -----
    ``"total"`` with two signal polarizations remains accepted for historical
    comparisons, although it double counts total power as a coupling driver.
    """
    convention = str(launch_power_convention)
    if convention not in LAUNCH_POWER_CONVENTIONS:
        msg = f"launch_power_convention must be one of {LAUNCH_POWER_CONVENTIONS}, got {launch_power_convention!r}."
        raise ValueError(msg)
    if (
        convention == "per_pol"
        and signal_polarizations is not None
        and int(signal_polarizations) != 2
    ):
        msg = (
            "launch_power_convention='per_pol' declares a per-polarization input, so the ODE state "
            "must be per-polarization: signal_polarizations must be 2, got "
            f"{int(signal_polarizations)}. Use launch_power_convention='total' for a "
            "per-channel total-power input."
        )
        raise ValueError(msg)
    return convention


def launch_power_scale(convention: str) -> float:
    """Return model power divided by physical total power.

    Parameters
    ----------
    convention : str
        Launch-power convention, either ``"total"`` or ``"per_pol"``.

    Returns
    -------
    float
        1.0 for total power and 0.5 for per-polarization power.

    Raises
    ------
    ValueError
        If the convention is unknown.
    """
    return 0.5 if resolve_launch_power_convention(convention) == "per_pol" else 1.0


def ase_polarization_scale(convention: str) -> float:
    """Return the scale for newly generated dual-polarization ASE.

    Parameters
    ----------
    convention : str
        Launch-power convention, either ``"total"`` or ``"per_pol"``.

    Returns
    -------
    float
        1.0 for total power and 0.5 for per-polarization power.

    Notes
    -----
    Apply the scale once to EDFA or ROADM booster ASE. Input ASE already in
    model coordinates must not be scaled again.

    Raises
    ------
    ValueError
        If the convention is unknown.
    """
    return launch_power_scale(convention)


def to_convention(power_w: Any, from_convention: str, to_convention: str) -> Any:
    """Convert power between total and per-polarization conventions.

    Parameters
    ----------
    power_w : array-like or scalar
        Power in watts.
    from_convention : str
        Convention of the input power.
    to_convention : str
        Desired output convention.

    Returns
    -------
    array-like or scalar
        Converted power. If both conventions match, the original object is
        returned unchanged. Converted lists and tuples become NumPy arrays.

    Raises
    ------
    ValueError
        If either convention is unknown.
    """
    source = resolve_launch_power_convention(from_convention)
    target = resolve_launch_power_convention(to_convention)
    if source == target:
        return power_w
    # Scale total to per-polarization power by 0.5, or by 2.0 in reverse.
    factor = launch_power_scale(target) / launch_power_scale(source)
    if isinstance(power_w, np.ndarray) or np.isscalar(power_w):
        # Preserve NumPy array and scalar behavior when scaling.
        return cast("Any", power_w) * factor
    # Convert sequences before multiplying by a floating-point scale.
    return np.asarray(power_w, dtype=np.float64) * factor
