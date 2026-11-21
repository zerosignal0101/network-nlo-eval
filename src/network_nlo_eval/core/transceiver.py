"""Transceiver modes and receiver budgets for optical service admission."""

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

OSNR_REFERENCE_BANDWIDTH_HZ = 12.5e9

_Nonnegative = Annotated[float, Field(ge=0)]


class ReceiverPenaltyCurve(BaseModel):
    """Piecewise-linear SNR penalty versus an impairment magnitude.

    Points are pairs of impairment magnitude and penalty in dB. Values below
    the first point use its penalty. Values above the last point are infeasible.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    points: tuple[tuple[_Nonnegative, _Nonnegative], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_points(self) -> "ReceiverPenaltyCurve":
        if any(
            left[0] >= right[0]
            for left, right in zip(self.points, self.points[1:], strict=False)
        ):
            msg = "Receiver penalty curve magnitudes must be strictly increasing."
            raise ValueError(msg)
        return self

    def penalty_db(self, magnitude: float) -> float:
        """Return the interpolated penalty, or infinity beyond the curve."""
        if not math.isfinite(magnitude) or magnitude > self.points[-1][0]:
            return math.inf
        index = bisect_right(self.points, magnitude, key=lambda point: point[0])
        if index == 0:
            return self.points[0][1]
        left, penalty = self.points[index - 1]
        if index == len(self.points) or left == magnitude:
            return penalty
        right, next_penalty = self.points[index]
        return penalty + (next_penalty - penalty) * (magnitude - left) / (right - left)


class TransceiverMode(BaseModel):
    """Opt-in transmitter geometry and receiver requirements for one carrier.

    GSNR uses the physical evaluator's channel bandwidth. OSNR requirements
    and transmitter OSNR use ``osnr_reference_bandwidth_hz``. Receiver power
    bounds refer to total signal power after the destination drop port.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    symbol_rate_hz: float = Field(gt=0)
    roll_off: float = Field(0, ge=0, lt=1)
    min_slot_width_hz: float = Field(0, ge=0)
    min_gsnr_db: float
    min_osnr_db: float | None = None
    tx_osnr_db: float | None = None
    osnr_reference_bandwidth_hz: float = Field(OSNR_REFERENCE_BANDWIDTH_HZ, gt=0)
    rx_power_min_dbm: float | None = None
    rx_power_max_dbm: float | None = None
    cd_penalty: ReceiverPenaltyCurve | None = Field(
        None, description="Penalty versus absolute chromatic dispersion in ps/nm"
    )
    pmd_penalty: ReceiverPenaltyCurve | None = Field(
        None, description="Penalty versus polarization mode dispersion in ps"
    )
    pdl_penalty: ReceiverPenaltyCurve | None = Field(
        None, description="Penalty versus polarization dependent loss in dB"
    )

    @model_validator(mode="after")
    def _validate_limits(self) -> "TransceiverMode":
        if (
            self.rx_power_min_dbm is not None
            and self.rx_power_max_dbm is not None
            and self.rx_power_min_dbm > self.rx_power_max_dbm
        ):
            msg = "Receiver minimum power must not exceed maximum power."
            raise ValueError(msg)
        if not math.isfinite(self.required_slot_width_hz):
            msg = "The required slot width must be finite."
            raise ValueError(msg)
        return self

    @property
    def required_slot_width_hz(self) -> float:
        """Return the minimum reservation for the signal and any guard band."""
        return max(self.symbol_rate_hz * (1 + self.roll_off), self.min_slot_width_hz)


@dataclass(frozen=True, slots=True)
class ReceiverFeasibility:
    """Receiver metrics, impairments, and margins for a tested service.

    Line SNR includes the validity policy. GSNR includes transmitter noise,
    ``effective_gsnr_db`` also includes receiver penalties and the validity
    policy. OSNR includes line ASE and transmitter noise at the reference
    bandwidth. Power margins describe the closest configured receiver bound.
    Unconfigured limits contribute infinite margin.
    """

    mode_name: str
    line_snr_db: float
    osnr_db: float
    gsnr_db: float
    effective_gsnr_db: float
    rx_power_dbm: float
    cd_ps_nm: float
    pmd_ps: float
    pdl_db: float
    penalty_db: float
    noise_bandwidth_hz: float
    osnr_reference_bandwidth_hz: float
    line_snr_margin_db: float
    osnr_margin_db: float
    gsnr_margin_db: float
    rx_power_margin_db: float

    @property
    def margin_db(self) -> float:
        """Return the smallest configured admission margin."""
        return min(
            self.line_snr_margin_db,
            self.gsnr_margin_db,
            self.osnr_margin_db,
            self.rx_power_margin_db,
        )

    @property
    def feasible(self) -> bool:
        """Return whether all receiver requirements are satisfied."""
        return self.margin_db >= 0


def _snr_db(signal: float, noise: float) -> float:
    if not (0 < signal < math.inf and noise >= 0):
        return -math.inf
    return 10 * (math.log10(signal) - math.log10(noise)) if noise > 0 else math.inf


def _add_noise_snr_db(line_db: float, tx_db: float) -> float:
    weaker = min(line_db, tx_db)
    if not math.isfinite(weaker):
        return weaker
    return weaker - 10 * math.log10(1 + 10 ** (-abs(line_db - tx_db) / 10))


def evaluate_receiver(
    mode: TransceiverMode,
    *,
    signal_power_w: float,
    ase_noise_w: float,
    noise_power_w: float,
    noise_bandwidth_hz: float,
    line_snr_db: float,
    snr_requirement_db: float,
    cd_ps_nm: float,
    pmd_ps: float,
    pdl_db: float,
    validity_adjustment_db: float = 0.0,
) -> ReceiverFeasibility:
    """Evaluate a mode using signal and noise powers in total watts.

    ``noise_power_w`` includes line ASE and all modeled nonlinear noise in
    the evaluator's bandwidth. Transmitter OSNR is treated as spectrally flat
    additive noise at the receiver. A validity adjustment of negative infinity
    rejects the receiver even when its raw GSNR is infinite.
    """
    reference_shift_db = 10 * (
        math.log10(mode.osnr_reference_bandwidth_hz) - math.log10(noise_bandwidth_hz)
    )
    osnr = _snr_db(signal_power_w, ase_noise_w) - reference_shift_db
    gsnr = _snr_db(signal_power_w, noise_power_w)
    if mode.tx_osnr_db is not None:
        osnr = _add_noise_snr_db(osnr, mode.tx_osnr_db)
        gsnr = _add_noise_snr_db(gsnr, mode.tx_osnr_db + reference_shift_db)
    rx_power = 10 * math.log10(signal_power_w) + 30 if signal_power_w > 0 else -math.inf
    penalty = sum(
        curve.penalty_db(magnitude)
        for curve, magnitude in (
            (mode.cd_penalty, abs(cd_ps_nm)),
            (mode.pmd_penalty, pmd_ps),
            (mode.pdl_penalty, pdl_db),
        )
        if curve is not None
    )
    effective_gsnr = (
        gsnr - penalty + validity_adjustment_db
        if penalty < math.inf and validity_adjustment_db > -math.inf
        else -math.inf
    )
    line_margin = -math.inf
    if line_snr_db > -math.inf and not math.isnan(snr_requirement_db):
        line_margin = (
            0.0
            if line_snr_db == snr_requirement_db
            else line_snr_db - snr_requirement_db
        )
    return ReceiverFeasibility(
        mode_name=mode.name,
        line_snr_db=line_snr_db,
        osnr_db=osnr,
        gsnr_db=gsnr,
        effective_gsnr_db=effective_gsnr,
        rx_power_dbm=rx_power,
        cd_ps_nm=cd_ps_nm,
        pmd_ps=pmd_ps,
        pdl_db=pdl_db,
        penalty_db=penalty,
        noise_bandwidth_hz=noise_bandwidth_hz,
        osnr_reference_bandwidth_hz=mode.osnr_reference_bandwidth_hz,
        line_snr_margin_db=line_margin,
        osnr_margin_db=(
            osnr - mode.min_osnr_db if mode.min_osnr_db is not None else math.inf
        ),
        gsnr_margin_db=effective_gsnr - mode.min_gsnr_db,
        rx_power_margin_db=min(
            rx_power - mode.rx_power_min_dbm
            if mode.rx_power_min_dbm is not None
            else math.inf,
            mode.rx_power_max_dbm - rx_power
            if mode.rx_power_max_dbm is not None
            else math.inf,
        ),
    )
