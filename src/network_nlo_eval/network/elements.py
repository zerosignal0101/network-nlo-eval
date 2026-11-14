"""Pydantic configuration models for physical network elements."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from network_nlo_eval.core.types import NDArrayFloat


class FiberSpanConfig(BaseModel):
    """Static fiber-link configuration.

    Reference values are corrected for the actual wavelength by the models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    length_km: float = Field(..., gt=0, description="Fiber link length (km)")
    max_span_length_km: float = Field(
        100.0, gt=0, description="Maximum amplified span length (km)"
    )

    # Attenuation
    attenuation_db_km_ref: float = Field(
        0.2, gt=0, description="Attenuation at the reference wavelength (dB/km)"
    )
    use_attenuation_polynomial: bool = Field(
        False, description="Use the alpha0/1/2 wavelength attenuation curve"
    )
    # Polynomial defaults follow the original GSNR calculator reference.
    attenuation_alpha2: float = Field(
        3.7685e-6, description="Quadratic attenuation-curve coefficient"
    )
    attenuation_alpha1: float = Field(
        -7.3764e-5, description="Linear attenuation-curve coefficient"
    )
    attenuation_alpha0: float = Field(
        0.162, description="Constant attenuation-curve coefficient"
    )
    reference_wavelength_nm: float = Field(
        1550.0, description="Attenuation-curve reference wavelength (nm)"
    )

    # Dispersion
    dispersion_parameter_d: float = Field(
        17.0,
        description="Dispersion parameter D at the reference wavelength (ps/(nm*km))",
    )
    dispersion_slope_s: float = Field(
        0.06,
        description="Dispersion slope S at the reference wavelength (ps/(nm^2*km))",
    )

    # Nonlinearity
    nonlinear_index_n2: float = Field(
        2.6e-20, ge=0, description="Fiber nonlinear refractive index n2 (m^2/W)"
    )
    effective_area_um2_ref: float = Field(
        80.0,
        gt=0,
        description="Effective mode area A_eff at the reference wavelength (um^2)",
    )
    aeff_slope_um2_nm: float = Field(
        0.05, description="Effective mode area slope with wavelength (um^2/nm)"
    )
    raman_pumps: list["RamanPumpConfig"] = Field(
        default_factory=list, description="Distributed Raman pumps on this link"
    )
    lumped_loss: list[tuple[float, float]] = Field(
        default_factory=list,
        description="In-span lumped losses (position_m, loss_db), such as connectors, splices, and attenuators",
    )

    @model_validator(mode="after")
    def validate_lumped_loss(self) -> "FiberSpanConfig":
        """Validate the positions and magnitudes of in-span losses.

        Positions must be finite and strictly inside the span. Losses must
        be finite and non-negative. Multiple losses at one position are
        allowed and are combined by the kernel in dB.

        Raises
        ------
        ValueError
            If a position or loss is non-finite or outside its valid range.
        """
        length_m = float(self.length_km) * 1e3
        for position_m, loss_db in self.lumped_loss:
            position = float(position_m)
            loss = float(loss_db)
            if not np.isfinite(position):
                msg = f"lumped loss position must be finite, got {position_m!r}."
                raise ValueError(msg)
            if not np.isfinite(loss):
                msg = f"lumped loss must be finite, got {loss_db!r}."
                raise ValueError(msg)
            if not 0.0 < position < length_m:
                msg = f"lumped loss position must lie strictly inside (0, {length_m}) m, got {position}."
                raise ValueError(msg)
            if loss < 0.0:
                msg = f"lumped loss must be non-negative, got {loss} dB."
                raise ValueError(msg)
        return self


class EDFAConfig(BaseModel):
    """Erbium-doped fiber amplifier (EDFA) configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_gain_db: float | None = Field(
        None, ge=0, description="Fixed gain; None compensates span loss automatically"
    )
    noise_figure_db: float | NDArrayFloat = Field(
        5.0, description="Amplifier noise figure (dB)"
    )
    # Optional per-channel gain ripple.
    gain_ripple_db: NDArrayFloat | None = Field(
        None, description="Gain ripple (dB), one value per channel"
    )

    @model_validator(mode="after")
    def validate_noise_figure(self) -> "EDFAConfig":
        """Require finite noise figure and ripple, and non-negative noise figure."""
        noise_figure = np.asarray(self.noise_figure_db)
        if np.any(~np.isfinite(noise_figure)) or np.any(noise_figure < 0):
            msg = "EDFA noise figure must contain finite non-negative values."
            raise ValueError(msg)
        if self.gain_ripple_db is not None and np.any(
            ~np.isfinite(self.gain_ripple_db)
        ):
            msg = "EDFA gain ripple must contain finite values."
            raise ValueError(msg)
        return self


class RamanPumpConfig(BaseModel):
    """Distributed Raman pump configuration corresponding to a waveform PumpSpec."""

    wavelength_nm: float = Field(..., gt=0, description="Pump wavelength (nm)")
    power_w: float = Field(..., ge=0, description="Pump power (W)")
    direction: Literal["forward", "backward"] = Field(
        "backward", description="Pump direction"
    )
    attenuation_db_per_km: float = Field(
        0.25, ge=0, description="Pump attenuation (dB/km)"
    )


class ROADMConfig(BaseModel):
    """ROADM loss, WSS transfer function, and amplifier configuration."""

    insertion_loss_db: float = Field(
        5.0, ge=0, description="WSS and connector insertion loss (dB)"
    )
    filtering_penalty_db: float = Field(
        0.2,
        ge=0,
        description="Cascaded filtering SNR penalty for statistical models (dB)",
    )
    passband_hz: float | None = Field(
        None, gt=0, description="WSS 3-dB passband; None uses channel spacing"
    )
    filter_order: int = Field(4, ge=1, description="Super-Gaussian WSS order")
    express_attenuation_db: float = Field(
        0.0, ge=0, description="Additional express-port attenuation (dB)"
    )
    add_drop_loss_db: float = Field(
        1.0, ge=0, description="Additional local add/drop port loss (dB)"
    )
    booster_noise_figure_db: float = Field(
        5.0,
        ge=0,
        description="Noise figure of the ROADM loss compensating booster (dB)",
    )
    equalize_output_power: bool = Field(
        True, description="Restore per-channel launch power with a booster"
    )
    blocked_channels: list[int] = Field(
        default_factory=list,
        description="Statically blocked channel indices; expanded into slots on a flexible grid",
    )
    blocked_slots: list[int] = Field(
        default_factory=list,
        description="Statically blocked slot indices (authoritative for flexible grids)",
    )

    @model_validator(mode="after")
    def validate_blocked_channels(self) -> "ROADMConfig":
        """Reject duplicate or negative channel and slot indices."""
        if any(index < 0 for index in self.blocked_channels):
            msg = "ROADM blocked channel indices must be non-negative."
            raise ValueError(msg)
        if len(set(self.blocked_channels)) != len(self.blocked_channels):
            msg = "ROADM blocked channel indices must be unique."
            raise ValueError(msg)
        if any(index < 0 for index in self.blocked_slots):
            msg = "ROADM blocked slot indices must be non-negative."
            raise ValueError(msg)
        if len(set(self.blocked_slots)) != len(self.blocked_slots):
            msg = "ROADM blocked slot indices must be unique."
            raise ValueError(msg)
        return self

    def path_loss_db(
        self, operation: Literal["express", "add", "drop"] = "express"
    ) -> float:
        """Return the deterministic loss for the requested switching operation."""
        port_loss = (
            self.express_attenuation_db
            if operation == "express"
            else self.add_drop_loss_db
        )
        return self.insertion_loss_db + port_loss
