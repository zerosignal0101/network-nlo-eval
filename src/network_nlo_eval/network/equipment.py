"""Named equipment settings resolved during topology construction."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

_EquipmentKind = Literal["fiber", "edfa", "roadm"]
_ProfileName = Annotated[str, Field(min_length=1)]
_ELEMENT_TYPES: dict[_EquipmentKind, type[BaseModel]] = {
    "fiber": FiberSpanConfig,
    "edfa": EDFAConfig,
    "roadm": ROADMConfig,
}


class EquipmentProfiles(BaseModel):
    """Reusable partial settings for fibers, amplifiers, and ROADMs.

    Each entry uses fields of the corresponding element configuration.
    Named settings are merged with placement overrides before the complete
    configuration is validated, so values such as fiber length can be supplied
    at the installation site.
    """

    model_config = ConfigDict(extra="forbid")

    fiber: dict[_ProfileName, dict[str, Any]] = Field(default_factory=dict)
    edfa: dict[_ProfileName, dict[str, Any]] = Field(default_factory=dict)
    roadm: dict[_ProfileName, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_profile_fields(self) -> "EquipmentProfiles":
        for kind, element in _ELEMENT_TYPES.items():
            profiles: dict[str, dict[str, Any]] = getattr(self, kind)
            for name, fields in profiles.items():
                unknown = fields.keys() - element.model_fields.keys()
                if unknown:
                    msg = f"Unknown fields for {kind} profile {name!r}: {sorted(unknown)}."
                    raise ValueError(msg)
        return self

    def resolve(
        self,
        kind: _EquipmentKind,
        specification: str | Mapping[str, Any],
        *,
        placement: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve a name or inline settings, applying local overrides last."""
        local = {} if placement is None else dict(placement)
        name: object
        if isinstance(specification, str):
            name = specification
            overrides: dict[str, Any] = {}
        else:
            overrides = dict(specification)
            if "profile" not in overrides:
                return local | overrides
            name = overrides.pop("profile")

        profiles: dict[str, dict[str, Any]] = getattr(self, kind)
        if not isinstance(name, str) or name not in profiles:
            msg = f"Unknown {kind} equipment profile {name!r}."
            raise ValueError(msg)
        unknown = overrides.keys() - _ELEMENT_TYPES[kind].model_fields.keys()
        if unknown:
            msg = f"Unknown overrides for {kind} profile {name!r}: {sorted(unknown)}."
            raise ValueError(msg)
        return profiles[name] | local | overrides
