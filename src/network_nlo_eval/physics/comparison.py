"""Compare statistical ISRS-GN and numerical waveform QoT on one topology route."""

# ruff: noqa: PLC0415  # Load the optional waveform engine on demand.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)

if TYPE_CHECKING:
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayFloat, NodeID
    from network_nlo_eval.models.base import PathQoTResult, QoTModel
    from network_nlo_eval.network.topology import NetworkTopology
    from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator


@dataclass(frozen=True)
class QoTComparisonResult:
    """Serializable comparison of two physical-layer engines on one path."""

    path: list[NodeID]
    statistical: PathQoTResult
    waveform_metrics: dict[str, NDArrayFloat] | None
    snr_delta_db: NDArrayFloat | None

    def to_dict(self) -> dict[str, Any]:
        """Convert the comparison to a JSON-compatible dictionary."""

        def values(array: NDArrayFloat) -> list[float | None]:
            return [float(value) if np.isfinite(value) else None for value in array]

        statistical = {
            "signal_power_w": values(self.statistical.signal_power_w),
            "spm_noise_w": values(self.statistical.spm_noise_w),
            "xpm_noise_w": values(self.statistical.xpm_noise_w),
            "ase_noise_w": values(self.statistical.ase_noise_w),
            "snr_db": values(self.statistical.snr_db),
            "span_count": self.statistical.span_count,
            "roadm_count": self.statistical.roadm_count,
        }
        waveform = None
        if self.waveform_metrics is not None:
            waveform = {
                name: values(array) for name, array in self.waveform_metrics.items()
            }
        return {
            "path": self.path,
            "statistical": statistical,
            "waveform": waveform,
            "snr_delta_db": None
            if self.snr_delta_db is None
            else values(self.snr_delta_db),
        }


class TopologyQoTComparator:
    """Run both physical-layer engines from one topology-derived route contract.

    ``launch_power_convention`` declares how ``launch_power_profile_w`` (the
    ``evaluate`` argument) is interpreted.  Each engine is a separate model with
    its own ``launch_power_convention``; the comparator converts the input to
    each coordinate independently, exactly once.  With the default ``"total"``
    on all three sides both conversions return the input object unchanged.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        gn_evaluator: QoTModel | None = None,
        waveform_evaluator: PyNLOQoTEvaluator | None = None,
        *,
        launch_power_convention: str = "total",
    ) -> None:
        """Initialize a topology-based model comparator.

        Parameters
        ----------
        topology : NetworkTopology
            Topology supplying physical path configurations.
        spectrum_grid : SpectrumGrid
            Frequency grid shared by the evaluators.
        gn_evaluator : QoTModel or None, optional
            Statistical evaluator; None creates a total-power ISRS-GN model.
        waveform_evaluator : PyNLOQoTEvaluator or None, optional
            Waveform evaluator; None creates one when first requested.
        launch_power_convention : {"total", "per_pol"}, optional
            Power convention of the input passed to ``evaluate``.

        Raises
        ------
        ValueError
            If the power convention is invalid.
        """
        self.topology = topology
        self.spectrum_grid = spectrum_grid
        self.gn_evaluator = gn_evaluator or MultiBandISRSGN(
            spectrum_grid, topology.get_default_fiber_config()
        )
        self.waveform_evaluator = waveform_evaluator
        self.launch_power_convention = resolve_launch_power_convention(
            launch_power_convention
        )

    def evaluate(
        self,
        path: list[NodeID],
        launch_power_profile_w: NDArrayFloat,
        *,
        run_waveform: bool = False,
        num_symbols: int = 1024,
    ) -> QoTComparisonResult:
        """Evaluate a path with the statistical and optional waveform models.

        Parameters
        ----------
        path : list of int
            Ordered internal node IDs.
        launch_power_profile_w : numpy.ndarray
            Channel launch powers in the comparator's power convention.
        run_waveform : bool, optional
            Whether to run the waveform evaluator.
        num_symbols : int, optional
            Number of symbols used for waveform evaluation.

        Returns
        -------
        QoTComparisonResult
            Statistical results, optional waveform results, and their SNR
            difference. Each evaluator receives power in its own convention.
        """
        physical_path = self.topology.build_optical_path(path)
        stat_power = to_convention(
            launch_power_profile_w,
            self.launch_power_convention,
            str(getattr(self.gn_evaluator, "launch_power_convention", "total")),
        )
        statistical = self.gn_evaluator.evaluate_path(physical_path, stat_power)
        waveform_metrics = None
        delta = None
        if run_waveform:
            if self.waveform_evaluator is None:
                from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator

                # A lazily created waveform engine uses total-power coordinates.
                self.waveform_evaluator = PyNLOQoTEvaluator(
                    self.spectrum_grid,
                    symbol_rate_hz=self.gn_evaluator.symbol_rate_hz,
                    launch_power_convention="total",
                )
            wf_power = to_convention(
                launch_power_profile_w,
                self.launch_power_convention,
                str(
                    getattr(self.waveform_evaluator, "launch_power_convention", "total")
                ),
            )
            _, waveform_metrics = self.waveform_evaluator.evaluate_path(
                physical_path, wf_power, num_symbols=num_symbols
            )
            delta = waveform_metrics["snr"] - statistical.snr_db
        return QoTComparisonResult(path.copy(), statistical, waveform_metrics, delta)
