"""Compare statistical ISRS-GN and numerical waveform QoT on one topology route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat, NodeID
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, PathQoTResult

if TYPE_CHECKING:
    from network_nlo_eval.network.topology import NetworkTopology
    from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator


@dataclass(frozen=True)
class QoTComparisonResult:
    """同一路径上两个物理层引擎的可序列化比较结果."""

    path: list[NodeID]
    statistical: PathQoTResult
    waveform_metrics: dict[str, NDArrayFloat] | None
    snr_delta_db: NDArrayFloat | None

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典."""

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
            waveform = {name: values(array) for name, array in self.waveform_metrics.items()}
        return {
            "path": self.path,
            "statistical": statistical,
            "waveform": waveform,
            "snr_delta_db": None if self.snr_delta_db is None else values(self.snr_delta_db),
        }


class TopologyQoTComparator:
    """Run both physical-layer engines from one topology-derived route contract."""

    def __init__(
        self,
        topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        gn_evaluator: MultiBandISRSGN | None = None,
        waveform_evaluator: PyNLOQoTEvaluator | None = None,
    ) -> None:
        """初始化比较器；波形评估器保持可选以支持快速批量仿真."""
        self.topology = topology
        self.spectrum_grid = spectrum_grid
        self.gn_evaluator = gn_evaluator or MultiBandISRSGN(spectrum_grid, topology.get_default_fiber_config())
        self.waveform_evaluator = waveform_evaluator

    def evaluate(
        self,
        path: list[NodeID],
        launch_power_profile_w: NDArrayFloat,
        *,
        run_waveform: bool = False,
        num_symbols: int = 1024,
    ) -> QoTComparisonResult:
        """评估路径并返回统计值、波形值及逐信道 SNR 差值."""
        physical_path = self.topology.build_optical_path(path)
        statistical = self.gn_evaluator.evaluate_path(physical_path, launch_power_profile_w)
        waveform_metrics = None
        delta = None
        if run_waveform:
            if self.waveform_evaluator is None:
                from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator

                self.waveform_evaluator = PyNLOQoTEvaluator(
                    self.spectrum_grid, symbol_rate_hz=self.gn_evaluator.symbol_rate_hz
                )
            _, waveform_metrics = self.waveform_evaluator.evaluate_path(
                physical_path, launch_power_profile_w, num_symbols=num_symbols
            )
            delta = waveform_metrics["snr"] - statistical.snr_db
        return QoTComparisonResult(path.copy(), statistical, waveform_metrics, delta)
