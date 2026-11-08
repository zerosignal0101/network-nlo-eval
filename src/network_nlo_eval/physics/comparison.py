"""Compare statistical ISRS-GN and numerical waveform QoT on one topology route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat, NodeID
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, PathQoTResult
from network_nlo_eval.models.power_conventions import resolve_launch_power_convention, to_convention

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
        gn_evaluator: MultiBandISRSGN | None = None,
        waveform_evaluator: PyNLOQoTEvaluator | None = None,
        *,
        launch_power_convention: str = "total",
    ) -> None:
        """初始化比较器；波形评估器保持可选以支持快速批量仿真。

        Args:
            topology: 网络拓扑。
            spectrum_grid: 频谱网格。
            gn_evaluator: 统计侧模型；``None`` 时新建一个总功率口径的
                ``MultiBandISRSGN``。
            waveform_evaluator: 波形侧模型；``None`` 时在首次要求波形时惰性创建。
            launch_power_convention: ``evaluate`` 入参的口径，``"total"``（默认）
                或 ``"per_pol"``。

        Raises
        ------
            ValueError: ``launch_power_convention`` 非法。
        """
        self.topology = topology
        self.spectrum_grid = spectrum_grid
        self.gn_evaluator = gn_evaluator or MultiBandISRSGN(spectrum_grid, topology.get_default_fiber_config())
        self.waveform_evaluator = waveform_evaluator
        self.launch_power_convention = resolve_launch_power_convention(launch_power_convention)

    def evaluate(
        self,
        path: list[NodeID],
        launch_power_profile_w: NDArrayFloat,
        *,
        run_waveform: bool = False,
        num_symbols: int = 1024,
    ) -> QoTComparisonResult:
        """评估路径并返回统计值、波形值及逐信道 SNR 差值.

        ``launch_power_profile_w`` 按构造时的口径解释；统计侧与波形侧分别换算到
        各自模型的 ``launch_power_convention``。
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

                # 波形引擎内部坐标是总功率：惰性创建时显式声明，别跟随比较器入参。
                self.waveform_evaluator = PyNLOQoTEvaluator(
                    self.spectrum_grid,
                    symbol_rate_hz=self.gn_evaluator.symbol_rate_hz,
                    launch_power_convention="total",
                )
            wf_power = to_convention(
                launch_power_profile_w,
                self.launch_power_convention,
                str(getattr(self.waveform_evaluator, "launch_power_convention", "total")),
            )
            _, waveform_metrics = self.waveform_evaluator.evaluate_path(
                physical_path, wf_power, num_symbols=num_symbols
            )
            delta = waveform_metrics["snr"] - statistical.snr_db
        return QoTComparisonResult(path.copy(), statistical, waveform_metrics, delta)
