r"""频段归属与逐波段发射功率（**WP-O3 只摘 flexgrid 接线片段**）。

⚠️ 本文件**不是** `wp/fg-uwb-fid` 的 `core/bands.py` 整文件搬运
------------------------------------------------------------------------------

`wp/fg-uwb-fid` 的 `core/bands.py` 共 728 行，其中约 430 行是该分支 **FU-2 刀**
（S+C+L+U 波段定义与跨距口径）的自有资产：模块级 S/C/L/U 论证 docstring、
`hz_to_nm`、ITU-T G.694.1 与"假设 U 波段"的波长常数、`U_BAND_ASSUMPTIONS`、
`_u_spec` 与 `CL/CLS/CLU/SCLU_BAND_SPECS` 四张预置波段表、
`BandPlan.from_channels`、`noise_figure_db_per_element`、
`launch_power_profile_w` / `per_band_launch_power_w`（逐**信道**口径）、
`guard_report`（保护间隔报告）、`sclu_span_report`（跨距口径报告）。

按 WP-O3 §2 步骤 0 的要求（"**只搬 flexgrid 接线**，其余不搬"），上述内容
**全部未移植**——它们既不是 flexgrid 接线所需，又会连带引入"U 波段噪声系数
无出处"这类与本 WP 无关的假设。逐片段裁定见
``results/ode_dev_flexgrid_sync.json``，摘要：

=================  =======================================================
移植               flexgrid 接线所需（见下）
不移植             FU-2 的 S/C/L/U 波段资产与报告逻辑
=================  =======================================================

本文件被移植的全部内容都是 **J7 = engineering**（纯数据结构与索引/校验逻辑，
不含任何 ASE/NLI/SRS 公式），因此不触及 J5。

移植了哪三个 flexgrid 接入点（WP-O3 §0 定位）
-------------------------------------------
1. :meth:`BandPlan.from_connections` —— 按**连接中心频率**把每条连接归入一个频段；
2. :meth:`BandPlan.assert_connections_single_band` —— 拒绝横跨频段边界的宽连接，
   而不是静默按中心频率归一段（那样会给出一个物理上不唯一的逐波段功率）；
3. :meth:`BandPlan.apply_launch_power` —— 把逐波段发射功率写回连接。
   :class:`~network_nlo_eval.core.flexgrid.ConnectionSpec` 是 frozen dataclass，
   故返回的是**替换了 ``launch_power_w`` 的新对象**列表，原列表不被修改。

为什么 BandSpec / nm_to_hz 也必须一起搬
---------------------------------------
:meth:`BandPlan.from_connections` 的第二个参数是 ``Sequence[BandSpec]``，
而 :attr:`BandSpec.low_hz` / :attr:`BandSpec.high_hz` 由 :func:`nm_to_hz`
换算。这三者构成**最小闭包**：只搬这三个方法而留下 ``BandSpec`` 不可用。

⚠️ 因此 `BandSpec` 是**通用**的边界+标注 dataclass，**不含**任何波段的具体
取值。C/L/S/U 的边界数值由调用方传入——本仓不预置任何波段表
（这正是与 fu4 的有意差异，见上表"不移植"栏）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import ConnectionSpec
from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat, NDArrayInt

__all__ = [
    "nm_to_hz",
    "BandSpec",
    "BandPlan",
]

_NM = 1e-9
_THz = 1e12


def nm_to_hz(wavelength_nm: float) -> float:
    """波长 (nm) -> 频率 (Hz)。

    Args:
        wavelength_nm: 波长 (nm)，须 > 0。

    Returns:
    -------
        频率 (Hz)。

    Raises:
    ------
        ValueError: ``wavelength_nm <= 0``。
    """
    if wavelength_nm <= 0:
        raise ValueError(f"wavelength_nm must be positive, got {wavelength_nm}.")
    return C_LIGHT / (wavelength_nm * _NM)


@dataclass(frozen=True)
class BandSpec:
    """一个频段的边界、放大器标注与发射功率。

    波长是**真源**：``low_nm`` < ``high_nm``（**短**波长在前、**长**波长在后），
    频率边界由其换算而来且**自动反向**（波长越长频率越低），所以
    :attr:`low_hz` < :attr:`high_hz`。

    ⚠️ 命名约定：``low_nm`` / ``high_nm`` 指**波长数值**的大小，不是频率大小。
    例：L 波段 ``BandSpec("L", 1565.0, 1625.0)`` —— 1565 nm 比 1625 nm **短**，
    故 ``low_nm = 1565``、``high_nm = 1625``；换算后 ``low_hz ≈ 184.49 THz``、
    ``high_hz ≈ 191.55 THz``。

    频率区间语义是**左闭右开** ``[low_hz, high_hz)``，与
    :class:`~network_nlo_eval.core.flexgrid.FlexibleGrid` 的半开区间约定一致。
    这样首尾相接的频段能无缝平铺而不产生双重归属。

    Attributes
    ----------
    name:
        频段名（如 ``"C"`` / ``"L"``）。
    low_nm:
        短波段边缘 (nm)，数值较小，频率**最高**。
    high_nm:
        长波段边缘 (nm)，数值较大，频率**最低**。
    amplifier:
        放大器类型标注。**不参与本仓任何数值计算**，仅作记录。
    noise_figure_db:
        本频段的 EDFA/TDFA 噪声系数 (dB)。**仅作记录**——本仓的噪声系数真源是
        :class:`~network_nlo_eval.network.elements.EDFAConfig`。
    launch_power_w:
        本频段的**逐波段发射功率** (W，双偏振合计)。``None`` 表示"未设定"，
        此时 :meth:`BandPlan.apply_launch_power` 会拒绝构造，而不是静默填 0
        （0 在既有语义里表示"未占用"）。
    source:
        边界出处标注。
    """

    name: str
    low_nm: float
    high_nm: float
    amplifier: str = "EDFA"
    noise_figure_db: float = 5.0
    launch_power_w: float | None = None
    source: str = "ITU-T G.694.1"

    def __post_init__(self) -> None:
        """强制边界不变式。"""
        if not self.name:
            raise ValueError("BandSpec.name must be a non-empty string.")
        if self.low_nm <= 0 or self.high_nm <= 0:
            raise ValueError(
                f"Band {self.name!r} wavelengths must be positive, "
                f"got [{self.low_nm}, {self.high_nm}] nm."
            )
        if self.low_nm >= self.high_nm:
            raise ValueError(
                f"Band {self.name!r} must satisfy low_nm < high_nm (low_nm is the "
                f"SHORTER wavelength, hence the HIGHER frequency), "
                f"got low_nm={self.low_nm} high_nm={self.high_nm}."
            )
        if self.launch_power_w is not None and self.launch_power_w < 0:
            raise ValueError(
                f"Band {self.name!r} launch_power_w must be non-negative or None, "
                f"got {self.launch_power_w}."
            )

    @property
    def low_hz(self) -> float:
        """频率下边缘 (Hz) = 由**长**波长端（``high_nm``）换算。"""
        return nm_to_hz(self.high_nm)

    @property
    def high_hz(self) -> float:
        """频率上边缘 (Hz) = 由**短**波长端（``low_nm``）换算。"""
        return nm_to_hz(self.low_nm)

    @property
    def span_hz(self) -> float:
        """频段频率跨度 (Hz)。"""
        return self.high_hz - self.low_hz

    def contains_hz(self, frequency_hz: float) -> bool:
        """频率是否落在本频段（**左闭右开** ``[low_hz, high_hz)``）。

        Args:
            frequency_hz: 待判频率 (Hz)。

        Returns:
        -------
            落在本频段内为 True；恰好等于上边缘时为 False（归更高频段所有）。
        """
        return self.low_hz <= frequency_hz < self.high_hz

    def contains_hz_array(self, frequencies_hz: NDArrayFloat) -> NDArrayBool:
        """向量版 :meth:`contains_hz`，掩码语义完全相同（左闭右开）。

        Args:
            frequencies_hz: 待判频率数组 (Hz)。

        Returns:
        -------
            与输入同形的 bool 掩码。
        """
        query = np.asarray(frequencies_hz, dtype=np.float64)
        return (query >= self.low_hz) & (query < self.high_hz)

    def as_dict(self) -> dict[str, object]:
        """机器可读描述，随证据 JSON 落盘。"""
        return {
            "name": self.name,
            "wavelength_nm": [self.low_nm, self.high_nm],
            "frequency_hz": [self.low_hz, self.high_hz],
            "frequency_thz": [self.low_hz / _THz, self.high_hz / _THz],
            "span_thz": self.span_hz / _THz,
            "amplifier": self.amplifier,
            "noise_figure_db": self.noise_figure_db,
            "launch_power_w": self.launch_power_w,
            "source": self.source,
        }


class BandPlan:
    """把一段频谱（逐连接粒度）切分到若干频段，并给出逐波段发射功率。

    校验沿用了 RL 侧 ``BandPlan`` 已经踩过的几条坑：非空、两两不重叠、
    **全覆盖**、每段至少一个元素。理由同样——任一条失守都会让上层拿到
    静默错误的频段归属，而不是一个显式异常。

    差异一处：频率区间是**左闭右开**，故相邻频段可无缝平铺。

    Raises
    ------
    ValueError:
        频段集合为空；存在重叠（含非相邻对）；有元素未被覆盖；有频段内无元素；
        频率数组含非正数。
    """

    def __init__(self, frequencies_hz: NDArrayFloat, specs: Sequence[BandSpec]) -> None:
        """按 ``specs`` 顺序为每个频率元素分配频段序号。

        Args:
            frequencies_hz: 待切分的频率元素 (Hz)，一维、严格递增或至少互异均可
                （本类只要求覆盖判定有意义，不要求递增）。
            specs: 频段定义，须按**频率从低到高**排列。

        Raises
        ------
            ValueError: 见类级 Raises；另加形状/正值校验。
        """
        if not specs:
            raise ValueError("BandPlan requires at least one band specification.")
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError(
                f"frequencies_hz must be a non-empty 1-D array, got shape {frequencies.shape}."
            )
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("frequencies_hz must be finite and strictly positive.")

        # 全对重叠检查，不能只看相邻：非相邻的一对若重叠，相邻比较一个都发现不了，
        # 而它会让一个元素被静默夺走。
        for i, left in enumerate(specs):
            for right in specs[i + 1:]:
                if left.high_hz > right.low_hz and right.high_hz > left.low_hz:
                    raise ValueError(
                        f"Bands {left.name!r} and {right.name!r} overlap "
                        f"({left.high_hz:.6e} Hz > {right.low_hz:.6e} Hz)."
                    )

        band_index = np.full(frequencies.size, -1, dtype=np.int_)
        for index, spec in enumerate(specs):
            band_index[spec.contains_hz_array(frequencies)] = index

        uncovered = int(np.count_nonzero(band_index < 0))
        if uncovered:
            raise ValueError(
                f"{uncovered} of {frequencies.size} frequency elements are uncovered by "
                f"{[spec.name for spec in specs]}. The grid must lie inside the union of "
                "the declared bands; widen the band set or narrow the grid."
            )
        empty = [spec.name for index, spec in enumerate(specs) if not np.any(band_index == index)]
        if empty:
            counts = {
                spec.name: int(np.count_nonzero(band_index == index))
                for index, spec in enumerate(specs)
            }
            raise ValueError(
                f"Bands {empty} contain 0 of {frequencies.size} frequency elements "
                f"(per-band counts: {counts}). Declaring a band the grid never reaches "
                "is always a configuration error."
            )

        self.frequencies_hz = frequencies
        self.specs = tuple(specs)
        self.band_index: NDArrayInt = band_index
        self._band_elements: tuple[NDArrayInt, ...] = tuple(
            np.flatnonzero(band_index == index).astype(np.int_)
            for index in range(len(specs))
        )

    # --- 构造 -------------------------------------------------------------
    @classmethod
    def from_connections(cls, connections: Sequence[ConnectionSpec], specs: Sequence[BandSpec]) -> BandPlan:
        """由灵活栅格连接构造（逐**连接**粒度）。★ flexgrid 接线

        每个连接的归属按其**中心频率**判定；跨边界的宽连接请先调用
        :meth:`assert_connections_single_band` 得到显式拒绝。

        Args:
            connections: 连接列表。
            specs: 频段定义，按频率从低到高。

        Returns:
        -------
            每条连接归属一个频段的 :class:`BandPlan`。

        Raises:
        ------
            ValueError: ``connections`` 为空。
        """
        if not connections:
            raise ValueError("BandPlan.from_connections requires at least one connection.")
        return cls(
            np.array([conn.centre_hz for conn in connections], dtype=np.float64),
            specs,
        )

    # --- 归属查询 ---------------------------------------------------------
    @property
    def num_bands(self) -> int:
        """频段数量。"""
        return len(self.specs)

    @property
    def num_elements(self) -> int:
        """被切分的频率元素数量（逐连接模式下 = 连接数）。"""
        return int(self.frequencies_hz.size)

    @property
    def band_names(self) -> tuple[str, ...]:
        """频段名称，按频率从低到高。"""
        return tuple(spec.name for spec in self.specs)

    def band_element_indices(self, band: int) -> NDArrayInt:
        """返回指定频段内的元素索引。

        Args:
            band: 频段序号。

        Returns:
        -------
            该频段内的元素索引数组，升序。

        Raises:
        ------
            IndexError: ``band`` 越界。
        """
        if isinstance(band, bool) or not isinstance(band, int):
            raise TypeError(f"band must be an integer, got {type(band).__name__}.")
        if not 0 <= band < self.num_bands:
            raise IndexError(f"Band index {band} out of range for {self.num_bands} bands.")
        return self._band_elements[band]

    def band_of(self, element_idx: int) -> int:
        """返回某元素所属的频段序号。

        Args:
            element_idx: 元素索引。

        Returns:
        -------
            频段序号。

        Raises:
        ------
            IndexError: ``element_idx`` 越界。
        """
        if isinstance(element_idx, bool) or not isinstance(element_idx, int):
            raise TypeError(f"element_idx must be an integer, got {type(element_idx).__name__}.")
        if not 0 <= element_idx < self.num_elements:
            raise IndexError(f"Element index {element_idx} out of range for {self.num_elements}.")
        return int(self.band_index[element_idx])

    def element_count(self, band: int) -> int:
        """返回某频段内的元素数量。

        Args:
            band: 频段序号。

        Returns:
        -------
            元素数量 (int)。

        Raises:
        ------
            IndexError: ``band`` 越界。
        """
        return int(self.band_element_indices(band).size)

    # --- 逐连接发射功率 ★ --------------------------------------------------
    def assert_connections_single_band(self, connections: Sequence[ConnectionSpec]) -> None:
        """拒绝任何横跨频段边界的连接。★ flexgrid 接线

        宽连接可能整体压在一个频段边界上。届时"它属于哪个频段"没有唯一答案——
        按中心频率归属会静默给出错误功率。

        Args:
            connections: 待检查的连接列表。

        Raises:
        ------
            ValueError: 某条连接的占用带跨越两个及以上频段。消息含连接 id、
                占用带 (THz) 与它跨越的频段名。
        """
        for conn in connections:
            half = conn.n_slots * conn.grid.slot_width_hz / 2
            low_hz, high_hz = conn.centre_hz - half, conn.centre_hz + half
            hit = [spec.name for spec in self.specs if low_hz < spec.high_hz and spec.low_hz < high_hz]
            if len(hit) > 1:
                raise ValueError(
                    f"Connection {conn.connection_id} occupies "
                    f"[{low_hz / _THz:.6f}, {high_hz / _THz:.6f}] THz, which straddles "
                    f"band boundary between {hit}. Per-band launch power is ambiguous for it; "
                    "split the connection or drop the crossing band from the plan."
                )

    def apply_launch_power(
        self, connections: Sequence[ConnectionSpec], *, allow_crossing: bool = False
    ) -> list[ConnectionSpec]:
        """把逐波段发射功率写入连接，返回新的连接列表。★ flexgrid 接线

        :class:`~network_nlo_eval.core.flexgrid.ConnectionSpec` 是 frozen dataclass，
        因此返回的是**替换了 ``launch_power_w`` 的新对象**，原列表不被修改。

        Args:
            connections: 原连接列表。
            allow_crossing: 设为 True 时跳过边界检查（调用方自负责）。

        Returns:
        -------
            ``launch_power_w`` 已换成所属频段功率的连接列表。

        Raises:
        ------
            ValueError: 有频段未设定功率；或（``allow_crossing=False`` 时）
                有连接横跨频段边界。
        """
        if not allow_crossing:
            self.assert_connections_single_band(connections)
        unset = [spec.name for spec in self.specs if spec.launch_power_w is None]
        if unset:
            raise ValueError(
                f"Bands {unset} have launch_power_w=None; per-band launch power cannot "
                "be applied to connections until every band has one."
            )
        return [
            replace(conn, launch_power_w=float(self.specs[self.band_of(i)].launch_power_w))
            for i, conn in enumerate(connections)
        ]

    # --- 报告 -------------------------------------------------------------
    def as_dict(self) -> dict[str, object]:
        """机器可读描述，随证据 JSON 落盘。

        ⚠️ 与 fu4 的 ``as_dict`` 有意不同：fu4 版含 ``guards`` 键（来自未移植的
        :meth:`guard_report`，属 FU-2 保护间隔报告）。本版不含该键。
        """
        return {
            "band_order_low_to_high_frequency": self.band_names,
            "num_elements": self.num_elements,
            "num_bands": self.num_bands,
            "elements_per_band": {
                spec.name: int(self._band_elements[index].size)
                for index, spec in enumerate(self.specs)
            },
            "bands": [spec.as_dict() for spec in self.specs],
        }
