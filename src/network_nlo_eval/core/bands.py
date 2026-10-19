r"""S/C/L/U 波段定义与**逐波段发射功率**口径（FU-2）。

本次合并补记（WP-MERGE-FG-FLEX）
--------------------------------
本文件在 `flexgrid-dev` 上曾是**相位守卫**的载体：模块 docstring 明写"只摘 flexgrid
接线片段"、FU-2 的 S/C/L/U 资产"全部未移植"，`BandPlan.as_dict` 也刻意**不含**
``guards`` 键（因其依赖未移植的 :meth:`BandPlan.guard_report`）。

WP-MERGE-FG-FLEX 的目的正是把这些资产移植进来，上述守卫的前提已消失，故按
"存在 + 只定义一次 + 可达"翻成正向不变量：FU-2 资产（:func:`hz_to_nm`、
:data:`U_BAND_ASSUMPTIONS`、:data:`SCLU_BAND_SPECS`、``BandPlan.from_channels``、
``noise_figure_db_per_element``、``guard_report``、:func:`sclu_span_report`）全部
保留并成为对外可达的正式 API，:meth:`BandPlan.as_dict` 重新输出 ``guards`` 键。
逐片段旧文本 → 新文本见 ``docs/wp_merge_fg_into_flex.md`` V5。

flexgrid 侧的三个接入点（数据模型边界，物理层 ↔ 业务层）
---------------------------------------------------------
1. :meth:`BandPlan.from_connections` —— 按**连接中心频率**把每条连接归入一个频段；
2. :meth:`BandPlan.assert_connections_single_band` —— 拒绝横跨频段边界的宽连接，
   而不是静默按中心频率归一段（那样会给出一个物理上不唯一的逐波段功率）；
3. :meth:`BandPlan.apply_launch_power` —— 把逐波段发射功率写回连接。
   :class:`~network_nlo_eval.core.flexgrid.ConnectionSpec` 是 frozen dataclass，
   故返回的是**替换了 ``launch_power_w`` 的新对象**列表，原列表不被修改。

这三个只描述「一条连接的频谱占位与速率」（物理层）；到达/分配/释放的状态机归业务层
（``traffic.py`` / ``network/slots.py`` / ``rwa/allocators.py``）。两套表示的转换函数是
:func:`network_nlo_eval.core.flexgrid.connection_spec_from_allocation`，禁止双写状态。

为什么 BandSpec / nm_to_hz 构成最小闭包
---------------------------------------
:meth:`BandPlan.from_connections` 的第二个参数是 ``Sequence[BandSpec]``，
而 :attr:`BandSpec.low_hz` / :attr:`BandSpec.high_hz` 由 :func:`nm_to_hz`
换算。这三者构成**最小闭包**，故 ``BandSpec`` 与 ``nm_to_hz`` 在本仓各**只定义一次**
（由 ``tests/test_ode3_flexgrid_sync.py`` 的正向不变量守卫）。

为什么需要这个模块
------------------
本仓（`network-nlo-eval`）此前**没有任何波段概念**：功率口径只有
`ConnectionSpec.launch_power_w`（逐连接）与 `LinkState.launch_power_profile_w`
（逐信道），两者都是**逐元素**的，没有"整段频谱用同一份功率配置"的中间层。
RL 侧 (`multiband-optical-rl`) 虽有 `BandPlan`，但它只到 `CL_BAND_SPECS` /
`CLS_BAND_SPECS` 为止，**没有 U 段**，也没有逐波段功率。

波段边界从哪来（S/C/L 对齐，U 是假设）
------------------------------------
本模块的边界**以波长 (nm) 为真源**再换算成频率，因为 ITU-T G.694.1 的频段表
本身就是按波长发布的。S/C/L 直接采用 G.694.1 的数值，与 RL 侧
`CL_BAND_SPECS` / `CLS_BAND_SPECS` 用的那一组**完全相同**（见下表），未作改动。

============  ============  ============  ====================================
波段          长波段 (nm)   长波段 (nm)   出处
============  ============  ============  ====================================
S             1460          1530          ITU-T G.694.1
C             1530          1565          ITU-T G.694.1（与 RL 侧同值）
L             1565          1625          ITU-T G.694.1（与 RL 侧同值）
U             1660          1770          ★**本刀假设**（见下）
============  ============  ============  ====================================

★ U 波段是假设，不是引用
-------------------------
**ITU-T G.694.1 没有定义 U 波段**，两仓在 FG-2 之前也都没有 U 段资产
（FG-4d 实测）。1660–1770 nm 这个区间是本刀按 brief 指定的**假设**，其
放大器类型与噪声系数同样是假设。三项假设逐条列在 :data:`U_BAND_ASSUMPTIONS`，
并随每个证据 JSON 一起落盘，便于第三方判断。

保护间隔（guard band）
---------------------
G.694.1 的 C/L/S 是**首尾相接**的（S|C 交于 1530 nm，C|L 交于 1565 nm），
所以既有的 C/L 做法就是 **guard = 0**：没有预留保护间隔，相邻频段共用边界。
本模块沿用该做法。

U 段则不同：U 的下沿 1660 nm 距 L 的上沿 1625 nm 有 **35 nm 的真实空隙**，
这不是"没预留 guard"，而是 1660–1770 nm 这个假设区间本身就把 1625–1660 nm
排除在外了。该空隙在频率上约 3.72 THz，是 S+C+L+U **不是连续频带**的直接
原因，也是跨距口径必须写清楚的原因（见 :func:`sclu_span_report`）。

guard 的实际取值不是一句断言而是**算出来的**：见 :meth:`BandPlan.guard_report`，
它对每一对相邻频段报出真实间隔 (nm / THz)。

逐波段功率如何进入 `launch_power_profile_w`
-------------------------------------------
既有实现的粒度（FG-2 之前）：

* **逐信道**：`LinkState.launch_power_profile_w`，长度 = 信道数，由
  `LinkState.allocate(channel_idx, launch_power_w, service_id)` 逐信道写入；
* **逐连接**：灵活栅格下 `ConnectionSpec.launch_power_w`，一条连接一个值，
  `MultiBandISRSGN` 按连接中心频率建节点。

也就是说**既有实现已经能表达逐波段功率**，只是要由调用方自己把"波段 -> 每个
信道/连接填同一个值"这件事重复做一遍，且没有校验（一个横跨 S|C 边界的宽连接
到底算 S 还是 C，没有任何人拦）。本模块补的是这个中间层：

* :meth:`BandPlan.launch_power_profile_w` —— 逐**信道**数组，可直接作为
  `launch_power_profile_w` / `evaluate_path(power_in_w=...)` 的入参；
* :meth:`BandPlan.apply_launch_power` —— 逐**连接**，返回把
  `ConnectionSpec.launch_power_w` 换成所属波段功率的新连接列表；
* :meth:`BandPlan.assert_connections_single_band` —— 拒绝横跨频段边界的
  连接，而不是静默按中心频率归一段。

⚠ 两条路**不互相覆盖**：`launch_power_profile_w` 是逐信道的，`ConnectionSpec`
是逐连接的。可变宽连接下两者粒度不同，本模块不假装它们等价，转换必须显式。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import ConnectionSpec
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat, NDArrayInt

__all__ = [
    "nm_to_hz",
    "hz_to_nm",
    "BandSpec",
    "BandPlan",
    "ITU_C_LOW_NM",
    "ITU_C_HIGH_NM",
    "ITU_L_LOW_NM",
    "ITU_L_HIGH_NM",
    "ITU_S_LOW_NM",
    "ITU_S_HIGH_NM",
    "ASSUMED_U_LOW_NM",
    "ASSUMED_U_HIGH_NM",
    "U_BAND_ASSUMPTIONS",
    "CL_BAND_SPECS",
    "CLS_BAND_SPECS",
    "CLU_BAND_SPECS",
    "SCLU_BAND_SPECS",
    "sclu_span_report",
]

_NM = 1e-9
_THZ = 1e12


def nm_to_hz(wavelength_nm: float) -> float:
    """波长 (nm) -> 频率 (Hz)。

    Args:
        wavelength_nm: 波长 (nm)，须 > 0。

    Returns
    -------
        频率 (Hz)。

    Raises
    ------
        ValueError: ``wavelength_nm <= 0``。
    """
    if wavelength_nm <= 0:
        raise ValueError(f"wavelength_nm must be positive, got {wavelength_nm}.")
    return C_LIGHT / (wavelength_nm * _NM)


def hz_to_nm(frequency_hz: float) -> float:
    """频率 (Hz) -> 波长 (nm)。与 :func:`nm_to_hz` 互逆。

    Args:
        frequency_hz: 频率 (Hz)，须 > 0。

    Returns
    -------
        波长 (nm)。

    Raises
    ------
        ValueError: ``frequency_hz <= 0``。
    """
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}.")
    return C_LIGHT / frequency_hz / _NM


# ---------------------------------------------------------------------------
# 波长边界：真源是波长 (nm)。改这里的常数即可让全链路重算（FU2-1 边界参数化）。
# ---------------------------------------------------------------------------

#: ITU-T G.694.1 C 波段下沿 (nm)。与 RL 侧 ``CLS_BAND_SPECS`` 同值。
ITU_C_LOW_NM: float = 1530.0
#: ITU-T G.694.1 C 波段上沿 (nm)。与 RL 侧同值。
ITU_C_HIGH_NM: float = 1565.0
#: ITU-T G.694.1 L 波段下沿 (nm)。与 RL 侧 ``CL_BAND_SPECS`` 同值。
ITU_L_LOW_NM: float = 1565.0
#: ITU-T G.694.1 L 波段上沿 (nm)。与 RL 侧同值。
ITU_L_HIGH_NM: float = 1625.0
#: ITU-T G.694.1 S 波段下沿 (nm)。与 RL 侧 ``CLS_BAND_SPECS`` 同值。
ITU_S_LOW_NM: float = 1460.0
#: ITU-T G.694.1 S 波段上沿 (nm)。与 RL 侧同值。
ITU_S_HIGH_NM: float = 1530.0

#: ★ **假设** U 波段下沿 (nm)。ITU-T G.694.1 未定义 U 波段。
ASSUMED_U_LOW_NM: float = 1660.0
#: ★ **假设** U 波段上沿 (nm)。ITU-T G.694.1 未定义 U 波段。
ASSUMED_U_HIGH_NM: float = 1770.0

#: U 波段的**逐条假设**，随证据落盘。编号与 `docs/fu2_scu_evaluation.md` 一致。
U_BAND_ASSUMPTIONS: tuple[str, ...] = (
    "U1: 1660-1770 nm 的区间本身是本刀假设 —— ITU-T G.694.1 只定义 C/L/S/E/O 等，"
    "没有 U 波段；两仓在 FG-2 之前也没有 U 段资产 (FG-4d 实测)。",
    "U2: U 波段按 TDFA 放大建模（同 S 波段的判断：频率更低、增益效率更差）。"
    "本仓 ODE 路径只用 noise_figure_db 参与数值，amplifier 字段是标注，不参与计算。",
    "U3: U 波段噪声系数假设 6.5 dB（高于 S 的 6.0 dB，高于 C 的 4.5 dB）。"
    "**无出处**，是按'频率更低 -> 增益效率更差'单调外推的一个占位值。",
    "U4: L 与 U 之间 1625-1660 nm 的 35 nm 空隙被当作 U/L 之间的保护间隔。"
    "它不是人为预留的 guard，而是假设区间 1660-1770 nm 本身排除该段的结果。",
)


@dataclass(frozen=True)
class BandSpec:
    """一个频段的边界、放大器特性与发射功率。

    波长是**真源**：``low_nm`` < ``high_nm``（**短**波长在前、**长**波长在后），
    频率边界由其换算而来且**自动反向**（波长越长频率越低），所以
    :attr:`low_hz` < :attr:`high_hz`。

    ⚠️ 命名约定：``low_nm`` / ``high_nm` 指**波长数值**的大小，不是频率大小。
    例：L 波段 ``BandSpec("L", 1565.0, 1625.0)`` —— 1565 nm 比 1625 nm **短**，
    故 ``low_nm = 1565``、``high_nm = 1625``；换算后 ``low_hz ≈ 184.49 THz``、
    ``high_hz ≈ 191.55 THz``。

    频率区间语义是**左闭右开** ``[low_hz, high_hz)``，与
    :class:`~network_nlo_eval.core.flexgrid.FlexibleGrid` 的半开区间约定一致。
    这样首尾相接的频段（S|C、C|L）能无缝平铺而不产生双重归属——RL 侧的
    :class:`~multiband_optical_rl.bands.BandPlan` 用的是**闭区间**，因此一个正好
    落在 1565 nm 上的信道会被 L 与 C 同时认领。本模块刻意不复制那个歧义。

    Attributes
    ----------
    name:
        频段名（``"S"`` / ``"C"`` / ``"L"`` / ``"U"``）。
    low_nm:
        短波段边缘 (nm)，数值较小，频率**最高**。
    high_nm:
        长波段边缘 (nm)，数值较大，频率**最低**。
    amplifier:
        放大器类型标注。**不参与本仓任何数值计算**，仅作记录。
    noise_figure_db:
        本频段的 EDFA/TDFA 噪声系数 (dB)。
    launch_power_w:
        本频段的**逐波段发射功率** (W，双偏振合计)。``None`` 表示"未设定"，
        此时 :meth:`BandPlan.launch_power_profile_w` 会拒绝构造 profile，
        而不是静默填 0。
    source:
        边界出处标注；U 段为 ``"assumption"``。
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
                f"LONGER wavelength), got low_nm={self.low_nm} high_nm={self.high_nm}."
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

        Returns
        -------
            落在本频段内为 True；恰好等于上边缘时为 False（归更高频段所有）。
        """
        return self.low_hz <= frequency_hz < self.high_hz

    def contains_hz_array(self, frequencies_hz: NDArrayFloat) -> NDArrayBool:
        """向量版 :meth:`contains_hz`，掩码语义完全相同（左闭右开）。

        Args:
            frequencies_hz: 待判频率数组 (Hz)。

        Returns
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
            "frequency_thz": [self.low_hz / _THZ, self.high_hz / _THZ],
            "span_thz": self.span_hz / _THZ,
            "amplifier": self.amplifier,
            "noise_figure_db": self.noise_figure_db,
            "launch_power_w": self.launch_power_w,
            "source": self.source,
        }


def _u_spec(launch_power_w: float | None) -> BandSpec:
    """构造 U 波段规格（假设值集中在此，便于一处改）。"""
    return BandSpec(
        name="U",
        low_nm=ASSUMED_U_LOW_NM,
        high_nm=ASSUMED_U_HIGH_NM,
        amplifier="TDFA",
        noise_figure_db=6.5,
        launch_power_w=launch_power_w,
        source="assumption (U1/U2/U3/U4, see U_BAND_ASSUMPTIONS)",
    )


#: 频段集合一律按**频率从低到高**排列（与 RL 侧约定相同）。故 S+C+L+U 的顺序是
#: U -> L -> C -> S（波长顺序正好相反）。
CL_BAND_SPECS: tuple[BandSpec, ...] = (
    BandSpec("L", ITU_L_LOW_NM, ITU_L_HIGH_NM, "EDFA", 5.0),
    BandSpec("C", ITU_C_LOW_NM, ITU_C_HIGH_NM, "EDFA", 4.5),
)

CLS_BAND_SPECS: tuple[BandSpec, ...] = CL_BAND_SPECS + (
    BandSpec("S", ITU_S_LOW_NM, ITU_S_HIGH_NM, "TDFA", 6.0),
)

#: C+L+U（不含 S）。注意 U 频率最低，故顺序是 U -> L -> C。
CLU_BAND_SPECS: tuple[BandSpec, ...] = (_u_spec(None),) + CL_BAND_SPECS

#: S+C+L+U 全集，**按频率从低到高**排列：U -> L -> C -> S（波长顺序恰好相反）。
CLS_BAND_SPECS_U: tuple[BandSpec, ...] = (_u_spec(None),) + CLS_BAND_SPECS

#: S+C+L+U 全集的规范名。
SCLU_BAND_SPECS: tuple[BandSpec, ...] = CLS_BAND_SPECS_U


class BandPlan:
    """把一段频谱（逐信道或逐连接）切分到若干频段，并给出逐波段发射功率。

    校验沿用了 RL 侧 :class:`~multiband_optical_rl.bands.BandPlan` 已经踩过的
    几条坑：非空、两两不重叠、**全覆盖**、每段至少一个元素。理由同样——任一条
    失守都会让上层拿到静默错误的频段归属，而不是一个显式异常。

    差异一处：频率区间是**左闭右开**，故相邻频段可无缝平铺（见 :class:`BandSpec`）。

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
                （本模块只要求覆盖判定有意义，不要求递增）。
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
        empty = [
            spec.name for index, spec in enumerate(specs) if not np.any(band_index == index)
        ]
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
    def from_channels(cls, grid: SpectrumGrid, specs: Sequence[BandSpec]) -> BandPlan:
        """由遗留 ``SpectrumGrid`` 构造（逐**信道**粒度）。

        Args:
            grid: 频谱栅格。
            specs: 频段定义，按频率从低到高。

        Returns
        -------
            每个信道归属一个频段的 :class:`BandPlan`。
        """
        return cls(grid.frequencies, specs)

    @classmethod
    def from_connections(cls, connections: Sequence[ConnectionSpec], specs: Sequence[BandSpec]) -> BandPlan:
        """由灵活栅格连接构造（逐**连接**粒度）。

        每个连接的归属按其**中心频率**判定；跨边界的宽连接请先调用
        :meth:`assert_connections_single_band` 得到显式拒绝。

        Args:
            connections: 连接列表。
            specs: 频段定义，按频率从低到高。

        Returns
        -------
            每条连接归属一个频段的 :class:`BandPlan`。

        Raises
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
        """被切分的频率元素数量（信道数或连接数）。"""
        return int(self.frequencies_hz.size)

    @property
    def band_names(self) -> tuple[str, ...]:
        """频段名称，按频率从低到高。"""
        return tuple(spec.name for spec in self.specs)

    def band_element_indices(self, band: int) -> NDArrayInt:
        """返回指定频段内的元素索引。

        Args:
            band: 频段序号。

        Returns
        -------
            该频段内的元素索引数组，升序。

        Raises
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

        Returns
        -------
            频段序号。

        Raises
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

        Returns
        -------
            元素数量 (int)。

        Raises
        ------
            IndexError: ``band`` 越界。
        """
        return int(self.band_element_indices(band).size)

    # --- 逐频段物理量 -----------------------------------------------------
    def noise_figure_db_per_element(self) -> NDArrayFloat:
        """返回逐元素噪声系数 (dB)，按频段常数填充。"""
        profile = np.zeros(self.num_elements, dtype=np.float64)
        for index, spec in enumerate(self.specs):
            profile[self._band_elements[index]] = spec.noise_figure_db
        return profile

    # --- 逐波段发射功率 ★ --------------------------------------------------
    def launch_power_profile_w(self) -> NDArrayFloat:
        """逐元素发射功率 (W)，由所属频段的 ``launch_power_w`` 填充。

        返回数组的语义与 :attr:`network_nlo_eval.network.state.LinkState.launch_power_profile_w`
        一致（**发射端注入功率**，双偏振合计，非接收功率），可直接用作该字段的
        赋值源，或作为 :meth:`~network_nlo_eval.models.isrs_gn.MultiBandISRSGN.evaluate_path`
        的 ``power_in_w`` 入参。

        Returns
        -------
            长度 = :attr:`num_elements` 的 float64 数组。

        Raises
        ------
            ValueError: 存在 ``launch_power_w is None`` 的频段——此时无法给出
                完整功率谱，**不**静默填 0（0 在既有语义里表示"未占用"）。
        """
        unset = [spec.name for spec in self.specs if spec.launch_power_w is None]
        if unset:
            raise ValueError(
                f"Bands {unset} have launch_power_w=None, so no complete launch power "
                "profile can be built. Set per-band launch_power_w (dataclasses.replace "
                "on the BandSpec) for every band; 0.0 means 'unoccupied' and must be "
                "set explicitly."
            )
        profile = np.zeros(self.num_elements, dtype=np.float64)
        for index, spec in enumerate(self.specs):
            profile[self._band_elements[index]] = float(spec.launch_power_w)
        return profile

    def per_band_launch_power_w(self) -> dict[str, float]:
        """返回逐频段发射功率字典，未设定的频段值为 ``None``。

        Returns
        -------
            ``{band_name: power_w or None}``，按频率从低到高。
        """
        return {spec.name: spec.launch_power_w for spec in self.specs}

    def assert_connections_single_band(self, connections: Sequence[ConnectionSpec]) -> None:
        """拒绝任何横跨频段边界的连接。

        宽连接（例如 8 槽 = 100 GHz 虽窄，但更宽的弹性连接可达数百 GHz）
        可能整体压在一个频段边界上。届时"它是 S 还是 C"没有唯一答案——
        按中心频率归属会静默给出错误功率。

        Args:
            connections: 待检查的连接列表。

        Raises
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
                    f"[{low_hz / _THZ:.6f}, {high_hz / _THZ:.6f}] THz, which straddles "
                    f"band boundary between {hit}. Per-band launch power is ambiguous for it; "
                    "split the connection or drop the crossing band from the plan."
                )

    def apply_launch_power(
        self, connections: Sequence[ConnectionSpec], *, allow_crossing: bool = False
    ) -> list[ConnectionSpec]:
        """把逐波段发射功率写入连接，返回新的连接列表。

        :class:`ConnectionSpec` 是 frozen dataclass，因此返回的是**替换了
        ``launch_power_w`` 的新对象**，原列表不被修改。

        Args:
            connections: 原连接列表。
            allow_crossing: 设为 True 时跳过边界检查（调用方自负责）。

        Returns
        -------
            ``launch_power_w`` 已换成所属频段功率的连接列表。

        Raises
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
    def guard_report(self) -> list[dict[str, object]]:
        """相邻频段之间的**实际**间隔（保护间隔）报告。

        G.694.1 的 S/C/L 首尾相接，故这三对之间的 guard 恒为 0——这正是既有
        C/L 做法。本函数把 guard **算出来**而不是断言它，任何人改了边界常数后
        都能立刻看出间隔变成了多少。

        Returns
        -------
            每对相邻频段一项：``lower_band`` / ``upper_band``（频率高者为 upper）、
            ``gap_nm``（>=0，相邻时为 0）、``gap_thz``、``touching``（是否相接）。
        """
        ordered = sorted(self.specs, key=lambda spec: spec.low_hz)
        report: list[dict[str, object]] = []
        for lower_freq, higher_freq in zip(ordered, ordered[1:], strict=False):
            # ordered 已按频率升序，故 higher_freq.low_hz 是较高频段的下沿、
            # lower_freq.high_hz 是较低频段的上沿。两者之差即中间的空隙（相接时为 0）。
            gap_hz = higher_freq.low_hz - lower_freq.high_hz
            gap_nm = hz_to_nm(lower_freq.high_hz) - hz_to_nm(higher_freq.low_hz)
            report.append(
                {
                    "lower_frequency_band": lower_freq.name,
                    "higher_frequency_band": higher_freq.name,
                    "gap_nm": gap_nm,
                    "gap_thz": gap_hz / _THZ,
                    "touching": bool(abs(gap_hz) < 1.0),
                }
            )
        return report

    def as_dict(self) -> dict[str, object]:
        """机器可读描述，随证据 JSON 落盘。"""
        return {
            "band_order_low_to_high_frequency": self.band_names,
            "num_elements": self.num_elements,
            "num_bands": self.num_bands,
            "elements_per_band": {
                spec.name: int(self._band_elements[index].size)
                for index, spec in enumerate(self.specs)
            },
            "bands": [spec.as_dict() for spec in self.specs],
            "guards": self.guard_report(),
        }


def sclu_span_report(specs: Sequence[BandSpec] = SCLU_BAND_SPECS) -> dict[str, object]:
    """S+C+L+U 的跨距口径——**连续跨度**与**包络跨度**必须分开说。

    两者不同，因为 U 与 L 之间有 35 nm 空隙（假设 U4）：

    * ``sum_of_band_spans_thz``：各频段跨度之和 = 真正有信道的频谱材料宽度；
    * ``envelope_span_thz``：最慢频率 (U 下沿 1770 nm) 到最高频率 (S 上沿 1460 nm)
      的**包络**，含 U/L 之间的空隙；
    * ``total_gap_thz``：``envelope - sum_of_band_spans``，应恰好等于空隙之和。

    FG-U1 的合成均匀栅格量到的是 **27 THz** 跨距——那是**连续口径**（把 U/L 的
    空隙也拉满信道），与这里的包络口径 35.96 THz 不同源。本刀会同时报出这几个数，
    使与 FG-U1 的对比有明确基准，而不是把两个口径混着比。

    Args:
        specs: 频段定义，默认 S+C+L+U 全集。

    Returns
    -------
        含两种跨度、各频段跨度、以及空隙数量的字典。

    Raises
    ------
        ValueError: ``specs`` 为空。
    """
    if not specs:
        raise ValueError("sclu_span_report requires at least one band specification.")
    ordered = sorted(specs, key=lambda spec: spec.low_hz)
    gaps_thz = [
        (higher.low_hz - lower.high_hz) / _THZ
        for lower, higher in zip(ordered, ordered[1:], strict=False)
    ]
    envelope_low_hz = min(spec.low_hz for spec in ordered)
    envelope_high_hz = max(spec.high_hz for spec in ordered)
    return {
        "band_order_low_to_high_frequency": tuple(spec.name for spec in ordered),
        "envelope_low_nm": hz_to_nm(envelope_high_hz),
        "envelope_high_nm": hz_to_nm(envelope_low_hz),
        "envelope_span_thz": (envelope_high_hz - envelope_low_hz) / _THZ,
        "sum_of_band_spans_thz": sum(spec.span_hz for spec in ordered) / _THZ,
        "inter_band_gaps_thz": gaps_thz,
        "largest_gap_thz": max(gaps_thz) if gaps_thz else 0.0,
        "total_gap_thz": sum(gaps_thz),
        "u_l_gap_thz": (
            gaps_thz[0]
            if len(ordered) > 1 and ordered[0].name == "U" and ordered[1].name == "L"
            else 0.0
        ),
    }
