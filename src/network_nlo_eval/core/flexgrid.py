"""灵活栅格 (Flexible Grid)：槽位化频谱模型。

本模块是 `docs/flexgrid_design.md` §3「冻结接口」的逐字实现，供 FG-2
(`network-nlo-eval`) 与 FG-3 (`multiband-optical-rl`) 共用。字段名、签名、单位与
错误语义均为冻结项，改动即破坏下游。

术语
----
* **槽 (slot)**：宽度 `slot_width_hz` 的频谱基本单位，半开区间
  `[band_low_edge_hz + i*W, band_low_edge_hz + (i+1)*W)`。
* **连接 (connection)**：占据一段**连续**槽 `[first_slot, first_slot+n_slots)`。
  一条连接 = 一个 ODE 节点 = 一个 RL 分配单元。

★ 半槽陷阱
----------
连接中心频率的分子是 ``n_slots / 2``（**区间**中心），不是 ``(n_slots - 1) / 2`
（点阵中心）。用后者会使整条栅格偏移半槽（12.5 GHz 槽下 6.25 GHz），
固定栅格作为特例的逐位等价（AC-1）立即失败。`test_fg2_flexible_grid.py::test_half_slot_trap`
显式锁定该行为。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MODULATION_FORMATS",
    "BITS_PER_SYMBOL",
    "bits_per_symbol",
    "FlexibleGrid",
    "ConnectionSpec",
    "slots_for_rate",
]

# ---------------------------------------------------------------------------
# 调制格式阶梯 (设计 §2.2)
# ---------------------------------------------------------------------------
# 谱效率定义沿用本仓既有约定：spectral_efficiency_bps_hz = bits_per_symbol，
# 且隐含 Rs = W（符号率等于信道宽度，无滚降裕量）。
MODULATION_FORMATS: tuple[str, ...] = ("BPSK", "QPSK", "8QAM", "16QAM", "64QAM")

BITS_PER_SYMBOL: dict[str, int] = {
    "BPSK": 1,
    "QPSK": 2,
    "8QAM": 3,
    "16QAM": 4,
    "64QAM": 6,
}


def bits_per_symbol(modulation: str) -> int:
    """返回调制格式的每符号比特数。

    Args:
        modulation: 调制格式名，必须出现在 :data:`MODULATION_FORMATS` 中。

    Returns
    -------
        每符号比特数 (bit/symbol)，>= 1。

    Raises
    ------
        ValueError: 调制格式不在 :data:`MODULATION_FORMATS` 中。
    """
    try:
        return BITS_PER_SYMBOL[modulation]
    except KeyError:
        raise ValueError(
            f"Unknown modulation format {modulation!r}; expected one of {list(MODULATION_FORMATS)}."
        ) from None


# ---------------------------------------------------------------------------
# §3.3 slots_for_rate
# ---------------------------------------------------------------------------
def slots_for_rate(
    bit_rate_bps: float,
    bits_per_symbol: int,
    roll_off: float,
    slot_width_hz: float,
) -> int:
    """由比特率算所需槽数。纯函数，无状态。

    公式:  n = ceil( (bit_rate_bps / bits_per_symbol) * (1 + roll_off) / slot_width_hz )
          n = max(n, 1)

    取整一律 ``ceil``、禁止 ``floor``：``floor`` 会让实际占用窄于该比特率所需的
    谱宽，低估容量并破坏 50 GHz 对齐。取整在**槽数空间**做，不在 Hz 空间做——
    避免浮点 ceil 边界。

    Args:
        bit_rate_bps: 比特率 (bit/s)，> 0
        bits_per_symbol: 每符号比特数 (bit/symbol)，>= 1
        roll_off: RRC 滚降系数，0 <= roll_off < 1。取 0.0 与现有
            ``Rs = W`` 约定自洽（默认，见 [需定夺-2]）。
        slot_width_hz: 槽宽 (Hz)，> 0

    Returns
    -------
        槽数 (int)，>= 1

    Raises
    ------
        ValueError: bit_rate_bps <= 0；bits_per_symbol < 1；
                    not (0 <= roll_off < 1)；slot_width_hz <= 0
    """
    if bit_rate_bps <= 0:
        raise ValueError(f"bit_rate_bps must be positive, got {bit_rate_bps}.")
    if bits_per_symbol < 1:
        raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol}.")
    if not (0 <= roll_off < 1):
        raise ValueError(f"roll_off must satisfy 0 <= roll_off < 1, got {roll_off}.")
    if slot_width_hz <= 0:
        raise ValueError(f"slot_width_hz must be positive, got {slot_width_hz}.")

    required_width_hz = (bit_rate_bps / bits_per_symbol) * (1.0 + roll_off)
    return max(1, math.ceil(required_width_hz / slot_width_hz))


# ---------------------------------------------------------------------------
# §3.1 FlexibleGrid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlexibleGrid:
    """灵活栅格：等宽槽的半开区间并集 [band_low_edge_hz, band_high_edge_hz)。

    单位：全部频率为 Hz。槽索引为 0-based 整数。
    频率真源唯一：centre(i, n) = band_low_edge_hz + (i + n/2) * slot_width_hz

    不变式（__post_init__ 强制）：
      1. slot_width_hz > 0
      2. num_slots 为正整数且非 bool
      3. band_low_edge_hz > 0
      4. band_low_edge_hz + num_slots*slot_width_hz > 0（隐含于 3）

    注意：**不做**频段覆盖校验。覆盖不变式施加在连接**中心**上（设计 §2.1 /
    [需定夺-1]），由上层 BandPlan 检查。把覆盖施加在槽边缘上与"230 道"在几何上
    互斥——C+L 边到边可用宽度 11.4551 THz < 230x50 GHz 需要的 11.5000 THz，
    缺口 44.9 GHz。槽边缘允许越出标称频段（下侧 25.0 GHz、上侧 19.9 GHz）。
    """

    slot_width_hz: float  # 标称槽宽 (Hz)，须 > 0
    num_slots: int  # 槽总数，须 >= 1
    band_low_edge_hz: float  # 槽 0 的**下边缘** (Hz)，须 > 0

    def __post_init__(self) -> None:
        """强制不变式。"""
        if self.slot_width_hz <= 0:
            raise ValueError(f"slot_width_hz must be positive, got {self.slot_width_hz}.")
        if isinstance(self.num_slots, bool) or not isinstance(self.num_slots, int) or self.num_slots <= 0:
            raise ValueError(f"num_slots must be a positive integer, got {self.num_slots!r}.")
        if self.band_low_edge_hz <= 0:
            raise ValueError(
                f"band_low_edge_hz must be positive (optical frequencies are positive), "
                f"got {self.band_low_edge_hz}."
            )

    # --- 派生量 -----------------------------------------------------------
    @property
    def band_high_edge_hz(self) -> float:
        """栅格上边缘 (Hz)，半开区间不含此值。"""
        return self.band_low_edge_hz + self.num_slots * self.slot_width_hz

    @property
    def band_center_hz(self) -> float:
        """频段中心频率 (Hz)，即 `f_rel_hz` 的参考原点（设计 §3.4）。"""
        return self.band_low_edge_hz + self.num_slots * self.slot_width_hz / 2

    @property
    def total_bandwidth_hz(self) -> float:
        """栅格总宽度 (Hz) = num_slots * slot_width_hz。"""
        return self.num_slots * self.slot_width_hz

    # --- 槽坐标 -----------------------------------------------------------
    def _check_slot(self, slot: int) -> int:
        if isinstance(slot, bool) or not isinstance(slot, (int,)):
            raise TypeError(f"slot must be an integer, got {type(slot).__name__}.")
        if not (0 <= slot < self.num_slots):
            raise IndexError(f"Slot index {slot} out of bounds for {self.num_slots} slots.")
        return slot

    def slot_low_edge_hz(self, slot: int) -> float:
        """槽 slot 的下边缘 (Hz)。IndexError if not 0 <= slot < num_slots."""
        return self.band_low_edge_hz + self._check_slot(slot) * self.slot_width_hz

    def slot_high_edge_hz(self, slot: int) -> float:
        """槽 slot 的上边缘 (Hz)，半开区间不含此值。IndexError 同上。"""
        return self.band_low_edge_hz + (self._check_slot(slot) + 1) * self.slot_width_hz

    def slot_center_hz(self, slot: int) -> float:
        """槽 slot 的中心 (Hz)。IndexError 同上。"""
        return self.band_low_edge_hz + (self._check_slot(slot) + 0.5) * self.slot_width_hz

    # --- 连接中心 ★ 频率真源 ---------------------------------------------
    def centre_hz(self, first_slot: int, n_slots: int) -> float:
        """占据 [first_slot, first_slot+n_slots) 的连接的**中心频率** (Hz)。

        ★ 分子是 n_slots/2（区间中心），不是 (n_slots-1)/2（点阵中心）。
          用后者会整体偏移半槽（12.5 GHz 槽下 6.25 GHz），AC-1 失败。

        这条连接代表一个 ODE 节点，其频率必须取连接中心而非首槽/末槽（设计 §2.4）。

        Raises
        ------
            IndexError: first_slot < 0 或 first_slot + n_slots > num_slots。
            ValueError: n_slots <= 0。
        """
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}.")
        if first_slot < 0 or first_slot + n_slots > self.num_slots:
            raise IndexError(
                f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for {self.num_slots} slots."
            )
        return self.band_low_edge_hz + (first_slot + n_slots / 2) * self.slot_width_hz

    def connection_edges_hz(self, first_slot: int, n_slots: int) -> tuple[float, float]:
        """返回连接带 [low, high) 的两个边缘 (Hz)。"""
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}.")
        if first_slot < 0 or first_slot + n_slots > self.num_slots:
            raise IndexError(
                f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for {self.num_slots} slots."
            )
        half = n_slots * self.slot_width_hz / 2
        centre = self.centre_hz(first_slot, n_slots)
        return centre - half, centre + half

    def edges_are_inside(self, low_hz: float, high_hz: float) -> bool:
        """连接带 [low, high) 是否完全落在本栅格内。仅供诊断。"""
        return low_hz >= self.band_low_edge_hz and high_hz <= self.band_high_edge_hz

    # --- 便捷构造 ---------------------------------------------------------
    @classmethod
    def from_legacy_grid(
        cls,
        num_channels: int,
        channel_spacing_hz: float,
        center_frequency_hz: float,
        slots_per_channel: int = 4,
    ) -> FlexibleGrid:
        """由遗留 ``SpectrumGrid`` 参数构造等价的灵活栅格。

        固定栅格是灵活栅格的特例（设计 §2.3）：
            slot_width = channel_spacing_hz / slots_per_channel
            num_slots   = num_channels * slots_per_channel
            band_low_edge = 第一个信道中心 - channel_spacing_hz/2

        在 50 GHz 间隔 + 4 槽时 slot_width = 12.5 GHz，230 信道 -> 920 槽。
        """
        if slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")
        slot_width = channel_spacing_hz / slots_per_channel
        # 遗留 frequencies[0] 是**第一个信道中心**（spectrum.py:28 的 lowest 是
        # 命名误导），带边需再减半个信道间隔。
        first_channel_centre = center_frequency_hz - (num_channels - 1) * channel_spacing_hz / 2
        return cls(
            slot_width_hz=slot_width,
            num_slots=num_channels * slots_per_channel,
            band_low_edge_hz=first_channel_centre - channel_spacing_hz / 2,
        )


# ---------------------------------------------------------------------------
# §3.2 ConnectionSpec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConnectionSpec:
    """一次实验中的一个 WDM 连接（= 一个 ODE 节点 = 一个 RL 分配单元）。

    不变式（__post_init__ 强制）：
      1. first_slot >= 0 且 first_slot + n_slots <= grid.num_slots
      2. n_slots >= 1
      3. n_slots == slots_for_rate(bit_rate_bps, bits_per_symbol(modulation),
         roll_off, grid.slot_width_hz)
         —— §2.2 公式的**强制**形式：不允许"给 4 槽却只跑 100 Gbps 的 64QAM"
         这类静默欠配。违反抛 ValueError。

    注：``grid`` 字段是 §3.2 不变式 1/3 必需的——两者分别引用
    ``grid.num_slots`` 与 ``grid.slot_width_hz``。它是**只读**引用。
    """

    grid: FlexibleGrid
    connection_id: int  # >= 0，同一实验内唯一
    first_slot: int  # 占据的起始槽 (0-based, 闭)
    n_slots: int  # 占据的槽数，>= 1
    bit_rate_bps: float  # 目标比特率 (bit/s)，> 0
    modulation: str  # 必须出现在 MODULATION_FORMATS 中
    symbol_rate_hz: float  # 符号率 (Hz)，> 0
    roll_off: float  # RRC 滚降系数，0 <= roll_off < 1
    launch_power_w: float  # 单连接入纤总功率 (W, 双偏振合计)，>= 0；0 = 未占用

    def __post_init__(self) -> None:
        """强制不变式 1-3。"""
        if isinstance(self.connection_id, bool) or not isinstance(self.connection_id, int) or self.connection_id < 0:
            raise ValueError(f"connection_id must be a non-negative integer, got {self.connection_id!r}.")
        if isinstance(self.n_slots, bool) or not isinstance(self.n_slots, int) or self.n_slots < 1:
            raise ValueError(f"n_slots must be a positive integer, got {self.n_slots!r}.")
        if isinstance(self.first_slot, bool) or not isinstance(self.first_slot, int):
            raise ValueError(f"first_slot must be an integer, got {self.first_slot!r}.")
        if self.first_slot < 0 or self.first_slot + self.n_slots > self.grid.num_slots:
            raise ValueError(
                f"Connection slot range [{self.first_slot}, {self.first_slot + self.n_slots}) "
                f"out of bounds for grid with {self.grid.num_slots} slots."
            )
        if self.bit_rate_bps <= 0:
            raise ValueError(f"bit_rate_bps must be positive, got {self.bit_rate_bps}.")
        if self.symbol_rate_hz <= 0:
            raise ValueError(f"symbol_rate_hz must be positive, got {self.symbol_rate_hz}.")
        if not (0 <= self.roll_off < 1):
            raise ValueError(f"roll_off must satisfy 0 <= roll_off < 1, got {self.roll_off}.")
        if self.launch_power_w < 0:
            raise ValueError(f"launch_power_w must be non-negative, got {self.launch_power_w}.")

        required_slots = slots_for_rate(
            self.bit_rate_bps,
            bits_per_symbol(self.modulation),
            self.roll_off,
            self.grid.slot_width_hz,
        )
        if self.n_slots != required_slots:
            raise ValueError(
                f"n_slots={self.n_slots} is under-/over-provisioned for "
                f"{self.bit_rate_bps / 1e9:g} Gbps {self.modulation} (roll_off={self.roll_off}): "
                f"slots_for_rate requires {required_slots} slots of "
                f"{self.grid.slot_width_hz / 1e9:g} GHz."
            )

    # --- 派生量 -----------------------------------------------------------
    @property
    def centre_hz(self) -> float:
        """连接中心频率 (Hz)。"""
        return self.grid.centre_hz(self.first_slot, self.n_slots)

    @property
    def occupied_bandwidth_hz(self) -> float:
        """连接占用带宽 (Hz) = n_slots * slot_width_hz。"""
        return self.n_slots * self.grid.slot_width_hz

    @property
    def is_occupied(self) -> bool:
        """是否有发射功率（launch_power_w > 0）。未占用连接仍占数组槽位。"""
        return self.launch_power_w > 0
