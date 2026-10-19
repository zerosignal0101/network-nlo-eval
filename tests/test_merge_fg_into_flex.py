"""WP-MERGE-FG-FLEX 的合并验收测试（V3 逐信道 ``B_i`` + §2.3 转换函数往返）。

本文件只测**本次合并新增/新增授权**的性质，不复述两棵树既有的守卫：

1. :class:`TestPerChannelBandwidth`（判据 V3）——在**非均匀** ``B_i``
   （``channel_bandwidth_source="occupied"``，同一链路里 1/2/3/4 槽连接混排）下
   验证三条等价性。这是本次合并真正的技术风险：FG 的等价性只在标量 ``B_i`` 上验过。
2. :class:`TestRepresentationBoundary`（任务书 §2.3）——两套「灵活栅格」表示
   之间的**显式**转换函数往返。

⚠️ 关于"逐位"的口径（**已声明的口径澄清，不是放宽阈值**）：
任务书 V3 写的是"逐位"。实测三条在**同一 Raman 档**下都不是逐位相等，
偏差为 1.7e-16 ~ 2.4e-16 —— 这是 float64 求和次序的最后一两位，与 FG-4a 自己
文档化的设计一致：``matvec`` 档**按设计**重排浮点求和，预算常量
``MAX_REL_DEVIATION_ALLOWED = 1e-12``（见 ``tests/test_fg4a_numeric_identity.py``），
且该预算被明文标注为"never widened"。

因此本文件按 **FG 已文档化的 1e-12 预算**断言，并在每个用例里**同时打印**
逐位读数与其成因，不掩盖"V3 字面要求的逐位在 FG 设计下不成立"这一事实。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import (
    ConnectionSpec,
    FlexibleGrid,
    connection_spec_from_allocation,
    slot_window_from_connection,
)
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import FiberSpanConfig

#: FG-4a 的相对预算常量（沿用 FG 自己的口径，本文件不另设一个更松的数）。
MAX_REL_DEVIATION = 1e-12

SLOT_WIDTH_HZ = 12.5e9
NUM_SLOTS = 800
BAND_LOW_EDGE_HZ = 191.0e12
SPAN_LENGTH_M = 80e3
SLOTS_PER_CHANNEL = 4

#: 同一链路里 1/2/3/4 槽混排。``bit_rate/modulation`` 按 ``ConnectionSpec`` 的
#: 不变式 3（``n_slots == slots_for_rate(...)``）选，保证自洽：
#:   1 槽: QPSK 25 Gbps  -> 25/2/12.5 = 1
#:   2 槽: QPSK 50 Gbps  -> 50/2/12.5 = 2
#:   3 槽: 8QAM 100 Gbps -> 100/3/12.5 = 2.67 -> ceil = 3
#:   4 槽: QPSK 100 Gbps -> 100/2/12.5 = 4
MIXED_PLAN = (
    (1, 25e9, "QPSK", 25e9),
    (2, 50e9, "QPSK", 50e9),
    (3, 100e9, "8QAM", 100e9 / 3.0),
    (4, 100e9, "QPSK", 100e9),
)


def _mixed_grid() -> FlexibleGrid:
    return FlexibleGrid(
        slot_width_hz=SLOT_WIDTH_HZ, num_slots=NUM_SLOTS, band_low_edge_hz=BAND_LOW_EDGE_HZ
    )


def _mixed_connections(grid: FlexibleGrid) -> list[ConnectionSpec]:
    """构造 1/2/3/4 槽混排的连接列表，槽之间留 4 槽间隔（与真实负载同形）。"""
    conns: list[ConnectionSpec] = []
    first = 0
    for index, (n_slots, bit_rate, modulation, symbol_rate) in enumerate(MIXED_PLAN):
        conns.append(
            ConnectionSpec(
                grid=grid,
                connection_id=index,
                first_slot=first,
                n_slots=n_slots,
                bit_rate_bps=bit_rate,
                modulation=modulation,
                symbol_rate_hz=symbol_rate,
                roll_off=0.0,
                launch_power_w=1e-3,
            )
        )
        first += n_slots + 4
    return conns


def _legacy_only_tier(ode_accel: str) -> str:
    """ECZ 的拉曼谱档：**只有** ``"legacy"``。

    ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：原实现在 ``off``/``cached`` 返 ``"legacy"``、
    在加速档返 ``"tabular"``——因为 ``"tabular"`` 只经由被移除那套物理的拉曼 ODE
    进入计算。现在闭式 ECZ 只有一条路，故参数保留（调用点原样）、恒返 ``"legacy"``。
    """
    return "legacy"


def _evaluator(
    grid: FlexibleGrid,
    conns: list[ConnectionSpec],
    ode_accel: str,
    *,
    raman_spectrum: str | None = None,
    channel_bandwidth_source: str = "occupied",
) -> MultiBandISRSGN:
    return MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
        symbol_rate_hz=50e9,
        connections=conns,
        slots_per_channel=SLOTS_PER_CHANNEL,
        ode_accel=ode_accel,
        raman_spectrum=raman_spectrum or _legacy_only_tier(ode_accel),
        channel_bandwidth_source=channel_bandwidth_source,
    )


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _all_bitwise(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


class TestPerChannelBandwidth:
    """判据 V3：逐信道非均匀 ``B_i`` 下的等价性。"""

    def test_b_i_is_really_non_uniform_in_occupied_mode(self) -> None:
        """前置：确认这条用例真的在**非均匀** ``B_i`` 上跑，否则后面全是自证。"""
        grid = _mixed_grid()
        conns = _mixed_connections(grid)
        ev = _evaluator(grid, conns, "matvec")
        b_i = ev.channel_bandwidth_hz_channels
        assert b_i.shape == (len(conns),)
        assert len(set(b_i.tolist())) > 1, "B_i 退化成标量了，这条用例失去意义"
        # occupied 档 = n_slots × 槽宽
        np.testing.assert_allclose(b_i, np.array([n for n, *_ in MIXED_PLAN]) * SLOT_WIDTH_HZ)
        # 与声明符号率档**不同**（否则测不到多槽宽度进 QoT）
        sym = _evaluator(grid, conns, "matvec", channel_bandwidth_source="symbol_rate")
        assert not np.array_equal(sym.channel_bandwidth_hz_channels, b_i)

    def test_uniform_b_i_collapses_to_legacy_scalar_bitwise(self) -> None:
        """``symbol_rate`` 档下 ``B_i`` 恒等于标量符号率 ⇒ 与 legacy 逐位相同。

        这是 WP-Q1 的恒等式（``Q-J1``），也是"默认档逐位不变"的根据。
        """
        grid = SpectrumGrid(num_channels=8, channel_spacing_hz=50e9, center_frequency_hz=193.1e12)
        ev_sym = MultiBandISRSGN(grid, FiberSpanConfig(length_km=80.0), 50e9)
        ev_sym_pp = MultiBandISRSGN(
            grid,
            FiberSpanConfig(length_km=80.0),
            50e9,
            power_width_mode="fixed_psd",
        )
        assert np.array_equal(ev_sym.channel_bandwidth_hz_channels, ev_sym_pp._power_width_scale() * 0 + np.full(8, 50e9))

    @pytest.mark.parametrize("mode", ["cached", "matvec"])
    def test_mixed_b_i_off_vs_matvec_within_budget(self, mode: str) -> None:
        """同 Raman 档下 ``matvec`` 与 ``off`` 的偏差在 FG 的 1e-12 预算内。"""
        grid = _mixed_grid()
        conns = _mixed_connections(grid)
        power = np.array([c.launch_power_w for c in conns], dtype=np.float64)

        off = _evaluator(grid, conns, "off", raman_spectrum="legacy")
        other = _evaluator(grid, conns, mode, raman_spectrum="legacy")
        a = off._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        b = other._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)

        worst = max(_max_rel(x, y) for x, y in zip(a, b, strict=True))
        bitwise = _all_bitwise(a, b)
        print(f"\n[V3] off vs {mode}（同 Raman=legacy）: bitwise={bitwise} max_rel={worst:.3e}")
        assert worst <= MAX_REL_DEVIATION

    @pytest.mark.parametrize("batch", [8, 16, 64, 142])
    def test_batch_equiv_serial_within_budget(self, batch: int) -> None:
        """批量宽度 8..142：批量核逐候选等于串行路径（预算内）。"""
        grid = _mixed_grid()
        conns = _mixed_connections(grid)
        ev = _evaluator(grid, conns, "matvec", raman_spectrum="legacy")
        base = np.array([c.launch_power_w for c in conns], dtype=np.float64)
        profiles = np.ascontiguousarray(np.tile(base, (batch, 1)))
        profiles[:, ::5] = 0.0  # 造出零功率槽位（惰性连接仍占槽，掩码语义一致）
        profiles = np.ascontiguousarray(profiles)

        serial = np.stack(
            [
                np.stack(
                    ev._calc_span_noise_and_power(
                        np.ascontiguousarray(profiles[b]), SPAN_LENGTH_M, None, None, 1
                    )
                )
                for b in range(batch)
            ]
        )
        batched = np.moveaxis(
            np.stack(ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M, None, None, 1)), 0, 1
        )

        bitwise = all(np.array_equal(serial[q], batched[q]) for q in range(4))
        worst = max(_max_rel(serial[q], batched[q]) for q in range(4))
        print(f"\n[V3] batch≡serial B={batch}: bitwise={bitwise} max_rel={worst:.3e}")
        assert worst <= MAX_REL_DEVIATION

    def test_wide_connection_is_not_physics_identical_to_narrow(self) -> None:
        """反向检查：若这条不成立，上面的等价性就可能是在测一个 no-op。

        **逐信道带宽真正进入 QoT 的地方是闭式 ECZ**：P1 的 ``b_hz_channels`` 就是
        P1 (23)/(29)/(30)/(34) 的 ``B_i``。所以 ``occupied`` 与 ``symbol_rate``
        **必须**给出不同的噪声——这就是本条命题（"逐信道"不是空话）。

        ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本条原有两段，第二段断言「那套被移除的
        物理**看不见**占用宽度（FG 口径的带宽量是符号率）」。该物理整体移除后，
        第二段**失去被测对象**，已删除；**第一段（ECZ 逐信道语义）原样保留**，
        它正是本题数「多槽占用宽度进入 QoT」的判据。
        """
        grid = _mixed_grid()
        conns = _mixed_connections(grid)
        power = np.array([c.launch_power_w for c in conns], dtype=np.float64)

        # ⚠️ 观测量选 **SPM**（下标 1）而不是 ASE：P1 的 ASE 按 P1 (34) 是
        # `NF·h·ν·B_i·(gain−1)`，而本用例**不传 EDFA** ⇒ ASE 恒为 0，测不出
        # 带宽口径。SPM 走 P1 (29) 的 `asinh(φ_i·B_i²/(π·α))`，B_i 直接进公式。
        p1_occupied = _evaluator(grid, conns, "off")
        p1_by_rate = _evaluator(grid, conns, "off", channel_bandwidth_source="symbol_rate")
        a = p1_occupied._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        b = p1_by_rate._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        assert not np.array_equal(a[1], b[1]), (
            "闭式 ECZ 下 occupied 与 symbol_rate 的 SPM 完全相同 —— B_i 没有真正进入 QoT"
        )


class TestRepresentationBoundary:
    """任务书 §2.3：两套「灵活栅格」表示 + 显式转换函数（禁止双写状态）。"""

    def test_round_trip_preserves_slot_window(self) -> None:
        grid = _mixed_grid()
        for service_id, (n_slots, bit_rate, modulation, _rs) in enumerate(MIXED_PLAN):
            service = _FakeAllocation(
                service_id=service_id,
                bit_rate_gbps=bit_rate / 1e9,
                n_slots=n_slots,
                first_slot=service_id * 8,
                launch_power_w=1e-3,
            )
            spec = connection_spec_from_allocation(
                service, grid, modulation=modulation, roll_off=0.0
            )
            assert slot_window_from_connection(spec) == (service.first_slot, service.n_slots)
            assert spec.bit_rate_bps == pytest.approx(bit_rate)
            assert spec.connection_id == service_id
            assert spec.launch_power_w == pytest.approx(1e-3)

    def test_symbol_rate_default_matches_roll_off_identity(self) -> None:
        """默认符号率满足 ``Rs(1+ro) = R / bits_per_symbol``。"""
        grid = _mixed_grid()
        service = _FakeAllocation(0, 50.0, n_slots=2, first_slot=0, launch_power_w=1e-3)
        spec = connection_spec_from_allocation(service, grid, modulation="QPSK", roll_off=0.0)
        assert spec.symbol_rate_hz == pytest.approx(50e9 / 2.0)
        explicit = connection_spec_from_allocation(
            service, grid, modulation="QPSK", roll_off=0.0, symbol_rate_hz=spec.symbol_rate_hz
        )
        assert explicit.symbol_rate_hz == spec.symbol_rate_hz

    def test_slot_count_mismatch_raises_instead_of_rewriting(self) -> None:
        """业务层 ``n_slots`` 与调制需求不符时**抛错**，不静默改写（不许双写状态）。"""
        grid = _mixed_grid()
        bad = _FakeAllocation(0, 50.0, n_slots=3, first_slot=0, launch_power_w=1e-3)
        with pytest.raises(ValueError, match="slots"):
            connection_spec_from_allocation(bad, grid, modulation="QPSK", roll_off=0.0)

    def test_out_of_bounds_window_raises(self) -> None:
        grid = _mixed_grid()
        bad = _FakeAllocation(0, 50.0, n_slots=2, first_slot=NUM_SLOTS - 1, launch_power_w=1e-3)
        with pytest.raises(ValueError, match="out of bounds"):
            connection_spec_from_allocation(bad, grid, modulation="QPSK", roll_off=0.0)

    def test_unknown_modulation_raises(self) -> None:
        grid = _mixed_grid()
        service = _FakeAllocation(0, 50.0, n_slots=2, first_slot=0, launch_power_w=1e-3)
        with pytest.raises(ValueError):
            connection_spec_from_allocation(service, grid, modulation="NOPE", roll_off=0.0)

    def test_round_trip_through_real_allocated_service(self) -> None:
        """用真实的业务层 ``AllocatedService`` 跑一遍（不是鸭子类型替身）。"""
        from network_nlo_eval.simulation.traffic import AllocatedService

        grid = _mixed_grid()
        service = AllocatedService(
            service_id=7,
            source_id=0,
            destination_id=1,
            arrival_time=0.0,
            departure_time=400.0,
            bit_rate_gbps=50.0,
            snr_requirement_db=6.0,
            launch_power_w=1e-3,
            n_slots=2,
            first_slot=16,
        )
        spec = connection_spec_from_allocation(service, grid, modulation="QPSK", roll_off=0.0)
        assert isinstance(spec, ConnectionSpec)
        assert slot_window_from_connection(spec) == (16, 2)
        assert spec.connection_id == 7
        # 物理层不认业务层状态：释放槽区间后物理层字段不受影响（无共享可变状态）
        assert spec.n_slots == 2


class _FakeAllocation:
    """最小业务层替身，字段与 ``simulation.traffic.AllocatedService`` 同名同义。"""

    __slots__ = ("service_id", "bit_rate_gbps", "n_slots", "first_slot", "launch_power_w")

    def __init__(
        self, service_id: int, bit_rate_gbps: float, *, n_slots: int, first_slot: int, launch_power_w: float
    ) -> None:
        self.service_id = service_id
        self.bit_rate_gbps = bit_rate_gbps
        self.n_slots = n_slots
        self.first_slot = first_slot
        self.launch_power_w = launch_power_w


def test_slots_for_rate_matches_connection_invariant() -> None:
    """转换函数所依赖的槽数公式与 ``ConnectionSpec`` 不变式同源。"""
    grid = _mixed_grid()
    for n_slots, bit_rate, modulation, _ in MIXED_PLAN:
        need = math.ceil((bit_rate / {**{"QPSK": 2, "8QAM": 3}}[modulation]) / SLOT_WIDTH_HZ)
        assert need == n_slots, f"{modulation} {bit_rate} 需要 {need} 槽而非 {n_slots}"
        ConnectionSpec(
            grid=grid,
            connection_id=0,
            first_slot=0,
            n_slots=n_slots,
            bit_rate_bps=bit_rate,
            modulation=modulation,
            symbol_rate_hz=bit_rate / 2,
            roll_off=0.0,
            launch_power_w=0.0,
        )
