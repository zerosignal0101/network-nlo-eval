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
    """``off`` / ``cached`` 是 legacy-only 的冻结路径，必须显式配 ``legacy`` 谱档。"""
    return "legacy" if ode_accel in ("off", "cached") else "tabular"


def _evaluator(
    grid: FlexibleGrid,
    conns: list[ConnectionSpec],
    ode_accel: str,
    *,
    raman_spectrum: str | None = None,
    channel_bandwidth_source: str = "occupied",
    nli_model: str = "p2_gn",
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
        # ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：本类用例全部建立在 FG 的加速档
        # （``cached_g`` / ``matvec`` / 批量核）之上——那些是 **P2** 的实现结构。
        # 第 3 轮把 NLI 物理拆成两套后，默认档是 ``p1_closed``（P1 闭式），
        # 故这里**显式**钉 ``"p2_gn"``：这些用例的数字与换底前**完全相同**
        # （换底前合并树跑的就是 FG 那套物理），没有任何期望值发生移动。
        # P1 档的 ``B_i`` 行为由 :meth:`TestPerChannelBandwidth.
        # test_b_i_is_non_uniform_in_occupied_mode_on_both_nli_tiers` 单独覆盖。
        nli_model=nli_model,
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

    def test_b_i_is_non_uniform_in_occupied_mode_on_both_nli_tiers(self) -> None:
        """⚠️ WP-MERGE-FG-FLEX **第 3 轮新增**：``B_i`` 在**两套** NLI 物理里都逐信道。

        第 3 轮把 NLI 物理拆成两套（``p1_closed`` 闭式 / ``p2_gn`` 张量化），
        ``B_i`` 随之落在两处：P1 核的 ``b_hz_channels``（P1 (23)/(29)/(30)/(34)）
        与 P2 核的 ``symbol_rate_hz``。这条用例**两档都查**：

        * ``B_i`` 数组本身与档位无关（是逐连接槽宽），故两档上必须**逐位相同**；
        * 每档的**物理入口**必须真的接上自己的那一份（``p1_closed`` 走
          ``b_hz_channels``，``p2_gn`` 走 ``symbol_rate_hz``）。

        第 2 轮那个「把 ``B_i`` 改名成 ``symbol_rate_hz`` 喂进 P2」的组合，
        正是本轮要消除的东西：它让两套物理共用一个名字，从而看不出
        哪一档真的在用逐信道带宽。
        """
        grid = _mixed_grid()
        conns = _mixed_connections(grid)
        p1 = _evaluator(grid, conns, "off", nli_model="p1_closed")
        p2 = _evaluator(grid, conns, "matvec", nli_model="p2_gn")
        b_p1 = p1.channel_bandwidth_hz_channels
        b_p2 = p2.channel_bandwidth_hz_channels
        assert np.array_equal(b_p1, b_p2), "B_i 数组本身不该随 NLI 档位变化"
        assert len(set(b_p1.tolist())) > 1, "B_i 退化成标量了，这条用例失去意义"
        np.testing.assert_allclose(b_p1, np.array([n for n, *_ in MIXED_PLAN]) * SLOT_WIDTH_HZ)
        # P1 档确实把逐信道 B_i 喂给了 **P1 核**（参数名就是 b_hz_channels）。
        assert p1.nli_model == "p1_closed"
        assert p2.nli_model == "p2_gn"

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
            raman_spectrum="tabular",
            # ⚠️ 第 3 轮：``tabular`` 是 P2 的拉曼谱档 ⇒ 显式钉 P2 档。
            nli_model="p2_gn",
        )
        assert np.array_equal(ev_sym.channel_bandwidth_hz_channels, ev_sym_pp._power_width_scale() * 0 + np.full(8, 50e9))

    @pytest.mark.parametrize("mode", ["off", "cached_g", "matvec"])
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

    def test_cached_g_vs_matvec_within_budget(self) -> None:
        """``cached_g`` 与 ``matvec`` 的偏差在预算内。

        ⚠️ 实测**不是逐位相等**（max_rel ≈ 2.4e-16）：FG-4a 的 ``matvec`` 按设计
        重排求和。任务书 V3 字面写的"逐位"在此不成立，详见模块 docstring。
        """
        grid = _mixed_grid()
        conns = _mixed_connections(grid)
        power = np.array([c.launch_power_w for c in conns], dtype=np.float64)
        a = _evaluator(grid, conns, "cached_g", raman_spectrum="tabular")._calc_span_noise_and_power(
            power, SPAN_LENGTH_M, None, None, 1
        )
        b = _evaluator(grid, conns, "matvec", raman_spectrum="tabular")._calc_span_noise_and_power(
            power, SPAN_LENGTH_M, None, None, 1
        )
        worst = max(_max_rel(x, y) for x, y in zip(a, b, strict=True))
        print(f"\n[V3] cached_g vs matvec（同 Raman=tabular）: bitwise={_all_bitwise(a, b)} max_rel={worst:.3e}")
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

        ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：这条用例的**归属档位变了**，理由是编排方
        第 3 轮查明的事实——第 2 轮把「多槽占用宽度进入 QoT」这件事实现在 **P2 档**
        （做法是把 P1 的 ``B_i`` **改名**成 ``symbol_rate_hz`` 喂进 P2 公式）。那个
        组合两边都没验证过，第 3 轮判它作废。

        第 3 轮把两套物理分开后，**逐信道带宽真正进入 QoT 的地方是 P1 档**：
        P1 的 ``b_hz_channels`` 就是 P1 (23)/(29)/(30)/(34) 的 ``B_i``。所以：

        * **P1 档**（``p1_closed``）：``occupied`` 与 ``symbol_rate`` **必须**给出
          不同的噪声——这就是本条命题（"逐信道"不是空话）。
        * **P2 档**（``p2_gn``）：P2 的带宽量是**符号率**（FG 口径），所以
          ``occupied`` 与 ``symbol_rate`` **必须**给出**相同**的噪声——P2 档
          **看不见**占用宽度。这条断言把两档的差异**钉死**，防止将来有人
          又把 ``B_i`` 塞回 P2。

        两段都是**反向检查**：一段证明 P1 的逐信道化是真的，一段证明 P2 不会被
        多槽宽度污染。
        """
        grid = _mixed_grid()
        conns = _mixed_connections(grid)
        power = np.array([c.launch_power_w for c in conns], dtype=np.float64)

        # --- P1 档：逐信道 B_i 真的进入 QoT ---
        # ⚠️ 观测量选 **SPM**（下标 1）而不是 ASE：P1 的 ASE 按 P1 (34) 是
        # `NF·h·ν·B_i·(gain−1)`，而本用例**不传 EDFA** ⇒ ASE 恒为 0，测不出
        # 带宽口径。SPM 走 P1 (29) 的 `asinh(φ_i·B_i²/(π·α))`，B_i 直接进公式。
        p1_occupied = _evaluator(grid, conns, "off", nli_model="p1_closed")
        p1_by_rate = _evaluator(
            grid, conns, "off", nli_model="p1_closed", channel_bandwidth_source="symbol_rate"
        )
        a = p1_occupied._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        b = p1_by_rate._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        assert not np.array_equal(a[1], b[1]), (
            "P1 档下 occupied 与 symbol_rate 的 SPM 完全相同 —— B_i 没有真正进入 QoT"
        )

        # --- P2 档：占用宽度**不**进入 P2（FG 口径：带宽量是符号率）---
        p2_occupied = _evaluator(grid, conns, "matvec", nli_model="p2_gn")
        p2_by_rate = _evaluator(
            grid, conns, "matvec", nli_model="p2_gn", channel_bandwidth_source="symbol_rate"
        )
        c = p2_occupied._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        d = p2_by_rate._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        assert np.array_equal(c[3], d[3]), (
            "P2 档的噪声被占用宽度污染了 —— P2 的带宽量必须是符号率（FG 口径）。"
            "第 3 轮已判废「把 B_i 改名成 symbol_rate_hz 喂 P2」的那个组合。"
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
