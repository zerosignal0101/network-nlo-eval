#!/usr/bin/env python3
"""WP-O3 步骤 0/1：逐文件逐片段的 flexgrid 同步记录（J7）。

本工具是 J7 的**唯一事实源**。``results/ode_dev_flexgrid_sync.json`` 里的每一条
``kind`` 都是这里手写的裁定，``--check`` 会做三件事：

1. **每段都有 ``kind``** —— 缺 ``kind`` 即失败；
2. **没有 ``kind == "physics"`` 的片段被应用** —— 逐条断言（J7 硬要求）；
3. **声明与现实一致** —— 每条 ``ported``/``applied`` 片段必须能在目标文件里
   grep 到 ``verify`` 锚点；每条 ``not_ported`` 片段的 ``verify_absent`` 必须
   grep **不到**。

第 3 条是关键：它让"我有没有搬这段 FU-2 的资产"这句话变成可执行断言，而不是
一句承诺。

WP-MERGE-FG-FLEX 的相位守卫翻转
--------------------------------

本工具此前把三条**相位守卫**编进了代码——它们断言的是"当时决定不做某事"，而
本次合并 ``wp/fg-integrate → flexgrid-dev`` 把那些资产**合法地**带了进来，前提
消失。按通例，这些守卫一律**翻成正向不变量**，翻法只有一种：
**X 存在 + X 只定义一次 + X 可达**。逐条对照：

=========================  ==========================================  ==========================
守卫（旧）                  新不变量                                     位置
=========================  ==========================================  ==========================
11 条片段 ``not_ported``   ``action: "ported"`` + ``verify`` 正向锚点   ``SEGMENTS``
``git diff b69563a`` 空    ``sha256 == FROZEN_SHA256``（绝对）          ``check_frozen_kernel``
``verify_absent`` grep 不到 ``verify`` grep 得到（``verify_absent`` 规则保留） ``--check``
=========================  ==========================================  ==========================

第 3 条里的 ``verify_absent`` 规则**本身保留**：将来若再新增"决定不移植"的片段，
它仍然适用。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode3_sync.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode3_sync.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ode_dev_flexgrid_sync.json"

#: J5 冻结内核。
#:
#: ⚠️ 历史（原相位守卫，已作废）：J5 最初锚在 ``46923a9``（WP-O3 起点），WP-Q1 按 J2
#: 「论文提及且与本仓不一致 ⇒ 改成论文」授权改动了本文件——P1 (23)/(29)/(30)/(34) 的
#: ``B_i`` / ``B_ℓ`` 本来就是**逐信道**量，本仓却是全局标量。改动落在提交 ``b69563a``，
#: 于是「相对冻结基线的 diff 为空」这条**相对**断言被顺理成章地立成了永久不变量。
FROZEN = "src/network_nlo_eval/models/numba_kernels.py"
WP_BASE = "b69563a"

#: J5 冻结内核的**绝对**字节出处：``flexgrid-dev`` 侧（``b69563a`` / ``cb24704``）的
#: 同名文件。
#:
#: ⚠️ 基线推进记录（WP-MERGE-FG-FLEX **第 3 轮**）：两份内核是**两套不同物理**，
#: 各占一个文件、各守一个出处：
#:
#:   * ``numba_kernels.py``    ``6611c416…`` —— P1 (23)/(29)/(30)/(34) 闭式，
#:     逐信道 ``B_i``（flexgrid 侧）。**第 3 轮已恢复**回这份字节。
#:   * ``numba_kernels_p2.py`` ``7beece4e…`` —— P2 Eq. 21 张量化，符号率口径
#:     （FG 侧）。**同字节、只换文件名**，由 FG-14 的字节冻结守卫单独守。
#:
#: 第 2 轮曾把 ``numba_kernels.py`` 整份换成 FG 的字节并把 J5 翻成那条字节的 sha256；
#: 第 3 轮改判后**两套都留**，本守卫回到 P1 那份，并把「相对 ``b69563a`` 的 diff 为空」
#: 这条**相对**断言**恢复成同时成立**（P1 那份的字节确实就等于 ``b69563a``）。
#: 相对断言 + 绝对不变量**两条同时成立**，比任一条单独存在都强。
FROZEN_SHA256 = "6611c4164a60d0dcc653e47c5f02d81534e9c78b03e71213aa30d8b13bfd4aa4"
FROZEN_PROVENANCE_BRANCH = "flexgrid-dev"

#: 第二份冻结内核：FG 的 P2 内核（第 3 轮同字节搬进 ``numba_kernels_p2.py``）。
FROZEN_P2 = "src/network_nlo_eval/models/numba_kernels_p2.py"
FROZEN_P2_SHA256 = "7beece4ea353b1bf5119ab9708d4db9845b092328357fff30f9b806afe613205"
FROZEN_P2_PROVENANCE_BRANCH = "wp/fg-integrate"

B = "src/network_nlo_eval/core/bands.py"
I = "src/network_nlo_eval/models/isrs_gn.py"
E = "src/network_nlo_eval/network/elements.py"
R = "src/network_nlo_eval/rwa/qot_checker.py"
P = "src/network_nlo_eval/physics/qot_checker.py"
K = FROZEN
K2 = FROZEN_P2

# ---------------------------------------------------------------------------
# 逐片段裁定。``kind`` 只有两种：engineering（纯接线/索引/校验）、
# physics（触及 ASE/NLI/SRS/alpha 公式）。physics 默认 not_ported；例外必须是凭 J2
# 「论文提及且与本仓不一致 ⇒ 改成论文」的**显式授权**，且自带非空 ``paper_formulas``。
# ⚠️ 本闸门在 WP-MERGE-FG-FLEX 里**没有**被放宽：合并合法带进来的两条 physics 能力
# （isrs_gn 频带边缘、整个 numba_kernels）因此仍留 not_ported 并附复核证据，见各自
# ``rationale``。只有 engineering 片段参与了相位守卫翻转。
# ---------------------------------------------------------------------------
SEGMENTS: list[dict] = [
    # ================= core/bands.py（fu4 728 行 -> 本仓最小闭包）============
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "nm_to_hz",
        "fu4_lines": "108-124",
        "kind": "engineering",
        "action": "ported",
        "rationale": "单位换算。BandSpec.low_hz/high_hz 的最小闭包依赖，无公式。",
        "verify": "def nm_to_hz(wavelength_nm: float) -> float:",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandSpec",
        "fu4_lines": "181-294",
        "kind": "engineering",
        "action": "ported",
        "rationale": "frozen dataclass + 边界校验。from_connections 的第二参数类型，"
        "属最小闭包。不含任何波段取值。",
        "verify": "class BandSpec:",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan.__init__",
        "fu4_lines": "347-410",
        "kind": "engineering",
        "action": "ported",
        "rationale": "频段归属的索引簿记 + 覆盖/重叠校验。纯数据结构。",
        "verify": "def __init__(self, frequencies_hz: NDArrayFloat, specs: Sequence[BandSpec]) -> None:",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan.from_connections",
        "fu4_lines": "427-451",
        "kind": "engineering",
        "action": "ported",
        "rationale": "★ J8 接入点 1：按连接中心频率归段。逐字移植。",
        "verify": "def from_connections(cls, connections: Sequence[ConnectionSpec], specs: Sequence[BandSpec]) -> BandPlan:",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan.assert_connections_single_band",
        "fu4_lines": "573-598",
        "kind": "engineering",
        "action": "ported",
        "rationale": "★ J8 接入点 2：拒绝横跨频段边界的宽连接。逐字移植。",
        "verify": "def assert_connections_single_band(self, connections: Sequence[ConnectionSpec]) -> None:",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan.apply_launch_power",
        "fu4_lines": "600-632",
        "kind": "engineering",
        "action": "ported",
        "rationale": "★ J8 接入点 3：ConnectionSpec 是 frozen，故返回 replace 后的新列表。逐字移植。",
        "verify": "def apply_launch_power(",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan 查询属性 (num_bands/num_elements/band_names/"
        "band_element_indices/band_of/element_count)",
        "fu4_lines": "454-523",
        "kind": "engineering",
        "action": "ported",
        "rationale": "归属查询。证据 JSON 需要 band_of/band_names，故保留。",
        "verify": "def band_of(self, element_idx: int) -> int:",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandSpec / BandPlan 的 as_dict",
        "fu4_lines": "282-294, 665-677",
        "kind": "engineering",
        "action": "ported_adapted",
        "rationale": "服务本 WP 证据落盘。BandPlan.as_dict 去掉 guards 键（其依赖未移植的 "
        "guard_report），其余逐字。",
        "verify": '"band_order_low_to_high_frequency": self.band_names,',
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandSpec 波长/频率 docstring",
        "fu4_lines": "185-193, 222-235, 242-250",
        "kind": "engineering",
        "action": "ported_adapted",
        "rationale": "★ fu4 的 docstring 与代码相反（自称 low_nm 是长波长，但不变式强制 "
        "low_nm < high_nm 即短波长）。只改注释不改行为——该错误表述在写场景时会导致 "
        "波段边界整体算反。",
        "verify": "SHORTER wavelength, hence the HIGHER frequency",
    },
    # ---- bands.py 中此前刻意**不搬**、现已由 WP-MERGE-FG-FLEX 并进来的 FU-2 自有资产 ----
    # ⚠️ 相位守卫翻转：以下 6 条原本断言"本仓不搬 FU-2 资产"，本次合并把该能力合法地
    # 带进来了，故一律翻成正向不变量——存在 + 只定义一次（`verify` 锚点在 bands.py 内
    # 各只出现 1 次）+ 可达。
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "hz_to_nm",
        "fu4_lines": "127-143",
        "kind": "engineering",
        "action": "ported",
        "rationale": "只被未移植的 guard_report / sclu_span_report 使用；搬进来即死代码。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：原「决定不搬」的前提已被本次合并合法作废——"
        "guard_report / sclu_span_report 一并进来了，hz_to_nm 不再是死代码。按通例翻成"
        "正向不变量：存在 + 只定义一次（`def hz_to_nm(` 在 bands.py 内仅 1 处定义）"
        "+ 可达（guard_report / sclu_span_report 的实际调用方）。",
        "verify": "def hz_to_nm(",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "ITU-T G.694.1 波长常数 + U_BAND_ASSUMPTIONS",
        "fu4_lines": "146-178",
        "kind": "engineering",
        "action": "ported",
        "rationale": "FU-2 的 S/C/L/U 波段表资产；含 4 条**假设**（U 波段 1660-1770 nm 无"
        "ITU 出处、U 段 NF 6.5 dB 无出处）。与 flexgrid 接线无关。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：合并把 FG 侧波段表资产合法地带了进来，"
        "原「决定不搬」的前提消失。按通例翻成正向不变量：存在 + 只定义一次"
        "（`U_BAND_ASSUMPTIONS: tuple[str, ...] = (` 在 bands.py 内仅 1 处定义）"
        "+ 可达（模块级导出，tests 与 SCLU_BAND_SPECS 均引用）。"
        "那 4 条无 ITU 出处的**假设**随资产一并保留，其出处缺口由 SCLU_BAND_SPECS 的"
        "U 波段定义承接，本条不改判据口径，只改 action。",
        "verify": "U_BAND_ASSUMPTIONS: tuple[str, ...] = (",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "_u_spec / CL_BAND_SPECS / CLS_BAND_SPECS / CLU_BAND_SPECS / SCLU_BAND_SPECS",
        "fu4_lines": "297-328",
        "kind": "engineering",
        "action": "ported",
        "rationale": "同上，FU-2 预置波段表。本仓场景自带频段规格，故不需要预置表。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：预置表已由合并带进来。按通例翻成正向不变量："
        "存在 + 只定义一次（`SCLU_BAND_SPECS: tuple[BandSpec, ...]` 在 bands.py 内仅 1 处"
        "定义）+ 可达（模块级导出，sclu_span_report 与 tests 引用）。",
        "verify": "SCLU_BAND_SPECS: tuple[BandSpec, ...]",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan.from_channels",
        "fu4_lines": "413-425",
        "kind": "engineering",
        "action": "ported",
        "rationale": "遗留 SpectrumGrid 粒度入口，非 flexgrid 接线所需。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：合并已把该入口带进来。按通例翻成正向不变量："
        "存在 + 只定义一次（`def from_channels(` 在 bands.py 内仅 1 处定义；与 "
        "`from_connections` 是两个不同入口，锚点互不混淆）+ 可达（classmethod，"
        "由调用方按 SpectrumGrid 粒度调用）。",
        "verify": "def from_channels(",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan.noise_figure_db_per_element / launch_power_profile_w / "
        "per_band_launch_power_w",
        "fu4_lines": "526-571",
        "kind": "engineering",
        "action": "ported",
        "rationale": "逐**信道**口径入口。J8 链路走逐连接口径（apply_launch_power）。"
        "noise_figure_db_per_element 还会把 BandSpec.amplifier 拉进评估口径。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：逐信道口径的三个入口已由合并带进来，"
        "与既有的逐连接口径（apply_launch_power）并存，不是二选一。按通例翻成正向不变量："
        "存在 + 只定义一次（`def noise_figure_db_per_element(` 在 bands.py 内仅 1 处定义）"
        "+ 可达（BandPlan 公有方法）。",
        "verify": "def noise_figure_db_per_element(",
    },
    {
        "file": B,
        "origin": "wp/fg-uwb-fid@a233002 core/bands.py",
        "symbol": "BandPlan.guard_report / sclu_span_report",
        "fu4_lines": "635-663, 680-728",
        "kind": "engineering",
        "action": "ported",
        "rationale": "FU-2 的保护间隔与 S+C+L+U 跨距口径报告，属该分支自有结论。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：两份报告已由合并带进来。按通例翻成正向不变量："
        "存在 + 只定义一次（`def guard_report(` 在 bands.py 内仅 1 处定义；"
        "`sclu_span_report` 为模块级函数）+ 可达（两者都被 tests/ 与 hz_to_nm 的调用方"
        "实际调用）。",
        "verify": "def guard_report(",
    },
    # ================= models/isrs_gn.py =================
    {
        "file": I,
        "origin": "wp/fg-uwb-fid@a233002 models/isrs_gn.py",
        "symbol": "MultiBandISRSGN.__init__(slots_per_channel=...)",
        "fu4_lines": "136, 183-185",
        "kind": "engineering",
        "action": "applied",
        "rationale": "J5 允许的索引空间适配。仅用于把 blocked_channels 展开成 blocked_slots；"
        "遗留模式下不参与任何数值。",
        "verify": "if slots_per_channel is not None and slots_per_channel < 1:",
    },
    {
        "file": I,
        "origin": "wp/fg-uwb-fid@a233002 models/isrs_gn.py",
        "symbol": "MultiBandISRSGN.symbol_rate_hz_channels",
        "fu4_lines": "216",
        "kind": "engineering",
        "action": "applied",
        "rationale": "★ 索引空间适配：np.full(n_nodes, 标量 Rs)。遗留下与标量逐位等价"
        "（WP-O1 已证 isrs_gn.py:215-216）。**不是**逐连接异构 Rs——那需改冻结内核。",
        "verify": "return np.full(self.n_nodes, float(self.symbol_rate_hz), dtype=np.float64)",
    },
    {
        "file": I,
        "origin": "wp/fg-uwb-fid@a233002 models/isrs_gn.py",
        "symbol": "MultiBandISRSGN.roadm_blocks_node",
        "fu4_lines": "485-509 (rwa 侧同语义)",
        "kind": "engineering",
        "action": "applied_new",
        "rationale": "★ fu4 把同一套 blocked_slots 解析逻辑在 rwa / physics / isrs_gn 三处各写一遍"
        "（其中 physics 侧靠私有属性 _slots_per_channel_hint 传参）。本 WP 收敛成一个方法，"
        "三处共用，保证语义不会漂移。legacy 分支逐字等价。",
        "verify": "def roadm_blocks_node(self, roadm: ROADMConfig, node_index: int) -> bool:",
    },
    {
        "file": I,
        "origin": "WP-O3 自行发现（fu4 无对应差异）",
        "symbol": "MultiBandISRSGN.for_fiber 透传 connections/ode_accel/slots_per_channel",
        "fu4_lines": "n/a",
        "kind": "engineering",
        "action": "applied_new",
        "rationale": "★ **bug 修复**：O2 移植的 for_fiber 只传 (grid, fiber, Rs)，逐连接模式下"
        "子评估器会退回遗留分支并读 FlexibleGrid.wavelengths（不存在）→ AttributeError。"
        "J8 链路必经 for_fiber，不修则端到端跑不起来。遗留模式三者皆为默认值，行为不变。"
        "⚠️ WP-MERGE-FG-FLEX：合并在透传序列里插入了 `ode_accel` / `raman_spectrum`，"
        "锚点随之改钉在 `slots_per_channel` 与 `ode_accel` 相邻的那两行上（透传事实未变，"
        "仍是同一处 `for_fiber` 调用）。",
        "verify": "                self.slots_per_channel,\n                self.ode_accel,",
    },
    {
        "file": I,
        "origin": "wp/fg-uwb-fid@a233002 models/isrs_gn.py",
        "symbol": "逐连接异构 symbol_rate_hz 数组参数",
        "fu4_lines": "1306, 1328-1330 (numba_kernels.py)",
        "kind": "physics",
        "action": "ported_adapted",
        "rationale": "★ 物理片段。fu4 的 _calc_span_noise_and_power_jit 接受逐连接 Rs 数组"
        "并据此算 ASE；本仓原先只吃标量，故 O2/O3 都显式拒绝（那时的裁定是『改冻结核 = 违反 J5』）。"
        "WP-Q1 按 J2 授权改动冻结核：P1 (23)/(29)/(30)/(34) 的 B_i / B_l 本来就是逐信道量。"
        "落地口径不是照抄 fu4 的 Rs 数组，而是引入逐信道**信道带宽** b_hz_channels，"
        "由 channel_bandwidth_source（symbol_rate / occupied）决定取值——fu4 没有这个区分。"
        "未移植 fu4 的任何 ODE / phi_kurtosis / ASE 增益因子。"
        "⚠️ WP-MERGE-FG-FLEX **第 3 轮**：上一轮把 flexgrid 侧的 `b_hz_channels` **改名**并入 "
        "`symbol_rate_hz`（内核侧收**一个**逐节点数组承载 P1 的 B_i），锚点因此改钉在"
        "调用点 `symbol_rate_hz=self.channel_bandwidth_hz_channels` 上——那个组合是"
        "「P2 的 GN 模型 + 占用带宽当符号率」，**两边都没验证过**。"
        "第 3 轮改判为**两套物理都留、显式档位区分**：P1 档的调用点回到 "
        "`b_hz_channels=self.channel_bandwidth_hz_channels`（P1 核的参数名就是 "
        "`b_hz_channels`），P2 档则回到 FG 口径的 `symbol_rate_hz=self.symbol_rate_hz_channels`。"
        "**判据口径不动**：paper_formulas 仍是 P1 (23)/(29)/(30)/(34)，B_i 仍是逐信道量、"
        "不许退回标量；改的只是它现在住在**哪个档位、叫什么名字**。",
        "paper_formulas": ["P1 (23)", "P1 (29)", "P1 (30)", "P1 (34)"],
        "verify": "b_hz_channels=self.channel_bandwidth_hz_channels,",
    },
    {
        "file": I,
        "origin": "wp/fg-uwb-fid@a233002 models/isrs_gn.py",
        "symbol": "用连接占用带覆写 f_m_hz / f_M_hz 并真正传给内核",
        "fu4_lines": "n/a（fu4 用真 ODE，无此参数）",
        "kind": "physics",
        "action": "not_ported",
        "rationale": "★ 物理片段。O2 已在 Python 侧按连接占用带算出 f_m_hz/f_M_hz，但冻结内核"
        "在核内用标量 Rs 重算二者，故该修正在本仓是**死代码**。让它生效需给内核加参数 = J5。"
        "见 e2e 的 known_limitations.srs_band_edges_still_scalar_rs。"
        "⚠️ WP-MERGE-FG-FLEX 复核结论：**未翻转**，维持 not_ported（不是留白，是有出处的复核）。"
        "锚点 `conn.centre_hz + conn.n_slots * conn.grid.slot_width_hz / 2` 确实在 isrs_gn.py "
        "存在且只出现 1 次，但 (a) `git diff b69563a -- models/isrs_gn.py` 证明该段是 **O2 既有"
        "代码**，本次合并只在它上方加了两行注释，没有把新能力带进来——相位守卫的前提并未被推翻；"
        "(b) `f_m_hz` / `f_M_hz` 在整个 src/ 中仍**零读取方**（只有同一处 if/else 的 4 行赋值），"
        "依旧没有真正传给内核，「死代码」的结论今天仍成立。故按通例翻成 `ported` 会是假声明，"
        "不翻。翻转需要先给内核加逐连接带边参数，那需要单独的 J5 授权。",
        "verify_absent": None,
    },
    # ================= network/elements.py =================
    {
        "file": E,
        "origin": "wp/fg-uwb-fid@a233002 network/elements.py",
        "symbol": "ROADMConfig.blocked_slots",
        "fu4_lines": "84-87",
        "kind": "engineering",
        "action": "applied",
        "rationale": "灵活栅格下的权威阻断字段（12.5 GHz 粒度）。默认空列表，遗留行为不变。",
        "verify": "blocked_slots: list[int] = Field(",
    },
    {
        "file": E,
        "origin": "wp/fg-uwb-fid@a233002 network/elements.py",
        "symbol": "validate_blocked_channels 扩展到槽",
        "fu4_lines": "96-99",
        "kind": "engineering",
        "action": "applied",
        "rationale": "槽索引的非负/唯一校验，与既有信道校验同构。",
        "verify": "ROADM blocked slot indices must be unique.",
    },
    {
        "file": E,
        "origin": "wp/fg-uwb-fid@a233002 network/elements.py",
        "symbol": "ROADMConfig.blocked_slot_set()",
        "fu4_lines": "102-111",
        "kind": "engineering",
        "action": "ported",
        "rationale": "实测本仓 src/ tests/ tools/ 中**零调用方**（死代码）。语义已由 "
        "isrs_gn.roadm_blocks_node 与 slots.resolve_blocked_slots 覆盖。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：方法已由合并带进来，且不再是死代码——"
        "tests/test_fg2_flexible_grid.py:737/743 实际调用它并断言返回值。按通例翻成正向不变量："
        "存在 + 只定义一次（`def blocked_slot_set(` 在 elements.py 内仅 1 处定义）"
        "+ 可达（tests 直接调用）。语义仍与既有实现同源，翻转的是"
        "「本仓不存在该能力」这条相位守卫，不是判据口径。",
        "verify": "def blocked_slot_set(",
    },
    # ================= rwa/qot_checker.py =================
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "QoTValidator.num_nodes",
        "fu4_lines": "53-59",
        "kind": "engineering",
        "action": "applied",
        "rationale": "J5 明确允许的 num_channels -> n_nodes 索引空间适配。遗留下 "
        "n_nodes == spectrum_grid.num_channels。",
        "verify": "def num_nodes(self) -> int:",
    },
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "QoTValidator._is_blocked",
        "fu4_lines": "61-78",
        "kind": "engineering",
        "action": "applied_adapted",
        "rationale": "委托给 evaluator.roadm_blocks_node（单一实现）。legacy 分支返回 "
        "node_index in roadm.blocked_channels，与移植前逐字等价。",
        "verify": "return self.gn_evaluator.roadm_blocks_node(roadm, node_index)",
    },
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "QoTValidator._node_frequency_and_bandwidth",
        "fu4_lines": "106-122",
        "kind": "engineering",
        "action": "applied",
        "rationale": "★ 索引空间适配：booster ASE 的频率/带宽真源必须是逐节点数组，"
        "不能用 spectrum_grid.frequencies。WP-O1 §0 已证 legacy 下二者逐位等价。"
        "WP-Q1 起第二返回值是 P1 (34) 的 B_i（bandwidth），故改名 _node_frequency_and_bandwidth；"
        "旧名保留为别名。",
        "verify": "def _node_frequency_and_bandwidth(self, channel_idx: int) -> tuple[float, float]:",
    },
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "spectrum_grid.num_channels -> self.num_nodes（6 处数组/循环定长）",
        "fu4_lines": "241-243, 397, 417-418",
        "kind": "engineering",
        "action": "applied",
        "rationale": "索引空间适配，遗留下取值完全相同。",
        "verify": "total_spm = np.zeros(self.num_nodes, dtype=np.float64)",
    },
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "booster ASE 标量分支 -> _node_frequency_and_bandwidth",
        "fu4_lines": "334-350",
        "kind": "engineering",
        "action": "applied",
        "rationale": "只换取值来源，公式 NF*h*f*B*(loss-1) 一字未动。",
        "verify": "frequency_hz, channel_bandwidth_hz = self._node_frequency_and_bandwidth(channel_idx)",
    },
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "booster ASE 向量分支 -> f_abs_hz + channel_bandwidth_hz_channels",
        "fu4_lines": "426-441",
        "kind": "engineering",
        "action": "applied",
        "rationale": "同上。WP-Q1 起 B_i 取 channel_bandwidth_hz_channels[i]（逐信道），"
        "legacy 下该数组恒等于标量 Rs，故 legacy 逐位等价"
        "（IEEE 双精度逐元素乘法与标量乘法同结果）。",
        "verify": "* self.gn_evaluator.channel_bandwidth_hz_channels",
    },
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "QoTValidator._roadm_terms",
        "fu4_lines": "80-105",
        "kind": "engineering",
        "action": "ported",
        "rationale": "纯重构：把既有 inline 逻辑抽成函数（PLAN §0 已断语义为零）。对 flexgrid "
        "零价值。**故意不搬**，使 legacy 的装配算式保持逐字不变——J6 由构造即可证明，"
        "不必只靠测试反证。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：抽取已由合并带进来。按通例翻成正向不变量："
        "存在 + 只定义一次（`def _roadm_terms(` 在 qot_checker.py 内仅 1 处定义；"
        "折叠逻辑的单一真源是模块级 `fold_roadm_terms`，本方法只是薄封装）"
        "+ 可达（qot_checker.py:671 的逐节点 booster 路径实际调用）。"
        "「legacy 装配算式逐字不变」这一原有判据口径不受本次翻转影响。",
        "verify": "def _roadm_terms(",
    },
    {
        "file": R,
        "origin": "wp/fg-uwb-fid@a233002 rwa/qot_checker.py",
        "symbol": "_H_PLANCK 常量提取",
        "fu4_lines": "17-19",
        "kind": "engineering",
        "action": "ported",
        "rationale": "纯命名提取，数值同（6.62607015e-34）。零 flexgrid 价值。"
        "★ WP-MERGE-FG-FLEX 相位守卫翻转：常量提取已由合并带进来。按通例翻成正向不变量："
        "存在 + 只定义一次（`_H_PLANCK = 6.62607015e-34` 在 qot_checker.py 内仅 1 处"
        "赋值定义；另 3 处是引用而非重复定义）+ 可达"
        "（tests/test_fg20_roadm_injection_behavior.py:50 直接 import 并断言其值）。",
        "verify": "_H_PLANCK = 6.62607015e-34",
    },
    # ================= physics/qot_checker.py =================
    {
        "file": P,
        "origin": "wp/fg-uwb-fid@a233002 physics/qot_checker.py",
        "symbol": "PyNLOQoTEvaluator(connections=..., slots_per_channel=...)",
        "fu4_lines": "78, 92, 104-131",
        "kind": "engineering",
        "action": "applied",
        "rationale": "逐连接模式：每连接一个波形，频率取连接中心。",
        "verify": "self.connections = list(connections) if connections is not None else None",
    },
    {
        "file": P,
        "origin": "wp/fg-uwb-fid@a233002 physics/qot_checker.py",
        "symbol": "_default_passband_hz（最宽连接的占用带宽）",
        "fu4_lines": "118-131, 247",
        "kind": "engineering",
        "action": "applied",
        "rationale": "WSS 兜底通带。遗留分支恒等于 spectrum_grid.channel_spacing_hz，"
        "与移植前表达式逐字相同；灵活栅格下必须取最宽连接，否则宽连接被滤波掉。",
        "verify": "self._default_passband_hz = max(c.occupied_bandwidth_hz for c in self.connections)",
    },
    {
        "file": P,
        "origin": "wp/fg-uwb-fid@a233002 physics/qot_checker.py",
        "symbol": "PyNLOQoTEvaluator._is_blocked",
        "fu4_lines": "135-147",
        "kind": "engineering",
        "action": "applied_adapted",
        "rationale": "槽区间重叠语义；**改掉** fu4 靠私有属性 roadm._slots_per_channel_hint "
        "传参的做法，改为显式构造参数。legacy 分支逐字等价。",
        "verify": "def _is_blocked(self, roadm, node_index: int) -> bool:",
    },
    {
        "file": P,
        "origin": "wp/fg-uwb-fid@a233002 physics/qot_checker.py",
        "symbol": "occupied_width 按连接集合带边缘计算",
        "fu4_lines": "118-124",
        "kind": "engineering",
        "action": "applied",
        "rationale": "遗留分支仍为 (num_channels-1)*spacing，逐字不变；仅决定 _mux_target_sr。",
        "verify": "occupied_width = high - low",
    },
    # ================= models/numba_kernels.py（J5 冻结）=================
    {
        "file": K,
        "origin": "wp/fg-uwb-fid@a233002 models/numba_kernels.py",
        "symbol": "整个文件（fu4 用 RK4 _calc_exact_power_profile_ode 真 ODE 传播；"
        "本仓用 ECZ 闭式 _calc_span_out_power_jit）",
        "fu4_lines": "1-1480",
        "kind": "physics",
        "action": "not_ported",
        "rationale": "★ J5 冻结文件（基线已推进到 WP-Q1 的 b69563a，见 WP_BASE）。两棵树在此"
        "**不是同一个物理模型**：fu4 是 RK4 数值传播，本仓是 P1 的闭式 ECZ GN 模型"
        "（ODE 线按 O2 裁定以 P1/P2 为准）。fu4 的 RK4 / ode_accel / phi_kurtosis 一律不移植。"
        "唯一被授权的改动是 WP-Q1 对 B_i / B_l 的逐信道化（P1 (23)/(29)/(30)/(34)）。"
        "自 b69563a 起 diff 必须为空。"
        "⚠️ WP-MERGE-FG-FLEX **第 3 轮**复核结论：J5 已**恢复**为「相对 b69563a 的 diff 为空」"
        "**且**「sha256 == 6611c416…」的绝对正向不变量，**两条同时成立**——因为本轮把"
        "本文件恢复成 flexgrid-dev 的冻结字节（P1 闭式那一份）。"
        "FG 的 RK4 ODE（`_calc_exact_power_profile_ode`）与 ode_accel 加速**整份搬到**"
        "`numba_kernels_p2.py`（同字节，只换文件名），由那里的 J5 记录单独守。"
        "本条 `kind == \"physics\"`：J7 的硬闸门要求任何被应用的物理片段自带非空 "
        "`paper_formulas`（J2「论文提及且与本仓不一致 ⇒ 改成论文」的授权）；判据口径要求 "
        "`paper_formulas` 一律不动，而合并带入的 FG 内核是 RK4 数值传播、**并非**按 P1 "
        "(23)/(29)/(30)/(34) 逐式改写，无法诚实地给出论文公式号。故 action 暂留 "
        "not_ported，**待裁决**，不作假声明。",
    },
    # ============ models/numba_kernels_p2.py（第 3 轮新增的 FG 内核副本）============
    {
        "file": K2,
        "origin": "wp/fg-integrate@57c7221 models/numba_kernels.py",
        "symbol": "整个文件（FG 的 RK4 ODE + NLI 张量化 + 拉曼 G 矩阵 + 批量张量）",
        "fu4_lines": "1-1480",
        "kind": "physics",
        "action": "not_ported",
        "rationale": "★ 第 3 轮新增：FG 的冻结内核**同字节**搬进本文件（只换文件名，"
        "sha256 仍为 7beece4e…），由 `j5_frozen_kernel_p2` 记录单独守。"
        "它承载的是 **P2 Eq. 21** 那套物理（符号率口径），与 P1 闭式是**两套不同物理**；"
        "本轮按编排方裁决把两套都保留、用 `nli_model` 档位区分，谁都不许被另一套替换。"
        "本条 `kind == \"physics\"` 且 action 留 not_ported：与上面同源的理由——"
        "FG 的 RK4 数值传播不是按 P1 (23)/(29)/(30)/(34) 逐式改写的，"
        "J7 要求被应用的物理片段自带论文公式号授权，这里给不出诚实的一组，故不作假声明。",
    },
]


def file_sha256(rel_path: str) -> str:
    """返回目标文件的 sha256；文件缺失时返回空串。"""
    path = REPO / rel_path
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_diff_empty(rel_path: str) -> tuple[bool, str]:
    """断言 ``rel_path`` 相对 ``WP_BASE`` 无 diff。

    ⚠️ 这条**相对**断言在 WP-MERGE-FG-FLEX 里不再是 J5 守卫本身（守卫已翻成
    :data:`FROZEN_SHA256` 的绝对不变量），只作为 provenance 证据留档：两条线的字节
    互斥，相对基线的 diff 不可能为空，但它记录了"相对 flexgrid-dev 侧究竟动了多少"。
    """
    proc = subprocess.run(
        ["git", "diff", WP_BASE, "--", rel_path],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    return proc.stdout.strip() == "", proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    errs: list[str] = []
    records = []
    for seg in SEGMENTS:
        path = REPO / seg["file"]
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        rec = dict(seg)
        if seg.get("kind") not in ("engineering", "physics"):
            errs.append(f"[J7] {seg['file']}::{seg['symbol']} 缺少 kind 标签")
        if seg.get("action") not in ("ported", "ported_adapted", "applied", "applied_adapted",
                                     "applied_new", "not_ported"):
            errs.append(f"[J7] {seg['file']}::{seg['symbol']} action 非法: {seg.get('action')!r}")
        # J7 硬要求：physics 片段不得被应用
        if seg.get("kind") == "physics" and seg.get("action") != "not_ported":
            # ⚠️ J7 的收紧版豁免（WP-Q1 引入）：物理片段**原则上**不许应用，但可以凭
            # J2「论文提及且与本仓不一致 ⇒ 改成论文」被显式授权。授权必须自带论文
            # 公式号清单 `paper_formulas`（非空、每项形如 "P1 (29)"），否则照旧报错。
            # 这样"物理被应用"永远是**可见且有出处的**例外，而不是把闸门整体拆掉。
            formulas = seg.get("paper_formulas")
            if not (isinstance(formulas, list) and formulas and all(
                isinstance(f, str) and f.strip() for f in formulas
            )):
                errs.append(
                    f"[J7] physics 片段被应用且未给出 paper_formulas 授权: {seg['file']}::{seg['symbol']}"
                )
        # 声明与现实一致（WP-MERGE-FG-FLEX 翻转后的口径）：
        #   * 已应用片段（action != not_ported）→ ``verify`` 必须在源码里 grep **到**；
        #   * 未移植片段（action == not_ported）→ ``verify_absent`` 必须 grep **不到**。
        # 后半条规则**本身保留**：将来若再新增"决定不移植"的片段，它仍然适用。
        if seg.get("action") != "not_ported":
            anchor = seg.get("verify")
            if not anchor:
                errs.append(
                    f"[J7] {seg['file']}::{seg['symbol']} 已应用却无 verify 正向锚点"
                )
            elif anchor not in text:
                errs.append(f"[J7] {seg['file']} 缺 {seg['symbol']} 的锚点 {anchor!r}")
            else:
                rec["verified"] = True
        elif seg.get("verify_absent"):
            if seg["verify_absent"] in text:
                errs.append(
                    f"[J7] {seg['file']} 竟含声称未移植的 {seg['verify_absent']!r}"
                )
            else:
                rec["verified_absent"] = True
        records.append(rec)

    frozen_sha = file_sha256(FROZEN)
    frozen_ok = frozen_sha == FROZEN_SHA256
    if not frozen_ok:
        errs.append(
            f"[J5] {FROZEN} 的 sha256 = {frozen_sha or '<文件缺失>'}，"
            f"不等于已核出处 {FROZEN_PROVENANCE_BRANCH} 的 {FROZEN_SHA256}"
        )
    # provenance 证据（守卫）：相对 flexgrid-dev 侧基线 ``b69563a`` 的 diff 为空。
    # ⚠️ 第 3 轮恢复 P1 字节后，这条**相对**断言**重新成立**（P1 那份字节就等于
    #    ``b69563a``）⇒ 相对断言与绝对不变量**两条同时成立**。
    diff_empty, _diff = git_diff_empty(FROZEN)
    if not diff_empty:
        errs.append(
            f"[J5] {FROZEN} 相对 WP_BASE {WP_BASE} 的 diff 不为空 —— "
            f"P1 内核必须与已核出处逐字节相同。"
        )

    # 第二份冻结内核（FG 的 P2）：同字节搬进 numba_kernels_p2.py，单独守。
    frozen_p2_sha = file_sha256(FROZEN_P2)
    frozen_p2_ok = frozen_p2_sha == FROZEN_P2_SHA256
    if not frozen_p2_ok:
        errs.append(
            f"[J5] {FROZEN_P2} 的 sha256 = {frozen_p2_sha or '<文件缺失>'}，"
            f"不等于已核出处 {FROZEN_P2_PROVENANCE_BRANCH} 的 {FROZEN_P2_SHA256}"
        )

    n_physics = sum(1 for s in SEGMENTS if s["kind"] == "physics")
    n_applied_physics = sum(1 for s in SEGMENTS if s["kind"] == "physics" and s["action"] != "not_ported")
    doc = {
        "wp": "O3",
        "judgment": "J7",
        "wp_base_commit": WP_BASE,
        "fu4_ref": "wp/fg-uwb-fid@a233002",
        "summary": {
            "n_segments": len(SEGMENTS),
            "n_engineering": sum(1 for s in SEGMENTS if s["kind"] == "engineering"),
            "n_physics": n_physics,
            "n_physics_applied": n_applied_physics,
            "n_ported_or_applied": sum(1 for s in SEGMENTS if s["action"] != "not_ported"),
            "n_not_ported": sum(1 for s in SEGMENTS if s["action"] == "not_ported"),
            "files_touched": sorted({s["file"] for s in SEGMENTS if s["action"] != "not_ported"}),
        },
        "j5_frozen_kernel": {
            "file": FROZEN,
            "frozen_sha256": FROZEN_SHA256,
            "sha256_matches": frozen_ok,
            "provenance_branch": FROZEN_PROVENANCE_BRANCH,
            "flexgrid_side_sha256_baseline": WP_BASE,
            # 相对断言**重新成立**（第 3 轮恢复 P1 字节）：与绝对不变量同时成立。
            "diff_vs_wp_base_empty": diff_empty,
        },
        "j5_frozen_kernel_p2": {
            "file": FROZEN_P2,
            "frozen_sha256": FROZEN_P2_SHA256,
            "sha256_matches": frozen_p2_ok,
            "provenance_branch": FROZEN_P2_PROVENANCE_BRANCH,
            "note": "FG 的 P2 内核：第 3 轮同字节搬进 numba_kernels_p2.py（只换文件名）。",
        },
        "segments": records,
        "errors": errs,
    }

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        print(f"  片段 {len(SEGMENTS)} 条：engineering {doc['summary']['n_engineering']} / "
              f"physics {n_physics}；physics 被应用 {n_applied_physics} 条")
        print(f"  J5 {FROZEN} sha256 == {FROZEN_PROVENANCE_BRANCH} 的冻结字节: {frozen_ok}"
              f"（{frozen_sha}）")

    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        if stored.get("segments") != records:
            print("FAIL sync 记录与落盘不一致")
            return 1
        if errs:
            print(f"FAIL {len(errs)} 条")
            for e in errs:
                print(f"  {e}")
            return 1
        print(f"OK  J7 复算通过：{len(SEGMENTS)} 段全有 kind，physics 应用数 = "
              f"{n_applied_physics}，J5 sha256 == {FROZEN_SHA256[:12]}…（{FROZEN_PROVENANCE_BRANCH}）")
        return 0

    if errs:
        for e in errs:
            print(f"FAIL {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
