# WP-O3：flexgrid 全链路同步到 `ode-dev`

- 判据：**J5**（`numba_kernels.py` diff 为空）/ **J6**（legacy 逐位）/ **J7**（逐片段 engineering|physics）/ **J8**（端到端跑通）
- 参照：`wp/fg-uwb-fid@a233002`；本 WP 起点：`b69563a`
- 本文件全部数字由 `tools/ode3_report.py` 从 `results/*.json` 渲染，无手抄。

## 0. 验收结论

| 判据 | 内容 | 结果 |
|---|---|---|
| J5 | `src/network_nlo_eval/models/numba_kernels.py` 逐字节等于已核出处 `wp/fg-integrate` 的冻结字节 | ✅ 通过（sha256 `7beece4ea353b1bf…`） |
| J6 | 6 个 legacy 场景 `delta_db == 0.0` | ✅ 通过（逐位相同 6/6） |
| J7 | 每段都有 kind，且 `kind == physics` 的片段数为 1 | ❌ 失败（共 37 段） |
| J8 | 端到端跑通，≥2 个不同 `n_slots`，逐连接 SNR 落盘 | ✅ 通过（4 个不同 n_slots：[2, 3, 6, 8]） |

## 1. 同步了什么（逐文件逐片段，J7 分类）

共裁定 **37** 段：engineering 34 段、physics 3 段；**被应用的 35 段里，physics 为 1 段**。

### 1.1 已应用（按文件）

| 文件 | 片段 | fu4 行 | kind | 动作 | 理由 |
|---|---|---|---|---|---|
| `core/bands.py` | nm_to_hz | `108-124` | **engineering** | `ported` | 单位换算。BandSpec.low_hz/high_hz 的最小闭包依赖，无公式。 |
| `core/bands.py` | BandSpec | `181-294` | **engineering** | `ported` | frozen dataclass + 边界校验。from_connections 的第二参数类型，属最小闭包。不含任何波段取值。 |
| `core/bands.py` | BandPlan.__init__ | `347-410` | **engineering** | `ported` | 频段归属的索引簿记 + 覆盖/重叠校验。纯数据结构。 |
| `core/bands.py` | BandPlan.from_connections | `427-451` | **engineering** | `ported` | ★ J8 接入点 1：按连接中心频率归段。逐字移植。 |
| `core/bands.py` | BandPlan.assert_connections_single_band | `573-598` | **engineering** | `ported` | ★ J8 接入点 2：拒绝横跨频段边界的宽连接。逐字移植。 |
| `core/bands.py` | BandPlan.apply_launch_power | `600-632` | **engineering** | `ported` | ★ J8 接入点 3：ConnectionSpec 是 frozen，故返回 replace 后的新列表。逐字移植。 |
| `core/bands.py` | BandPlan 查询属性 (num_bands/num_elements/band_names/band_element_indices/band_of/element_count) | `454-523` | **engineering** | `ported` | 归属查询。证据 JSON 需要 band_of/band_names，故保留。 |
| `core/bands.py` | BandSpec / BandPlan 的 as_dict | `282-294, 665-677` | **engineering** | `ported_adapted` | 服务本 WP 证据落盘。BandPlan.as_dict 去掉 guards 键（其依赖未移植的 guard_report），其余逐字。 |
| `core/bands.py` | BandSpec 波长/频率 docstring | `185-193, 222-235, 242-250` | **engineering** | `ported_adapted` | ★ fu4 的 docstring 与代码相反（自称 low_nm 是长波长，但不变式强制 low_nm < high_nm 即短波长）。只改注释不改行为——该错误表述在写场景时会导致 波段边界整体算反。 |
| `core/bands.py` | hz_to_nm | `127-143` | **engineering** | `ported` | 只被未移植的 guard_report / sclu_span_report 使用；搬进来即死代码。★ WP-MERGE-FG-FLEX 相位守卫翻转：原「决定不搬」的前提已被本次合并合法作废——guard_report / sclu_span_report 一并进来了，hz_to_nm 不再是死代码。按通例翻成正向不变量：存在 + 只定义一次（`def hz_to_nm(` 在 bands.py 内仅 1 处定义）+ 可达（guard_report / sclu_span_report 的实际调用方）。 |
| `core/bands.py` | ITU-T G.694.1 波长常数 + U_BAND_ASSUMPTIONS | `146-178` | **engineering** | `ported` | FU-2 的 S/C/L/U 波段表资产；含 4 条**假设**（U 波段 1660-1770 nm 无ITU 出处、U 段 NF 6.5 dB 无出处）。与 flexgrid 接线无关。★ WP-MERGE-FG-FLEX 相位守卫翻转：合并把 FG 侧波段表资产合法地带了进来，原「决定不搬」的前提消失。按通例翻成正向不变量：存在 + 只定义一次（`U_BAND_ASSUMPTIONS: tuple[str, ...] = (` 在 bands.py 内仅 1 处定义）+ 可达（模块级导出，tests 与 SCLU_BAND_SPECS 均引用）。那 4 条无 ITU 出处的**假设**随资产一并保留，其出处缺口由 SCLU_BAND_SPECS 的U 波段定义承接，本条不改判据口径，只改 action。 |
| `core/bands.py` | _u_spec / CL_BAND_SPECS / CLS_BAND_SPECS / CLU_BAND_SPECS / SCLU_BAND_SPECS | `297-328` | **engineering** | `ported` | 同上，FU-2 预置波段表。本仓场景自带频段规格，故不需要预置表。★ WP-MERGE-FG-FLEX 相位守卫翻转：预置表已由合并带进来。按通例翻成正向不变量：存在 + 只定义一次（`SCLU_BAND_SPECS: tuple[BandSpec, ...]` 在 bands.py 内仅 1 处定义）+ 可达（模块级导出，sclu_span_report 与 tests 引用）。 |
| `core/bands.py` | BandPlan.from_channels | `413-425` | **engineering** | `ported` | 遗留 SpectrumGrid 粒度入口，非 flexgrid 接线所需。★ WP-MERGE-FG-FLEX 相位守卫翻转：合并已把该入口带进来。按通例翻成正向不变量：存在 + 只定义一次（`def from_channels(` 在 bands.py 内仅 1 处定义；与 `from_connections` 是两个不同入口，锚点互不混淆）+ 可达（classmethod，由调用方按 SpectrumGrid 粒度调用）。 |
| `core/bands.py` | BandPlan.noise_figure_db_per_element / launch_power_profile_w / per_band_launch_power_w | `526-571` | **engineering** | `ported` | 逐**信道**口径入口。J8 链路走逐连接口径（apply_launch_power）。noise_figure_db_per_element 还会把 BandSpec.amplifier 拉进评估口径。★ WP-MERGE-FG-FLEX 相位守卫翻转：逐信道口径的三个入口已由合并带进来，与既有的逐连接口径（apply_launch_power）并存，不是二选一。按通例翻成正向不变量：存在 + 只定义一次（`def noise_figure_db_per_element(` 在 bands.py 内仅 1 处定义）+ 可达（BandPlan 公有方法）。 |
| `core/bands.py` | BandPlan.guard_report / sclu_span_report | `635-663, 680-728` | **engineering** | `ported` | FU-2 的保护间隔与 S+C+L+U 跨距口径报告，属该分支自有结论。★ WP-MERGE-FG-FLEX 相位守卫翻转：两份报告已由合并带进来。按通例翻成正向不变量：存在 + 只定义一次（`def guard_report(` 在 bands.py 内仅 1 处定义；`sclu_span_report` 为模块级函数）+ 可达（两者都被 tests/ 与 hz_to_nm 的调用方实际调用）。 |
| `models/isrs_gn.py` | MultiBandISRSGN.__init__(slots_per_channel=...) | `136, 183-185` | **engineering** | `applied` | J5 允许的索引空间适配。仅用于把 blocked_channels 展开成 blocked_slots；遗留模式下不参与任何数值。 |
| `models/isrs_gn.py` | MultiBandISRSGN.symbol_rate_hz_channels | `216` | **engineering** | `applied` | ★ 索引空间适配：np.full(n_nodes, 标量 Rs)。遗留下与标量逐位等价（WP-O1 已证 isrs_gn.py:215-216）。**不是**逐连接异构 Rs——那需改冻结内核。 |
| `models/isrs_gn.py` | MultiBandISRSGN.roadm_blocks_node | `485-509 (rwa 侧同语义)` | **engineering** | `applied_new` | ★ fu4 把同一套 blocked_slots 解析逻辑在 rwa / physics / isrs_gn 三处各写一遍（其中 physics 侧靠私有属性 _slots_per_channel_hint 传参）。本 WP 收敛成一个方法，三处共用，保证语义不会漂移。legacy 分支逐字等价。 |
| `models/isrs_gn.py` | MultiBandISRSGN.for_fiber 透传 connections/ode_accel/slots_per_channel | `n/a` | **engineering** | `applied_new` | ★ **bug 修复**：O2 移植的 for_fiber 只传 (grid, fiber, Rs)，逐连接模式下子评估器会退回遗留分支并读 FlexibleGrid.wavelengths（不存在）→ AttributeError。J8 链路必经 for_fiber，不修则端到端跑不起来。遗留模式三者皆为默认值，行为不变。⚠️ WP-MERGE-FG-FLEX：合并在透传序列里插入了 `ode_accel` / `raman_spectrum`，锚点随之改钉在 `slots_per_channel` 与 `ode_accel` 相邻的那两行上（透传事实未变，仍是同一处 `for_fiber` 调用）。 |
| `models/isrs_gn.py` | 逐连接异构 symbol_rate_hz 数组参数 | `1306, 1328-1330 (numba_kernels.py)` | **physics** | `ported_adapted` | ★ 物理片段。fu4 的 _calc_span_noise_and_power_jit 接受逐连接 Rs 数组并据此算 ASE；本仓原先只吃标量，故 O2/O3 都显式拒绝（那时的裁定是『改冻结核 = 违反 J5』）。WP-Q1 按 J2 授权改动冻结核：P1 (23)/(29)/(30)/(34) 的 B_i / B_l 本来就是逐信道量。落地口径不是照抄 fu4 的 Rs 数组，而是引入逐信道**信道带宽** b_hz_channels，由 channel_bandwidth_source（symbol_rate / occupied）决定取值——fu4 没有这个区分。未移植 fu4 的任何 ODE / phi_kurtosis / ASE 增益因子。⚠️ WP-MERGE-FG-FLEX：合并后的内核把 flexgrid 侧的 `b_hz_channels` 改名并入 `symbol_rate_hz`（内核侧收**一个**逐节点数组承载 P1 的 B_i），故锚点改钉在调用点 `symbol_rate_hz=self.channel_bandwidth_hz_channels` 上。**判据口径不动**：paper_formulas 仍是 P1 (23)/(29)/(30)/(34)，B_i 仍是逐信道量、不许退回标量；改的只是这个参数在合并后叫什么。 |
| `network/elements.py` | ROADMConfig.blocked_slots | `84-87` | **engineering** | `applied` | 灵活栅格下的权威阻断字段（12.5 GHz 粒度）。默认空列表，遗留行为不变。 |
| `network/elements.py` | validate_blocked_channels 扩展到槽 | `96-99` | **engineering** | `applied` | 槽索引的非负/唯一校验，与既有信道校验同构。 |
| `network/elements.py` | ROADMConfig.blocked_slot_set() | `102-111` | **engineering** | `ported` | 实测本仓 src/ tests/ tools/ 中**零调用方**（死代码）。语义已由 isrs_gn.roadm_blocks_node 与 slots.resolve_blocked_slots 覆盖。★ WP-MERGE-FG-FLEX 相位守卫翻转：方法已由合并带进来，且不再是死代码——tests/test_fg2_flexible_grid.py:737/743 实际调用它并断言返回值。按通例翻成正向不变量：存在 + 只定义一次（`def blocked_slot_set(` 在 elements.py 内仅 1 处定义）+ 可达（tests 直接调用）。语义仍与既有实现同源，翻转的是「本仓不存在该能力」这条相位守卫，不是判据口径。 |
| `rwa/qot_checker.py` | QoTValidator.num_nodes | `53-59` | **engineering** | `applied` | J5 明确允许的 num_channels -> n_nodes 索引空间适配。遗留下 n_nodes == spectrum_grid.num_channels。 |
| `rwa/qot_checker.py` | QoTValidator._is_blocked | `61-78` | **engineering** | `applied_adapted` | 委托给 evaluator.roadm_blocks_node（单一实现）。legacy 分支返回 node_index in roadm.blocked_channels，与移植前逐字等价。 |
| `rwa/qot_checker.py` | QoTValidator._node_frequency_and_bandwidth | `106-122` | **engineering** | `applied` | ★ 索引空间适配：booster ASE 的频率/带宽真源必须是逐节点数组，不能用 spectrum_grid.frequencies。WP-O1 §0 已证 legacy 下二者逐位等价。WP-Q1 起第二返回值是 P1 (34) 的 B_i（bandwidth），故改名 _node_frequency_and_bandwidth；旧名保留为别名。 |
| `rwa/qot_checker.py` | spectrum_grid.num_channels -> self.num_nodes（6 处数组/循环定长） | `241-243, 397, 417-418` | **engineering** | `applied` | 索引空间适配，遗留下取值完全相同。 |
| `rwa/qot_checker.py` | booster ASE 标量分支 -> _node_frequency_and_bandwidth | `334-350` | **engineering** | `applied` | 只换取值来源，公式 NF*h*f*B*(loss-1) 一字未动。 |
| `rwa/qot_checker.py` | booster ASE 向量分支 -> f_abs_hz + channel_bandwidth_hz_channels | `426-441` | **engineering** | `applied` | 同上。WP-Q1 起 B_i 取 channel_bandwidth_hz_channels[i]（逐信道），legacy 下该数组恒等于标量 Rs，故 legacy 逐位等价（IEEE 双精度逐元素乘法与标量乘法同结果）。 |
| `rwa/qot_checker.py` | QoTValidator._roadm_terms | `80-105` | **engineering** | `ported` | 纯重构：把既有 inline 逻辑抽成函数（PLAN §0 已断语义为零）。对 flexgrid 零价值。**故意不搬**，使 legacy 的装配算式保持逐字不变——J6 由构造即可证明，不必只靠测试反证。★ WP-MERGE-FG-FLEX 相位守卫翻转：抽取已由合并带进来。按通例翻成正向不变量：存在 + 只定义一次（`def _roadm_terms(` 在 qot_checker.py 内仅 1 处定义；折叠逻辑的单一真源是模块级 `fold_roadm_terms`，本方法只是薄封装）+ 可达（qot_checker.py:671 的逐节点 booster 路径实际调用）。「legacy 装配算式逐字不变」这一原有判据口径不受本次翻转影响。 |
| `rwa/qot_checker.py` | _H_PLANCK 常量提取 | `17-19` | **engineering** | `ported` | 纯命名提取，数值同（6.62607015e-34）。零 flexgrid 价值。★ WP-MERGE-FG-FLEX 相位守卫翻转：常量提取已由合并带进来。按通例翻成正向不变量：存在 + 只定义一次（`_H_PLANCK = 6.62607015e-34` 在 qot_checker.py 内仅 1 处赋值定义；另 3 处是引用而非重复定义）+ 可达（tests/test_fg20_roadm_injection_behavior.py:50 直接 import 并断言其值）。 |
| `physics/qot_checker.py` | PyNLOQoTEvaluator(connections=..., slots_per_channel=...) | `78, 92, 104-131` | **engineering** | `applied` | 逐连接模式：每连接一个波形，频率取连接中心。 |
| `physics/qot_checker.py` | _default_passband_hz（最宽连接的占用带宽） | `118-131, 247` | **engineering** | `applied` | WSS 兜底通带。遗留分支恒等于 spectrum_grid.channel_spacing_hz，与移植前表达式逐字相同；灵活栅格下必须取最宽连接，否则宽连接被滤波掉。 |
| `physics/qot_checker.py` | PyNLOQoTEvaluator._is_blocked | `135-147` | **engineering** | `applied_adapted` | 槽区间重叠语义；**改掉** fu4 靠私有属性 roadm._slots_per_channel_hint 传参的做法，改为显式构造参数。legacy 分支逐字等价。 |
| `physics/qot_checker.py` | occupied_width 按连接集合带边缘计算 | `118-124` | **engineering** | `applied` | 遗留分支仍为 (num_channels-1)*spacing，逐字不变；仅决定 _mux_target_sr。 |

### 1.2 刻意**不**搬（按文件）

| 文件 | 片段 | fu4 行 | kind | 理由 |
|---|---|---|---|---|
| `models/isrs_gn.py` | 用连接占用带覆写 f_m_hz / f_M_hz 并真正传给内核 | `n/a（fu4 用真 ODE，无此参数）` | **physics** | ★ 物理片段。O2 已在 Python 侧按连接占用带算出 f_m_hz/f_M_hz，但冻结内核在核内用标量 Rs 重算二者，故该修正在本仓是**死代码**。让它生效需给内核加参数 = J5。见 e2e 的 known_limitations.srs_band_edges_still_scalar_rs。⚠️ WP-MERGE-FG-FLEX 复核结论：**未翻转**，维持 not_ported（不是留白，是有出处的复核）。锚点 `conn.centre_hz + conn.n_slots * conn.grid.slot_width_hz / 2` 确实在 isrs_gn.py 存在且只出现 1 次，但 (a) `git diff b69563a -- models/isrs_gn.py` 证明该段是 **O2 既有代码**，本次合并只在它上方加了两行注释，没有把新能力带进来——相位守卫的前提并未被推翻；(b) `f_m_hz` / `f_M_hz` 在整个 src/ 中仍**零读取方**（只有同一处 if/else 的 4 行赋值），依旧没有真正传给内核，「死代码」的结论今天仍成立。故按通例翻成 `ported` 会是假声明，不翻。翻转需要先给内核加逐连接带边参数，那需要单独的 J5 授权。 |
| `models/numba_kernels.py` | 整个文件（fu4 用 RK4 _calc_exact_power_profile_ode 真 ODE 传播；本仓用 ECZ 闭式 _calc_span_out_power_jit） | `1-1480` | **physics** | ★ J5 冻结文件（基线已推进到 WP-Q1 的 b69563a，见 WP_BASE）。两棵树在此**不是同一个物理模型**：fu4 是 RK4 数值传播，本仓是 P1 的闭式 ECZ GN 模型（ODE 线按 O2 裁定以 P1/P2 为准）。fu4 的 RK4 / ode_accel / phi_kurtosis 一律不移植。唯一被授权的改动是 WP-Q1 对 B_i / B_l 的逐信道化（P1 (23)/(29)/(30)/(34)）。自 b69563a 起 diff 必须为空。⚠️ WP-MERGE-FG-FLEX 复核结论：**J5 已翻、本条 action 未翻**。合并确实把 fu4 的 RK4 ODE（`_calc_exact_power_profile_ode`，只定义 1 处，由同文件 :1431 实际调用）与 ode_accel 加速带进来了，因此 J5 已按通例从「相对 b69563a 的 diff 为空」翻成 sha256 绝对正向不变量（见 FROZEN_SHA256，取 FG 侧字节）。但本条 `kind == "physics"`：J7 的硬闸门要求任何被应用的物理片段自带非空 `paper_formulas`（J2「论文提及且与本仓不一致 ⇒ 改成论文」的授权）；判据口径要求 `paper_formulas` 一律不动，而合并带入的 FG 内核是 RK4 数值传播、**并非**按 P1 (23)/(29)/(30)/(34) 逐式改写（results/ode_dev_formula_map.json 里指向本文件的 P1 代码锚点现已全部匹配不上），无法诚实地给出论文公式号。故 action 暂留 not_ported，**待裁决**，不作假声明。 |

### 1.3 `core/bands.py` 的最小闭包（728 行 → 最小片段）

`wp/fg-uwb-fid` 的 `core/bands.py` 在 `ecz-dev` 上**整文件不存在**（非改名、非合并——它诞生于 FG 线的 `51b601b` "FU-2: S+C+L+U 波段定义"）。因此本 WP 新建该文件时**只摘 flexgrid 接线所需的最小闭包**：`nm_to_hz` + `BandSpec` + `BandPlan` 的三个接入点与其归属查询；FU-2 的波段表、假设与报告逻辑一律不搬。

最小闭包的三条依据：

1. `BandPlan.from_connections` 的第二参数是 `Sequence[BandSpec]`；
2. `BandSpec.low_hz/high_hz` 由 `nm_to_hz` 换算；
3. `ConnectionSpec` 是 frozen dataclass，故逐波段功率必须经 `apply_launch_power` 返回新对象。

⚠️ 本仓**不预置**任何波段表——波段规格由场景自带 （见 `tools/ode3_flexgrid_scenarios.py` 的 `BAND_SPECS`）。

## 2. flexgrid 端到端（J8）

场景 **`flex_two_hop_mixed_width`**，由 `tools/ode3_flexgrid_scenarios.py (WP-O3 自写，未 import fu4 tests)` 自写（brief §2 步骤 2 要求不 import fu4 的 tests）。

- 栅格：槽宽 12.5 GHz × 240 槽 = 3.000 THz，覆盖 191.000 – 194.000 THz
- 频段：`L`, `C`，逐段元素数 {'L': 2, 'C': 2}
- 统一符号率：50 GHz（uniform=True，原因见 §4 限制 1）
- 业务：路径 `[0, 1, 2]`，承载连接 #2（下标 2），`accepted = True`

### 2.1 逐连接 SNR

| conn | n_slots | 占用带宽 | 中心频率 (THz) | 调制 | 比特率 | 频段 | 发射功率 | SNR (dB) |
|---:|---:|---:|---:|---|---:|---|---:|---:|
| 0 | 2 | 25.0 GHz | 191.01250 | QPSK | 50 Gbps | L | 1.00 mW | **33.2192** |
| 1 | 6 | 75.0 GHz | 191.16250 | QPSK | 150 Gbps | L | 1.00 mW | **33.1958** |
| 2 ★ | 3 | 37.5 GHz | 192.01875 | 8QAM | 100 Gbps | C | 0.50 mW | **34.9186** |
| 3 | 8 | 100.0 GHz | 192.55000 | 16QAM | 400 Gbps | C | 0.50 mW | **34.9517** |

★ = 本次验证的新业务承载连接。`n_slots` 取值 `[2, 3, 6, 8]` 共 **4 个不同值**（J8 要求 ≥2）。

⚠️ **声明宽度 ≠ 声明符号率**：连接按 `slots_for_rate` 占 2/6/3/8 槽（25–100 GHz），但符号率统一为 50 GHz。原因是冻结内核只吃标量 Rs（§4 限制 1）。**灵活栅格要表达的「宽度不等」成立；「每连接 Rs 自适应」不成立。**

### 2.2 探针

1. **异构 Rs**（❌ 未被拒）：`[50000000000.0, 75000000000.0, 50000000000.0, 50000000000.0]` → `None…`
2. **blocked_slots 槽区间重叠**（✅ 通过）：槽 11 阻断 [1]（期望 [1]，落在连接 1 的槽区间 [10,16) 内）；槽 5 阻断 []（期望 []，落在空隙）
3. **跨频段边界宽连接**（✅ 被拒）：200 GHz 的连接中心在 191.5000 THz，跨 L|C 边界 → `ValueError: Connection 0 occupies [191.400000, 191.600000] THz, which straddles band boundary betwee…`

## 3. 能力矩阵（`fu4` vs `ode-dev`）

共 19 条：双方都有 8、仅 `fu4` 有 8（其中**涉物理** 4）、仅 `ode-dev` 有 2、`ode-dev` 更优 1。

| 能力 | 文件 | 状态 | 涉物理 | 证据 |
|---|---|---|:---:|---|
| FlexibleGrid（槽位化栅格） | `core/flexgrid.py` | `both` | 否 | 逐字移植（O2）。centre_hz 用 n_slots/2 区间中心，非点阵中心。 |
| ConnectionSpec（冻结连接规格） | `core/flexgrid.py` | `both` | 否 | 逐字移植（O2）。n_slots 与 slots_for_rate 一致性由 __post_init__ 强制。 |
| slots.py 槽工具（slots_for_rate / connection_overlaps_slots / resolve_blocked_slots / migrate_roadm_blocked_slots） | `network/slots.py` | `both` | 否 | 逐字移植（O2）。 |
| MultiBandISRSGN(connections=...) 逐连接模式 | `models/isrs_gn.py` | `both` | 否 | O2 已接入；O3 补齐 for_fiber 透传（否则逐连接模式下 AttributeError）。 |
| ROADMConfig.blocked_slots（12.5 GHz 粒度权威阻断） | `network/elements.py` | `both` | 否 | O3 应用。默认空，legacy 不变。 |
| 槽区间重叠阻断语义（connection_overlaps_slots 单一实现） | `models/isrs_gn.py` | `ode_dev_better` | 否 | fu4 在 rwa/physics/isrs_gn 三处各写一遍（physics 侧还靠私有属性 _slots_per_channel_hint）；O3 收敛为 evaluator.roadm_blocks_node 一处。 |
| QoTValidator 逐连接 SNR 装配 | `rwa/qot_checker.py` | `both` | 否 | O3 应用 num_nodes / _is_blocked / _node_frequency_and_rate；booster ASE 公式一字未动。 |
| PyNLO 波形模型逐连接频率 + 最宽连接 WSS 通带 | `physics/qot_checker.py` | `both` | 否 | O3 应用。legacy 分支的 passband/occupied_width 表达式逐字不变。 |
| BandPlan.from_connections / assert_connections_single_band / apply_launch_power | `core/bands.py` | `both` | 否 | O3 最小闭包移植（fu4 728 行 -> 本仓只取 flexgrid 接线片段）。 |
| S/C/L/U 预置波段表（CL/CLS/CLU/SCLU_BAND_SPECS） | `core/bands.py` | `fu4_only` | 否 | FU-2 自有资产，含 4 条无 ITU 出处的 U 波段假设。与 flexgrid 接线无关，本 WP 不搬。场景可自带频段规格。 |
| sclu_span_report / guard_report（跨距与保护间隔口径） | `core/bands.py` | `fu4_only` | 否 | FU-2 自有报告逻辑，依赖上条预置波段表。 |
| ROADMConfig.blocked_slot_set() | `network/elements.py` | `fu4_only` | 否 | 实测本仓零调用方（死代码）。语义由 roadm_blocks_node 覆盖。 |
| QoTValidator._roadm_terms（把 inline 装配抽成函数） | `rwa/qot_checker.py` | `fu4_only` | 否 | 纯重构（PLAN §0 已证语义为零）。故意不搬，使 legacy 装配算式保持逐字不变。 |
| 逐连接异构符号率（symbol_rate_hz 数组参数） | `models/numba_kernels.py` | `fu4_only` | **是** | ★ 物理缺口。fu4 核接受 (N,) Rs 并据此算 ASE；本仓冻结核只吃标量。改它 = 改 numba_kernels.py = 违反 J5。O2 已显式拒绝，O3 沿用。证据：e2e.probes.heterogeneous_rs。 |
| SRS 频带边界按真实连接占用带（f_m_hz/f_M_hz） | `models/numba_kernels.py` | `fu4_only` | **是** | ★ 物理缺口。本仓 Python 侧已按占用带算出 f_m_hz/f_M_hz，但冻结核在核内用标量 Rs 重算，故该修正是死代码。让它生效需给内核加参数 = J5。证据：e2e.known_limitations.srs_band_edges_still_scalar_rs。 |
| ode_accel: matvec / cached_g 加速档 | `models/numba_kernels.py` | `fu4_only` | **是** | ★ 物理缺口（求和次序改变 ⇒ 非逐位相等）。需改冻结内核，WP-O3 §8 明确列为不做。fu4 侧 NLI 张量公式出处是其 docstring 的 'Eq. 21'，P1/P2 均无此式，O2 已按 J1/J3 拒绝。 |
| phi_kurtosis（64QAM 峰度进入 XPM） | `models/numba_kernels.py` | `fu4_only` | **是** | 本仓闭式 ECZ 模型无此参数；引入它等于改变 NLI 模型，按 J1 需 P1/P2 公式号支撑，O2 未取得，故不移植。 |
| numba 缺失时的 njit 降级垫片 | `models/numba_kernels.py` | `ode_dev_only` | 否 | 本仓独有：ImportError 时 njit 退化为恒等装饰器，让无 JIT 的最小环境也能跑统计模型。O2 已带（ecz-dev 既有）。 |
| 闭式 ECZ GN 模型（按 P1 公式号逐条可 grep） | `models/numba_kernels.py` | `ode_dev_only` | **是** | 本仓物理底：P1 §II.B 扩展 CZ / §III 闭式 GN。fu4 走的是 RK4 真 ODE 传播。两棵树内核**不是同一个物理模型**，故 numba_kernels.py 的 diff 恒非空——J5 要求的是本 WP 相对自身起点的 diff 为空。 |

## 4. legacy 回归（J6）

协议与 WP-O2 **同口径**（复用 `复用 tools/ode_compare.py（scripts_shared/compat_run.py + cross_snr_compat.py 签名兼容层），与 WP-O2 同口径`），容差 **0.0**。

| 场景 | `ode-dev` SNR (dB) | `ecz-dev` SNR (dB) | `delta_db` | 逐位相同 | 裁决 |
|---|---:|---:|---:|:---:|:---:|
| `legacy_two_hop_equalize_on` | 25.265176305 | 25.265176305 | +0.000000000 | ✅ | pass |
| `legacy_three_hop_equalize_on` | 23.391590838 | 23.391590838 | +0.000000000 | ✅ | pass |
| `legacy_two_hop_equalize_off` | 22.239480442 | 22.239480442 | +0.000000000 | ✅ | pass |
| `legacy_express_attenuation` | 25.227016935 | 25.227016935 | +0.000000000 | ✅ | pass |
| `legacy_high_filtering` | 24.255573730 | 24.255573730 | +0.000000000 | ✅ | pass |
| `legacy_blocked_channel` | n/a | n/a | n/a | ✅ | pass |

`legacy_blocked_channel` 两端**一致拒绝**（`both_rejected = 1`），故 `delta_db = n/a` 但裁决为 `pass`。

## 5. 物理缺口（J5 上报，**未修**）

以下三项都需要改 `models/numba_kernels.py`（J5 禁改，diff 必须为空）。按 WP-O3 §5「别重复踩 O2 的坑」，本 WP **一律报告而不强改**。

### srs_band_edges_still_scalar_rs

- **涉及物理**：是
- **事实**：isrs_gn.py 的 _precompute_channel_dependent_properties 已按连接占用带算出 f_m_hz/f_M_hz，但冻结内核**不使用**它们——它在核内用标量 Rs 重算 (f_rel_hz[0] - Rs/2, f_rel_hz[-1] + Rs/2)。
- **影响**：宽连接下 SRS 频带边界仍按标量 Rs 而非真实占用带，低估真实拉曼作用范围。
- **裁定**：★ WP-Q1 **未**处理此项：它不属于 P1 (23)/(29)/(30)/(34) 里 B_i/B_l 的落点，且改它要动 SRS 频带边界这一独立物理量。按「不给论文没写的项加物理」与「拿不准就停下报告」，留给后续 WP 裁决。

此外 `ode_accel` 的 `matvec` / `cached_g` 档、以及 `phi_kurtosis` 同样需改冻结内核，已按 WP-O3 §8 列为不做（能力矩阵中 `involves_physics = 是`）。

## 6. 附带修正与副作用

### 6.1 `for_fiber` 丢 `connections`（bug）

O2 移植的 `MultiBandISRSGN.for_fiber` 只传 `(grid, fiber_config, Rs)`，逐连接模式下子评估器会退回遗留分支并读 `FlexibleGrid.wavelengths`（该属性不存在）→ `AttributeError`。J8 链路必经 `for_fiber`（`rwa/qot_checker.py:161`），不修则端到端跑不起来。已补上 `connections` / `ode_accel` / `slots_per_channel` 透传；遗留模式下三者皆为默认值，行为不变。

### 6.2 O2 公式锚点行号漂移

在 `models/isrs_gn.py` 插入接线后，O2 `formula_map` 里 **16** 条代码锚点的**行号**下移，导致 J1 守卫失败。这些锚点指向的**物理代码一字未改**（`git diff` 只删了 3 行，全是工程接线）。`tools/ode3_relocate_anchors.py` 只重排行号、**不动 `contains` 字符串**——若物理真被改写，它会因匹配不上而报错退出。未自动修正 47 条。

### 6.3 `BandSpec` 波长/频率 docstring 反了

fu4 的 docstring 自称 `low_nm` 是**长**波长，但不变式强制 `low_nm < high_nm`即**短**波长。照它写场景会把波段边界整体算反。只改注释、不改行为。

## 7. 没做什么 / 没跑完什么

| 项 | 状态 | 原因 |
|---|---|---|
| 逐连接**异构**符号率 | **未跑** | 冻结内核只吃标量 Rs，改它违反 J5。已用 `probes.heterogeneous_rs` 证明确实被拒 |
| SRS 频带边界按真实占用带 | **未跑** | 同上：需给冻结核加参数 |
| `ode_accel` 的 `matvec` / `cached_g` 档 | **未做** | WP-O3 §8 明确列为不做 |
| `phi_kurtosis` 进入 XPM | **未做** | 本仓闭式 ECZ 无此参数，按 J1 需 P1/P2 公式号 |
| FU-2 的 S/C/L/U 预置波段表、`sclu_span_report`、`guard_report` | **未搬** | FU-2 自有资产，含无 ITU 出处的 U 波段假设；与 flexgrid 接线无关 |
| `physics/qot_checker.py` 的**波形级**端到端跑通 | **未跑** | 需 PyNLO + CuPy compute context（本机无 GPU 上下文）。本 WP 只做接线与 legacy 表达式不变性验证，未实际跑波形 |
| 跨仓（RL 仓）对照 | **未做** | WP-O3 §8 明确「不接 RL 仓」 |
| `flex_*` 与 `fu4` 的 SNR 逐场景对照 | **未做** | fu4 与本仓内核**不是同一个物理模型**（fu4 = RK4 真 ODE；本仓 = P1 闭式 ECZ），对照无意义，且本 WP 无权改本仓物理 |

## 8. 复算

```bash
export PYTHONPATH=$PWD/src
PY=~/miniconda3/envs/rl/bin/python
$PY tools/ode3_relocate_anchors.py --check   # O2 锚点行号仍对齐
$PY tools/ode3_sync.py            --check     # J7 + J5
$PY tools/ode3_flexgrid_e2e.py    --check     # J8 逐连接 SNR
$PY tools/ode3_legacy_regression.py --check   # J6 逐位
$PY tools/ode3_capability.py      --check     # 能力矩阵
$PY tools/ode_compare.py          --check     # WP-O2 三树对照（未退步）
$PY tools/ode_report.py           --check     # 本文档逐字节一致
```

