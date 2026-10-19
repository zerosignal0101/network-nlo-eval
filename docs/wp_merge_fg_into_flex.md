# WP-MERGE-FG-FLEX 合并报告：把 `wp/fg-integrate` 的加速实现并入 `flexgrid-dev`

- 分支：`wp/merge-fg-into-flex`
- 起点：`flexgrid-dev @ cb24704`
- 并入：`wp/fg-integrate @ 57c7221`（只读）
- 合并提交：`6b8304f`（其后为测试/证据/文档的增量提交）
- 解释器：`~/miniconda3/envs/rl/bin/python`（torch 2.14.0+cu126 / numpy 2.5.3 / numba 0.68.0）
- 跑测试一律带 `PYTHONPATH=$PWD/src`

> ⚠️ 本仓的 site-packages 里装着一份 `network_nlo_eval`（commit cb24704 = flexgrid-dev）。
> **不设 `PYTHONPATH` 时测试会测到别人家的树。** FG 侧另需 `PYTHONPATH=$WT/src`。

---

## 0. 结论摘要

88 个冲突块全部逐块裁决完毕，无一遗留。合并后合并树同时具备两边能力：

| 能力 | 来自 | 状态 |
|---|---|---|
| ODE 加速三档（`off`/`cached_g`/`matvec`，默认 `matvec`） | FG | 保留，`off` 档与 FG 金标准**逐位相同** |
| 跨段记忆化 `cached` 档 | flexgrid | 保留（与 `off` 逐位相等） |
| NLI 张量化 / 拉曼 G 缓存 / 批量入口 | FG | 保留，`numba_kernels.py` 逐字节取 FG |
| 逐信道 `B_i` / `P_i` 参数化 | flexgrid | 保留，默认档下与 FG 逐位相同 |
| 多槽灵活栅格（槽区间分配/释放） | flexgrid | 保留 |
| S+C+L+U 波段表与 FU-2 报告 | FU-2 | **本次移植**，并翻转其相位守卫 |
| 两套灵活栅格表示的显式转换函数 | 新增 | `flexgrid.connection_spec_from_allocation` / `slot_window_from_connection` |

两条判据**客观互斥**、本次只能取其一，已在 §5 逐条给出可复算证据（J5 内核哈希、Q-J5 锚点字面量）。

---

## 1. 合并前基线（跑前记录）

| 树 | 读数 | 耗时 |
|---|---|---|
| `flexgrid-dev @ cb24704` | **2 failed, 66 passed, 1 skipped** | 3.6 s |
| `wp/fg-integrate @ 57c7221` | **477 passed, 2 skipped** | 36.2 s |

`flexgrid-dev` 的 2 个失败是 `test_q_evidence_tools_check_reproduces[bwqot_legacy_regression.py]`
与 `[ode_compare.py]`，根因是 `ModuleNotFoundError: fu4_scenarios_legacy` —— **环境预存问题，
与合并无关**，两棵树都有 `physics_nlo_eval` 缺失导致的 skip。

权威冲突面（`git merge-tree --write-tree`，不依赖工作树状态）：

| 文件 | 冲突块 |
|---|---:|
| `core/bands.py` | 28 |
| `models/isrs_gn.py` | 22 |
| `models/numba_kernels.py` | 15 |
| `network/elements.py` | 2 |
| `network/state.py` | 3 |
| `physics/qot_checker.py` | 8 |
| `rwa/qot_checker.py` | 10 |
| **合计** | **88** |

> 处置表按这个**权威粒度**记录。注：中途用 `git checkout -m` 重新生成冲突时，`isrs_gn.py`
> 的粒度从 22 变成 21（两块被并成一块），因此下表以 `merge-tree` 为准拆回 22 行。

---

## 2. 88 行逐冲突块处置表

| # | 文件 | 块号 | 起始行 | 所属函数 / 类 | 取哪边 | 理由 |
|---:|---|---:|---:|---|---|---|
| 1 | `core/bands.py` | 1 | 1 | 模块 docstring | **取 FG + 追加合并补记** | FG 的 S/C/L/U 论证 docstring 是 FU-2 资产本体；flexgrid 那份写着「FU-2 全部未移植」的相位守卫前提已消失，改为在 FG 版上追加合并补记段 |
| 2 | `core/bands.py` | 2 | 127 | 模块 imports | **取 FG** | SpectrumGrid 只被 FG 侧用到；AST 差分确认 flexgrid 无独有 import |
| 3 | `core/bands.py` | 3 | 135 | __all__ | **取 FG** | FG 的导出面含全部 FU-2 资产 |
| 4 | `core/bands.py` | 4 | 173 | nm_to_hz | **取 FG** | docstring 措辞差异（Returns: vs Returns），公式逐字相同 |
| 5 | `core/bands.py` | 5 | 194 | BandSpec | **取 FG** | BandSpec 两侧方法集**完全相同**（各 7 个），是去重而非取舍 |
| 6 | `core/bands.py` | 6 | 283 | BandSpec 字段文档 | **取 FG + 并入 flexgrid** | FG 的 low_nm/high_nm 说明把长短波长写反了（"low_nm: 长波段边缘，频率最高"自相矛盾）；代码两边一致（high_hz=nm_to_hz(low_nm)），故采用 flexgrid 的正确描述 |
| 7 | `core/bands.py` | 7 | 339 | BandSpec.__post_init__ | **取 FG + 并入 flexgrid** | 报错文案取 flexgrid 的清晰版（SHORTER wavelength, hence the HIGHER frequency），它同时是 J1 锚点 |
| 8 | `core/bands.py` | 8 | 354 | BandSpec.low_hz | **取 FG + 并入 flexgrid** | 补回「由长波长端（high_nm）换算」的具体端点说明 |
| 9 | `core/bands.py` | 9 | 363 | BandSpec.high_hz | **取 FG + 并入 flexgrid** | 补回「由短波长端（low_nm）换算」 |
| 10 | `core/bands.py` | 10 | 381 | BandSpec.contains_hz | **取 FG** | 纯 docstring 措辞 |
| 11 | `core/bands.py` | 11 | 397 | BandSpec.contains_hz_array | **取 FG** | 纯 docstring 措辞 |
| 12 | `core/bands.py` | 12 | 414 | BandSpec.as_dict | **取 FG** | _THz→_THZ 常量改名；值同为 1e12 且该常量只在 bands.py 内部使用（实测无外部引用），统一为 FG 命名 |
| 13 | `core/bands.py` | 13 | 428 | BandPlan | **取 FG** | BandPlan 侧 FG 多 5 个方法、flexgrid 多 0 个（AST 差分）⇒ 严格超集 |
| 14 | `core/bands.py` | 14 | 494 | BandPlan.__init__ | **取 FG** | docstring 措辞（「本类」→「本模块」） |
| 15 | `core/bands.py` | 15 | 536 | BandPlan.__init__ | **取 FG** | 折行格式差异，逻辑相同 |
| 16 | `core/bands.py` | 16 | 564 | BandPlan.from_connections | **取 FG** | flexgrid 的三个接入点之一，FG 侧同名同签名且是超集 |
| 17 | `core/bands.py` | 17 | 593 | BandPlan.from_connections | **取 FG + 并入 flexgrid** | 保留 flexgrid 的「★ flexgrid 接线」层级标注（数据模型边界说明要用） |
| 18 | `core/bands.py` | 18 | 624 | BandPlan.num_elements | **取 FG + 并入 flexgrid** | flexgrid 的「（逐连接模式下 = 连接数）」更准确 |
| 19 | `core/bands.py` | 19 | 642 | BandPlan.band_element_indices | **取 FG** | 纯 docstring 措辞 |
| 20 | `core/bands.py` | 20 | 670 | BandPlan.band_of | **取 FG** | 纯 docstring 措辞 |
| 21 | `core/bands.py` | 21 | 698 | BandPlan.element_count | **取 FG** | 纯 docstring 措辞 |
| 22 | `core/bands.py` | 22 | 716 | 逐连接发射功率区块 | **取 FG** | FG 新增的 apply_launch_power + as_dict + guard_report 整块；as_dict 的 guards 键随 guard_report 一起落地（相位守卫翻转） |
| 23 | `core/bands.py` | 23 | 782 | apply_launch_power | **取 FG** | 纯 docstring 措辞 |
| 24 | `core/bands.py` | 24 | 798 | assert_connections_single_band | **取 FG** | _THz→_THZ 常量改名；错误文案保留 FG 的更完整版本 |
| 25 | `core/bands.py` | 25 | 810 | apply_launch_power docstring | **取 FG + 并入 flexgrid** | 保留 flexgrid 的层级边界说明与 ConnectionSpec frozen 语义 |
| 26 | `core/bands.py` | 26 | 826 | apply_launch_power | **取 FG** | 纯 docstring 措辞 |
| 27 | `core/bands.py` | 27 | 857 | BandPlan.as_dict | **取 FG（相位守卫翻转）** | flexgrid 版**刻意删掉** guards 键并注明「本版不含该键（guard_report 未移植）」。该前提已消失 ⇒ 取 FG 版输出 guards；这属 §2 通例的「存在+只定义一次+可达」翻转 |
| 28 | `core/bands.py` | 28 | 907 | sclu_span_report | **取 FG** | FU-2 的跨距口径报告函数，随 merge 落地 |
| 29 | `models/isrs_gn.py` | 1 | 8 | 模块 imports | **并集** | FG 需 `from typing import Any`；其余两边一致 |
| 30 | `models/isrs_gn.py` | 2 | 19 | 模块 imports | **并集** | batch_kernels（FG 批量入口）+ numba_kernels 两侧符号并集，一次写全 |
| 31 | `models/isrs_gn.py` | 3 | 34 | numba_kernels 导入表 | **并集** | 并入 FG 的 g_cache_stats/nli_cache_stats/nli_tensors/raman_g_* 与 flexgrid 的 _db_to_lin |
| 32 | `models/isrs_gn.py` | 4 | 51 | ode_accel 档位常量 | **两侧并集（自定义合并）** | 两套机制互补：flexgrid 的 "cached"=跨段记忆化，FG 的 "cached_g"=G 矩阵缓存、"matvec"=去分支 matvec。ODE_ACCEL_MODES 取并集 ("off","cached","cached_g","matvec")；默认按任务书 §3 保持 FG 的 "matvec"（禁止改动该默认档）。同时并入 flexgrid 的 CHANNEL_BANDWIDTH_*/POWER_WIDTH_* 与 FG 的 RAMAN_SPECTRUM_*、_BATCH_CACHE_STATS、batch_cache_stats() |
| 33 | `models/isrs_gn.py` | 5 | 183 | MultiSpanOpticalPath.__post_init__ | **取 FG** | 仅句号样式（. vs 。） |
| 34 | `models/isrs_gn.py` | 6 | 199 | PathQoTResult | **取 FG** | 仅句号样式 |
| 35 | `models/isrs_gn.py` | 7 | 225 | __init__ 形参 | **并集** | phi_kurtosis(FG) + connections/slots_per_channel/ode_accel/channel_bandwidth_source/power_width_mode(flexgrid)；raman_spectrum 按 FG 守卫要求置于**末位**且默认值为字面 RAMAN_SPECTRUM_DEFAULT |
| 36 | `models/isrs_gn.py` | 8 | 242 | __init__ docstring + 校验 | **自定义合并** | 形参文档两侧合并；校验保留全部四条（ode_accel/slots_per_channel/channel_bandwidth_source/power_width_mode）**并** FG 的 raman_spectrum 校验与「legacy-only 档位显式拒绝 tabular」口径（off 与 cached 都属 legacy-only） |
| 37 | `models/isrs_gn.py` | 9 | 312 | __init__ 属性赋值 | **取 flexgrid** | self.k_bar = min(Σ泵浦功率, 1.0)；FG 侧无此属性。注：合并后内核不再消费 k_bar（解析倾斜路径已删），但属性保留以满足 test_ode_dev 的可达性断言 |
| 38 | `models/isrs_gn.py` | 10 | 320 | __init__ 属性赋值 | **并集** | connections 列表化(FG) + slots_per_channel 校验(FG) + ode_accel/raman_spectrum/phi_kurtosis/channel_bandwidth_source/power_width_mode 赋值；**实参顺序必须与块 7 的签名一致**（顺序错会导致位置实参错配，实测已踩） |
| 39 | `models/isrs_gn.py` | 11 | 340 | n_nodes 与 B_i 属性 | **取 flexgrid** | flexgrid 的 symbol_rate_hz_channels / channel_bandwidth_hz_channels / _power_width_scale / roadm_blocks_node 是 **B_i 不许退回标量**的载体，FG 侧全无 |
| 40 | `models/isrs_gn.py` | 12 | 456 | _precompute 频率/波长 | **取 flexgrid + 弃 FG 的 eager 赋值** | **真实互斥点**：FG 赋值 self.symbol_rate_hz_channels = ...（实例属性），flexgrid 定义同名 @property（无 setter）⇒ 赋值会 AttributeError。取 flexgrid 的 property 形态，值与 FG 的 eager 赋值逐位相同（同一恒等式），legacy 路径 self.grid.wavelengths 经验证恒等于 C_LIGHT/f_abs_hz |
| 41 | `models/isrs_gn.py` | 13 | 513 | 频带边缘计算 | **并集** | 采用 FG 更完整的注释（设计 §2.5），代码结构不变 |
| 42 | `models/isrs_gn.py` | 14 | 535 | FG-4a/4b/4e 加速区块 | **取 FG** | _g_full/_g_gain/_g_loss 缓存、_nli_cache、_batch_constants、_batch_constants_for_span、_nli_kernel_kwargs、_nli_tensors_for_span、g_cache_hits 共 6 个方法，flexgrid 侧全无。⚠ 其中 rs= 传入已由 symbol_rate_hz_channels 改为 channel_bandwidth_hz_channels（逐信道 B_i） |
| 43 | `models/isrs_gn.py` | 15 | 723 | EDFA NF 解析 | **flexgrid 形状校验 + **FG 默认值**** | flexgrid 的 ndim==0 判标量与 (n_nodes,) 形状校验更严，保留；但其 else 分支写的是 np.ones_like(...) = **NF 0 dB**，实测使串行 sigma2_ase_w 比 FG 小 3.981 倍并导致 batch≢serial 差 2.981 ⇒ else 分支必须取 FG 的 self.nf_lin_channels |
| 44 | `models/isrs_gn.py` | 16 | 754 | _calc_span_noise_and_power：缓存外壳 ↔ njit 内核调用 | **自定义合并** | 权威块 16 的 ours 是 flexgrid 的 "cached" 记忆化外壳（cached=None + cache_key），theirs 是 FG 的 njit 调用。两段**互补而非二选一**：缓存外壳保留（并把缓存值改为 4 元组以匹配 FG 内核的 s2_ase 返回），内核实参取 FG 全套（g_full/g_gain/g_loss/**nli_kernel_kwargs/phi_kurtosis/delta_f_co_hz），带宽实参改为 symbol_rate_hz=self.channel_bandwidth_hz_channels（逐信道 B_i；默认档下与 FG 的 symbol_rate_hz_channels 逐位相同）。⚠ 合并后内核形参已无 b_hz_channels/k_bar（numba_kernels 逐字节取 FG） |
| 45 | `models/isrs_gn.py` | 17 | 871 | _calc_span_noise_and_power：缓存命中分支 / EDFA-NF 段 ↔ 批量入口与 ROADM 折叠 | **自定义合并** | 权威块 17：ours 是缓存命中分支 + EDFA/NF 段 + flexgrid 的 inline ROADM，theirs 是批量入口的 NF 覆盖 + FG 的 fold_roadm_terms ROADM。裁决：缓存命中分支保留；**ROADM 装配取 FG**（fold_roadm_terms 单一真源、booster 改注入 ASE 而非衰减信号、filtering_penalty 只乘噪声），丢弃 flexgrid 的 inline p_out/=loss（否则两处装配并存且与 booster 重复计入）；**跨段 EDFA 段整段删除**（见报告「跨段 EDFA」节：与调用方 ASE 重复计入且违反 V1）；批量入口 calc_span_noise_and_power_batch 取 FG；for_fiber 取 FG 并补入 flexgrid 的接线说明与 return |
| 46 | `models/isrs_gn.py` | 18 | 968 | for_fiber docstring | **并集** | 权威块 18（ours=「⚠️ WP-O3 接线修正：逐连接模式下必须把 connections 一并传下去」；theirs=FG 的 FU-2 漏传 raman_spectrum 修正）。两段**都保留**：前者是逐连接接线的前提，后者是 raman_spectrum 漏传的修正——合并后 for_fiber 必须同时透传 connections 与 raman_spectrum |
| 47 | `models/isrs_gn.py` | 19 | 989 | for_fiber 实参 | **并集** | 按块 7 的签名顺序透传全部 7 个配置（phi_kurtosis/connections/slots_per_channel/ode_accel/channel_bandwidth_source/power_width_mode/raman_spectrum） |
| 48 | `models/isrs_gn.py` | 20 | 1008 | _resolve_blocked_node_indices + evaluate_path | **自定义合并** | 保留 FG 的 _resolve_blocked_node_indices（阻断判定单一真源）；evaluate_path 取 FG 头并补入 flexgrid 的 power = power * self._power_width_scale()（P_i/B_i 归一化口径的显式生效处） |
| 49 | `models/isrs_gn.py` | 21 | 1088 | n_hops + 懒导入 | **取 FG** | evaluate_path 需要的 n_hops 与 fold_roadm_terms 的懒导入（避免与 rwa 的模块级循环导入） |
| 50 | `models/isrs_gn.py` | 22 | 1120 | ROADM 逐跳装配 | **取 FG** | fold_roadm_terms(is_last_hop=hop_index==n_hops-1) 为权威口径；blocked 走 _resolve_blocked_node_indices。booster 的 B_i 改用 channel_bandwidth_hz_channels。另：roadm_blocks_node 已改为**委派** _resolve_blocked_node_indices，消除两份槽重叠判定 |
| 51 | `models/numba_kernels.py` | 1 | 6 | 模块头 | **两者合并（ours 胜）** | ours 的 njit 无-JIT 降级 shim；**后续被合并裁决覆盖**：内核最终逐字节取 FG 侧（见报告「J5 互斥」节） |
| 52 | `models/numba_kernels.py` | 2 | 728 | _calc_alpha_tilde_and_kappa_jit | **取 FG** | FG 张量化结构的地基，删则批量核整体塌掉 |
| 53 | `models/numba_kernels.py` | 3 | 814 | _calc_spm_variance_jit | **取 FG** | 取 Eq.21 主体；逐信道 b_i 由形参 rs_hz 保住（块 8 调用点） |
| 54 | `models/numba_kernels.py` | 4 | 847 | _calc_xpm_variance_term_jit 签名 | **取 FG + 并入 flexgrid** | 取 FG 的 14 参签名，把 flexgrid 的逐信道语义（B_ℓ/B_i 各取各的）写进 docstring |
| 55 | `models/numba_kernels.py` | 5 | 879 | _calc_xpm_variance_term_jit 体首 | **取 FG** | a_tilde/kappa 构建 |
| 56 | `models/numba_kernels.py` | 6 | 913 | pre_factor | **取 FG** | (32/27)Γ²p_ℓ²p_i/rs_l_hz 的张量化对应式；rs_l_hz 分母即逐信道 B_ℓ |
| 57 | `models/numba_kernels.py` | 7 | 951 | _compute_nli_variances | **两者合并** | 取 FG 骨架；把 flexgrid 的 b_hz_channels 逐信道契约整体迁入（重命名为 symbol_rate_hz 并注明它就是 B_i） |
| 58 | `models/numba_kernels.py` | 8 | 1036 | SPM 调用点 | **两者合并** | 实际传 symbol_rate_hz[i]（= B_i），P1(29) 语义以注释保留 |
| 59 | `models/numba_kernels.py` | 9 | 1058 | XPM 循环 | **两者合并** | rs_l_hz=symbol_rate_hz[l_idx]、rs_i_hz=symbol_rate_hz[i] —— B_ℓ 与 B_i 各取各的，未合并成同一标量 |
| 60 | `models/numba_kernels.py` | 10 | 1511 | _calculate_ase_noise_variance 形参 | **两者合并** | b_hz_channels 重命名为 symbol_rate_hz，保留 P1(34) 的 B_i 语义 |
| 61 | `models/numba_kernels.py` | 11 | 1536 | 同上 docstring | **两者合并** | 同上 |
| 62 | `models/numba_kernels.py` | 12 | 1552 | total_ase_variance 算式 | **两者合并** | 两侧算式逐字符相同，只差形参名 |
| 63 | `models/numba_kernels.py` | 13 | 1581 | _calc_span_noise_and_power_jit 形参 | **取 FG + 并入 flexgrid** | 取 FG 单数组签名（无 b_hz_channels/k_bar），注释写明它承载逐信道 B_i |
| 64 | `models/numba_kernels.py` | 14 | 1634 | 形状守卫 | **取 FG** | FG 全套长度守卫等价覆盖 flexgrid 的 b_hz_channels 守卫 |
| 65 | `models/numba_kernels.py` | 15 | 1679 | NLI 调度 + ASE 调用 | **取 FG** | _compute_nli_variances_t 分派（八个 nli_* 门控） |
| 66 | `physics/qot_checker.py` | 1 | 16 | 模块 imports | **取 FG** | FG 单一装配需 rwa.qot_checker 与 ROADMConfig；slots/metrics 两边一致 |
| 67 | `physics/qot_checker.py` | 2 | 30 | _active_channel_mask | **取 FG** | FG-20 裁决（None 成员跳过），test_fg20_none_roadm_adjudication 直接 import 该符号 |
| 68 | `physics/qot_checker.py` | 3 | 135 | __init__ 形参 | **取 flexgrid** | flexgrid 逐连接接线入口（slots_per_channel），既有测试按关键字传 |
| 69 | `physics/qot_checker.py` | 4 | 150 | __init__ 赋值 | **取 flexgrid** | self.slots_per_channel 是 _is_blocked 的权威展开比例 |
| 70 | `physics/qot_checker.py` | 5 | 197 | _is_blocked docstring | **取 flexgrid** | 记录「与 roadm_blocks_node 同序、缺参抛错」的冻结顺序 |
| 71 | `physics/qot_checker.py` | 6 | 211 | _is_blocked 展开比例 | **并集** | 构造参数优先(flexgrid)，FG 的 _slots_per_channel_hint 降为兜底，两者都缺才抛错 |
| 72 | `physics/qot_checker.py` | 7 | 327 | propagate_path | **取 FG** | ROADM 装配全部走 fold_roadm_terms（FG-17 收敛） |
| 73 | `physics/qot_checker.py` | 8 | 414 | evaluate_path | **取 FG** | 活动掩码只留 _active_channel_mask 一处口径 |
| 74 | `rwa/qot_checker.py` | 1 | 7 | 模块 imports(stdlib) | **取 FG** | Callable/Iterable/dataclass 支撑 FG 的共享核与 RoadmTerms |
| 75 | `rwa/qot_checker.py` | 2 | 18 | 模块 imports(包内) | **取 FG（去掉 slots 导入）** | ROADMConfig 必须留；connection_overlaps_slots 因 _is_blocked 改为委派而成为未用导入 |
| 76 | `rwa/qot_checker.py` | 3 | 312 | num_nodes docstring | **取 flexgrid** | 措辞等价 |
| 77 | `rwa/qot_checker.py` | 4 | 324 | _is_blocked + _roadm_terms | **并集** | _roadm_terms 取 FG（三元组封装，FG 新增测试依赖）；阻断判定取 flexgrid 的委派 roadm_blocks_node（单一真源） |
| 78 | `rwa/qot_checker.py` | 5 | 380 | _node_frequency_and_bandwidth docstring | **取 flexgrid** | 第二返回值是 P1(34) 的 B_i |
| 79 | `rwa/qot_checker.py` | 6 | 395 | 取值器返回值 + 旧名别名 | **并集** | 装配走 B_i；_node_frequency_and_rate 恢复为独立方法保留 FG 的符号率语义 |
| 80 | `rwa/qot_checker.py` | 7 | 527 | _calc_link_noise | **取 FG** | 任务书硬要求：噪声算术只有一份，委派 assemble_link_noise |
| 81 | `rwa/qot_checker.py` | 8 | 610 | _eval_service_snr | **取 FG（喂 B_i）** | 递推交给 accumulate_service_snr，共享核形参按 WP-Q1 传 channel_bandwidth_hz |
| 82 | `rwa/qot_checker.py` | 9 | 772 | _build_path_snr_array booster | **取 FG（带宽换 B_i）** | symbol_rate_hz_channels 换成 channel_bandwidth_hz_channels，不把逐信道 B_i 退回符号率 |
| 83 | `rwa/qot_checker.py` | 10 | 806 | _build_path_snr_array 滤波 | **取 FG** | 用折叠出的 filtering_penalty，不再就地 10**(db/10) |
| 84 | `network/elements.py` | 1 | 83 | ROADMConfig 阻断字段 | **自定义合并** | 两侧字段同型同名，仅描述不同；取 flexgrid 的「遗留/只读兼容输入」+「权威字段」措辞 |
| 85 | `network/elements.py` | 2 | 111 | ROADMConfig.blocked_slot_set | **取 FG** | FG 独有的方法（冻结解析顺序：有 blocked_slots 用之，否则按 slots_per_channel 展开） |
| 86 | `network/state.py` | 1 | 51 | LinkState 槽字段 | **全取 flexgrid** | 实测 FG 侧 state.py 对 slots_per_channel / occupied_slots / allocated_service_ids_slots / allocate_slot_range 的引用数均为 **0** ⇒ 无等价行为被丢弃 |
| 87 | `network/state.py` | 2 | 67 | __post_init__ 槽初始化 | **全取 flexgrid** | 同上；slots_per_channel 正整数校验与槽数组默认构造 |
| 88 | `network/state.py` | 3 | 126 | allocate_slot_range / release_slot_range | **全取 flexgrid** | 同上；93 行槽区间分配/释放（业务层状态机），FG 侧无对应行为 |

---

## 3. 语义侦察的关键结论（决定处置的依据）

1. **`BandSpec` 是同一份定义的两次落地。** AST 差分：两侧方法集**完全相同**（各 7 个），
   flexgrid 侧独有符号只有 `_THz` 常量（外部无引用）。⇒ 去重，不是取舍。
2. **`BandPlan` 上 FG 是严格超集**：FG 多 5 个方法（`from_channels`、
   `noise_figure_db_per_element`、`launch_power_profile_w`、`per_band_launch_power_w`、
   `guard_report`）、flexgrid 多 0 个；模块函数 FG 多 3 个（`hz_to_nm`、`_u_spec`、
   `sclu_span_report`）。
3. **`numba_kernels.py` 两条线各自独立演化。** merge-base 上 `ode_accel` 出现 0 次，
   flexgrid 侧 11 次、FG 侧 22 次 —— 两侧各自实现了同一套加速接口。
4. **`B_i` 逐信道参数化在 flexgrid 侧，`ode_accel`/批量结构在 FG 侧**，互不覆盖 ⇒ 「结构取 FG、
   参数取 flexgrid」可执行。
5. **ROADM 阻断判定两侧各写了一遍**（FG 的 `_resolve_blocked_node_indices` 与 flexgrid 的
   `roadm_blocks_node`）。合并后 `roadm_blocks_node` **委派**前者，单一真源。

---

## 4. 数据模型边界（任务书 §2.3 的转换函数契约）

两套「灵活栅格」表示**都保留**，边界与转换契约：

| 层 | 表示 | 拥有什么 | 归属 |
|---|---|---|---|
| 物理层 | `core/flexgrid.py::FlexibleGrid` / `ConnectionSpec` | 一条连接的频谱占位与速率：`grid`/`first_slot`/`n_slots`/`bit_rate_bps`/`modulation`/`symbol_rate_hz`/`roll_off`/`launch_power_w` | FG（FG-2） |
| 业务层 | `traffic.py::ServiceRequest.n_slots`、`AllocatedService.first_slot`、`network/slots.py`、`rwa/allocators.py::KSPFirstFitFlexAllocator` | 到达 / 分配 / 释放的状态机 | flexgrid（WP-M1） |

**转换函数**（本次新增，`src/network_nlo_eval/core/flexgrid.py`）：

- `connection_spec_from_allocation(service, grid, *, modulation, roll_off, symbol_rate_hz=None, connection_id=None) -> ConnectionSpec`
- `slot_window_from_connection(spec) -> tuple[int, int]`（逆方向投影）

契约要点：

1. **真源分工，禁止双写状态。** `first_slot` / `n_slots` / `bit_rate_gbps` / `launch_power_w`
   只从业务层对象**读**；`modulation` / `roll_off` / `symbol_rate_hz` 只由物理层参数决定。
   转换函数**不缓存、不写回**，两侧不可能漂移。
2. **不一致时抛错，不静默改写。** 业务层 `n_slots` 与所选调制算出的需求不符时抛 `ValueError`
   （`ConnectionSpec` 不变式 3），而不是把槽窗口的真源偷偷搬回物理层。
3. **不反向依赖。** 参数用鸭子类型而非直接 import `AllocatedService`，使 `core` 层不依赖
   `simulation` 层。
4. **往返有测试。** `tests/test_merge_fg_into_flex.py::TestRepresentationBoundary`（6 条），
   含真实 `AllocatedService` 的往返。

---

## 5. 两条判据客观互斥（含可复算证据）

### 5.1 J5 冻结内核 vs FG-14 冻结内核哈希

| 对象 | `sha256(src/network_nlo_eval/models/numba_kernels.py)` |
|---|---|
| `b69563a`（flexgrid 的 WP-O3 起点） | `6611c4164a60d0dcc653e47c5f02d81534e9c78b03e71213aa30d8b13bfd4aa4` |
| `flexgrid-dev @ cb24704` | `6611c416…`（同上，两侧未变） |
| `wp/fg-integrate @ 57c7221` | `7beece4ea353b1bf5119ab9708d4db9845b092328357fff30f9b806afe613205` |
| 合并树（最终） | `7beece4e…`（取 FG 侧） |

- `tests/test_ode3_flexgrid_sync.py::test_guard2_j5_frozen_kernel_diff_is_empty` 要求
  `git diff b69563a -- numba_kernels.py` **为空**，即内核必须等于 `6611c416…`。
- `tests/test_fg14_assembly_provenance.py::test_fg14_ode_kernel_is_byte_frozen` 要求内核
  `sha256 == 7beece4e…`。

**同一文件不可能同时有两个 SHA-256** —— 这是客观互斥，不是取舍偏好。
判据 **V1（首要判据）** 要求合并树与 `wp/fg-integrate` 逐位相同，故取 FG 侧字节。

处置：按 addendum §2 通例把 **J5 守卫翻成正向不变量**（不再断言「diff 为空」这一**相对**断言，
改为断言「内核逐字节等于已核出处的 sha256」这一**绝对**不变量，更强：同时挡住「被悄悄改一行」
与「合并出第三份实现」）。

### 5.2 Q-J5 要求内核出现 `b_hz_channels[i]` 字面量

`tests/test_bwqot_bandwidth.py:190/193` 要求公式锚点里出现字面量 `b_hz_channels[l_idx]` /
`b_hz_channels[i]`。而：

```
$ grep -rc -F 'b_hz_channels[i]' src/     → 0
$ grep -rc -F 'b_hz_channels[l_idx]' src/ → 0
```

合并后内核（逐字节取 FG）用单数组形参 `symbol_rate_hz` 承载 `B_i`。该守卫同时还要求
「锚点文本必须能在源码里 grep 到」，两条要求互相矛盾 ⇒ **结构上不可满足**。

处置：**未伪造**。逐信道 `B_i` 的语义保留在
`isrs_gn.py` 的 `channel_bandwidth_hz_channels` / `_power_width_scale` 与调用点
`symbol_rate_hz=self.channel_bandwidth_hz_channels`，并有 18 条新增测试守着。
代价与未闭合项如实记在 §7。

### 5.3 公式锚点：47 → 31，剩余 31 条的实现已被合并删除

`tools/ode_formula_map.py --check` 的失配锚点由 **47 条降到 31 条**（16 条重指向到合并后
真正实现该公式的行，逐条 grep 验证）。剩余 31 条**无法诚实重指向**，因为它们的实现已随
V1 的裁决被删除：

- `p1_eq18` / `p1_eq20` / `p1_eq22` / `p1_eq31`：flexgrid 的**解析倾斜**路径
  （`_calc_raman_profile_jit`、`_calc_span_out_power_jit`）已被 FG 按死代码删除 ——
  `tests/test_fg1_physics_parity.py:107-108` 现在**断言它们不存在**。
- `p1_eq29`（SPM）/ `p1_eq30`（XPM）：合并后的实现是 **P2 Eq. 21** 的张量化形式，不是 P1 (29)/(30)
  的闭式。常数与结构都不同，例如 `(4/9)` vs `(16/27)`、`asinh(φB²/(πα))` vs
  `asinh(3φB²/(8π·ã_t))`、`φ_i = (3/2)π²(β₂+2πfβ₃)` vs `φ_i = −4π²(β₂+2πfβ₃)`。
  把 P1 (29) 指到这些行等于**伪造出处**。
- `p1_eq34_ase`：合规的 ASE（EDFA 增益块）随「跨段 EDFA」一并删除，现在唯一活着的是
  `_calculate_ase_noise_variance`（原 `p1_eq34_ase_dead_kernel`）。两条记录角色对调。

这些条目**没有被删除、也没有改判据口径**（公式号、`paper` 出处、`kind` 一律未动），
只是它们的锚点在合并后确实找不到对应实现。

### 5.4 「跨段内施加 EDFA」：为满足 V1 而删除

flexgrid 侧在 `_calc_span_noise_and_power` 末尾有一段「显式应用 EDFA」
（`p_out = p_out_fiber * gain`，并用增益与 NF 重算 ASE）。合并取回 FG 口径，理由三条：

1. **V1（首要判据）**：实测保留该块时 `cl_full_100km/p_out_w` 由 `3.775e-05`（光纤输出）
   变成 `1.000e-03`（EDFA 后），`sigma2_ase_w` 同步被重写 ⇒ **7 场景 29 项全部不一致**。
   删除后同口径复测 **0 不一致**。
2. **重复计入**：RWA 层（`rwa/qot_checker.py` / `rwa/link_noise.py`）自己按
   `NF·h·ν·B_i·(gain−1)` 算放大器 ASE，两处都算会重复。
3. FG 的场景/脚本（`fg1_scenarios` / `fg4a_scenarios` / `fg14_divergence_probe`）都把本函数
   的返回值直接喂回下一跨段，语义就是「跨段输出」。

### 5.5 `use_attenuation_polynomial`：为满足 V1 而不由 isrs_gn 读取

该开关在 **FG 树里本就只对波形级 `physics/qot_checker` 生效**，`models/isrs_gn.py` 从不读它
—— 这正是 FG-14 记录并裁决过的「装配口径分歧」（`docs/fg14_assembly_provenance.md`）。
flexgrid 侧让 isrs_gn 也读它，默认档（`False` ⇒ 常数衰减）会让 **alpha 与 FG 不同 ⇒ 每个物理量都变**。
实测：修复前 `off` 档 7 场景 29 项全不一致；修复后 0 不一致。
flexgrid 侧**没有任何测试**依赖 isrs_gn 读这个开关（`git grep use_attenuation_polynomial HEAD -- tests/` 无命中）。

### 5.6 `raman_spectrum` 与 legacy-only 档位

FG 的守卫 `test_fg15_legacy_bitwise.py:116` 要求
`MultiBandISRSGN.__init__.__defaults__[-1] == "tabular"`（字面量且位于末位）。
而 flexgrid 的 `test_ode_dev.py` 用 `ode_accel="off"/"cached"` 构造且**不传**该参数。
`off` 与 `cached` 都是 **legacy-only** 的冻结路径（前者核内逐对重建 legacy 折线，后者只是把它
记忆化），结构上无法换谱。

处置：签名恢复 FG 口径（末位、默认 `RAMAN_SPECTRUM_DEFAULT = "tabular"`），保留 FG 的
「显式拒绝而非静默回退」；flexgrid 侧那 3 处构造**显式**配 `raman_spectrum="legacy"`
（与 FG 自己在 `tests/fg1_scenarios.py` 的做法一致）。属已声明的测试改动，见 §6。

---

## 6. 判据实测读数（V1–V5）

全部命令可复算。物理树一律由 `PYTHONPATH` 切换。

### V1 —— FG 判决摘要逐位相同（**通过**）

探针：`evidence/merge/v1_rollout_probe.py`（新增）。口径与 RL 仓
`tools/fg11_rollout_trace.py` 逐字一致（同一策略 `KSPFirstFitPolicy`、同一
`build_env`、同一逐步决策三元组），只把 seeds 从 `(0, 7)` 扩成任务书要求的 **0..3**，
场景 `assets/pan_europe_multiband.json`，`num_requests=40`、
`avg_arrival_interval=10.0`、`avg_holding_time=400.0`。决策摘要 = 每步
`(action, channel, allocated)` 序列的 SHA-256。

```bash
cd /home/zsig/Documents/gits/wt/fg-rt
PYTHONPATH=$PWD/src:<TREE>/src \
  ~/miniconda3/envs/rl/bin/python <MERGE_WT>/evidence/merge/v1_rollout_probe.py \
  --out <json> [--compare <other.json>]
```

| seed | steps | `wp/fg-integrate @ 57c7221` | 合并树 | reward | alloc | blocked |
|---:|---:|---|---|---:|---:|---:|
| 0 | 40 | `ef766e29aa05c479` | `ef766e29aa05c479` | 16.0 | 21 | 480 |
| 1 | 40 | `078175297a017c6e` | `078175297a017c6e` | 18.0 | 21 | 544 |
| 2 | 40 | `614c144a69191b8a` | `614c144a69191b8a` | 18.0 | 20 | 484 |
| 3 | 40 | `d03602fde3e2ff20` | `d03602fde3e2ff20` | 20.0 | 24 | 452 |

**`COMPARE: seeds=4 mismatched=0`** ⇒ V1 通过。
证据：`evidence/merge/v1_{fg_integrate,merged_tree}_rollout.json`（各含 `provenance`
记录两棵物理树的 commit，交叉核对无误）。

另有更细的物理层逐位对照：`off` 档 7 个场景 × 29 项读数，合并树 vs `wp/fg-integrate`
**0 不一致**（`fg4a_scenarios.build_scenarios(ode_accel="off", raman_spectrum="legacy")`）。

### V2 —— flexgrid 多槽分配决策逐位相同（**通过**）

**为什么换探针**：RL 仓 `wp/fg-rt` 上的 `multiband_optical_rl.config` 已硬依赖 FG 符号
（`from network_nlo_eval.models.isrs_gn import RAMAN_SPECTRUM_DEFAULT`），而
`flexgrid-dev` 没有这些符号 ⇒ 同一条探针在 flexgrid 树上**根本 import 不了**：

```
ImportError: cannot import name 'RAMAN_SPECTRUM_DEFAULT' from
'network_nlo_eval.models.isrs_gn' (.../wt/flexgrid-dev/src/...)
```

任务书 V2 允许「同一探针口径**或多槽专用探针**」，故新增
`evidence/merge/v2_multislot_probe.py`，**只用两棵树都有的符号**
（`LinkState.allocate_slot_range` / `release_slot_range`、
`first_fit_contiguous_windows`、`connection_overlaps_slots`），请求序列 1/2/3/4 槽混排
（`SLOT_DEMANDS`，12 步，240 槽，每信道 4 槽），逐步记录
`first_slot`/`n_slots`/占用摘要，最后**逐个释放**验证释放路径。

| 对象 | `phys_commit` | `decision_sha256` |
|---|---|---|
| `flexgrid-dev @ cb24704` | `cb247045dab6…` | `895d9d04d5f4e6efee3d75209f1b0da470458d33fcad69434887e159e1680345` |
| 合并树 | `6b8304fbaeb1…` | `895d9d04d5f4e6efee3d75209f1b0da470458d33fcad69434887e159e1680345` |

**`steps_equal=True final_equal=True sha256_equal=True`** ⇒ V2 通过。
证据：`evidence/merge/v2_{flexgrid_dev,merged_tree}_multislot.json`。

### V3 —— 逐信道 `B_i` 等价性（**在 FG 自己的 1e-12 预算内通过；字面「逐位」不成立**）

`tests/test_merge_fg_into_flex.py::TestPerChannelBandwidth`，8 个等价性用例。
前置断言已确认这条链路上 `B_i` **真的非均匀**（`occupied` 档 = `n_slots × 12.5 GHz`
= `[1.25e10, 2.50e10, 3.75e10, 5.00e10]`，且与 `symbol_rate` 档不同），
否则「逐信道」就是空话。

| 判据 | 逐位? | max_rel | 备注 |
|---|---|---:|---|
| `off ≡ off`（同 Raman 档） | **是** | `0.000e+00` | 自洽 |
| `off ≡ cached_g`（同 Raman 档） | **是** | `0.000e+00` | |
| `off ≡ matvec`（同 Raman=legacy） | 否 | `2.041e-16` | 纯浮点重排 |
| `cached_g ≡ matvec`（同 Raman=tabular） | 否 | `2.415e-16` | 纯浮点重排 |
| `batch ≡ serial`，B=8 | 否 | `1.707e-16` | |
| `batch ≡ serial`，B=16 | 否 | `1.707e-16` | |
| `batch ≡ serial`，B=64 | 否 | `1.707e-16` | |
| `batch ≡ serial`，B=142 | 否 | `1.707e-16` | |

**关于任务书写的「逐位」**：三条都在 float64 求和次序的最后一两位（1.7e-16 ~ 2.4e-16），
比 FG 自己定的 `MAX_REL_DEVIATION_ALLOWED = 1e-12` 小四个数量级。**这不是本次合并引入的**：
FG-4a 自己的文档写明 `matvec` 档**按设计**重排浮点求和，该预算常量被明文标注为
"never widened"，且它给三个既有测试（`test_f4a_1`…）定的标准就是
`off`/`cached_g` 逐位、`matvec` 有界。⇒ V3 字面要求的「`cached_g ≡ matvec` 逐位」
与 FG 的设计本身互斥；本次按 FG 已文档化的预算断言，并**在每个用例里打印逐位读数**，
不掩盖事实。

另注：`off` 与 `matvec` 若**不固定** Raman 档，`max_rel = 2.056e-04` —— 因为 FG-15 把
`"tabular"` 定为全局默认，而 `off` 是 legacy-only 的冻结路径。这不是数值误差，是
**物理档位不同**，所以比较"纯加速维度"时必须把档位钉住（表中已如此处理）。

### V4 —— 全部测试（**11 failed / 578 passed / 2 skipped**，38.3 s）

基线对照：`flexgrid-dev @ cb24704` 是 `2 failed / 66 passed / 1 skipped`，
`wp/fg-integrate @ 57c7221` 是 `477 passed / 2 skipped`；合并后套件共 591 项。

11 个失败的归因（**没有一个是"为了让测试变绿"而留下的**）：

| 失败 | 归因 | 是否本次合并造成 |
|---|---|---|
| `ode_dev[ode_compare.py]` | `ModuleNotFoundError: fu4_scenarios_legacy` | **否**，基线既有 |
| `bwqot_legacy_regression.py` | 同上 | **否**，基线既有 |
| `bwqot_probe.py` | 导入 `_calc_raman_profile_jit`（该解析内核已被 V1 裁决删除） | 是 |
| `bwqot_cost.py` | Q-J3「内核不得出现 ODE 标识符」——合并后内核**就是** FG 的 ODE 内核 | 是 |
| `qj5_per_channel_bandwidth_mapped_to_p1_23_29_30_34` | 要求字面量 `b_hz_channels[i]`（见 §5.2） | 是 |
| `qj5_formula_map_check_reproduces`、`ode3[guard8]`、`ode3[guard9b]`、`ode3[guard9[relocate_anchors]]`、`ode3[guard9[flexgrid_e2e]]`、`ode_dev[ode_formula_map]` | 公式图谱 **32 条**锚点失效（见 §5.3） | 是 |

即：**2 个是环境预存问题（与合并无关），9 个全部溯源到同一个根因** —— V1 要求内核逐字节
取 FG 侧，而 flexgrid 的 J1/J2 证据体系是为**它自己的闭式实现**建的记录。已重定位 15 条
（47 → 32），剩余 32 条**不伪造**。

### V5 —— 已声明的测试/守卫改动（全部旧文本 → 新文本）

| # | 位置 | 旧文本 / 旧语义 | 新文本 / 新语义 | 理由 |
|---:|---|---|---|---|
| 1 | `tests/test_ode3_flexgrid_sync.py` guard 7 | 断言 `SCLU_BAND_SPECS: tuple[BandSpec, ...]`、`def sclu_span_report(`、`def guard_report(`、`def hz_to_nm(`、`U_BAND_ASSUMPTIONS: tuple[str, ...] = (` **均不在** `bands.py` 中 | 三条接入点断言原样保留；翻成正向不变量：五个符号**存在**（`hz_to_nm`/`sclu_span_report`/`BandPlan.guard_report` 可调用、`U_BAND_ASSUMPTIONS`/`SCLU_BAND_SPECS` 为非空 tuple 且含 S/C/L/U）、`class BandSpec` 与 `class BandPlan` 在 `bands.py` 里**各只定义一次**（正则计数 == 1）、且真能调用 | 原为相位守卫（docstring 自称「不搬 FU-2 资产」），前提已消失 |
| 2 | `tests/test_ode3_flexgrid_sync.py` guard 2（J5） | `git diff b69563a -- numba_kernels.py` **必须为空** | 内核 `sha256` **必须等于** `7beece4e…`（= `wp/fg-integrate @ 57c7221` 的冻结字节）；`results/…flexgrid_sync.json` 的 `j5_frozen_kernel` 改用 `frozen_sha256` / `provenance_branch` 字段 | §5.1 的客观互斥；绝对不变量比相对断言更强 |
| 3 | `results/ode_dev_flexgrid_sync.json` 的 `j5_frozen_kernel` | `{"file": …, "diff_vs_wp_base_empty": true}` | `{"file": …, "frozen_sha256": "7beece4e…", "provenance_branch": "wp/fg-integrate", "provenance_commit": "57c7221", "flip_note": …}` | 同上 |
| 4 | `results/ode_dev_flexgrid_sync.json` 的 11 个 `not_ported` 段 | `action: "not_ported"` + `verify_absent` | `action: "ported"` + `verify`（每条都实测 grep 到）+ `verified: true`，`rationale` 追加翻转说明 | addendum §1.2；实测锚点见处置表与 §5.3 |
| 5 | `tools/ode3_sync.py` | 上述 11 段硬编码 `not_ported`；J5 断言 diff 为空；`--check` 断言 `verify_absent` grep 不到 | 同 3、4；`--check` 改为「已应用段的 `verify` 必须 grep 到；`not_ported` 段的 `verify_absent` 仍须 grep 不到」（后者保留） | 工具是事实源，JSON 由它派生 |
| 6 | `tools/ode3_report.py` | 渲染 J5 为「相对 `b69563a` 的 diff 为空 ✅/❌」 | 读绝对哈希字段渲染，schema 不符时回退旧路径 | 否则会把已通过的 J5 印成 ❌ |
| 7 | `tests/test_fg4a_numeric_identity.py` | `BIT_EXACT_MODES = ("off", "cached_g")` | `BIT_EXACT_MODES = ("off", "cached", "cached_g")` | `test_f4a_1_all_modes_are_exercised` 要求**每个声明的档位都有标准**。`cached` 是纯记忆化 ⇒ 与 `off` 按构造逐位相等，标准即 bit-exact。`BIT_EXACT_IDS` 会用金标准**真跑一遍** `cached` ⇒ 这是**扩证据**，不是放宽 |
| 8 | `tests/test_ode_dev.py`（3 处构造） | `MultiBandISRSGN(grid, fiber, 32e9, ode_accel=mode)`（不传 Raman 档） | 显式配 `raman_spectrum=_raman_tier_for(mode)`；`off`/`cached` 配 `"legacy"`，其余 `"tabular"` | `off`/`cached` 是 legacy-only 冻结路径；与 FG 自己在 `tests/fg1_scenarios.py` 的做法一致 |
| 9 | `tests/test_ode_dev.py`（断言 (d)） | `assert np.allclose(auto[0], power)`（依赖"跨段内施加 EDFA、`target_gain_db=None` 时自动均衡回入纤功率"） | `assert np.all(np.isfinite(auto[0]))` 且 `0 < auto[0] < power` | §5.4 删除该段后，跨段放大由调用方负责；新断言钉合并后的真实契约，并显式防"加回去" |
| 10 | `src/.../isrs_gn.py`（非测试，行为声明） | flexgrid 版在 edfa 为 None 时 `current_nf_lin = np.ones_like(...)`（= NF **0 dB**） | 取 FG 的 `self.nf_lin_channels`（形状校验仍用 flexgrid 更严的版本） | 实测该差异让串行 `sigma2_ase_w` 比 FG 小 **3.981 倍**，`batch≡serial` 直接差 2.981（预算 1e-12） |
| 11 | `src/.../isrs_gn.py`（非测试，行为声明） | `if self.ref_fiber_config.use_attenuation_polynomial:` 二分支 | 恒用 `_calc_fiber_attenuation_npm` 多项式 | §5.5；FG 树里该开关本就只对 `physics/qot_checker` 生效，属 FG-14 已裁决的装配分歧 |

---

## 7. 我没做什么 / 没跑完什么

如实列出，**不粉饰**：

1. **V4 未达成「零失败」**。11 个失败里 9 个是本次合并造成的（根因单一，见 §5.3/§5.2）：
   公式图谱 32 条锚点指向的 P1 闭式实现已随 V1 的内核裁决被删除，`Q-J3` 的「内核不得含
   ODE 标识符」与 V1 直接对撞，`Q-J5` 要求内核出现 `b_hz_channels[i]` 字面量而合并后内核
   没有该标识符。我**没有**把这些锚点指到语义不同的行来凑绿，也没有改预算/加 skip。
   可选的收尾路径（需你裁决）：把 P1 (29)/(30) 的记账整体重指向 P2 Eq. (21)（**这会改动
   flexgrid 的判据口径**，任务书不允许我做）；或正式退役这批条目并改 `ode_formula_map.py`
   的口径（同样是口径变更）；或补回 P1 闭式内核（与 V1 对撞，不可行）。
2. **V5 的 "全部测试绿" 未达成**，且我改了 11 处（§6 V5 表）。其中 #1/#2/#3/#4/#5 是
   addendum §2 授权的相位守卫翻转；#7–#9 是我认为必须声明的测试改动，但它们确实**动了测试**。
3. **V3 没有做到任务书字面要求的「逐位」**（三条都是 1.7e-16 ~ 2.4e-16 的浮点重排）。
   这是 FG 自身的设计（`matvec` 按设计重排，预算 1e-12），不是本次合并引入的。详见 §6 V3。
4. **两条 `results/…flexgrid_sync.json` 的段我最终没有翻成 `ported`**：
   - `f_m_hz/f_M_hz` 频带边缘那段：代码是 WP-O2 **既有**的（本次只合并了注释），且
     `f_m_hz`/`f_M_hz` 全仓**零读取者**。翻成「已移植」是假陈述，故保留 `not_ported`。
   - `numba_kernels.py` 整个文件：内核确实在树里（sha256 已匹配），但 J7 要求 physics 段
     必须有 `paper_formulas` 授权，而它的物理出处是 FG 的 RK4 ODE / P2 Eq. 21，
     **不是** P1 的闭式。授权链在另一套纪律里（`docs/fg4b_nli_vectorization.md`、
     `docs/fg14_assembly_provenance.md`、FG 自己的金标准与 1e-12 闸门）。
     我没有为了过闸门而编造公式号，保留 `not_ported` 并在此登记为未闭合项。
5. **`wt/fg-integrate` 与 `network-nlo-eval` 主检出全程只读**；没有 push；
   `wp/fg-rt` 只被读取（探针靠 `PYTHONPATH` 切物理树），未改动。
6. **性能/规模维度没有覆盖**：FG 的基准（`results/fg4e_speedup_ab.json` 等）与 RL 侧探针的
   批量化收益没有重测，只验证了**等价性**（批量 ≡ 串行）与正确性。
7. **`evidence/merge/baseline_flexgrid_pytest.txt` 只记了 flexgrid 侧的基线**；
   FG 侧的 477/2 基线取自当时 `wt/fg-integrate` 的一次运行，未单独落盘成文件。

---

## 8. 第 3 轮：两套 NLI 物理**都留**，用 `nli_model` 档位区分

> 本章是**第 3 轮**（编排方裁决）的交付。第 1、2 轮的章节（§0–§7）**原样保留**，
> 不追改：第 2 轮结束时那些读数在当时是真的。本章只说明**第 3 轮改了什么、
> 为什么改、实测读数是多少**。
>
> ⚠️ **第 4 轮的追改说明**（历史陈述保留，但**当前状态以 §9 为准**）：
> 第 3 轮给 `p1_closed` 定的默认加速档是 `"off"`，**原因**是「本轮不做 P1 核的
> 加速（另立一轮）」。**第 4 轮**（§9）补上了 P1 的 `matvec` 档，那个原因消失，
> 默认档按实测裁决改成了 **`"matvec"`**。本章与 §8.4 里的「off 档」读数**仍然是
> 真的**——它们说的是**冻结参考路径**这个档位，而那个档位**一个字没动**。

### 8.1 第 3 轮改判的来由（**是判据错了，不是执行错了**）

第 2 轮把 `models/numba_kernels.py` 整份换成 FG 的字节（sha256 `7beece4e…`），
并把 flexgrid 的逐信道 `B_i` **改名成** `symbol_rate_hz` 喂进 P2 的公式。
第 3 轮编排方查明：这两份核是**两套不同物理**，那个组合**两边都没验证过**。

| | `numba_kernels.py`（P1，flexgrid WP-Q1） | `numba_kernels_p2.py`（P2，FG） |
|---|---|---|
| 物理 | P1 (23)/(29)/(30)/(34) 闭式 | P2 Eq. 21 张量化 |
| 带宽量 | `b_hz_channels[i]` = 信道带宽 `B_i` / `B_ℓ` | `rs_hz` = **符号率** |
| 指纹 | `asinh(φ_i·B_i²/(πα))`、`atan(φ_{i,ℓ}·B_i/α_ℓ)` | `(16/27)·γ²P³/rs²·T²`、`asinh(3φ·rs²/(8π·a_t))` |
| 字节 | `6611c4164a60d0dcc653e47c5f02d81534e9c78b03e71213aa30d8b13bfd4aa4` | `7beece4ea353b1bf5119ab9708d4db9845b092328357fff30f9b806afe613205` |

⇒ **第 3 轮目标状态**：两套都保留，用显式档位 `nli_model` 区分，谁都不许被另一套替换。

**第 3 轮还查出第 2 轮漏掉的一处**：那个改名**不止**在核的形参上，它还漏在
**五处** P2 路径上（内核调用、张量化 NLI 的**构建期张量**、批量常量、两处
booster ASE）。第 2 轮的套件是绿的，因为那 11 条用例**都测 P2 路径**，
而改名让 P2 的三条加速路径（`off` / `matvec` / 批量）**互相打架**：
实测 `off` vs `matvec` 的 SPM 相对偏差 **2.166**（预算 1e-12）。
第 3 轮把这五处全部改回 FG 口径后，`off ≡ matvec` 恢复 **2.851e-16**。

### 8.2 第 3 轮的实现

**档位表**（`models/isrs_gn.py`）：

| 档位 | 核 | 物理 | `ode_accel` 允许值 | `ode_accel` 默认 | `raman_spectrum` 允许值 | 默认 |
|---|---|---|---|---|---|---|
| `p1_closed`（**默认**） | `numba_kernels.py`（`6611c416…`） | P1 闭式，逐信道 `B_i` | `off` / `cached` | `off` | `legacy` | `legacy` |
| `p2_gn` | `numba_kernels_p2.py`（`7beece4e…`） | P2 Eq. 21，符号率 | `off` / `cached` / `cached_g` / `matvec` | `matvec` | `legacy` / `tabular` | `tabular` |

**不匹配即显式拒绝**（不静默回退）：给 `p1_closed` 配 `matvec` / `cached_g` /
`raman_spectrum="tabular"` 会在构造期抛 `ValueError`。理由：P1 核里**没有**拉曼 G
矩阵、**没有**去分支 matvec、**没有**张量化 NLI、**没有** tabular 谱；硬跑等于
**静默**把请求丢掉——那是最坏的一种错：数字对、账对不上、没人知道加速没生效。

**P1 档不是「FG 减去点什么」**：它的四处口径是**逐字取自** `flexgrid-dev @ cb24704`
的 ——① NF 默认值（未指定时 `np.ones_like`，即 NF 0 dB）；② 跨段后放大在 Python 层
**显式**施加，P1 (34) 的 ASE 按 `NF·h·ν·B_i·(G−1)` 逐信道算（核里那个 ASE 分支是
**死代码**，锚点 `p1_eq34_ase_dead_kernel` 就是这条事实的可执行形式）；
③ 衰减口径读 `use_attenuation_polynomial`（默认 ⇒ **常数**衰减）；
④ ROADM 装配按 `path_loss_db(operation)`、等化档「先除以 loss 再乘回 loss」、
legacy 模式的阻断集合只看 `blocked_channels`。

### 8.3 88 行处置表**追加说明行**（原 88 行**继续有效**）

第 3 轮**没有**推翻第 2 轮的任何一行裁决，只是把其中 5 行的「取哪边」**按档位
分列**（两套物理各取各的），并为新的文件形态补 3 行。

| # | 文件 | 块号 | 所属函数 / 类 | 第 3 轮处置 | 理由 |
|---:|---|---:|---|---|---|
| 89 | `models/numba_kernels.py` | 1–15（整份） | 整个文件 | **拆成两份**：`numba_kernels.py` 恢复 flexgrid 的 `6611c416…`（P1），FG 的 `7beece4e…` **同字节**搬进 `numba_kernels_p2.py` | 第 3 轮改判：两套物理都留。第 2 轮的行 51–65（逐块取 FG）**在 P2 那份里原样成立** |
| 90 | `models/isrs_gn.py` | 14（行 42） | FG-4a/4b/4e 加速区块 | **分列**：`_nli_tensors_for_span` / `_batch_constants_for_span` 的 `rs` 实参改回 `symbol_rate_hz_channels` | 第 2 轮那一行写着「⚠ 其中 rs= 传入已由 symbol_rate_hz_channels 改为 channel_bandwidth_hz_channels（逐信道 B_i）」——**这正是第 3 轮判废的那个改名** |
| 91 | `models/isrs_gn.py` | 15（行 43） | EDFA NF 解析 | **分列**：P1 档取 flexgrid 的 `np.ones_like(...)`，P2 档取 FG 的 `self.nf_lin_channels` | 第 2 轮取的是 FG 的默认值（因当时只有 P2 一套物理）；P1 档要逐位等于 cb24704，必须取回 flexgrid 的 |
| 92 | `models/isrs_gn.py` | 16–17（行 44–45） | `_calc_span_noise_and_power` | **拆成派发口 + 两条实现**：P1 档走 `_calc_span_noise_and_power_p1`（P1 核 + 显式 EDFA + flexgrid-dev 的 ROADM 口径），P2 档走 `_calc_span_noise_and_power_p2`（FG 原样） | 第 2 轮把这一块「取 FG」是因为当时只有 P2；第 3 轮两套都留 |
| 93 | `models/isrs_gn.py` | 20（行 48） | `for_fiber` 实参 | **追加**透传 `nli_model` | 子评估器不能悄悄换档（默认档已是 `p1_closed`） |
| 94 | `rwa/qot_checker.py` | —（新增行） | `_node_frequency_and_bandwidth` | 改读 `evaluator.ase_bandwidth_hz_channels` | 两套物理的 **ASE 带宽量**本来就不同（P1 用 `B_i`、P2 用 `rs`）。该属性在**物理仓**按档位解析好，**装配层仍不出现任何档位字样**（FG-4g 的「装配层对档位无知」守卫原样通过） |

### 8.4 判据实测读数（V1'–V5'）

全部命令可复算。物理树一律由 `PYTHONPATH` 切换。

#### V1' —— 两套物理各自与自己的源头逐位相同（**两条都通过**）

**`p2_gn` 档 vs `wp/fg-integrate @ 57c7221`**

探针：`evidence/merge/v1p_p2_probe.py`（新增）。口径与第 2 轮完全一致：
`fg4a_scenarios.build_scenarios(ode_accel="off", raman_spectrum="legacy")` 的 **7 个场景**。

```bash
cd /home/zsig/Documents/gits/wt/fg-integrate
PYTHONPATH=$PWD/src:$PWD/tests ~/miniconda3/envs/rl/bin/python \
  <MERGE_WT>/evidence/merge/v1p_p2_probe.py --out <json>
cd <MERGE_WT>
PYTHONPATH=$PWD/src:$PWD/tests ~/miniconda3/envs/rl/bin/python \
  evidence/merge/v1p_p2_probe.py --out <json> --compare <fg json>
```

| 项 | 读数 |
|---|---|
| 场景数 | **7** |
| 逐项读数 | **6364** |
| 不一致 | **0** |
| 读数 sha256（两侧） | `12ddc4bd12f77ea009410be2fa6a94c62bfd7bb5351a14c4c912e9cee06a703c`（**相同**） |

> ⚠️ **这一条是重测的，不是复用**。编排方说「上轮已拿到这个读数，直接复用」，
> 但第 3 轮改了 P2 路径的 5 处带宽实参（**物理**改动，量级远超浮点噪声），
> 复用旧读数会**失真**。故如实重测。
>
> 另：第 2 轮的 **RL 侧**判决摘要（4 个 seed，决策三元组 sha256
> `ef766e29…` / `078175297a017c6e`…）**仍然有效**——RL 的默认带宽口径是
> `channel_bandwidth_source="symbol_rate"`，该档下
> `channel_bandwidth_hz_channels` 与 `symbol_rate_hz_channels` 是**同一个恒等式**
> （逐位相同），故第 3 轮的改名修复**不改变** RL 读数。这条复用有据，不是偷懒。

**`p1_closed` 档 vs `flexgrid-dev @ cb24704`**

探针：`evidence/merge/v1p_p1_probe.py`（新增）。**只用两棵树都有的符号**（`SpectrumGrid` /
`MultiBandISRSGN` / `FiberSpanConfig` / `EDFAConfig` / `ROADMConfig` / `MultiSpanOpticalPath`），
所以**同一个脚本**能分别量两棵树。6 个场景（5 个跨段 + 1 个端到端含 ROADM）、
逐项读数含 `p_out_w` / `sigma2_spm_w` / `sigma2_xpm_w` / `sigma2_ase_w` /
`signal_power_w` / 三项噪声 / `snr_db` / `span_count` / `roadm_count`。

```bash
cd /home/zsig/Documents/gits/wt/flexgrid-dev
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python \
  <MERGE_WT>/evidence/merge/v1p_p1_probe.py --out <json>
cd <MERGE_WT>
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python \
  evidence/merge/v1p_p1_probe.py --nli-model p1_closed --out <json> --compare <flexgrid json>
```

| 项 | 读数 |
|---|---|
| 场景数 | **6** |
| 逐项读数 | **5752** |
| 不一致 | **0** |
| 读数 sha256（两侧） | `14d77108b5616d39635b914a1fb92f30c65568f08eb461f17ee06a158903ae63`（**相同**） |

> ⚠️ **「分配判决」这一半的读数边界**：端到端的 RL 判决摘要**不能在** flexgrid 树上跑
> ——RL 仓 `wp/fg-rt` 的 `multiband_optical_rl.config` 硬依赖 FG 符号
> （`from network_nlo_eval.models.isrs_gn import RAMAN_SPECTRUM_DEFAULT`），
> flexgrid 树**连 import 都过不去**（第 2 轮 §6 V2 已实测并记录）。故「分配判决」
> 在 `p1_closed` 档由**两条**证据合起来覆盖：
> ① **物理层逐位相同**（上表 5752 个读数，0 不一致）——判决的**输入**逐位相同；
> ② **分配判决机制本身逐位相同**（V2' 的多槽探针，sha256 相同）——判决的**规则**逐位相同。
> 输入与规则都逐位相同 ⇒ 判决必然逐位相同。
> ⚠️ 这是**由两条读数推出**的，不是端到端跑出来的；我不把它说成端到端实测。

#### V2' —— flexgrid 侧判决与 Q 证据链（**通过**）

| 项 | 读数 |
|---|---|
| 多槽分配判决 vs `cb24704` | `steps_equal=True` / `final_equal=True` / **`sha256_equal=True`**（`895d9d04d5f4e6efee3d75209f1b0da470458d33fcad69434887e159e1680345`） |
| Q-J1..Q-J6 | **12 passed**（`tests/test_bwqot_bandwidth.py -k "qj1 or qj2 or qj3 or qj5 or qj6"`） |
| `b_hz_channels[i]` 字面量 | **保留在 `numba_kernels.py`**：`459`、`492`、`552`（docstring）、`578` 四处 |
| `qj5_per_channel_bandwidth_mapped_to_p1_23_29_30_34` | **绿** |
| `qj5_formula_map_check_reproduces` | **绿** |
| 公式图谱 | `18 条物理公式全部有可 grep 的论文锚点；8 条工程能力均有 paper_note` |

#### V3' —— P1 核上的逐信道 `B_i`（**两条通过、一条未达成**）

探针：`evidence/merge/v3p_probe.py`（新增）。前置已确认 `B_i` **真的**非均匀：
`occupied` 档下 `B_i = [1.25e10, 2.50e10, 3.75e10, 5.00e10]`（= `n_slots × 12.5 GHz`），
且与 `symbol_rate` 档**不同**（`differs_from_symbol_rate_source=True`）。

| 判据 | 档位 | 逐位? | max_rel | 结论 |
|---|---|---|---:|---|
| `cached ≡ off` | `p1_closed` | **是** | `0.000e+00` | ✅ **通过** |
| `batch ≡ serial` | `p1_closed` | — | — | ❌ **未达成**（见下） |
| `matvec ≡ off` | `p2_gn` | 否 | `2.851e-16` | ✅ 预算内（FG 既有 1e-12，未放宽） |
| `cached_g ≡ matvec` | `p2_gn` | 否 | `2.977e-16` | ✅ 预算内（同上） |

> ❌ **`p1_closed` 的 `batch ≡ serial`：本轮未达成，如实记为未达成。**
> **P1 档暂无批量路径**——批量核（`models/batch_kernels.py`）是 **P2 的实现结构**
> （它批量求值的是 P2 的**张量化** NLI `_compute_nli_variances_t`），而 P1 的 NLI
> 是**闭式**、没有对应的批量形式。批量入口对 `p1_closed` **显式抛 `ValueError`**
> （不是静默退回串行、也不是假装跑通）。
> 第 3 轮范围明确写着「P1 核的加速（matvec/批量）不在本轮，另立一轮」，
> 故本条**记为未达成**，按要求**不假装**。
>
> ⚠️ **第 4 轮已闭合这一条**（见 §9.4 V2''）：P1 有了**自己的**批量实现
> （`models/numba_kernels_p1_accel.py`），`batch ≡ serial` 实测
> `max_rel 4.9e-16 ~ 1.5e-15`（预算 1e-12），批宽 8/16/32/68/142 全绿。
> 上面这段保留为**第 3 轮当时的状态**；**当前状态以 §9 为准**。

#### V4' —— 套件（**零失败**，除基线既有的 2 个环境问题）

```bash
cd <MERGE_WT>
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python -m pytest -q -p no:randomly
```

| | 读数 |
|---|---|
| 合并树（**第 3 轮**） | **`2 failed / 587 passed / 5 skipped`，37.86 s** |
| 合并树（第 2 轮基线） | `11 failed / 578 passed / 2 skipped`，38.40 s |
| `flexgrid-dev @ cb24704`（未改动的基线） | `2 failed / 66 passed / 1 skipped` |
| `wp/fg-integrate @ 57c7221`（未改动的基线） | `477 passed / 2 skipped` |

**11 个失败里的 9 个全部消失**（逐条对照）：

| 第 2 轮的失败 | 第 3 轮 | 怎么修的 |
|---|---|---|
| `bwqot[probe.py]` | ✅ 消失 | 导入的 `_calc_raman_profile_jit` 回到 P1 那份核里 |
| `bwqot[cost.py]` | ✅ 消失 | Q-J3「内核不得含 ODE 标识符」——P1 内核确实没有 |
| `qj5_per_channel_bandwidth_mapped_to_p1_23_29_30_34` | ✅ 消失 | `b_hz_channels[i]` 字面量回到 `numba_kernels.py` |
| `qj5_formula_map_check_reproduces` | ✅ 消失 | 公式图谱恢复 P1 锚点 |
| `ode3[guard8]` | ✅ 消失 | 锚点全部重新对齐（`n_unfixable: 0`） |
| `ode3[guard9[relocate_anchors]]` | ✅ 消失 | 重定位记录重生成（`n_stale: 0` / `n_unfixable: 0`） |
| `ode3[guard9[flexgrid_e2e]]` | ✅ 消失 | 同上 |
| `ode3[guard9b]`、`ode_dev[ode_formula_map]` | ✅ 消失 | 公式图谱与工具一致 |
| `ode_dev[ode_compare.py]` | ❌ **保留** | `ModuleNotFoundError: fu4_scenarios_legacy`——**基线既有** |
| `bwqot[legacy_regression.py]` | ❌ **保留** | 同上——**基线既有** |

**基线既有的 2 个**与 `flexgrid-dev @ cb24704`（**未改动**）**同名同因**
（`ModuleNotFoundError: No module named 'fu4_scenarios_legacy'`），单独标出。

> ⚠️ 顺带说明：第 3 轮过程中**新出现过**大量失败——全部是「默认档从 P2 变成 P1」的
> **必然连带**（FG 的既有期望值默认跑的是 FG 那套物理）。实测的收敛过程：
> **365 → 25 → 13 → 10 → 2**（中途一次回弹到 9 是我加的 RWA 层改动引起的，
> 改完即回 2）。它们**不是**靠改期望值变绿的，而是靠**把 FG 侧那些构造点显式钉到
> `p2_gn`**（§8.5 的第 23 条）——钉住的值与换底前**完全相同**，故**没有任何期望值
> 发生移动**；`tests/fg4b_tier_pin_policy.py` 的 token 计数守卫（行级、按文件）
> 继续按原样守门。

#### V5' —— 已声明的改动（逐条旧 → 新）

第 2 轮的 11 条声明**继续有效**（§6 V5 表）。第 3 轮**新增/改写**的声明如下
（**只**列第 3 轮动过的；第 2 轮那 11 条里被第 3 轮**部分推翻**的，在末尾单列）：

| # | 位置 | 旧文本 / 旧语义 | 新文本 / 新语义 | 理由 |
|---:|---|---|---|---|
| 1 | `src/.../models/numba_kernels.py` | sha256 `7beece4e…`（FG 的 P2 字节） | sha256 **`6611c416…`**（flexgrid-dev 的 P1 闭式字节） | 第 3 轮改判：两套物理都留，P1 那份住回它自己的文件名 |
| 2 | `src/.../models/numba_kernels_p2.py` | **不存在** | **新增**：逐字节 = FG 的 `7beece4e…`（**同字节、只换文件名**） | FG 的核搬家 |
| 3 | `models/isrs_gn.py` 导入表 | 两侧符号并集，都从 `numba_kernels` 导入 | P1 符号取自 `numba_kernels`、P2 符号取自 `numba_kernels_p2` | 两套物理**不共用**任何一个物理函数 |
| 4 | `models/isrs_gn.py` 构造签名 | `ode_accel: str = ODE_ACCEL_DEFAULT`、`raman_spectrum: str = RAMAN_SPECTRUM_DEFAULT`、**无** `nli_model` | `ode_accel: str \| None = None`、`raman_spectrum: str \| None = None`、**新增** `nli_model: str = NLI_MODEL_DEFAULT` | 档位默认值必须**按 NLI 档位解析**：P1 档没有 matvec、没有 tabular 谱。常量 `ODE_ACCEL_DEFAULT`（`"matvec"`）与 `RAMAN_SPECTRUM_DEFAULT`（`"tabular"`）**一个字没动**（它们仍是 `p2_gn` 档的生产默认） |
| 5 | `models/isrs_gn.py` 档位常量 | 无 | 新增 `NLI_MODELS` / `NLI_MODEL_DEFAULT` / `ODE_ACCEL_MODES_BY_NLI_MODEL` / `ODE_ACCEL_DEFAULT_BY_NLI_MODEL` / `RAMAN_SPECTRUM_MODES_BY_NLI_MODEL` / `RAMAN_SPECTRUM_DEFAULT_BY_NLI_MODEL` | 显式档位 |
| 6 | `models/isrs_gn.py::_calc_span_noise_and_power` | 单一实现（FG 口径） | **派发口** + `_calc_span_noise_and_power_p1` + `_calc_span_noise_and_power_p2` | 两套物理 |
| 7 | `models/isrs_gn.py` P2 路径的 5 处带宽实参 | `channel_bandwidth_hz_channels`（内核调用 / 张量构建 / 批量常量 / 2 处 booster ASE） | `symbol_rate_hz_channels` | 第 3 轮判废「把 `B_i` 改名成 `symbol_rate_hz` 喂 P2」。P1 路径的 `b_hz_channels` / P1 (34) 的 ASE **不动**（`B_i` 属于 P1） |
| 8 | `models/isrs_gn.py` 衰减口径 | 恒用 alpha0/1/2 波长多项式 | **P1 档读** `use_attenuation_polynomial`、**P2 档**恒用多项式 | P1 档要逐位等于 cb24704（alpha 不同 ⇒ `p_out_fiber` 不同 ⇒ P1 (34) 的 ASE 不同） |
| 9 | `models/isrs_gn.py` NF 默认值 | `self.nf_lin_channels` | **P1 档** `np.ones_like(...)`、**P2 档** `self.nf_lin_channels` | 同上 |
| 10 | `models/isrs_gn.py` ROADM 装配 | 单一 `fold_roadm_terms` | **P1 档** `_apply_roadm_p1`（flexgrid-dev 口径）、**P2 档** `fold_roadm_terms` | `x/loss*loss ≠ x`（IEEE-754），两套口径差最后几位 |
| 11 | `models/isrs_gn.py` 阻断节点解析 | 单一 `_resolve_blocked_node_indices` | **P1 档** `_blocked_nodes_p1`（legacy 模式只看 `blocked_channels`） | 同上 |
| 12 | `models/isrs_gn.py::for_fiber` | 透传 10 项 | **追加**透传 `nli_model` | 子评估器不能悄悄换档 |
| 13 | `models/isrs_gn.py::calc_span_noise_and_power_batch` | 无档位检查 | `nli_model != "p2_gn"` **显式抛 `ValueError`** | P1 档**暂无批量路径**，不静默退回串行 |
| 14 | `rwa/qot_checker.py::_node_frequency_and_bandwidth` | 读 `channel_bandwidth_hz_channels` | 读 **`ase_bandwidth_hz_channels`**（物理仓按档位解析好的） | 两套物理的 ASE 带宽量本来就不同。**层内仍不出现任何档位字样**——FG-4g 的「装配层对档位无知」守卫原样通过 |
| 15 | `models/batch_kernels.py` | `from ...numba_kernels import H_PLANCK, _NLI_SPAN_EXPONENT` | `from ...numba_kernels_p2 import ...` | 批量核是 P2 的实现结构 |
| 16 | `tests/test_fg14_assembly_provenance.py` | 字节冻结守 `numba_kernels.py`，断言 sha256 `7beece4e…` | 改守 **`numba_kernels_p2.py`**，**断言值一字未改** | **改指向，不是放宽**：被守的仍是 FG 那份字节 |
| 17 | `tests/test_ode3_flexgrid_sync.py` guard 2（J5） | sha256 **必须等于** `7beece4e…`（相对断言已不成立） | sha256 **必须等于** `6611c416…`（**绝对**不变量）**且恢复** `git diff b69563a -- numba_kernels.py` **必须为空**——**两条同时成立** | 第 3 轮恢复 P1 字节 ⇒ P1 那份字节**就等于** `b69563a` ⇒ 相对断言**重新成立**。相对 + 绝对两条同时成立，比任一条单独存在都强 |
| 18 | `tests/test_ode3_flexgrid_sync.py` | 无 | **新增** `test_guard2b_j5_frozen_p2_kernel_is_byte_frozen` | FG 那份 P2 内核**另立一条**守卫，没有被放掉 |
| 19 | `results/ode_dev_flexgrid_sync.json` | `j5_frozen_kernel` 指向 FG 字节 | `j5_frozen_kernel` 指向 **P1** 字节（`6611c416…` / `flexgrid-dev`）+ **新增** `j5_frozen_kernel_p2` | 同上 |
| 20 | `tools/ode3_sync.py` | `FROZEN_SHA256 = 7beece4e…`、相对断言不成立 | `FROZEN_SHA256 = 6611c416…`、相对断言**恢复成立**；**另加** `FROZEN_P2` / `FROZEN_P2_SHA256` / `FROZEN_P2_PROVENANCE_BRANCH`；片段表**新增 1 段**（共 38 段） | 同上；工具是事实源，JSON 由它派生 |
| 21 | `tools/ode_formula_map.py` + `results/ode_dev_formula_map.json` | P1 锚点被**改钉到 P2 代码**上（`b_hz_channels[i]` → `symbol_rate_hz[i]`、`asinh` → `(16/27)` 张量化式） | P1 锚点**恢复**到 `numba_kernels.py`（P1 闭式，`b_hz_channels[i]` 4 处），**另立** `K2`（= `numba_kernels_p2.py`）锚点记 P2 的张量化实现 | 两套物理**各自在册**，谁都没被抹掉。`results/ode_dev_anchor_relocation.json` 由 `n_unfixable: 32` 回到 **`0`** |
| 22 | `tests/fg4b_tier_pin_policy.py` | 只授权 `ode_accel="off"` 与 `raman_spectrum="legacy"` 两种钉子 | **新增** `NLI_MODEL_AUTHORIZED_FILES` + `NLI_MODEL_PIN_LITERAL = 'nli_model="p2_gn"'` | 第 3 轮的档位钉子，**先例同 FG-15**：钉住的值与换底前**完全相同**，故没有期望值移动；token 计数守卫继续按行级守门 |
| 23 | FG 侧测试 / 脚本的构造点 | 不传档位 | **显式** `nli_model="p2_gn"`（`tests/fg1_scenarios.py`、`tests/fg4a_scenarios.py`、`tests/fu4_scenarios.py`、9 个 `tests/test_fg*.py`、`scripts/fg13_probe_roadm.py`、`scripts/fg4g_q2_trace_diff.py`） | 默认档已是 P1；这些文件量的是 FG 那套物理，**必须自己声明** |
| 24 | `tests/test_fg15_legacy_bitwise.py`、`tests/test_fg_u1_raman_spectrum.py` | `MultiBandISRSGN.__init__.__defaults__[-1] == "tabular"` | `__defaults__[-1] == NLI_MODEL_DEFAULT`（= `"p1_closed"`）+ 断言**按档位解析**的拉曼默认（`p2_gn`→`tabular`、`p1_closed`→`legacy`） | 构造签名的**末位**默认参数已换成 `nli_model` ⇒ `[-1]` 不再是拉曼档位。**常量本身一字未动** |
| 25 | `tests/test_ode_dev.py` 断言 (d) | `0 < auto[0] < power`（FG 口径） | **拆成两段**：P1 档 `allclose(auto[0], power)`（**恢复 flexgrid-dev 原断言**）+ P2 档 `0 < auto[0] < power` | 两套物理各自的契约；第 2 轮二选一的那一刀，第 3 轮拆回两段 |
| 26 | `tests/test_ode_dev.py` guard 4 | 遍历 `ODE_ACCEL_MODES` 并集 | 遍历「**每个 NLI 档位 × 该档位支持的每个档位**」的**笛卡尔积**（6 组）+ **新增** `test_guard4b_incompatible_ode_accel_is_rejected` | 覆盖**扩大**（原来 4 组，现在 6 组）；新增的守卫要求不匹配组合**显式报错** |
| 27 | `tests/test_merge_fg_into_flex.py::TestPerChannelBandwidth` | 全部隐式跑在 P2 档 | P2 档的用例**显式**钉 `p2_gn`；**新增** `test_b_i_is_non_uniform_in_occupied_mode_on_both_nli_tiers`；`test_wide_connection_is_not_physics_identical_to_narrow` **拆成两段**（P1 档必须不同 + P2 档必须相同） | 逐信道 `B_i` 的语义**按档位分属**：P1 档 `B_i` 进 QoT，P2 档**看不见**占用宽度 |
| 28 | `tests/test_fu2_sclu_bands.py` | 不传档位 | 显式 `nli_model="p2_gn"` + **新增** `for_fiber` 透传档位的断言 | 拉曼谱两档（legacy/tabular）都是 **P2** 的能力（P1 核没有 G 矩阵） |
| 29 | `evidence/merge/v1_rollout_probe.py` | 不钉档位 | **新增** `--nli-model` | RL 仓是**只读**的、它的 `build_env` 用默认参数造 evaluator ⇒ 默认档会被静默选中 |
| 30 | `docs/ode3_flexgrid_sync.md`、`docs/ode2_papers_and_diff.md` | 旧渲染 | 重新生成 | 派生文档 |

**第 2 轮那 11 条里，被第 3 轮「按档位分列」的 3 条**：#9（`test_ode_dev.py` 断言 (d)）、
#10（NF 默认值）、#11（`use_attenuation_polynomial`）——第 2 轮取的都是 **P2 侧**的口径
（因为当时只有 P2 一套物理）；第 3 轮**在 `p2_gn` 档原样保留**它们，在 `p1_closed` 档
**取回 flexgrid-dev 的口径**。这三条**不是**被推翻，是被**分列**。

### 8.5 第 3 轮我没做什么 / 没跑完什么

如实列出，**不粉饰**：

1. **P1 核的加速没做**（matvec / 批量）。这是编排方划到「另立一轮」的范围。
   ⇒ **V3' 的 `p1_closed` 的 `batch ≡ serial` 一条未达成**，如实记为未达成（§8.4 V3'）。
   批量入口对 `p1_closed` 显式抛 `ValueError`，不静默退回串行。
2. **两套物理没有「统一」**。`p1_closed` 与 `p2_gn` 的读数**不可互相比较**——
   它们是两套物理，数字本来就该不同。我没有做任何「让两套数字看起来一样」的事。
3. **没有删任何测试、没有加 `skip` / `xfail`、没有放宽任何阈值、没有伪造公式号。**
   阈值 `MAX_REL_DEVIATION = 1e-12` 与 `MAX_REL_DEVIATION_ALLOWED` 原样沿用。
   `results/ode_dev_formula_map.json` 的 18 条物理公式号**一条没动**。
4. **下游 RL 仓 `wp/fg-rt` 需要一行改动才能继续跑 FG 那套物理，本轮没改那个仓。**
   它的 `env.py` 构造 evaluator 时显式传 `raman_spectrum=config.raman_spectrum`
   （= `RAMAN_SPECTRUM_DEFAULT` = `"tabular"`），而 **P1 档没有 tabular 谱**
   ⇒ 构造期抛 `ValueError`。确切改法：在 `multiband_optical_rl/env.py:154` 的
   构造里加 `nli_model="p2_gn"`（该仓在 `wp/fg-rt` 分支上，**不在本轮范围**，
   且我按纪律只读它）。**受本仓影响的证据路径已经钉好**：
   `scripts/fg4g_q2_trace_diff.py` 已显式钉 `nli_model="p2_gn"`（它量的是 FG 那套物理），
   `evidence/merge/v1_rollout_probe.py` 新增了 `--nli-model`。
   ⚠️ 这是本轮**已知、未闭合**的一条，我把它摆在这里而不是藏起来。
5. **`flexgrid-dev` / `wp/fg-integrate` 两棵树全程只读**；没有 push；
   `wp/fg-rt` 只被读取（探针靠 `PYTHONPATH` 切物理树），未改动。
6. **性能 / 规模维度仍未覆盖**：FG 的基准（`results/fg4e_speedup_ab.json` 等）没有重测，
   本轮只验证**等价性**与**正确性**（这与第 2 轮一致）。
7. **「分配判决」在 `p1_closed` 档是两条读数推出**的（物理层逐位 + 分配机制逐位），
   **不是**端到端 RL 实测——因为 RL 仓在 flexgrid 树上连 import 都过不去（§8.4 V1'）。
   我不把它说成端到端读数。
8. **P1 档的 ROADM 装配有第二处实现**（`_apply_roadm_p1`）。FG-11 的「折叠逻辑只
   允许有一处」是**P2 那套物理**的性质；P1 的装配是**另一套物理**的，按第 3 轮裁决
   「两套物理都留」而必须独立存在。`tests/test_fg13_evaluate_path_roadm.py` 那条
   结构守卫扫的是 `evaluate_path` 的**函数体**——P1 的装配被放进**另一个方法**，
   那条守卫**一个字都没改**、照样通过。
9. **`p1_closed` 与 `p2_gn` 的拉曼谱档位不共用**：P1 核的拉曼模型是**固定的**
   （legacy 折线），没有 tabular 这条路。所以「P1 档 + `raman_spectrum="tabular"`」
   是**显式拒绝**而不是静默降级——与 FG-15 的「显式拒绝而非静默回退」同一口径。

---

# 9. 第 4 轮：把加速结构移植到 **P1 闭式核**上

本轮只做一件事：**在 P1 闭式核上实现加速，不碰 P2**。
`p2_gn` 档的行为一字未动（它的 6364 项 0 不一致照旧，§9.4 V1''）。

## 9.1 为什么 P1 的 XPM 不能整块收成一次 matvec

FG 侧 P2 Eq. 21 的 XPM 在张量化后是 `l` 方向**线性**的
（`gamma_sq[i,l] * u_k[b,l]`），所以能整块收成 GEMM。
**P1 (30) 不是**：

```
term(i,l) = (P_l²/B_l)·exp(2·Cr·k̄·L_eff_l·r_fl) · (gamma_il²/(phi_il·3·alpha_l²))
            · [ c1_l·atan(phi_il·B_i/alpha_l) + c2_l·atan(phi_il·B_i/(2·alpha_l)) ]
```

`atan` 的自变量 `phi_il·B_i/alpha_l` **同时含信道对量 `phi_il` 与逐信道量 `B_i`、
`alpha_l`**，而 `atan` 是非线性函数 ⇒ 这一项**写不成** `f(i)·g(l)`。
任务书 §1.1 明文允许这种「分段」：**矩阵化能矩阵化的部分，剩余部分保持逐对循环**。

本轮的分段是这样切的（细节在 `models/numba_kernels_p1_accel.py` 的模块 docstring）：

```
term(i,l) = G1[i,l]·w1[l] + G2[i,l]·w2[l]

G1[i,l] = (gamma_il²/(phi_il·3·alpha_l²))·atan(phi_il·B_i/alpha_l)      ← evaluator 级 (N,N)
G2[i,l] = (gamma_il²/(phi_il·3·alpha_l²))·atan(phi_il·B_i/(2·alpha_l))  ← evaluator 级 (N,N)
w1[l]   = (P_l²/B_l)·exp(2·Cr·k̄·L_eff_l·r_fl)·c1_l                     ← 逐候选 (N,)
w2[l]   = (P_l²/B_l)·exp(2·Cr·k̄·L_eff_l·r_fl)·c2_l                     ← 逐候选 (N,)
```

⇒ `l` 方向的求和降成**两次矩阵-向量乘**；`G1`/`G2` 只依赖信道几何
（波长、频率、带宽、衰减、色散），是 **evaluator 级**常量，**构建一次、跨跨段与跨候选复用**。

**唯一没有矩阵化的是原实现逐对做的 `max(·, 0)` 截断**。它不是「忘了做」，
而是**在数学上挡路**，本轮用「符号论证 + 少量逐对回退」处理（§9.2）。

## 9.2 逐对 `max(·, 0)` 截断的处理：符号论证，而不是「拍脑袋安全」

把求和项按符号摊开（`A_il = |gamma_il²/(phi_il·3·alpha_l²)| >= 0`，
`A1_il = |atan(phi_il·B_i/alpha_l)|`，`A2_il = |atan(phi_il·B_i/(2·alpha_l))|`）：

```
result(i,l) = v_l · A_il · Q_il ,   Q_il = c1_l·A1_il + c2_l·A2_il
```

`sign(result) = sign(Q_il)`，所以截断**当且仅当** `Q_il < 0` 时才真的生效。
又 `A1_il >= A2_il`（`atan` 单调）且 `A1_il <= 2·A2_il`（`atan` 在 `[0,inf)` 上凹且
`atan(0)=0` ⇒ `atan(2y) <= 2·atan(y)`），于是得到**逐通道 O(1) 的充分判据**：

| 条件 | 判据 | 理由 |
|---|---|---|
| `c1 >= 0` 且 `c2 >= 0` | 恒安全 | 两项都非负 |
| `c1 >= 0` 且 `c2 < 0` | `c1 + c2 >= 0` | `G1·c1 + G2·c2 >= G1·(c1+c2)`（`G2 <= G1`） |
| `c1 < 0` 且 `c2 >= 0` | `2·c1 + c2 >= 0` | `G1·c1 + G2·c2 >= G2·(2·c1+c2)`（`G1 <= 2·G2`） |
| 两者都负 | **不安全** | `Q < 0`，截断一定生效 |

> ⚠️ **这一版判据是第一版的修正**。第一版写成「`c1 >= 0` **且** `c2 >= 0` 才安全」，
> 实测发现**全部 68 个通道都落进不安全分支**（`c2_l < 0` 出现在 `t_l > 4·alpha_l²`
> 那一侧，即 `f_rel < 0` 的半边），于是矩阵化**一次都没生效**、全部退回逐对实现
> （结果反而逐位相同，但**没有加速**）。改成上表的「bracket 符号」判据后，
> 68/142 信道、混合与均匀两档**全部通道都安全**，矩阵化才真正走通。
> 这条修正记在这里，因为它正是「逐位相同但没有加速」这种假成功的来源。

判据不满足的通道**不多不少**地走**原实现逐对算**：直接调用冻结核里的
`_calc_xpm_variance_term_jit`（**逐位相同**，包括 `max(·,0)` 本身），加回 GEMV 的结果。
**这不是「静默回落」**：它只影响个别通道**对**的求和项，且走的是**同一个冻结核函数**。

**实测**：68 与 142 信道、混合与均匀两档、批宽 8/16/32/68/142，
**安全通道数 = 全部**（回退分支**一次都没触发**）。回退分支仍然必须存在——
否则那是个「在别的参数下悄悄算错」的实现。

## 9.3 出纤功率的公共子表达式消除（恒等变换，不是改算式）

XPM 之外，P1 的**出纤功率**也是 O(N²)，且是加速后的**最大剩余成本**：

```
denom_i = Σ_j p_in[j]·exp( (-(1-k_bar)·_SRS_CR·L_eff[i]) · r_f[j] )
```

`u_i = (-(1-k_bar)·_SRS_CR)·L_eff[i]` 只依赖**逐信道衰减**。在 `p1_closed` 的
**默认**衰减口径（常数衰减）下全部 `alpha_i` 相等 ⇒ 全部 `u_i` **逐位相等**
⇒ 那 N 条指数向量**是同一条**。

本轮据此把指数向量的条数从 N 降到「不同 `u` 的个数」（默认口径下 = 1），
指数计算量 **O(N²) → O(N)**。**这是恒等变换，不是改算式**：`u_i` 逐位相等
⇒ `exp(u_i·r_f[j])` 的输入逐位相同 ⇒ 输出逐位相同。
实测**出纤功率与冻结核逐位相同**（§9.4 V2''）。

`u` 互不相等的场合（`use_attenuation_polynomial=True` 的波长多项式衰减）
自动退回逐 `i` 计算，代价与冻结核相同——**不更差**。

## 9.4 判据实测读数（V1''–V5''）

探针：`evidence/merge/v4p_probe.py`（新增）。

```bash
cd /home/zsig/Documents/gits/wt/merge-fg-into-flex
PYTHONPATH=$PWD/src:$PWD/tests ~/miniconda3/envs/rl/bin/python \
  evidence/merge/v4p_probe.py --out evidence/merge/v4p_probe.json
```

### V1'' —— 公式不动（**通过**）

| 项 | 读数 |
|---|---|
| 冻结核 `models/numba_kernels.py` 的 sha256 | `6611c4164a60d0dcc653e47c5f02d81534e9c78b03e71213aa30d8b13bfd4aa4` |
| 与 `flexgrid-dev @ cb24704` 的字节是否相同 | **是**（`diff` 空） |
| `b_hz_channels[i]` 出现次数 | **4** |
| `b_hz_channels[l_idx]` 出现次数 | **1** |
| P1 的 `asinh` / `atan` 参数形式 | **逐字仍在**（4 条字面量，见 `tests/test_p1_closed_accel.py`） |
| `p1_closed` 的 **off 档** vs `cb24704` | **5752 项 / 0 不一致**（sha256 相同） |
| `p2_gn` vs `wp/fg-integrate @ 57c7221` | **6364 项 / 0 不一致**（sha256 相同，**本轮未动**） |

> ⚠️ 5752 项那一对**现在显式钉在 `ode_accel="off"`**（§9.5 默认档裁决）。
> 探针 `evidence/merge/v1p_p1_probe.py` 在两棵树上都传 `ode_accel="off"`
> （`flexgrid-dev` 的 `ODE_ACCEL_MODES` 含 `"off"`、默认也是 `"off"`）⇒ **对等对照**，
> 没有任何一侧被静默换档。任务书 V1'' 的原文也正是「`p1_closed` 的 **off 档**」。

### V2'' —— 加速正确性（**通过**；预算 1e-12 **未放宽**）

两种构造：`mixed_n_slots`（`B_i` 真非均匀 = `[1.25e10, 2.50e10, 3.75e10, 5.00e10]`）
与 `uniform`（`B_i` 恒为 `1.25e10`）。N = 68。

**`matvec ≡ off`**（单候选）：

| 构造 | 逐位? | max_rel | max ULP | 结论 |
|---|---|---:|---:|---|
| mixed_n_slots | **否** | `1.269e-15` | 11 | ✅ 预算内（余量 ~790x） |
| uniform | **否** | `1.288e-15` | 7 | ✅ 预算内 |

**`batch ≡ serial`**（逐批宽 max_rel）：

| 批宽 B | mixed_n_slots | uniform | max ULP（mixed） | 结论 |
|---:|---:|---:|---:|---|
| 8 | `1.375e-15` | `1.154e-15` | 12 | ✅ |
| 16 | `6.230e-16` | `4.898e-16` | 4 | ✅ |
| 32 | `1.307e-15` | `1.483e-15` | 11 | ✅ |
| 68 | `7.572e-16` | `6.153e-16` | 6 | ✅ |
| 142 | `7.572e-16` | `7.602e-16` | 6 | ✅ |

**逐项拆开**（mixed，B=68）：`p_out` / `sigma2_spm` / `sigma2_ase` 与 `off` 档
**逐位相同**（max_rel = 0、max ULP = 0）；**只有 `sigma2_xpm_w` 漂移**
（`max_rel 1.269e-15`、max 11 ULP）。漂移**只来自 XPM 的 `l` 方向求和次序**
（成对求和 → GEMM 累加），与 FG 侧 `matvec` 档的成因**同类**
（FG 侧实测 `2.851e-16`）。

### V3'' —— 收益（**全部为正**；带 loadavg 与线程数）

**计时上下文**（没有这两个数的倍率不算证据）：

| 项 | 读数 |
|---|---|
| loadavg（1/5/15 min） | **1.59 / 1.74 / 1.42**（落盘那一轮；四轮实测在 1.5–3.1 之间波动） |
| CPU 核数 / 进程亲和度 | **20 / 20** |
| MKL 线程数 | **12** |
| BLAS 实现 | `mkl-sdl` 2025 |
| numba threading layer | `default` |
| (N,N) 对矩阵的一次性构建 | **0.079 ms**（evaluator 级，跨跨段/跨候选复用） |

N = 68，混合 n_slots。`serial_off` = 逐候选跑 `off` 档；`serial_matvec` = 逐候选跑
`matvec` 档；`batch_matvec` = 一次批量调用。

| 批宽 B | serial_off (ms) | serial_matvec (ms) | batch_matvec (ms) | 档位倍率（单候选串行 vs matvec） | 批量倍率（批量 vs 串行） | 端到端（批量 vs off 串行） | 冷启动首次调用 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.687 | 0.247 | 0.060 | **2.78x** | **4.10x** | **11.37x** | 0.136 |
| 16 | 1.372 | 0.491 | 0.134 | **2.79x** | **3.66x** | **10.22x** | 0.111 |
| 32 | 2.829 | 1.007 | 0.182 | **2.81x** | **5.53x** | **15.52x** | 0.132 |
| 68 | 5.836 | 2.079 | 0.331 | **2.81x** | **6.28x** | **17.62x** | 0.128 |
| 142 | 12.418 | 4.408 | 0.639 | **2.82x** | **6.90x** | **19.44x** | 0.129 |

> ⚠️ **计时读数是这台共享机器的一次快照，重跑必有抖动。**
> `evidence/merge/v4p_probe.json` 里落盘的就是**其中一轮**（取 7 次的**最小值**）；
> 本表抄的就是那一轮。**四轮实测**（loadavg 在 1.5–3.1 之间波动）的**区间**是：
>
> | 量 | 四轮区间 |
> |---|---|
> | 档位倍率（单候选串行 vs matvec） | **2.71 – 2.94x** |
> | 批量倍率（批量 vs 串行） | **3.66 – 6.91x** |
> | 端到端（批量 vs off 串行） | **10.22 – 19.86x** |
> | (N,N) 对矩阵一次性构建 | **0.077 – 0.081 ms** |
> | 冷启动首次调用 | **0.111 – 0.145 ms** |
>
> ⇒ **区间**才是结论。我把四轮读数都摆出来，**不挑好看的那个**。

> ⚠️ **收益为负的那一点，照实报**：**冷启动的第一次调用比 `off` 慢**——
> 0.111–0.145 ms vs `off` 的单候选 0.086 ms（= 0.687/8），即 **~1.3–1.7x 慢**。
> 原因是要多付一次 (N,N) 对矩阵的构建（0.079 ms）。
> **第二次调用起就反超**（构建摊销），第 3 个候选起已是净收益。
> ⇒ 对**只算一个候选**的调用方，加速档**不划算**；对跨段、跨候选的调用方才划算。

### V4'' —— 套件（**2 failed / 612 passed / 2 skipped**）

```
2 failed, 612 passed, 2 skipped in 38.21s
```

那 2 个 failed 仍是**基线既有的环境问题**（`ModuleNotFoundError: No module named
'fu4_scenarios_legacy'`），与本轮无关。**skip = 2**（未超过 2）。
passed 从基线的 590 涨到 612（本轮新增 21 条守卫 + 1 条改写的守卫）。

### V5'' —— 声明（逐条旧 → 新）

**新增函数**（全部在 `models/numba_kernels_p1_accel.py`，新文件）：

| 新函数 | 作用 |
|---|---|
| `build_p1_xpm_pair_matrices` | 构建 P1 XPM 的 (N,N) 对矩阵 `G1`/`G2` + `phi_raw`/`gamma`（evaluator 级常量） |
| `calc_span_noise_and_power_p1_matvec` | P1 闭式核的 (B,N) 加速形态（XPM 降成两次 matvec） |
| `_p1_span_pre_nli_batch_jit` | NLI 之前的跨段物理（拉曼剖面、出纤功率 + CSE、t 因子） |
| `_p1_xpm_weights_jit` | 逐候选权重 `w1`/`w2` 与**逐对安全掩码** |
| `_p1_spm_jit` | SPM 批量形态（调用**同一个**冻结核函数 ⇒ 逐位相同） |
| `_p1_xpm_clamp_correction_jit` | 少数不安全通道对的**原实现逐对回退**（逐位相同） |
| `_p1_xpm_gemv` | `w1 @ G1ᵀ + w2 @ G2ᵀ` |
| `P1XpmMatrices` | (N,N) 常量包（`g1`/`g2`/连续转置/`phi_raw`/`gamma`） |

**新增档位**：

| 档位 | 之前 | 现在 |
|---|---|---|
| `p1_closed` 的 `ode_accel` | `("off", "cached")` | `("off", "cached", "matvec")` |
| `p1_closed` 的批量入口 | **显式抛 `ValueError`** | **可用**（按 `nli_model` 分派到 P1 自己的批量核） |
| `p2_gn` 的 `ode_accel` | `("off","cached","cached_g","matvec")` | **不变** |

**默认档选择依据**（本轮唯一的裁决）：

| | 读数 |
|---|---|
| **取** | `p1_closed` 的默认 `ode_accel` = **`"matvec"`**（原为 `"off"`） |
| **延迟** | 单候选串行 **2.7–2.9x**；批量 **3.7–6.9x**；端到端 **10.2–19.9x**（四轮实测的**区间**，见 §9.4 V3''） |
| **精度** | 只有 `sigma2_xpm_w` 漂移：`max_rel 1.269e-15`（~790x 余量，预算 1e-12 **未放宽**）；`p_out`/`spm`/`ase` **逐位相同** |
| **代价（如实记）** | 默认档的读数**不再与 `cb24704` 逐位相同**；逐位相同的是**显式指定**的 `"off"` 档 |
| **先例** | FG-4a 已把生产默认档定成加速档 `"matvec"`（同一 1e-12 预算），并把逐位对照的测试**钉在** `"off"`；FG-15 对拉曼档同一手法 |
| **为什么第 3 轮是 `"off"`** | 第 3 轮自己的注释写着「本轮**不做** P1 核的加速（另立一轮）」——那个**原因**在本轮已经消失 |

**被改动的守卫**（逐条旧 → 新）：

| # | 位置 | 旧 | 新 | 为什么 |
|---:|---|---|---|---|
| 1 | `tools/ode_formula_map.py` | 19 条 `isrs_gn.py` 代码锚点的行号 | 按 `contains` 字面量**逐条重新定位**（`tools/ode3_relocate_anchors.py --write` 复算「0 条待修」） | 本轮在 `isrs_gn.py` 插入代码 ⇒ 行号下移。**`contains` 字符串一个字没动**（物理代码未改） |
| 2 | `tools/ode3_capability.py` | 「ode_accel: matvec / cached_g 加速档」= `fu4_only`（物理缺口） | 改判为 **`both`**，证据改写 | 合并后本仓已有这些档，且**冻结核未被改动**；加速全在另立的新模块里 |
| 3 | `tools/ode3_capability.py` | — | **新增**一条「P1 闭式核的 matvec 档 + 批量入口」= `ode_dev_only` | 第 4 轮新增的能力 |
| 4 | `tests/test_ode_dev.py` `guard4b` | `p1_closed` + `("cached_g","matvec")` 都必须抛错 | `cached_g` **仍抛**；`matvec` **现在合法**（配 legacy）；`tabular` **仍抛** | `p1_closed` 有了自己的 matvec 档。**命题没变**（「没有的能力不许静默回落」），变的是「哪些组合算不匹配」 |
| 5 | `tests/test_ode_dev.py` `_raman_tier_for` | 按 `ode_accel` 判 | 额外按 `nli_model` 判（`p1_closed` 只有 `legacy`） | `p1_closed` 的拉曼谱口径与 `ode_accel` 无关 |
| 6 | `tests/test_ode_dev.py` | — | **新增** `guard4b_p1_matvec_tier_is_reachable_and_not_a_silent_off` | 盯住 `matvec` 不许退化成静默的 `off`（用**可观测差异**判，不是「构造成功」） |
| 7 | `evidence/merge/v1p_p1_probe.py` | 跟默认档走 | **显式钉** `ode_accel="off"` | 探针的命题是「冻结参考路径逐位相同」；两棵树都支持 `"off"` ⇒ 对等对照 |
| 8 | `results/ode_dev_formula_map.json` / `ode_dev_anchor_relocation.json` / `ode_dev_capability_matrix.json` / `docs/ode3_flexgrid_sync.md` | 旧行号/旧能力矩阵 | 由工具重新生成 | 派生产物 |

**没被改动的**：冻结核 `models/numba_kernels.py`（sha256 仍是 `6611c416…`）、
`models/numba_kernels_p2.py`、P2 档的**任何**行为、`MAX_REL_DEVIATION = 1e-12`、
18 条物理公式号、任何 `skip`。

## 9.5 第 4 轮我没做什么 / 没跑完什么

1. **`verify_candidates` 在本仓里不存在**。任务书 §1.3 说的「`verify_candidates`
   对 `p1_closed` 可用」，本仓全文 grep 不到这个符号（`grep -rn "verify_candidates"`
   无命中）。⇒ 本轮把该判据**落到实际存在的批量入口**
   `MultiBandISRSGN.calc_span_noise_and_power_batch` 上（它就是 FG-4e 建的库内批量入口，
   `batch_kernels.py` 的批量核就是给它用的）。**如果编排方指的是别处的入口，请指出，
   我按那个入口重测**——但批量核本身（`calc_span_noise_and_power_batch`）已经通了。
2. **P1 档的 XPM 有一处**没有**完全**矩阵化：逐对 `max(·, 0)` 截断的**判定**是 O(1) 充分判据
   （§9.2），不是逐对精确判定。判据不满足时走原实现逐对算（逐位相同）。
   实测**回退分支一次都没触发**，但它**不是**「已被证明不可能触发」——只是在本轮
   实测的参数下没触发。
3. **没有重测 FG 的既有基准**（`results/fg4e_speedup_ab.json` 等）。本轮只测了
   **P1 档**的新加速路径；P2 档的读数**一字未动**。
4. **规模维度**：只测到 N = 68（任务书点名的批宽 B = 8/16/32/68/142）。
   N = 142 的信道数**没有**单独测（批宽 142 ≠ 信道数 142）。`B_i` 非均匀性、
   ROADM 路径、跨段路径**有**既有测试覆盖（套件 612 passed），但**没有**为本轮
   单独出一份 A/B 基准表。
5. **下游 RL 仓 `wp/fg-rt`** 那条第 3 轮记下的未闭合项（构造 evaluator 时显式传
   `raman_spectrum="tabular"`，而 P1 档没有 tabular 谱 ⇒ 构造期抛 `ValueError`）
   **本轮仍未闭合**。⚠️ 但**本轮把默认档改成了 `matvec`**，`p1_closed` 的默认拉曼谱
   仍是 `"legacy"`，所以那条 RL 路径**仍然**需要显式 `nli_model="p2_gn"`。
   该仓**不在本轮范围**，我只读它。
6. **没有删任何测试、没有加 `skip` / `xfail`、没有放宽任何阈值、没有伪造公式号。**
   `MAX_REL_DEVIATION = 1e-12` 原样沿用；`results/ode_dev_formula_map.json` 的
   **18 条物理公式号一条没动**（只重排了 `isrs_gn.py` 那几条**代码锚点的行号**，
   `contains` 字面量一字未改）。
7. **`flexgrid-dev` / `wp/fg-integrate` 两棵树全程只读**；没有 push；
   `wp/fg-rt` 只被读取。合并树没有 detached HEAD。
