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
