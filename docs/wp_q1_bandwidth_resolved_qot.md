# WP-Q1 报告：把闭式 QoT 改成逐连接带宽分辨率

> 继承 `_PREAMBLE_QOT.md`；判据以 `WP-Q1.md` §4 为准，**跑前写死、跑完未改**。
> 本文件里每个数字都能在 `results/bwqot_*.json` 里找到。

## 0. 一句话

`B_i` 从**全局标量**变成**逐信道数组**，贯通 P1 (23)/(29)/(30)/(34)；
legacy（均匀 50 GHz）**逐位不变**；多槽连接（n_slots 1…8 ⇒ 12.5–100 GHz）
的 SNR 第一次有了真实的带宽依赖。

## 1. 侦察结论：改动前 B 是怎么流的

`B` 的唯一来源是构造参数 `MultiBandISRSGN.symbol_rate_hz`（标量），
经 `_calc_span_noise_and_power_jit` 一路传到三个公式项：

| P1 公式 | 落点（改动前行号） | 当时的 `B` |
|---|---|---|
| **(29) SPM** `P_i³/B_i²`、`asinh(φ_i B_i²/πα_i)` | `numba_kernels.py:321,327,328` | 标量 `rs_hz` |
| **(30) XPM** `P_ℓ²/B_ℓ` | `numba_kernels.py:375` | 标量 `rs_hz`（作 `rs_l_hz`） |
| **(30) XPM** `atan(φ_{i,ℓ} B_i/α_ℓ)` | `numba_kernels.py:377,378` | **同一个** `rs_hz`（作 `rs_i_hz`） |
| **(30) 装配** `(32/27)P_i·N·Σ_ℓ` | `numba_kernels.py:493` | — 已与 P1 一致 |
| **(34) ASE** 核内死代码 | `numba_kernels.py:556` | 标量 `rs_hz`（**返回值被丢弃**） |
| **(34) ASE** 实际生效 | `isrs_gn.py:361` | 标量 `self.symbol_rate_hz` |
| (34) booster ASE（RWA 侧） | `rwa/qot_checker.py:302,395` | 标量 / `symbol_rate_hz_channels`（常量填充） |

两个关键发现（决定了本 WP 的改法）：

1. **`B_ℓ` 与 `B_i` 原本是同一个数**。`_compute_nli_variances` 把 `rs_hz` 同时
   传给 `rs_l_hz` 与 `rs_i_hz`，所以 P1 (30) 里"干扰信道自己的带宽"和"COI 的带宽"
   恒等——这正是宽连接进不了 QoT 的根因。
2. **核内 ASE 是死代码**。`_calculate_ase_noise_variance` 的返回值在
   `isrs_gn.py:325` 被 `_` 丢弃；真正生效的 ASE 是 Python 侧按同一形状 (34) 的算式
   算的。两处都改了 `B_i`，但**没有**借机改核内的 `G_i = p_in/p_out`（那是既有
   差异，属 J4 已裁定冲突项，不在本 WP 范围）。

## 2. 改了什么（每项挂 P1 公式号）

### 2.1 内核 `models/numba_kernels.py`

- (29) SPM：`rs_hz` → `b_i_hz`，用于 `P_i³/B_i²` 与两个 `asinh` 的 `B_i²`。
- (30) XPM：**`rs_l_hz`/`rs_i_hz` 拆开**为 `b_l_hz`（= `P_ℓ²/B_ℓ`）与
  `b_i_hz`（= `atan(φ_{i,ℓ} B_i/α_ℓ)`），调用点分别是 `b_hz_channels[l_idx]`
  与 `b_hz_channels[i]`。
- (34) ASE：`_calculate_ase_noise_variance` 的 `rs_hz` → `b_hz_channels` 数组。
- `_compute_nli_variances` / `_calc_span_noise_and_power_jit` 新增
  `b_hz_channels` 参数；**保留** `symbol_rate_hz` 标量（它仍用于 SRS 频带边界
  `f_m/f_M`，见 §5 未闭合项）。
- 核内加了 `len(b_hz_channels) == num_channels` 断言：漏接参数立刻炸，
  不静默错位。

### 2.2 接线 `models/isrs_gn.py`

- 新属性 `channel_bandwidth_hz_channels`：**P1 四个公式里 `B_i`/`B_ℓ` 的唯一取值源**。
- 解除 `ValueError: per-connection heterogeneous symbol rates are not supported`，
  改为对 `B_i` 的合法域校验（正、有限）。
- 实际生效的 (34) 与 ROADM booster ASE 都改读逐信道 `B_i`。
- `rwa/qot_checker.py` 的 `_node_frequency_and_rate` 更名
  `_node_frequency_and_bandwidth`（它返回的是 (34) 的 `B_i`，不是声明符号率），
  旧名保留为别名。

### 2.3 两个**显式**口径参数（Q-J6）

| 参数 | 取值 | 含义 |
|---|---|---|
| `channel_bandwidth_source` | `symbol_rate`（默认） | `B_i` = 连接声明的 `symbol_rate_hz` |
| | `occupied` | `B_i` = `occupied_bandwidth_hz` = `n_slots` × 槽宽 —— **本 WP 的目标档** |
| `power_width_mode` | `fixed_power`（默认） | `P_i` 是连接自己的总功率，与 `B_i` 无关 |
| | `fixed_psd` | 恒定 PSD：`P_i_eff = P_i × B_i / symbol_rate_hz` |

两档都不是新物理：取值分别来自 `ConnectionSpec` 已有的字段，未改任何公式系数。
**论文没写清 `B_i` 是符号率还是占用带**（P1 只说 "the channel bandwidth"，
P2 只说 "bandwidth of the channel k"），所以本仓把它做成显式参数而不是替读者隐含。

### 2.4 公式映射 `tools/ode_formula_map.py`

新增 P1 **(23)** 条目（§3.5 要求把「`B_i` 逐信道」挂到 (23)/(29)/(30)/(34)），
并为 (29)/(30)/(34) 补上 `b_hz_channels[i]` / `b_hz_channels[l_idx]` 逐信道锚点。
锚点因插行整体重定位：`tools/ode3_relocate_anchors.py --write`（只改行号，不改
`contains`）。共 18 条 physics。

## 3. 判据读数（跑前写死，未放宽）

### Q-J1 legacy 逐位不变 —— **硬闸门，通过**

6 个 legacy 场景 vs 只读基线 `ecz-dev`，`delta_db` **全部恰为 0.0**
（`max_abs_delta_db = 0.0`），1 条拒绝场景两端都拒。

额外做了**构造性**证明（不靠测试反证）：legacy 走 `connections is None` 分支，
`channel_bandwidth_hz_channels` 直接 `np.full(n_nodes, symbol_rate_hz)`。
测试按 `int64` 位模式比对，确认与那个全局标量**逐位相等**
（`equals_scalar_bitwise = true`）。这就是 legacy 数字不动的根据。

### Q-J2 带宽真的进了公式 —— 通过

做法不是"扫一遍看有没有变"，而是照 P1 原文**独立重写**一份标量参考实现
（`tools/bwqot_probe.py::reference_terms`），逐项与内核对照。三项的 `B` 依赖
互不相同，这正是判别力所在：

| 项 | `B` 依赖 | 期望随 B |
|---|---|---|
| (29) SPM | `B^{-2}` 前因子 × `asinh(φB²/πα)` 括号 | **减** |
| (30) XPM | `atan(φ B_i/α)` | **增** |
| (34) ASE | `f_i·B_i` | **增** |

`fixed_power` 口径（`P = 1 mW` 恒定）逐项实测：

| B_i (GHz) | SPM (W) | XPM (W) | ASE (W) | SNR (dB) |
|---|---|---|---|---|
| 12.5 | 3.536822e-07 | 8.783457e-08 | 2.785661e-07 | 32.4271 |
| 25 | 2.933385e-07 | 9.709567e-08 | 5.571502e-07 | 31.2347 |
| 50 | 1.563652e-07 | 1.162075e-07 | 1.114373e-06 | 29.5802 |
| 100 | 6.142301e-08 | 1.944716e-07 | 2.229034e-06 | 27.0474 |

`fixed_psd` 口径（`P ∝ B`）—— 这里出现了教科书级的自洽性检查：ASE 在四个宽度上
**完全相同**（2.785661e-07），因为恒定 PSD 意味着恒定噪声谱密度，而 (34) 的
ASE 本就与 B 线性；与此同时 SPM 随 B **增长**（`P³/B²` 在 `P∝B` 下净剩 `∝B`）。

- `max_rel_err = 0.000e+00`（SPM/XPM/ASE/SNR 四项全对），容差 `1e-9`
- `max_scaling_rel_err = 0.000e+00`（逐项比值与公式比值全对）
- SNR 跨度：`fixed_power` **5.3797 dB**、`fixed_psd` **1.0382 dB**

**容差与推导**：`REL_TOL = 1e-9`。参考实现是 Python 标量、内核是编译后循环，
浮点运算顺序不同，IEEE 双精度末位差异可达 ~1e-16 但不会超过 1e-9；1e-9 足以把
"B 进错位置"这种**结构性**错误（通常差几个数量级）与纯舍入噪声分开。
解析标度的推导直接见 §2.1 各公式位置。

⚠️ 一个诚实的混淆项：灵活栅格下 `n_slots` 变化会**连带平移该连接的中心频率**，
所以"B 单独隔离"的 ASE 标度必须连 `f_i` 一起比（`(34)` 比值 =
`(f_i·B_i)_new/(f_i·B_i)_old`）。这一点已写进 `bwqot_probe.py`，否则会把频移误
当成 B 的作用。

### Q-J3 不引入 ODE —— 通过

- 单次 QoT 验证中位耗时 **0.1558 ms**（预算 10 ms），口径与 WP-M1 的 M-J11 **完全一致**
- 内核 tokenize 扫描：禁用符号命中 **0**（`ode`/`rk4`/`ode_accel`/`phi_kurtosis`）
- 额外补一组异构 `B_i` 的热路径计时：中位 **0.0330 ms**（确认把标量改成数组没掉出预算）

### Q-J4 测试不退步 —— 通过

全量 **68 passed / 1 skipped**，0 failed（基线 46 passed / 1 skipped，本 WP 新增 22 条）。

### Q-J5 公式号可 grep —— 通过

`results/ode_dev_formula_map.json` 含 P1 (23)/(29)/(30)/(34) 四条 physics 条目，
逐信道锚点在源码里可 grep；(30) **分别**锚定 `B_ℓ`（`b_hz_channels[l_idx]`）
与 `B_i`（`b_hz_channels[i]`）。`ode_formula_map.py --check` 可独立复算通过。

### Q-J6 归一化口径显式 —— 通过

`fixed_power` 与 `fixed_psd` 都是**显式构造参数**（`POWER_WIDTH_MODES`），
物理含义写在模块常量 docstring 里；两种口径的读数都在 `results/bwqot_probe.json`，
且给出**不同**的 SNR 曲线。非法取值显式报错，不静默取默认。

## 4. 复算命令

```bash
cd /home/zsig/Documents/gits/wt/bw-qot

# Q-J1（硬闸门，与只读 ecz-dev 对照）
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_legacy_regression.py --write
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_legacy_regression.py --check

# Q-J2 + Q-J6
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_probe.py --write
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_probe.py --check

# Q-J3
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_cost.py --write
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_cost.py --check

# Q-J5
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/ode_formula_map.py --check

# Q-J4
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python -m pytest -q
```

逐行复算 Q-J2：读 `results/bwqot_probe.json` 的 `rows[*].measured` 与
`rows[*].reference_p1`，两者应逐位相等；再读 `term_scaling_vs_p1`，每项的
`{spm,xpm,ase}_measured_ratio` 应等于对应的 `_formula_ratio`。

## 5. 未闭合项 / 明确没做的

1. **SRS 频带边界仍用标量 Rs**（`srs_band_edges_still_scalar_rs`）。
   Python 侧已按连接占用带算出 `self.f_m_hz`/`f_M_hz`，但核内仍用标量 `symbol_rate_hz`
   重算，故那两个值是**死代码**。本 WP **没动**它：`f_m/f_M` 是 SRS 增益谱的截断
   边界，不是 (23)/(29)/(30)/(34) 里 `B_i` 的落点；按「不给论文没写的项加物理」与
   「拿不准就停下报告」，留给后续 WP 裁决。
2. **核内 ASE 的 `G_i = p_in/p_out`** 与 P1 (34) 的 `G(f_i)` 不符。这是 WP-O2 就
   记录过的既有差异（J4 已裁定与 fu4 的 ASE 增益因子冲突）。本 WP 只把它的 `B_i`
   参数化为逐信道量，**没有**改 `G_i`。
3. **`channel_bandwidth_source` 的默认值是 `symbol_rate`**，不是 `occupied`。
   默认档保证 legacy 与既有逐连接场景逐位不变；让 `n_slots` 驱动 QoT 需要**显式**
   传 `channel_bandwidth_source="occupied"`。这是刻意的保守选择，代价是"开箱即用"
   看不到 n_slots 的作用。
4. **`symbol_rate_hz_channels` 语义变了**：它原本恒等于标量填充；现在在逐连接模式下
   返回各连接**声明**的符号率。legacy 下二者相同（逐位不变），但外部若有代码依赖
   "它是个常量"，逐连接模式下会看到不同值。
5. **J5 冻结基线被推进**（`46923a9` → `b69563a`）。冻结核按 J2 被授权改动，基线随之
   推进；J7 相应改成"物理片段应用必须自带 `paper_formulas` 授权"的**例外制**，
   并由测试断言授权的公式号在公式映射中真实存在——闸门是收紧成例外，不是拆掉。
6. **`multiband-optical-rl` 未动**（brief §5 边界）。
7. `ode3_flexgrid_e2e.json` 里 O3 时代的 `known_limitations.uniform_symbol_rate_only`
   已删除（该限制不存在了）；`heterogeneous_rs` 探针由「必须被拒」翻转为
   「必须被接受」，并记录 `b_hz_channels` 真值。

## 6. 提交

| 提交 | 内容 |
|---|---|
| `b69563a` | `feat(bwqot)` B_i/B_ℓ 逐信道化，贯通 P1 (29)/(30)/(34) |
| `0b76012` | `chore(sync)` J5 基线推进 + J7 例外制 + 公式映射补 (23) |
| `0f9e3a6` | `test(bwqot)` 三份证据落盘 + 22 条验收守卫 |

## 7. 结论

Q-J1…Q-J6 全部通过，判据未放宽。`B_i` 现在是逐信道量，P1 (23)/(29)/(30)/(34)
四个公式的带宽依赖都被独立参考实现逐位验证；legacy 逐位不变有构造性保证。
未闭合项见 §5，均为**明确选择**而非遗漏。