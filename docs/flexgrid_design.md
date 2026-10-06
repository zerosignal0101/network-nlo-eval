# 灵活栅格（Flexible Grid）设计与接口冻结

- **状态**：设计冻结候选，待 review。**本文档不含任何实现代码。**
- **本 WP 产物**：仅本文件（`docs/flexgrid_design.md`）。FG-1 在同一仓库改代码，本 WP 未触碰任何代码文件。
- **下游**：FG-2（`network-nlo-eval` 实现）、FG-3（`multiband-optical-rl` 适配）**照本文照抄**。
- **性质**：凡标注 **[需定夺]** 的，是产品取舍，本设计**不替编排方决定**，只给备选与代价。

---

## 1. 证据基线

| 仓库 | 检出 | HEAD | 角色 |
|---|---|---|---|
| `network-nlo-eval` | `/home/zsig/Documents/gits/wt/fg-design`（本 worktree） | `87850a1` | `origin/main`，物理/统计侧 |
| `network-nlo-eval` | `/home/zsig/Documents/gits/network-nlo-eval` | `70f851d` | `ecz-dev`，波形 QoT 侧 |
| `multiband-optical-rl` | `/home/zsig/Documents/gits/multiband-optical-rl` | `3b1b891` | 下游消费者，`stage6-integration` |

两分支 merge-base 为 `8729e63`（已分叉），本设计不假设二者已合并。

### 1.1 与 brief 的一处更正（重要，影响 FG-2 的落点）

brief §1 把四处在"同一个仓库"里描述。实测：**四处焊点分属两个互不重叠的分支版本**，必须分开处理，否则 FG-2 会去改一个不存在的行号。

| brief 所述焊点 | 实际所在 | 证据 |
|---|---|---|
| `qot_checker.py:109` `occupied_width = (num_channels-1)*channel_spacing_hz` | **仅 ecz-dev** | `network-nlo-eval:src/network_nlo_eval/physics/qot_checker.py:109`（main 同行是 `for channel_idx in range(...)`，`main:.../physics/qot_checker.py:89`，**无** `occupied_width`） |
| `qot_checker.py:212` ROADM passband 用 `channel_spacing_hz` | **仅 ecz-dev** | `ecz-dev:.../physics/qot_checker.py:212` `passband_hz=roadm.passband_hz or self.spectrum_grid.channel_spacing_hz` |
| `roadm.blocked_channels`（按索引） | **仅 ecz-dev** | `ecz-dev:.../network/elements.py:83`；main 的 `ROADMConfig` 只有 2 个字段（`main:.../network/elements.py:48-52`，全文件 52 行） |
| `models/numba_kernels.py::_calc_exact_power_profile_ode` | **仅 main** | `main:.../models/numba_kernels.py:174`；ecz-dev 中该名字出现 **0** 次 |

**推论**：`ROADMConfig` 的 `blocked_channels` / `passband_hz` / `filter_order` / `path_loss_db` 是 **ecz-dev 独有**（`ecz-dev:.../network/elements.py:77,78,82,83,94`），而精确功率演化 ODE 是 **main 独有**。而 `multiband-optical-rl` 消费的是 **ecz-dev 风格 API**（`multiband:src/multiband_optical_rl/topology.py:36` 读 `node["roadm"]["blocked_channels"]`，`topology.py:73` 构造 `EDFAConfig(noise_figure_db=nf)` 数组）。

→ **[需定夺-0]** 分支合并顺序会决定灵活栅格是"一次改完"还是"改两遍"。**本设计按"最终形态 = main 的 ODE + ecz-dev 的 ROADM 语义"给出接口**（即假定 `wp/fg-reconcile` 会把两边合流），并在 §4 标注哪些改动在哪个分支上。若合流顺序相反，§4 的"机械改"清单需重排，但**冻结接口（§3）不变**——这是把接口冻结与实现顺序解耦的原因。

---

## 2. 八个设计问题

### 2.1 频谱模型：槽位化 vs 任意中心频率+任意宽度

**结论：槽位化（slot-based）**，标称槽宽 **12.5 GHz**；对外暴露的物理量仍是每连接的 `(中心频率, 宽度)`。

**备选**：

| 方案 | 说明 | 为何不选 |
|---|---|---|
| A. 纯固定栅格 | 现状 `SpectrumGrid`（`main:.../core/spectrum.py:11-54`） | 无法表达变宽连接，题设 |
| B. 任意中心频率 + 任意宽度 | 字节完全自由 | 连续性/邻接性判定退化为浮点相等；资产阻断、碎片指标、RL 动作空间全部失去整数语义；**且无法满足 §2.3 的"逐位不变"验收** |
| C. 6.25 GHz 槽 | 更细 | 同样 50 GHz 需 8 槽，资产体积与占用数组翻倍；12.5 GHz 已对齐 ITU-T G.694.1 ODU 粒度（ODU0=12.5, ODU1=25, ODU2=50, ODU3=75, ODU4=100, ODU4e1=150, ODU4e2=200, ODU4e3=250, ODU4e4=400, ODU4e5=500, ODU4e6=800, ODU4e7=1000, ODU4e8=1600, ODU4e9=2000 GHz），**ODU 阶梯恰好是 12.5 GHz 的整数倍** |
| D. **12.5 GHz 槽（采用）** | — | 50 GHz = 4 槽整除；ODU 阶梯整除；整数语义 |

**槽宽选择理由**：12.5 GHz 使 50 GHz 整除为 4（§2.3 逐位不变的前提），且 12.5 的整数倍集合覆盖 G.694.1 全部标准 ODU 宽度，连接→槽数（§2.2）可做成查表。

**关键不变量（必须在实现中显式断言）**：

1. `slot_width_hz > 0`；`num_slots >= 1`。
2. `band_low_edge_hz > 0`（物理光频率为正，对齐 `spectrum.py:28-30` 的既有保护）。
3. **频段覆盖不变式施加在连接"中心"上，不是槽边缘上。** 理由见下，这是本设计最容易被实现错的一处。

> **实测陷阱（务必读）**：把频段覆盖检查施加在**槽边缘**上，会与"230 道"**在几何上互斥**。
> - C+L 可用边到边宽度 = `C_HIGH - L_LOW` = 195.9428 − 184.4877 = **11.4551 THz**。
> - 230 道 50 GHz 需要的边到边宽度 = `230 × 50 GHz` = **11.5000 THz**。
> - **缺口 44.9 GHz ⟹ 无论怎么锚定，边缘都不可能全部落在带内。**
> - 中心到中心只需 11.4500 THz（余 5.1 GHz）⟹ **中心方案可行，边缘方案不可行**。
> - 且这不是锚点选择问题：若把 `band_low_edge` 抬到 `L_LOW`，230 个中心会**整体 +25.0 GHz**，
>   AC-1 的逐位相等立刻失败；而上边缘仍超 `C_HIGH` 44.9 GHz。
>
> 根因：`SpectrumGrid.frequencies` 是**中心点阵**（`spectrum.py:40` 用 `(i - (N-1)/2) * spacing`），不是边缘阵列。`spectrum.py:28` 把 `lowest` 算成 `center - (N-1)*spacing/2`，它其实是**第一个信道中心**，被命名成 "lowest" 极易被误读为带边。
> **决定**：覆盖不变式施加在连接**中心**上（`BandPlan` 零改动）。代价是**最外侧连接的槽边缘越出标称频段**（下侧 25.0 GHz、上侧 19.9 GHz）。这是标准做法（带边是放大器/WSS 滚降起点，ODU 中心在带内即可），但需 **[需定夺-1]** 确认是否接受该越出量。

---

### 2.2 连接 → 槽数：由什么决定

**结论：由比特率 ÷ 调制谱效率决定，取 `ceil`；保护带显式参数化。**

本仓已有的谱效率定义（**必须沿用，不要另立公式**）：

- `spectral_efficiency_bps_hz = bits_per_symbol`（`multiband:src/multiband_optical_rl/modulation.py:45-48`）——注意这是 bit/s/Hz，数值等于 bits/symbol。
- `max_supported_rate_gbps(fmt, W) = spectral_efficiency_bps_hz * W / 1e9`（`modulation.py:87-89`）。
- 格式阶梯 `BPSK=1, QPSK=2, 8QAM=3, 16QAM=4, 64QAM=6`（`modulation.py:66-72`）。

即本仓隐含约定 **`Rs = W`**（符号率等于信道宽度，无滚降裕量）。

**冻结公式**：

```
required_width_hz  = bit_rate_bps / bits_per_symbol * (1 + roll_off)
occupancy_slots    = ceil(required_width_hz / slot_width_hz)        # 整数，>= 1
guard_slots        = 0            # 显式保留；见下方说明
n_slots            = occupancy_slots + guard_slots
```

**取整规则（冻结）**：

- 一律 **`ceil`，禁止 `floor`**。理由：`floor` 会让实际占用窄于该比特率所需的谱宽，低估容量并破坏 §2.3 的 50 GHz 对齐。
- `occupancy_slots >= 1`（1 Gbps 的 BPSK 也占 1 槽，不允许 0 槽连接）。
- 取整在**槽数空间**做，不在 Hz 空间做——避免浮点 ceil 边界。

**`guard_slots` 语义**：本仓 `Rs = W` 意味着 `roll_off = 0`。若要显式留 RRC 滚降保护，用 `roll_off` 参数（默认 `0.0`，与 `modulation.py:87-89` 现状一致；波形侧 `rrc_roll_off=0.1` 见 `ecz-dev:.../physics/qot_checker.py:74`）。

**[需定夺-2]** `roll_off` 取 0 还是 0.1，直接决定 §2.3 的对齐算例是否成立（见算例 4）。默认取 `0.0` 以保证与现有 `max_supported_rate_gbps` 自洽。

**算例**（`slot_width_hz = 12.5e9`）：

| # | 比特率 | 格式 | bps | roll_off | required_width | 槽数 | 占用宽度 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 1 | 100 Gbps | QPSK | 2 | 0.0 | 50.0 GHz | **4** | **50.0 GHz** | ✅ **与固定 50 GHz 对齐**：恰好 1 个遗留信道 |
| 2 | 400 Gbps | 16QAM | 4 | 0.0 | 100.0 GHz | **8** | 100.0 GHz | ⚠️ 遗留需 **2** 个 50 GHz 信道 → 灵活栅格下是 **1 个连接**（语义差异，见 §4） |
| 3 | 200 Gbps | 64QAM | 6 | 0.0 | 33.333 GHz | **3** | 37.5 GHz | 非整数宽度 → `ceil` 生效；遗留 50 GHz 多占 12.5 GHz |
| 4 | 100 Gbps | QPSK | 2 | 0.1 | 55.0 GHz | **5** | 62.5 GHz | 若启用滚降保护，**对齐破坏**（4→5 槽） |

**算例可复算**：附录 A3 给出 `ceil` 规则的逐例计算。

---

### 2.3 固定栅格必须是特例（FG-2/3 的验收前提）

**结论：固定 50 GHz 栅格 = 灵活栅格的以下参数组合**：

```
slot_width_hz            = 12.5e9        (= 50e9 / 4)
num_slots                = 230 * 4 = 920
band_low_edge_hz         = SpectrumGrid.frequencies[0] - 50e9/2
first_slot ≡ 0 (mod 4)   恒成立
n_slots ∈ {4k : k >= 1}  恒成立
guard_slots = 0, roll_off = 0.0, contiguous_slots = True, alignment_slots = 4
```

**中心频率公式（冻结，唯一的频率真源）**：

```
centre(first_slot, n_slots) = band_low_edge_hz + (first_slot + n_slots / 2) * slot_width_hz
```

> ⚠️ **半槽陷阱（已实测，必须避开）**：分子是 `n_slots / 2`（区间**中心**），**不是** `(n_slots - 1) / 2`（点阵中心）。用后者会整体偏移 **6.25 GHz = 半槽**，230 道全部错位，§2.3 的逐位不变立刻失败。
> 两者混淆的根源就是 §2.1 提到的：`SpectrumGrid.frequencies` 用点阵写法 `(i-(N-1)/2)`，而槽是**区间**。见附录 A1 实测。

**可测判据（三条，全部可断言）**：

- **AC-1（频率逐位相等）**：`slot_width=12.5e9, num_slots=920, band_low_edge = SpectrumGrid(230, 50e9, 190.2127e12).frequencies[0] - 25e9` 下，把 230 个 4 槽块按上式求中心，得到 `centres`，断言 `np.all(centres == legacy.frequencies)`（**float64 原始 `==`，非 `allclose`**）。**已实测通过**：最大偏差 `0.0 Hz`，不一致元素 `0` 个（附录 A1）。
- **AC-2（占用掩码逐位相等）**：固定参数下，920 位槽掩码按每 4 槽 OR 归约，与遗留 230 位 `occupied_channels` 逐位相等。
- **AC-3（QoT 输出逐位相等）**：满载 230 连接时，`_calc_span_noise_and_power_jit` 的四个返回数组与现行实现 `np.array_equal` 为 True。
  *依赖*：§3.5 的"每连接一个节点、中心频率取连接中心"，且 `f_abs_hz` 逐位等于遗留网格（AC-1 已保证）。因为所有下游量（`alpha_power_npm`、`a_eff_m2`、`f_rel_hz`、`t_tilde`）都是 `f_abs_hz` 的确定性函数，逐位相等会传递下去。

**AC-1/2/3 合起来构成"既有测试逐位不变"的充分条件。**

---

### 2.4 ODE 网格粒度 vs 槽粒度（本设计的物理核心）

**结论：槽边界可以落在 ODE 网格点之间——因为 ODE 网格是"每连接一个点"，不是"每槽一个点"。不需要加密 ODE 网格；加密反而是错的。**

**机制（证据）**：`_calc_exact_power_profile_ode`（`main:.../models/numba_kernels.py:174-214`）对 `N = len(p_in_w)` 个节点各持一个 `f_hz[i]` 与 `p[i]`，耦合系数 `G[i,k]` 只由**节点中心间距** `|f_hz[i]-f_hz[k]|` 决定（`:184`），RK4 积分 `dp/dz`（`:152-170`）。**代码里没有任何对频率区间做积分的量**——没有 `Δf` 的带宽因子，没有 `f_lo/f_hi`。它不是"频率轴上的离散化"，而是"一组耦合 ODE 在代表点上的解"。

因此：**一条连接 = 一个节点，位于该连接的中心频率。** 槽边界在物理上位于该节点两侧，与 ODE 无关。

**量化估计（本节所有数字均可复算，附录 A2）**：

拉曼耦合在 `Δf ≤ 13.2 THz` 是**线性**的：`g = Cr·Δf`，`Cr = 2.8e-17 1/(W·m·Hz)`（`main:.../models/isrs_gn.py:26`；`numba_kernels.py:185-187`）。物理上正确的耦合是两条连接**带重叠加权**：

```
g_eff(Δf_c) = ∫ g(Δf_c + w) · T(w) dw ,  T = 宽度 (B_i+B_j) 的三角分布
```

对**线性** `g`，三角分布零均值 ⟹ `g_eff = g(Δf_c)` **精确成立**（与带宽无关）。误差只在卷积支撑跨越 `g` 的**折点**时出现，折点在 13.2 THz 与 15.6 THz（`numba_kernels.py:185,188-191`）。

实测最大相对误差（扫描全部物理合法的非重叠连接对，附录 A2）：

| 频段资产 | 频段跨度 | 连接宽度 50 GHz | 100 GHz | 200 GHz | 400 GHz | 800 GHz | 1600 GHz |
|---|---|---|---|---|---|---|---|
| **C+L**（230×50G，跨度 11.45 THz） | < 13.2 THz 折点 | 2.7e-16 | 2.4e-16 | 2.6e-16 | 2.4e-16 | 2.4e-16 | **2.9e-16** |
| C+L+S**（417×50G，跨度 20.80 THz） | > 13.2 THz 折点 | 9.2e-6 | 3.1e-3 | **1.0e+0** | **2.3e+0** | **3.0e+0** | 9.7e+0 |

（C+L 那一行全部是浮点机器精度，差异来自梯形积分网格本身，非模型误差。）

**读法**：

- **C+L 上，中心取样是精确的（机器精度），对任意连接宽度都成立**——因为卷积支撑 `[Δf_c ± (B_i+B_j)/2]` 永不越过 13.2 THz（最极端的边沿连接中心也内缩了半个连接宽度，支撑右端恰好停在频段边 11.45/20.80 THz）。**故 C+L 上 ODE 网格无需任何改动、无需加密。**
- **C+L+S 上会真的错**：100 GHz 连接已 0.31%，200 GHz 起进入 O(1)。**这是 C+L+S 资产的固有代价，与灵活栅格无关也部分相关**——注意 50 GHz 的**现行**固定栅格在 C+L+S 上已有 9.2e-6 误差，不是新引入的。
- **误差源不是 ODE 网格，而是"代表点选哪个"。** 若用**最低槽**频率代替中心，代表点偏移最多 `(n-1)/2 · slot`；8 槽（100 GHz）连接偏移 43.75 GHz，相对耦合误差 `0.04375/Δf_thz`，在 Δf=11.45 THz 处为 **0.38%**。**因此冻结规则：代表点必须是连接中心（§2.3 公式），不得用首槽或末槽。**

**若坚持要加密会怎样（反面论证）**：把 C+L 拆成 12.5 GHz 槽级节点 = 920 节点。由 §2.8 实测 `t(N) ∝ N²`，代价 `5.25 ms × (920/230)² = 84 ms/跨段`（**16×**），且会改变 `Δf` 语义（槽间 Δf = 12.5 GHz，使 SRS 耦合矩阵退化为近邻主导，与物理不符）。**结论：不要加密。**

**[需定夺-3]** C+L+S（417 槽）是否要支持？若要，则 200 GHz 以上连接的 SRS 耦合需改用带重叠积分（`g_eff`），不能沿用中心取样。本设计对 C+L 给出精确结论，对 C+L+S 只给出误差上界与可选修正路线，**不替编排方决定是否做**。

---

### 2.5 QoT 接口：变宽连接如何进入 ODE；功率谱密度如何定义

**先说一个必须澄清的事实**：**本仓不存在"功率谱密度"对象。** 全文搜索无 PSD 抽象。宽度在本物理模型里只通过**三个标量/数组**起作用：

1. **中心频率** → SRS 耦合 `Δf`（`numba_kernels.py:184`）与色散相干项 `phi_i`（`:381`）、`phi_base`（`:397`）。
2. **符号率 `Rs`** → SPM（`:386`）、XPM（`:405-406`）、ASE（`:482`）。
3. **波长相关量** → `alpha_power_npm`、`a_eff_m2`、`nf_lin`，按 `grid.wavelengths` 逐点算（`main:.../models/isrs_gn.py:73-94`）。

所以"宽连接进入 ODE"的答案是：**它不作为一个宽节点进入，而是作为一个 `n_slots` 已知、中心已知的普通节点进入；`Rs` 变成逐连接数组。**

**功率谱密度的定义（冻结，诚实版）**：本模型把每条连接的谱视为**以其中心频率为原点的扁平带** `[f_c - n_slots·W_s/2, f_c + n_slots·W_s/2]`，谱宽 `n_slots·W_s`，带内 PSD 视为常数。该平坦谱**从不显式进入计算**——它只用于**推导 §2.4 的 `g_eff` 误差上界**。在中心取样规则下（C+L 上精确），该 PSD 假设不产生任何数值误差。**这比引入一个真 PSD 对象更诚实**：没有对象被凭空发明。

**精确签名变更**（FG-2 照抄）：

```python
# 变更 1：唯一破坏性签名改动。标量 Rs -> 逐连接数组。
#   位置：main:src/network_nlo_eval/models/numba_kernels.py:507
@njit(cache=True)
def _calc_span_noise_and_power_jit(
    power_in_w: NDArrayFloat,       # (N,)  每连接入纤功率 (W)，未占用=0
    span_length_m: float,           # (m)
    f_rel_hz: NDArrayFloat,         # (N,)  连接中心 - band_center (Hz)
    f_abs_hz: NDArrayFloat,         # (N,)  连接中心绝对频率 (Hz)
    lambdas_m: NDArrayFloat,        # (N,)  C_LIGHT / f_abs_hz (m)
    alpha_power_npm: NDArrayFloat,  # (N,)  按连接中心波长算 (Np/m)
    a_eff_m2: NDArrayFloat,         # (N,)  按连接中心波长算 (m^2)
    beta2_s2_m: float,              # (s^2/m)  标量，见下"注意"
    beta3_s3_m: float,              # (s^3/m)  标量，见下"注意"
    n2_m2_w: float,                 # (m^2/W)
    nf_lin_channels: NDArrayFloat,  # (N,)  线性 NF
    symbol_rate_hz: NDArrayFloat,   # (N,)  ★ 由 float 改为逐连接数组 (Hz)
    n_effective_spans: int,
    cr_w_m_hz: float,               # 1/(W*m*Hz)  = 2.8e-17
    delta_f_co_hz: float,           # Hz
    phi_kurtosis: float = 0.0,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """返回 (p_out_w, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w)，各形状 (N,)，单位 W / W^2。"""
```

`Rs` 标量→数组的**全部 4 个使用点**（`main:.../models/numba_kernels.py`）：

| 行 | 现状 | 改为 |
|---|---|---|
| `:386` | `_calc_spm_variance_jit(p_in_w[i], rs_hz, ...)` | `rs_hz[i]` |
| `:405` | `rs_l_hz=rs_hz` | `rs_l_hz=symbol_rate_hz[l_idx]` |
| `:406` | `rs_i_hz=rs_hz` | `rs_i_hz=symbol_rate_hz[i]` |
| `:482` | `... * rs_hz`（ASE） | `... * symbol_rate_hz[i]` |

**不变的部分（好消息）**：

- `_calc_exact_power_profile_ode`（`:174`）**签名完全不变**——它本来就只吃数组。调用点 `:521-523` 只需传入"连接中心"数组。
- `_compute_nli_variances` 的 `f_rel_hz`/`f_abs_hz`/`lambdas_m`/`a_eff`/`alpha` 已全是数组，**只把 `rs_hz: float`（`:348`）改成数组并按上表取值**。
- `MultiBandISRSGN` 里 `alpha_power_npm_channels` / `a_eff_m2_channels` / `nf_lin_channels`（`isrs_gn.py:73-94`）从 `grid.wavelengths` 改为 `connection_center_wavelengths`，**纯机械改**。
- `isrs_gn.py:109-110` 的 `f_m_hz` / `f_M_hz`（`= f_rel[0] - Rs/2`、`f_rel[-1] + Rs/2`）**必须改**：变宽后单条 `Rs/2` 不再是带边。冻结为 `f_m_hz = min(centre - n_slots*W_s/2)`、`f_M_hz = max(centre + n_slots*W_s/2)`。

**注意（未决，但不影响接口）**：`beta2_s2_m` / `beta3_s3_m` 在 `isrs_gn.py:99-105` 是按**参考波长**算的**标量**，与连接无关。因此**不需要**为灵活栅格改成数组——但这本身是一个既有的建模简化（多波段下 D 随 λ 变化被忽略），灵活栅格会把它暴露得更明显。**[需定夺-4]** 是否顺带把 `beta2/beta3` 改成逐连接数组（本设计认为属独立改进，不应绑进 FG-2）。

**数组形状错误语义（冻结）**：新增校验，任一数组长度 ≠ `N` 抛 `ValueError`，消息含实际形状：

```python
if power_in_w.shape != (n_conn,):
    raise ValueError(f"power_in_w must have shape ({n_conn},), got {power_in_w.shape}.")
```

**ROADM 侧（ecz-dev）**：`occupied_channels` 是长度 `num_channels` 的 bool 数组（`main:.../network/state.py:41`、`:121`）。变宽连接需要 `occupied_channels[first_slot : first_slot + n_slots] = True`（**切片**），这是 §4 的一处**语义改**。同时 `passband_hz` 兜底 `channel_spacing_hz`（`ecz-dev:.../physics/qot_checker.py:212`）应改为**该连接的 `n_slots * slot_width_hz`**——否则宽连接会被 WSS 滤波掉。这在物理上**必须**改，且按连接而非按网格取值。

---

### 2.6 资产 schema 迁移：`roadm.blocked_channels` → 什么

**结论：新增 `blocked_slots: list[int]`（槽索引），`blocked_channels` 保留为**只读**兼容输入**。**

**现状证据**：旧字段挂在**节点**上（不是边），`multiband:src/multiband_optical_rl/topology.py:35-36` 读 `node["roadm"]["blocked_channels"]`；资产里 28 处 `blocked_channels` 全部在 `nodes[*].roadm` 下。ecz-dev 的 `ROADMConfig` 校验（`ecz-dev:.../network/elements.py:86-90`）拒绝负索引与重复。`topology.py:27-41` 额外做越界前置拦截（`0 <= index < num_channels`），并有详细注释说明上游会**静默丢弃**越界索引——**这个拦截必须同步迁移到槽空间**，否则越界阻断会重演静默失效。

**新旧 schema**：

```jsonc
// 旧（继续可读，仅作兼容输入）
{ "roadm": { "blocked_channels": [3, 17] } }

// 新（权威）
{ "roadm": { "blocked_slots": [12, 13, 14, 15, 68, 69, 70, 71] } }
```

**兼容策略（冻结）**：

1. 解析顺序：若存在 `blocked_slots` → 用它。**否则**若存在 `blocked_channels` → 按 `first_slot = index * slots_per_channel` 展开为区间并记 `legacy_expanded=True`。
2. `slots_per_channel = round(channel_spacing_hz / slot_width_hz)`；在固定 50 GHz + 12.5 GHz 槽下 `= 4`。若 `channel_spacing_hz` 不是槽宽整数倍 → **`ValueError`**（不做取整猜测）。
3. 两者**同时**出现且展开后不一致 → **`ValueError`**，不做静默合并。
4. 越界：任一槽 `0 <= s < num_slots`，否则 `ValueError`（沿用 `topology.py:38-41` 的"构建期拦截"纪律）。
5. 迁移脚本**只写新字段，保留旧字段原样**，保证迁移可重入、diff 可读。

**迁移函数接口（冻结，FG-3 照抄）**：

```python
def migrate_roadm_blocked_slots(
    graph_json: dict[str, Any],
    slot_width_hz: float,
    channel_spacing_hz: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """把 node["roadm"]["blocked_channels"] 迁移为 node["roadm"]["blocked_slots"]。

    Args:
        graph_json: load_graph() 的返回值（原样传入，不修改）。
        slot_width_hz: 槽宽 (Hz)，须能整除 channel_spacing_hz。
        channel_spacing_hz: 遗留信道间隔 (Hz)。

    Returns:
        (新 graph_json, 统计)。统计键固定为：
          "nodes_migrated"  实际发生字段新增的节点数
          "nodes_skipped"   已有 blocked_slots 而未改动的节点数
          "slots_written"   写入 blocked_slots 的槽总数
          "legacy_channels_read" 从旧字段读到的信道索引总数

    Raises:
        ValueError: channel_spacing_hz 不是 slot_width_hz 的整数倍；
                    新旧字段同时存在且展开结果不一致；
                    展开后出现越界槽索引。
    """
```

**语义差异（必须写进迁移 PR 的说明）**：旧语义"阻断信道 i"= 该 50 GHz 不可用。新语义"阻断槽 s"= 该 12.5 GHz 不可用。**灵活栅格下一个宽连接若与任一被阻断槽重叠即不可行**——即阻断粒度变细，**可能让原本可行的解变得不可行**。这是**语义改**，不是机械改（见 §4）。

---

### 2.7 对 `multiband-optical-rl` 的破坏面清单

图例：**机**=机械改（不改变任何数值结论）／**语**=语义改（会改变实验结论）。

| # | 文件:行 | 现状 | 改什么 | 类 |
|---|---|---|---|---|
| 1 | `config.py:33` | `num_channels: int = 230` | 改 `slot_width_hz=12.5e9` + `num_slots=920`；`num_channels` 降级为派生量 | 机 |
| 2 | `config.py:34` | `channel_spacing_hz = 50e9` | 保留为**兼容输入**，只用于 §2.6 的 `slots_per_channel` 推导 | 机 |
| 3 | `config.py:35` | `center_frequency_hz = 190.2127e12` | 保留；**新增 `band_low_edge_hz`**（§2.3 公式），二者关系必须断言 | 机 |
| 4 | `config.py:36` | `symbol_rate_hz = 50e9`（标量） | 降级为**默认** `Rs`；实际 `Rs` 由连接推导 | **语** |
| 5 | `config.py:44-50` | `spectrum_grid()` 造 `SpectrumGrid` | 造 `FlexibleGrid`；`spectrum_grid()` 保留为**派生投影** | 机 |
| 6 | `config.py:52-55` | `channel_width_hz = channel_spacing_hz` | 语义从"每信道宽度"变为"每连接宽度"，**逐连接取值** | **语** |
| 7 | `config.py:42` | `max_bitrate_gbps = 400.0` | 观测归一化尺度；算例 2 显示 400 Gbps 在 50 GHz 下需 bps=8（阶梯无此项）→ **[需定夺-5]** | **语** |
| 8 | `modulation.py:87-89` | `max_supported_rate_gbps(fmt, W)` | **签名不变**，`W` 由调用方传连接宽度 | 机 |
| 9 | `topology.py:27-41` | 越界检查按 `num_channels` | 改为按 `num_slots`；旧 `blocked_channels` 走 §2.6 展开 | **语** |
| 10 | `topology.py:59-66` | `nf` 形状 `(num_channels,)` | 形状按**连接数**（动态），不再是 `(num_slots,)` | **语** |
| 11 | `env.py:392,405,435,470` | `occupied_channels[ch]` 单点 | 连续区间判定（切片 `.any()` / first-fit 扫连续空段） | **语** |
| 12 | `env.py:405-413` `_try_first_fit` | 找**单条**空闲信道 | 找**长度 ≥ 需求槽数**的连续空闲段 | **语** |
| 13 | `env.py:16-20` 观测向量 | 空闲信道数/平均信道索引/碎片 | 改为空闲**槽**数、最大连续空段长度（碎片语义更强） | **语** |
| 14 | `reopt/trial.py:190,267` | `0 <= channel < num_channels` | 范围校验改为 `(first_slot, n_slots)` 区间；重叠检测改区间交 | **语** |
| 15 | `scan.py:153,298` | `_num_channels` / `zeros(_num_channels)` | 数组长度语义：噪声数组按**连接数**（动态），占用掩码按**槽数** | **语** |
| 16 | `metrics.py:76-83,110-141` | 碎片指数/占用率按信道 | 碎片按**槽 + 连续段**；`legacy_fragmentation_index` 需保留为固定栅格专用 | **语** |
| 17 | `bands.py:84,100,119,159-167` | 按信道索引映射频段 | 按连接中心频率映射；覆盖校验按**中心**（§2.1） | **语** |
| 18 | `scan_prune.py:152,168,182-186` | `occupied_channels[band]` | 形状语义变更同上 | **语** |
| 19 | `tools_build_rate_table.py:109-140` | `occupancy_buckets(num_channels)` | 按**槽总数**与连续段可行性生成档位；**会改变档位数值** | **语** |
| 20 | `tools_build_rate_table.py:366` | 用 `config.channel_width_hz` | 改用连接的 `n_slots*slot_width` | **语** |
| 21 | `assets/pan_europe_multiband.json` | 28 处 `blocked_channels` | 跑 `migrate_roadm_blocked_slots`；**资产 diff 必须逐条 review** | **语** |
| 22 | `policies/search.py:45`, `oracle_provision.py:192`, `reopt/strategies/oracle.py:287` | `occupied_channels[ch]` | 区间判定 | **语** |
| 23 | `network/state.py:41,44,46,121`（上游） | 长度 `num_channels` 的数组 | 占用掩码长度 `num_slots`；`launch_power_profile_w` 长度**连接数** | **语** |

**统计：23 处中 4 处机械改、19 处语义改。** 语义改占比高是**预期且正确**的——灵活栅格不是重构，是**问题定义的改变**（"选一条信道"变成"选一个连续频段"）。**[需定夺-6]** 建议 FG-3 按"先 #1–#8 打通固定栅格回归，再逐个语义改"的顺序，每步单独出结论。

---

### 2.8 算力影响

**结论：若采用"每连接一个节点"（§2.4 结论），灵活栅格对 QoT 计算的直接影响为 0。**

**实测（本 worktree，`numba_kernels.py:489` 热路径，单 100 km 跨段，5 次取中位数，附录 A5 可复跑）**：

| 连接数 N | 空闲机器 (load 1.00) | **重载机器 (load 21.3)** | N² | 预测/实测比 |
|---|---|---|---|---|
| 115 | 1.31 ms | 3.34 ms | 13 225 | — |
| **230（现状）** | **5.25 ms** | **12.93 ms** | 52 900 | 1.00（基准） |
| 417 | 16.72 ms | 41.99 ms | 173 889 | 3.29 预测 / 3.19–3.25 实测 |
| 830 | 66.05 ms | 165.34 ms | 688 900 | 13.03 预测 / 12.60–12.79 实测 |

> **计时必须连同负载一起读（跨 WP 规则 3）**：两次测量分别在本机 load average **1.00** 与 **21.30** 时取得（21.3 那一档是同机另一条线在跑，与本 WP 无关）。重载使**所有** N 的绝对耗时统一放大约 **2.5×**（3.34/1.31 = 2.55，12.93/5.25 = 2.46，41.99/16.72 = 2.51，165.34/66.05 = 2.50）——即**纯粹的 CPU 争用，与 N 无关**。
> **因此本节所有结论只依赖比值，不依赖绝对毫秒数**：把重载数据除以 2.5 得 1.34 / 5.17 / 16.80 / 66.14 ms，与空闲机数据在 2% 内一致。
> **FG-3 的 F3-8 判据也据此定义为比值**（相对同机器同负载下的迁移前基线），而非绝对阈值。

**O(N²) 确认**：230→417 预测 3.29×、实测 3.19–3.25×；230→830 预测 13.03×、实测 12.60–12.79×。**两种负载下都成立。**

**三种方案的算力账**（比值形式，与负载无关）：

| 方案 | 节点数（C+L） | 相对现状耗时 | 评价 |
|---|---|---|---|
| **每连接一个节点（采用）** | ≤ 230（连接数，与槽数无关） | **1.00×** | ✅ 推荐 |
| 每槽一个节点 | 920 | **16.0×** | ❌ 且 Δf 语义错误（§2.4） |
| 槽+连接混合（谱离散化） | > 920 | ≫ 16× | ❌ 超出本模型结构（无 PSD 对象，§2.5） |

**真正的算力风险不在 QoT，在 RL 搜索空间**：

- **动作空间**：`env.py` 的信道选择从 230 选 1 变成"起点 × 长度"组合。C+L 下连续段数约 `O(920²/2) ≈ 4.2e5`（未计长度约束），**比现状大 3 个数量级**。这会直接拖垮 `policies/search.py` 的有界重试与 `scan.py` 的剪枝。
  **缓解（不替编排方决定）**：用 §2.2 的 `n_slots` 查表把长度限定在合法档位（实测合法档位仅 5–6 档：算例 1–4），动作空间降到 `~920 × 6 ≈ 5.5e3`，仍是现状的 24×。**[需定夺-7]** 动作编码方案（起点+档位 vs 打包整数）需编排方定夺。
- **缓存失效**：`scan.py:153` 的 `LinkNoiseCache` 按频谱签名缓存。灵活栅格下"频谱状态"的种类数从 `2^230` 变成"连续段组合"，缓存命中率会**大幅下降**。这是**真实的、可能主导总运行时间**的成本，且**本设计未测量**（需要真实 workload）。**[标记为未测量]**。

**诚实声明**：本节只测了**单跨段 QoT 内核**。端到端 RL 实验墙钟时间的变化**未测量**，因为它取决于 §4 的动作空间设计（未冻结）。**本文件中的所有算力数字都只覆盖 QoT 内核。**

---

## 3. 冻结接口

> FG-2 / FG-3 照抄。所有单位显式标注；错误语义显式标注。

### 3.1 `FlexibleGrid`（新增，替代 `SpectrumGrid` 的网格角色）

```python
@dataclass(frozen=True)
class FlexibleGrid:
    """灵活栅格：等宽槽的半开区间并集 [band_low_edge_hz, band_high_edge_hz)。

    单位：全部频率为 Hz。槽索引为 0-based 整数。
    频率真源唯一：centre(i, n) = band_low_edge_hz + (i + n/2) * slot_width_hz
    """
    slot_width_hz: float      # 标称槽宽 (Hz)，须 > 0
    num_slots: int            # 槽总数，须 >= 1
    band_low_edge_hz: float   # 槽 0 的**下边缘** (Hz)，须 > 0

    # --- 不变式（__post_init__ 强制）---
    # 1. slot_width_hz > 0
    # 2. num_slots 为正整数且非 bool
    # 3. band_low_edge_hz > 0
    # 4. band_low_edge_hz + num_slots*slot_width_hz > 0（隐含于 3）
    # 注意：**不做**频段覆盖校验——覆盖不变式施加在连接"中心"上（见 §2.1），
    #       由 BandPlan 在上层检查，理由见 §2.1 的实测陷阱。

    def slot_low_edge_hz(self, slot: int) -> float:
        """槽 slot 的下边缘 (Hz)。IndexError if not 0 <= slot < num_slots."""

    def slot_high_edge_hz(self, slot: int) -> float:
        """槽 slot 的上边缘 (Hz)，半开区间不含此值。IndexError 同上。"""

    def slot_center_hz(self, slot: int) -> float:
        """槽 slot 的中心 (Hz)。IndexError 同上。"""

    def centre_hz(self, first_slot: int, n_slots: int) -> float:
        """占据 [first_slot, first_slot+n_slots) 的连接的**中心频率** (Hz)。

        ★ 分子是 n_slots/2（区间中心），不是 (n_slots-1)/2（点阵中心）。
          用后者会整体偏移半槽（12.5 GHz 槽下 6.25 GHz），AC-1 失败。

        Raises:
            IndexError: first_slot < 0 或 first_slot + n_slots > num_slots。
            ValueError: n_slots <= 0。
        """

    def edges_are_inside(self, low_hz: float, high_hz: float) -> bool:
        """连接带 [low, high) 是否完全落在本栅格内。仅供诊断。"""
```

### 3.2 `ConnectionSpec`（新增：一条连接）

```python
@dataclass(frozen=True)
class ConnectionSpec:
    """一次实验中的一个 WDM 连接（= 一个 ODE 节点 = 一个 RL 分配单元）。"""
    connection_id: int          # >= 0，同一实验内唯一
    first_slot: int             # 占据的起始槽 (0-based, 闭)
    n_slots: int                # 占据的槽数，>= 1
    bit_rate_bps: float         # 目标比特率 (bit/s)，> 0
    modulation: str             # 必须出现在 MODULATION_FORMATS 中
    symbol_rate_hz: float       # 符号率 (Hz)，> 0
    roll_off: float             # RRC 滚降系数，0 <= roll_off < 1
    launch_power_w: float       # 单连接入纤总功率 (W, 双偏振合计)，>= 0；0 = 未占用

    # --- 不变式 ---
    # 1. first_slot >= 0 且 first_slot + n_slots <= grid.num_slots
    # 2. n_slots >= 1
    # 3. n_slots == slots_for_rate(bit_rate_bps, bits_per_symbol(modulation), roll_off, grid.slot_width_hz)
    #    —— §2.2 公式的**强制**形式：不允许"给 4 槽却只跑 100 Gbps 的 64QAM"这类
    #       静默欠配。违反抛 ValueError。
```

### 3.3 `slots_for_rate()`（§2.2 公式的可执行形式，FG-2/3 共用）

```python
def slots_for_rate(
    bit_rate_bps: float,
    bits_per_symbol: int,
    roll_off: float,
    slot_width_hz: float,
) -> int:
    """由比特率算所需槽数。纯函数，无状态。

    公式:  n = ceil( (bit_rate_bps / bits_per_symbol) * (1 + roll_off) / slot_width_hz )
          n = max(n, 1)

    Args:
        bit_rate_bps: 比特率 (bit/s)，> 0
        bits_per_symbol: 每符号比特数 (bit/symbol)，>= 1
        roll_off: RRC 滚降系数，0 <= roll_off < 1。取 0.0 与现有
            modulation.py:87-89 的 `Rs = W` 约定自洽（默认）。
        slot_width_hz: 槽宽 (Hz)，> 0

    Returns:
        槽数 (int)，>= 1

    Raises:
        ValueError: bit_rate_bps <= 0；bits_per_symbol < 1；
                    not (0 <= roll_off < 1)；slot_width_hz <= 0
    """
```

**单位表**：

| 量 | 单位 | 备注 |
|---|---|---|
| `slot_width_hz` / 所有频率 | Hz | 全仓频率一律 Hz，**不用 THz**（`spectrum.py:15` 明确） |
| `bit_rate_bps` | bit/s | `config.py:42` 现用 Gbps，边界处换算 |
| `symbol_rate_hz` | Hz | |
| `launch_power_w` | W | **双偏振合计**（`qot_checker.py:127-128` 的既有约定：网络发射功率是 DP 总功率，IQModulator 需 per-pol） |
| `alpha_power_npm` | Np/m | `_calc_fiber_attenuation_npm` |
| `a_eff_m2` | m² | |
| `cr_w_m_hz` | 1/(W·m·Hz) | `isrs_gn.py:26` = 2.8e-17 |
| `sigma2_*` | W²（功率谱密度量纲） | 见 §2.5 |

### 3.4 数组形状契约（冻结）

设活跃连接数 `N = len(connections)`。以下数组**必须**长度恰为 `N`、同序对齐：

```
power_in_w, f_abs_hz, f_rel_hz, lambdas_m, alpha_power_npm,
a_eff_m2, nf_lin_channels, symbol_rate_hz
```

`f_abs_hz[i] = grid.centre_hz(conn[i].first_slot, conn[i].n_slots)`
`lambdas_m[i] = C_LIGHT / f_abs_hz[i]`
`f_rel_hz[i] = f_abs_hz[i] - grid.band_center_hz`（`band_center_hz` = `band_low_edge_hz + num_slots*slot_width_hz/2`）

**未占用连接约定**：`launch_power_w == 0` 的连接**仍占一个数组槽位**（保持对齐），其噪声输出为 `NaN`（对齐 `qot_checker.py:247-249` 的既有 `np.nan` 约定）或 `0.0`——**[需定夺-8]**：现有 `_receiver_metrics` 用 `NaN`，统计路径 `_calculate_ase_noise_variance`（`:478`）用 `0.0` 跳过。**本设计不统一这两者**，因为统一会改变既有输出。

### 3.5 冻结的**不变**签名（FG-2 不应改动）

```python
# main:src/network_nlo_eval/models/numba_kernels.py:174 —— 签名完全不变
@njit(cache=True)
def _calc_exact_power_profile_ode(
    p_in_w: NDArrayFloat,      # (N,) 每连接入纤功率 (W)
    f_hz: NDArrayFloat,        # (N,) 每连接**中心**频率 (Hz)  ← 唯一约定变更
    alpha_npm: NDArrayFloat,   # (N,) Np/m
    L_m: float,                # 跨段长 (m)
    cr_w_m_hz: float,          # 1/(W·m·Hz)
    n_steps: int = 20,         # RK4 步数，保持 20（:520 注释称精度/速度最佳）
) -> NDArrayFloat:             # (N,) 出纤功率 (W)，已 clamp 到 >= 0（:210-212）
```

**唯一变更**：调用方（`isrs_gn.py:150-167`）传入的 `f_abs_hz` 从"栅格频率"变为"连接中心频率"。函数内部无任何改动需求。

---

## 4. 迁移清单（按文件 → 改什么 → 机/语）

见 §2.7 的 23 行表。**分支归属**（依 §1.1 的更正）：

| 阶段 | 分支 | 改动 | 判据 |
|---|---|---|---|
| **S1** | `main` | #1–#8 上游侧：加 `FlexibleGrid`/`ConnectionSpec`/`slots_for_rate`；改 `symbol_rate_hz: float → NDArrayFloat`（4 处）；`isrs_gn.py:73-94,109-110` 机械改 | AC-1/AC-3 通过；**main 现有测试全绿且未改期望值** |
| **S2** | `ecz-dev` | `ROADMConfig` 加 `blocked_slots`（`elements.py:83` 邻位）；`qot_checker.py:209` 阻断判定改槽区间；`:212` passband 改按连接 `n_slots*slot_width`；`:109` `occupied_width` 改按实际连接集合 | AC-2 通过 |
| **S3** | 合并后 | `multi_band` #9–#23 | 固定栅格回归等价于现状（§5 FG-3 AC-3） |

**G2 纪律**：S1/S2 期间**禁止**修改任何既有测试的期望值。AC-1/AC-3 之所以设计成"逐位相等"而不是"误差 < ε"，就是为了让这一点可被执行。

---

## 5. 验收判据

### FG-2（`network-nlo-eval` 实现）

| ID | 判据 | 断言 |
|---|---|---|
| F2-1 | 频率逐位不变 | `np.all(centres == SpectrumGrid(230,50e9,190.2127e12).frequencies)`，raw `==`。**已实测 = True，偏差 0.0 Hz** |
| F2-2 | 半槽陷阱已避 | 显式测试：用 `(n-1)/2` 公式的实现会偏移 6.25 GHz（防止回归到错公式） |
| F2-3 | QoT 输出逐位不变 | 满载 230 连接，`_calc_span_noise_and_power_jit` 四返回 `np.array_equal` 现状 |
| F2-4 | 槽数公式 | `slots_for_rate` 四个算例 = `[4, 8, 3, 5]`（算例 1–4，roll_off 分别 0,0,0,0.1） |
| F2-5 | 形状校验 | 长度 ≠ N 的数组 → `ValueError`，消息含 `(N,)` 与实际形状 |
| F2-6 | C+L 上 SRS 精确性 | 任意连接宽度 ≤ 1600 GHz，中心取样 vs 带重叠积分相对误差 `< 1e-12`（实测 ≤ 2.9e-16） |
| F2-7 | ODE 未被改动 | `_calc_exact_power_profile_ode` 的**签名与函数体逐行未变**（`git diff` 为空） |
| F2-8 | 既有测试 | main 全量测试绿，**零期望值修改** |

### FG-3（`multiband-optical-rl` 适配）

| ID | 判据 | 断言 |
|---|---|---|
| F3-1 | 占用掩码归约 | 920 位槽掩码按 4 槽 OR 归约 == 遗留 230 位 `occupied_channels` |
| F3-2 | 资产迁移幂等 | `migrate_roadm_blocked_slots` 跑两次，第二次 `nodes_migrated == 0` |
| F3-3 | 迁移一致性 | 迁移后 `len(blocked_slots) == 4 * len(blocked_channels)`，**对全部 28 处**成立 |
| F3-4 | 越界拦截 | 构造越界槽 → `ValueError`（不静默丢弃，延续 `topology.py:27-41` 纪律） |
| F3-5 | 固定栅格回归等价 | 固定参数下，固定种子的 oracle 分配序列与**迁移前**逐位相同 |
| F3-6 | 连续段分配 | 需求 8 槽的连接只被分配到长度 ≥ 8 的连续空段；不产生分裂分配 |
| F3-7 | 频段覆盖 | 230 连接中心全部落在 `[L_LOW, C_HIGH]`（与 `config.py:12-13` 现有结论一致） |
| F3-8 | 算力 | 满载 230 连接，单跨段 QoT 中位耗时 **≤ 1.10×** 迁移前基线（**同机器、同负载时段**测量；比值判据，不受 CPU 争用影响，见 §2.8） |

**F3-8 的 1.10 阈值是按 §2.8 的 O(N²) 实测定的**：若 FG-3 误把节点数做成槽数（920），耗时会是 16×，远超阈值——该阈值同时充当 §2.4 结论的回归守卫。

---

## 6. 不确定项与需定夺项

| ID | 问题 | 选项 | 后果 | 本设计倾向 |
|---|---|---|---|---|
| **需定夺-0** | 分支合流顺序 | (a) 先合 main+ecz-dev 再做灵活栅格；(b) 两边各自做，之后再合 | (b) 会让 §2.5 的接口改动写两遍 | (a) |
| **需定夺-1** | 槽边缘越出频段 25.0/19.9 GHz 是否可接受 | (a) 覆盖按中心（`BandPlan` 不动，AC-1 成立）；(b) 覆盖按边缘 | (b) **几何上不可行**：边缘方案缺 44.9 GHz，且重新锚定会破坏 AC-1 | (a) —— (b) 实测已排除，但越出量需物理侧确认可接受 |
| **需定夺-2** | `roll_off` = 0.0 还是 0.1 | (a) 0.0；(b) 0.1 | (b) 使算例 1 变 5 槽，**50 GHz 对齐破坏** | (a) |
| **需定夺-3** | 是否支持 C+L+S（417 槽） | (a) 只做 C+L；(b) 做并实现 `g_eff` 带重叠积分 | (b) 200 GHz+ 连接 SRS 误差 O(1) | 先 (a)，(b) 列为独立项 |
| **需定夺-4** | `beta2/beta3` 是否改逐连接 | (a) 不改（保持标量，沿用 `isrs_gn.py:99-105`）；(b) 改 | (b) 是独立的模型改进 | (a)，不绑进 FG-2 |
| **需定夺-5** | `max_bitrate_gbps=400` 与阶梯不自洽 | (a) 保留 400 作归一化尺度；(b) 降到 300（64QAM@50GHz 上限） | (a) 下算例 2 的 400 Gbps 需 bps=8，超出阶梯 | (a) + 显式文档化 |
| **需定夺-6** | FG-3 改造顺序 | (a) 先机后语；(b) 一次性 | (b) 无法定位回归 | (a) |
| **需定夺-7** | 动作空间编码 | (a) (起点, 档位)；(b) 打包整数；(c) 掩码 | 决定 RL 收敛性 | (a)，最易测 |
| **需定夺-8** | 未占用连接的噪声输出 | (a) `NaN`（波形侧）；(b) `0.0`（统计侧） | 统一会改变既有输出 | **不统一**，保持现状 |

**未测量项（诚实声明）**：

1. **端到端 RL 墙钟时间变化** —— 未测量。取决于需定夺-7（未冻结）与真实 workload。§2.8 只有 QoT 内核数字。
2. **`LinkNoiseCache` 命中率下降幅度** —— 未测量。代码证据：`scan.py:153` 缓存键含频谱签名，灵活栅格下状态空间组合数增加，命中率必降，但**降幅未测**。
3. **C+L+S 上的 SRS 修正（`g_eff`）实现后的实际误差** —— 未测量。§2.4 只给了误差上界（200 GHz 连接 O(1)），修正后的收敛性未验证。
4. **宽带滚降下的 XPM 修正** —— `_calc_xpm_variance_term_jit`（`:288-342`）的 `delta_f_hz` 路径用 `(2|Δf| - Rs_l)/(2|Δf| + Rs_l)`（`:336-337`）。当 `|Δf| < Rs_l/2`（即邻接的宽连接）该式进入未验证区间。**未检查其行为。**

---

## 附录 A：可重跑命令

**统一前置**（跨 WP 硬要求：`.pth` 指向主检出，不设 `PYTHONPATH` 会跑到另一棵树）：

```bash
cd /home/zsig/Documents/gits/wt/fg-design
export PYTHONPATH="$PWD/src"          # 硬要求
export NUMBA_CACHE_DIR=/tmp/fg-nb     # 避免 numba 往仓库写 __pycache__
PY="$HOME/miniconda3/envs/rl/bin/python"

# 硬断言：import 到的包必须落在本 worktree 内
$PY -c 'import network_nlo_eval as n; print(n.__file__)'
# 必须打印 /home/zsig/Documents/gits/wt/fg-design/src/network_nlo_eval/__init__.py

# 报计时须同时报负载
echo "loadavg: $(cat /proc/loadavg)"
```

### A1 — §2.3 AC-1 频率逐位相等（**已实测：偏差 0.0 Hz，0 不一致**）

```bash
$PY - <<'PY'
import numpy as np
from network_nlo_eval.core.spectrum import SpectrumGrid
N, SP, SLOT = 230, 50e9, 12.5e9; K = 4
g = SpectrumGrid(num_channels=N, channel_spacing_hz=SP, center_frequency_hz=190.2127e12)
band_low = g.frequencies[0] - SP/2.0
starts, nslots = np.arange(N)*K, np.full(N, K)
centres = band_low + (starts + nslots/2.0)*SLOT          # ★ n/2，不是 (n-1)/2
print("bit-identical:", bool(np.all(centres == g.frequencies)))
print("max |diff| Hz:", float(np.max(np.abs(centres - g.frequencies))))
wrong = band_low + (starts + (nslots-1)/2.0)*SLOT       # 错公式
print("trap offset GHz:", float(np.max(np.abs(wrong-g.frequencies)))/1e9)
PY
```

### A2 — §2.4 SRS 中心取样精确性（**已实测：C+L ≤ 2.9e-16；C+L+S 50GHz 已 9.2e-6**）

```bash
$PY - <<'PY'
import numpy as np
CR = 0.028/1e3/1e12
def g(d):
    d = np.abs(d)
    return np.where(d<=13.2, CR*d*1e12,
           np.where(d<15.6, (CR*13.2e12)*(15.6-d)/2.4, 0.0))
def g_eff(dfc, b, n=400001):
    tot = 2*b; w = np.linspace(-tot/2, tot/2, n)
    p = np.maximum(1-2*np.abs(w)/tot, 0.0); p /= np.trapezoid(p, w)
    return np.trapezoid(g(dfc+w)*p, w)
for span, name in [(11.45, "C+L 230x50G"), (20.80, "C+L+S 417x50G")]:
    for w_ghz in [50, 100, 200, 400, 800, 1600]:
        b = w_ghz/1e3
        if 2*b >= span: continue
        dfcs = np.linspace(b, span-b, 200)          # 非重叠、且中心已内缩
        rel = [abs(g_eff(d,b)-float(g(np.array([d]))[0]))/float(g(np.array([d]))[0])
               for d in dfcs if float(g(np.array([d]))[0])>0]
        print(f"{name:15s} width={w_ghz:5d}GHz  max|rel|={max(rel):.3e}")
PY
```

### A3 — §2.2 槽数公式（**已实测输出 `4 8 3 5`**）

```bash
$PY - <<'PY'
import math
SLOT, bps = 12.5e9, {"QPSK":2, "16QAM":4, "64QAM":6}
for rate_g, fmt, ro in [(100,"QPSK",0.0), (400,"16QAM",0.0),
                        (200,"64QAM",0.0), (100,"QPSK",0.1)]:
    n = max(1, math.ceil((rate_g*1e9/bps[fmt])*(1+ro)/SLOT))
    print(f"{rate_g:4d}Gbps {fmt:5s} roll_off={ro} -> {n:2d} slots = {n*SLOT/1e9:.1f} GHz")
PY
```

**输出**（实测）：

```
 100Gbps QPSK  roll_off=0.0 ->  4 slots = 50.0 GHz
 400Gbps 16QAM roll_off=0.0 ->  8 slots = 100.0 GHz
 200Gbps 64QAM roll_off=0.0 ->  3 slots = 37.5 GHz
 100Gbps QPSK  roll_off=0.1 ->  5 slots = 62.5 GHz
```

### A4 — §2.1 频段覆盖按中心 vs 按边缘（**已实测：边缘方案缺 44.9 GHz，几何上不可行**）

```bash
$PY - <<'PY'
from network_nlo_eval.core.spectrum import SpectrumGrid
L_LOW, C_HIGH, N, SP, K = 184.4877e12, 195.9428e12, 230, 50e9, 4
g = SpectrumGrid(num_channels=N, channel_spacing_hz=SP, center_frequency_hz=190.2127e12)
lo = g.frequencies[0] - SP/2
band_w, need = C_HIGH - L_LOW, N*SP
print(f"available edge-to-edge : {band_w/1e12:.4f} THz")
print(f"230 x 50GHz need        : {need/1e12:.4f} THz   shortfall {(need-band_w)/1e9:.1f} GHz")
print(f"centre-to-centre need  : {(N-1)*SP/1e12:.4f} THz   spare {(band_w-(N-1)*SP)/1e9:.1f} GHz")
print(f"centres inside both     : {L_LOW <= g.frequencies[0] and g.frequencies[-1] <= C_HIGH}")
print(f"edge overhang  low/high: {(L_LOW-lo)/1e9:.1f} / {(lo+N*K*12.5e9-C_HIGH)/1e9:.1f} GHz")
lo2 = L_LOW
print(f"re-anchor low to L_LOW  : all centres shift +{(lo2-lo)/1e9:.1f} GHz (AC-1 broken); "
      f"upper edge still +{(lo2+need-C_HIGH)/1e9:.1f} GHz")
PY
```

**输出**（实测）：

```
available edge-to-edge : 11.4551 THz
230 x 50GHz need        : 11.5000 THz   shortfall 44.9 GHz
centre-to-centre need  : 11.4500 THz   spare 5.1 GHz
centres inside both     : True
edge overhang  low/high: 25.0 / 19.9 GHz
re-anchor low to L_LOW  : all centres shift +25.0 GHz (AC-1 broken); upper edge still +44.9 GHz
```

### A5 — §2.8 算力（**已实测：O(N²) 在 load 1.00 与 load 21.30 下均成立；绝对值随负载变，勿单独引用**）

```bash
echo "loadavg: $(cat /proc/loadavg)"      # 必须与计时同时打印
$PY - <<'PY'
import time, numpy as np
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.numba_kernels import _calc_span_noise_and_power_jit
from network_nlo_eval.network.elements import FiberSpanConfig
fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
for n in [115, 230, 417, 830]:
    ctr = 190.2127e12 if n <= 230 else 194.8877e12
    ev = MultiBandISRSGN(grid=SpectrumGrid(n, 50e9, ctr),
                         ref_fiber_config=fiber, symbol_rate_hz=50e9)
    a = dict(power_in_w=np.full(n,1e-3), span_length_m=100e3, f_rel_hz=ev.f_rel_hz,
             f_abs_hz=ev.f_abs_hz, lambdas_m=ev.lambdas_m,
             alpha_power_npm=ev.alpha_power_npm_channels, a_eff_m2=ev.a_eff_m2_channels,
             beta2_s2_m=ev.beta2_s2_m, beta3_s3_m=ev.beta3_s3_m, n2_m2_w=ev.n2_m2_w,
             nf_lin_channels=ev.nf_lin_channels, symbol_rate_hz=50e9,
             n_effective_spans=1, cr_w_m_hz=ev.cr_w_m_hz, delta_f_co_hz=ev.delta_f_co_hz)
    _calc_span_noise_and_power_jit(**a)                      # 预热/编译
    ts = [(lambda t0: (_calc_span_noise_and_power_jit(**a), time.perf_counter()-t0)[1])(
          time.perf_counter()) for _ in range(5)]
    print(f"N={n:4d}  {np.median(ts)*1e3:8.2f} ms")
PY
```

### A6 — `g()` 转写校验（`numba_kernels.py:182-195` 的转写 vs 真实 njit 内核）

```bash
$PY - <<'PY'
import numpy as np
from network_nlo_eval.models.isrs_gn import _SRS_CR
from network_nlo_eval.models.numba_kernels import _calc_exact_power_profile_ode
for df in [1.0, 6.0, 11.45]:
    f = np.array([190.0e12, (190.0+df)*1e12]); p = np.array([1e-3, 1e-3])
    o = _calc_exact_power_profile_ode(p, f, np.zeros(2), 1.0, _SRS_CR, n_steps=20)
    G = (o[0]/p[0]-1.0)/(p[1]*1.0)
    print(f"dF={df:5.2f}THz  G_ode={G:.6e}  Cr*dF={_SRS_CR*df*1e12:.6e}  rel={abs(G-_SRS_CR*df*1e12)/(_SRS_CR*df*1e12):.2e}")
PY
# 实测 rel ~ 1e-8（RK4 离散残差），确认 §2.4 的转写与内核一致
```

---

## 变更记录

| 版本 | 内容 |
|---|---|
| v1（本文） | 初版冻结候选。8 问作答 + 冻结接口 + 迁移清单 + 验收判据 + 8 项需定夺 + 4 项未测量。 |
