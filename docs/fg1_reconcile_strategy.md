# FG-1 协调策略：把 ecz-dev 的路径级管线移植到 main 的 ODE 物理上

- worktree：`/home/zsig/Documents/gits/wt/fg-reconcile`，分支 `wp/fg-reconcile`
- 基线：`origin/main` = `87850a1`（好物理），参照：`origin/ecz-dev` = `70f851d`（好管线）
- merge-base：`8729e63`
- 本轮**不**做：灵活栅格（FG-2）、`multiband-optical-rl`（FG-3）、推 `origin`

---

## 0. 一句话结论

两个分支**同源不同物理实现**，不是"同一份代码的两个版本"：

- main 的内核是 **RK4 多波段 ODE + 峰 Kurtosis 修正**，功率演化是**数值积分**出来的。
- ecz-dev 的内核是 **解析 SRS 倾斜模型**（`r_f` 多项式 + 指数出纤功率），功率演化是**闭式**算出来的。

因此本轮的物理裁决很明确：**`models/numba_kernels.py` 整文件取 main，一个字节都不改**；
ecz-dev 的管线在其上跑。分歧点只有 2 个文件（`numba_kernels.py`、`models/isrs_gn.py`），
其余全是可加性（additive）改动。

---

## 1. 逐文件映射表

| 文件 | main 侧现状 | ecz-dev 侧现状 | 采用 | 为什么 | 谁调用谁 |
|---|---|---|---|---|---|
| `models/numba_kernels.py` | `_calc_dp_dz` + `_calc_exact_power_profile_ode`（RK4，`n_steps=20`）；`_calc_spm_variance_jit(t_i_tilde, L_m)`；`_calc_xpm_variance_term_jit(phi_kurtosis=...)`；`_compute_nli_variances(..., p_out_exact_w, ..., phi_kurtosis)`；`_calc_span_noise_and_power_jit(..., phi_kurtosis)` 内部调 ODE 并返回 ASE | 用 `_calc_raman_profile_jit` + `_calc_span_out_power_jit`（解析 `r_f`/`L_eff`）**替换**了 ODE；SPM/XPM 签名换成 `(t_i, r_fi, cr, k_bar, l_eff)`；`_compute_nli_variances` 收 `t_factors/r_f_values/l_eff_m/cr/k_bar` 而非 `p_out_exact_w`；**无 kurtosis**；内核返回的 ASE 被调用方丢弃 | **main（整文件）** | brief 明确禁止用旧 inline 演化替换 ODE；且 kurtosis（64QAM 修正）只在 main 存在，是物理增量。两侧 SPM/XPM 公式**不自洽混用**：换掉功率演化就必须连 SPM/XPM 一起换，否则 `t_tilde` 反演失去物理意义 | 被 `models/isrs_gn.py` 调用 |
| `models/isrs_gn.py` | `MultiSpanOpticalPath`（无校验、`link_keys=None`）；`MultiBandISRSGN(grid, fiber, rs, phi_kurtosis)`；`_calc_span_noise_and_power` **转发 main 内核**（含 `phi_kurtosis`，用内核返回的 ASE，ROADM 只除 `insertion_loss+filtering_penalty`） | 新增 `PathQoTResult`、`MultiSpanOpticalPath.__post_init__`/`link_keys`/`node_path`、`for_fiber` 缓存、`evaluate_path` 跨段累积；`__init__` 去掉 `phi_kurtosis`、加 `k_bar`；`_calc_span_noise_and_power` 改调解析内核并在 **Python 层显式应用 EDFA 增益 + 自算 ASE + ROADM booster/均衡** | **合并**：管线/数据类/`for_fiber`/`evaluate_path` 取 ecz-dev；`_calc_span_noise_and_power` 的**物理与内核调用**取 main；`phi_kurtosis` 保留 | 管线是纯增量的；物理必须留在 main。冲突点：EDFA 增益/ASE 归谁算（见 §3 冲突 C1） | 被 `physics/qot_checker.py`、`physics/comparison.py`、`rwa/qot_checker.py`、`__main__.py` 调用 |
| `network/elements.py` | `FiberSpanConfig`/`EDFAConfig`/`ROADMConfig` 基础字段；`EDFAConfig.target_gain_db=20.0`（必填默认）、ROADM 仅 2 个损耗字段 | 新增 `max_span_length_km`、`use_attenuation_polynomial`、`raman_pumps`、`RamanPumpConfig`；`target_gain_db: float\|None`；NF/纹波校验器；ROADM 扩展为 WSS/booster/blocked + `path_loss_db()` | **ecz-dev** | 纯增量参数与校验；`__main__` 与新测试依赖 `path_loss_db`/`blocked_channels`。注意：`attenuation_db_km_ref` 与多项式默认值在参考波长处**数值一致**（0.2 dB/km），故默认值下两条路径同解 | 被 `topology.py`、`isrs_gn.py`、`qot_checker.py` 使用 |
| `network/topology.py` | 只按 `weight` 覆盖 `length_km`；EDFA/ROADM 只用节点默认值；无路径 API | 链路级 fiber/edfa 覆盖、`get_link_edfa_config`、`get_network_graph`、`build_optical_path(path)`、`export_to_dict` 导出物理参数 | **ecz-dev** | `build_optical_path` 是拓扑→物理路径的唯一入口，两个引擎都依赖它；纯增量 | 被 `rwa/*`、`physics/*`、`__main__` 调用 |
| `physics/qot_checker.py` | 只有 `PyNLOQoTEvaluator` 的单链路旧 API（`run_through_fiber`） | 新增 `_load_physical_layer`/`_fiber_dispersion`/`_edfa_values`/`propagate_path`/`_receiver_metrics`/`evaluate_path`/`evaluate_topology_path` | **ecz-dev** | 波形引擎与统计内核**正交**：它调外部 `physics_nlo_eval`（本机缺失→测试 skip），只从本仓取 `_calc_beta2_from_d`/`_calc_beta3_from_s_d`/`_w_to_dbm` 三个纯单位换算函数，**这三个在 main 中存在且未被改动** | 独立；经 `comparison.py` 与统计侧并列 |
| `physics/comparison.py` | 不存在 | 新增 `QoTComparisonResult`/`TopologyQoTComparator` | **ecz-dev** | 纯新增；只做两引擎结果序列化对比 | 调 `isrs_gn.evaluate_path` + `topology.build_optical_path` |
| `core/spectrum.py` | 无校验 | 仅新增 `__post_init__` 校验（栅格语义不变） | **ecz-dev**（仅此 hunk） | brief 明确许可"纯改进"；**不引入 slot、不改栅格语义**（FG-2 范围） | 被所有网格使用者 |
| `physics/metrics.py` | `SNR = 1/EVM² - 1`，EVM=0 返回 40 dB | `SNR = 1/EVM²`，下限 −20 dB，EVM=0 返回 inf | **ecz-dev** | 修正 EVM-SNR 关系与哨兵值；波形侧与统计侧共用 | 被 `qot_checker.py` |
| `rwa/path_computation.py` | 直接摸 `._network_raw`；`except Exception` 打印 | 用 `get_network_graph()`；窄化异常为 `warnings.warn` | **ecz-dev** | 配套 topology 的只读访问器；不再吞异常 | 被 `rwa/allocators.py` |
| `rwa/qot_checker.py` | `fixed_span_length_km=100.0` 固定；ROADM 从**节点**取；不走 `for_fiber` | `fixed_span_length_km=None`→用 `fiber.max_span_length_km`；EDFA 从**链路**取；`for_fiber` 缓存；ROADM booster/均衡/阻断进入累积 | **ecz-dev** | 链路级参数化是管线的一部分；但其 ROADM 分支与 C1 同源，需按 §3 裁决对齐 main 物理 | 调 `isrs_gn` 与 `topology` |
| `__main__.py` | 内联默认配置、`except Exception`、`simulate` 单命令 | `_load_topology` 提取、窄化异常、新增 `compare` 命令 | **ecz-dev** | CLI 是管线出口 | 调上面全部 |
| `assets/example_physical_topology.json` | 不存在 | 新增示例拓扑 | **ecz-dev** | `compare` 命令的输入样例 | — |
| `tests/test_topology_physical_integration.py` | 不存在 | 新增 5 passed/1 skipped | **ecz-dev** | 交付物 4 | 全部 |
| `README.md` / `.gitignore` / 删除 `.github/`、`.claude/` | 有 CI 与 agent 配置 | 文档更新 + 仓库清理 | **ecz-dev** | 清理按 brief 照做 | — |

---

## 2. 接口契约：`physics/qot_checker.py` ↔ `models/numba_kernels.py`

这是本轮最容易出错的地方，单独写清。

### 2.1 层次与调用方向

```
physics/qot_checker.py  (PyNLOQoTEvaluator)  ── 波形引擎
        │  只取 3 个纯函数：_calc_beta2_from_d, _calc_beta3_from_s_d, _w_to_dbm
        │  ※ 不触碰功率演化/噪声内核
        ▼
physics/comparison.py   (TopologyQoTComparator)  ── 顶层编排
        ▼
models/isrs_gn.py       (MultiBandISRSGN.evaluate_path)  ── 路径级管线
        │  _calc_span_noise_and_power  ← 唯一物理入口
        ▼
models/numba_kernels.py (_calc_span_noise_and_power_jit)  ← 物理内核，njit
        ├─ _calc_exact_power_profile_ode ──► _calc_dp_dz   (RK4 × n_steps=20)
        ├─ _compute_nli_variances ──► _calc_spm_variance_jit
        │                        └─► _calc_xpm_variance_term_jit
        └─ _calculate_ase_noise_variance
```

**关键约束**：`physics/qot_checker.py` 永远不直接 import 功率演化内核。
它与统计侧只通过 `MultiSpanOpticalPath`（数据）和 `comparison.py`（结果）耦合。

### 2.2 `_calc_span_noise_and_power_jit` 签名（main 版，本轮不改）

```python
_calc_span_noise_and_power_jit(
    power_in_w:      (N,) f64,   # 各信道入纤功率  [W]
    span_length_m:   float,      # 跨段长度        [m]
    f_rel_hz:        (N,) f64,   # 相对中心频率    [Hz]
    f_abs_hz:        (N,) f64,   # 绝对频率        [Hz]
    lambdas_m:       (N,) f64,   # 波长            [m]
    alpha_power_npm: (N,) f64,   # 功率衰减系数    [Np/m]
    a_eff_m2:        (N,) f64,   # 有效模面积      [m^2]
    beta2_s2_m:      float,      # 色散 β2         [s^2/m]
    beta3_s3_m:      float,      # 色散 β3         [s^3/m]
    n2_m2_w:         float,      # 非线性折射率    [m^2/W]
    nf_lin_channels: (N,) f64,   # 线性噪声系数    [-] = 10^(NF_dB/10)
    symbol_rate_hz:  float,      # 符号速率        [Hz]
    n_effective_spans: int,      # NLI 累积跨段数  [-]
    cr_w_m_hz:       float,      # SRS 增益系数    [1/(W·m·Hz)]
    delta_f_co_hz:   float,      # SRS 截断频率    [Hz]
    phi_kurtosis:    float = 0.0,# 0.0 高斯；-0.6190 64QAM
) -> (p_out_after_fiber, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w)   # 各 (N,) f64
```

### 2.3 长度与单位约定

- **一切按信道索引对齐**：`N == grid.num_channels`；索引 `i` 在 `power_in_w`、`f_rel_hz`、`f_abs_hz`、`lambdas_m`、`alpha_power_npm`、`a_eff_m2`、`nf_lin_channels`、两个 `sigma2_*`、`sigma2_ase`、`p_out` 中含义一致。**任何一侧都不得重排或截断这些数组。**
- **功率**：`power_in_w` / `p_out` / `sigma2_*_w` 全部是 **W（线性）**，不是 dBm。dBm 只在 `__main__` 边界用 `_dbm_to_w`/`_w_to_dbm` 转换。`sigma2_*` 是**方差**（W² 量纲的噪声功率），不是标准差。
- **频率**：`f_abs_hz` 与 `f_rel_hz` **同时**传入且长度相同；ODE 内部用 `f_abs_hz` 建拉曼增益矩阵（只用频差，`f_rel_hz` 数值上亦可，但契约固定为 `f_abs_hz`）；NLI 相位用 `f_rel_hz`。`f_rel_hz = f_abs_hz - grid.center_frequency_hz`。
- **零功率信道**：`p_in_w[i] <= 0` 的信道跳过 NLI，且**不得**产生 ASE（`_calculate_ase_noise_variance` 内有 `p_in_w[i] > 0 and p_out_w[i] > 0` 守卫）。管线必须保持这一约定，否则 SNR 会出现 `0/0`。
- **`n_effective_spans`**：NLI 累积因子。路径级管线逐跨段调用时**恒传 1**，累积由 Python 侧完成；只有整条链路当一跨段算时才 >1。

### 2.4 ODE 契约（main 独有，移植方向的关键）

```python
_calc_exact_power_profile_ode(p_in_w, f_hz, alpha_npm, L_m, cr_w_m_hz, n_steps=20) -> (N,) W
_calc_dp_dz(p, G, f_hz, alpha_npm) -> (N,) W/m
```

- `G` 是 `(N, N)` 拉曼增益矩阵 `[1/(W·m·Hz)]`，由 `cr_w_m_hz` × 谱形（13.2 THz 线性增长、15.6 THz 归零）构造。
- `_calc_dp_dz` 内**能量守恒**：`f_hz[k] < f_hz[i]` 时损失项按 `f_hz[k]/f_hz[i]` 加权。
- RK4 每步后**钳位非负**。
- `n_steps=20` 固定，不随跨段长度自适应。

### 2.5 ODE → NLI 的耦合（移植时最容易悄悄破坏的地方）

`_compute_nli_variances` **不做** SRS 前向建模，而是**反演**一个等效 `t_tilde[i]`，
使 GN 解析积分的前后功率与 ODE 的前后功率对齐：

```python
p_lin_out = p_in_w[i] * exp(-alpha[i] * L)
t_tilde[i] = (p_out_exact_w[i] / p_lin_out - 1) / (1 - exp(-alpha[i] * L))
```

**因此 `p_out_exact_w` 必须是同一 span、同一 `p_in_w`、同一 `alpha`/`L` 下 ODE 的输出。**
任何"用别的功率演化喂进这个函数"的改动都会让 `t_tilde` 失去意义且偏差不可解释。
ecz-dev 的 `t_factors = (2α − Cr·r_f)²` 是**另一套**参数化，与本函数不兼容，不可混用。

---

## 3. 冲突裁决

### C1（唯一实质冲突）：EDFA 增益与 ASE 由谁计算

- **ecz-dev**：`isrs_gn._calc_span_noise_and_power` 丢弃内核返回的 ASE，在 **Python 层**显式施加
  EDFA 增益（`target_gain_db` 或自动补偿）并自算 `ASE = NF·h·f·R_s·(G−1)`；
  `rwa/qot_checker` 里 ROADM 还带 booster 均衡（`equalize_output_power`）。
- **main**：ASE 由内核 `_calculate_ase_noise_variance` 算，**用纤损反推的增益**
  `G = p_in/p_out_fiber`，并乘 `n_span`；EDFA 增益不进入功率路径；ROADM 只做一次线性衰减。

**裁决：取 main。** 理由：
1. 验收标准是"与 `origin/main` 的黄金输出逐点比对，偏差为 0"。改 ASE 归属必然改变数值。
2. brief 规定物理内核单向保留 main；ASE 属于内核输出的一部分。
3. 显式 EDFA 增益模型是**另一套物理假设**（`G` 由配置而非纤损决定），不是管线层适配。

**代价（如实记录）**：`ROADMConfig.equalize_output_power` / `booster_noise_figure_db`
/ `blocked_channels` / `raman_pumps` / `use_attenuation_polynomial` 这些字段在**统计内核**中
**不起作用**。其中：
- `blocked_channels`：`evaluate_path` 仍**执行**阻断置零（纯门控，零功率化，不改噪声公式），
  与统计数值无关的部分是安全的。
- `booster_*` / `equalize_output_power` / `raman_pumps` / `use_attenuation_polynomial`：
  **仅波形引擎**（`physics/qot_checker.py`）使用；统计侧保留字段但不消费。
  这是本轮**明确的功能取舍**，不是遗漏。

### C2：`for_fiber` 缓存 vs `phi_kurtosis`

`for_fiber` 以 `fiber_config.model_dump_json()` 为键新建 `MultiBandISRSGN`。
必须保证 `phi_kurtosis` 与 `symbol_rate_hz` 随实例传递，不能被缓存键吞掉。
`for_fiber` 用 `self.symbol_rate_hz` 构造新实例（与 `self` 同值），`phi_kurtosis` 亦须显式透传。

### C3：`fixed_span_length_km`

ecz-dev 改为 `None`→`fiber.max_span_length_km`。`rwa/qot_checker` 与 `isrs_gn.evaluate_path`
必须用**同一套跨段切分规则**，否则统计侧与拓扑侧路径不等价。本轮统一用
`while remaining_km > 1e-12: span_km = min(remaining_km, max_span_length_km)`。

---

## 4. 等价性证据设计（本 WP 核心验收）

本仓测试极弱（main 仅 1 个 `test_main.py`），**"测试绿"不构成验收**。证据链：

1. **黄金输出**：从 `origin/main` 侧用**固定场景**跑出功率演化曲线与噪声，存
   `results/fg1_golden_main.json`。场景参数（栅格、纤参、EDFA/ROADM、跨段序列、符号速率、
   `phi_kurtosis`）全部内联在场景定义里，可复算。
2. **同场景重算**：在 `wp/fg-reconcile` 上跑**同一个场景**：
   - (a) 经**低层** `_calc_span_noise_and_power` 逐跨段循环 → 应与 main **逐位相同**；
   - (b) 经**新管线** `evaluate_path` → 应与 (a) 逐位相同。
   (a) 证明内核未被改动；(b) 证明管线是物理中性的——这是本轮最关键的一条。
3. **落盘**：`results/fg1_physics_parity.json`，含场景参数、两侧数值、逐点最大绝对/相对偏差。
4. **进 CI**：`tests/test_fg1_physics_parity.py` 读黄金文件做断言，容差 `0.0`（严格相等），
   任一信道功率/噪声偏离即失败。

> 关于"改动内核"的说明：本轮**不改** `numba_kernels.py`。
> `isrs_gn.py` 只做**接口适配**（给内核传 `phi_kurtosis` 而非 `k_bar`；取内核返回的 ASE 而非丢弃），
> 适配后偏差必须为 0——这是 (a) 项直接验证的。

---

## 5. 提交顺序（逐文件、可审查）

1. `models/numba_kernels.py` — 保持 main 原样（**不产生 diff**，作为基线声明）
2. `core/spectrum.py` — 仅 `__post_init__`
3. `network/elements.py`、`network/topology.py` — 参数层 + 路径 API
4. `physics/metrics.py` — SNR 修正
5. `models/isrs_gn.py` — 管线 + main 物理（**核心提交**）
6. `physics/qot_checker.py`、`physics/comparison.py` — 波形侧 + 对比器
7. `rwa/qot_checker.py`、`rwa/path_computation.py` — RWA 侧对齐
8. `__main__.py`、`assets/`、`README.md`、`.gitignore`、仓库清理
9. `tests/test_topology_physical_integration.py`（移植）+ `tests/test_fg1_physics_parity.py`（新增）
10. `docs/fg1_reconcile_strategy.md`、`results/fg1_*.json`

## 6. 已知环境事实

- `physics_nlo_eval`（波形引擎）**本机未安装** → `test_waveform_route_smoke` 走
  `pytest.importorskip`，对应 brief 里的 **1 skipped**。波形路径本轮**未实测**。
- 测试命令必须带 `PYTHONPATH=$PWD/src`（editable 装的 `.pth` 指向主检出，与 cwd 无关）。

---

## 7. 实施结果（实施后回填）

### 7.1 等价性证据

黄金文件 `results/fg1_golden_main.json` 由 `git archive origin/main`（`87850a1`）导出的
**纯净 main 树**跑出，文件内记录了 `source_branch` / `source_commit` / `source_tree`。
场景定义 `tests/fg1_scenarios.py` **只用 main 上就存在的 API**，两侧跑的是同一份代码。

`results/fg1_physics_parity.json`（`parity_ok: true`，`failures: 0`，`tolerance: 0.0`）：

| 场景 | 信道数 | phi_kurtosis | 低层 vs main | 逐跨段曲线 vs main | `evaluate_path` vs 参考 |
|---|---|---|---|---|---|
| `c_band_single` | 8 | 0.0 | 逐位相同 | 4 点全同 | 逐位相同 |
| `multiband_wide` | 80 | 0.0 | 逐位相同 | 8 点全同 | 逐位相同 |
| `kurtosis_64qam` | 16 | −0.619 | 逐位相同 | 4 点全同 | 逐位相同 |
| `roadm_hoop` | 12 | 0.0 | 逐位相同 | 8 点全同 | 逐位相同 |

所有条目 `max_abs_dev = 0.0`、`max_rel_dev = 0.0`。

**核心结论**：`evaluate_path` 与 main 逐位相同 ⇒ 新管线是**物理中性**的编排层，
没有重复计入、没有重排跨段、没有改变噪声语义。这是本轮最关键的一条证据。

**非退化性**：`multiband_wide` 首跨段 SRS 功率倾斜 1.189×（`P∈[2.147e-5, 2.553e-5] W`），
说明 RK4 ODE 的信道间拉曼耦合确实被激发；`kurtosis_64qam` 对 `phi_kurtosis` 敏感
（测试 `test_kurtosis_scenario_is_sensitive_to_phi_kurtosis` 断言其 SNR 随该参数变化）。
因此这不是一组"恰好相等"的平凡数字。

**测试有牙**（负控制）：在临时副本里把 ODE 积分步数 `n_steps` 20→21，
8 条 parity 断言全部失败（`8 failed, 5 passed`）。

### 7.2 测试

`PYTHONPATH=$PWD/src python -m pytest -q` → **18 passed, 1 skipped**
（ecz-dev 的 5 passed/1 skipped + main 的 `test_main.py` + 12 条 FG-1 parity 断言）。

### 7.3 与计划的偏差（如实记录）

1. **`.github/` 与 `.claude/` 未删除**。brief 把"仓库清理"列在交付物 2，
   但同时要求 parity 测试"进 CI 集合"。删掉 `.github/workflows/` 就删掉了 CI 本身，
   二者直接冲突。删除不可逆（且本 WP 不允许推 `origin`），
   故**保留**并在此显式标出，**留给用户决定**。`README.md` 已按计划移植。
2. **`.gitignore` 未照搬 ecz-dev**。ecz-dev 把 `/results/` 改成 `results/`，
   但两者都会忽略整个 `results/` 目录，会把本 WP 的交付产物悄悄排除掉。
   改为保留 `/results/*` 并用否定规则显式放行两个 parity 产物。
3. **`rwa/qot_checker.py` 的 ROADM 分支被改写**（偏离照搬 ecz-dev）：
   原版用 `path_loss_db()` + booster 均衡并另加 booster ASE，与 §3 C1 冲突，
   且会与 `evaluate_path` 的行为不一致。已统一为 main 的规则
   （`insertion_loss + filtering_penalty` 施加于信号与已累积噪声，不引入新噪声），
   两处 ROADM 分支改法相同。
4. **修掉一个 main 侧的潜在 bug**：`edfa_config` 非空但 `noise_figure_db` 为 `None` 时
   `current_nf_lin` 未绑定。改为回落到 `nf_lin_channels`。
   这**只影响原本会抛 `NameError` 的输入**，对所有可达路径数值**零影响**
   （parity 逐位相同即为证）。
5. **C1 的功能代价（重申）**：`ROADMConfig.equalize_output_power` /
   `booster_noise_figure_db` / `raman_pumps` / `use_attenuation_polynomial`
   在**统计内核中不生效**，仅波形引擎使用；`blocked_channels` 在统计侧仍执行置零。
6. **波形路径未实测**：`physics_nlo_eval` 未安装，1 skipped。属环境限制，非本轮缺陷。

### 7.4 复算命令

```bash
cd /home/zsig/Documents/gits/wt/fg-reconcile
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python -c \
  "import network_nlo_eval as m; print(m.__file__)"   # 须落在本 worktree 内

# 重新生成本分支侧比对结果
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python scripts/fg1_parity.py \
  --check results/fg1_golden_main.json --out results/fg1_physics_parity.json

# 重新生成黄金（须在纯净 main 树上执行）
FG1_SOURCE_COMMIT=$(git rev-parse origin/main) \
PYTHONPATH=<main-tree>/src ~/miniconda3/envs/rl/bin/python scripts/fg1_parity.py \
  --emit-golden results/fg1_golden_main.json

PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python -m pytest -q
```
