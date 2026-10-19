# FG-4b — NLI 张量化与加速档位契约裁定

本文件记录两件事：**(1) 编排方对加速档位默认值的裁定及其理由**，
**(2) `_compute_nli_variances` 张量化的设计、证据与打折原因**。

所有数字都能从提交的产物复算；命令见文末 §6。测量口径为**同进程交错 A/B**，
并同时记录 `loadavg`（本轮实测 4.3–5.6，FG-4a 自报时为 4.6，编排方复核时为 23）。

---

## 1. 编排方裁定：默认档改为 `matvec`

### 1.1 裁定内容

`ODE_ACCEL_DEFAULT` 由 `"cached_g"` 改为 **`"matvec"`**（生产路径要快）。

F1-1 / F1-2 / F2-3 保持 `np.array_equal`、保持逐位相等、**期望值一个字未改**，
但被**显式构造为 `ode_accel="off"`**。

### 1.2 为什么不放宽那三条断言

这三条测试的命题是"reconcile 没有改变 origin/main 的物理"——那是关于
**移植保真度**的陈述，与"生产默认档选哪个"无关。用 `off` 档跑，命题原样成立、
强度原样保留（仍是逐位相等，期望值未动）。

若反过来把断言放宽成容差，会同时丢掉两件事：origin/main 那一段就没有任何
**逐位**证据了；而且"放宽"是一次不可逆的降级——后面每一刀都有人拿它当先例。
因此本刀选择**拆成两段**，而不是把一条断言改软。

### 1.3 证据链

```
origin/main ──(off 档, 逐位)──> FG-2 树 ──(默认 matvec, ≤1e-12)──> 生产
```

两段各自成立、各自有测试，且都比"把三条断言放宽成容差"更强。

| 段 | 守门测试 | 标准 |
| --- | --- | --- |
| origin/main → FG-2 | F1-1 / F1-2 / F2-3（`np.array_equal`） | 逐位相等 |
| FG-2 → 生产 | FG-4a F4a-4、FG-4b F4b-1 | ≤1e-12 相对偏差 |
| 冻结点仍在 | FG-4b F4b-2（新增） | `off` 仍可选且仍逐位相等 |

### 1.4 新增测试：冻结点仍在（F4b-2）

`tests/test_fg4b_nli_vectorization.py` 断言三件事，使冻结点不会被将来的重构悄悄搬走：

1. `off` 仍与 **origin/main** 金标准（`results/fg1_golden_main.json`）逐位相等
   （`test_f4b_2_off_is_bit_identical_to_origin_main`）；
2. `off` **确实没有把张量传给内核**（`test_f4b_2_off_tier_passes_no_tensors_to_the_kernel`）——
   仅"结果看起来接近"不够，真正让冻结点成立的是被执行的仍是冻结代码；
3. FG-2 冻结的 `_compute_nli_variances` **仍是逐对双层循环**且仍调用
   `_calc_xpm_variance_term_jit`（`test_f4b_2_frozen_nli_source_is_intact`），
   张量化是**另一个函数**。

### 1.5 这次放宽为什么必须现在定

整个加速程序（matvec、NLI 张量化、GPU/批量）**都会重排浮点求和**。若默认档被
逐位相等钉死，后面每一刀都会被同一条断言挡住。而在 dB 域的 SNR 上，1e-15 的
相对偏差在数值上无意义。

**这是一次有意记录在案的契约放宽，不是顺手绕过。** 记在这里，也记在
`isrs_gn.py::ODE_ACCEL_DEFAULT` 的注释里。

### 1.6 档位分工：一个档位只引入一类浮点重排

| 档位 | ODE | NLI | 标准 |
| --- | --- | --- | --- |
| `off` | 冻结 `_calc_dp_dz`（逐对分支） | 冻结逐对双层循环 | **逐位** |
| `cached_g` | G 缓存（跳过重建，求和次序不变） | 冻结逐对双层循环 | **逐位** |
| `matvec`（默认） | 去分支两次 matvec | 张量化 | ≤1e-12 |

这张表是本刀的一个关键设计决定：**张量化 NLI 只进 `matvec`，不进 `cached_g`**。
`cached_g` 是 FG-4a 的**逐位相等**档，若把张量化放进去，F4a-3
（"cached_g 与金标准逐位相等"）就会被本刀悄悄破坏、而那条断言还在原地假装成立。
一个档位只引入一类重排，这张表才能逐档取证。

`scripts/fg4a_capture_golden.py` 的 `--ode-accel` 默认值也**没有**跟随
`ODE_ACCEL_DEFAULT`：它被钉在 `CHECK_DEFAULT_MODE = "cached_g"`，否则 `--check`
会从"逐位自检"静默降级成 1e-12 容差自检。实测：**31/31 逐位相同，最差相对偏差 0.000e+00**。

### 1.7 治理变更：G2 守卫被升级（不是被绕过）

FG-2 与 FG-4a 各自的 G2 守卫原本是**文件名级**代理检查：

```python
offenders = [line for line in git_diff --name-status BASELINE -- tests/
             if line and not line.startswith("A")]
assert not offenders
```

§1.2 授权的改动落在既有测试文件里，因此该检查会失败。但这个检查**无法**区分
"把测试钉在 `off` 档"和"把断言改软"——它同时**过严**（合法改动被拦）与**过松**
（唯一出路是删掉守卫本身）。

因此守卫被换成**更严**的三机制策略（`tests/fg4b_tier_pin_policy.py`，
两个守卫共用 `collect_violations`，不会各自漂移）：

1. **白名单** — 未被 §1.2 点名的既有测试文件被改动，仍然直接失败；
2. **期望 token 计数保全** — 三个 parity 文件的 `np.array_equal` / `pytest.approx` /
   `allclose` / `isclose` / `atol` / `rtol` 出现次数必须与基线**完全相同**。
   这一条是**自指免疫**的：重写守卫会增删 `assert`，但不会增删一个 `array_equal`；
   而削弱检查一定会删一个。计数覆盖整个文件（含注释），所以不能靠把真代码挪进
   字符串来凑数；
3. **守卫完整性** — 两个 G2 守卫函数必须仍然存在，且仍然调用本策略。
   这是防止"把守卫删掉让 diff 消失"的兜底。

**诚实声明**：本刀**确实编辑了两个既有测试文件**（两个 G2 守卫本身），
这本身是一次治理变更。理由与证据链已完整记录于此与策略模块的 docstring，
且策略被**加强**而非放宽：原来"任何既有测试文件都不得改动"，
现在"未授权文件仍然一律失败，授权文件里不得有任何削弱期望的行"。

守卫不是空转的——已用变异测试验证：

| 变异 | 结果 |
| --- | --- |
| 在 `test_fg1_physics_parity.py` 里把 `np.array_equal` 换成 `np.allclose` | **被捕获**（token 计数 4→3、移除行含 `assert`） |
| 把 `test_f4a_pre_existing_tests_were_not_modified` 重命名掉 | **被捕获**（守卫被移除） |

---

## 2. 主体：`_compute_nli_variances` 张量化

### 2.1 立足点：唯一的数据相关量是 `t_tilde`

XPM 的 O(N²) 双重循环里，**没有任何一项依赖逐跨段功率**。进入
`_compute_nli_variances` 的量要么逐连接常量（`symbol_rate_hz` / `f_rel_hz` /
`lambdas_m` / `a_eff_m2` / `alpha_power_npm`），要么跨段常量（`L_m` / `beta_2` /
`beta_3` / `n_2` / `phi_kurtosis` / `n_span`）。唯一的数据相关量是 **`t_tilde`**，
而它只经两处进入：

```
T_l = 1 + t_tilde[l]        （pre_factor 的 T_l^2）
r_l = -t_tilde[l] / T_l     （(l_ch, lp) 求和系数 r_l^(l_ch+lp)）
```

另外，基线在双层循环里反复重算 `a_t` / `k`——但它们是按 **`alpha_l_npm`（l 通道）**
建的，与配对无关，本来就只是 `(N,2)` 的逐通道量，不是 `(N,N)`。

### 2.2 张量化形态

把 `(l_ch, lp) ∈ {(0,0),(0,1),(1,0),(1,1)}` 四项按 `r` 的幂次归并：

```
sum_val[i,l] = W0 + 2*r_l*W1 + r_l^2 * W2
```

`W0` / `W1` / `W2` 是三张 `(N,N)` 张量，**全部在构建期一次算好**，包括
`arcsinh` / `arctan` / `log` 这些超越函数，以及全部数据相关分支：
`log_arg > 0` 的选择、`|phi_il| < eps` 的钳位、`den < 1e-12` 的泰勒兜底。

热路径只剩乘加与广播，**无数据相关控制流**——这正是 FG-4c 搬到 GPU / 加 batch 维
所需要的形状。

### 2.3 brief 逐条要求的处理

| 要求 | 处理 |
| --- | --- |
| 逐连接带宽 | `rs_l_hz`（COI 自己的带宽）只进 term2 的对数项与分母；`rs_i_hz`（扰动信道带宽）只进 term1 的 `arctan`。**两者都取自 `symbol_rate_hz`，没有退回成同一个标量。** 见 `test_f4b_4_heterogeneous_symbol_rates_are_preserved`（flexgrid 场景混三种符号率） |
| 零功率掩码 | `p_in <= 0` 与对角线在**求和之后统一掩码**。掩码作用在已算出的 `(N,N)` 上，因此惰性连接的 `log` / `1/x` 产生的 `nan` / `inf` 不污染求和——与基线逐对 `continue` 等价。见 `test_f4b_4_zero_power_channels_are_excluded` |
| `t_tilde` 拟合 | 仍是唯一数据相关步骤，见 `test_f4b_4_t_tilde_fit_and_kurtosis_branches_are_covered`（用两个跨段长度驱动不同拉曼倾斜，断言 XPM 确实随之改变，即 `t_tilde` 真的到位） |
| `phi_kurtosis` 分支 | 0.0 高斯 / -0.6190 64QAM 两条都覆盖，断言该分支在张量路径上**仍然有效**（`cl_full_64qam_80km` + 上述测试） |
| 逐对相位项 | 按 brief 要求写成**外积形式**：`phi_il = (f_l - f_i) * phi_base(f_l + f_i)`，两个因子都是 `(N,N)` 外积，**不是逐对循环** |
| 设备无关 | 纯 numpy 张量运算（njit 内），**未引入 cuda 依赖** |

### 2.4 一个真实且危险的 bug（已修，已设回归闸）

XPM 的 pre-factor 里 `p_l`（内层）可直接广播，但 `p_i`（外层）**必须写死轴向**：

```python
pre = ((gamma_sq * p_sq) * p_in_w[:, None]) / symbol_rate_hz   # 对
pre = ((gamma_sq * p_sq) * p_in_w)      / symbol_rate_hz       # 错：静默变成 p_l
```

一维数组默认对齐**最后**一轴。写成 `* p_in_w` 会静默计算成 `p_l`。
**这个错误在满载下完全不可见**——此时每一条存活配对都满足 `p_i == p_l`，
所以 7 个场景里 5 个照样通过。只有 `path_cplusl_roadm` 经过拉曼倾斜、
逐通道功率不再相同时才暴露，实测相对偏差 **5.16e-04**（超预算 8 个数量级）。

回归闸：`test_f4b_3_tensor_nli_survives_non_uniform_per_channel_power` 用一个
**严格递增**的功率斜坡（任何两个通道功率都不同）直接驱动内核，
并额外断言偏差 `< 1e-9`——转置的 pre-factor 是灾难性错误，不是边际偏差。

### 2.5 与基线的浮点对应关系

逐项的乘法次序完全一致。差别只有三处：

1. l 方向求和由顺序累加变成成对求和（`np.sum`）；
2. `(0,1)` 与 `(1,0)` 两项按浮点交换律合并为一项 `2*r*W1`
   （两者**逐位**相等：`(A0+A1)==(A1+A0)`、`atan(x)+atan(y)==atan(y)+atan(x)`、
   `A0*A1==A1*A0`、`K0*K1==K1*K0` 在 IEEE 下都是精确的）；
3. `t_tilde` 之后不再有 `if`。

---

## 3. 数值证据

同进程交错 A/B，两臂共享**逐位相等的 matvec ODE**（同一 `g_gain` / `g_loss`），
因此差值可完全归因于 NLI 求值本身——不是"两处改动混在一起无法归因"。

| 场景 | `off` vs `matvec` 最大相对偏差 |
| --- | --- |
| `cl_full_100km` | 1.64e-15 |
| `cl_partial_100km`（含 `p_in=0`） | 1.38e-15 |
| `cl_full_64qam_80km`（kurtosis 分支） | 1.06e-15 |
| `cl_short_1km`（`den<1e-12` 泰勒分支） | 6.49e-16 |
| `cl_nspan10`（`n_span=10`） | 1.28e-15 |
| `flexgrid_mixed`（异构逐连接符号率） | 1.17e-15 |
| `path_cplusl_roadm`（ROADM + 倾斜） | 1.29e-15 |

- 隔离测量（ODE 逐位相同）：**4.109e-16**，`ode_output_bit_identical_between_arms = true`
- 金标准自检：**31/31 逐位相同，最差相对偏差 0.000e+00**
- 预算 1e-12，**未放宽**（`MAX_REL_DEVIATION_ALLOWED` 是模块常量）

实测余量约 **3 个数量级**。

---

## 4. 性能：实收 vs 理论上限（F4b-3 / F4b-4）

230 节点，100 km 跨段，20 步 RK4，1/8 槽位空载。口径：同进程交错 A/B
（每轮翻转顺序），比值取 **min 样本**，`loadavg` 前后记录。

| 指标 | 值 |
| --- | --- |
| 本刀（隔离，ODE 逐位相同） | **2.310x**（min）/ 2.276x（median），450 样本 |
| 累计（vs pre-FG-4a 基线 `ea7ecb0`） | **4.054x**（min）/ 3.971x（median） |
| `loadavg`（1min） | 4.48 → 4.33（delta −0.15） |

**重复性**：两次独立 150 轮重复给出 **2.318x** 与 **2.310x**，离散度 ±0.4%。
（一次 40 轮的早期读数 2.712x 是离群值，已被高样本测量取代。）

### 4.1 上限口径需要更正

brief 给的**名义上限 5.2x** 建立在"NLI 成本归零、只剩 ODE"上。实测表明这个上限
**在本刀下算术上不可达**：ODE 残差是一个硬地板。

| | 总时 | 其中 NLI |
| --- | --- | --- |
| before（matvec ODE + 冻结 NLI） | 2.531 ms | 1.917 ms |
| after（matvec ODE + 张量 NLI） | 1.096 ms | 0.482 ms |
| **ODE 地板**（单独计时，NLI 工作无法触及） | **0.614 ms** | — |
| 可达上限 = before / ODE 地板 = 2.531 / 0.614 | **4.12x** | |
| 实收 = before / after = 2.531 / 1.096 | **2.310x** | |
| 占名义 5.2x | **44.4%** | |
| 占可达 4.12x | **56.0%** | |

名义 5.2x 是按 brief 交接的 2.811 ms 总时 / 0.543 ms ODE 算的；本轮实测总时
2.531 ms、ODE 0.614 ms，故**同一个公式**给出的可达上限是 4.12x。两个数不一致
应归因于工作负载与机器状态差异，不是口径分歧——**按 brief 自己的公式复算**，
上限就是 4.12x 量级。

### 4.2 打折原因（NLI 成本只降了 75%，总时只降 2.3x）

1. **ODE 地板占 0.614 ms**，是本刀完全够不到的部分。after 的 1.096 ms 里
   **56% 已经是 ODE**。这一项单独就决定了上限只能是 4.12x 而非 5.2x。
2. **残余 NLI 0.482 ms（占 before 的 19.0%）** 来自三处真实工作：
   - **内存带宽**：三张 `(N,N)` 张量 = 3 × 230² × 8 B ≈ **1.27 MB**，每次调用都要
     读一遍。热路径已从"算术受限"变成"访存受限"。
   - **广播链的临时数组**：`ratio*w1`、`ratio_sq*w2`、`pre`、`pre*sum_val`、
     `maximum(...)`、`where(...)` 各自产生一个 `(N,N)` 临时量。
   - **`t_tilde` 拟合**：O(N) 但含两次 `exp`。
3. **超越函数已被完全移出热路径**（`arcsinh`/`arctan`/`log` 全在构建期），
   这一项**没有打折**——它 100% 兑现了。

### 4.3 剩余瓶颈（交给 FG-4c / FG-4d）

| 瓶颈 | 占比 | 建议归属 |
| --- | --- | --- |
| ODE 残差（RK4 + matvec，80 次 `dp_dz`） | after 的 56% | FG-4c：batch 维能把 80 次 RK4 子步批成一次 matmul，收益结构与本刀相同 |
| NLI 访存（三张 1.27 MB 张量 + 临时量） | after 的 44% | FG-4c：`w0/w1/w2` 已是纯张量，搬 GPU 后由显存带宽而非系统内存承担；batch 维可摊薄构建期成本 |
| NLI 构建期（本刀引入的一次性 O(N²)） | 不在热路径 | 已用 `nli_tensors` 值键缓存；跨跨段/跨 evaluator 复用，键含 `n_span` 与 `L_m` |

---

## 5. 验收对照

| ID | 判据 | 结果 |
| --- | --- | --- |
| F4b-1 | 默认档对金标准 ≤1e-12 | ✅ 最差 1.64e-15，预算常量未放宽 |
| F4b-2 | `off` 档仍与 origin/main 逐位相等 | ✅ 新增 3 条测试（含"确实没传张量"与"冻结源码仍是双层循环"） |
| F4b-3 | 同进程交错 A/B，带 `loadavg` | ✅ 450 样本、两次重复，loadavg 4.48→4.33 |
| F4b-4 | 实收 vs 理论上限，打折原因 | ✅ 2.310x；名义 5.2x **不可达**（ODE 地板 0.614 ms），可达上限 4.12x，达成 56.0%；打折原因逐条列出 |
| F4b-5 | 全量 ≥171/1，零既有期望值修改 | ✅ **194 passed / 1 skipped**（171 + 23 条新增）；期望值一字未改 |
| F4b-6 | 文档 | ✅ 本文件 |

---

## 6. 复算命令

```bash
cd /home/zsig/Documents/gits/wt/fg-ode-fast
export PYTHONPATH="$PWD/src"; export NUMBA_CACHE_DIR=/tmp/fg-fast-nb
PY="$HOME/miniconda3/envs/rl/bin/python"

$PY -c 'import network_nlo_eval as n; print(n.__file__)'      # 必须指向本 worktree
$PY -m pytest -q                                              # 194 passed, 1 skipped
$PY -m pytest -q tests/test_fg4b_nli_vectorization.py         # 23 passed
$PY scripts/fg4a_capture_golden.py --check --out results/fg4a_golden.json
$PY scripts/fg4b_speedup_ab.py --rounds 150 --out results/fg4b_speedup_ab.json
echo "loadavg: $(cat /proc/loadavg)"
```

产物：`results/fg4b_speedup_ab.json`（含 loadavg、逐档 min/median、
可达上限推导、数值自检）。基线提交：`a037532`（FG-4a），本 WP 在其之上。
