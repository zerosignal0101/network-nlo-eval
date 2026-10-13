# WP-M1 — 多槽需求贯通到 RWA 分配层 + 业务模型

基线 `ode-dev @ 0a95837`，分支 `wp/ms-alloc`。
一句话成果：**一个「需要 n 槽」的请求现在能被分配到一段连续槽窗口，
并把 `first_slot` / `n_slots` 一路带到 `AllocatedService`；释放按槽区间回收。**

---

## 0. 侦察：动手前的数据流（brief §2 步骤 1）

`ode-dev` 已经把灵活栅格打通到栅格层与物理层，但分配层仍是单信道。实测确认：

| 环节 | 改动前的位置与粒度 |
|---|---|
| 业务生成 | `simulation/traffic.py:ServiceRequest` 只有 `bit_rate_gbps`，**无槽需求** |
| 分配 | `rwa/allocators.py:KSPFirstFitAllocator` 按 `occupied_channels` 布尔数组选**单信道** `wavelength=channel_idx` |
| 占用 | `network/state.py:LinkState` 的 `occupied_channels` / `launch_power_profile_w` / `allocated_service_ids` 三个数组，长度均为 `spectrum_grid.num_channels` |
| 释放 | `KSPFirstFitAllocator.release` → `NetworkState.release_service` → `LinkState.release(wavelength)` |
| 阻断 | 只在物理层（`roadm_blocks_node` / `connection_overlaps_slots`），**分配器完全不看** |

**占用粒度与 `slots_per_channel` 的关系**：占用数组定长为 `num_channels`
（50 GHz 粒度）；灵活栅格槽宽 12.5 GHz，故 `slots_per_channel = 50/12.5 = 4`，
总槽数 `= num_channels * 4`。这条换算由 `network/slots.py:slots_per_channel`
单点负责，并在 `ms_alloc_scenarios.py` 中以 `SLOTS_PER_CHANNEL = 4` 固化。

**阻塞点**：分配层没有槽概念 ⇒ 一个「100 Gbps ⇒ 8 槽」的请求没有任何路径
被分配到连续 8 槽窗口。这正是本 WP 要打通的。

---

## 1. 索引空间的关键决定（必须先说清楚）

物理内核 `_calc_span_noise_and_power_jit` 吃的是**每信道标量功率向量**，
长度 = `num_channels`。要让物理逐槽运行就得改它的参数签名 = 改物理 = 破 M-J5。

因此本 WP 的分工是：

* **槽是分配层的真源**：决定窗口连续性、与谁冲突、释放清哪一段。
* **物理仍是信道粒度**：入纤功率只写在**锚信道**
  `anchor = first_slot // slots_per_channel` 上，QoT 验证也只在该锚信道上做。

这条分工不是权宜之计，而是 WP-M1 §2 步骤 3 自己给出的口径
（「定义它与 first_slot 的关系，例如 `wavelength = first_slot // slots_per_channel`」）。
代价与影响见 §5「未闭合项」。

---

## 2. 改了哪些文件

### 源码（3 个）

| 文件 | 改动 |
|---|---|
| `src/network_nlo_eval/simulation/traffic.py` | `ServiceRequest` 增 `n_slots`（默认 1）；新增 `slots_for_bit_rate()`；`generate_services` 增 `slot_width_hz` 开关；`AllocatedService` 增 `first_slot` / `slots_per_channel` 与 `slot_interval` |
| `src/network_nlo_eval/network/state.py` | `LinkState` 增 `slots_per_channel` / `occupied_slots` / `allocated_service_ids_slots` 与 `allocate_slot_range()` / `release_slot_range()`；`NetworkState.__init__` 增 `slots_per_channel`；`allocate_service` / `release_service` 按新旧分流；新增 `occupied_slots_on_path()` |
| `src/network_nlo_eval/rwa/allocators.py` | 新增 `first_fit_contiguous_windows()` 与 `KSPFirstFitFlexAllocator`；**`KSPFirstFitAllocator` 一个字节未动** |

### 证据与测试（新增）

`tools/ms_alloc_scenarios.py`、`tools/ms_alloc_{e2e,degenerate,legacy_regression,cost}.py`、
`tests/test_wp_m1_multislot.py`（15 条）、`results/ms_alloc_*.json`（4 份）。

### 一并恢复的构建脚手架（非本 WP 功能，但 M-J12 依赖它）

`scripts_shared/{compat_run,cross_snr_compat}.py` 与 `papers/*.md` 在 `ode-dev`
里**从未被 git 跟踪**，故任何全新 checkout 上全量测试只有 27 passed/4 failed/1 skipped。
本 WP 把它们按 `ode-dev` 原件逐字节复制进来（sha256 一致），M-J12 才能复现。
详见 §4 的 M-J12 行。

---

## 3. 设计要点

### 3.1 槽需求入请求
`ServiceRequest.n_slots: int = 1`，尾字段带默认值，**既有 7 处构造点
（含外部的 `scripts_shared/compat_run.py`）一个字不改**。
`slots_for_bit_rate()` 只是 `core.flexgrid.slots_for_rate` 的单位换算外壳
（Gbps→bit/s、调制名→bits/symbol），取整语义仍由 flexgrid 单点负责。

`generate_services(slot_width_hz=None)` 默认不推导槽需求 ⇒ 遗留流量不变；
给了槽宽才推导，且**该分支不消耗随机数**，故两种模式下到达时间/比特率序列逐位一致。

### 3.2 新旧分配器的分流开关
`AllocatedService.slots_per_channel`：
* `None` = 遗留信道粒度（`KSPFirstFitAllocator` 造出的对象恒为此值）；
* 正整数 = 灵活栅格槽区间。

`NetworkState.allocate_service` / `release_service` 据此分流，**旧路径逐位不变**。

### 3.3 信道对齐展开（退化等价的关键）
`allocate_slot_range` 把与窗口相交的信道的**全部子槽**一并置占用。
原因：`slots_per_channel=4` 且 `n_slots=1` 时，单槽窗口必须占满整个信道，
否则下一个 1 槽请求会落到同一信道的另一个子槽上，而旧分配器已把该信道标为占用。
这条规则是 M-J10 成立的前提。

### 3.4 窗口卷积
`first_fit_contiguous_windows` 用**前缀和** O(N) 求全空闲窗口起点；
同锚信道只保留第一个候选——QoT 只吃锚信道，同信道内不同起点 QoT 结果完全相同，
重复验证是纯浪费（M-J11）。

---

## 4. 判据逐条读数

全部数字来自 `results/*.json`，可用 §6 的命令逐条复算。

| 判据 | 读数 | 落盘位置 |
|---|---|---|
| **M-J5** 物理内核零 diff | `git diff ode-dev -- models/numba_kernels.py` 输出 **0 行** | 见 §6 命令 1 |
| **M-J6** 6 场景逐位 | 逐位相同 **6/6**，可比较 5 条 `delta_db` **全为 0.0**，`max\|delta_db\| = 0.0`，两端一致拒绝 1 条 | `results/ms_alloc_legacy_regression.json` |
| **M-J9** 多槽端到端 | 分配成功 **6/6**，不同 `n_slots = [1,2,3,4,6,8]`（要求 ≥3）；释放后窗口槽读数 `[False,False,False]`，再分配复用同一 `first_slot=0` | `results/ms_alloc_e2e.json` |
| **M-J10** 退化等价 | 逐请求 `(path, wavelength)` 一致 **12/12**，`all_identical=true` | `results/ms_alloc_degenerate.json` |
| **M-J11** 无 ODE | 单次 QoT 中位耗时 **0.1537 ms**（预算 10 ms，p95 0.1570 ms，n=200）；内核标识符 960 个，ODE/RK4 命中 **0** | `results/ms_alloc_cost.json` |
| **M-J12** 测试数 | **46 passed / 1 skipped / 0 failed**（基线 31 passed / 1 skipped，新增 15 条） | 见 §6 命令 2 |

### M-J9 逐请求读数（`results/ms_alloc_e2e.json`）

| service_id | n_slots | first_slot | wavelength | 占用带宽 | SNR (dB) |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 12.5 GHz | 31.647 |
| 1 | 2 | 4 | 1 | 25.0 GHz | 30.541 |
| 2 | 3 | 8 | 2 | 37.5 GHz | 30.066 |
| 3 | 4 | 12 | 3 | 50.0 GHz | 29.770 |
| 4 | 6 | 16 | 4 | 75.0 GHz | 29.556 |
| 5 | 8 | 24 | 6 | 100.0 GHz | 30.057 |

`first_slot` 依次为 0/4/8/12/16/24，全部**按信道对齐**且首尾相接——
这正是「连续窗口 + 信道对齐展开」的可见证据（4 槽宽的请求把信道 3 占满，
故 6 槽请求从信道 4 起；8 槽请求占信道 6、7）。

### M-J10 退化等价（`results/ms_alloc_degenerate.json`）
12 条请求全部 `n_slots=1`，两套分配器 `wavelength` 逐条相同。
其中包含真实的「释放后再复用」用例，证明**槽回收语义也与旧分配器一致**：

* `svc0 → w=0`，释放后 `svc7 → w=0`
* `svc1 → w=1`，释放后 `svc4 → w=1`
* `svc2 → w=2`，释放后 `svc6 → w=2`

---

## 5. 未闭合项（如实上报）

### 5.1 宽连接的物理仍是「锚信道」近似 —— 最重要的一条
一个占 8 槽（100 GHz）的连接，在物理上**只在它的锚信道上注入功率并计算 SNR**，
其 7 个同信道子槽没有独立的物理贡献。原因是冻结内核吃每信道标量功率向量
（见 §1）。

* 影响：宽连接的**频谱占用**是真的（槽层面），
  但**噪声耦合/非线性**按锚信道的标量 Rs 算，未按 100 GHz 的真实带宽计。
* 为何不修：修正需改 `numba_kernels.py` 的参数签名 = 改物理 = 直接破 M-J5。
  按 `_PREAMBLE.md` §8 与 brief §5，停手上报，不自行决定改物理。
* 衔接：`ode-dev` 的 `known_limitations` 已记录同一根因
  （`srs_band_edges_still_scalar_rs`、`uniform_symbol_rate_only`）。

### 5.2 阻塞率/波长利用率沿用遗留口径
`MetricsCollector` 仍按 `wavelength`（锚信道）统计。槽级碎片指标
（如槽级碎片率）本 WP 未实现——brief 未要求，但报告在此标注，
以免下游误把现有利用率读数当作槽级利用率。

### 5.3 `results/ode_dev_*.json` 未重新生成
本 WP 没有改动 `ode3_*` 工具的输出内容，故未触碰那批历史产物。
`tests/test_ode3_flexgrid_sync.py` 的守卫（读 `results/ode_dev_*.json`）仍全绿，
其 `--check` 守卫亦通过。

### 5.4 构图脚手架（`scripts_shared/`、`papers/`）的版本化是本 WP 的附带决定
它们属于 `ode-dev` 的未跟踪文件，本 WP 选择提交进来，否则 M-J12 在任何全新
checkout 上都无法复现。这是**环境修复**，不是判据放宽；若编排方更希望它们
走别的方式分发（例如只保留说明文档），请指示，我可以撤掉这两个提交。

### 5.5 M-J11 的「无 ODE」判定口径
判据原文是「`numba_kernels.py` 中 grep 无 `ode`/`RK4` 符号」。裸 `grep -c ode`
在冻结内核上有 **2 处命中**，但都在字符串字面量里，是 ode-dev 既有的**参考文献标题**：

* 第 183 行：`"...SNR Estimation of Multi-Band Systems"`（`M**ode**l`）
* 第 295 行：`"Modeling and mitigation of fiber nonlinearity..."`（`M**odeling**`）

本 WP 用 `tokenize` 剥离注释与字符串字面量后**逐标识符**比对（960 个标识符，命中 0），
并把裸 grep 的原始计数（`ode=2 / RK4=0 / ode_accel=0 / phi_kurtosis=0`）
原样存进 `no_ode_scan.grep_raw`，不藏数。
**口径变化已显式声明**：判定的是代码标识符，而非任意子串。

### 5.6 我自己测错并纠正的两处（留档，避免复算方重蹈）
1. **耗时第一版测的是早退分支**：`verify_allocation` 开头就有
   `if occupied_channels[channel_idx]: return False`，原实现固定在已被占用的
   channel 0 计时，测出 0.0002 ms 的假读数。现动态选空闲信道（channel 5）
   并断言预验证返回 True，真实值 0.1537 ms。
2. **首次无 ODE 扫描把 `Model`/`Modeling` 误判成 ODE 符号**，见 §5.5。

---

## 6. 复算命令

前置：`cd /home/zsig/Documents/gits/wt/ms-alloc`（下文 `$REPO`），解释器 `~/miniconda3/envs/rl/bin/python`。

```bash
# 1. M-J5 物理内核零 diff（无输出即通过）
git diff ode-dev -- src/network_nlo_eval/models/numba_kernels.py

# 2. M-J12 全量测试
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python -m pytest -q

# 3. M-J6 legacy 逐位回归
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/ms_alloc_legacy_regression.py --check

# 4. M-J9 多槽端到端（含释放回收读数）
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/ms_alloc_e2e.py --check

# 5. M-J10 退化等价
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/ms_alloc_degenerate.py --check

# 6. M-J11 耗时 + 无 ODE
PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/ms_alloc_cost.py --check

# 7. 旧分配器既有行为未被改动（无 '-' 行即纯新增）
git diff ode-dev -- src/network_nlo_eval/rwa/allocators.py | grep -E '^-' | grep -v '^---'
```

重算全部证据（会覆盖 `results/ms_alloc_*.json`）：

```bash
for t in legacy_regression e2e degenerate cost; do
  PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/ms_alloc_$t.py --write
done
```

**从 JSON 逐行复算 M-J6 / M-J9 / M-J10 读数**：

```bash
# M-J6：每条 delta_db == 0.0
python3 -c "import json;d=json.load(open('results/ms_alloc_legacy_regression.json'));print([r['delta_db'] for r in d['scenarios'] if not r['both_rejected']])"

# M-J9：>=3 个不同 n_slots + 释放回收
python3 -c "import json;d=json.load(open('results/ms_alloc_e2e.json'));print(d['distinct_n_slots_allocated']);print(d['release_probe']['after_release']['occupied_slots_in_window'])"

# M-J10：逐请求一致
python3 -c "import json;d=json.load(open('results/ms_alloc_degenerate.json'));print(d['summary'])"
```

---

## 7. 提交序列（每步一个提交）

| 提交 | 内容 | 判据读数 |
|---|---|---|
| `5f27a53` | 恢复未跟踪的 `scripts_shared/` + `papers/` | pytest 27p/4f/1s → **31p/1s** |
| `ac52f65` | `ServiceRequest.n_slots` / `AllocatedService.first_slot` | 31 passed / 1 skipped |
| `8d83a87` | `NetworkState`/`LinkState` 槽区间分配与释放 | 31 passed / 1 skipped；M-J5 diff 0 行 |
| `fec4e94` | `KSPFirstFitFlexAllocator` + 窗口卷积 | 31 passed / 1 skipped；旧分配器 diff 无 `-` 行 |
| `5d4c49d` | M-J9 端到端证据 + 修 `slots` dataclass 的 `super()` 崩溃 | M-J9 **6/6 分配成功**，不同 n_slots `[1,2,3,4,6,8]` |
| `4576e8f` | M-J10 退化等价证据 | 逐请求一致 **12/12** |
| `2624e30` | M-J11 耗时与无 ODE 证据 | 中位 **0.1537 ms**；标识符命中 0 |
| `907551f` | M-J6 legacy 逐位回归 | 逐位相同 **6/6**，`max\|delta\|=0.0` |
| `00c830f` | 15 条验收守卫测试 | pytest **46 passed / 1 skipped / 0 failed** |