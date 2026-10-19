# FG-5：物理仓两条线合并 + RL 与物理仓重新整合

> 本刀的报告。两个仓都改了：物理仓 `wt/fg-integrate`（分支 `wp/fg-integrate`）
> 与 RL 仓 `wt/fg-rt`（分支 `wp/fg-rt`）。
>
> **运行环境说明**：物理仓与 RL 仓都必须在 `PYTHONPATH=<树>/src` 下跑，并断言
> import 落在被测树内——两仓都是 `pip install -e .`，`.pth` 指向**主检出**且与 cwd
> 无关，不设这一行就会 import 到另一棵树。RL 侧用
> `$HOME/miniconda3/envs/rl/bin/python`（numpy 2.5.3 + gymnasium）；
> `$HOME/miniconda3/envs/nlo/bin/python` 是 numpy 1.26.4，**缺 `np.trapezoid`，
> 用它跑全量测试会产生 9 个纯环境性失败**（见 §8.4）。

---

## 0. 一句话结论

两条线合并后**没有一个数发生变化**：ΔSNR 表 max|Δ| = **0.0**（8 个可比较场景逐位
相同），FG-4a 金标准 31/31 逐位相同、FG-4a 的 ≤1e-12 契约 108 条全过、FU-4 保真档
金标准逐位复现、全量 **325 passed / 1 skipped**。RL 侧把链路噪声装配的**逐字复刻**
换成对物理仓的委托，并用**结构性守卫**（字节码断言 + 同一函数对象）钉住"今后不可能
再分叉"。

**但与 brief 的预期有两处实测不符**，如实记在 §8，不为了对齐预期去改任何期望值。

---

## 1. 分支图与合并

```
                        15b3c2c（分叉点）
                       /            \
      a233002                          a06d34c
      wp/fg-uwb-fid（保真线 +4）        wp/fg-ode-fast（速度线 +6）
       │  tabular 谱 / S+C+L+U 波段      │  matvec 默认档 / CPU 批量核
       │  FU-4 并集 SNR 装配             │  FG-4a 金标准 / FG-4g 归因
       │                                │
       └──────────────┬─────────────────┘
                      ▼
              95a55ed（merge commit）
              wp/fg-integrate  =  HEAD of 本刀
```

合并提交 `95a55ed`，双亲 `a233002` + `a06d34c`。

**交付物在位核对**（合并后实测，非目测）：

| 来源 | 交付物 | 合并后 |
|---|---|---|
| 速度线 | `models/batch_kernels.py` | 在位 |
| 速度线 | `MultiBandISRSGN.calc_span_noise_and_power_batch` | 在位 |
| 速度线 | `MultiBandISRSGN._batch_constants_for_span` | 在位 |
| 速度线 | `batch_cache_stats()` | 在位 |
| 速度线 | `ode_accel` 默认档 = `matvec` | 在位 |
| 保真线 | `RAMAN_SPECTRUM_MODES = ('legacy','tabular')` | 在位 |
| 保真线 | `MultiBandISRSGN(..., raman_spectrum=)` | 在位，默认 `legacy` |
| 保真线 | `for_fiber` 的 `raman_spectrum` 透传（FU-2） | 在位 |
| 保真线 | `physics/raman_spectrum.py`、`core/bands.py` | 在位 |
| FU-4 | `rwa/qot_checker.py` 并集 SNR 装配 | 在位，**与 `a233002` 逐字节相同** |

---

## 2. 两处冲突的逐处处置

`git merge-tree` 预检出 2 处冲突。原则：**两侧都是增量式交付，按并集处置，不丢任何
一条线的成果**；`rwa/qot_checker.py` 以 FU-4 的并集为准。

### 2.1 `.gitignore`

| | 内容 |
|---|---|
| 冲突成因 | 两条线各自在文件尾追加了一组 `!/results/*.json` 反选规则，**落点相邻** |
| 保真线侧 | `!/results/fg_u1_raman.json`、`!/results/fg_u2_scu.json` |
| 速度线侧 | `!/results/fg4e_speedup_ab.json`、`!/results/fg4e_speedup_ab_run2.json`、`!/results/fg4g_attribution.json` |
| 是否重叠 | **否**。两组规则各自独立，语义都是"让被 `/results/*` 忽略的验收产物重新入库" |
| 处置 | **两组都保留**（保真线在前、速度线在后，另加本刀的 `fg5_integration.json`） |

取舍理由：这是 `results/*` 反选清单，删任何一条都会让对应的验收产物不再入库，直接
违反"每个数字都能从提交的产物复算"。没有语义冲突，只有位置冲突。

### 2.2 `src/network_nlo_eval/models/isrs_gn.py`

| | 内容 |
|---|---|
| 冲突成因 | 两侧都在 `ODE_ACCEL_*` 常量之后追加模块级定义，**位置相接** |
| 保真线侧 | `RAMAN_SPECTRUM_MODES` / `RAMAN_SPECTRUM_DEFAULT`（拉曼谱档位） |
| 速度线侧 | `_BATCH_CACHE_STATS` + `batch_cache_stats()`（批量常量包命中计数） |
| 是否重叠 | **否**。两者语义无关：一个是"用哪条谱"，一个是"缓存命中了几次" |
| 处置 | **并集保留**，顺序为保真线在前、速度线在后 |

该文件其余部分由 git 自动合并即已正确，逐处核对过：

* 顶部 import：`batch_kernels` 的三个名字（速度线）与 `raman_g_matrix_tabular` /
  `raman_g_pair_matrices_tabular`（保真线）**共存于同一个 `numba_kernels` import 块**，
  自动合并正确；
* `__init__` 的 `raman_spectrum` 形参与校验（保真线）与批量入口所需的
  `_batch_constants_*`（速度线）分处不同区域，无交叠；
* `for_fiber` 的 `raman_spectrum` 透传（FU-2）与 `calc_span_noise_and_power_batch`
  （FG-4e）分处不同区域，无交叠。

### 2.3 没有冲突但需要说明的文件：`rwa/qot_checker.py`

**速度线在这��文件上没有任何改动**（`git diff a233002 -- rwa/qot_checker.py` 为空），
因此不存在"速度线改动如何取舍"的问题：FU-4 的并集 SNR 装配原样保留，合并后与
`a233002` **逐字节相同**。

⚠️ 但 FG-5 在物理仓对这一个文件做了**行为不变的重构**（见 §5.1）：把噪声累加提取到
`rwa/link_noise.py`，让 RL 侧能委托到同一个函数对象。重构前后四条门槛各跑一遍，结果
见 §3，两侧一致。

---

## 3. 四条门槛的实测结果

全部在合并后、`qot_checker` 重构后、`PYTHONPATH=$PWD/src` 且 import 断言落在
`wt/fg-integrate` 内、`flock` 串行化下实测。

| # | 门槛 | 命令 | 结果 |
|---|---|---|---|
| G1 | legacy 数值金标准复现 | `python scripts/fg4a_capture_golden.py --check` | ✅ 31/31 **bit-identical**，worst relative deviation `0.000e+00`，0 differing elements |
| G2 | FU-4 保真档金标准复现 | `python scripts/fu4_capture_golden.py --check` | ✅ 10 scenarios，**bit-exact** |
| G3 | FG-4a ≤1e-12 契约（`off` vs 默认档） | `pytest tests/test_fg4a_numeric_identity.py -q` | ✅ **108 passed** |
| G4 | 全量测试 | `pytest -q` | ✅ **325 passed, 1 skipped** |

环境：`rl` 解释器（numpy 2.5.3），G4 用时 36.54 s，该次运行 loadavg 3.04 → 5.50
（**远低于 12**）。重构**前**的同一条全量测试也是 325 passed / 1 skipped（68.73 s，
loadavg 7.81 → 9.41），即本刀的物理仓重构**没有改变任何测试结果**。

---

## 4. ΔSNR 表（整合前 / 整合后）

数据源：`results/fg5_integration.json`，由
`scripts/fg5_integration_evidence.py` 生成，`--check` 可逐字节重渲。

两侧都**用各自那棵树自己提交在库里的 `scripts/fu4_capture_golden.py`** 产出，因此
"整合没改物理"与"本刀的近似代码没改物理"是分得开的。选 FU-4 场景而不是 FG-4a 场景，
是因为 FU-4 走 `rwa/qot_checker.py`——正是 RL 依赖、也是历史上两仓分叉的那条路径；
FG-4a 走 `evaluate_path`，根本不调这个文件。

| 场景 | 整合前 SNR (dB) | 整合后 SNR (dB) | ΔSNR (dB) | 接受性变化 |
|---|---:|---:|---:|---|
| `legacy_two_hop_equalize_on` | 26.034730430 | 26.034730430 | 0.000000000 | 无 |
| `legacy_three_hop_equalize_on` | 24.152139266 | 24.152139266 | 0.000000000 | 无 |
| `legacy_two_hop_equalize_off` | 23.029699791 | 23.029699791 | 0.000000000 | 无 |
| `legacy_express_attenuation` | 25.989211917 | 25.989211917 | 0.000000000 | 无 |
| `legacy_high_filtering` | 25.025781932 | 25.025781932 | 0.000000000 | 无 |
| `legacy_blocked_channel` | — | — | — | 无 — rejected on both sides (无 SNR 可比；接受性仍单独比对) |
| `flex_two_hop_equalize_on` | 26.034730430 | 26.034730430 | 0.000000000 | 无 |
| `flex_three_hop_equalize_on` | 24.152139266 | 24.152139266 | 0.000000000 | 无 |
| `flex_two_hop_equalize_off` | 23.029699791 | 23.029699791 | 0.000000000 | 无 |
| `flex_blocked_slots` | — | — | — | 无 — rejected on both sides (无 SNR 可比；接受性仍单独比对) |

**结论**：10 个场景中 8 个可比较 SNR，
**max|ΔSNR| = 0.0**，逐位相同；接受性翻转的场景：
`无`。两个 `rejected` 场景在两侧都被拒（无 SNR 可比），
其**接受性**仍逐场景比对了。

对照树：`before_tree` = `/tmp/fg5-before-phys`
（`git archive a233002` 还原，期望 commit `a233002`），
`after_tree` = `wp/fg-integrate = merge(a233002, a06d34c)`。

产��重渲：`python scripts/fg5_integration_evidence.py --before-repo /tmp/fg5-before-phys --check`


---

## 5. RL 侧：消除重复实现

详见 `multiband-optical-rl` 仓的 `docs/fg5_rl_integration.md`。此处只给结论。

### 5.1 物理仓侧：提取共享核（行为不变）

新增 `network_nlo_eval/rwa/link_noise.py`，导出 `assemble_link_noise(...)`；
`QoTValidator._calc_link_noise` 的算术逐字搬进该函数，方法本身只剩"从拓扑取值"。

### 5.2 RL 侧：删除副本，改为委托

`ScanValidator._calc_link_noise` 的逐字复刻被删掉，改为调用**同一个** `assemble_link_noise`
函数对象。RL 保留的差异只有**取值来源**（缓存的 `_LinkPlan` vs 每次查拓扑）与
**噪声缓存写时序**，不含一行算术——逐条列在 RL 仓文档 §2。

### 5.3 结构性守卫（能防住下次改动物理仓时的静默分叉）

`tests/test_fg5_no_divergent_noise_assembly.py`，**7 passed**：

* 两个入口的字节码 `co_names` 都**不得**出现 `_calc_span_noise_and_power`（有人把累加
  内联回任一侧就红）；
* 两个入口都**必须**引用 `assemble_link_noise`；
* 两侧 import 到的 `assemble_link_noise` 与 `network_nlo_eval.rwa.link_noise` 里的必须是
  **同一个对象**（`is`）——vendored 副本只有这条能抓住；
* RL 侧方法体里不得出现 `if num_full_spans` / `if plan.remainder_length_m` / `+= `；
* 随机化输入上两个入口逐位相等；零跨段分支返回三个**互不共享**的全零数组；
  余段阈值是**严格** `> 1.0 m`。

"跑一次相等"防不住下次改动物理仓——所以前四条才是分叉守卫，语义层只作对照。

---

## 6. RL 侧三份测试文件的逐文件通过数

`PYTHONPATH=<fg-rt>/src:<fg-integrate>/src`，import 两侧分别断言落在
`wt/fg-rt` 与 `wt/fg-integrate` 内。

| 测试文件 | 整合前（重构前） | 整合后（重构后） |
|---|---|---|
| `tests/test_scan_fastpath.py` | 11 passed, **1 failed** | 11 passed, **1 failed** |
| `tests/test_scan_ci.py` | 10 passed | 10 passed |
| `tests/test_scan_pruning.py` | 14 passed | 14 passed |
| `tests/test_fg5_no_divergent_noise_assembly.py`（本刀新增） | — | **7 passed** |

唯一失败项是 `test_fast_path_is_faster_than_verify_allocation`，**重构前后同样失败**，
因此**不是本刀重构引入的**。它是一条计时断言。在 loadavg < 12 的干净窗口复测：

* 重构后 RL + 整合树：**0.994×**（base 6.4137 ms / fast 6.4524 ms，138 候选），loadavg 5.10 → 4.75
* 重构前 RL + 整合树：**1.021×**，loadavg 5.58 → 12.43

两者都在 1.0 附近、都低于断言下限 1.6×。即快速通道相对上游的优势在整合树上**已经
消失**，原因是速度线把上游 `verify_allocation` 本身变快了，不是 RL 变慢。
详细归因见 §8.3；该测试的期望值按 G2 纪律**一个字节未改**，是否重新标定交回编排方。

---

## 7. 决策序列等价性

数据源：`multiband-optical-rl` 仓的 `evidence/fg5_decision_identity.json`
（`python evidence/fg5_decision_identity.py --check` 可逐字节重渲）。

**设置**：策略固定为 `argmax(action_masks())` + 默认 first-fit 信道——确定性、无学习
权重，因此**唯一变量是被测的代码**。seed `0/1/2`，每个 seed 20 步，每步记录
`(action, path_index, band_index, path, channel, allocated)`。资产图固定用同一份。

### 7.1 归因矩阵（这是本节的重点）

只比 before/after 两点会得出误导性结论："决策变了"既可能是 RL 的重构改坏了，也可能是
**物理仓换了**。四个配置把两者分开：

| 配置 | RL | 物理 | 用途 |
|---|---|---|---|
| `before_eczdev` | `cf91d9d`（重构前） | 主检出 `ecz-dev` | 整合前基线 |
| `before_merge_uwb` | `cf91d9d`（重构前） | `a233002`（保真线，未合并） | 隔离保真线 |
| `before_merge_full` | `cf91d9d`（重构前） | `wt/fg-integrate`（合并后） | 隔离合并 |
| `after_integrated` | `wp/fg-rt`（重构后） | `wt/fg-integrate`（合并后） | 整合后 |

| 对照 | 变量 | 结果（seed 0/1/2） |
|---|---|---|
| `refactor_effect` | **物理相同**，只有 RL 的委托重构不同 | ✅ 逐位相同 — 0/1/2 全同 |
| `merge_effect` | **RL 相同**，只有"把速度线并进来"不同 | ✅ 逐位相同 — 0/1/2 全同 |
| `fidelity_effect` | **RL 相同**，只有物理从 `ecz-dev` 换到 `a233002` | ❌ 有分叉 |
| `total` | 编排方要求的主读数（`before_eczdev` vs `after_integrated`） | ❌ 有分叉 |

**结论（本刀最实质的一条）**：

1. **本刀的 RL 重构对决策序列零影响**——`refactor_effect` 三个 seed 各 20 步全部逐位
   相同。这与 §5.3 的字节码守卫、§3 的四条门槛互相印证。
2. **这次合并（速度线并入）对决策序列零影响**——`merge_effect` 同样全同。
3. **决策的全部变化来自保真线**（`fidelity_effect`）——即 FU-4 的**并集 SNR 装配**等
   物理改进。那是保真线**刻意**做的物理改变（它本来就该改结果，并有自己的金标准），
   不是回归。

### 7.2 第一次分叉的位置与形态（`total` 与 `fidelity_effect` 完全一致）

| seed | 第一次分叉的 step | 路径/频段是否相同 | 路径 / 频段 | 信道 |
|---|---|---|---|---|
| 0 | 2 | ✅ 逐位相同 | [4, 12, 10, 23, 27, 14] / band 0 | 信道 **53** → **52** |
| 1 | 3 | ✅ 逐位相同 | [10, 23, 27, 14] / band 0 | 信道 **3** → **1** |
| 2 | 7 | ✅ 逐位相同 | [3, 26, 25, 17, 16] / band 0 | 信道 **47** → **45** |

形态一致且值得注意：

* **路径与频段在每个 seed 的第一次分叉点上都相同**，**只有信道变了**；
* 变化方向一致：整合后选到的是**更小**的信道下标（52<53、1<3、45<47）。first-fit
  从低到高扫，所以在"整合后选得更靠下"意味着**低号信道在整合后变得可行了**——即
  整合后的树对 QoT 判定**略宽松**，可行信道集合变大了；
* 四个配置的 `blocked` / `blocked_qot` / `blocked_no_spectrum` 计数**逐 seed 完全相同**
  （seed 0: 5/5/0，seed 1: 8/8/0，seed 2: 8/8/0）。即**接受/拒绝的判定没变**，变的
  只是"在可行集合里挑到哪一个"。这与"保真线把 SNR 抬高了一点、于是某些原本刚好不过
  的低号信道现在能过"是自洽的。

因此：**不是 RL 换了策略，也不是合并引入的回归，而是保真线的物理改进顺带放宽了
部分信道的 QoT 门限**。这属于预期内的结果改变，本刀**没有**为了让任何旧金标准变绿
而调整物理或期望值。


---

## 8. 与 brief 预期不符之处 / 未测量 / 未闭合

### 8.1 brief 说"约 16 个逐位断言失败"——**未复现**

brief §0.4 称 RL 跑在整合树上时 `test_scan_fastpath` / `test_scan_ci` /
`test_scan_pruning` 三文件共约 16 个逐位断言失败。**实测不成立**：在整合树
（`wt/fg-integrate`）上，三份文件共 35 个测试，**34 通过、1 失败**，且唯一失败项是
计时断言，**不是逐位断言**。全部 bit-equality 测试
（`test_fast_path_is_bit_exact_across_occupancy_levels_paths_and_bands`、
`..._on_roadm_blocked_and_non_boosted_topology`、`..._does_not_perturb_upstream_...`、
`test_k_at_full_count_is_bit_identical_to_full_enumeration` 等）**均通过**。

因此 brief 的前提"**RL 现在无法吸收 FG 线的任何物理**"与实测不符：RL 的逐位闸在整合
树上**本来就是绿的**。本刀仍然做了委托重构——因为"跑一次相等"防不住结构性问题，
重复实现本身就是历史上分叉的成因，这与 brief 的动机一致；但**"修复 16 个失败"这个
具体目标并不存在**，如实记录。

### 8.2 brief 要求"若仍不绿，定位到哪一层语义不同"——**不适用**

因为逐位闸本来就是绿的，无需层次定位。

### 8.3 加速比 —— **本刀的 RL 重构不是原因；快速通道相对上游的优势在整合树上已消失**

`test_fast_path_is_faster_than_verify_allocation` 在两次实测里都红。最初两次都在高负载下
（loadavg 8–25），本刀因此一度把它标为"未测量"。随后等到 loadavg < 12 的窗口做了
**干净复测**，并补了一次"只换 RL、不换物理"的对照：

| RL | 物理 | loadavg（实测区间） | 实测加速比 | 断言下限 | 结果 |
|---|---|---|---:|---:|---|
| `wp/fg-rt`（重构后） | `wt/fg-integrate` | 5.10 → 4.75 | **0.994×**（base 6.4137 ms / fast 6.4524 ms，138 候选） | ≥ 1.6× | ❌ |
| `cf91d9d`（重构前） | `wt/fg-integrate` | 5.58 → 12.43 | **1.021×** | ≥ 1.6× | ❌ |

**结论**：干净负载下**两个版本都不过**，且两者都在 1.0 附近。因此

* **不是本刀的委托重构造成的**——把 RL 换回重构前的版本，比值同样不达标（1.021× vs
  0.994×，差异在噪声量级）；
* 快速通道原本相对 `verify_allocation` 的优势，是建立在"上游慢"这个前提上的。速度线
  把 `ode_accel` 默认档改成 `matvec` 并做了 NLI 向量化之后，**上游自己变快了**，于是
  两侧拉平到约 1.0×。这是**物理仓变快**的结果，不是 RL 变慢。

**本刀的处理**：按 G2 纪律，**没有**修改既有测试的期望值（`tests/test_scan_fastpath.py`
一个字节未动），也**没有**把 `>= 1.6` 放宽。该测试的前提是否应当随上游提速而重新
标定，属于**速度线的决策范围**，不在本刀授权内，记为**未闭合**并交回编排方。

同时按纪律：本刀**不报任何加速比声明**——本刀是整合与正确性刀，且上表是"两个版本都
不达标"的实测记录，不是可用于对外声明的性能读数。

### 8.4 `nlo` 解释器下全量测试有 9 个失败——**纯环境性**

用 `$HOME/miniconda3/envs/nlo/bin/python`（numpy **1.26.4**）跑物理仓全量测试：
7 个 `AttributeError: module 'numpy' has no attribute 'trapezoid'`（`np.trapezoid` 是
numpy ≥ 2.0 才有的）、1 个 `ModuleNotFoundError: No module named 'gymnasium'`（fg4g 差分
探针要 RL 依赖）、1 个同源。换 `rl` 解释器后 **325 passed / 1 skipped**。与合并无关。

### 8.5 仍未闭合的 RL 侧重复实现

`_service_snr` / `_path_channel_snr`（各约 50 行，复刻上游单信道标量版本）与
`_roadm_terms` / `_HopTerms` 的常量折叠**仍是 RL 侧独立的实现**，同样存在改物理仓时
静默分叉的结构性风险。本刀只处理了 `_calc_link_noise`（brief 授权范围）。逐条列在
RL 仓 `docs/fg5_rl_integration.md` §5。

### 8.6 其它未测量

* 本刀**未**跑任何加速基准（`fg4e_speedup_ab`、`fg4c`、`fg4d` 等一律未跑）。
* `fg4g_attribution.json` 的归因结论沿用速度线原提交，本刀**未**重渲复核。

---

## 9. 复现命令

```bash
# ---- 物理仓（合并 + 共享核）----
cd /home/zsig/Documents/gits/wt/fg-integrate
export PY=$HOME/miniconda3/envs/rl/bin/python
export PYTHONPATH=$PWD/src
$PY -c "import network_nlo_eval.models.isrs_gn as m; assert m.__file__.startswith('$PWD')"

flock /tmp/fg5.lock -c "$PY scripts/fg4a_capture_golden.py --check"          # G1
flock /tmp/fg5.lock -c "$PY scripts/fu4_capture_golden.py --check"           # G2
flock /tmp/fg5.lock -c "$PY -m pytest tests/test_fg4a_numeric_identity.py -q"  # G3
flock /tmp/fg5.lock -c "$PY -m pytest -q"                                     # G4

# ΔSNR 表（需要先还原整合前那棵树）
git -C /home/zsig/Documents/gits/wt/fg-integrate archive a233002 | tar -x -C /tmp/fg5-before-phys
flock /tmp/fg5.lock -c "$PY scripts/fg5_integration_evidence.py \
    --before-repo /tmp/fg5-before-phys --out results/fg5_integration.json"
flock /tmp/fg5.lock -c "$PY scripts/fg5_integration_evidence.py \
    --before-repo /tmp/fg5-before-phys --check"

# ---- RL 仓 ----
cd /home/zsig/Documents/gits/wt/fg-rt
export PY=$HOME/miniconda3/envs/rl/bin/python
export PYTHONPATH=/home/zsig/Documents/gits/wt/fg-rt/src:/home/zsig/Documents/gits/wt/fg-integrate/src
flock /tmp/fg5.lock -c "$PY -m pytest tests/test_scan_fastpath.py -q"
flock /tmp/fg5.lock -c "$PY -m pytest tests/test_scan_ci.py -q"
flock /tmp/fg5.lock -c "$PY -m pytest tests/test_scan_pruning.py -q"
flock /tmp/fg5.lock -c "$PY -m pytest tests/test_fg5_no_divergent_noise_assembly.py -q"

# 决策序列等价性（需要先还原整合前的 RL 树）
git -C /home/zsig/Documents/gits/multiband-optical-rl archive cf91d9d | tar -x -C /tmp/fg5-before-rl
flock /tmp/fg5.lock -c "$PY evidence/fg5_decision_identity.py --out evidence/fg5_decision_identity.json"
flock /tmp/fg5.lock -c "$PY evidence/fg5_decision_identity.py --check"
```