# FG-15：物理换底 FU-3 —— 默认拉曼档 `legacy` → `tabular`

> **本文档的每个数字都必须能从提交的产物复算。** 凡本刀没有测到的，一律显式标成
> **未测量**，不用"某档位的数字"顶替。本文档写在
> `wp/fg-tabdflt`（物理仓）与 `wp/fg-rt-tab`（RL 仓）两棵 worktree 上，
> 跨仓 import 用
> `PYTHONPATH=<fg-rt-tab>/src:<fg-tabdflt>/src`，并且**每次运行都打印实际
> import 到的包路径**作为证据（见 §7）。

## 0. 这一刀为什么是物理的，不是偏好

`legacy` 档的拉曼耦合是**合成折线**，带两个硬截断：

```
_NUMBA_RAMAN_LINEAR_LIMIT_THZ ≈ 13.2 THz   # 线性爬升段上界
_NUMBA_RAMAN_FLOOR_THZ        ≈ 15.6 THz   # 跌到零的位置，之后恒零
```

也就是说：**相隔超过 15.6 THz 的信道之间，耦合被强制为零**。
本程序（S+C+L+U）跨约 27 THz，远超这个截断；FG-U1 实测保真误差也集中在
S/C 段（最差 −8.260 dB）。`tabular` 档是**有出处的实测熔融石英拉曼增益谱**
（Lin & Agrawal 2006 响应函数经傅里叶变换得到的闭式增益谱），是这个运行区间
的**正确模型**。

两档共用同一语义：`G = cr_w_m_hz * 13.2e12 * S(|df|)`，`cr_w_m_hz` 管幅度、
`S` 管谱形，且在 13.2 THz 锚点处**逐位相等**。所以这一刀只换**谱形**，不换幅度，
legacy/tabular 的差异不会被幅度混淆。

## 1. 取证：档位到底在哪决定（**先取证，后动手**）

### 1.1 全部决定点（file + line，物理仓）

| # | 位置 | 性质 | 换底前的取值 |
|---|---|---|---|
| 1 | `src/network_nlo_eval/models/isrs_gn.py:89` | `RAMAN_SPECTRUM_MODES` 常量 | `("legacy", "tabular")` |
| 2 | `src/network_nlo_eval/models/isrs_gn.py:90` | **`RAMAN_SPECTRUM_DEFAULT`** | **`"legacy"`** ← **唯一的默认档来源** |
| 3 | `src/network_nlo_eval/models/isrs_gn.py:155` | `__init__` 关键字默认值 | `raman_spectrum: str = RAMAN_SPECTRUM_DEFAULT` |
| 4 | `src/network_nlo_eval/models/isrs_gn.py:181-189` | 校验 + **`tabular`×`off` 互斥** | 非法组合 `raise ValueError` |
| 5 | `src/network_nlo_eval/models/isrs_gn.py:197` | 实例属性（**读回点**） | `self.raman_spectrum = raman_spectrum` |
| 6 | `src/network_nlo_eval/models/isrs_gn.py:295-304` | G 矩阵构建分派 | `raman_g_matrix_tabular` vs `raman_g_matrix` |
| 7 | `src/network_nlo_eval/models/isrs_gn.py:615` | `for_fiber` 子评估器透传 | 传 `self.raman_spectrum`（FU-2 补的） |
| 8 | `src/network_nlo_eval/physics/raman_spectrum.py:136-169` | `tabular` 档谱源 | 闭式 + 插值 |

### 1.2 全部决定点（RL 仓）

| # | 位置 | 性质 |
|---|---|---|
| 1 | `src/multiband_optical_rl/config.py:20-53` | `SystemConfig`（11 个字段，**没有 physics 字段**） |
| 2 | `src/multiband_optical_rl/env.py:155-159` | 构造 `MultiBandISRSGN`，**不传 `raman_spectrum`** |
| 3 | `src/multiband_optical_rl/modulation.py:183-187` | 同上，**不传 `raman_spectrum`** |

### 1.3 复核 FG-12 的发现，并补齐它没说的三件事

FG-12 的 pilot 说「`SystemConfig` 里根本没有 physics 字段」。**复核结论：属实。**
本刀补齐三点 FG-12 未展开的事实：

1. **没有任何环境变量或 CLI 开关**。全仓 grep `getenv|environ` 在
   `src/network_nlo_eval/` 下只命中 `physics/qot_checker.py` 的 ImportError 文案，
   与档位无关；`__main__.py` 也**没有**任何 `--raman` 之类选项。所以 FG-12 所说的
   「进程级 patch 绑定」，实质就是**唯一那条通路：构造时传关键字**——
   没有第二个开关可以绕。

2. **`tabular` 此前只能由谁选中**：只有**显式传 `raman_spectrum="tabular"`** 的
   调用方。见 `scripts/fg_u1_raman_spectrum_evidence.py`、
   `scripts/fu2_scu_evidence.py`、`tests/test_fu2_sclu_bands.py`。生产路径
   （`env.py`）此前**结构上到不了** tabular。

3. **没有类属性、没有 fixture 级默认值**。`MultiBandISRSGN` 的档位只经由
   `__init__` 形参进入，落成一个普通实例属性（#5）。这意味着
   **「谁没传，就拿到全局默认」**——这正是换底会波及的全部机制。

### 1.4 直接回答 brief 的两个问题

**Q：`SystemConfig` 没有 physics 字段 ⇒ RL 现在根本无法表达"用哪一档"，这是配置面缺口。对吗？**

**对，而且比"无法表达"更严重**：RL 不仅无法表达，它连"自己正在用哪一档"都无法
**查询**——`env.evaluator.raman_spectrum` 存在（FG-12 正是读回它），但
`SystemConfig` 里没有任何东西把它和「本次实验的参数」关联起来。换句话说，
一次 RL 实验的产物里**没有任何字段**能回答"这次跑的是哪条拉曼谱"，
而 FG-7 已经证明 `results/*.json` 会忠实记录 `physical_regime` 字符串——
**谱档缺席在那个字段里**。

**Q：为什么"改默认"而不是"加档位开关"？**

本刀**两者都做**，理由不同，不是二选一：

- **改默认**：因为本程序的物理就该是 tabular。留一个默认=legacy 的"开关"，
  等于让**最省事的那条路**（不传参）继续跑在错物理上——而"不传参"恰恰是
  生产代码 `env.py` 的现状。开关不会让默认变对。
- **同时留一个显式取回 `legacy` 的入口**：这是**复现性**的唯一保证。
  FG-1 的 (A) 命题逐位对 `origin/main`、FG-4a 的金标准、旧 RL 结果，全部
  依赖 legacy 仍可达。brief §4 要求的那条测试就是钉这件事。

---

## 2. 最小配置面设计

### 2.1 只加"表达这一档"所必需的东西

brief §2 要求**最小**，且**不许**加抽象／插件框架／注册表。本刀只加三处：

1. **`SystemConfig` 增加一个字段** `raman_spectrum: str = RAMAN_SPECTRUM_DEFAULT`
   —— 默认值**直接引用物理仓的常量**，而不是在 RL 侧另写一个字符串。
   这样"默认是什么"只有**一个**事实来源：物理仓的 `RAMAN_SPECTRUM_DEFAULT`。
   换底因此**只需要改物理仓那一行**，RL 自动跟随，两边不可能漂移。

2. **`SystemConfig.__post_init__` 校验**档名在 `RAMAN_SPECTRUM_MODES` 内。
   这不是抽象，是把一个本该在**构造配置时**就暴露的打字错误提前暴露
   （否则它要等到 `MultiBandISRSGN` 构造时才炸，栈深且难归因）。

3. **`env.py` / `modulation.py` 两处构造点把 `config.raman_spectrum` 透传下去。**

**明确不做**：不加 tier 注册表、不加 `PhysicsBackend` 协议、不加工厂函数、
不加环境变量、不加 CLI 开关、不新增依赖。

### 2.2 为什么选档必须被读回验证

brief §2 要求「RL 的选档必须被读回验证（不许靠传参意图）」。本刀的读回点是
**`env.evaluator.raman_spectrum`**（FG-12 的做法）。

理由在 §1.3 第 3 点：档位经由 `__init__` 形参进入、落成实例属性，
**中间没有任何改写**。因此读回 `evaluator.raman_spectrum` 就是这条链的末端事实；
而断言「我传了 `tabular`」只证明了**调用点写了什么字**，证明不了
**生效的是什么**——两者的差正是换底这类改动最常见的静默失败面。

测试同时断言**三件事**，缺一不可：
- `SystemConfig().raman_spectrum == RAMAN_SPECTRUM_DEFAULT`（配置面读回）；
- `env.evaluator.raman_spectrum == SystemConfig().raman_spectrum`（**穿过了 env 到核**）；
- `MultiBandISRSGN.__init__.__defaults__[-1] == "tabular"`（形参默认值本身）。

### 2.3 与 §1.4 的"留一个 legacy 入口"呼应

`legacy` 不需要新增入口——它**本来就是**显式可传的形参。换底后
`raman_spectrum="legacy"` 依旧构造成功且**逐位不变**，这一点由
`tests/test_fg15_legacy_bitwise.py` 钉住（§4）。换言之：
**入口是早就有的，本刀要保证的是它换底后没被悄悄改语义。**

## 3. 翻默认 + 重建下游基线（逐条归因）

本节的规矩：**每一项移动的期望值都先给机制再重钉；不变量不许重钉。**
换句话说，下面每一个被改动的测试都必须能回答"它原来在证明什么"——
如果改动之后它证明的东西变了，那不是归因，那是偷换命题。

### 3.1 换底后立刻暴露的两类失败（物理仓）

翻 `RAMAN_SPECTRUM_DEFAULT` 那一行之后，物理仓全量从 338/1 变成
**48 failed / 356 passed / 1 skipped**。这 48 条**全部**落在两个机制上，
逐条如下：

#### 机制 A：`tabular` 与 `ode_accel="off"` 结构互斥 ⇒ 46 条直接 `ValueError`

`src/network_nlo_eval/models/isrs_gn.py:183-189` 有一条**早于本刀**就存在的拒绝规则：
`off` 是 FG-2 冻结的参照路径，它在核内**逐对重建 legacy 折线**，结构上无法换谱，
所以 `tabular` + `off` 被显式拒绝，而不是悄悄退回 legacy。

换底前这些测试不传谱档、于是落到 `legacy`，相安无事；换底后它们继承 `tabular`，
撞上这条规则。分布：

| 文件 | 条数 | 处理 |
|---|---|---|
| `tests/test_fg4g_matvec_attribution.py` | 43 | 构造 helper 显式钉 `raman_spectrum="legacy"` |
| `tests/test_fg4b_nli_vectorization.py` | 1 | `test_f4b_6` 的 evaluator 钉 `legacy` |
| `tests/test_fg2_flexible_grid.py` | 1 | F2-3 钉 `legacy` |
| `tests/test_fg13_evaluate_path_roadm.py` | 1 | 经 `scripts/fg13_probe_roadm.py` 三处构造钉 `legacy` |

**为什么是"钉 legacy"而不是"改成 matvec"**：这些测试的命题是
「**两条 ODE 路径之间的浮点重排预算**」或「**装配规则逐位不变**」。
谱档对它们是**正交变量**。特别是 `test_fg4g_matvec_attribution.py`：
它比较 `off` vs `matvec`，若只给其中一侧换谱，这条测试就变成
「在两套不同物理之间比预算」——**命题被换掉了**，而且它会**必然超预算**。
所以两侧必须钉在同一条谱上。

`scripts/fg4g_q2_trace_diff.py` 另有一处**进程级改绑定**（这正是 FG-12 提到的
那种手法）：它在 `build_env` 之后把 `env.evaluator.ode_accel` 改写掉。
FG-15 起这里必须**连谱档一起改**，因为 `for_fiber` 会把
`(ode_accel, raman_spectrum)` 一起透传给子 evaluator，只改前者会让子构造
`tabular` + `off` 而抛错。

#### 机制 B：「默认档是什么」本身就是命题 ⇒ 2 条期望值必须移动

| 文件 | 原期望 | 新期望 | 机制 |
|---|---|---|---|
| `tests/test_fg_u1_raman_spectrum.py::test_default_tier_is_legacy` | `__defaults__[-1] == "legacy"` | `== "tabular"`（函数改名 `test_default_tier_is_tabular`） | 这条测试的命题**就是**"默认档是什么"。换底必然让它移动——这是 FG-15 授权移动的既有期望值。同一条测试里对 legacy G 矩阵的核对保持显式 `_evaluator("legacy")`，因此**legacy 的数值仍被守住**。 |
| `tests/test_fu2_sclu_bands.py::test_legacy_path_still_bit_identical_for_fiber` | 默认构造后 `child.raman_spectrum == "legacy"` | 构造时**显式**钉 `legacy`，断言不变 | 这条测试的命题是「FU-2 补的 `for_fiber` **透传**」。跟着默认走会把命题悄悄换成"默认档透传"，而测试名与文档都在说 legacy。故**不改断言，改构造**。 |

**§3 的诚实声明**：机制 B 的第一行是本刀唯一**真正移动了既有期望值**的地方。
其余 46 条全部是"把原先隐式继承的值写成显式值"，**数值一个都没动**。

### 3.2 明确**不许**动、且确实没动的四处

| 项 | 为什么不许动 | 证据 |
|---|---|---|
| `results/fg1_golden_main.json` | 它声明 `source_branch: origin/main`；FG-13 的纪律是**任何声称 origin/main 出处的文件都不得写入本刀数值** | 本刀**未触碰**该文件；`tests/fg1_physics_parity.py` 与 `tests/fg1_scenarios.py` 的 evaluator 显式钉 `legacy`，因此 FG-1 (A) 仍逐位对 origin/main |
| FG-1 (B) 装配分歧框架（FG-14 刚立的） | 那是**出处**不变量，与物理档位无关 | FG-14 的守卫 `fg14_relocation_violations` 在本刀后仍绿 |
| `results/fg4a_golden.json` 的数值 | 加速线的金标准；谱档与加速是正交变量 | `--check` **31/31 逐位相等**，worst 相对偏差 `0.000e+00`（见 §6 门 3） |
| `results/fu4_fidelity_golden.json` 的数值 | FU-4 问的是 SNR **装配**（并集口径），不是选哪条谱 | `--check` **逐位复现**（见 §6 门 3） |

FG-4a 与 FU-4 两个金标准之所以能**一个 bit 都不动**，正是因为本刀
**没有**把它们重钉，而是在它们的构造点上把谱档显式钉成它们采集时实际生效的值
（`legacy`）。这比"重新采集一次并记出处链"更诚实：重新采集会把一次
**物理换底**伪装成一次**金标准刷新**，从而永久失去"加速线没动过"的证据。

### 3.3 G2 守卫：授权是**记进去**的，不是绕过的

`tests/fg4b_tier_pin_policy.py` 是一道既有守卫：它对 FG-4a 基线 `ea7ecb0`
之后被修改的**既有**测试文件做授权检查。本刀改动的文件里，**两个早于基线**
（因而受守卫管辖）：

- `tests/fg1_scenarios.py` —— 在 `STRICT_TIER_PIN_FILES` 里，受**行级**检查：
  新增的代码行必须"要么是 tier pin 本身，要么是被删行的碎片"。
  本刀新增的 `raman_spectrum="legacy",` 按原规则**会被判违规**。
- `tests/test_fg1_physics_parity.py` —— 在 `FG14_AUTHORIZED_FILES` 里，
  额外受"迁走的测试不得超出授权名单"与 token 计数检查。

处理方式**不是放宽守卫**，而是按 FG-14 已有的先例，在策略文件里**记一条
FG-15 授权**（`FG15_AUTHORIZED_FILES` + `FG15_PIN_LITERAL`），并让行级检查
认这个字面量。授权内容被夹得很死：

- 只允许**一个**新增字面量 `raman_spectrum="legacy"`；
- 该字面量钉的值，与这些文件换底前**实际生效的值完全相同**，故无期望值移动；
- `token_count_violations`（`assert` / `array_equal` / 容差 token 的个数）
  继续按文件守门，一个都不许少。

守卫实测：换底后 `test_f4a_pre_existing_tests_were_not_modified` 等 3 条**全绿**。

## 4. `legacy` 档没被删，且结果逐位不变

brief §4 要求一条测试钉住："`legacy` 仍可显式取到，且其结果与换底前逐位不变"。
落在 `tests/test_fg15_legacy_bitwise.py`。

**关键设计：逐位不"重新钉一个新数字"来证明，而是复用换底前就已经存在的金标准。**

| 命题 | 断言 | 为什么这样钉 |
|---|---|---|
| 档还在 | `"legacy" in RAMAN_SPECTRUM_MODES` 且显式构造成功 | §4 的字面要求 |
| 生效值是读回的 | `ev.raman_spectrum == "legacy"` | 不断言"我传了什么"，断言"生效的是什么" |
| **逐位不变** | 对 `ode_accel ∈ {off, cached_g}`，`build_scenarios(..., raman_spectrum="legacy")` 的**每一个**输出与 `results/fg4a_golden.json` **逐位相等** | 这份 golden 采集于 FG-4a 基线 `ea7ecb0`，即**换底之前**。它因此就是"换底前 legacy 的读数"本身。 |
| 覆盖没缩水 | golden 里的场景集 == 显式钉 legacy 时声明的场景集 | 防止"少跑几个场景所以没漂" |
| 换底确有牙齿 | 默认构造的 `tabular` 与显式 `legacy` 的 G 矩阵**不相同** | 若两者相同，这一刀就是空操作 |

最后一条尤其重要：它证明本刀**不是**一个把两个名字指向同一条曲线的改名。

参数化规模：7 个场景 × 输出键 × 2 个 bit-exact 档位，合计 **65 条**用例。
换底前跑这套用例时，唯一会红的是"默认档是 tabular"那一条（换底前它理应红），
其余 64 条全绿——这直接说明**换底没有碰 legacy 的任何一个 bit**。

复算：

```bash
cd <fg-tabdflt>
PYTHONPATH=$PWD/src $HOME/miniconda3/envs/rl/bin/python \
    -m pytest tests/test_fg15_legacy_bitwise.py -v
```

### 3.4 RL 侧：决策基线重钉（**唯一真正移动期望值的下游**）

物理仓被钉住之后，RL 是**唯一**还会被换底影响的地方。

**改前/改后读数**（同一组 10 个数值/决策测试文件，`rl` 解释器，跨仓
`PYTHONPATH=<rl>/src:<physics>/src`）：

| | passed | failed | skipped | 墙钟 |
|---|---|---|---|---|
| 换底前 | 163 | 0 | 1 | 773 s |
| 换底后（重钉前） | 161 | **2** | 1 | 817 s |

失败的恰好两条，且是**同一个 fixture** 的两条守卫：

- `tests/test_env_channel_choice.py::test_channel_none_is_bit_identical_to_the_stage4_baseline`
- `tests/test_env_bounded_retry.py::test_channels_none_path_is_bit_identical_to_pre_change_baseline`

两者都读 `tests/fixtures/stage4_env_channel_baseline.json`（40 步决策序列）。

**归因取证**：`tools/fg15_tier_rebaseline.py` → `evidence/fg15_tier_rebaseline.json`
（**只取证，不改期望值**，与重钉分开，这样归因证据不会被重钉动作污染）。读数：

- 谱档读回：`legacy→legacy`、`tabular→tabular`（不信传参）。
- **自检通过**：用 `legacy` 独立重跑，**逐位复现**已提交的旧 fixture
  （`mismatching_steps: []`）。这条很关键——它证明除换底外**没有别的漂移**，
  所以下面的分叉可以干净地归因给换底。
- **9 处分叉，全部只落在 `channel` 一个字段上**：

| step | legacy 选 | tabular 选 |
|---:|---:|---:|
| 2 | 52 | 51 |
| 6 | 0 | 126 |
| 10 | 61 | 126 |
| 17 | 69 | 2 |
| 19 | 94 | 101 |
| 20 | 94 | 114 |
| 22 | 96 | 117 |
| 30 | 17 | 18 |
| 32 | 41 | 56 |

- **分桶总数完全不变**：`steps=40`、`blocked=12`、`blocked_qot=12`、
  `blocked_no_spectrum=0`、`buckets_consistent=True`（换底前后一致）。

**这个"总数不变、选道改变"的组合是一个实质结论，不是巧合**：换底改变的是
**每条候选信道的 SNR 数值**，因此 first-fit 的"第一个通过门限的信道"会换人；
但**是否可达门限**这一层判定在这次换底下没有改变，所以阻塞计数与奖励分布不动。
换句话说：**换底没有改变可达性拓扑，只改变了同层内的排序。**

**重钉**：跑 `tools/gen_env_channel_baseline.py` 重新生成，并在 `provenance` 里
补上谱档字段——这正是 §1.4 指出的缺口（此前 fixture 里**没有任何字段**能回答
"这次跑的是哪条谱"）：

```
"raman_spectrum": "tabular",
"raman_spectrum_source": "env.evaluator.raman_spectrum（运行期读回，非传参意图）",
"physical_regime": "...；FG-15 谱档默认 tabular",
"previous_commit": "0a123e9",
"previous_physics_commit": "FG-14 a27e42a 整合物理，但全局默认谱档为 legacy（合成折线）",
```

比较字段、逐位比较、步数（40）与场景参数**一律未动**；重钉后两条守卫
连同新配置面测试共 **22 passed**。

---

## 5. 速度线：换底的代价

口径：同进程交错 A/B（`A,B,B,A,A,B…`，逐轮翻转顺序），取 **min 样本**之比，
并记录全程 loadavg。计时是**补充**证据，结论以确定性计数为准。

| `ode_accel` | legacy (min) | tabular (min) | tabular 相对开销 | peak loadavg | 可报比值？ |
|---|---:|---:|---:|---:|---|
| `matvec`（默认） | 1.262 ms | 1.273 ms | **+0.83 %** | 7.31 | ✅ < 12 |
| `cached_g` | 5.451 ms | 5.473 ms | **+0.40 %** | 7.19 | ✅ < 12 |
| `off` | — | — | **结构上不可能** | — | 见下 |

确定性计数（`results/fg15_tier_speed_ab.json`）：`matvec` 下
`g_misses=2, nli_hits=1, nli_misses=1`；`cached_g` 下 `g_misses=2`。

**怎么读这张表**：

1. **换底几乎不要钱**：两条可比的 ODE 档位开销分别 +0.83 % 与 +0.40 %，都在
   测量噪声量级。原因是两档都走"预计算 G 矩阵"这条路——`g_misses=2` 说明每个
   档位的 G 只在预热时建一次，缓存命中后热路径**不再触碰谱形**，因此换一条谱
   几乎不改变热路径成本。
2. **`ode_accel="off"` 在 tabular 下**结构上不可能**：它是 FG-2 冻结的、只在核内
   逐对重建 legacy 折线的参照路径，`tabular`+`off` 被显式拒绝（已实测抛
   `ValueError`）。这不是性能结论，是**能力边界**，换底只影响可跑的档位集合，
   不影响它们的相对速度。

---

## 6. 验收门读数

| 门 | 结果 | 说明 |
|---|---|---|
| 门 1 物理仓全量 | **404 passed / 1 skipped** | 基线 FG-14 是 338/1；差额 +66 全部来自本刀新增的 §4 逐位用例（65 条）+ 改名后的"默认档是 tabular"那一条。**零失败**。 |
| 门 2 RL 仓全量 | 见 §8（**未跑完**） | 见下方未闭合项 |
| 门 3 FG-4a golden `--check` | **OK，31/31 逐位相等，worst 相对偏差 `0.000e+00`** | 金标准**未重钉**，故"加速线没动过"仍是可查的事实 |
| 门 3 FU-4 golden `--check` | **OK，逐位复现** | 同上 |
| 门 4 `legacy` 逐位 | 64 条 bit-exact 全绿 | 见 §4 |

---

## 7. 复算命令

两棵树：

```bash
TAB=/home/zsig/Documents/gits/wt/fg-tabdflt          # 物理仓 wp/fg-tabdflt
RT=/home/zsig/Documents/gits/wt/fg-rt-tab            # RL 仓  wp/fg-rt-tab
PY=$HOME/miniconda3/envs/rl/bin/python               # 必须用 rl 解释器（nlo 缺 np.trapezoid/gymnasium）
```

**先证明 import 落进 worktree，而不是主检出**（editable 安装的 `.pth` 指向主检出，
与 cwd 无关；不设 `PYTHONPATH` 就会跑另一棵树的代码，绿得毫无意义）：

```bash
PYTHONPATH=$TAB/src $PY -c 'import network_nlo_eval as m; print(m.__file__)'
PYTHONPATH=$RT/src:$TAB/src $PY -c 'import multiband_optical_rl as r; print(r.__file__)'
```

| 门 | 命令 |
|---|---|
| 门 1 物理仓全量 | `cd $TAB && PYTHONPATH=$TAB/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb $PY -m pytest -q` |
| 门 2 RL 仓全量 | `cd $RT && PYTHONPATH=$RT/src:$TAB/src NUMBA_CACHE_DIR=/tmp/nb $PY -m pytest -q` |
| 门 3 FG-4a golden | `cd $TAB && PYTHONPATH=$TAB/src $PY scripts/fg4a_capture_golden.py --check` |
| 门 3 FU-4 golden | `cd $TAB && PYTHONPATH=$TAB/src $PY scripts/fu4_capture_golden.py --check` |
| 门 4 legacy 逐位 | `cd $TAB && PYTHONPATH=$TAB/src $PY -m pytest tests/test_fg15_legacy_bitwise.py -v` |
| §2 配置面读回 | `cd $RT && PYTHONPATH=$RT/src:$TAB/src $PY -m pytest tests/test_fg15_raman_tier_surface.py -v` |
| §5 档位计时 A/B | `cd $TAB && PYTHONPATH=$TAB/src $PY scripts/fg15_tier_speed_ab.py --out results/fg15_tier_speed_ab.json` |
| §3 RL 决策基线重钉 | `cd $RT && PYTHONPATH=$RT/src:$TAB/src $PY -m tools.gen_env_channel_baseline` |
