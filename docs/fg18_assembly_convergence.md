# FG-18：关掉源码守卫的盲区 + 收敛 `isrs_gn.py` 的两处合并式装配

- **worktree（物理）**：`wp/fg-isrs` @ `/home/zsig/Documents/gits/wt/fg-isrs`，从 `1004458` 切
- **worktree（RL）**：`wp/fg-rt-is` @ `/home/zsig/Documents/gits/wt/fg-rt-is`，`9ca3ec2`（**未改 RL 侧任何文件**）
- **解释器**：物理仓全量 + RL 侧 = `rl`；波形级 / `physics_nlo_eval` / `cupy` = `nlo`
- **未推 `origin`**；`origin/main` 保持 `87850a1`

> 每条命令都显式带 `PYTHONPATH=<repo>/src` 并打印实际 import 路径。理由：本仓是
> editable 安装，`.pth` 指向**主检出** `/home/zsig/Documents/gits/network-nlo-eval/src`，
> 与 cwd ��关。实测确认：不带 `PYTHONPATH` 时 `rl` 仓测试 import 到的是主检出。

---

## 0. TL;DR

| 问题 | 结论 |
|---|---|
| 两处是不是 FG-13 同型缺陷？ | **是**。拿不出"刻意简化"的设计契约原文，按缺陷处理（与 FG-13 同一裁决口径） |
| 是不是 ODE 内核？ | **不是**。两处都在内核 `return` **之后**的 Python 装配层，不触碰 `_calc_span_noise_and_power_jit` / `numba_kernels.py` |
| 本仓是否真走到？ | **没走到**：790 次调用，`roadm_config` 非 None **0 次** |
| 现有场景的读数会动吗？ | **不动**。fg4a 31/31 逐位（worst `0.000e+00`）、fu4 逐位、RL 决策序列 30/30 逐项相同 |
| 守卫盲区关掉了吗？ | **关掉了**。两向变异：新守卫 A/B 均 **RED**；旧守卫对 A 向 **GREEN**（盲区被实证） |

---

## 1. §0 取证

### 1.1 两处的调用面

| 施加点 | file:line | 收敛前形状 |
|---|---|---|
| 逐跨段入口 | `src/network_nlo_eval/models/isrs_gn.py:528`（`if roadm_config:`） | `_db_to_lin(insertion_loss_db + filtering_penalty_db)` 后 `p_out_fiber /= loss_lin` |
| batch 入口 | `src/network_nlo_eval/models/isrs_gn.py:596`（`if roadm_config is not None:`） | 同一行，后 `p_out /= loss_lin` |

**同名方法不等于同一个函数**——逐个确认过接收者：

- 两处都是 `MultiBandISRSGN` 的**实例方法**，`self` 相同，但**布局不同**：串行返回
  `(N,)`，batch 返回 `(B, N)`。收敛时 booster 项按 `(N,)` 广播到 `(B, N)`。
- `_db_to_lin` **不在 `isrs_gn.py` 里定义**，是从 `models/numba_kernels.py:43`
  `import` 进来的 `@njit(cache=True)` 函数。本刀收敛后该 import 成为未使用，
  已删除（已核实无人从 `isrs_gn` 转出 `_db_to_lin`；`physics/metrics.py` 直接从
  `numba_kernels` import）。

**`roadm_config` 从哪来 / 在什么档位下被走到**：

| 调用方 | file:line | 传了什么 `roadm_config` |
|---|---|---|
| `rwa/link_noise.py` 整跨 | `src/network_nlo_eval/rwa/link_noise.py:72` | 位置参数 `None` |
| `rwa/link_noise.py` 余段 | `src/network_nlo_eval/rwa/link_noise.py:85` | **不传** → 默认 `None` |
| `evaluate_path` 编排 | `src/network_nlo_eval/models/isrs_gn.py:712` | **不传** → 默认 `None`（**关键**） |

⇒ **`evaluate_path` 这个真正的端到端路径，恰恰也不给逐跨段入口传 ROADM**。
它在**编排层**自己调 `fold_roadm_terms`（`isrs_gn.py:730`），不指望下层处理 ROADM。
这是"两处在本仓不可达"的**结构性**原因，不是巧合。

**RL 侧**：`/home/zsig/Documents/gits/wt/fg-rt-is/src` 里 grep 不到任何对这两个方法的
调用（只有 `scan.py` 的注释提到 `_calc_span_noise_and_power`）；RL 走的是
`scan.py:234` 直接调 `fold_roadm_terms`。

### 1.2 实测可达性（不是看调用点，是运行时挂钩）

`scripts/fg18_reachability_probe.py` + `scripts/fg18_probe_plugin.py`：把两个方法的
`roadm_config` 形参包一层计数器，逐测试文件跑。

**读数**（`results/fg18_reachability.json`）：

| 测试文件 | `_calc_span_noise_and_power` 调用数 | 非 None | `..._batch` 调用数 | 非 None |
|---|---|---|---|---|
| `test_fg4e_cpu_batching.py` | 515 | **0** | 50 | **0** |
| `test_fg4g_matvec_attribution.py` | 92 | 0 | 0 | 0 |
| `test_fu4_assembly_union.py` | 66 | 0 | 0 | 0 |
| `test_fg4b_nli_vectorization.py` | 22 | 0 | 0 | 0 |
| `test_fg13_evaluate_path_roadm.py` | 22 | 0 | 0 | 0 |
| `test_fg1_physics_parity.py` | 16 | 0 | 0 | 0 |
| `test_topology_physical_integration.py` | 4 | 0 | 0 | 0 |
| `test_fg2_flexible_grid.py` | 3 | 0 | 0 | 0 |
| **合计** | **790** | **0** | **50** | **0** |

⚠️ **本探针自己踩过一次坑，必须写下来**：初版只在"非 None"时落盘计数，于是全套件
读出 `0/0`，看起来像"方法从没被调用"——那是**假阴性**。修法是每次调用都落盘，
并加 `probe_live = totals["calls"] > 0` 自检，`calls == 0` 时打印
`PROBE INVALID` 且**禁止**把结果当可达性证据。上表的 790 就是自检通过后的读数。

**结论的诚实边界**：这证明的是**本仓选定测试面不可达**，不等于"全宇宙不可达"。
外部调用方（用户代码、RL 仓）仍可能传非 None——本仓只是**没有**在传。

### 1.3 公式级对照

收敛前的合并式（FG 树口径）：

```
loss_lin = 10 ** ((insertion_loss_db + filtering_penalty_db) / 10)
p_out   /= loss_lin          # 只衰减信号
```

权威折叠 `rwa/qot_checker.py:115 fold_roadm_terms`（FG-5 并集裁定为权威）：

```
loss_db = config.path_loss_db("express")
        = insertion_loss_db + express_attenuation_db      # elements.py:113-116
loss    = 10 ** (loss_db / 10)
filtering_penalty = 10 ** (filtering_penalty_db / 10)
booster_base      = 10 ** (booster_noise_figure_db / 10) * H_PLANCK

if equalize:  # booster 把信号拉回入纤值 ⇒ 信号不衰减，改注入 ASE
    ase += booster_base * f * Rs * (loss - 1)
else:
    p /= loss ; spm /= loss ; xpm /= loss ; ase /= loss
```

逐项差异：

| 项 | 收敛前（合并式） | 收敛后（权威折叠） | 是否等价 |
|---|---|---|---|
| `insertion_loss_db` | 用 | 用 | — |
| `express_attenuation_db` | **丢弃** | 并入 `path_loss_db("express")` | ❌ |
| `filtering_penalty_db` | **并进信号光功率** | 独立噪声域乘子，**不进光功率** | ❌（量纲错误：SNR 代价被当成功率损耗） |
| `add_drop_loss_db` | 丢弃 | 取决于 `operation`；逐跨段层固定 express，故取 0 | ❌（但见下"端口"行） |
| `booster_noise_figure_db` | **完全未用** | 进 `booster_base` | ❌ |
| `equalize_output_power` | **忽略**，恒衰减信号 | 为真则不衰减 + 注 booster ASE | ❌ |
| 端口 vs 路径 | 不区分 | 末跳 `drop`、其余 `express` | ❌ |
| 噪声是否随之缩放 | 否（只动 `p_out`） | `equalize=False` 时三个噪声分量**同时**除 `loss` | ❌ |

**末跳 `drop` 的取舍（必须显式记录）**：逐跨段 API **拿不到 hop 序号**——
`evaluate_path` 靠 `is_last_hop = (hop_index == n_hops - 1)` 决定，而那层才知道拓扑。
两处收敛后固定用 `is_last_hop=False`（express）。这是本层唯一可得的确定性选择，
已写进代码注释。`drop` 与 `express` 只差 `add_drop_loss_db`（默认 1.0 dB）。

### 1.4 判性质

**判为 FG-13 同型缺陷**，不是刻意简化。理由：拿不出任何"这两处刻意简化"的设计契约
原文；同一文件的 `evaluate_path` 已在编排层用权威折叠并写明"与另外两处施加点同口径"
（`isrs_gn.py:678-684`），**同一文件里两个 ROADM 施加点用两套口径**，本身即是缺陷证据。

---

## 2. §1 收敛（最小改动）

两处都改为懒导入 `rwa.qot_checker.fold_roadm_terms` 并按其口径装配：

- **不新增文件**（除取证脚本与守卫测试）、**不新增抽象**、**不加开关**
- **对外签名不变**：`roadm_config` 仍是第 4 个位置形参
- 懒导入理由与 `evaluate_path:700-702` 相同：`rwa.qot_checker` 在模块级反过来取
  `MultiBandISRSGN`，模块级互相导入会成环
- 删除随之未使用的 `_db_to_lin` import

⚠️ **ODE 内核未改动**（brief §2 的停手条件）：两处都在
`_calc_span_noise_and_power_jit(...)` / `batched_span_noise_and_power(...)`
**返回之后**，属装配层。`numba_kernels.py` 一字未动，内核逐位保证不受影响。

⚠️ **"往返误差" vs "语义变化"**（brief §2 的区分要求）：本刀的移动**全部是语义变化**，
**没有一项是 dB↔线性往返误差**。理由：收敛后直接用折叠的 `loss`
（`10 ** (loss_db / 10)`），**没有**做 `10*log10(loss)` 的反算，因此不引入 FG-17 记录的
1 ulp 级往返误差。这一点在 §3 的数字上可直接验证：所有 `delta_dB` 都是 `0.2 / 2.0 /
1.8` 这类**量级差异**，没有一个是 1e-16 量级。

---

## 3. §2 守卫：从单文件扩到全 `src/` 树

新增 `tests/test_fg18_lumped_assembly_tree_guard.py`（17 个用例）。
**既有的 `test_fu4_assembly_union.py` 一字未改**（断言未放宽）。

### 3.1 为什么用 AST 而不是正则

正则扫全树会**误伤合法写法**——FG-18 自己在 `isrs_gn.py:527` 和 `:622` 的注释里就写了
`insertion_loss_db + filtering_penalty_db` 这串字面量（说明"不再在本处合并…"）。
正则会把注释和文档字符串判成回归。**踩到自己的注释正是把断言改宽的起点。**

按 AST 判定：只匹配真正参与加法的 `ast.BinOp(op=Add)`，两侧分别是
`.insertion_loss_db` 与 `.filtering_penalty_db` 的**属性访问**。注释、docstring、
字符串常量在 AST 里根本不出现 → 天然不误伤。**不用文件白名单**排除任何 `src/` 文件。

### 3.2 两向变异测试读数

`scripts/fg18_mutation_test.py` → `results/fg18_mutation_test.json`。
在**临时树副本**上注入变异（不碰工作树），跑守卫读判决。

| # | 注入位置 | 守卫 | baseline（无变异） | 变异后 |
|---|---|---|---|---|
| A | `models/isrs_gn.py`（FG-17 盲区） | **新**全树守卫 | GREEN | 🔴 **RED**（2 failed: 全树扫描 + 盲区文件专项） |
| A2 | `models/isrs_gn.py` | **旧**单文件守卫 | GREEN | 🟢 **GREEN** ← **盲区被实证** |
| B | `rwa/qot_checker.py` | **新**全树守卫 | GREEN | 🔴 **RED**（2 failed: 全树扫描 + 旧单文件专项） |
| B2 | `rwa/qot_checker.py` | **旧**单文件守卫 | GREEN | 🔴 **RED** |

**A2 = GREEN 就是 FG-17 报告里那条盲区的直接实验证据**：把合并式装配塞回
`isrs_gn.py`，旧守卫**通过**。FG-18 的新守卫在同一边**变红**。

**突变测试在临时树上跑，不改工作树**，跑完删除。

⚠️ **变异测试自己也踩了两个坑，都已修**（都是"探针失效被误读成判决"）：
1. 初版只复制 `src/ tests/`，缺 `results/` ⇒ 旧守卫的兄弟测试**收集失败**（rc=4），
   被我按 `rc != 0` 误判成 RED。修法：复制 `results/`，并把 rc=4/5 单列为
   `INVALID`，不计入判决。
2. 相对 node id `tests/...::name` 在临时树 rootdir 下被解析成缺前缀的路径 ⇒
   rc=4。修法：一律传**绝对路径**。

修完后 `verdict.all_valid = true`，四个方向全部有效读数。

### 3.3 检测器自检（防"改宽"）

- `test_detector_catches_regression_shapes`（7 例）：换接收者、换左右顺序、
  删空格、加括号、嵌进深层表达式、batch 入口形状 —— **必须全部被抓**
- `test_detector_does_not_fire_on_legal_forms`（7 例）：注释字面量、docstring、
  `config.path_loss_db(...)` 调用、分别取值、减法、`path_loss_db` 自身的
  `insertion_loss_db + port_loss` —— **必须全部不被抓**
  （最后一条尤其重要：`elements.py:116` 本身就是加法，但加的是
  `insertion_loss + port_loss`，**不是** `filtering_penalty`，是合法口径实现）

---

## 4. §3 碰撞面逐条归因

`scripts/fg18_assembly_ab.py` → `results/fg18_assembly_ab.json`。

**先说清楚为什么必须在"装配函数层"做 A/B**：§1.2 已证明本仓
`roadm_config` 恒为 `None`，所以**任何现有场景上改前改后必然逐位相同**——
门 2 的 `0.000e+00` 正是这件事的读数。机制只能靠直接对装配函数做 A/B 暴露。

| config | equalize | before dB | after dB | ΔdB | 逐位相同 | 机制 |
|---|---|---|---|---|---|---|
| `fu4_base` | True | 5.200 | 5.000 | −0.200 | ❌ | 惩罚 0.2 dB 不再并进光功率 |
| `high_filtering` | True | 7.000 | 5.000 | −2.000 | ❌ | 同上，放大到 2.0 dB |
| `express_attn_nonzero` | True | 5.200 | 7.000 | **+1.800** | ❌ | 新纳入 `express_attenuation_db=2.0` |
| `equalize_off` | False | 5.200 | 5.000 | −0.200 | ❌ | 惩罚移出；两分支都衰减信号 |
| `zero_filtering` | True | 5.000 | 5.000 | **0.000** | ❌ | ΔdB=0 **仍非逐位相同** |
| `zero_express_attn` | True | 5.200 | 5.000 | −0.200 | ❌ | 惩罚移出 |

**合计：6/6 移动，0 逐位相同。** 两类机制，逐条：

1. **`filtering_penalty_db` 移出光功率** → Δ = −`filtering_penalty_db`。
   这是**量纲修正**：级联滤波的 SNR 代价本来就不该被当成信号功率损耗。
2. **`express_attenuation_db` 纳入** → Δ += `express_attenuation_db`。

⚠️ **`zero_filtering` 一行值得单独说**：ΔdB 恰好 `0.000`，但**不是**逐位相同。
原因是它 `equalize_output_power=True`：收敛后走 booster 分支（**完全不衰减信号**，
改注 ASE），而收敛前无论 `equalize` 与否都衰减。**只看 ΔdB 会漏掉这类语义变化**，
所以本表同时给"逐位相同"列。FG-13 的 booster 事故正是这一类——FG-14 实测末跳
`3.55e-15` → `0.857183 dB`。

**本刀是保真线，允许并预期移动；但这些移动在本仓落不到任何现有读数上**，
因为两个分支在本仓不可达（§1.2）。

### 4.1 不变量重钉

**无。** 本刀**没有**重钉任何既有期望值：`fg4a_golden.json`、`fu4_fidelity_golden.json`、
`fu4_assembly_delta.json` 全部**未改**，`git status` 里没有它们。原因即上：
不可达 ⇒ 无读数移动。FG-14 建立的 (A)/(B) 出处框架**未被扩大**——
`provenance.repins[]` 零新增。

---

## 5. §4 门读数

### 门 1：全量门

| 仓 | 基线 | 本刀 | 判定 |
|---|---|---|---|
| 物理仓 | `1004458`：404 passed / 1 skipped | **421 passed / 1 skipped / 0 failed** | ✅ 新增 17 = 新守卫全部用例，无新增失败 |
| RL 仓 | `9ca3ec2`：763 passed / 0 failed / 2 skipped | 见 §5.1 | ⏳ |

### 门 2：golden 逐位

| golden | 读数 |
|---|---|
| `fg4a_capture_golden.py --check` | **31/31 cases bit-identical，worst relative deviation `0.000e+00`，0 differing elements** |
| `fu4_capture_golden.py --check` | **10 scenarios，OK: fidelity golden reproduces bit-exactly** |

两者都**未移动**，故无需归因与出处链。

### 门 3：grep + 变异测试

收敛后 `src/` 全树只剩**注释**里那两处字面量（`isrs_gn.py:527`、`:622`），
**没有可执行代码**残留：

```
$ grep -rn "insertion_loss_db + filtering_penalty_db" --include=*.py src/
src/network_nlo_eval/models/isrs_gn.py:527:  # 不再在本处合并 ``insertion_loss_db + filtering_penalty_db``。
src/network_nlo_eval/models/isrs_gn.py:622:  # 不再在本处合并 ``insertion_loss_db + filtering_penalty_db``。
```

（这正是 §3.1 选 AST 而非正则的原因——正则会把这两行判成回归。）

`filtering_penalty_db` 的**唯一可执行定义点**仍是 `rwa/qot_checker.py:145`
（`fold_roadm_terms` 内），字段声明仍是 `network/elements.py:76`。

变异测试读数见 §3.2 表：**新守卫 A/B 均 RED，旧守卫 A 向 GREEN**。

### 门 4：RL 决策序列

固定 seed rollout（`seeded_random`，seed=0，30 requests），逐步
`(action, channel, allocated)`；两侧 RL 源码同为 `wt/fg-rt-is`（本刀未改 RL），
唯一变量是物理仓。

`results/fg18_decision_compare.json`：

```
meta_equal        : true
steps_new/base    : 30 / 30
n_mismatches      : 0
identical         : true
```

新侧 import `/home/zsig/Documents/gits/wt/fg-isrs/src/network_nlo_eval`，
基线侧 `/tmp/fg17_baseline_1004458/src/network_nlo_eval`（`git archive 1004458`
解出的只读副本，已核实其 `isrs_gn.py:528/596` 仍是合并式）。

**为什么本刀不影响决策路径**（grep/import 链）：
RL 侧 `scan.py:234` 调的是 `fold_roadm_terms`，**从不**经过 `isrs_gn` 的两个
`roadm_config` 施加点；且 §1.2 已证明那两个形参在本仓恒为 `None`。决策路径与本刀
**在结构上不相交**，与实测 `n_mismatches = 0` 一致。

---

## 6. §5 顺带取证：`physics/qot_checker.py::evaluate_path` 的 `None` 崩溃

**只取证，未修改**（brief 要求：修不修交编排方裁）。

`src/network_nlo_eval/physics/qot_checker.py:319-320`：

```python
for roadm in optical_path.roadm_configs or []:
    active[[index for index in roadm.blocked_channels if index < len(active)]] = False
```

`scripts/fg18_crash_probe.py` → `results/fg18_crash_probe.json`：

| `roadm_configs` | 结果 |
|---|---|
| `[ROADMConfig(), ROADMConfig()]` | 正常 |
| **`[ROADMConfig(), None]`** | 🔴 `AttributeError: 'NoneType' object has no attribute 'blocked_channels'` |
| **`[None, None]`** | 🔴 同上 |
| `None` | 正常（`or []` 兜住） |
| `[]` | 正常 |

⇒ **确实会抛**，FG-17 报告属实。

**有没有活调用方？会不会发生在生产路径？**

- 活调用方 3 处：`physics/comparison.py:90`（`run_waveform=True`）、
  `qot_checker.py:331`（`evaluate_topology_path`）、`qot_checker.py:347`
  （`evaluate_channel_evm_and_ber`，自建 path 且 `roadm_configs=None`）。
- **生产路径安全**：`NetworkTopology.build_optical_path`
  （`network/topology.py:173`）只 `append(self.get_roadm_config(v_idx))`，
  返回类型是 `ROADMConfig`，**结构上不可能产出 `None`**。
- 仓内**没有**任何活调用方会传含 `None` 的 `roadm_configs`。
  旁证：`scripts/fg17_waveform_ab.py` 的模块 docstring 明写"不构造含 `None` 的
  `roadm_configs`"，即 FG-17 已经知道这个坑并绕开了。

**结论**：这是一个**潜伏的契约不一致**，不是当前在发生的故障——
同一份 `MultiSpanOpticalPath` 传给统计侧 `isrs_gn.evaluate_path` 没问题，
传给波形侧 `physics.evaluate_path` 含 `None` 就炸。**修不修交编排方裁**：修它要动
`evaluate_path` 的输入契约（`None` 是"无 ROADM"还是"未知"），可能波及调用方。

⚠️ 本探针初版用 Python list 复刻那段代码，先撞上
`TypeError: list indices must be integers`，把真正的 `AttributeError` 掩盖了；
改用 numpy 后才是上面的读数。**与守卫那次的假阴性同源**——复刻必须与被复刻代码
的数据类型一致。

---

## 7. 未闭合项（交回编排方）

1. **RL 全量门未在本回合跑完**。FG-17 实测 ~2h+，本机负载高（见 §5.1）。
   若回合结束前仍未收齐，已完成分片的 JSON 在 `results/fg18_rl_gates/`，
   缺片用 `--only` 补跑（分片会随父进程一起死：**只认 JSON 汇总文件，
   不认控制台"看起来在跑"**）。
2. **`is_last_hop=False` 的取舍**：逐跨段层拿不到 hop 序号，本刀固定用 express。
   若将来让逐跨段入口支持末跳 `drop`，需要显式的 hop 位置参数——**本刀不加**（不加开关）。
3. **`physics.evaluate_path` 的 `None` 崩溃**：取证完毕，未修，等裁决。
4. **可达性结论的边界**：790 次调用只覆盖本仓选定测试面。外部调用方仍可传非 `None`；
   收敛对它们是**行为变更**（§3 表格），发布前需在 release note 里点名。
5. **未新增"两处施加点"的回归测试**：既然它们在本仓不可达，为它们写行为测试需要
   人工构造 `roadm_config`；本刀选择用**源码级守卫**（全树 AST 扫描）覆盖"不再退回
   合并式"这一性质。若要补行为测试，建议按 §3 表格的 6 个配置参数化。

---

## 8. 复算命令

全部命令都在 `/home/zsig/Documents/gits/wt/fg-isrs` 下执行，
**每条都带 `PYTHONPATH` 前缀**（否则 import 到主检出，跑的是另一棵树）。

```bash
RL=/home/zsig/miniconda3/envs/rl/bin/python
export PYTHONPATH=/home/zsig/Documents/gits/wt/fg-isrs/src

# 0) 证明 import 落在这棵 worktree 内
$RL -c "import network_nlo_eval; print(network_nlo_eval.__file__)"

# 1) §0 可达性（790 calls / 0 非 None）
$RL scripts/fg18_reachability_probe.py

# 2) §1 收敛的 grep 证据（只剩注释）
grep -rn "insertion_loss_db + filtering_penalty_db" --include=*.py src/

# 3) §2 守卫 + 既有守卫
$RL -m pytest tests/test_fg18_lumped_assembly_tree_guard.py tests/test_fu4_assembly_union.py -q

# 4) §2 两向变异测试（新守卫 A/B 均 RED；旧守卫 A 向 GREEN）
$RL scripts/fg18_mutation_test.py

# 5) §3 碰撞面归因表
$RL scripts/fg18_assembly_ab.py

# 6) 门 1 物理仓全量（421 passed / 1 skipped）
$RL -m pytest tests/ -q -p no:randomly

# 7) 门 2 golden
$RL scripts/fg4a_capture_golden.py --check
$RL scripts/fu4_capture_golden.py --check

# 8) 门 4 RL 决策序列 A/B（需要 RL 侧 src 也在 PYTHONPATH）
cd /home/zsig/Documents/gits/wt/fg-rt-is
export PYTHONPATH=/home/zsig/Documents/gits/wt/fg-rt-is/src:/home/zsig/Documents/gits/wt/fg-isrs/src
$RL scripts/fg17_rl_decision_trace.py -o .../results/fg18_rl_decision_trace.json
git -C /home/zsig/Documents/gits/wt/fg-isrs archive 1004458 | tar -x -C /tmp/fg17_baseline_1004458
PYTHONPATH=/home/zsig/Documents/gits/wt/fg-rt-is/src:/tmp/fg17_baseline_1004458/src \
  $RL scripts/fg17_rl_decision_trace.py -o .../results/fg18_rl_decision_trace_baseline.json
cd /home/zsig/Documents/gits/wt/fg-isrs && $RL scripts/fg18_decision_compare.py \
  --new results/fg18_rl_decision_trace.json \
  --baseline results/fg18_rl_decision_trace_baseline.json

# 9) 门 1 RL 全量（分片；只认 JSON 汇总）
$RL scripts/fg18_shard_suite.py --repo /home/zsig/Documents/gits/wt/fg-rt-is \
  --out results/fg18_rl_gates --jobs 2 --shards 8 \
  --pythonpath /home/zsig/Documents/gits/wt/fg-rt-is/src:/home/zsig/Documents/gits/wt/fg-isrs/src

# 10) §5 崩溃点取证
$RL scripts/fg18_crash_probe.py
```

### 产物清单

| 文件 | 内容 |
|---|---|
| `results/fg18_reachability.json` | 逐测试文件调用数 / 非 None 计数 |
| `results/fg18_mutation_test.json` | 两向变异判决 + baseline + `all_valid` |
| `results/fg18_assembly_ab.json` | 6 配置逐项 A/B + 机制 |
| `results/fg18_crash_probe.json` | `None` 崩溃复现 + 调用方 + 生产路径判定 |
| `results/fg18_decision_compare.json` | RL 决策序列逐项对比 |
| `results/fg18_rl_decision_trace*.json` | 两侧 rollout 原始轨迹 |
| `results/fg18_rl_gates/` | RL 全量分片日志 + 汇总 |