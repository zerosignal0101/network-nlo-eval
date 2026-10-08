# FG-20 §4：门读数与交付账目

- worktree：`wp/fg-guard` @ `/home/zsig/Documents/gits/wt/fg-guard`，从 `9115da2` 切
- RL 树（**未改**）：`/home/zsig/Documents/gits/wt/fg-rt` @ `9ca3ec2`
- **未推 `origin`**；`origin/main` 保持 `87850a1`
- 本机负载：门 1 前后 `loadavg` 2.45 / 2.69（详见各节），**全程未长时间占满 CPU**

> 每条命令都显式带 `PYTHONPATH=<repo>/src` 并打印实际 import 路径。

---

## 0. TL;DR

| 门 | 结果 |
|---|---|
| 门 1 全量 | **477 passed / 2 skipped / 0 failed**（基线 421/1；新增 56 passed + 1 skipped，**无新增失败**） |
| 门 2 golden | fg4a **31/31 逐位**（worst `0.000e+00`，0 differing）；fu4 **10 场景 bit-exactly**。**均未移动** |
| 门 3 变异 | 行为测试 **5/5 全 RED**；FG-18 AST 守卫**只**抓 M5 ⇒ 盲区实证 |
| 门 4 RL 决策 | **30/30 逐项相同**，`n_mismatches = 0` |

## 1. 门 1：物理仓全量不得新增失败

| 侧 | 读数 | 计时 | `loadavg`（前 / 后） |
|---|---|---|---|
| 基线 `9115da2` | **421 passed / 1 skipped / 0 failed** | 61.24 s | 2.87 / 2.83 |
| 本刀 | **477 passed / 2 skipped / 0 failed** | 36.27 s | 2.45 / 2.69 |

**完整账目**（新增 N 条 ⇒ 最终 M 条，不拿某个数字顶替）：

| 项 | 数 |
|---|---|
| 基线 passed | 421 |
| 新增 passed（`test_fg20_roadm_injection_behavior.py` 45 + `test_fg20_none_roadm_adjudication.py` 11） | **+56** |
| 本刀 passed | **477** ✅ 与 421+56 相符 |
| 基线 skipped | 1 |
| 新增 skipped（`test_evaluate_path_tolerates_none_member_end_to_end`，需 `physics_nlo_eval`） | **+1** |
| 本刀 skipped | **2** ✅ |
| **新增失败** | **0** ✅ |

⚠️ **两处计时不可当加速比读**：两跑的**测试集合不同**（477 vs 421 条），且
`numba` 的 `@njit(cache=True)` 缓存第二次是热的。**本刀不报任何加速比**
（brief 纪律：计时须同集合同口径）。`loadavg` 全程 < 3，未干扰 FG-19 的负载敏感档。

## 2. 门 2：golden 逐位

| golden | 读数 |
|---|---|
| `fg4a_capture_golden.py --check` | **31/31 cases bit-identical，worst relative deviation `0.000e+00`，0 differing elements** |
| `fu4_capture_golden.py --check` | **10 scenarios，OK: fidelity golden reproduces bit-exactly** |

两者都**未移动** ⇒ **无需逐条归因与出处链**。与预期一致：本刀只加守卫、
`None` 防御与 `is_last_hop` 取证，改动**全部不可达**
（FG-18 §1.2 已证 `roadm_config` 在本仓恒为 `None`；门 4 再次独立确认）。

## 3. 门 3：新行为测试的变异读数

见 [`fg20_roadm_injection_behavior.md`](fg20_roadm_injection_behavior.md) §4
与 `results/fg20_mutation_test.json`。

**核心一行**：baseline 🟢/🟢；M1–M4 行为测试 🔴 而 AST 守卫 🟢
⇒ **源码守卫的盲区被实证**，行为测试 **5/5** 抓得住。

## 4. 门 4：RL 决策序列（**不必跑 RL 全量**，理由在下面）

`results/fg20_decision_compare.json`：

```
meta_equal        : true
steps_new/base    : 30 / 30
n_mismatches      : 0
identical         : true
n_allocated 新/基  : 10 / 10
```

- 新侧 import `/home/zsig/Documents/gits/wt/fg-guard/src/network_nlo_eval`
- 基线侧 import `/tmp/fg20_base_9115da2/src/network_nlo_eval`
  （`git archive 9115da2` 解出的只读副本；已核实其
  `physics/qot_checker.py:319` 仍是旧的崩溃写法、`_active_channel_mask` 出现 **0** 次）
- RL 侧两侧同为 `/home/zsig/Documents/gits/wt/fg-rt/src`（本刀**未改** RL）

### 为什么不必跑 RL 全量（结构证据 + 实测证据）

**结构证据（grep 全链，非推断）**：

1. `grep -r "physics\.qot_checker" /home/zsig/Documents/gits/wt/fg-rt/src`
   → **无任何匹配**。RL 树**从不** import 本刀改动的那个模块。
2. 物理仓内 `physics.qot_checker` 只被 `physics/comparison.py:16,85` import，
   而 `comparison.py` 只在 `run_waveform=True` 时才走波形分支。RL 走**统计**侧。
3. 其余改动全在 `tests/` 与 `scripts/`，RL 运行时**不加载**。

**实测证据**：固定 seed（`seed=0`）rollout 30 步，逐步
`(action, channel, allocated)` **30/30 逐项相同**，`n_mismatches = 0`。

⇒ 若本刀真触到了可达路径，`n_mismatches` 会立刻非零。**它没有。**
故**未跑** RL 全量分片（基线 `9ca3ec2` 的 763 passed / 0 failed / 2 skipped
本刀**未复跑**，故**不引用**该数作为本刀的通过证据）。

⚠️ **与 FG-18 的读数不可直接对比**：FG-18 记录的 trace 是它**自己**临时写的
策略/场景（`n_allocated = 30`）；本刀的生成器也是新写的（`n_allocated = 10`）。
两者**互不可比**。可比的是本刀**自己两侧**的 30/30 逐项相同——两侧用**同一个**
生成器，唯一变量是物理仓的 `src/`。

## 5. 交付物清单

| 文件 | 性质 |
|---|---|
| `src/network_nlo_eval/physics/qot_checker.py` | **§1 裁决落地**（唯一改动的 `src/` 文件） |
| `tests/test_fg20_none_roadm_adjudication.py` | §1 行为守卫（12 收集：11 passed + 1 skipped） |
| `tests/test_fg20_roadm_injection_behavior.py` | §2 行为回归（45 条） |
| `scripts/fg20_mutation_test.py` | 门 3 变异读数 |
| `scripts/fg20_is_last_hop_forensics.py` | §3 取证（只报不改） |
| `scripts/fg20_decision_trace.py` | 门 4 rollout 生成器 |
| `docs/fg20_none_adjudication.md` | 裁决 + §3 取证 |
| `docs/fg20_roadm_injection_behavior.md` | 行为测试设计 + 期望值来源 + 变异 |
| `docs/fg20_gates.md` | 本文件 |
| `results/fg20_*.json` | 全部数字的可复算产物 |

**未改**：`models/numba_kernels.py`、ODE 内核、`rwa/` 并集口径的语义、
`learn/`、`oracle_provision.py`、`feasibility.py`、`reopt/`、
任何既有测试的期望值、`origin/*`。**未新增依赖。**

## 6. 复算命令（全部）

```bash
RL=/home/zsig/miniconda3/envs/rl/bin/python
cd /home/zsig/Documents/gits/wt/fg-guard
export PYTHONPATH=/home/zsig/Documents/gits/wt/fg-guard/src

# 0) 证明 import 落在这棵 worktree 内（不带 PYTHONPATH 会跑到主检出）
$RL -c "import network_nlo_eval; print(network_nlo_eval.__file__)"
#   -> /home/zsig/Documents/gits/wt/fg-guard/src/network_nlo_eval/__init__.py

# 1) 门 1：全量
$RL -m pytest -q            # -> 477 passed, 2 skipped

# 2) 门 2：golden
$RL scripts/fg4a_capture_golden.py --check   # -> 31/31 bit-identical, worst 0.000e+00
$RL scripts/fu4_capture_golden.py --check    # -> 10 scenarios bit-exactly

# 3) 门 3：五向变异
$RL scripts/fg20_mutation_test.py           # -> results/fg20_mutation_test.json

# 4) 门 4：RL 决策序列（两侧同一生成器，唯一变量是物理仓 src/）
cd /home/zsig/Documents/gits/wt/fg-rt
PYTHONPATH=/home/zsig/Documents/gits/wt/fg-guard/src:/home/zsig/Documents/gits/wt/fg-rt/src \
  $RL /home/zsig/Documents/gits/wt/fg-guard/scripts/fg20_decision_trace.py \
      --out /home/zsig/Documents/gits/wt/fg-guard/results/fg20_rl_decision_trace.json

B=/tmp/fg20_base_9115da2 && mkdir -p $B
cd /home/zsig/Documents/gits/wt/fg-guard && git archive 9115da2 | tar -x -C $B
cd /home/zsig/Documents/gits/wt/fg-rt
PYTHONPATH=$B/src:/home/zsig/Documents/gits/wt/fg-rt/src \
  $RL /home/zsig/Documents/gits/wt/fg-guard/scripts/fg20_decision_trace.py \
      --out /home/zsig/Documents/gits/wt/fg-guard/results/fg20_rl_decision_trace_baseline.json

# 5) §3 取证
$RL scripts/fg20_is_last_hop_forensics.py
```

## 7. 未闭合项（交回编排方）

1. **波形侧端到端读数未测量**：本机**无 CUDA toolkit**（有 P100 驱动与 GPU，
   但无 `/usr/local/cuda` / `nvcc` / `libcudart`），`cupy` 无法编译 kernel。
   既有的 `test_waveform_route_smoke` 在 `nlo` 下报**同一个错**（已复核），
   故是环境限制。⇒ `physics/qot_checker.py` 的改动**只被 `rl` 下的单元级用例覆盖**，
   **未经真实波形端到端验证**。
2. **RL 全量门未跑**（理由见 §4，附结构 + 实测双证据）。基线 `9ca3ec2` 的
   763/0/2 未复跑，**不引用**为通过证据。
3. **`add_drop_loss_db` 在逐跨段层结构性不可见**、**`filtering_penalty_db`
   在逐跨段层不施加**——两条都如实记录，未改语义，待裁决。
4. **`is_last_hop` 的末跳支持**未落地（brief 要求只取证）。外部调用方传末跳
   ROADM 时会拿到 express 口径，默认配置差 **0.074 dB**。建议 release note 点名。
5. **可达性结论的边界**：`roadm_config` 非 `None` 计数 0 覆盖的是**本仓选定
   测试面**（FG-18 探针 790 次 + 本刀 §3 的 AST 全链扫描），不等于"全宇宙不可达"。
   外部调用方仍可传非 `None` ⇒ 对它们是**行为变更**。
6. **batch 入口只测了 `B=1`**（见 `fg20_roadm_injection_behavior.md` §7.3）。
