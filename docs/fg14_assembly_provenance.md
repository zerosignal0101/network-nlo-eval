# FG-14 · 阶段 0：出处分歧框架

> **本阶段零物理变化。** `src/` 下一个字节都没动。
> 本阶段只做一件事：把 `tests/test_fg1_physics_parity.py` 里揉在一起的
> **两句话**拆成 (A) 与 (B) 两条可分别裁决、可分别钉住的命题。

---

## §1 编排方 §0 那句待裁的回答

FG-13 §5 停在原地，问的是一句话：

> *"`evaluate_path` 与 `origin/main` 的逐位一致性（装配层面）算不算 §25 那四条不变量之一？"*

**答：算。** 但算它的内容**不是**"这么算是对的"，而是
**"这次重构系列没有改动 legacy 物理"**。

这是**出处**不变量，不是**物理**不变量。所以处置方式**不是把数字改掉**，
而是**把它所声称的命题说准确**：

| 命题 | 内容 | 归谁 | 阶段 0 的处置 |
| --- | --- | --- | --- |
| **(A)** 内核逐位 | ODE / 噪声内核与 `origin/main` 逐位一致 | `tests/test_fg1_physics_parity.py` | **一格不动**，仍是真正的重构保证 |
| **(B)** 装配按裁决 | 装配遵循仓内权威定义，其与 main 的**分歧被授权并记录** | `tests/test_fg14_assembly_provenance.py` | **新建**，分歧幅度被钉住 |

### 1.1 出处纪律：`source_branch` 的金标准一个字节都没动

`results/fg1_golden_main.json` 声明 `source_branch: "origin/main"`。
**把 FG 树的端口口径写进去就是伪造出处** —— 那正是 FG-13 拒绝做的事。

⇒ 本刀**没有**修改那个文件的任何一个数字，并由 sha256 钉死：

```
results/fg1_golden_main.json            4f4d9060ca273f337e71927ac0f116d9e2bf666ed9867102e81c230200a03794
src/…/models/numba_kernels.py           7beece4ea353b1bf5119ab9708d4db9845b092328357fff30f9b806afe613205
```

分歧只记录在**另一个**文件里：`results/fg14_assembly_divergence.json`。

---

## §2 门 0 读数（与 FG-13 时逐项相同）

全部在 `c621cda` 之后的本 worktree 上实测，`PYTHONPATH=$PWD/src`、
`rl` 解释器、`flock` 串行（`nlo` 环境缺 `np.trapezoid` / `gymnasium`）：

| 闸门 | FG-13 读数 | 本刀阶段 0 读数 | 一致 |
| --- | --- | --- | --- |
| 物理仓全量 | `333 passed / 1 skipped` | **`337 passed / 1 skipped`** | ✅ 见下方账 |
| `fg4a_capture_golden.py --check` | `31/31` 逐位 | `31/31 cases bit-identical` | ✅ |
| `fu4_capture_golden.py --check` | bit-exact | `fidelity golden reproduces bit-exactly` | ✅ |
| `test_fg1_physics_parity` | 全绿 | 全绿（(B) 已迁出） | ✅ |
| `numba_kernels.py` | 未动 | sha256 未变 | ✅ |

**全量计数为什么从 333 变成 337（如实记账，不是新增失败）**：

```
333（FG-13 基线）
+ 8   tests/test_fg14_assembly_provenance.py 新增守卫
− 4   test_pipeline_matches_origin_main 迁出（它按 4 个场景参数化）
─────
337 passed / 1 skipped          # 无任何失败项
```

brief 门 0 要求"与 FG-13 读数逐项相同"。**逐项相同的是物理读数**
（`31/31` 逐位、`bit-exactly`、内核逐位、`numba_kernels.py` 的 sha256）；
**全量计数必然随守卫的增删而变**，此处给的是完整账目，不是"用某个数字顶替"。

计时：全量 `41.12 s`（`flock` 串行），load average `7.24 → 7.95`。
**该数字仅用于"闸门开销可接受"，不含任何加速比含义**（本机同机另有一条线在跑）。

---

## §3 拆法的自证：变异测试（§7 纪律）

命题 (B) 的钉子不是"必须逐位等于 main"——阶段 1 之后授权幅度非零，
它检查的是**上界**：*分歧不许悄悄扩大*。

**变异测试**：把候选读数整体抬高 `1e-6` 再跑钉子。

```bash
PYTHONPATH=$PWD/src python scripts/fg14_divergence_probe.py --mutation 1e-6
# MUTATION_CAUGHT: 钉子变红，共 40 条：
#   - c_band_single.signal_power_w.max_abs_dev: 实测 9.999999999999972e-07
#     超过授权幅度 0.0（超出 1.000000e-06）——分歧悄悄扩大了
```

同样的变异也被 `tests/test_fg14_assembly_provenance.py::test_fg14_the_pin_is_not_a_vacuous_assertion`
钉住。⇒ **(B) 不是真空断言。**

### 3.1 G2 授权的夹逼证明

本刀要编辑 `tests/test_fg1_physics_parity.py`，而 FG-2 / FG-4a 的 G2 守卫
禁止改动既有测试的期望值。这是本刀遇到的第一道**真实冲突**，处置见 §4。
守卫的夹逼函数 `fg14_relocation_violations` 也做了变异验证：

```
# 变异：多删一条**不在**授权清单里的测试（命题 A 的内核守卫）
tests/test_fg1_physics_parity.py: tests removed beyond the FG-14 authorisation
  ('test_pipeline_matches_origin_main',): ['test_kernel_uses_ode_and_not_the_analytic_tilt_variant']
是否捕获: True
# 还原后：[]
```

⇒ 授权**只**放行命题 (B) 迁出；命题 (A) 少一条测试就会变红。

---

## §4 遇到的第一道冲突：G2 守卫 vs brief 的拆分要求

**如实记录，不掩盖。**

- brief §1 要求"拆 `tests/test_fg1_physics_parity.py` 的命题"；
- 而 `tests/test_fg2_flexible_grid.py::test_f2_8` 与
  `tests/test_fg4a_numeric_identity.py::test_f4a_pre_existing_tests_were_not_modified`
  机械禁止该文件的任何期望值改动（基线 `8f55e14` / `ea7ecb0`）。

首次实测**两条守卫同时变红**（`np.array_equal` 4x→3x 等）。三条路：

1. **放弃拆分** —— 不可能：阶段 1 的修复必然让那条断言失败，
   拆与不拆都要动这个文件。
2. **删掉或放宽守卫** —— brief §3 明令禁止（G2 纪律），且 §4.2 已证明
   守卫本身是被 token 保全反查过的，删它会被抓到。
3. **把授权记进设计它的那套机制** —— ✅ 采用。

`tests/fg4b_tier_pin_policy.py` 正是 FG-4b §1.2 为"有书面裁决的既有测试改动"
建立的授权台账（`AUTHORIZED_TIER_PIN_FILES` / `POLICY_GUARD_FILES`）。
本刀按同一模式新增 `FG14_AUTHORIZED_FILES` / `FG14_RELOCATED_TESTS`，
并加 `fg14_relocation_violations` 作夹逼。

**为什么这不是"为了过测试而改期望值"**：被移除的那条断言**没有被放宽**——
命题 (B) 在新文件里以更强的形式（上限 + 变异自证）重新落地，
而 `results/fg1_golden_main.json` 的出处**一个字节都没动**。

> 附注：该策略文件相对两个基线都是**新增**文件（`--name-status` 为 `A`），
> 因此 `unauthorized_test_edits` 本就不检查它；本刀**没有**借这个漏洞放宽任何守卫，
> 只增加了**新**的、更严的检查。

---

## §5 采纳档：完整并集（含 booster）

按编排方 §0 采用 **完整并集**，对齐 `physics/qot_checker.py:248-261` 的波形级装配
与 `rwa/qot_checker.py:118-128` 的解析级装配（三处里两处已按端口 + booster，
字段文档 `network/elements.py:76` 亦支持）。

**"仅端口化"那一档的 ΔSNR 按要求留在记录里**（`adopted_tier` 旁边的对照档）。

### 5.1 一个必须摆出来的实测结论：端口语义在既有场景上**几乎不产生 dB**

| 档位 | `fg1/roadm_hoop` | `fg4a/path_cplusl_roadm` |
| --- | --- | --- |
| 仅端口化 | `3.5527e-15 dB`（浮点残差） | `0.0`（**逐位**） |
| 完整并集 | **`0.978809 dB`** | **`5.186963 dB`** |

**机制**（两条，各自可查）：

1. `fg1/roadm_hoop` 的 ROADM 在**末跳** ⇒ 端口语义是 `drop`，
   `path_loss_db("drop") = 5.0 + add_drop_loss_db(1.0) = 6.0 dB`，
   比 `insertion_loss_db` 多 1 dB；但 `evaluate_path` 把损耗**同时除信号与全部累积噪声**，
   末跳损耗在 `power/(spm+xpm+ase)` 里**代数抵消** ⇒ 只剩 1e-15 浮点残差。
   （这正是 FG-13 §2.1 记录的结构性事实。）
2. `fg4a/path_cplusl_roadm` 的 ROADM 在**非末跳** ⇒ 端口语义是 `express`，
   而 `express_attenuation_db` **默认值是 `0.0`** ⇒
   `path_loss_db("express") = 5.0 + 0.0 = insertion_loss_db` ⇒ **逐位不变**。

⇒ **既有场景集上，被授权的那笔分歧几乎全部来自 booster/equalize 重排，
而不是端口语义。** 这不是说端口字段该被读（FG-13 的惰性字段普查依然成立：
`add_drop_loss_db` 在 `drop` 端口上是活的），而是说：
**若当初采纳"仅端口化"，两套金标准一个字节都不会动，缺陷却只修了一半**。
这条对照留在记录里，正是为了让采纳哪一档可复核。

---

## §6 阶段 0 交付物

| 产物 | 作用 |
| --- | --- |
| `scripts/fg14_divergence_probe.py` | 三档实测 + `--emit` / `--check` / `--emit-baseline` / `--mutation` |
| `results/fg14_assembly_divergence.json` | (B) 的分歧记录：幅度、机制、授权来源、上界钉子 |
| `results/fg14_pre_fix_baseline.json` | **冻结的修复前读数**（门 2 的 ΔSNR 表参照） |
| `tests/test_fg14_assembly_provenance.py` | (B) 的 8 项守卫，含变异自证与冻结基线守卫 |
| `tests/fg4b_tier_pin_policy.py` | 新增 FG-14 授权条目 + `fg14_relocation_violations` 夹逼 |
| `scripts/fg1_parity.py` | 装配分歧从 `failures` 移入 `assembly_divergences`（**数据**，不是失败项） |
| `docs/fg14_assembly_provenance.md` | 本文 |

---

## §7 复算命令

```bash
export PYTHONPATH="$PWD/src"
PY=/home/zsig/miniconda3/envs/rl/bin/python          # nlo 缺 np.trapezoid / gymnasium
$PY -c 'import network_nlo_eval as n; print(n.__file__)'   # 必须打印本 worktree 的路径

# 门 0
flock /tmp/fg14-physics.lock $PY -m pytest -q          # 337 passed / 1 skipped（账见 §2）
$PY scripts/fg4a_capture_golden.py --check             # 31/31 bit-identical
$PY scripts/fu4_capture_golden.py  --check             # bit-exactly

# 本阶段证据
$PY scripts/fg14_divergence_probe.py --check results/fg14_assembly_divergence.json
$PY scripts/fg14_divergence_probe.py --mutation 1e-6   # MUTATION_CAUGHT（钉子非真空）
$PY scripts/fg14_divergence_probe.py --emit-baseline results/fg14_pre_fix_baseline.json
```

---

## §8 未闭合项

1. **阶段 1 尚未落地**（本提交只有框架）。`evaluate_path` 的装配**仍然是**合并式，
   四个端口/booster 字段**仍然惰性**。这是刻意的：两阶段分离是为了让
   "授权框架"与"物理改动"落在两个可分别复核的提交上。
2. **`prospective_*` 三档在阶段 0 全部等于 `committed`**：探针此时量的是
   "若现在修复会动多少"，不是"已经修复"。阶段 1 合入后 `committed` 会变成
   满并集的读数，那两档随之变成历史对照。
3. **RL 侧的 0.2878 dB 用例**（FG-4 记为未复现）本刀**未触及**，需要 RL 侧补参数。
4. **`fg1_parity.py` 的装配检查用的是生产默认档 `matvec`**，因此报告里那 13 条
   `assembly_divergences` 全部 ≤`3.55e-15`——那是**加速档位残差**，不是装配分歧
   （FG-4b §1.2 已把这条断言钉在 `ode_accel="off"` 上）。本刀把它们从 `failures`
   移到 `assembly_divergences`，**没有**删掉，也没有改动内核比较那一半。
5. **未做任何计时声明**（见 §2）。