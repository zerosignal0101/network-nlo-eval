# FG-14 · 阶段 1：`evaluate_path` 的端口化修复

> **单向向权威定义靠拢。** `models/isrs_gn.py` 的 `evaluate_path` 不再合并式施加
> ROADM，改为消费 `rwa.qot_checker.fold_roadm_terms`（即
> `ROADMConfig.path_loss_db()` + booster + 独立滤波乘子）。
> **没有反向替换任何物理**，**没有触碰 ODE 内核**。
>
> 阶段 0（框架，零物理变化）见 [`fg14_assembly_provenance.md`](fg14_assembly_provenance.md)。
> 本文所有数字可复算，命令见 §7。

---

## §1 改了什么（只有 ROADM 那一段）

```python
# 改前：合并式，纯线性，同时除信号与全部已累积噪声
loss = 10 ** ((roadm.insertion_loss_db + roadm.filtering_penalty_db) / 10)
power /= loss; total_spm /= loss; total_xpm /= loss; total_ase /= loss

# 改后：权威口径
terms = fold_roadm_terms(roadm, is_last_hop=hop_index == n_hops - 1)
if terms.equalize:
    total_ase += terms.booster_base * f_abs_hz * symbol_rate_hz_channels * (terms.loss - 1.0)
else:
    power /= terms.loss; total_spm /= terms.loss; total_xpm /= terms.loss; total_ase /= terms.loss
total_spm *= terms.filtering_penalty; total_xpm *= terms.filtering_penalty; total_ase *= terms.filtering_penalty
```

跨段循环、`_calc_span_noise_and_power` 调用、阻断处理、SNR 公式**一字未改**。

> **实现细节**：`fold_roadm_terms` 是**惰性导入**的 —— `rwa.qot_checker` 从
> `models.isrs_gn` 取 `MultiBandISRSGN`，模块级互相导入会成环。
> 惰性导入保住了 FG-11 的"折叠逻辑只有一处"，没有为了避开 import 就复制一份。

---

## §2 门 1：全量不新增失败

| 项 | 读数 |
| --- | --- |
| 基线（FG-13 交付） | `333 passed / 1 skipped` |
| 阶段 0（框架落地） | `337 passed / 1 skipped` |
| **阶段 1（本次）** | **`338 passed / 1 skipped`**，`EXIT=0` |

阶段 0 → 阶段 1 的 `+1` 是新增的
`test_fg13_fu4_golden_is_structurally_immune_to_the_repin`。**无任何失败项。**

`flock` 串行，`41.23 s`，load average `6.93 → 7.38`（同机另有一条线在跑）。
该计时**不含任何加速比含义**。

| 既有金标准 | 读数 |
| --- | --- |
| `fg4a_capture_golden.py --check` | `31/31 cases bit-identical`，worst rel dev `0.000e+00` |
| `fu4_capture_golden.py --check` | `fidelity golden reproduces bit-exactly` |
| `fg14_divergence_probe.py --mutation 1e-6` | `MUTATION_CAUGHT`（钉子仍非真空） |

---

## §3 碰撞面逐条归因（FG-13 预测 18 → 实测 17）

FG-13 §4 预测"17 项属 FG-4a golden（可正当重钉）、1 项是 FG-1 的 (B)"。
实测**与预测一致**：第 1 类 17 项全部落地；第 2 类 1 项在阶段 0 已由 (B) 接手。

### 3.1 17 项失败：机制与处置

| # | 失败项 | 数量 | 机制 | 处置 |
| --- | --- | --- | --- | --- |
| 1 | `test_f4a_3_bit_identical_to_golden[{off,cached_g}::path_cplusl_roadm::*]` | 10 | 该场景的 5 个输出量随新装配移动（见 §3.2） | 重钉 `fg4a_golden.json` |
| 2 | `test_f4a_4_matvec_within_budget[matvec::path_cplusl_roadm::*]` | 5 | `expected` 取自 `_golden_array`（`test_fg4a_numeric_identity.py:281`），**经由 golden 间接比较档位** | 随 #1 一并重钉 |
| 3 | `test_f4a_4_measured_deviation_is_bounded_and_reported` | 1 | 汇总 #1/#2 的实测偏差 | 随 #1 |
| 4 | `test_f4a_capture_script_check_mode_passes` | 1 | `--check` 子进程对旧 golden 失败 | 随 #1 |

### 3.2 金标准重钉的逐项证据（不是只改数字）

改前 vs 改后，**26 项逐位一致、5 项移动**，移动的**全部**是 `path_cplusl_roadm`：

| 场景 | 输出 | 改前 vs 改后 |
| --- | --- | --- |
| `cl_full_100km` / `cl_partial_100km` / `cl_full_64qam_80km` / `cl_short_1km` / `cl_nspan10` / `flexgrid_mixed` | 全部 4 个输出 | **逐位一致（24 项）** |
| `path_cplusl_roadm` | `roadm_count` / `span_count` | **逐位一致** |
| `path_cplusl_roadm` | `ase_noise_w` | `max_abs=1.03932e-08`，rel `9.263e-03` |
| `path_cplusl_roadm` | `signal_power_w` | `max_abs=4.15965e-08`，rel `2.315e+00` |
| `path_cplusl_roadm` | `spm_noise_w` | `max_abs=1.60688e-10`，rel `2.471e+00` |
| `path_cplusl_roadm` | `xpm_noise_w` | `max_abs=6.09554e-10`，rel `2.471e+00` |
| `path_cplusl_roadm` | `snr_db` | **`5.18696` dB**，rel `2.891e-01` |

**机制**：6 个单跨段场景不经 `evaluate_path`，所以逐位不变 —— 这同时证明
**ODE 内核与跨段物理未被触碰**。唯一移动的是那个带 ROADM 的路径场景。

> `signal/spm/xpm` 的 `rel` 偏大（>1）是因为 `equalize=True` 时信号不再被 `loss`
> 除、噪声按 booster 重算，部分信道的基线值极小，相对量因此被放大；
> **绝对量在 `1e-8` W 量级**，SNR 的 `rel=0.289` 才是有意义的读数。

### 3.3 重钉的出处：不是静默改写参考

`fg4a_capture_golden.py` 新增 `--repinned-by`，把重钉写进金标准的出处链
（`provenance.repins[]`），并保留**最初**的采集修订：

```json
"repins": [{"git_rev": "ea7ecb0…", "captured_at_stage": "pre-FG-4a baseline",
            "adjudicator": "FG-4a baseline capture"}]
```

`test_f4a_1_golden_records_its_provenance` 相应改为断言**链根**仍是 FG-4a 基线，
且每一跳都带授权人 —— 只断言"最新 `git_rev`"会让日后任何一次重采集悄悄盖掉出处。

### 3.4 不变量复核：四条不许重钉，一条没碰

FG-13 §4.2 的疑虑（`matvec_within_budget` 名字里带 "invariant"）经复核**不成立**：
#2 的 `expected` 取自 golden 而非同树两档。修复后的同树对比：

```
fg4a_capture_golden --check: 31/31 cases bit-identical, worst rel dev 0.000e+00
```

`off` / `cached_g` / `matvec` 三档对 `path_cplusl_roadm` 逐位一致 ⇒
**FG-4a 的 `off ≡ matvec` 档位不变量完好**，`numba_kernels.py` 的 sha256
`7beece4e…` 自阶段 0 起未变。§25 四条不变量**一条未重钉**。

---

## §4 门 2：逐场景 ΔSNR（含"仅端口化"对照档）

对照基准是**阶段 0 冻结的修复前读数** `results/fg14_pre_fix_baseline.json`
（不是记忆、不是 git diff）。

| 场景 | 仅端口化（对照） | **完整并集（采纳）** |
| --- | --- | --- |
| `fg1/c_band_single` | `0.0`（逐位） | `0.0`（逐位） |
| `fg1/multiband_wide` | `0.0`（逐位） | `0.0`（逐位） |
| `fg1/kurtosis_64qam` | `0.0`（逐位） | `0.0`（逐位） |
| `fg1/roadm_hoop` | `3.5527e-15 dB` | **`0.978809 dB`**（rel `5.539e-02`） |
| `fg4a/path_cplusl_roadm` | **`0.0`（逐位）** | **`5.186963 dB`**（rel `2.891e-01`） |

**"仅端口化"这一档必须留着**（编排方 §0）。它说明的事比数字本身更重要：

- `roadm_hoop` 的 ROADM 在**末跳**，端口语义是 `drop`：
  `path_loss_db("drop") = 5.0 + 1.0 = 6.0 dB`，比 `insertion_loss_db` 多 1 dB，
  但修复前的代数抵消让 SNR 只差 `3.55e-15`；
- `path_cplusl_roadm` 的 ROADM 在**非末跳**，端口语义是 `express`，
  而 `express_attenuation_db` **默认 `0.0`** ⇒ 与 `insertion_loss_db` **完全相等**。

⇒ **在既有场景集上，端口语义贡献几乎为 0；被授权的那笔分歧几乎全部来自
booster/equalize 重排。** 若当初采纳"仅端口化"，两套金标准一个字节都不会动，
而 `equalize_output_power` / `booster_noise_figure_db` 仍是惰性字段 ——
**缺陷只修一半，且看不出来**。

### 4.1 受影响的调用方

| 调用方 | 拿输出做什么 | 是否受影响 |
| --- | --- | --- |
| `physics/comparison.py:80` | 统计模型 SNR；`run_waveform=True` 时与波形引擎 SNR 相减得 `snr_delta_db`（`:93`） | **是**，且这正是统计/波形**跨模型一致性**的度量 |
| `tests/fg4a_scenarios.py` | FG-4a 金标准采集入口 | 是（已重钉，§3.2） |
| `physics/qot_checker.py:327/343` | ⚠️ **不是**本路径的调用方，是波形引擎同名方法（FG-13 §4.1 更正） | 否 |

已复核：`grep -rn "evaluate_path" src/` 确认统计侧在 `src/` 内的**唯一**调用方
仍是 `physics/comparison.py:80`。

---

## §5 FG-13 守卫的翻转：被授权的分歧，不是掩盖

FG-13 的守卫钉的是"当前状态"。修复合入后它们**必然变红** —— 这是**预期的**。
本刀把它们**翻转成新状态的守卫**：

| 守卫 | 翻转后钉的是 |
| --- | --- |
| `test_fg13_port_and_booster_fields_are_inert_today` | 四个字段**已转为活字段**，且惰性清单为空 |
| `test_fg13_evaluate_path_does_not_call_the_union_folding` → `..._calls_the_union_folding` | `evaluate_path` **正在**消费 `fold_roadm_terms`，且合并式写法**不**回来 |
| `test_fg13_unpatched_tree_still_reproduces_both_goldens` → `..._goldens_are_reproducible_after_the_repin` | 重钉后两套金标准仍逐位重现 |
| `test_fg13_impact_is_physics_scale_not_float_noise` | FG-13 量化的 `0.978809 dB` 与 FG-14 独立实测**必须一致**（跨产物交叉校验） |

**这不是掩盖**，因为：

1. 每条翻转都指向**新状态应当是什么**，而不是"让它变绿"；
2. FG-13 的原始数字（惰性清单、影响量级）**没有被删掉**，
   而是被 FG-14 的冻结基线与分歧记录**继续担保**，且两者必须逐位相等；
3. 授权来源写在 §1 与 `results/fg14_assembly_divergence.json` 的
   `adjudicated_by` 里。

### 5.1 一个必须记录的结构性反转

FG-13 §2.1 记录过：**末跳** ROADM 的损耗在 SNR 比值里代数抵消（残差 `3.55e-15`）。
**修复后这条恒等被打破了**：

| | 修复前 | 修复后 |
| --- | --- | --- |
| 仅末跳 ROADM | `3.55e-15 dB`（抵消） | **`0.857183 dB`（可观测）** |
| 仅非末跳 ROADM | `2.550735 dB` | `0.096296 dB` |

**机制**：`equalize_output_power` 默认 `True`，booster 以
`NF·h·ν·Rs·(loss−1)` 在损耗**之后注入** ASE，这部分噪声不再被同一损耗约掉。

FG-13 §2.1 当初警告"修复时不能因为末跳损耗看起来无效就跳过端口语义"——
**这个警告是对的**，而且修复后末跳损耗**确实重新变得重要**。
守卫已相应改为钉住"末跳也必须可观测"。

---

## §6 字段存活普查（改前 vs 改后）

| 字段 | 修复前 max\|ΔSNR\| | 修复后 max\|ΔSNR\| |
| --- | --- | --- |
| `insertion_loss_db` | `2.550735` | `0.900832` |
| `filtering_penalty_db` | `1.499416` | `2.701649` |
| `express_attenuation_db` | **`0.0`（死）** | **`0.047846`** |
| `add_drop_loss_db` | **`0.0`（死）** | **`0.509148`** |
| `booster_noise_figure_db` | **`0.0`（死）** | **`0.686972`** |
| `equalize_output_power` | **`0.0`（死）** | **`2.524285`** |

**惰性字段清单：修复前 4 个 → 修复后 0 个。**

`insertion_loss_db` / `filtering_penalty_db` 的数值也变了，因为它们现在进入的是
"损耗 + booster + 独立滤波乘子"这套口径，而不是被合并进一个线性除数。

---

## §7 复算命令

```bash
export PYTHONPATH="$PWD/src"
PY=/home/zsig/miniconda3/envs/rl/bin/python      # nlo 缺 np.trapezoid / gymnasium
$PY -c 'import network_nlo_eval as n; print(n.__file__)'   # 必须打印本 worktree

# 门 1
flock /tmp/fg14-physics.lock $PY -m pytest -q      # 338 passed / 1 skipped
$PY scripts/fg4a_capture_golden.py --check         # 31/31 bit-identical
$PY scripts/fu4_capture_golden.py  --check         # bit-exactly

# 门 2：ΔSNR 表（含"仅端口化"对照档）
$PY scripts/fg14_divergence_probe.py --check results/fg14_assembly_divergence.json

# 重钉（仅在有意重钉时执行，且必须带授权人）
$PY scripts/fg4a_capture_golden.py --repinned-by "<谁授权、为什么>"

# 字段存活普查（改后）
$PY scripts/fg13_probe_roadm.py --check results/fg13_roadm_probe.json
```

---

## §8 未闭合项（不猜，列清楚缺什么、谁能给）

1. **`physics/comparison.py` 的 `snr_delta_db` 未重跑。** 本刀只保证仓内全量绿；
   统计/波形跨模型一致性度量的**数值**变化需要 RL 侧跑批才能给，
   超出本刀授权（brief §3：不改 RL 仓、不新增依赖）。
   —— 需要编排方决定是否开下一刀。
2. **RL 的 0.2878 dB 用例**（FG-4 记为未复现）本刀**未触及**，仍需 RL 侧补参数。
3. **`physics/qot_checker.py` 的波形级装配仍是它自己那套写法**
   （`current["ROADM"](…)` + `current["EDFA"](…)`），本刀**没有**统一它。
   FG-14 只改了统计侧的 `evaluate_path`；三处现在**语义等价**，
   但**代码仍有两份**。要不要收敛成单一实现需要另开一刀并重新评估碰撞面。
4. **FU-3（把 `legacy` 默认翻成 `tabular`）按 brief 未做**，框架已就绪，下一刀直接用。
5. **未做任何计时声明**（见 §2）。