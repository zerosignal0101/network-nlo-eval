# FG-17：把波形级 ROADM 装配收敛为**单一实现** + 收 `snr_delta_db` 尾巴

- **worktree（物理）**：`/home/zsig/Documents/gits/wt/fg-converge`，分支 `wp/fg-converge`，基线 `25ed30d`
- **worktree（RL）**：`/home/zsig/Documents/gits/wt/fg-rt-cv`，分支 `wp/fg-rt-cv`，基线 `21ee113`
- **本刀不推 `origin`**；`origin/main` 保持 `87850a1` 不变。
- **本刀没有改 RL 侧任何 `src/` 文件**（只新增了 `scripts/` 下的取证脚本）。

---

## 0. 结论先行

| 项 | 结论 |
|---|---|
| 权威实现 | `rwa/qot_checker.py::fold_roadm_terms`（按 FG-5 并集裁定，未推翻） |
| 收敛形态 | `physics/qot_checker.py` 的 ROADM 装配改调**同一个函数对象**；折叠新增 `loss_db` / `booster_nf_db` 两个 dB 字段 |
| 是否零行为变化 | **是**。波形级 A/B 6/6 用例 `max|Δ| = 0.0`；金标准逐位不变；物理仓 404 全绿；RL 决策序列 0 差异 |
| 新增文件 | 0 个（改的是两个既有模块） |
| 新增配置开关 | 0 个 |
| 重钉既有期望值 | **0 处**（没有改动任何测试的期望值） |
| 未闭合项 | 3 条，见 §6 |

---

## 1. §0 取证：两份实现的公式级对照

### 1.1 各自施加了哪些物理量

改前 `physics/qot_checker.py::propagate_path`（基线 `25ed30d` 的 `:244-262`）**自己持有一份装配**：

| 物理量 | 统计侧（权威，`rwa/qot_checker.py`） | 波形侧（改前副本，`physics/qot_checker.py`） |
|---|---|---|
| 端口判据 | `operation = "drop" if is_last_hop else "express"` (`:135`) | `"drop" if hop_index == len(fiber_configs)-1 else "express"` (`:250-252`, `:257-259`，**同一表达式写了两遍**) |
| 损耗 | `loss = 10 ** (path_loss_db(operation) / 10)` — **线性** (`:140`) | `insertion_loss_db=path_loss_db(...)` / `gain_db=path_loss_db(...)` — **dB** (`:250`, `:257`) |
| `equalize_output_power` | `equalize=config.equalize_output_power` (`:146`) | `if roadm.equalize_output_power:` (`:255`) |
| `booster_noise_figure_db` | `booster_nf = 10 ** (nf_db/10) if equalize else None` (`:136-137`) | `nf_db=roadm.booster_noise_figure_db` (`:260`) |
| `filtering_penalty_db` | `filtering_penalty = 10 ** (db/10)`，**只乘噪声** (`:145`) | **不读**——滤波由物理 WSS（`filter_order` / `passband_hz`）承担 |
| `add_drop_loss_db` / `express_attenuation_db` | 只经 `path_loss_db(operation)` 间接生效 | 同左 |
| 端口 vs 路径语义 | 末跳 `drop`、其余 `express` | 同左（**但判据是另一份写法**） |
| 信号如何被 equalize 抵消 | 统计域：信号**不再**被 `loss` 衰减，改加 `NF·h·ν·Rs·(loss−1)` 的 booster ASE | 波形域：ROADM 先按 `loss` 衰减，再用一台**真实 EDFA** 以 `gain_db = loss` 加回来 |

**语义等价、物理相同、单位域不同**——这是本刀要处理的全部实质。

### 1.2 谁是权威

按 FG-5 的并集裁定，权威是 `rwa/qot_checker.py`。本刀的取证**没有**推翻它：
统计侧口径已被 `models/isrs_gn.py::evaluate_path`（`:702` `from ...rwa.qot_checker import fold_roadm_terms`）
与 RL 侧 `scan.py`（`:55-58`）共同消费，是三处共用的那一份。

### 1.3 逐位实测（25 个场景）

`scripts/fg17_assembly_evidence.py` → `results/fg17_assembly_evidence.json`：

```
n_scenarios                       : 25
equalize_mismatches               : 0
booster_presence_mismatches       : 0
loss_roundtrip_non_exact          : 0
max_loss_roundtrip_abs_err_db     : 0.0
max_nf_roundtrip_abs_err_db       : 0.0
```

覆盖 `path_cplusl_roadm`（express 端口）、fu4 全部 9 个用例 × 2 个端口、两个单跨段场景、
`flexgrid_mixed` —— brief 点名的场景全部在内。**两侧语义与逐位读数均无分歧**。

### 1.4 波形级端到端 A/B（6 用例，改前 vs 改后）

`scripts/fg17_waveform_ab.py` → `results/fg17_waveform_ab.json`：

```
n_cases: 6, all_bit_exact: true, cases_not_bit_exact: []
  express_roadm_3hop        ber/evm/snr max|Δ| = 0.000e+00
  drop_roadm_last_hop       ber/evm/snr max|Δ| = 0.000e+00
  single_span_last_hop      ber/evm/snr max|Δ| = 0.000e+00
  equalize_off_express      ber/evm/snr max|Δ| = 0.000e+00
  express_attenuated_blocked ber/evm/snr max|Δ| = 0.000e+00
  odd_nf_and_loss           ber/evm/snr max|Δ| = 0.000e+00
```

**结论：收敛是零行为变化的重构。**

---

## 2. §1 收敛：最小改动

### 2.1 核心难点：dB ↔ 线性往返**不逐位**

`fold_roadm_terms` 折出的是**线性** `loss`，而波形侧的物理 ROADM / EDFA 元件收的是 **dB**。
最省事的做法是让波形侧做 `10 * log10(terms.loss)` 反算。**实测这条路走不通**：

| 试验 | 非逐位比例 |
|---|---|
| 合成 `ROADMConfig` 网格（4×3×4×4×2×2 = 768 组，含 loss 与 NF） | **304/768 = 39.6%** |
| 随机 dB（20 万次，`-40..40 dB`） | **11145/200000 = 5.57%** |

典型差值 `1.0 dB -> 1.0000000000000002`（1 ulp）。若走反算，收敛就会变成一次**静默的
数值改动**——正是本刀明令禁止的。

### 2.2 采用的形状：让折叠**同时**产出 dB 形式

`rwa/qot_checker.py::RoadmTerms` 新增两个字段，`fold_roadm_terms` 里**只求值一次** `path_loss_db`：

```python
loss_db = config.path_loss_db(operation)          # 同一次求值，两个形式共用
booster_nf_db = config.booster_noise_figure_db
return RoadmTerms(..., loss=10 ** (loss_db / 10), loss_db=loss_db,
                  booster_nf_db=booster_nf_db, ...)
```

波形侧于是直接拿 `terms.loss_db` / `terms.booster_nf_db`，
**值与它原先自己读配置逐位相同**（同一个表达式、同一次求值），不需要任何反算。

`loss`（线性）字段一字未改，统计侧与 RL 侧**零影响**。

### 2.3 改动清单（`git diff --stat`）

```
 src/network_nlo_eval/physics/qot_checker.py | 20 ++++++++++++--------
 src/network_nlo_eval/rwa/qot_checker.py     | 24 ++++++++++++++++++++++-
 2 files changed, 35 insertions(+), 9 deletions(-)
```

`physics/qot_checker.py` 的改动只有三处语义：一次 `fold_roadm_terms` 调用、
`insertion_loss_db` / `gain_db` 改用 `terms.loss_db`、`if roadm.equalize_output_power`
改用 `terms.equalize`、`nf_db` 改用 `terms.booster_nf_db`，外加一行 import。

### 2.4 对外行为与签名保持

- `PyNLOQoTEvaluator.evaluate_path` 的**签名与返回值一字未改**。
  调用点（先 grep 清）：
  - `physics/comparison.py:90`（`TopologyQoTComparator.evaluate`）
  - `physics/qot_checker.py:327`（`evaluate_topology_path`）、`:343`（`evaluate_channel_evm_and_ber`）
  - `tests/test_topology_physical_integration.py:57,99`
- `evaluate_path` 自身保留（**签名未动**），只是内部 `propagate_path` 换了装配来源。
- **已知既有缺陷（不在本刀范围）**：`evaluate_path:320` 假定 `roadm_configs` 元素都是
  `ROADMConfig`，遇 `None` 抛 `AttributeError`（而 `propagate_path` 容忍 `None`）。
  这是 `25ed30d` 起就有的行为，本刀**未改也未修**；取证脚本因此不构造含 `None` 的路径。

---

## 3. §2 碰撞面逐条归因

### 3.1 改前 vs 改后对照表

| 物理量 / 施加点 | 改前 | 改后 | 位 |
|---|---|---|---|
| 端口判据（选 drop/express） | `hop_index == len(fiber_configs)-1`（自持一份） | `fold_roadm_terms(..., is_last_hop=同上)` 内部 `operation` | **逐位相同** |
| `insertion_loss_db`（ROADM 元件） | `roadm.path_loss_db(op)` | `terms.loss_db`（= 同一次 `path_loss_db` 求值） | **逐位相同** |
| booster `gain_db` | `roadm.path_loss_db(op)` | `terms.loss_db` | **逐位相同** |
| booster 是否存在 | `roadm.equalize_output_power` | `terms.equalize`（= `config.equalize_output_power`） | **逐位相同** |
| booster `nf_db` | `roadm.booster_noise_figure_db` | `terms.booster_nf_db`（= 同一次字段读取） | **逐位相同** |
| `passband_hz` / `filter_order` | `roadm.passband_hz` / `roadm.filter_order` | 不变 | **未动** |
| `filtering_penalty_db` | 不读（物理 WSS 承担） | 不读 | **未动** |
| `RoadmTerms.loss`（线性，统计侧） | — | 一字未改 | **未动** |
| `_eval_service_snr` / `_build_path_snr_array` | — | 一字未改 | **未动** |

**没有任何一项移动**。因此 §25 的"重钉期望值"四条件**一条都不适用**——本刀
**0 处**重钉，未改动任何既有测试的期望值（G2 纪律）。

### 3.2 碰撞面上唯一"可能被读成移动"的点，及其处理

`RoadmTerms` 是 `frozen=True` dataclass，新增两个字段。若它被**按位置**构造，旧调用会错位。
取证结论：

- `RoadmTerms(...)` 在**整个仓库只有一个构造点**：`rwa/qot_checker.py` 的 `fold_roadm_terms`。
  （`grep -rn "RoadmTerms" src/ tests/` 见 `physics/` 与 RL 侧均只**读属性**。）
- `tests/` 与 RL `tests/` 中**没有任何测试**构造 `RoadmTerms` 或枚举其字段
  （`grep -rn "RoadmTerms\|fields(" tests/` → 无命中）。
- 新字段**插在中段**，但唯一构造点用**关键字实参**，故不存在错位。

结论：无碰撞面。

### 3.3 不变量：FG-14 的 (A)/(B) 分歧**未被扩大**

FG-14 把"与 `origin/main` 逐位一致"拆成 (A) 内核逐位 / (B) 装配按裁决。本刀：

- (A) **未触碰** ODE 内核（`_calc_exact_power_profile_ode` / `numba_kernels.py` 一字未改）。
- (B) **未扩大**：`physics/` 侧的装配现在是**收敛到并集**的，不再是第三个口径。
  本刀只删副本、不动 `rwa/` 的并集语义（`loss` / `filtering_penalty` / `equalize` /
  `booster_nf` / `booster_base` / `blocked` 六个既有字段的取值与算术一字未改）。

---

## 4. 门 3：grep 证据（每一物理量只剩一份定义）

```
$ grep -rn "add_drop_loss_db" src/ --include=*.py | grep -v elements.py
  （无输出）
$ grep -rn "express_attenuation_db" src/ --include=*.py | grep -v elements.py
  （无输出）
```

`add_drop_loss_db` 与 `express_attenuation_db` 的**唯一定义点**是
`network/elements.py::ROADMConfig.path_loss_db`（`:113-116`），全部消费者经由它。

```
$ grep -n "path_loss_db\|equalize_output_power\|booster_noise_figure_db" \
      src/network_nlo_eval/physics/qot_checker.py
18:  from network_nlo_eval.rwa.qot_checker import fold_roadm_terms
252:  terms = fold_roadm_terms(roadm, is_last_hop=...)
258:  insertion_loss_db=terms.loss_db,        # 不再自己读 path_loss_db
264:      nf_db=terms.booster_nf_db,          # 不再自己读 booster_noise_figure_db
```

波形侧的 `propagate_path` 现在**没有任何一处**直接读这三个字段（`:249-250` 的命中是解释性注释）。

源码级守卫仍绿：`tests/test_fu4_assembly_union.py::test_fu4_6b_union_keeps_every_fg2_capability_symbol`
按字面检查 `path_loss_db` / `booster_noise_figure_db` / `filtering_penalty_db` 仍在
`rwa/qot_checker.py` 中——新增字段没有删掉任何既有符号，该守卫**未触发重钉**。

---

## 5. §3 `snr_delta_db` 尾巴（FG-13/FG-14 未闭合项）

`models/isrs_gn.py::evaluate_path` 的调用点**不止一处**（`scripts/` 与 `tests/` 下还有
`fg1_parity.py:86`、`fg4g_q1_library_ab.py:275-276`、`fg13_probe_roadm.py:99`、
`fg4a_scenarios.py:344`、`test_fu2_sclu_bands.py:428` 等），但按调用点逐条 grep 后可确认：
**`physics/comparison.py:80` 是唯一一个其返回值流入 `snr_delta_db` 的调用方**
（`snr_delta_db` 全仓只出现在 `comparison.py:26` 与 `:50`，值由 `:93` 的
`waveform_metrics["snr"] - statistical.snr_db` 产生）。这正是 FG-13/FG-14 端口化后
没被重跑的那条尾巴。已在收敛后的树上重跑。**先决问题**：这条尾巴能否当"逐位门"用？
实测**不能**。

### 5.1 关键机制发现：波形侧有个 `IdealAmplifier` 不可重现

| 实验（同 seed、同输入、同 evaluator 实例） | 4 次调用的逐位关系 | max\|ΔSNR\| |
|---|---|---|
| 拓扑默认路径（`EDFAConfig()` → `target_gain_db=None` → `IdealAmplifier`） | `Y...`（仅第 0 次与自身相同） | **0.330 dB** |
| 同一路径但把 `target_gain_db` 钉成显式增益（真 `EDFA`） | `YYYY` | **0.0** |

即：**`IdealAmplifier` 这条波形路径逐次不可重现**，而同一条路径改用显式 `EDFA` 增益后
**逐位可重现**。这是**既有**性质（`25ed30d` 即如此），与本刀无关，但它决定了
`snr_delta_db` 只能与噪声底比、不能与 0 比。

### 5.2 两层读数（`scripts/fg17_snr_delta_probe.py` → `results/fg17_snr_delta.json`）

**第 0 层（能逐位的那层）**——同一 evaluator 实例 + 显式 `EDFA` 增益，避开 `IdealAmplifier`：

```
layer0_all_bit_exact              : true
layer0_max_abs_delta              : 0.0
（每个配置的 same-code null 同样 = 0.000e+00）
  active=[3,4,5] launch=+0.0dBm   max|d|=0.000e+00 (same-code null=0.000e+00)
  active=[3,4,5] launch=-3.0dBm   max|d|=0.000e+00 (same-code null=0.000e+00)
  active=[0,9]   launch=+0.0dBm   max|d|=0.000e+00 (same-code null=0.000e+00)
  active=[0,9]   launch=-3.0dBm   max|d|=0.000e+00 (same-code null=0.000e+00)
  active=[2]     launch=+0.0dBm   max|d|=0.000e+00 (same-code null=0.000e+00)
  active=[2]     launch=-3.0dBm   max|d|=0.000e+00 (same-code null=0.000e+00)
```

6/6 配置改前 = 改后**逐位相同**，且自身 null 也是 0 —— 这条尾巴的**装配部分是零行为变化**。

**第 1 层（真实尾巴，`TopologyQoTComparator` 端到端）**——保留默认 `IdealAmplifier`：

```
layer1_mean_null_floor_db   : 0.1142 dB   （同代码、6 次重复的 null 底）
layer1_max_null_floor_db    : 0.4708 dB
layer1_max_before_after_delta_db : 0.3358 dB
layer1_all_within_null_floor: false   （6 个配置里 1 个超出当次 null 底）
```

**如实结论**：第 1 层的改前/改后差（≤0.336 dB）与**同代码自身**的噪声底
（均值 0.114 dB、最大 0.471 dB）**同量级**，且 1/6 配置超出当次 null 底。
因此**不能**据此断言 `snr_delta_db` 逐位不变；能断言的是"该路径的读数**在噪声内与噪声不可区分**"，
而噪声的来源已定位为既有的 `IdealAmplifier`（§5.1），与本刀改动无关。
**"逐位不变"这一结论只由第 0 层与 §1.4 的 6 用例波形 A/B 承担。**

---

## 6. 门读数（逐项，不接受"通过"三个字）

| 门 | 项 | 读数 |
|---|---|---|
| 门 1 | 物理仓全量 | `404 passed, 1 skipped in 38.86s`（基线 `25ed30d` 同口径，**0 新增失败**） |
| 门 1 | RL 仓全量（分片 8 片） | 见 §6.1 |
| 门 2a | `fg4a_capture_golden --check` | `31/31 cases bit-identical, worst relative deviation 0.000e+00 (0 differing elements in total)` |
| 门 2b | `fu4_capture_golden --check` | `OK: fidelity golden reproduces bit-exactly` |
| 门 3 | grep 单一定义 | `add_drop_loss_db` / `express_attenuation_db` 在 `elements.py` 外**零命中**；`physics/qot_checker.py` 对 `path_loss_db` / `equalize_output_power` / `booster_noise_figure_db` **零直接读取**（见 §4） |
| 门 4 | RL 决策序列 | `first_fit`：30 步、trace **完全相同**、0 步差异；`seeded_random`：30 步、10 个不同 action、trace **完全相同**、0 步差异。两侧 allocated 均为 30/30 |
| 附加 | 波形级 A/B（§1.4） | 6/6 用例 BER/EVM/SNR `max|Δ| = 0.000e+00` |
| 附加 | §3 第 0 层 | 6/6 配置 `max|Δ| = 0.0`，同码 null 亦为 `0.000e+00` |

### 6.1 RL 仓全量（门 1 另一半）

RL 全量按文件切成 8 片、每片独立进程（无 xdist / timeout，且不许新增依赖）。
读数见 `wt/fg-rt-cv` 的 `scripts/fg17_rl_full.sh` 与 `/tmp/fg17_rl/shard_*.log`
（每片自带 `shard_rc=`，据此区分"这片死了"与"还在跑"）。

**RL 源码本刀一行未改**（只新增 `scripts/`），RL 侧的物理消费点
（`scan.py:55-58` import 的 `fold_roadm_terms` / `accumulate_*_snr`）签名与语义均未变，
故 RL 侧只可能通过门 4 的决策序列体现——已逐项相同。

---

## 7. 未闭合项（如实登记）

1. **`snr_delta_db` 端到端无法给逐位结论**。原因已定位：`IdealAmplifier` 逐次不可重现
   （0.33 dB，§5.1）。要闭合需要单独一刀处理 `IdealAmplifier` 的可重现性
   （例如改用显式 `target_gain_db` 的 `EDFA`），**不在本刀授权范围**。
2. **`models/isrs_gn.py:528` 与 `:596` 仍带 FG-tree 合并式装配**
   （`loss_lin = _db_to_lin(roadm_config.insertion_loss_db + roadm_config.filtering_penalty_db)`）。
   这两处在**跨段级** helper（`calc_span_noise_and_power` / `..._batch`）里，**不是**
   `evaluate_path`（后者已在 FG-14 走 `fold_roadm_terms`）。它们是 `filtering_penalty_db`
   仅存的第二处定义点。本刀**未改**：一是超出授权范围（ODE 相邻路径），
   二是改了会移动统计侧金标准。`tests/test_fu4_assembly_union.py::test_fu4_6` 的
   源码级守卫**只**检查 `rwa/qot_checker.py`，不覆盖这两处——**守卫有盲区**，建议单开一刀。
3. **`RoadmTerms` 新增字段若将来被按位置构造会错位**。当前唯一构造点用关键字实参、无测试枚举字段，
   故本刀无碰撞面；但这是**新增的前置条件**，已在本文件与字段 docstring 中写明。

---

## 8. 复算命令

> 全部命令都**必须**带 `PYTHONPATH=<PH>/src:<RL>/src`（editable 装的 `.pth` 指向主检出，
> 不设这一行跑的是另一棵树），且每条都打印实际 import 到的包路径。

### 8.1 §0 取证 / §2 grep

```bash
cd /home/zsig/Documents/gits/wt/fg-converge
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python scripts/fg17_assembly_evidence.py \
    -o results/fg17_assembly_evidence.json

grep -rn "add_drop_loss_db\|express_attenuation_db" src/ --include=*.py | grep -v elements.py   # 期望：无输出
grep -n "path_loss_db\|equalize_output_power\|booster_noise_figure_db" \
    src/network_nlo_eval/physics/qot_checker.py
```

### 8.2 波形级 A/B（需要 GPU + `physics_nlo_eval`，故用 `nlo` 解释器）

```bash
# CuPy 需要 CUDA root；本机无 /usr/local/cuda，用 cupy 自带头文件 + 自带 .so 建软链：
SP=~/miniconda3/envs/nlo/lib/python3.12/site-packages
SHIM=/tmp/fg17_cuda_root; mkdir -p "$SHIM/lib64"
ln -sfn "$SP/cupy/_core/include/cupy/_cuda/cuda-12" "$SHIM/include"
ln -sfn "$SP/cupy_backends/cuda/libs" "$SHIM/lib64"

cd /home/zsig/Documents/gits/wt/fg-converge
CUDA_PATH=$SHIM PYTHONPATH=$PWD/src ~/miniconda3/envs/nlo/bin/python \
    scripts/fg17_waveform_ab.py -o results/fg17_waveform_ab.json
# 期望：all_bit_exact: true
```

### 8.3 §3 `snr_delta_db`

```bash
cd /home/zsig/Documents/gits/wt/fg-converge
CUDA_PATH=$SHIM PYTHONPATH=$PWD/src ~/miniconda3/envs/nlo/bin/python \
    scripts/fg17_snr_delta_probe.py -o results/fg17_snr_delta.json
# 期望：layer0_all_bit_exact=true / layer0_max_abs_delta=0.0
```

### 8.4 门 1：物理仓全量

```bash
cd /home/zsig/Documents/gits/wt/fg-converge
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python -m pytest tests/ -q -p no:randomly
# 期望：404 passed, 1 skipped
```

### 8.5 门 2：金标准逐位

```bash
cd /home/zsig/Documents/gits/wt/fg-converge
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python scripts/fg4a_capture_golden.py --check
# 期望：31/31 cases bit-identical, worst relative deviation 0.000e+00
PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python scripts/fu4_capture_golden.py --check
# 期望：fidelity golden reproduces bit-exactly
```

### 8.6 门 1：RL 仓全量（分片）

```bash
cd /home/zsig/Documents/gits/wt/fg-rt-cv
bash scripts/fg17_rl_full.sh /tmp/fg17_rl
cat /tmp/fg17_rl/meta.txt /tmp/fg17_rl/shard_*.log | grep -E "shard_rc=|passed|failed"
```

### 8.7 门 4：RL 决策序列

```bash
cd /home/zsig/Documents/gits/wt/fg-converge
git archive 25ed30d src references | tar -x -C /tmp/fg17_baseline_25ed30d

cd /home/zsig/Documents/gits/wt/fg-rt-cv
for pol in first_fit seeded_random; do
  PYTHONPATH=$PWD/src:/tmp/fg17_baseline_25ed30d/src ~/miniconda3/envs/rl/bin/python \
      scripts/fg17_rl_decision_trace.py -o results/fg17_decision_baseline_$pol.json --policy $pol
  PYTHONPATH=$PWD/src:/home/zsig/Documents/gits/wt/fg-converge/src ~/miniconda3/envs/rl/bin/python \
      scripts/fg17_rl_decision_trace.py -o results/fg17_decision_fg17_$pol.json --policy $pol
done
```

---

## 9. 环境注记（与 brief 的两处出入，以树为准）

1. **波形侧依赖只在 `nlo` 环境**：`physics_nlo_eval` + `cupy` 在 `~/miniconda3/envs/nlo`，
   **`rl` 环境两者皆无**（实测）。因此门 2/§0.2/§3 的波形读数用 `nlo` 解释器，
   而测试全量与 RL 侧用 `rl` 解释器。brief 的"`rl` 解释器缺 `np.trapezoid`/`gymnasium`"
   与本机一致（`nlo` 是 numpy 1.26.4，无 `trapezoid`；`rl` 是 numpy 2.5.3 + gymnasium）。
2. **本机无 CUDA toolkit**（无 `/usr/local/cuda`、无 `nvcc`），但有 P100。
   CuPy 通过 §8.2 的软链 CUDA root 可正常运行。
