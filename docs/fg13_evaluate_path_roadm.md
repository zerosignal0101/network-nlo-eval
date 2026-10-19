# FG-13：`evaluate_path` 的 ROADM 施加点 —— 裁决

> **裁决：缺陷（DEFECT），不是"刻意不同的模型"。**
> `MultiBandISRSGN.evaluate_path` 用合并式 ROADM 装配、完全不读 `express_attenuation_db`
> ——连同另外三个有明确物理语义的字段——这是缺陷。
>
> **但本刀没有改它的数值。** 修复被一道**出处约束**挡住，见 §5。
> 这是有意的：绕过那道约束去改，就是伪造金标准出处。
>
> 本文所有数字均可复算：
>
> ```bash
> PYTHONPATH=$PWD/src python scripts/fg13_probe_roadm.py --check   # 字段存活普查
> PYTHONPATH=$PWD/src python scripts/fg13_impact_probe.py          # 修复影响量化
> PYTHONPATH=$PWD/src python scripts/fg4a_capture_golden.py --check # 既有金标准
> PYTHONPATH=$PWD/src python scripts/fu4_capture_golden.py --check
> ```

---

## §1 裁决要回答的问题与本刀给出的答案

brief 问：`evaluate_path` 的合并式写法、忽略 `express_attenuation_db`——**是缺陷，还是刻意不同的模型**？

答案是**缺陷**。判断依据不是"两处看起来应该一致"，而是三条各自独立的事实：

1. **权威定义在本物理仓，且按端口区分。** `ROADMConfig.path_loss_db()`
   （`src/network_nlo_eval/network/elements.py:113-116`）按 `operation` 返回
   `insertion_loss_db + (express_attenuation_db if express else add_drop_loss_db)`。
   `evaluate_path` 不读它，不是"另有定义"，是**没有定义**。
2. **同仓另外两处 ROADM 施加点都是按端口 + booster 的**，二比一。
   - 波形级：`src/network_nlo_eval/physics/qot_checker.py:248-261`——用
     `path_loss_db("drop" if 末跳 else "express")`，并且**显式**走
     `equalize_output_power` + `booster_noise_figure_db` 的 EDFA 分支；
   - 解析 SNR 级：`src/network_nlo_eval/rwa/qot_checker.py:118-128`（`fold_roadm_terms`）。
3. **FU-4 明确把这一处列为未决，而不是已裁。** `docs/fu4_qot_assembly_adjudication.md`
   §5 第 2 条与待办清单最后一项："统一**或明确论证** `models/isrs_gn.py` 与
   `rwa/qot_checker.py` 的 ROADM 装配"。
   ⇒ **从未有人裁过它是"刻意"**。FG-11 的复核只是把"express 是死的"这条反证的
   适用范围收窄到本路径，并没有改变它的定性。

### 1.1 docstring 里的理由已经失效（这是关键的一步）

`evaluate_path` 的 docstring（`models/isrs_gn.py:665-671`）给的理由是两条：

> "ROADM 衰减在此处按 main 的约定统一施加于信号与已累积噪声，
> **与 `rwa/qot_checker` 的链路切分规则一致**。"

第一条是**出处**（provenance），不是物理依据——"main 这么写的"不构成"这么写是对的"。
第二条在 FU-4 之后**已经不成立**：FU-4 改了 `rwa/qot_checker.py`，两处现在结构上不同。

⇒ 合并写法**没有任何仍然有效的书面理由**。若它是刻意的模型差异，应该有一句
"本路径是另一套口径，故不读端口字段"；这句话在仓库里不存在。
**"没有被论证过的遗留写法"不等于"刻意"。**

### 1.2 端口信息在这条路径上并非结构性不可得

合并写法常被解释成"这条路径拿不到末跳信息"。**实测不成立**：循环里
`hop_index` 就在手边（`models/isrs_gn.py:681`），末跳判定就是
`hop_index == len(optical_path.fiber_configs) - 1`——**波形级那处用的正是同一个表达式**
（`physics/qot_checker.py:249`）。

---

## §2 实测：惰性字段是逐位死的（核心证据）

`scripts/fg13_probe_roadm.py`，两跳各挂一个 ROADM（hop0 = 非末跳 ⇒ express 语义，
hop1 = 末跳 ⇒ drop 语义），`ode_accel="off"`，逐字段扫描：

| 字段 | `evaluate_path` | max\|ΔSNR\| (dB) | RWA 折叠的 express / drop 损耗变化 |
| --- | --- | --- | --- |
| `insertion_loss_db` | 活 | `2.5507350388219034` | `+3.147295785` / `+3.962210642` |
| `filtering_penalty_db` | 活 | `1.4994157059056317` | `0` / `0`（它是独立噪声乘子，不进损耗） |
| `express_attenuation_db` | **死** | **`0.0`（逐位不变）** | **`+1.849594676`** / `0` |
| `add_drop_loss_db` | **死** | **`0.0`（逐位不变）** | `0` / **`+2.328501739`** |
| `booster_noise_figure_db` | **死** | **`0.0`（逐位不变）** | `0` / `0` |
| `equalize_output_power` | **死** | **`0.0`（逐位不变）** | `0` / `0` |

**读法**：四个字段的 max\|ΔSNR\| 是**恰好 `0.0` 且逐位相等**，不是"影响很小"。
同时最后两列用同一把尺子显示：它们在 RWA 并集路径上是**活的**（express 与 drop 各自
只在自己那个端口上生效——这也顺带证明 `path_loss_db` 的端口语义确实被消费了）。
**没有最后两列，这张表只能说明"`evaluate_path` 没读"，说明不了"该读却没读"。**

brief 特别点名的 `equalize_output_power` 默认值是 `True`、`booster_noise_figure_db`
默认值是 `5.0`（`network/elements.py:81-82`）：**默认配置下 booster 物理整段缺失**。

### 2.1 一条必须挡住的假阳性：末跳抵消 ≠ 字段是死的

`evaluate_path` 把损耗**同时除信号与全部已累积噪声**（`models/isrs_gn.py:706-709`）。
于是**末跳** ROADM 的损耗在 `power/(spm+xpm+ase)` 里代数抵消。
若拿一个**根本没被读**的字段去测末跳，同样得到"逐位不变"——两种解释被混为一谈。

所以对照必须用一个 `evaluate_path` **确实读取**的字段。用 `insertion_loss_db`（5→8 dB）：

| 位置 | 逐位不变 | max\|ΔSNR\| (dB) |
| --- | --- | --- |
| 仅末跳挂 ROADM | 否 | `3.552713678800501e-15`（浮点残差，非严格逐位） |
| 仅**非末跳**挂 ROADM | 否 | **`2.5507350388219034`**（可观测） |

**结论**：末跳抵消是**代数恒等**（残差 1e-15 量级），非末跳可观测。
⇒ **修复时不能因为"末跳损耗看起来无效"就跳过端口语义**；末跳该走 drop 端口，
只是它对 SNR 恰好不敏感。这一点已由 `tests/test_fg13_evaluate_path_roadm.py`
的 `test_fg13_last_hop_cancellation_is_not_mistaken_for_inertness` 钉住。

---

## §3 修的话，物理量是这些（影响量化）

`scripts/fg13_impact_probe.py` 在**进程内**把 `evaluate_path` 的 ROADM 段换成权威口径
（逐字对齐 `accumulate_service_snr`，见文件头），**不改动仓库任何文件**，
再跑既有场景：

| 场景集 / 场景 | ΔSNR (dB) | 相对偏差 | 说明 |
| --- | --- | --- | --- |
| `fg1_scenarios/c_band_single` | `0.000000`（逐位不变） | — | 无 ROADM，构造性不受影响 |
| `fg1_scenarios/multiband_wide` | `0.000000`（逐位不变） | — | 无 ROADM |
| `fg1_scenarios/kurtosis_64qam` | `0.000000`（逐位不变） | — | 无 ROADM |
| `fg1_scenarios/roadm_hoop` | **`0.978809`** | `5.539e-02` | ROADM 在**末跳** |
| `fg4a_scenarios/path_cplusl_roadm` | **`5.186963`** | `2.891e-01` | ROADM 在**非末跳** |

**受影响调用方**（brief §1.3 要求说明各自拿输出做什么）：

| 调用方 | 拿 `evaluate_path` 的输出做什么 | 是否受影响 |
| --- | --- | --- |
| `physics/comparison.py:80` `TopologyQoTComparator.evaluate` | 统计模型 SNR；`run_waveform=True` 时与波形引擎 SNR 相减得 `snr_delta_db`（`comparison.py:93`） | **是**。这正是统计/波形**跨模型一致性**的度量，统计侧忽略端口损耗会直接污染这个 delta |
| `physics/qot_checker.py:327/343` | ⚠️ **不是本路径的调用方**，见 §4.1 | 否 |
| `tests/fg1_scenarios.py` / `tests/fg4a_scenarios.py` | 两套既有金标准的采集入口 | 是（§4） |
| `scripts/fu1_parity.py`、`fg4a_bench.py`、`fg4g_*` | 诊断/基准脚本 | 是（随金标准漂移） |

---

## §4 修的话，会撞到哪些既有期望值（逐条归因）

在 scratch 副本（`/tmp/fg13-scratch-*`，**不含本 worktree 的任何改动**）实际应用修复后
跑全量：**`18 failed, 307 passed, 1 skipped`**（基线 `325 passed / 1 skipped`）。
逐条机制归因：

| # | 失败项 | 数量 | 机制 | 能否正当重钉 |
| --- | --- | --- | --- | --- |
| 1 | `test_fg1_physics_parity.py::test_pipeline_matches_origin_main[3]` | 1 | `roadm_hoop` 的 SNR 对 `results/fg1_golden_main.json` 偏 `0.978809 dB` | ❌ **不能**，见 §5 |
| 2 | `test_f4a_3_bit_identical_to_golden[off/cached_g::path_cplusl_roadm::*]` | 10 | `results/fg4a_golden.json` 的 5 个输出量（signal/spm/xpm/ase/snr）随新装配移动 | ✅ 可，`fg4a_capture_golden.py --write` |
| 3 | `test_f4a_4_matvec_within_budget[matvec::path_cplusl_roadm::*]` | 5 | 同上——该断言的 `expected` 取自 `_golden_array`（`test_fg4a_numeric_identity.py:281`），经由 golden 间接比较档位 | ✅ 随 #2 一并重钉 |
| 4 | `test_f4a_4_measured_deviation_is_bounded_and_reported` | 1 | 汇总 #2/#3 的实测偏差 | ✅ 随 #2 |
| 5 | `test_f4a_capture_script_check_mode_passes` | 1 | `--check` 子进程对旧 golden 失败 | ✅ 随 #2 |

### 4.1 顺带更正 brief 的一处事实

brief §0 写"`evaluate_path` **确实有活调用方**：`physics/qot_checker.py:327/343`"。
**实测不成立**：那两行属于 `PyNLOQoTEvaluator.evaluate_path`
（定义在 `physics/qot_checker.py:305`，`:327` 是 `evaluate_topology_path` 转发、
`:343` 是 `evaluate_channel_evm_and_ber` 转发），是**波形引擎自己的同名方法**。
该文件只从 `isrs_gn` 导入 `MultiSpanOpticalPath`（`physics/qot_checker.py:14`），
**从不调用统计侧的 `MultiBandISRSGN.evaluate_path`**。

`src/` 内统计侧 `evaluate_path` 的**唯一**调用方是 `physics/comparison.py:80`
（`grep -rn "evaluate_path" src/` 的完整结果见 §8）。
⇒ 这**不改变裁决**（惰性字段由 `ROADMConfig` 的字段文档与 `path_loss_db` 定义坐实，
不依赖调用方数量），但它改变"影响面"的描述：受影响的生产调用方**只有一处**。

### 4.2 不变量复核：修复没有破坏 FG-4a 的档位一致性

#3 名字里有 "matvec_within_budget"，容易误读成"档位不变量被打破"。**实际没有。**
该断言的 `expected` 取自 golden，而不是同树的 `off` 档位；它在修复后失败，是因为
**golden 记录的是旧的合并式物理**。直接在修复后的树上比较同树两档：

```
off vs matvec（修复后，path_cplusl_roadm）
  signal_power_w  max_rel=1.768e-15      snr_db  max_rel=7.829e-16
  spm_noise_w     max_rel=1.491e-15      span_count / roadm_count  逐位不变
  xpm_noise_w     max_rel=1.282e-15
  ase_noise_w     max_rel=8.141e-16
```

`≤1.8e-15`，**远在 `1e-12` 预算之内** ⇒ **FG-4a 的加速档位不变量（off ≡ matvec）未被破坏**，
`_calc_exact_power_profile_ode` 内核未被触碰（修复只改 ROADM 装配那一段）。
**唯一移动的是"记录旧物理的 golden"，不是不变量。**

---

## §5 为什么本刀没有落代码修复（诚实标注）

**修复在物理上明确、在工程上被一道出处约束挡住：**

`results/fg1_golden_main.json` 声明 `source_branch: "origin/main"`，且
`tests/test_fg1_physics_parity.py` 的文件头明确写着它**只能**从一份干净的 main 检出
重新生成（`scripts/fg1_parity.py --emit-golden`）。该测试的命题是
"`evaluate_path` 逐位复现 `origin/main`"。

一旦修复 `evaluate_path`，这个命题按定义不再成立。要让 #1 变绿，只能：
把 FG 树的端口口径写进一个**声称来自 `origin/main`** 的金标准文件。
**那就是伪造出处。** 我没有这个授权，也不认为应当自己做这个决定。

于是：
- FG-4a 那 17 项（#2–#5）**本来可以正当重钉**（golden 可从本树重新采集）；
- 但只要 #1 还红，**门 1（全量不新增失败）就无法满足**；
- 为让门 1 变绿而去重钉 #1，是本 brief 明令禁止的（G2 纪律 + 跨仓规则 4/6）。

⇒ **按 §24 的规矩，这里停下来归因，而不是硬改。** 需要编排方裁的是一句话：
**"`evaluate_path` 与 `origin/main` 的逐位一致性（设备装配层面）算不算 §25 那四条不变量之一？"**

- 若**算** ⇒ 本缺陷需要一条显式的"与 main 分歧"授权，然后重钉 FG-4a 那 17 项并保留 #1 作为已知分歧记录；
- 若**不算** ⇒ 需要一条"允许重钉 FG-1 金标准"的授权，但它将**永久失去对 main 的可追溯性**，
  我建议改为把 FG-1 的 golden 拆成"内核逐位"与"装配按裁决"两档，而不是覆盖原文件。

### 5.1 `filtering_penalty_db` 的一个反面证据（如实记录）

有一条**支持"刻意"**的证据必须摆出来：`filtering_penalty_db` 的字段文档是
*"统计模型中的级联滤波 SNR 代价 (dB)"*（`network/elements.py:76`）——
它**明确把自己限定在统计模型里**，而 `evaluate_path` 正是统计模型。这给"合并"提供了部分掩护。

本刀**不采纳**它作为整体理由，理由有二：
1. 它至多解释 `filtering_penalty_db` 一个字段，**完全解释不了**四个端口/booster 字段的惰性；
2. 把它并进一个**同时除信号和噪声**的线性损耗，与它自身"SNR 代价"的语义冲突：
   并进去之后它在比值里被抵消掉约一半（FU-4 §1.3 实测 −1.444 dB vs 理论 2×0.2 dB 量级）。
   FU-4 §1.3 的裁决对象虽限定在 `rwa/qot_checker.py`，但其**依据**（字段文档即判据）
   对本路径同样成立。

**故本裁决的强度分层**：端口 + booster 三个字段 = **证据强，判为缺陷**；
`filtering_penalty_db` 的合并写法 = **证据中等**，倾向缺陷但保留上表的反面证据。

---

## §6 本刀交付了什么

**没有**改任何物理数值。交付的是**裁决 + 可复算证据 + 防回归守卫**：

| 产物 | 作用 |
| --- | --- |
| `scripts/fg13_probe_roadm.py` | 字段存活普查 + 末跳/非末跳结构性对照；`--emit` / `--check` |
| `scripts/fg13_impact_probe.py` | 进程内替换装配，量化 ΔSNR 与碰撞面；不改仓库文件 |
| `results/fg13_roadm_probe.json` | 上述普查的固化读数（`--check` 可重验） |
| `results/fg13_impact_probe.json` | 上述影响量化的固化读数 |
| `tests/test_fg13_evaluate_path_roadm.py` | 8 项守卫（惰性清单 / 权威口径位置 / 假阳性防线 / 影响量级 / 两套金标准仍重现） |
| `docs/fg13_evaluate_path_roadm.md` | 本文 |

**守卫的设计意图**：它们钉的是**当前状态**，不是"当前状态正确"。
一旦有人合入修复，惰性清单与结构守卫会**变红**——这是**预期的**，
逼那次改动显式更新期望值并在文档归因，而不是悄悄改绿。
同时 §2.1 的对照测试挡住"把末跳代数抵消误当成字段惰性"这个反向错误。

---

## §7 未闭合项（不猜，列清楚缺什么、谁能给）

1. **§5 那一句话的裁定**：FG-1 金标准能否为装配口径而重钉？谁有授权？
   —— 这是唯一挡住修复的东西。需要编排方。
2. **`evaluate_path` 应当采用哪一档粒度**：完整并集（含 booster ASE 注入）还是仅端口化？
   本刀的量化按**完整并集**做（与 `accumulate_service_snr` 逐字对齐）。
   若只做端口化、不加 booster，`equalize_output_power`/`booster_noise_figure_db`
   仍是惰性字段，缺陷只修一半。本刀**未**替编排方选这一档。
3. **RL 的 0.2878 dB 用例**：FG-4 已记为未复现，本刀**未触及**（不在授权范围，且需要 RL 侧补参数）。
4. **未做任何计时声明**。本 WP 全量 59.97 s（`flock` 串行，load 2.04→2.27），
   该数字仅用于"闸门开销可接受"，**不含**任何加速比含义。

---

## §8 复算与关键行号

```bash
# 闸门：editable 的 .pth 指向主检出，与 cwd 无关，PYTHONPATH 是硬要求；
# 断言 import 到的包落在本树内，全量用 flock 串行，并同时报 load average。
export PYTHONPATH="$PWD/src"
PY=~/.miniconda3/envs/rl/bin/python           # nlo 环境缺 np.trapezoid / gymnasium
$PY -c 'import network_nlo_eval as n; print(n.__file__)'   # 必须打印本 worktree 的路径
flock /tmp/fg13.lock $PY -m pytest -q          # 基线 325 passed / 1 skipped

# 本刀证据
PYTHONPATH=$PWD/src python scripts/fg13_probe_roadm.py    --emit results/fg13_roadm_probe.json
PYTHONPATH=$PWD/src python scripts/fg13_impact_probe.py  --emit results/fg13_impact_probe.json
PYTHONPATH=$PWD/src python scripts/fg13_probe_roadm.py   --check results/fg13_roadm_probe.json

# 门 3（两套既有金标准）
PYTHONPATH=$PWD/src python scripts/fg4a_capture_golden.py --check   # 31/31 逐位
PYTHONPATH=$PWD/src python scripts/fu4_capture_golden.py  --check   # bit-exactly
```

| 位置 | 内容 |
| --- | --- |
| `models/isrs_gn.py:705` | 被裁决的合并式一行（`insertion_loss_db + filtering_penalty_db`） |
| `models/isrs_gn.py:706-709` | 同时除信号与全部累积噪声（末跳代数抵消的来源，§2.1） |
| `models/isrs_gn.py:665-671` | docstring 的失效理由（§1.1） |
| `network/elements.py:79-80` | `express_attenuation_db` / `add_drop_loss_db` 的字段文档 |
| `network/elements.py:81-82` | `booster_noise_figure_db` / `equalize_output_power`（默认 `5.0` / `True`） |
| `network/elements.py:113-116` | **权威定义** `path_loss_db` |
| `rwa/qot_checker.py:118-128` | `fold_roadm_terms`：按端口 + 滤波独立 + booster |
| `rwa/qot_checker.py:133-186` | `accumulate_service_snr`（影响量化所对齐的口径） |
| `physics/qot_checker.py:248-261` | 波形级：同仓另一处按端口 + booster 的施加点 |
| `physics/qot_checker.py:305/327/343` | brief 误记为调用方，实为波形引擎同名方法（§4.1） |
| `physics/comparison.py:80` | 统计侧 `evaluate_path` 在 `src/` 内的唯一调用方 |
| `docs/fu4_qot_assembly_adjudication.md` §5.2 | "统一**或明确论证**"——本处的未决状态出处 |