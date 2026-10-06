# FU-4：`rwa/qot_checker.py` 两套 SNR 装配的裁决与合并

> **这是保真线 WP，它会改变结果。** 本文件不含任何速度/加速比声明——FG-4h 在并行跑同一台机器，
> 混进来的任何计时都会被那边的负载污染。
>
> 全文数字均来自可 `--check` 重渲的产物，**不手抄**：
>
> ```bash
> PYTHONPATH=$PWD/src python scripts/fu4_assembly_delta.py --check   # ΔSNR 表
> PYTHONPATH=$PWD/src python scripts/fu4_capture_golden.py --check   # 保真档金标准
> PYTHONPATH=$PWD/src python scripts/fu4_probe_terms.py              # 机理探针
> ```

---

## §0 背景：为什么这不是"一新一旧"

编排方已否决"把物理仓基线 rebase 到 `ecz-dev` 的 `70f851d`"，因为三方哈希实测显示
三个版本互不相同、两边各有对方没有的东西：

| 版本 | `rwa/qot_checker.py` | SNR 装配 | 灵活栅格阻断 |
| --- | --- | --- | --- |
| `origin/main`（`87850a1`） | `3cae234a…` | `insertion_loss_db + filtering_penalty_db` 合并施加 | 无（按信道索引） |
| FG 树（`164135d`/`a06d34c`） | `31d76135…` | 同 main | **有**（FG-2 移植） |
| `ecz-dev`（`70f851d`） | `a092132b…` | `path_loss_db(drop/express)` + booster ASE + 独立滤波乘子 | 无 |

本刀的立场：**权威版本是二者的并集**——保留 FG 树的灵活栅格阻断，采用一套有物理依据的
ROADM/booster/filtering 装配。

### 0.1 一个必须先讲清楚的事实：legacy 金标准**不会**变

brief 预期"legacy 金标准应该变"。**实测结论是它不变**，而且原因是结构性的：

- legacy 金标准（`results/fg4a_golden.json`，7 场景 × 31 输出量）走的是
  `MultiBandISRSGN.evaluate_path`，该路径的 ROADM 装配在 `models/isrs_gn.py` 内部，
  **从不调用** `rwa/qot_checker.py`。
- 硬证据：`QoTValidator` 在整个 `src/` 里只被 `__main__.py` 与 `rwa/allocators.py` 导入；
  `models/isrs_gn.py` 不引用它（`grep -rn "qot_checker import" src/`）。
- 改动后实测：`tests/test_fg4a_numeric_identity.py` **108 passed**，legacy 31/31 逐位不变。

**这条事实的真正含义比"绿灯"重要**：既然 legacy 金标准照不到 `rwa/qot_checker.py`，
那么在 FU-4 之前，**RWA 验证器的数字从来没有被任何逐位闸门钉住**。这正是 FG-4g 能发现
跨仓 0.2878 dB 分歧、而全仓没有一个测试变红的原因。所以 §3 的保真档金标准不是"顺手加的"，
而是这条代码路径**本来就缺**的第一道闸门。

---

## §1 三项裁决（逐项给依据）

### 1.1 ROADM 损耗：`path_loss_db("drop"/"express")` 胜出

**裁定**：采用按端口区分的 `path_loss_db`，不采用合并施加。

**依据（仓库内既有实现，不是"看起来更详细"）**：

1. 本仓的**波形级黄金模型** `src/network_nlo_eval/physics/qot_checker.py:250` 早就写成
   `insertion_loss_db=roadm.path_loss_db("drop" if last_hop else "express")`。
   也就是说，**同一仓库里物理最严格的那条路径已经采用了按端口取损耗**；FG 树的
   `rwa/qot_checker.py` 是少数几个没跟上的。
2. `ROADMConfig` 把 `express_attenuation_db`（直通端口附加衰减）与
   `add_drop_loss_db`（本地 add/drop 端口附加损耗）定义为**两个独立字段**
   （`network/elements.py:79-80`）。合并写法只读 `insertion_loss_db`，使这两个字段
   **完全失效**——它们就是为"末跳 drop 走不同物理路径"而存在的。
3. **实测反证**：在改动前捕获的基线里，`legacy_express_attenuation` 场景
   （把 `express_attenuation_db` 调到 2 dB）得到的 SNR 是 `23.125814425161867`，
   与基准场景 `legacy_two_hop_equalize_on` 的 `23.125814425161867` **逐位相同**。
   即 FG 树写法下 express 端口衰减对结果零影响——字段是死的。

**物理结构**：末跳 drop 多穿一对 WSS/解复用端口，损耗高于直通端口；这正是
`path_loss_db` 用不同端口参数表达的东西。合并写法在结构上无法表达末跳与中间跳的差异。

### 1.2 booster：`equalize_output_power` 分支胜出，但**只在它自己的拓扑下成立**

**裁定**：采用 booster 分支。`equalize_output_power=True` 时信号**不再**被端口损耗衰减，
改为加入 booster ASE `NF · h · ν · Rs · (loss − 1)`；为假时按端口损耗衰减信号与噪声。

**依据**：

1. 同一仓库的波形模型 `physics/qot_checker.py:255-262`：
   ```python
   if roadm.equalize_output_power:
       current = EDFA(gain_db=roadm.path_loss_db(...), nf_db=roadm.booster_noise_figure_db)
   ```
   物理含义一目了然——ROADM 后面挂一台 EDFA，**增益恰好抵消该 ROADM 的损耗**，
   于是出纤功率被拉回入纤值，代价是引入该 EDFA 自身的 ASE。这只在"输出功率均衡"
   拓扑下成立，代码里的 `if` 正是这个条件的显式表达。
2. `ROADMConfig.booster_noise_figure_db` 的字段文档写的是"补偿 ROADM 损耗的 booster NF"——
   它只在有补偿动作时才有意义。
3. FG 树写法里根本没有 booster，于是 `equalize_output_power` 与 `booster_noise_figure_db`
   两个字段**都是死的**。实测佐证：改动前 `legacy_two_hop_equalize_on` 与
   `legacy_two_hop_equalize_off` 逐位相同（均 `23.125814425161867`）。

**brief 要求实测的那一条——关掉 equalize 会不会退回 FG 树的结果？**

> **实测结论：不会。ΔSNR = −0.096115 dB，不是 0。**

机理（`scripts/fu4_probe_terms.py` 的同缓存分解，在**完全相同**的噪声缓存上对比两套规则）：

```
FG-tree : signal=9.120108e-05 W  noise=4.440364e-07 W  SNR=23.125814 dB
union   : signal=7.943282e-05 W  noise=3.953940e-07 W  SNR=23.029700 dB
delta   : -0.096115 dB
```

FG 树把 `insertion_loss(5 dB) + filtering(0.2 dB) = 5.2 dB` 一起除；并集把 express 5 dB
与 drop 6 dB 分别除、并把 0.2 dB 滤波惩罚**只乘在噪声上**。两者的信号/噪声比因此不同，
所以**关掉 booster 只抹平了 booster 这一个差异，滤波的差异依然在**。

> ⚠️ **一条被实测推翻的直觉，值得留在记录里**：我最初以为"损耗同时除信号和噪声会在 SNR 比值里
> 完全抵消，因此 FG 树的 ROADM 块对 SNR 是 no-op"。**这是错的**。噪声是逐跳累加的：
> 第 1 跳的噪声不会经历第 2 个 ROADM 的衰减，所以 SNR 实际形如 `P / (n₁ + n₂·L₁)`，
> 中间跳的损耗 `L₁` **不会**抵消。实测支持后者——`insertion_loss_db` 从 0 → 5 → 10 → 20 dB
> 时 SNR 为 26.160 → 23.030 → 18.839 → 9.227 dB，明显随损耗下降。
> 本仓实现的结构是正确的，被推翻的是我的推理，不是代码。

### 1.3 filtering penalty：独立噪声乘子胜出

**裁定**：`filtering_penalty_db` 作为**只作用于噪声**的独立乘子，不与插入损耗合并。

**依据**：

1. **字段文档就是判据**：`network/elements.py:76` 对该字段的描述是
   *"统计模型中的级联滤波 SNR 代价 (dB)"*，且约束为 `ge=0`。它被定义成
   **噪声域的量（SNR 代价）**，不是光功率损耗。把它并进 `insertion_loss_db` 等于把它
   变成"插入损耗"，与字段自身语义冲突。
2. 合并写法会让滤波惩罚与信号**同比例**衰减，于是它与插入损耗在 SNR 上**不可区分**——
   而"滤波惩罚只作用于噪声/串扰"是既有约定（它是级联滤波造成的、随级数累加的信噪比代价，
   不改变每信道的平均光功率）。实测佐证：改动前把 `filtering_penalty_db` 从 0.2 调到 2.0
   使 SNR 从 23.126 降到 21.682（−1.444 dB），量级只有一半左右，正是因为它一半的效应
   在信号上也施加了、随后在比值里抵消掉。

**三项都不完整时的第三种形式**：本仓 `physics/qot_checker.py` 用的是**波形级** ROADM
（WSS 超高斯传递函数 + 显式 EDFA），`isrs_gn.py` 用的是**统计级** lumped 规则，
`rwa/qot_checker.py` 用的是**解析 SNR 级**装配。三者粒度不同，本刀只在
`rwa/qot_checker.py` 的粒度上裁决，不试图统一另外两处（见 §5 待办）。

---

## §2 并集实现

改动文件：`src/network_nlo_eval/rwa/qot_checker.py`。

### 2.1 保留的 FG-2 能力（一处都没删）

`num_nodes`、`_is_blocked`（槽区间重叠）、`blocked_slots`、`connection_overlaps_slots`、
`slots_per_channel` 全部原样保留。阻断判定仍是 FG-2 的切片语义，与装配改动正交。
由 `tests/test_fu4_assembly_union.py::test_fu4_4b`（符号存在性守卫）逐项钉住。

### 2.2 新装配（`_roadm_terms` 单点定义）

标量路径（`_eval_service_snr`）与向量路径（`_build_path_snr_array`）现在**共用同一个
`_roadm_terms`**，不再各写一份：

```python
operation = "drop" if is_last_hop else "express"
loss = 10 ** (roadm.path_loss_db(operation) / 10)
filtering_penalty = 10 ** (roadm.filtering_penalty_db / 10)
booster_nf_lin = 10 ** (roadm.booster_noise_figure_db / 10) if roadm.equalize_output_power else None
```

### 2.3 一处两边都没有的第三种形式：booster 用**逐连接**频率

`ecz-dev` 与 RL 的 `scan.py` 都用 `spectrum_grid.frequencies[channel_idx]` 与标量
`gn_evaluator.symbol_rate_hz` 取 booster ASE 的 `ν` 与 `Rs`。**这在灵活栅格下是错的**：
那里 `channel_idx` 索引的是**连接（ODE 节点）**，与固定栅格的信道索引不是同一个索引空间。

本刀新增 `_node_frequency_and_rate`，改走 `gn_evaluator.f_abs_hz[channel_idx]` 与
`gn_evaluator.symbol_rate_hz_channels[channel_idx]`。遗留模式下
`f_abs_hz is spectrum_grid.frequencies` 且 `symbol_rate_hz_channels` 全等于标量，
故 legacy 路径逐位等价，flexgrid 路径得到修正。
由 `test_fu4_4c`（flex ≡ legacy 逐位）与 `test_fu4_4d`（逐连接频率）钉住。

---

## §3 变更记录与 ΔSNR

### 3.1 ΔSNR 表（`results/fu4_assembly_delta.json`，`--phase compare` 重渲）

| 场景 | 类型 | before (dB) | after (dB) | ΔSNR (dB) |
| --- | --- | --- | --- | --- |
| `legacy_two_hop_equalize_on` | legacy | 23.125814 | 26.034730 | +2.908916 |
| `legacy_three_hop_equalize_on` | legacy | 17.604538 | 24.152139 | +6.547601 |
| `legacy_two_hop_equalize_off` | legacy | 23.125814 | 23.029700 | -0.096115 |
| `legacy_express_attenuation` | legacy | 23.125814 | 25.989212 | +2.863397 |
| `legacy_high_filtering` | legacy | 21.681811 | 25.025782 | +3.343971 |
| `legacy_blocked_channel` | legacy | rejected | rejected | n/a（两端都被阻断） |
| `flex_two_hop_equalize_on` | flex | 23.125814 | 26.034730 | +2.908916 |
| `flex_three_hop_equalize_on` | flex | 17.604538 | 24.152139 | +6.547601 |
| `flex_two_hop_equalize_off` | flex | 23.125814 | 23.029700 | -0.096115 |
| `flex_blocked_slots` | flex | rejected | rejected | n/a（两端都被阻断） |

汇总（同一 JSON 的 `summary` 字段）：10 场景、8 个移动、2 个两端均被阻断、
`max_abs_delta_db = 6.547600967785943`、`min_delta_db = -0.09611463424176492`。

**场景集是冻在 `tests/fu4_scenarios.py` 的**：字面量、无 RNG、无时钟、无环境读取，
每次重跑必须逐位复现。

### 3.2 这是物理量级，不是数值量级

brief 要求证明这一刀动的是物理量级。证据链全部可复算：

1. **符号与量级由物理量直接解释**，不是浮点噪声。`equalize=True` 时 SNR **上升**
   （+2.9 ~ +6.5 dB），因为 booster 抵消了 ROADM 损耗、不再衰减信号；跳数越多、
   累积的端口损耗越大，收益越大（2 跳 +2.91 dB，3 跳 +6.55 dB）——这正是
   `loss − 1` 项随 `loss` 增长的直接后果。
2. **同一个已知参数变动，在改动后能被正确反映**。改动前 `equalize_output_power`
   与 `express_attenuation_db` 对 SNR **零影响**（§1.1、§1.2 的实测）；改动后
   equalize 开关给出 +2.91 dB 的差别。能把一个本该有物理意义的旋钮接通，
   本身就是"改到了物理"而不是"改到了浮点"的证据。
3. **与 ODE 核无关**。分解对比在**完全相同的噪声缓存**上进行（`fu4_probe_terms.py`），
   差异只来自装配规则本身，量级 ±0.1 dB 到数 dB，与 ODE 求解精度（FG-4a 闸门 ≤1e-12）
   相差 10 个数量级以上。
4. **数量级对照**：设备相关的量级。booster 收益 +2.9 ~ +6.5 dB 对应 5~11 dB 的端口损耗
   被补偿后减去新增 ASE 的代价；滤波惩罚 −0.096 dB 对应 2×0.2 dB 量级的噪声侧代价。
   都不是 "1e-13 相对偏差"那一类。

### 3.3 现有场景集的逐个 ΔSNR

brief 要求对 `fg1_scenarios` / `fg4a_scenarios` 也逐个报 ΔSNR。**这两个集合的 ΔSNR 恒为
0.000000，且是构造性的必然**，必须写清楚而不是含糊过去：

| 场景集 | 是否经过 `rwa/qot_checker.py` | ΔSNR |
| --- | --- | --- |
| `tests/fg1_scenarios.py`（4 场景） | 否——直接调 `evaluator._calc_span_noise_and_power` | **0.000000** |
| `tests/fg4a_scenarios.py`（7 场景 × 31 输出量） | 否——走 `evaluate_path`，装配在 `models/isrs_gn.py` | **0.000000** |
| RL 的 0.2878 dB 用例 | 是 | **未复现**（见 §5） |

验证方式：`tests/test_fg4a_numeric_identity.py` 改动后 **108 passed**（legacy 31/31 逐位
不变），全量 **261 passed / 1 skipped**。`0.000000` 不是"我没测"，是"该路径结构上到不了"，
有 §0.1 的导入关系作证。

### 3.4 保真档金标准与既有闸的处置

- **新开保真档金标准**：`results/fu4_fidelity_golden.json`（10 场景），
  由 `scripts/fu4_capture_golden.py` 采集、`--check` 重验，配套
  `tests/test_fu4_assembly_union.py` 13 项测试。
- **既有逐位闸（legacy 31/31）：未被改动，无需重定向。** brief 要求"若因此变红，必须显式
  重定向到参考档"——**它没有变红**，因为它照不到这条路径（§0.1）。此处如实记录：**没有发生
  需要裁定的事件**，不伪造一个。
- **零既有期望值修改**：`git diff` 中 `tests/` 的新增只有
  `tests/fu4_scenarios.py` 与 `tests/test_fu4_assembly_union.py` 两个新文件，
  既有 `tests/*` 一行未动（G2 纪律）。
- 变更记录即本节（§3），并随 `results/fu4_assembly_delta.json` 的
  `before` / `after` / `delta` 三列固化。

### 3.5 验收数字

| 项 | 数值 |
| --- | --- |
| FU-2 后基线 | 248 passed / 1 skipped |
| FU-4 后全量 | **261 passed / 1 skipped**（+13 项新测试） |
| legacy 金标准 | **31/31 逐位不变**（`test_fg4a_numeric_identity.py` 108 passed） |
| 保真档金标准 | 10 场景，`--check` 逐位重现 |
| 场景数 / 移动数 | 10 / 8 |
| `max_abs_delta_db` | 6.547600967785943 |
| 计时环境 | 本 WP 未做计时（保真线）。全量测试在 load average 21.5/16.6/13.6 下串行完成（`flock`），共 29.52 s |

---

## §4 RL 的 `scan.py` 需要复刻哪一套，以及哪一处会与 FG 线不符

本刀**不改** RL 的 `scan.py`（RL 仓，只读）。但必须写清复刻要求。

### 4.1 `scan.py` 现在已经是"并集"那一套

实测 RL 侧 `fg-rl-rebase/src/multiband_optical_rl/scan.py` 当前实现：

- `_roadm_terms` 取 `config.path_loss_db("drop" if is_last else "express")`（第 206 行）——
  与本刀 §1.1 裁定一致；
- `equalize_output_power` 为真时加 `booster_base · f · Rs · (loss − 1)`，为假时除 `loss`
  （第 377–387 行）——与 §1.2 裁定一致；
- `filtering_penalty_db` 作为独立乘子只乘噪声（第 389–391 行）——与 §1.3 裁定一致。

**所以 FG-4g 观测到的 0.2878 dB 分歧，根因就是 FG 树的 `rwa/qot_checker.py` 仍用
lumped 规则，而 RL 已经用并集规则。** 本刀把 FG 线改成并集后，FG-4g 那条分歧的成因被消除。

### 4.2 那一处仍会与 FG 线不符（必须点名）

**booster ASE 的 `ν` 与 `Rs` 取法。** `scan.py:154-155` 用的是：

```python
self._symbol_rate_hz = self.gn_evaluator.symbol_rate_hz   # 标量
self._frequencies = self.grid.frequencies                 # 固定栅格信道频率
```

而本刀 FG 线改用 `gn_evaluator.f_abs_hz[channel_idx]` 与
`gn_evaluator.symbol_rate_hz_channels[channel_idx]`（逐节点）。

- **legacy（固定栅格）下两者逐位相同**，不会分歧；
- **灵活栅格下两者必然分歧**：`grid.frequencies` 的长度是信道数，而 `channel_idx` 索引的是
  连接；且每连接的 `symbol_rate_hz` 可互不相同，标量取不到。
  ⇒ **RL 若要在灵活栅格下与 FG 线一致，`scan.py` 必须改成读逐节点数组。**
  这是本刀识别出的、FG-4g 那条用例**没有覆盖到**的唯一残留分歧点，已列入待办。

---

## §5 如实标注：未测 / 证据不足 / 不确定

1. **RL 的 0.2878 dB 用例：未复现。** brief 只给了两个数（25.385654 vs 25.099058）与
   "频段 L、142 信道、3 跳"这类线索，没有给完整场景构造（拓扑、EDFA 参数、ROADM 配置、
   符号率、功率），因此**无法**在本仓重建同一条用例，也就无法报它逐场景的 ΔSNR。
   已做的是：确认分歧根因（装配规则不同）、并给出机制解释与量级。**未声称该用例已闭合。**
2. **`isrs_gn.py` 的装配未统一（已知不一致，本次未动）。** `models/isrs_gn.py:445` 与 `:561`
   仍是 FG 树的 lumped 规则，而同仓 `physics/qot_checker.py` 用并集规则。于是本仓现在
   存在**统计级路径（`evaluate_path`）与解析 SNR 级路径（`verify_allocation`）对 ROADM
   处理不一致**的状态。本刀的文件所有权只覆盖 `rwa/qot_checker.py`，故未改。
   ⇒ **列为待办**：统一 `isrs_gn.py` 的装配，或明确写下"两条路径物理不同"的理由。
3. **booster 的 `booster_ase[signal_power <= 0] = 0` 保留，但物理上可疑。** 空槽/被阻断信道
   同样会由 booster 产生 ASE。保留它是为了与 RL `scan.py` 逐位一致（RL 显式实现了该守卫，
   并以其作为"逐位等价"契约）。实测该守卫**不改变任何输出**（信号为 0 的节点 SNR 必为
   −inf，下游按 `signal_power > 0` 过滤），所以保留它没有数值代价——但它是一个
   **为对齐而保留的物理近似**，记入待办。
4. **三项裁决的证据强度不同。** §1.3（filtering）证据最强——字段文档直接把它定义为
   噪声域 SNR 代价。§1.1（端口损耗）证据强——同仓波形黄金模型已如此实现，且被 merge 写法
   会让两个端口字段失效的反证所支撑。§1.2（booster）的**物理机制**证据强（同仓波形模型
   有对应 EDFA），但"RL 的这条用例闭合后差多少"未测（见第 1 条）。
   **没有一项是靠"另一套看起来更详细"当选边理由的。**
5. **未做的加速声明**：本文件不含任何 timing/加速比。FU-4h 在并行跑同一台机器，
   本 WP 的 load average 常年 17~21，计时不可信。
6. **本机负载**：全量测试用 `flock` 串行化，未触碰同机其它任务的进程
   （load average 21.5/16.6/13.6 → 21.7/17.1/13.8）。

### 待办清单

- [ ] RL 侧补齐 0.2878 dB 用例的完整参数，闭合该用例的 ΔSNR。
- [ ] RL `scan.py` 的 booster `ν`/`Rs` 改读逐节点数组（灵活栅格下必需）。
- [ ] 统一或明确论证 `models/isrs_gn.py` 与 `rwa/qot_checker.py` 的 ROADM 装配。
- [ ] 复核 `booster_ase[signal_power <= 0] = 0` 的物理合理性（当前为对齐 RL 而保留）。

---

## §6 产物清单

| 文件 | 作用 |
| --- | --- |
| `docs/fu4_qot_assembly_adjudication.md` | 本文件：裁决 + 变更记录 + ΔSNR 表 |
| `results/fu4_assembly_delta.json` | ΔSNR 产物（`before`/`after`/`delta`/`summary`），`--check` 可重验 |
| `results/fu4_fidelity_golden.json` | 保真档金标准（10 场景），`--check` 逐位重现 |
| `scripts/fu4_assembly_delta.py` | ΔSNR 采集/对比/重验 |
| `scripts/fu4_capture_golden.py` | 保真档金标准采集/重验 |
| `scripts/fu4_probe_terms.py` | 机理探针：灵敏度扫描 + 同缓存 FG/并集分解 |
| `tests/fu4_scenarios.py` | 冻结场景集（`tests/` 为真源，`scripts/` 反向导入） |
| `tests/test_fu4_assembly_union.py` | 13 项新测试（结构 / FG-2 回归 / 保真） |
| `src/network_nlo_eval/rwa/qot_checker.py` | 改动后的并集装配 |