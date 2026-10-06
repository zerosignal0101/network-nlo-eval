# FG-4g 归因：matvec 与参考档的分歧裁决

> 本文件由 `scripts/fg4g_render_doc.py` 从 `results/fg4g_attribution.json` 重渲，
> **不手抄任何数字**。复算：`$PY scripts/fg4g_attribution.py --check`。

> **关于两个 dB 数的口径。** brief 里的 *0.314 dB* 是编排方在**全部 800 例**上报的最差值；
> 本刀在自己的定位子集（FG-4e 树下 2 例）上实测到
> **0.287773 dB**。两者不是同一个统计量，**本刀未独立复算 800 例/433 失配的计数**
> （见第 8 节）。但本刀的裁决**不依赖** 0.314 这个具体数值：它依赖的是
> "换档前后该数值**逐位不变**"这一条。

## 0. 结论

**(B)** —— 0.314 dB 与 ode_accel 无关：同一棵 FG-4e 树下把档位从 off 换成 matvec，最差差值逐位相同；根因是两棵树 qot_checker.py 的 ROADM/booster QoT 语义不同，而 scan.py 复刻的是较新（ecz-dev）的那一套。

一句话版本：**0.314 dB 与 `ode_accel` 没有任何关系。** 把档位从 `off` 换成
`matvec`，差分结果**逐位不变**；真正的分歧是 `rwa/qot_checker.py` 装配 SNR 的那段
代码在两棵树之间**已经改过了**，而 RL 的 `scan.py` 复刻的是较新（`ecz-dev`）的那一套。

因此：

* FG-4a 金标准的 <= 1e-12 契约**没有被证伪**；
* 速度线的 4.678x / 6.42x **不需要重新取证**；
* 需要重新对齐的是 **RL 与物理仓的基线**，不是加速档位。

## 1. 编排方的那张表为什么具有误导性

那张表把"主检出"叫做"参考档"，把"FG-4e 树"叫做"`matvec` 档"。但这两棵树的差异
远不止加速档位：

| 维度 | `ecz-dev`（主检出） | `wp/fg-ode-fast`（FG-4e 树） |
|---|---|---|
| 提交 | `70f851d` Add topology-aware QoT comparison across both engines. | `164135d` |
| `qot_checker.py` | `3ef3ac8a5f861fea` | `593b9eb68fac7b11` |
| 有无 `ode_accel` | 否 | 是 |
| 有无 FG-4e 批量核 | 否 | 是 |

`ecz-dev` 上有、而 `wp/fg-ode-fast` 上没有的、且**动过 SNR 装配**的提交：

| 提交 | 标题 |
|---|---|
| `70f851d` | Add topology-aware QoT comparison across both engines. |
| `e78fc62` | Fix SNR check and add noise cache. |
| `e0392e9` | Add physical qot checker. |

也就是说，被命名为"参考档"的那一侧，其 SNR 语义已经是**更新的一版**。

## 2. Q1：库自己的路径，两档差多少？

在真实 RL 工况（230 x 50 GHz C+L、多跳 1/2/3/7、双频段、
非空占用）下，用**库自己的公开入口**比较 `ode_accel="off"` 与 `"matvec"`，
两边喂同一份输入、同一份 `NetworkState`：

| 入口 | 用例 | 有差异例 | 最大绝对差 | 最大相对差 |
|---|---|---|---|---|
| `QoTValidator.verify_allocation`（SNR，dB） | 102 | **0** | 0.000e+00 dB | 0.000e+00 |
| `_calc_span_noise_and_power`（逐跨段，W） | 48 | 10 | 2.647e-23 | 3.138e-16 |
| `evaluate_path`（端到端） | 30 | 5 | 1.034e-25 | 3.714e-16 |

**判据：任何一层 > 1e-12 就要报 (A) 候选。三层都在 1e-15 量级，(A) 被排除。**

其中 `verify_allocation` 这一层是**逐位相等**的（0/102 有差异，
最大相对差 0.000e+00）—— 这是最强的一条：SNR 闸门自己的输出在
两档之间一个 bit 都没动。另有 698 例两档都判不可行（不参与比较），
判定不一致 0 例。

## 3. Q2：分歧到底在哪一层？

在 FG-4e 树里分别强制 `off` 与 `matvec`，跑同一组 RL 差分用例，并对
`_calc_span_noise_and_power` 的**每次调用**（入纤谱、跨段长度、NF、四个输出）做并列比对：

| 树 / 档位 | 用例 | 失配 | 最差绝对差 (dB) | 编排并列结果 |
|---|---|---|---|---|
| FG-4e / `off` | 2 | 2 | 0.287773 | identical |
| FG-4e / `matvec` | 2 | 2 | 0.287773 | identical |
| `ecz-dev` 参考树 | 101 | 0 | 0.000e+00 | identical, power_profile, span_length_m |

两条读数合起来就是裁决：

1. **换档不动。** `off` 与 `matvec` 的最差差值
   0.287773 vs 0.287773，
   `bit_identical_across_tiers = True`。
   档位对 0.287773 dB 这件事**零贡献**。
2. **物理核两边完全一致。** 编排并列结果是
   `identical` —— 两条路径发给内核的调用、以及内核给出的输出，**逐位相同**。
   既然输入与输出都相同，SNR 却不同，那么分歧**必然在物理核之外**，
   在把噪声装配成 SNR 的那段代码里。

### 3.1 第一个不同的量

在 FG-4e 树上，同一个用例（1 跳 / band 0 / ch 0）：

| 侧 | `verify_allocation` 的 SNR | `verify_for_scan` 的 SNR |
|---|---|---|
| FG-4e 树 | 25.385654 dB | 25.100024 dB |
| `ecz-dev` 树 | 25.099058 dB | 25.099058 dB |

**动的是 `verify_allocation`，不是 `scan.py`。** `scan.py` 在两棵树上给出几乎相同的值
（25.0991 vs 25.1000），
而 `verify_allocation` 从 25.0991 跳到
25.3857。差值 0.2866 dB。

### 3.2 机制

`ecz-dev` 的 `_build_path_snr_array` / `_eval_service_snr` 相对 FG-4e 树多三件事：

* ref _build_path_snr_array/_eval_service_snr 用 roadm.path_loss_db('drop'|'express') 区分末端/中间损耗；FG-4e 树用合并的 10**((insertion_loss_db+filtering_penalty_db)/10)
* ref 在 equalize_output_power 为真时把 booster ASE 加进 total_ase；FG-4e 树没有这个分支，一律除以 loss
* ref 把 filtering_penalty_db 单独乘一遍 total_*；FG-4e 树把它折进上面的合并 loss

这三处叠加，在默认拓扑（ROADM 的 `equalize_output_power` 全为真）上给 `total_ase`
加了一份 **booster ASE**，噪声变大、SNR 变小 —— 正好是
0.2878 dB 这个量级与方向。

FG-4e 树里 `rwa/qot_checker.py` 完全不出现 `ode_accel` 这个词：SNR 闸门对加速档位无知。
所以"换档导致 SNR 变"在结构上就不可能发生。

## 4. Q3：FG-4a 金标准当时到底比了什么？

如实回答（口径自查，不是指控）：

* **比在哪一层**：
  * MultiBandISRSGN._calc_span_noise_and_power  (6 span scenarios x 4 arrays)
  * MultiBandISRSGN.evaluate_path              (1 path scenario x 7 arrays)
* **输入怎么构造**：全部经 tests/fg4a_scenarios.py 的稳定评估器 API 构造，不含 RL 的 NetworkTopology / QoTValidator / ROADM 编排
* **规模**：7 个场景、31 个输出
* **是否覆盖 RL 工况**：
  * N=230 / 50 GHz / C+L：是
  * 非空占用：是
  * `evaluate_path` 多跳编排：否
  * 两频段 `band_plan` 选择：否
  * ROADM `equalize` booster ASE：否

金标准覆盖 230×50 GHz C+L 与非空占用（cl_partial_100km 每 7 个空槽），也覆盖 evaluate_path 的 3 跳编排（path_cplusl_roadm），但用的是固定 ROADMConfig，不含 equalize booster 分支；也不经 QoTValidator，因此不含 band_plan 的两频段选择。

**它并没有比错。** 金标准比的是**物理层**（内核输出），RL 比的是**编排后的 SNR**。
brief 里"两句话不可能同时为真"的根源候选**不成立**：本刀实测证明物理层两档 <= 1e-15，
编排层的分歧另有来源（基线偏斜）。FG-4a 真正没覆盖的是
`roadm_equalize_booster_ase` 与 `band_plan` 两频段，而那两条**在物理层内**，
本刀的 `tests/test_fg4g_matvec_attribution.py` 已把它们补上（并保持 1e-12 不放松）。

## 5. 能推翻本结论的具体观测

本结论是 (B)。下列任一读数出现，就必须改判：

| 若观测到 | 则结论改为 |
|---|---|
| 同一棵 FG-4e 树下 `off` 与 `matvec` 的差分**结果不同** | (A)，且需重新取证 4.678x / 6.42x |
| `verify_allocation` 相对 SNR 大于 1e-12（当前 102 例为 0.000e+00） | (A) |
| 编排并列出现 `kernel_output`（同输入不同输出） | (A)，指向内核不确定性 |
| `rwa/qot_checker.py` 开始引用 `ode_accel` | (A) 的必要条件成立，须重新取证 |

反向自查：若 `ecz-dev` 树上也出现同样的失配，则本结论的"基线偏斜"归因作废，
需要重新找第三种解释。

## 6. 影响面：哪些既有数字仍然可信

| 数字 | 判定 | 依据 |
|---|---|---|
| FG-4a 金标准（matvec <= 1.09e-15） | **仍可信** | Q1：102 例 SNR 逐位相等，最差相对差 0.000e+00 |
| FG-4b 4.678x | **仍可信** | 归因落在 QoT 装配，与 NLI 张量化无关 |
| FG-4e 6.42x | **仍可信** | 同上；本刀实测批量核 vs 串行 matvec 亦在 1e-15 量级 |
| FG-U1 / FG-U2 | **需重看，但方向相反** | 它们若引用了 `qot_checker` 的 SNR 语义，则基线偏斜会污染它们；这与 `matvec` 无关 |

## 7. RL 侧最小改动面（只写，不改）

分歧在 RL 与物理仓的**基线**，不在加速档位。二选一：

1. **把物理仓基线对齐到 `ecz-dev`**（推荐）。FG-4e 树 rebase 到
   `70f851d` 之后，`scan.py` 的复刻与之逐位相符，
   差分回到 0 不一致，FG-4f 的测试直接转绿。
2. **把 `scan.py` 的 ROADM 复刻改回 FG-4e 树那一版**（合并 loss、去掉 booster ASE 分支）。
   代价是丢掉 `ecz-dev` 的拓扑感知 QoT 改进，且需要改
   `tests/test_scan_fastpath.py` 的 ROADM 变体用例。

无论选哪条，`scan.py:645` 的批量门控（`ode_accel != "matvec"` 就回落串行）
**都不是**本问题的成因：失效用例 `test_fast_path_is_bit_exact_...` 逐条比对单个候选，
`BATCH_MIN_WIDTH = 8` 永远不触发，FG-4e 的批量核一次都没被调用。

## 8. 未测量 / 未归因（显式列出）

* **未做**任何加速比声明：本机 loadavg 长期在 20+（本刀各次运行时的实测值见
  JSON 的 `environment.loadavg`），远超 FG-4f 定的 "load > 12 不做加速比声明" 门槛。
* **未**在 `ecz-dev` 树上跑 Q1：该树没有 `ode_accel` 参数，无法做两档比较（它只有一档）。
* **未**逐例覆盖 RL 的全部 800 个差分用例：本刀取 2 例做定位，
  全量 800 例的 433/800 与 0.314 dB 是编排方给出的，本刀未独立复算该计数。
  已独立复算的是"换档不动"这个方向性结论。
* **未**验证 rebase 方案 1 是否真的让差分转绿（本刀只做归因，不改基线）。
