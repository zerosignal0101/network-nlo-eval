# FG-20 §1：`physics.evaluate_path` 的 `None` 崩溃裁决落地

- **worktree（物理）**：`wp/fg-guard` @ `/home/zsig/Documents/gits/wt/fg-guard`，从 `9115da2` 切
- **worktree（RL，只读参考）**：`/home/zsig/Documents/gits/wt/fg-rt` @ `9ca3ec2`（**未改 RL 侧任何文件**）
- **解释器**：物理仓全量 + RL 侧 = `rl`；波形级 = `nlo`
- **未推 `origin`**；`origin/main` 保持 `87850a1`

> 每条命令都显式带 `PYTHONPATH=<repo>/src` 并打印实际 import 路径。理由：本仓是
> editable 安装，`.pth` 指向**主检出**，与 cwd 无关；不设这一行，跑的是**另一棵树**。

---

## 1. 裁决

**列表里的 `None` 成员视为"无 ROADM"并跳过**，与字段级 `roadm_configs or []`
的兜底**同义**。

落地位置：`src/network_nlo_eval/physics/qot_checker.py` 新增模块级
`_active_channel_mask(launch_power_profile_w, roadm_configs)`，
`evaluate_path` 改为调它，不再自己抄一遍阻断逻辑。

| 输入 | 掩码读数（`_LAUNCH = [1e-3, 0, 2e-3]`） |
|---|---|
| `[cfg(blocked=[0]), None]` | `[F, F, T]` |
| `[cfg(blocked=[0])]` | `[F, F, T]` ← **逐位相同** |
| `[None, None]` | `[T, F, T]` |
| `[]` | `[T, F, T]` ← **逐位相同** |
| `None`（字段级） | `[T, F, T]` ← **逐位相同** |

## 2. ⚠️ 残余风险（照写，不淡化）

**这样处理会掩盖调用方的畸形输入。**

`[None]` 这样的列表本应被当成"这跳的 ROADM 声明是漏填还是确实没有？"报出来；
现在被**静默**当成"确实没有 ROADM"。本函数把"**不知道**有没有 ROADM"和
"**确定没有** ROADM"折叠成同一件事，而这两者在 SNR 语义上并不等价——
少一段 WSS 滤波与一段插入损耗，功率仍算得出来，**但依据被丢掉了**，
且没有任何日志或告警提示发生过这次折叠。

为什么仍选"跳过"而不是"报错"：生产路径
`NetworkTopology.build_optical_path`（`network/topology.py:173`）只
`append(self.get_roadm_config(v_idx))`，而 `get_roadm_config` 返回类型是
`ROADMConfig`，**结构上不可能产出 `None`**。含 `None` 的列表只可能来自仓外的
畸形构造。为它把报错抛进 `evaluate_path` 的正常控制流，会把一个潜伏的输入问题
变成生产故障。

若将来要区分"漏填"与"确实没有"，正确做法是**在调用方入口校验**并给出明确的
字段级错误，而不是在这里猜——**本函数不做猜测**。

该风险已写进 `_active_channel_mask` 的 docstring，并用
`test_fg20_none_roadm_adjudication.py::test_none_member_is_masked_not_crashed`
钉成显式契约（含"这是有意的"注释锚点：将来改裁为报错时，那条**应当**红）。

## 3. 三处活调用方逐个说明

| # | 调用方 | 传入的 `roadm_configs` | 是否受本裁决影响 | 依据 |
|---|---|---|---|---|
| 1 | `physics/comparison.py:90`（`run_waveform=True`） | 来自 `:79` 的 `topology.build_optical_path(path)` | **否** | `build_optical_path` 只 append `ROADMConfig`；`test_build_optical_path_never_emits_a_none_member` 断言 `all(isinstance(cfg, ROADMConfig))` |
| 2 | `qot_checker.py:331`（`evaluate_topology_path`） | `topology.build_optical_path(path)` | **否** | 同上 |
| 3 | `qot_checker.py:347`（`evaluate_channel_evm_and_ber`） | **不传**该参数 ⇒ 字段级 `None` | **否** | `MultiSpanOpticalPath.roadm_configs` 默认 `None`（`isrs_gn.py:117`），由 `or []` 兜住，FG-18 取证里它本就不崩 |

⇒ **三处活调用方都不受本裁决影响**：没有一处会传含 `None` 成员的列表。
本裁决对它们**逐位惰性**；它只是把一个**潜伏**的崩溃变成不再崩溃。

## 4. 为什么原先会崩（FG-18 §6 取证的复核）

原实现直接解引用：

```python
for roadm in optical_path.roadm_configs or []:
    active[[index for index in roadm.blocked_channels if index < len(active)]] = False
```

`or []` 只兜住**字段级** `None`；列表**成员**的 `None` 照样进循环体，
`roadm.blocked_channels` 抛
`AttributeError: 'NoneType' object has no attribute 'blocked_channels'`。

另注：`propagate_path` 早就用 `if roadm is None: continue` 处理同样的成员，
所以本裁决的口径与它**一致**——FG-18 记录的崩溃只发生在活动掩码这一处。

## 5. §3 顺带取证：逐跨段入口的 `is_last_hop`（**只报不改**）

问题：逐跨段层拿不到 hop 序号，FG-18 收敛后**固定** `is_last_hop=False`（express）。
这个选择在本仓可达面上**会不会**给出与权威口径不同的结果？

`scripts/fg20_is_last_hop_forensics.py` → `results/fg20_is_last_hop_forensics.json`

### (A) 结构性读数：现在有没有风险

AST 扫描全部 `src/`，逐跨段入口的**全部**调用点：

| 调用点 | 形参实参 | keywords |
|---|---|---|
| `models/isrs_gn.py:750`（`evaluate_path`） | 3 位置 | `['n_effective_spans']` |
| `rwa/link_noise.py:72` | 4 位置 | `['n_effective_spans']` |
| `rwa/link_noise.py:85` | 3 位置 | `['n_effective_spans']` |

**三处都不传 `roadm_config` 关键字** ⇒ 逐跨段入口拿到的 `roadm_config` 恒为
默认值 `None`。加上 FG-18 §1.2 的运行时挂钩读数（790 次调用 / 非 `None` **0** 次），
两条独立证据一致。

⇒ **现在没有差异**：本仓不存在"逐跨段入口带着 ROADM 且自己该是末跳"的调用面。

### (B) 构造性读数：若接上末跳，差多少

这是上界读数，**不是**"现有场景会不会动"。差值机制经核验（`delta_rule_holds: true`）：

```
loss_db_drop - loss_db_express == add_drop_loss_db - express_attenuation_db
```

即**只**来自 `add_drop_loss_db`（默认 1.0 dB），与 `express_attenuation_db` 无关。

| config | loss_db (express → drop) | SNR Δ 若按末跳 drop |
|---|---|---|
| `defaults` | 5.000 → 6.000 | **−0.074012 dB** |
| `express_attenuation_2db` | 7.000 → 6.000 | +0.091417 dB |
| `add_drop_loss_5db` | 5.000 → 10.000 | **−0.582497 dB** |
| `equalize_off` | 5.000 → 6.000 | **+0.000000 dB** |
| `equalize_off_add_drop_5db` | 5.000 → 10.000 | **+0.000000 dB** |

⚠️ **`equalize=False` 两行值得单独说**：SNR 差**恰好为 0**。
因为 `equalize=False` 时信号与**全部三个**噪声分量同时除以 `loss`，
`10·log10(P/loss / (N/loss))` 里 `loss` 整除掉了 —— **SNR 对端口损耗不敏感**。
⇒ 端口选择的影响**只在 `equalize_output_power=True` 时**才体现为 SNR 差。
默认 `equalize_output_power=True`，所以默认情形下差 0.074 dB（非零，不可忽略）。

### (C) 要支持真正的末跳 drop，需要什么

**本刀不改签名、不加开关**（brief 要求），只给结论：

需要**显式的 hop 位置信息**，三条路各有权衡，交编排方裁：

1. **加显式位置参数**（如 `hop_index` + `num_hops`，或在 `roadm_config` 之外
   传 `is_last_hop`）。最直接，但逐跨段入口**结构上拿不到**它——
   `MultiSpanOpticalPath` 有 `node_path`，可由拓扑反推"这是不是最后一跳"，
   但逐跨段 API 本身不知道拓扑。
2. **由调用方（`evaluate_path`）自己施加**。它**已经**在做这件事
   （`isrs_gn.py:768`），且**不**给逐跨段入口传 `roadm_config`
   （`isrs_gn.py:750` 的调用只带 `n_effective_spans`）。
   ⇒ 其实"正确口径的末跳 drop **已经在编排层生效**"，逐跨段入口的 express
   取值只在**外部调用方**传 `roadm_config` 时才有影响。
3. **维持现状 + 文档化**（本刀的现状）。代价是外部调用方若传末跳 ROADM，
   会拿到 express 口径，差 0.074 dB（默认配置）。

⇒ **本仓可达面无差异**（(A) 读数），**外部调用方差 0.074 dB**（(B) 读数）。
建议在 release note 里点名这一条。

## 6. 复算命令

```bash
RL=/home/zsig/miniconda3/envs/rl/bin/python
export PYTHONPATH=/home/zsig/Documents/gits/wt/fg-guard/src

# 0) 证明 import 落在这棵 worktree 内
$RL -c "import network_nlo_eval; print(network_nlo_eval.__file__)"

# 1) §1 裁决的 11 条用例
$RL -m pytest tests/test_fg20_none_roadm_adjudication.py -q

# 2) §3 取证（(A) 结构 + (B) 构造）
$RL scripts/fg20_is_last_hop_forensics.py
```

## 7. 未闭合项

1. **波形侧端到端读数未取到**（`nlo` 环境）。`physics_nlo_eval` 与 `cupy`
   在 `nlo` 里都可用，但**本机无 CUDA toolkit**（有 P100 驱动与 GPU，
   无 `/usr/local/cuda`、`nvcc`、`libcudart`），`cupy` 无法编译 kernel：
   `RuntimeError: Failed to auto-detect CUDA root directory`。
   ⇒ `test_evaluate_path_tolerates_none_member_end_to_end` 在 `rl` 下
   `importorskip` 跳过；**在 `nlo` 下同样跑不通**，但这是**环境限制**而非本刀
   缺陷——既有 `test_topology_physical_integration.py::test_waveform_route_smoke`
   在 `nlo` 下**报同一个错**（实测，已复核）。该波形读数标为**未测量**。
2. **`add_drop_loss_db` 在逐跨段层结构性不可见**（`test_add_drop_loss_does_not_enter_express_reading`）。
   这是 §3 (A) 结论的直接推论，不是不一致。
3. **`filtering_penalty_db` 在逐跨段层不施加**，而编排层 `evaluate_path:781-783`
   施加。这是**如实记录现状**（`test_filtering_penalty_is_not_applied_at_the_per_span_layer`），
   不是本刀判定的"正确"或"错误"——理由见该用例 docstring。
