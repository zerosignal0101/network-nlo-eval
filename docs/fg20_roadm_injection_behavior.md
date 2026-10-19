# FG-20 §2：两处 ROADM 施加点的**行为**回归测试（FG-18 未闭合项 5）

- worktree：`wp/fg-guard` @ `/home/zsig/Documents/gits/wt/fg-guard`，从 `9115da2` 切
- 新增：`tests/test_fg20_roadm_injection_behavior.py`（**45** 条）、`scripts/fg20_mutation_test.py`
- **未改** `models/isrs_gn.py`、未改 `numba_kernels.py`、未改 ODE 内核、未改任何物理
- **未推 `origin`**

---

## 1. 为什么源码守卫不够

FG-18 的 `tests/test_fg18_lumped_assembly_tree_guard.py` 是**源码级 AST 守卫**。
它证明的是"`insertion_loss_db + filtering_penalty_db` 这个**字面形状**不再出现"，
**证明不了"口径正确"**。

这正是 FG-13 出事的形态：当时源码看着没问题（守卫是绿的），漏的是**语义**——
booster 注入没有发生。守卫能挡住"退回旧写法"，挡不住"用新写法干错的事"。

⇒ 本文件补**行为**层。两把尺子都要有，且要能互相印证（§4 的变异读数）。

## 2. 期望值从哪来（**不许**把"当前实现跑出来的数"当期望）

期望值**一律**由权威折叠 `rwa/qot_checker.py::fold_roadm_terms`（`:115-151`）
独立算出，形状是：

```python
terms  = fold_roadm_terms(cfg, is_last_hop=...)   # 权威折叠
baseline = <用 roadm_config=None 跑同一入口>       # ODE 内核读数，与 ROADM 无关
expected = _expected(baseline, terms, ν, Rs)       # 按 fold 文档语义重算
```

`_expected()` 逐条抄 `fold_roadm_terms` 的文档语义与
`accumulate_service_snr`（`rwa/qot_checker.py:196-205`）的实际循环体：

* `equalize` 为真 → 信号**不衰减**，噪声注入 `booster_base·ν·Rs·(loss−1)`；
* `equalize` 为假 → 四个量**全部**除以 `loss`。

⚠️ **非循环性**：`_expected` 与被测入口是**两个独立函数**，前者只读 `fold_roadm_terms`
返回的字段，**不读**被测入口的任何输出。§4 的变异读数（删掉 booster 注入 ⇒ 变红）
是这条非循环性的实证。

`baseline` 用 `roadm_config=None` 取，是合法的公共起点：ODE 内核本身与 ROADM 无关。

## 3. 覆盖矩阵（FG-18 §3 对照表的 6 个参数逐个拧动）

`CONFIG_CASES` 九个配置，每个**只拧动一个参数**，失败时可直接归因到那个参数：

| 参数 | 用例 | 覆盖的命题 |
|---|---|---|
| `insertion_loss_db` | `insertion_loss_3db` | 进 `loss`；`path_loss_db` 逐位 = `insertion + express` |
| `filtering_penalty_db` | `filtering_penalty_zero` / `filtering_penalty_does_not_touch_signal_power` | **不**进光功率（FG-18 §3 表里的量纲错误） |
| `express_attenuation_db` | `express_attenuation_2db` / `express_attenuation_enters_loss` | **express 端口**语义 |
| `add_drop_loss_db` | `drop_loss_5db_express_zero` / `test_add_drop_loss_does_not_enter_express_reading` | 逐跨段层**看不到**它（末跳不可达） |
| `equalize_output_power` | `equalize_on/off` 六例 | 真时信号**不再**被 `loss` 衰减 |
| `booster_noise_figure_db` | `booster_nf_7db` / `booster_nf_zero_equalize_on` | 按 `nf_lin` 线性进入 |

另外覆盖：端口 vs 路径、末跳 vs 非末跳（在本入口**不可做**，故写成取证性用例）、
串行 `(N,)` vs batch `(B, N)` 逐候选同口径、`None` 时两个入口都不介入。

## 4. 门 3：变异读数（`results/fg20_mutation_test.json`）

五个方向在**临时树副本**上注入，不碰工作树。baseline 先跑且**必须绿**。

| # | 变异 | 行为测试 | FG-18 AST 守卫 | 语义类别 |
|---|---|---|---|---|
| — | 无变异（baseline） | 🟢 **GREEN** | 🟢 **GREEN** | — |
| **M1** | 删掉**串行**施加点 booster 注入 | 🔴 **RED**（19 failed） | 🟢 GREEN | booster 漏注入（FG-13 同型） |
| **M2** | 删掉 **batch** 施加点 booster 注入 | 🔴 **RED**（14 failed） | 🟢 GREEN | booster 漏注入 |
| **M3** | `equalize=True` 时**照样**衰减信号 | 🔴 **RED**（16 failed） | 🟢 GREEN | 信号口径 |
| **M4** | `filtering_penalty` 并进光功率 | 🔴 **RED**（21 failed） | 🟢 GREEN | 量纲错误（FG 树旧口径） |
| **M5** | 退回合并式 `insertion+filtering` | 🔴 **RED**（16 failed） | 🔴 **RED** | FG-17 盲区同款 |

**这张表就是本刀存在的理由**：AST 守卫只抓得住 **M5**，
M1–M4 四个**语义**回归它一律放过；行为测试 **5/5 全抓**。

### 探针自检（本脚本自己翻过一次车，如实记录）

初版临时树只复制 `src/ tests/ results/`，**漏了 `references/`**。
拉曼谱表 CSV 按**仓库根相对路径**加载（`physics/raman_spectrum.py`），
于是 `MultiBandISRSGN.__init__` 抛 `FileNotFoundError`，
45 条用例变成 **44 errors + 1 passed**，而 pytest 对 error **也给 rc=1**——
初版把它读成"RED = 抓到了"。

**探针失效被误读成判决**，与 FG-18 那两次同源。修法两条：

1. 临时树补 `references/`；
2. `_run()` 把**任何出现 ERROR 的输出**显式判为 `INVALID`，不计入判决。

⇒ **教训：`rc` 相同不代表判决相同。** 现在 `all_valid: true`。

## 5. 两条如实记录的"现状"用例（非规范断言）

1. `test_filtering_penalty_is_not_applied_at_the_per_span_layer`：
   逐跨段层**不施加** `filtering_penalty`，而编排层 `evaluate_path:781-783` 施加。
   理由：它乘的是**跨跳累积后**的噪声量，逐跨段层只返回**本跨段**噪声，
   那里无处可施。**本刀不判这是对是错**，只把它钉住——将来"顺手补上"会变红，
   从而被迫显式讨论。
2. `test_is_last_hop_is_hardwired_false_at_the_per_span_layer`：
   固定 express 是契约，不是"已经处理了端口"。取证见
   [`fg20_none_adjudication.md`](fg20_none_adjudication.md) §5。

## 6. 复算命令

```bash
RL=/home/zsig/miniconda3/envs/rl/bin/python
export PYTHONPATH=/home/zsig/Documents/gits/wt/fg-guard/src

# 0) 证明 import 落在这棵 worktree 内
$RL -c "import network_nlo_eval; print(network_nlo_eval.__file__)"

# 1) 45 条行为用例（baseline，必须全绿）
$RL -m pytest tests/test_fg20_roadm_injection_behavior.py -q
#   -> 45 passed

# 2) 五向变异读数（baseline 先跑且必须绿）
$RL scripts/fg20_mutation_test.py
#   -> baseline GREEN/GREEN；M1..M5 行为测试全 RED；AST 守卫只 M5 RED
#   -> results/fg20_mutation_test.json

# 3) 与既有守卫一起跑，证明没有放宽它们
$RL -m pytest tests/test_fg18_lumped_assembly_tree_guard.py \
              tests/test_fu4_assembly_union.py \
              tests/test_fg20_roadm_injection_behavior.py -q
#   -> 75 passed（FG-18 守卫 17 + FU-4 13 + FG-20 行为 45）
```

## 7. 未闭合项

1. **两处施加点在本仓不可达**：逐跨段入口的 `roadm_config` 恒为 `None`
   （三处调用点都不传该关键字，见 `fg20_none_adjudication.md` §5 (A)）。
   ⇒ 本文件是**外部调用方**的行为契约，**不是**现有场景的读数。
   本仓现有读数由 `9115da2` 基线与门 1 保证（见 `fg20_gates.md`）。
2. **`filtering_penalty_db` 在逐跨段层的归属未裁定**（§5.1）。
   本刀如实记录现状，未改语义。**交回编排方**。
3. **batch 入口只测了 `B=1`**：FG-18 记 batch 相对串行的预算是 1e-12，
   本文件只验"逐候选等于串行"，未单独验 `B>1` 的批内交互。若需要，应另立一刀。
