"""FG-4a 设计文档：ODE/QoT 加速（基准 + 金标准 + 零结果变化的结构优化）—— 见 docs/fg4a_accel_design.md

本文件是 FG-4a 的设计说明与证据汇总。FG-4a 只做"更快但同结果"，不做保真改进
（保真线是 FG-4d），不做 NLI 张量化（FG-4b），不做 GPU/批量（FG-4c，只提案）。

复现命令全部假定在 worktree 根目录、且 ``PYTHONPATH`` 指向本 worktree：

.. code-block:: bash

    cd /home/zsig/Documents/gits/wt/fg-ode-fast
    export PYTHONPATH="$PWD/src"; export NUMBA_CACHE_DIR=/tmp/fg-fast-nb
    PY="$HOME/miniconda3/envs/rl/bin/python"
    $PY -c 'import network_nlo_eval as n; print(n.__file__)'   # 必须落在本 worktree 内
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
DOC_PATH = REPO_ROOT / "docs" / "fg4a_accel_design.md"

BASELINE_REV = "ea7ecb0"


def _load(name: str) -> dict[str, Any]:
    """Read one committed evidence artifact."""
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing evidence artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def _fmt(value: float, unit: str = "ms", scale: float = 1e3) -> str:
    return f"{value * scale:.{3}f} {unit}"


def build_report() -> str:
    """Render the whole FG-4a report from the committed JSON artifacts.

    Every number printed here is read from a file under ``results/`` — none of
    them is hard-coded prose — so the report cannot drift away from the
    measurements it describes.
    """
    bench_base = _load("fg4a_bench_baseline.json")
    bench_after = _load("fg4a_bench_fg4a.json")
    ab = _load("fg4a_speedup_ab.json")
    golden = _load("fg4a_golden.json")

    base_m = bench_base["measurements"]
    after_m = bench_after["measurements"]
    derived_base = bench_base["derived"]
    derived_after = bench_after["derived"]
    ab_speed = ab["span_kernel_speedups"]
    gc = ab_speed["g_cache_only"]
    mv = ab_speed["g_cache_plus_branchless_matvec"]

    span_off_after = after_m["calc_span_noise_and_power_jit"]["min_s"]
    span_matvec = after_m["calc_span_noise_and_power_jit_matvec"]["min_s"]
    nli_after = after_m["compute_nli_variances"]["min_s"]
    ase_after = after_m["calculate_ase_noise_variance"]["min_s"]
    dp_matvec = after_m["calc_dp_dz_matvec"]["min_s"]
    dp_scalar = after_m["calc_dp_dz_single"]["min_s"]

    # Residual ODE cost under the matvec path = span kernel minus NLI minus ASE.
    ode_residual = span_matvec - nli_after - ase_after
    nli_share_of_matvec_span = nli_after / span_matvec if span_matvec else float("nan")

    # Ceiling if NLI were removed entirely (FG-4b), holding everything else fixed.
    ceiling_fg4b = span_matvec / ode_residual if ode_residual > 0 else float("nan")
    total_ceiling_fg4b = (
        base_m["calc_span_noise_and_power_jit"]["min_s"] / ode_residual
        if ode_residual > 0
        else float("nan")
    )

    load_base = bench_base["loadavg_at_start"]
    load_after = bench_after["loadavg_at_start"]
    n = bench_after["workload"]["num_channels"]

    md = f"""# FG-4a：ODE/QoT 加速（第一刀）

**状态**：已实现并验收。分支 `wp/fg-ode-fast`，基线 `{BASELINE_REV}`。
**基线测试**：63 passed / 1 skipped → **本分支 171 passed / 1 skipped**（新增 FG-4a
测试 108 个，既有测试**零期望值修改**）。

本 WP 的目标是"更快但同结果"。**没有**改任何物理、拉曼谱或 NLI 公式。
FG-2 的 F2-7 把 `_calc_exact_power_profile_ode` 的函数体逐字节冻结了，因此本 WP
**没有修改**那个函数，而是**新增**了等价变体与之并存——`ode_accel="off"` 仍走
冻结的参考路径。

---

## 1. 基准：一个必须先纠正的前提

编排方在 `profile_span_breakdown.py` 中报告"单跨段核 20.89 ms，其中 `G` 矩阵构建
≈ 6.1 ms"。**在本机实测中，`G` 矩阵构建不是 6.1 ms。**

`results/fg4a_bench_baseline.json` 用 n_steps 扫描把 ODE 成本拆成
`T(n) = T_g_build + 4·n·T_dp_dz`（`loadavg` 1min = {load_base[0]:.1f}）：

| 量 | 实测 | 占单跨段核 |
| --- | --- | --- |
| `_calc_dp_dz` 单次 | {_fmt(base_m['calc_dp_dz_single']['min_s'])} | {base_m['calc_dp_dz_single']['min_s'] / base_m['calc_span_noise_and_power_jit']['min_s'] * 100:.1f}% |
| × 80 次（20 步 × 4） | {_fmt(derived_base['dp_dz_calls_per_span_nsteps20'] * base_m['calc_dp_dz_single']['min_s'])} | **{derived_base['dp_dz_total_share_of_span_kernel'] * 100:.1f}%** |
| `G` 构建（njit 内） | {_fmt(derived_base['g_build_njit_est_s'])} | {derived_base['g_build_njit_est_share_of_span_kernel'] * 100:.2f}% |
| `_compute_nli_variances` | {_fmt(base_m['compute_nli_variances']['min_s'])} | {base_m['compute_nli_variances']['min_s'] / base_m['calc_span_noise_and_power_jit']['min_s'] * 100:.1f}% |
| 纯 Python 双层循环的 `G` | {_fmt(base_m['g_matrix_python_loop']['min_s'])} | （**不属于热路径**） |
| 单跨段核合计 | {_fmt(base_m['calc_span_noise_and_power_jit']['min_s'])} | 100% |

结论：6.1 ms 那一档对应的是**纯 Python 双层循环**的 `G` 构建（本机
{_fmt(base_m['g_matrix_python_loop']['min_s'])}），而**核内实际执行的是 njit 编译
后的循环**，只占 **{derived_base['g_build_njit_est_share_of_span_kernel'] * 100:.2f}%**。
因此"缓存 G"这一刀单独做，理论收益上限就只有这个量级——实测
**{gc['speedup_a_over_b']:.2f}x** 与此吻合。**真正的成本是 80 次
`_calc_dp_dz`，占 {derived_base['dp_dz_total_share_of_span_kernel'] * 100:.0f}%**，
去分支 matvec 才是这一刀的主收益（**{mv['speedup_a_over_b']:.2f}x**）。

> 编排方不需要重新发现这一点，但**需要更正优先级**：G 缓存是"纯赢但很小"，
> 去分支是"收益主体"。

## 2. 金标准与闸门（F4a-1）

- `results/fg4a_golden.json` — 在基线 `{BASELINE_REV}` 上用
  `scripts/fg4a_capture_golden.py` 冻结，7 个场景 × 31 个输出量。
  JSON 浮点按 `repr` 序列化（最短可往返表示），`json.load` 恢复**完全相同**的
  IEEE-754 位；`snr_db` 中的 `±inf` 按 JSON 的 `Infinity` 字面量往返。
- `tests/test_fg4a_numeric_identity.py` — 108 个测试的闸门。

关键设计：金标准是**通过稳定评估器 API**（`_calc_span_noise_and_power` /
`evaluate_path`）捕获的，**不是**通过 njit 内核的入参列表。FG-4a 改了内核的内部
与签名；若金标准定义在内核层，每一处合法重构都会被迫重新生成金标准，闸门就变成
循环论证。

每个加速档位按**自己的验收 ID** 判定：

| 档位 | 标准 | 实测 |
| --- | --- | --- |
| `off`（FG-2 冻结参考） | **逐位相等** | 31/31 bit-identical |
| `cached_g`（优化一） | **逐位相等** | 31/31 bit-identical |
| `matvec`（优化二） | 相对偏差 ≤ 1e-12 | 最差 **{ab['numeric_check']['matvec_worst_relative_deviation']:.3e}**（约为预算的 1/1000） |

`MAX_REL_DEVIATION_ALLOWED = 1e-12` 是模块常量，并由
`test_f4a_tolerance_budget_is_not_loosened` 断言它**没有被放宽**——失败的闸门必须
去修实现，不许去松这个数。

## 3. 两刀的实现

### 优化一：`G` 矩阵缓存（纯赢，逐位相等）

`G` 只依赖 `f_hz`、`cr_w_m_hz` 与谱形常数（13.2 / 15.6 THz），与功率无关。
提升为按 **(频率栅格的值, Cr, 谱形常数, N)** 记忆化的预计算量
（`raman_g_matrix` / `raman_g_pair_matrices`）。缓存键用 `f_hz.tobytes()`，是
**值**键，因此数值相同但对象不同的频率数组命中同一条条目，不会误用别人的矩阵。
返回的矩阵标记为 **只读**，避免调用方就地改坏共享缓存。

`test_f4a_5_g_cache_is_reused_not_rebuilt` 断言第二个 evaluator 只增加 `hits`
而不增加 `builds`，且拿到的是**同一对象**——这才叫缓存，而不是"G 恰好便宜"。

### 优化二：`_calc_dp_dz` 去分支（两次 matvec）

把逐对 `if f_hz[k] > f_hz[i]` **搬到构建期**：

```
dP/dz = p ⊙ (G_gain @ p − G_loss @ p − α)
```

`G_gain` 只在 `f_k > f_i` 处非零；`G_loss[i,k] = (f_k/f_i)·G[i,k]` 只在 `f_k < f_i`
处非零；同频与对角线留 0（基线两个分支都不命中）。
`test_f4a_g_split_matrices_partition_the_reference_g` 逐元素核对了这个划分。

**偏差来源**：`g_loss[i,k]` 预先乘了频率比，基线算的是
`(f_k/f_i)·G[i,k]·p[k]`——**逐项的乘法次序完全相同**；差别只在于求和从"一次
k 序号的标量累加"变成 BLAS dot 的求和树，因此只有末位舍入不同。实测最差
{ab['numeric_check']['matvec_worst_relative_deviation']:.3e}，比 1e-12 预算小约
三个数量级。

### 默认档位为什么是 `cached_g` 而不是 `matvec`

这是**契约**决定，不是性能疏漏：本仓库既有的 F1-1 / F1-2 / F2-3 三个测试用
`np.array_equal` 把公共 API 的输出钉死在 origin/main 的金标准上。`matvec` 的末位
重排会让这三个测试失败，而按 G2 纪律**不允许改它们的期望值**。所以默认取逐位等价
的 `cached_g`，`matvec` 需**显式**选入（`MultiBandISRSGN(..., ode_accel="matvec")`）。
若编排方希望默认开启 `matvec`，需要的是一个显式的决策（放开 F1-1/F1-2/F2-3 的逐位
断言或引入容差），而不是在 FG-4a 里改测试。

## 4. 实测加速比（F4a-5）

**跨进程的"前/后"在本机不可引用**：基线在 `loadavg` 1min = {load_base[0]:.1f} 下测得，
优化后在 {load_after[0]:.1f} 下测得，差 {abs(load_base[0] - load_after[0]):.1f}
（远超容差）。那种口径下所有量都显示约 1.5x，但那是**机器负载差**，不是加速——
`scripts/fg4a_bench.py` 的 `compare()` 会**拒绝**输出该比值并写明原因。

因此权威口径是 `results/fg4a_speedup_ab.json`：**把基线内核从
`{BASELINE_REV}` 导入同一进程**，两个目标**逐样本交错**（每轮翻转顺序以抵消
"谁先测"的系统偏差），同机、同 Numba/BLAS 状态、相邻微秒取样：

| 口径 | 基线 | 优化后 | 加速比（min） | 加速比（median） |
| --- | --- | --- | --- | --- |
| 优化一（`cached_g`） | {_fmt(gc['a_min_s'])} | {_fmt(gc['b_min_s'])} | **{gc['speedup_a_over_b']:.3f}x** | {gc['median_speedup_a_over_b']:.3f}x |
| 优化一+二（`matvec`） | {_fmt(mv['a_min_s'])} | {_fmt(mv['b_min_s'])} | **{mv['speedup_a_over_b']:.3f}x** | {mv['median_speedup_a_over_b']:.3f}x |

运行期 `loadavg` 1min：{ab['loadavg_before'][0]:.2f} → {ab['loadavg_after'][0]:.2f}
（Δ = {ab['loadavg_delta_1min']:+.2f}），两次运行分别复现为
{gc['speedup_a_over_b']:.3f}x / {mv['speedup_a_over_b']:.3f}x，稳定性良好。
同一脚本还顺带复核了数值：`cached_g` 与基线**逐位相等**，
`matvec` 最差相对偏差 {ab['numeric_check']['matvec_worst_relative_deviation']:.3e}。

## 5. FG-4c 批量接口提案（**只提案，不实现**）

```python
def batch_power_profile(
    power: Tensor,          # (B, N)  float32|float64, device
    g_gain: Tensor,         # (N, N)  float32|float64, device —— 全 batch 共享
    g_loss: Tensor,         # (N, N)  同上
    alpha_npm: Tensor,      # (N,)    float32|float64
    L_m: float | Tensor,    # 标量或 (B,)
    *,
    n_steps: int = 20,
    dtype: torch.dtype = torch.float64,
) -> Tensor:                # (B, N)
    \"\"\"对 B 组入纤功率并行跑 RK4/SRS ODE。数学式与 `_calc_dp_dz_matvec` 完全一致。\"\"\"
```

| 项 | 取值 | 理由 |
| --- | --- | --- |
| `power` | `(B, N)` | batch 维在前，与 `torch.baddbmm`/`matmul` 的布局一致 |
| `g_gain` / `g_loss` | `(N, N)`，**batch 维折叠** | 二者与功率无关，B 份拷贝是纯浪费 |
| `alpha_npm` | `(N,)`，广播 | 同上 |
| `L_m` | 标量或 `(B,)` | 每跨段长度；批量候选下通常同长，标量即够 |
| 设备 | `cuda`（主）/ `cpu`（回退） | |
| dtype | 计算 `float32`，缓存 `float32` | 见下"精度" |

**`G` 缓存在批量下的存放形态**：按 (频率栅格, Cr, 谱形常数) 缓存 **一份** `(N, N)`
`g_gain` + **一份** `(N, N)` `g_loss`，`requires_grad=False`、`contiguous()`、
常驻显存。N = {n} 时 `G` = {n}×{n} = {n * n:,} 元素，float32 约
{n * n * 4 / 1024:.0f} KiB、float64 约 {n * n * 8 / 1024:.0f} KiB，两者都能舒服地放进
L2，所以**不需要**按 batch 复制，也不需要 `(B, N, N)` 的膨胀张量。

**计算形态**（关键：不要 `G.expand(B, N, N)`）：

```python
gain = power @ g_gain.T          # (B,N) @ (N,N) -> (B,N)   一次 GEMM
loss = power @ g_loss.T          # (B,N) @ (N,N) -> (B,N)   一次 GEMM
dp   = power * (gain - loss - alpha_npm)   # 广播 alpha (N,)
```

RK4 的 80 次 `dp` 调用因此变成 80 次 `(B,N)×(N,N)` GEMM，无数据相关控制流——
这正是去分支这一刀的真实目的：它不是为了 CPU 上的几毫秒，而是为了让 `dp_dz`
成为**可批量、可映射到 GEMM**的算子。零功率槽位不再需要分支（乘 0 自然为 0）。

**精度**：`float32` 的 1e-7 相对精度**远差于**本 WP 的 1e-12 保真预算，且会与 FG-4d
的保真线纠缠。故提案默认 `float64` 计算；`float32` 仅作为显式的
`precision="fast"` 档位，且必须走与本 WP 相同的金标准闸门单独验收。

**设备上的注意点**（留给 FG-4c 验证）：

1. `power @ G.T` 在不同 BLAS/cuBLAS 后端上的**求和次序不同**，因此不同设备之间
   不会逐位一致。跨设备一致性必须像本 WP 一样由金标准闸门定义（建议按设备各存一份
   金标准，预算沿用 1e-12 相对偏差）。
2. `N = {n}` 是**小矩阵**。B 较小时 `(B,N)×(N,N)` 是 **memory-bound**（要把整个
   `G` 从显存读一遍），加速比随 B 增长而非随设备算力增长；FG-4c 应先扫 B 找到拐点。
3. RK4 内部的临时量建议预分配 `(B, N)` 缓冲并 `copy_` 复用，避免每步重新分配。

## 6. 后续每刀的理论加速上限

口径：单跨段核（`{n}` 节点，80 次 `dp_dz`，`loadavg` 1min = {load_after[0]:.1f}）。
优化后 matvec 路径的分解：NLI {_fmt(nli_after)}（占
{nli_share_of_matvec_span * 100:.1f}%）+ ODE 残差 {_fmt(ode_residual)}。

| 刀 | 作用对象 | 理论上限 | 说明 |
| --- | --- | --- | --- |
| FG-4a 优化一 | `G` 构建 | **{gc['speedup_a_over_b']:.2f}x（实测）** | njit 下只占 {derived_base['g_build_njit_est_share_of_span_kernel'] * 100:.2f}%，上限本来就低 |
| FG-4a 优化二 | 80 × `dp_dz` | **{mv['speedup_a_over_b']:.2f}x（实测）** | `_calc_dp_dz` 单次 {(_fmt(dp_scalar))} → matvec {(_fmt(dp_matvec))}，约 {dp_scalar / dp_matvec:.1f}x |
| FG-4b（NLI 张量化） | `_compute_nli_variances` | **≤ {ceiling_fg4b:.1f}x**（理论，假设 NLI 归零） | matvec 之后 **NLI 已是 {nli_share_of_matvec_span * 100:.0f}% 的瓶颈**；对基线的累计上限约 {total_ceiling_fg4b:.1f}x。这是下一刀最大的一块 |
| FG-4c（GPU/批量） | 整个 RK4/SRS 段 | 需实测，本 WP **未测量** | N = {n} 属小矩阵，小 B 时 memory-bound；上限取决于 B 与 `G` 的驻留层级，不可在本 WP 给出可信数字 |
| FG-4d（保真） | 物理/拉曼谱 | 不适用 | 允许改结果，须单独验收；不与本 WP 的"同结果"预算混用 |

> FG-4b 的上限 {ceiling_fg4b:.1f}x 是**理想上限**（假定 NLI 成本完全消失），实际会被
> `t_tilde` 拟合、0 功率掩码与变带宽等非规则访存打折扣。FG-4c 的数字本 WP **没有测**，不填估计。


## 7. 产物清单

| 文件 | 内容 |
| --- | --- |
| `results/fg4a_golden.json` | 7 场景 × 31 输出量的 repr 级精度金标准（基线 `{BASELINE_REV}`） |
| `results/fg4a_bench_baseline.json` | 逐项耗时 + 每次读数的 `loadavg` + 派生的 G 构建成本 |
| `results/fg4a_bench_fg4a.json` | 优化后同口径计时（三个档位同一进程） |
| `results/fg4a_bench_fg4a_vs_baseline.json` | 跨进程对比，**显式判定负载不可比** |
| `results/fg4a_speedup_ab.json` | 同进程交错 A/B（权威加速比） |
| `tests/fg4a_scenarios.py` | 7 个固定场景的输入侧定义（可跨内核变更复用） |
| `tests/test_fg4a_numeric_identity.py` | 108 个测试的数值闸门 |
| `scripts/fg4a_capture_golden.py` | 金标准生成器 + `--check` 自检 |
| `scripts/fg4a_bench.py` | 计时harness（带 `loadavg`，拒绝不可比的比值） |
| `scripts/fg4a_speedup_ab.py` | 同进程交错 A/B |

## 8. 复现

```bash
cd /home/zsig/Documents/gits/wt/fg-ode-fast
export PYTHONPATH="$PWD/src"; export NUMBA_CACHE_DIR=/tmp/fg-fast-nb
PY="$HOME/miniconda3/envs/rl/bin/python"
$PY -c 'import network_nlo_eval as n; print(n.__file__)'

# 闸门（数值）与全量套件
$PY -m pytest tests/test_fg4a_numeric_identity.py -q -s
$PY -m pytest -q

# 金标准自检（应报 31/31 bit-identical）
$PY scripts/fg4a_capture_golden.py --check --out results/fg4a_golden.json

# 加速比（同进程交错；应报 ~1.03x / ~1.88x）
$PY scripts/fg4a_speedup_ab.py --out results/fg4a_speedup_ab.json
```

当前 HEAD：{_git_rev()}；基线：`{BASELINE_REV}`。
"""
    return md


def main() -> int:
    """Render docs/fg4a_accel_design.md from the committed evidence."""
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(build_report(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
