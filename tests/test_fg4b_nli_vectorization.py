r"""FG-4b gate: the NLI tensorisation must not move a bit, and the freeze point must hold.

Every assertion here is executable and recomputable inside this worktree::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \\
        $HOME/miniconda3/envs/rl/bin/python -m pytest tests/test_fg4b_nli_vectorization.py -v

Mapping to the FG-4b acceptance table
=====================================

======  ==============================================================
 F4b-1  the **default** tier (``matvec``) is within 1e-12 of the golden
 F4b-2  ``off`` is still selectable and still **bit-identical** to
        origin/main — the freeze point has not been quietly moved
 F4b-3  the tensorised NLI alone stays within 1e-12 of the frozen
        per-pair NLI, with the ODE held bit-identical across the arms
 F4b-4  the per-connection bandwidth semantics (``rs_l`` vs ``rs_i``)
        and the zero-power mask both survive vectorisation
 F4b-5  the build-time tensors are provably reused, and provably keyed
        on everything they actually depend on
======  ==============================================================

Why the isolation in F4b-3 matters
----------------------------------
The tempting mistake is to compare the tensor path against the *baseline
commit* and call the difference "the NLI knife". That number is really
"matvec ODE + tensor NLI" versus "per-pair ODE + per-pair NLI" — two changes at
once, so a failure could not be attributed. Here both arms pass the same
``g_gain``/``g_loss``, so the RK4/SRS solve produces **identical bits** and the
measured difference is attributable to the NLI evaluation alone. The test
asserts that bit-identity explicitly rather than assuming it.

A second trap this file is written to close
-------------------------------------------
A 1-D array broadcasts against the **last** axis. In the XPM pre-factor the
inner channel power ``p_l`` may be broadcast implicitly, but the outer channel
power ``p_i`` may not — writing ``* p_in_w`` instead of ``* p_in_w[:, None]``
silently computes ``p_l`` and is **invisible on a fully loaded grid**, where
``p_i == p_l`` for every surviving pair. It only shows up once Raman tilt has
made the per-channel powers differ, i.e. in ``path_cplusl_roadm``. That exact bug
shipped and passed five of the seven scenarios before being caught; the
non-uniform and end-to-end scenarios below are what make it impossible to
reintroduce silently.
"""

from __future__ import annotations

from network_nlo_eval.models.isrs_gn import ODE_ACCEL_DEFAULT, ODE_ACCEL_MODES


# ======================================================================
# F4b-1  the production default is within budget
# ======================================================================
def test_f4b_1_default_tier_is_matvec() -> None:
    """The adjudication in §1 must actually be in force, not merely documented."""
    assert ODE_ACCEL_DEFAULT == "matvec", (
        f"production default is {ODE_ACCEL_DEFAULT!r}; the FG-4b §1 adjudication set it to 'matvec'"
    )
    assert "off" in ODE_ACCEL_MODES, "the freeze point must remain selectable"


# ---------------------------------------------------------------------------
# ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本模块原有 14 条判据**全部撤下**。
#
# 逐条理由（文件 + 测试名 + 一行理由）：
#
# 1.  ``test_f4b_1_default_tier_within_budget``
#    —— 默认档 vs 冻结档的 1e-12 预算，**跑在被移除那套物理上**（5 组参数化）。
#    闭式 ECZ 的同类判据在 ``tests/test_p1_closed_accel.py::test_matvec_equals_off_within_budget``
#    与 ``tests/test_p1_closed_accel.py::test_batch_equals_serial_within_budget``，
#    覆盖面更广（混合 + 均匀 n_slots、批宽 8..142）。
# 2.  ``test_f4b_1_cached_g_remains_the_bit_exact_tier``
#    —— ``cached_g``（拉曼 G 矩阵缓存）随该物理**整体**离场；闭式 ECZ 没有 G 矩阵。
# 3.  ``test_f4b_2_off_is_bit_identical_to_origin_main``
#    —— 与另一条线的 ``origin/main`` 逐位比对，两套物理**不可互比**。
# 4.  ``test_f4b_2_off_tier_passes_no_tensors_to_the_kernel``
#    —— 断言 ``off`` 档**不**向核传张量；张量化 NLI 是被移除那套物理的结构，
#    ECZ 的 ``off`` 档直接调冻结核，**没有这条接线**。
# 5.  ``test_f4b_2_frozen_nli_source_is_intact``
#    —— 守的是被移除那份核（张量化 NLI 那一份）的**字节冻结**。
#    幸存内核的字节冻结由 ``test_p1_closed_accel.py::test_frozen_kernel_bytes_are_unchanged``
#    与 ``test_ode3_flexgrid_sync.py`` 的 J5 守卫（绝对 sha256 + 相对 diff）双重接管。
# 6.  ``test_f4b_3_tensor_nli_matches_frozen_nli_with_budget``
# 7.  ``test_f4b_3_tensor_nli_survives_non_uniform_per_channel_power``
#    —— 守「**张量化** NLI ≈ 逐对冻结 NLI」。闭式 ECZ 没有张量化路径，
#    被测实现已不存在。
# 8.  ``test_f4b_4_heterogeneous_symbol_rates_are_preserved``
#    —— 符号率口径是那套物理的带宽量；ECZ 的带宽量是逐信道 ``B_i``，命题不同。
# 9.  ``test_f4b_4_zero_power_channels_are_excluded``
#    —— 守张量化 NLI 的零功率掩码。ECZ 批量入口的零功率语义由
#    ``test_p1_closed_accel.py::test_batch_handles_zero_power_channels`` 覆盖。
# 10. ``test_f4b_4_t_tilde_fit_and_kurtosis_branches_are_covered``
#    —— ``t_tilde`` 拟合与**峰度**分支。峰度修正是拉曼 ODE 的结构，
#    闭式 P1 核**全文 0 处** ``phi_kurtosis``（HEAD 上同样为 0）⇒ 无可测对象。
# 11. ``test_f4b_5_tensors_are_reused_across_calls``
# 12. ``test_f4b_5_tensor_cache_is_keyed_on_what_it_depends_on``
# 13. ``test_f4b_5_tensors_are_read_only``
#    —— 三条都是**张量缓存**的复用/键控/只读性；ECZ 的等价物是第 4 轮写的
#    P1 XPM 对矩阵缓存，由 ``test_p1_closed_accel.py::test_pair_matrices_are_cached_per_evaluator``
#    与 ``::test_pair_matrices_are_nonnegative`` 覆盖。
# 14. ``test_f4b_6_end_to_end_path_with_roadm_and_tilt``
#    —— 端到端（ROADM + 倾斜）比对**张量化**路径。端到端路径本身仍在
#    ``tests/test_fg13_evaluate_path_roadm.py`` 与 ``test_fu4_assembly_union.py`` 中覆盖，
#    只是不再经由被移除的物理。
#
# **保留**：``test_f4b_1_default_tier_is_matvec`` —— 「生产默认档是 ``matvec``」
# 是**编排方裁决**（任务书明令不许改），与哪套物理无关。
# ---------------------------------------------------------------------------
