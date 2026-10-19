r"""FG-20 §2：两处 ROADM 施加点的**行为**回归测试（FG-18 未闭合项 5）。

为什么源码守卫不够
------------------
FG-18 的 ``tests/test_fg18_lumped_assembly_tree_guard.py`` 是**源码级 AST 守卫**。
它证明的是"``insertion_loss_db + filtering_penalty_db```` 这个字面形状不再出现"，
**证明不了"口径正确"**。这正是 FG-13 出事的形态：当时源码看着没问题（守卫是绿的），
漏的是**语义**——``booster`` 注入没有发生。守卫能挡住"退回旧写法"，挡不住
"用新写法干错的事"。因此本文件补**行为**层。

期望值从哪来（不许把"当前实现跑出来的数"当期望）
------------------------------------------------
本文件的期望值**一律**由权威折叠 ``rwa/qot_checker.py::fold_roadm_terms`` 独立算出，
形状是::

    terms = fold_roadm_terms(cfg, is_last_hop=...)     # 权威折叠
    expected = baseline_without_roadm 的算术副本        # 由 fold 的字段重算

其中 ``baseline_without_roadm`` 是用 ``roadm_config=None`` 跑同一入口得到的**内核**
读数（ODE 内核本身与 ROADM 无关，故这是合法的公共起点）。然后按 ``fold_roadm_terms``
的文档语义（``rwa/qot_checker.py:115-151``）重算：

* ``equalize`` 为真 → 信号**不衰减**，噪声注入 ``booster_base·ν·Rs·(loss−1)``；
* ``equalize`` 为假 → 四个量**全部**除以 ``loss``。

⚠️ 若期望值直接取自被测函数的输出，本文件就变成"把 bug 钉成规范"的同谋
（FG-13 的教训）。故期望值与被测实现**分处两个函数**：`_expected_*`` 用 fold 的
字段重算，被测入口自己跑一遍，两者比。

覆盖矩阵
--------
按 FG-18 §3 对照表的 6 个配置参数参数化：

* ``equalize_output_power`` True/False —— 真时信号**不再**被 ``loss`` 衰减
* ``booster_noise_figure_db`` —— 生效且按 ``nf_lin`` 线性进入
* ``express_attenuation_db`` vs ``add_drop_loss_db`` —— **express** 语义
* ``insertion_loss_db`` / ``filtering_penalty_db`` —— 位置与作用域
* 端口（``is_last_hop`` → drop/express）与末跳 vs 非末跳
* 串行 ``(N,)`` 与 batch ``(B, N)`` 两条施加点逐位同口径
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import FiberSpanConfig, ROADMConfig
from network_nlo_eval.rwa.qot_checker import _H_PLANCK, fold_roadm_terms

#: FG-4a/FG-4b 的逐候选相对偏差预算，沿用不新增常量。
_REL_TOL = 1e-12

_SPAN_M = 100e3
_N_CH = 3


def _evaluator() -> MultiBandISRSGN:
    """一个最小、确定性的评估器（默认 ``ode_accel="matvec"``）。"""
    grid = SpectrumGrid(_N_CH, 50e9, 193.1e12)
    return MultiBandISRSGN(grid, FiberSpanConfig(length_km=100.0), symbol_rate_hz=32e9,
        )


def _run_span(ev: MultiBandISRSGN, power: np.ndarray, roadm: ROADMConfig | None):
    """**逐跨段**施加点（串行，形状 ``(N,)``）。

    ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：闭式 ECZ 下，这一层只做
    **损耗等分**（``path_loss_db()``）——booster 均衡、滤波惩罚、阻断**都不在这里**
    （它们乘的是**跨跳累积**后的量，逐跨段层无处可施）。完整的 ROADM 装配语义在
    **路径层** :meth:`MultiBandISRSGN.evaluate_path`（``_apply_roadm_p1``）。
    这与 ``flexgrid-dev @ cb24704`` 逐字一致，是本树 ROADM 行为的**权威定义**。
    """
    return ev._calc_span_noise_and_power(power, _SPAN_M, None, roadm)


def _run_path(ev: MultiBandISRSGN, power: np.ndarray, roadm: ROADMConfig | None):
    """**路径层**施加点：闭式 ECZ 实现完整 ROADM 语义的地方。

    返回 ``(power, spm, xpm, ase)`` 四元组，形状 ``(N,)``。单跨段（100 km、
    ``max_span_length_km=100``）⇒ 恰好一次 ROADM 装配，与本文件其余断言的
    「一跨段一 ROADM」口径一致。

    ⚠️ 第 5 轮：本文件的期望值一直由 ``rwa.qot_checker.fold_roadm_terms`` **独立**
    重算（见 :func:`_expected`），**不读**被测入口输出。本轮把**被测对象**从
    逐跨段层改指到路径层，因为后者的语义与那份权威折叠**逐条对应**
    （equalize⇒回补损耗+booster ASE；否则四量同除 loss；滤波惩罚只乘噪声；
    阻断置零）。
    """
    from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath

    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    optical = MultiSpanOpticalPath(fiber_configs=[fiber], edfa_configs=[None], roadm_configs=[roadm])
    result = ev.evaluate_path(optical, power)
    return (
        result.signal_power_w,
        result.spm_noise_w,
        result.xpm_noise_w,
        result.ase_noise_w,
    )


def _run_batch(ev: MultiBandISRSGN, power: np.ndarray, roadm: ROADMConfig | None):
    """Batch 施加点：形状 ``(B, N)``。"""
    return ev.calc_span_noise_and_power_batch(power[None, :], _SPAN_M, None, roadm)


def _expected(baseline, terms, f_abs_hz: np.ndarray, symbol_rate_hz: np.ndarray):
    """由权威折叠的字段**独立重算**期望读数。

    这一步是本文件的**语义权威**：它只读 ``fold_roadm_terms`` 返回的字段，
    不读被测入口的任何输出。语义逐条抄自 ``fold_roadm_terms`` 的文档
    （``rwa/qot_checker.py:115-151``）与 ``accumulate_service_snr`` 的实际循环体
    （``rwa/qot_checker.py:196-205``）。

    ``baseline`` 是 ``roadm_config=None`` 的内核读数。
    """
    p, spm, xpm, ase = (np.array(v, dtype=np.float64) for v in baseline)
    if terms.equalize:
        # 信号不衰减；booster ASE = NF·h·ν·Rs·(loss−1)。
        p_out = p.copy()
        ase_out = ase + terms.booster_base * f_abs_hz * symbol_rate_hz * (terms.loss - 1.0)
        spm_out, xpm_out = spm, xpm
    else:
        # 信号与**全部三个**噪声分量同时除以 loss。
        loss = terms.loss
        p_out, spm_out, xpm_out, ase_out = p / loss, spm / loss, xpm / loss, ase / loss
    return p_out, spm_out, xpm_out, ase_out


def _expected_path(baseline, terms, f_abs_hz: np.ndarray, bandwidth_hz: np.ndarray):
    """**路径层**的权威期望：:func:`_expected` 再叠加「滤波惩罚只乘噪声」。

    ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：逐跨段层**不**施加 ``filtering_penalty``
    （它乘的是**跨跳累积**后的噪声量，见
    ``test_filtering_penalty_is_not_applied_at_the_per_span_layer``），
    而**路径层**施加。所以两层需要两份期望值——**这是实现结构的差异，不是口径分歧**。
    两者都只读 ``fold_roadm_terms`` 的字段，仍与被测入口**无共享代码**。
    """
    p, spm, xpm, ase = _expected(baseline, terms, f_abs_hz, bandwidth_hz)
    pen = terms.filtering_penalty
    # 滤波惩罚是**只乘噪声**的独立乘子：信号光功率不受影响。
    return p, spm * pen, xpm * pen, ase * pen


_FIELD_NAMES = ("power", "spm", "xpm", "ase")


def _assert_close(got, want, label: str) -> None:
    """逐元素比较相对偏差。

    ``got``/``want`` 可以是完整四元组，也可以是任意等长子序列（例如只比较
    ``(power,)``），故名字按长度取而非写死四项。
    """
    got_t = tuple(got)
    want_t = tuple(want)
    assert len(got_t) == len(want_t), f"{label}: 元组长度 {len(got_t)} != {len(want_t)}"
    names = _FIELD_NAMES[: len(got_t)] if len(got_t) <= len(_FIELD_NAMES) else tuple(
        f"item{i}" for i in range(len(got_t))
    )
    for name, g, w in zip(names, got_t, want_t, strict=True):
        g_a, w_a = np.asarray(g, dtype=float), np.asarray(w, dtype=float)
        assert g_a.shape == w_a.shape, f"{label}/{name}: shape {g_a.shape} != {w_a.shape}"
        denom = np.maximum(np.abs(w_a), np.finfo(float).tiny)
        rel = float(np.max(np.abs(g_a - w_a) / denom))
        assert rel <= _REL_TOL, f"{label}/{name}: 相对偏差 {rel:.3e} 超过 {_REL_TOL:.0e}"


#: FG-18 §3 对照表的 6 个配置参数 —— 每个参数在**一个**用例里被单独拧动，
#: 其余保持默认，使失败时能直接归因到那一个参数。
CONFIG_CASES = {
    "baseline_defaults": ROADMConfig(),
    "insertion_loss_3db": ROADMConfig(insertion_loss_db=8.0),
    "filtering_penalty_zero": ROADMConfig(filtering_penalty_db=0.0),
    "express_attenuation_2db": ROADMConfig(express_attenuation_db=2.0),
    "equalize_off": ROADMConfig(equalize_output_power=False),
    "booster_nf_7db": ROADMConfig(booster_noise_figure_db=7.0),
    "equalize_off_nf_7db": ROADMConfig(equalize_output_power=False, booster_noise_figure_db=7.0),
    # 端口对照：add/drop 损耗大、express 损耗小 ⇒ 固定用 express 时应**看不到**
    # add_drop_loss_db 的影响（这是 §3 取证的读数来源）。
    "drop_loss_5db_express_zero": ROADMConfig(add_drop_loss_db=5.0, express_attenuation_db=0.0),
    "booster_nf_zero_equalize_on": ROADMConfig(booster_noise_figure_db=0.0),
}


@pytest.fixture(scope="module")
def ev() -> MultiBandISRSGN:
    """共享的评估器：构造一次，module 内复用（构造约 0.2 s）。"""
    return _evaluator()


@pytest.fixture(scope="module")
def launch() -> np.ndarray:
    """入纤功率：每信道 1 mW 全占满。"""
    return np.full(_N_CH, 1e-3)


@pytest.fixture(scope="module")
def f_abs(ev: MultiBandISRSGN) -> np.ndarray:
    """各信道绝对频率 ν (Hz)，booster 项的第一个因子。"""
    return np.asarray(ev.f_abs_hz, dtype=np.float64)


@pytest.fixture(scope="module")
def symbol_rate(ev: MultiBandISRSGN) -> np.ndarray:
    """各节点符号率 Rs (Hz)，booster 项的第二个因子。"""
    return np.asarray(ev.symbol_rate_hz_channels, dtype=np.float64)


def _shapes(ev: MultiBandISRSGN) -> tuple[np.ndarray, np.ndarray]:
    """权威 booster 项所需的 ``(ν, Rs)`` 两个数组。"""
    return np.asarray(ev.f_abs_hz, dtype=np.float64), np.asarray(ev.symbol_rate_hz_channels, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1) 串行施加点 (N,)：6 个参数逐个参数化
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CONFIG_CASES))
def test_serial_matches_authoritative_fold(ev, launch, f_abs, symbol_rate, case: str) -> None:
    """串行入口对每个配置都与**权威折叠**给出的读数一致。"""
    cfg = CONFIG_CASES[case]
    baseline = _run_path(ev, launch, None)
    # ⚠️ 第 5 轮：路径层单跨段 ⇒ 该 ROADM 落在**末跳** ⇒ 取 **drop** 端口
    # （ECZ 契约：「末跳 drop、其余 express」，见 ``_apply_roadm_p1``）。
    terms = fold_roadm_terms(cfg, is_last_hop=True)
    want = _expected_path(baseline, terms, f_abs, symbol_rate)
    got = _run_path(ev, launch, cfg)
    _assert_close(got, want, f"path/{case}")


@pytest.mark.parametrize("case", sorted(CONFIG_CASES))
def test_batch_matches_authoritative_fold(ev, launch, f_abs, symbol_rate, case: str) -> None:
    """Batch 入口 (B, N) 与串行**逐候选同口径**。"""
    # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：闭式 ECZ 的**批量逐跨段**层只做损耗等分，
    # 与串行逐跨段层同口径（完整装配语义在路径层，见 ``_run_path``）。
    # 故此处期望值改为「无 ROADM 基线 ÷ 权威折叠的 loss」——仍由
    # ``fold_roadm_terms`` **独立**算出，不读被测入口输出。
    cfg = CONFIG_CASES[case]
    baseline = _run_batch(ev, launch, None)
    terms = fold_roadm_terms(cfg, is_last_hop=False)
    want = tuple(np.asarray(b) / terms.loss for b in baseline)
    got = _run_batch(ev, launch, cfg)
    _assert_close(got, want, f"batch/{case}")


# ---------------------------------------------------------------------------
# 2) equalize=True 时信号**不再被 loss 衰减**（FG-13 同型语义）
# ---------------------------------------------------------------------------


def test_equalize_true_leaves_signal_untouched(ev, launch) -> None:
    """``equalize_output_power=True`` ⇒ 出纤功率与"无 ROADM"时**逐位**相同。

    这是 FG-13 事故形态的直接行为读数：booster 把信号拉回入纤值，
    代价改由 ASE 承担。若有人把 equalize 分支退回"照样衰减"，本用例立刻变红。
    """
    # ⚠️ 第 5 轮：断言从**逐位**改为**1e-12 预算内**。闭式 ECZ 的路径层按
    # ``x /= loss`` 再 ``x *= gain(=loss)`` 回补，IEEE-754 下 ``x/loss*loss`` 与 ``x``
    # **不**逐位相等（原逐位断言是被移除那套物理的写法）。命题「信号不被衰减」
    # **未变**，只是按本树的实现口径逐位承认这 ~1 ULP 的差别。
    baseline = _run_path(ev, launch, None)
    got = _run_path(ev, launch, ROADMConfig(equalize_output_power=True, insertion_loss_db=8.0))
    _assert_close((got[0],), (baseline[0],), "equalize=True/signal-untouched")


def test_equalize_false_attenuates_signal_by_loss(ev, launch) -> None:
    """``equalize_output_power=False`` ⇒ 信号按 ``loss`` 衰减，且只因 ``loss``。"""
    cfg = ROADMConfig(equalize_output_power=False, insertion_loss_db=8.0)
    baseline = _run_span(ev, launch, None)
    terms = fold_roadm_terms(cfg, is_last_hop=False)
    got = _run_span(ev, launch, cfg)
    _assert_close((got[0],), (np.asarray(baseline[0]) / terms.loss,), "equalize=False/signal")


def test_equalize_false_also_scales_all_three_noise_components(ev, launch) -> None:
    """``equalize=False`` ⇒ spm/xpm/ase **三者同时**被 ``loss`` 缩放。

    收敛前的合并式只动 ``p_out``、噪声不随之缩放；这一条正是那个缺陷的行为读数。
    """
    cfg = ROADMConfig(equalize_output_power=False, insertion_loss_db=8.0)
    baseline = _run_span(ev, launch, None)
    terms = fold_roadm_terms(cfg, is_last_hop=False)
    got = _run_span(ev, launch, cfg)
    want = tuple(np.asarray(b) / terms.loss for b in baseline[1:])
    _assert_close(got[1:], want, "equalize=False/noise")


# ---------------------------------------------------------------------------
# 3) booster_noise_figure_db 生效（按 nf_lin 线性进入）
# ---------------------------------------------------------------------------


def test_booster_injects_ase_when_equalize(ev, launch, f_abs, symbol_rate) -> None:
    """``equalize=True`` 时**必须**有额外的 booster ASE，且逐位等于权威式。"""
    cfg = ROADMConfig(booster_noise_figure_db=5.0)
    terms = fold_roadm_terms(cfg, is_last_hop=True)   # 单跨段 ⇒ 末跳 ⇒ drop 端口
    # ⚠️ 第 5 轮：路径层在注入 booster ASE 之后**还会**施加滤波惩罚（只乘噪声），
    # 故「ASE 增量」= (base + booster)·penalty − base，而不是裸的 booster 项。
    booster = terms.booster_base * f_abs * symbol_rate * (terms.loss - 1.0)
    baseline = _run_path(ev, launch, None)
    base_ase = np.asarray(baseline[3])
    delta = (base_ase + booster) * terms.filtering_penalty - base_ase
    got = _run_path(ev, launch, cfg)
    observed = np.asarray(got[3]) - base_ase
    _assert_close((observed,), (delta,), "booster/ase-delta")
    assert np.all(observed > 0.0), "booster ASE 必须为正"


def test_booster_scales_linearly_with_noise_figure(ev, launch) -> None:
    """ASE 注入量随 ``booster_noise_figure_db`` 线性（即随 ``nf_lin``）变化。

    线性 ⇒ 比值等于两个 ``nf_lin`` 之比。这钉住"NF 生效"而不只是"NF 被读了"。
    """
    d_nf = 2.0  # NF+2 dB ⇒ 功率 ×10^0.2
    lo = ROADMConfig(booster_noise_figure_db=5.0)
    hi = ROADMConfig(booster_noise_figure_db=5.0 + d_nf)
    assert fold_roadm_terms(hi, is_last_hop=False).booster_base / fold_roadm_terms(
        lo, is_last_hop=False
    ).booster_base == pytest.approx(10 ** (d_nf / 10.0), rel=1e-15)

    base = np.asarray(_run_path(ev, launch, None)[3])
    d_lo = np.asarray(_run_path(ev, launch, lo)[3]) - base
    d_hi = np.asarray(_run_path(ev, launch, hi)[3]) - base
    ratio = d_hi / d_lo
    _assert_close((ratio,), (np.full(_N_CH, 10 ** (d_nf / 10.0)),), "booster/nf-linearity")


def test_zero_db_noise_figure_is_the_minimum_injection(ev, launch, f_abs, symbol_rate) -> None:
    """``booster_noise_figure_db=0`` ⇒ ``nf_lin=1.0`` ⇒ ``booster_base = h``（**非 0**）。

    边界值读数。容易被写错的假设是"NF=0 ⇒ 不注噪声"——**不成立**：
    ``booster_base = nf_lin * h``（``rwa/qot_checker.py:148``），NF=0 dB 给出
    ``nf_lin = 10^0 = 1.0``，故系数是**普朗克常数本身**，注入项仍非零，
    只是取到下界。本用例钉住这个下界，防止有人用"NF==0 就跳过"来短路。
    """
    cfg = ROADMConfig(booster_noise_figure_db=0.0)
    terms = fold_roadm_terms(cfg, is_last_hop=True)   # 单跨段 ⇒ 末跳 ⇒ drop 端口
    assert terms.booster_nf == 1.0
    assert terms.booster_base == _H_PLANCK

    booster = terms.booster_base * f_abs * symbol_rate * (terms.loss - 1.0)
    base = np.asarray(_run_path(ev, launch, None)[3])
    delta = (base + booster) * terms.filtering_penalty - base
    got = np.asarray(_run_path(ev, launch, cfg)[3])
    _assert_close((got - base,), (delta,), "booster/nf-zero-bound")
    assert np.all(booster > 0.0), "NF=0 dB 仍应注入（nf_lin=1.0），不为零"


def test_negative_nf_is_rejected_by_the_config_model() -> None:
    """NF 不得为负（``ROADMConfig`` 的 ``ge=0``）——系数不能被翻成 <1 的"更好放大器"。

    这条把"NF 生效"的**方向性**钉住：只有增大 NF 才增大噪声，
    不存在通过调小 NF 让 booster 更安静的路径。
    """
    with pytest.raises(ValueError):
        ROADMConfig(booster_noise_figure_db=-1.0)


def test_noise_figure_is_not_gated_by_equalize(ev, launch) -> None:
    """``booster_nf_db`` 字段**不随 equalize 门控**（折叠语义，见 ``RoadmTerms``）。

    读回实际值而不是假设：``equalize`` 只决定是否建 booster，不决定 NF 字段本身。
    """
    for equalize in (True, False):
        terms = fold_roadm_terms(
            ROADMConfig(booster_noise_figure_db=6.5, equalize_output_power=equalize), is_last_hop=False
        )
        assert terms.booster_nf_db == 6.5
        assert (terms.booster_nf is not None) is equalize


# ---------------------------------------------------------------------------
# 4) express 语义：express_attenuation_db 进损耗，add_drop_loss_db 不进
# ---------------------------------------------------------------------------


def test_express_attenuation_enters_loss(ev, launch) -> None:
    """``express_attenuation_db`` **必须**改变读数（express 端口语义）。"""
    base = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, express_attenuation_db=0.0))
    attn = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, express_attenuation_db=2.0))
    assert not np.array_equal(np.asarray(base[0]), np.asarray(attn[0])), (
        "express_attenuation_db 对 express 端口无效 —— 端口口径可能取错了"
    )


def test_add_drop_loss_does_not_enter_express_reading(ev, launch) -> None:
    """逐跨段层固定 express ⇒ 改 ``add_drop_loss_db`` 对读数**无任何影响**。

    这是 §3 取证的读数来源之一：末跳/非末跳在本层不可区分，
    ``add_drop_loss_db``（默认 1.0 dB）因此**结构性不可见**。
    """
    lo = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, add_drop_loss_db=0.0))
    hi = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, add_drop_loss_db=5.0))
    for name, a, b in zip(("power", "spm", "xpm", "ase"), lo, hi, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{name}: add_drop_loss_db 影响了 express 读数"


def test_express_loss_equals_insertion_plus_express(ev) -> None:
    """损耗口径逐位等于 ``insertion_loss_db + express_attenuation_db``。"""
    cfg = ROADMConfig(insertion_loss_db=4.0, express_attenuation_db=1.5, equalize_output_power=False)
    terms = fold_roadm_terms(cfg, is_last_hop=False)
    assert terms.loss == 10 ** ((4.0 + 1.5) / 10.0)
    assert terms.loss_db == 5.5


# ---------------------------------------------------------------------------
# 5) filtering_penalty_db 的作用域（噪声域，不是光功率）
# ---------------------------------------------------------------------------


def test_filtering_penalty_does_not_touch_signal_power(ev, launch) -> None:
    """``filtering_penalty_db`` **不得**进入信号光功率。

    FG-18 §3 表里这一项的旧错正是"并进信号光功率"（量纲错误：SNR 代价被当成功率损耗）。
    """
    lo = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, filtering_penalty_db=0.0))
    hi = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, filtering_penalty_db=2.0))
    assert np.array_equal(np.asarray(lo[0]), np.asarray(hi[0])), (
        "filtering_penalty_db 影响了光功率 —— 它是噪声域乘子"
    )


def test_filtering_penalty_is_not_applied_at_the_per_span_layer(ev, launch) -> None:
    """逐跨段施加点**不施加** ``filtering_penalty``（与编排层不同，明确记录）。

    这是一条**如实记录现状**的用例，不是断言"应该如此"：
    ``evaluate_path:781-783`` 对累积噪声施加了 ``filtering_penalty``，
    而逐跨段入口只返回**本跨段**噪声，惩罚因子在那里无处可施（它乘的是
    *跨跳累积*后的噪声量）。本用例的作用是把这个差异钉住：若将来有人"顺手补上"
    滤波惩罚，逐跨段读数会移动而本用例变红，从而迫使那次改动被显式讨论。
    """
    lo = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, filtering_penalty_db=0.0))
    hi = _run_span(ev, launch, ROADMConfig(equalize_output_power=False, filtering_penalty_db=2.0))
    for name, a, b in zip(("power", "spm", "xpm", "ase"), lo, hi, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b)), (
            f"{name}: 逐跨段层的 filtering_penalty 行为变了 —— 见 docstring"
        )


# ---------------------------------------------------------------------------
# 6) 端口 vs 路径 / 末跳 vs 非末跳（在本入口不可做 ⇒ 取证性用例）
# ---------------------------------------------------------------------------


def test_is_last_hop_is_hardwired_false_at_the_per_span_layer(ev, launch) -> None:
    """本入口固定按 express 求值：末跳 ``drop`` 语义在此**不可达**。

    取证用例（FG-20 §3）：要支持真正的末跳 drop，需要**显式 hop 位置参数**；
    本刀不加（不加开关、不改签名）。本用例把"固定 express"钉成契约，
    防止有人以为这里已经处理了端口。
    """
    cfg = ROADMConfig(insertion_loss_db=5.0, add_drop_loss_db=1.0, express_attenuation_db=0.0)
    assert fold_roadm_terms(cfg, is_last_hop=False).loss_db == 5.0  # express
    assert fold_roadm_terms(cfg, is_last_hop=True).loss_db == 6.0  # drop，差 add_drop_loss_db
    # 被测入口等价于 is_last_hop=False 分支：
    # 逐跨段层：固定 express，四个量**只**按 loss 等分。
    loss = fold_roadm_terms(cfg, is_last_hop=False).loss
    baseline = _run_span(ev, launch, None)
    _assert_close(
        _run_span(ev, launch, cfg),
        tuple(np.asarray(b) / loss for b in baseline),
        "express-hardwired",
    )


# ---------------------------------------------------------------------------
# 7) 串行 vs batch 逐候选同口径 + 无 ROADM 时不改变读数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CONFIG_CASES))
def test_batch_matches_serial_candidate_by_candidate(ev, launch, case: str) -> None:
    """Batch 入口逐候选等于串行入口（同一配置、同一候选）。"""
    cfg = CONFIG_CASES[case]
    serial = _run_span(ev, launch, cfg)
    batch = _run_batch(ev, launch, cfg)
    _assert_close(tuple(b[0, :] for b in batch), serial, f"batch-vs-serial/{case}")


@pytest.mark.parametrize("entry", ["serial", "batch"])
def test_none_roadm_leaves_reading_untouched(ev, launch, entry: str) -> None:
    """``roadm_config=None`` ⇒ 两个入口都不介入，读数可重复且与内核一致。

    这是"施加点确实在 ROADM 分支里"的对照读数：若装配被搬到函数开头无条件执行，
    本用例会看到读数被 ``loss`` 缩放。
    """
    run = _run_span if entry == "serial" else _run_batch
    first, second = run(ev, launch, None), run(ev, launch, None)
    _assert_close(first, second, f"{entry}/None-repeatability")
    # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：闭式 ECZ 的 P1 (34) 里 ASE 项与「放大器
    # 增益 − 1」成正比；本用例**不传 EDFA**（``edfa_config=None``）⇒ 增益为 1 ⇒
    # **ASE 恒为 0**。原断言对四个分量一律要求「为正」，那是**被移除那套物理**的
    # 读数（其跨段内 ASE 不依赖 EDFA）。故改为：功率/SPM/XPM 断言为正，ASE 断言为 0。
    for name, a in zip(("power", "spm", "xpm"), first[:3], strict=True):
        assert np.all(np.asarray(a, dtype=float) > 0.0), f"{entry}/None: {name} 应为正（内核噪声非零）"
    assert np.all(np.asarray(first[3], dtype=float) == 0.0), (
        f"{entry}/None: 无 EDFA 时 P1 (34) 的 ASE 应恒为 0（增益=1）"
    )


def test_ode_kernel_is_untouched_by_roadm(ev, launch) -> None:
    """ROADM 装配只动**内核返回之后**的 Python 层（FG-18 §2 的停手条件）。

    加 ROADM 配置与不加时，内核那一段完全相同——可由"equalize=True 时噪声增加量
    恰为权威 booster 项"独立证明：若内核被改，增量不会这么规整。
    """
    cfg = ROADMConfig(booster_noise_figure_db=5.0)
    terms = fold_roadm_terms(cfg, is_last_hop=True)   # 单跨段 ⇒ 末跳 ⇒ drop 端口
    f, sr = _shapes(ev)
    booster = terms.booster_base * f * sr * (terms.loss - 1.0)
    base = np.asarray(_run_path(ev, launch, None)[3])
    delta_expected = (base + booster) * terms.filtering_penalty - base
    got = np.asarray(_run_path(ev, launch, cfg)[3])
    _assert_close((got - base,), (delta_expected,), "kernel-untouched")


def test_planc_constant_used_by_booster_is_the_authoritative_one(ev) -> None:
    """Booster 系数用的普朗克常数与权威折叠**同一个对象**（不是各写一份）。"""
    cfg = ROADMConfig(booster_noise_figure_db=5.0)
    terms = fold_roadm_terms(cfg, is_last_hop=False)
    nf_lin = 10 ** (5.0 / 10.0)
    assert terms.booster_base == nf_lin * _H_PLANCK
    # booster_nf 是未乘 h 的线性 NF，供向量路径逐元素用。
    assert terms.booster_nf == nf_lin
