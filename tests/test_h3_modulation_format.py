"""``pcfm_h3_raman`` 的 Semrau 2019b 调制格式修正（任务 AN）验收测试。

覆盖：

* 默认 ``modulation_format_correction=False`` 时逐位不变（C / C+L / S+C+L）；
* ``Φ=0`` 的显式恒等修正逐位不变；
* ``Φ>0`` 只改 XCI，SCI 与信号/ASE 不动，且方向与 2019b 一致；
* 逐信道 XCI 修正比例就是 2019b 的 XPM 比例（残差 ~0），总 NLI 修正比例与
  ``semrau2019(egn=True)`` 的差 <0.1 dB；
* 单跨时修正比例精确等于 Eq. (16) 的 ``1 + 5Φ/6``；
* 非法 ``Φ``（超出 [-2,100]、NaN、inf）与 ``egn=True`` 叠加被拒绝；
* 真实 64-QAM 的负 ``Φ`` 允许，并使 XCI/SNR 朝 2019b 方向变化；
* 给了 ``excess_kurtosis`` 却没开开关时警告。

审计口径与残差分项见 ``H3_TASK_REPORT.md`` 与
``scripts/semrau2019b_modformat_audit.py``。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import SpanContext
from network_nlo_eval.network.elements import EDFAConfig

#: C 波段 10 跨（残差对照的主算例）。
CASE_C = build_case("c_57ch")
#: C 波段 1 跨（Eq. (16) 的 n=1 精确极限）。
CASE_C_1SPAN = build_case("scaling_40ch")

#: 基线 ``9477ae6`` 上 ``pcfm_h3_raman`` 的中点信道读数。
#: 用于锁住"默认路径逐位不变"，不是本轮生成的值。
BASELINE_MID: dict[tuple[str, tuple[tuple[str, object], ...]], dict[str, float]] = {
    ("c_57ch", (("egn", False),)): {
        "snr_db": 16.166088786236465,
        "spm_w": 1.2052889597270188e-05,
        "xpm_w": 1.8386248309100966e-05,
        "ase_w": 1.1975549019292502e-05,
    },
    # 纯默认：egn="auto" 在无泵链路上解析为 True。
    ("c_57ch", ()): {
        "snr_db": 16.27693330798564,
        "spm_w": 1.0182401660669834e-05,
        "xpm_w": 1.9187888735377772e-05,
        "ase_w": 1.1975549019292502e-05,
    },
    ("c_57ch", (("egn", False), ("poly_degree", 5))): {
        "snr_db": 16.160212433404112,
        "spm_w": 1.2046154514715045e-05,
        "xpm_w": 1.8450412711037557e-05,
        "ase_w": 1.1975549019292502e-05,
    },
    ("cl_152ch", (("egn", False),)): {
        "snr_db": 17.35001997087385,
        "spm_w": 1.0474111533947026e-06,
        "xpm_w": 1.922077135764031e-06,
        "ase_w": 1.2276459036748751e-05,
    },
    ("scl_277ch", (("egn", False),)): {
        "snr_db": 16.007762868561677,
        "spm_w": 8.632838640319448e-07,
        "xpm_w": 1.9785432646785446e-06,
        "ase_w": 1.589706641986973e-05,
    },
}


def _run(case, **options):
    """按算例构造模型并跑一次路径。"""
    model = make_model("pcfm_h3_raman", case, **options)
    return model.evaluate_path(build_path(case), launch_profile(case))


def _nli(result) -> np.ndarray:
    """总 NLI（SCI + XCI）。"""
    return np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)


def _ratio_db(numerator, denominator) -> np.ndarray:
    """两个 NLI 向量之比的 dB 值。"""
    return 10.0 * np.log10(np.asarray(numerator) / np.asarray(denominator))


# ---------------------------------------------------------------------------
# 默认不变 / Φ=0 不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(BASELINE_MID, key=lambda item: (item[0], str(item[1]))))
def test_default_path_is_bit_for_bit_the_baseline(key) -> None:
    """默认构造的 NLI/ASE/SNR 与基线 9477ae6 逐位相同。"""
    case_name, options = key
    case = build_case(case_name)
    result = _run(case, **dict(options))
    mid = case.num_channels // 2
    golden = BASELINE_MID[key]
    assert result.snr_db[mid] == golden["snr_db"]
    assert result.spm_noise_w[mid] == golden["spm_w"]
    assert result.xpm_noise_w[mid] == golden["xpm_w"]
    assert result.ase_noise_w[mid] == golden["ase_w"]


def test_zero_kurtosis_is_the_identity_correction() -> None:
    """``modulation_format_correction=True`` 配 ``Φ=0`` 必须逐位等于默认。"""
    default = _run(CASE_C, egn=False)
    identity = _run(CASE_C, egn=False, modulation_format_correction=True, excess_kurtosis=0.0)
    assert np.array_equal(default.spm_noise_w, identity.spm_noise_w)
    assert np.array_equal(default.xpm_noise_w, identity.xpm_noise_w)
    assert np.array_equal(default.ase_noise_w, identity.ase_noise_w)
    assert np.array_equal(default.snr_db, identity.snr_db)


# ---------------------------------------------------------------------------
# Φ>0：方向、分项与参考模型的对照
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.2, 0.5])
def test_positive_kurtosis_changes_only_the_xci(phi: float) -> None:
    """Φ>0 只放大 XCI：SCI、信号、ASE 不动，NLI 与 SNR 按 2019b 方向变化。"""
    base = _run(CASE_C, egn=False)
    corrected = _run(CASE_C, egn=False, modulation_format_correction=True, excess_kurtosis=phi)

    # SCI 与信号/ASE 完全不动。
    assert np.array_equal(base.spm_noise_w, corrected.spm_noise_w)
    assert np.array_equal(base.signal_power_w, corrected.signal_power_w)
    assert np.array_equal(base.ase_noise_w, corrected.ase_noise_w)

    # 2019b 的 XPM 修正在 Φ>0 时增大 NLI。
    active = np.asarray(base.xpm_noise_w) > 0.0
    assert np.all(np.asarray(corrected.xpm_noise_w)[active] > np.asarray(base.xpm_noise_w)[active])
    assert np.all(_nli(corrected)[active] > _nli(base)[active])
    assert np.all(np.asarray(corrected.snr_db)[active] < np.asarray(base.snr_db)[active])


@pytest.mark.parametrize("phi", [0.2, 0.5])
def test_xci_ratio_matches_the_semrau2019b_xpm_ratio(phi: float) -> None:
    """逐信道 XCI 修正比例与 ``semrau2019`` 的 XPM 比例一致（验收主判据）。"""
    base = _run(CASE_C, egn=False)
    corrected = _run(CASE_C, egn=False, modulation_format_correction=True, excess_kurtosis=phi)

    sem_gn = make_model("semrau2019", CASE_C, egn=False).evaluate_path(
        build_path(CASE_C), launch_profile(CASE_C)
    )
    sem_egn = make_model("semrau2019", CASE_C, egn=True, excess_kurtosis=phi).evaluate_path(
        build_path(CASE_C), launch_profile(CASE_C)
    )

    active = np.asarray(sem_gn.xpm_noise_w) > 0.0
    ratio_pcfm = np.asarray(corrected.xpm_noise_w)[active] / np.asarray(base.xpm_noise_w)[active]
    ratio_semrau = np.asarray(sem_egn.xpm_noise_w)[active] / np.asarray(sem_gn.xpm_noise_w)[active]

    residual_db = np.abs(_ratio_db(ratio_pcfm, ratio_semrau))
    assert residual_db.max() < 0.1, f"XCI 比例残差 {residual_db.max():.4f} dB 超过 0.1 dB"

    # 总 NLI 的修正比例同样落在 0.1 dB 内（含 SCI/XCI 配比差异）。
    ratio_total_pcfm = _nli(corrected)[active] / _nli(base)[active]
    ratio_total_semrau = _nli(sem_egn)[active] / _nli(sem_gn)[active]
    assert np.abs(_ratio_db(ratio_total_pcfm, ratio_total_semrau)).max() < 0.1


def test_single_span_ratio_is_the_paper_factor() -> None:
    """单跨（n=1）时 2019b 的 XPM 系数精确是 ``1 + 5Φ/6``。

    Eq. (16) 的渐近项带 ``n_tilde``，单跨时为 0，所以修正只剩统计部分；这条
    测试同时验证"逐跨等效"口径没有把跨数因子重复计一次。
    """
    phi = 0.2
    base = _run(CASE_C_1SPAN, egn=False)
    corrected = _run(CASE_C_1SPAN, egn=False, modulation_format_correction=True, excess_kurtosis=phi)

    active = np.asarray(base.xpm_noise_w) > 0.0
    ratio = np.asarray(corrected.xpm_noise_w)[active] / np.asarray(base.xpm_noise_w)[active]
    assert np.allclose(ratio, 1.0 + 5.0 * phi / 6.0, rtol=1e-12)


def test_multi_span_ratio_is_not_the_bare_statistical_factor() -> None:
    """10 跨时修正比例明显大于 ``1 + 5Φ/(6N)``：渐近项不可省略。

    2019b Eq. (16) 的 XPM 修正有 ``n + 5Φ/6`` 的统计部分与 ``(5/3)Φπ`` 的渐近
    部分；只迁移前者会漏掉约 0.35 dB（Φ=0.2）。这条测试锁住"必须用参考核的
    完整比值"，防止以后有人用公式化简把它退化掉。
    """
    phi = 0.2
    base = _run(CASE_C, egn=False)
    corrected = _run(CASE_C, egn=False, modulation_format_correction=True, excess_kurtosis=phi)

    active = np.asarray(base.xpm_noise_w) > 0.0
    ratio_db = _ratio_db(np.asarray(corrected.xpm_noise_w)[active], np.asarray(base.xpm_noise_w)[active])
    struct_db = 10.0 * np.log10(1.0 + 5.0 * phi / (6.0 * CASE_C.span_count))
    assert ratio_db.min() - struct_db > 0.2, (
        f"10 跨的修正比例 {ratio_db.mean():.3f} dB 不该退化到结构项 {struct_db:.3f} dB"
    )


def test_amplitudes_stay_consistent_with_the_corrected_power() -> None:
    """多跨时复振幅的 ``|a|²`` 必须等于修正后的 XCI 功率。"""
    case = CASE_C
    model = make_model("pcfm_h3_raman", case, egn=False, modulation_format_correction=True, excess_kurtosis=0.3)
    fiber = build_path(case).fiber_configs[0]
    edfa = EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)
    power = launch_profile(case)
    context = SpanContext(span_index=0, total_spans=case.span_count, cumulative_length_m=0.0, span_length_m=100e3)
    element = model.span_step(fiber, edfa, power, 100e3, span_context=context)
    assert element.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(element.nli_amplitude_xpm) ** 2, element.xpm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(element.nli_amplitude_spm) ** 2, element.spm_noise_w, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 非法输入与互斥
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [-2.1, -3.0, float("nan"), float("inf")])
def test_out_of_range_kurtosis_is_rejected(phi: float) -> None:
    """超出 [-2, 100] 或非有限的 ``Φ`` 一律 ``ValueError``。"""
    with pytest.raises(ValueError, match="excess_kurtosis"):
        make_model("pcfm_h3_raman", CASE_C, egn=False, modulation_format_correction=True, excess_kurtosis=phi)


def test_negative_kurtosis_is_allowed_and_reduces_xci() -> None:
    """真实 64-QAM 的 Φ=-0.619 允许，且 XCI 修正比例 <1。"""
    base = _run(CASE_C, egn=False)
    corrected = _run(
        CASE_C,
        egn=False,
        modulation_format_correction=True,
        excess_kurtosis=-0.619,
    )
    assert np.all(np.isfinite(corrected.xpm_noise_w))
    assert float(np.mean(corrected.xpm_noise_w)) < float(np.mean(base.xpm_noise_w))
    # 2019b 的负 Φ 修正应使 SNR 略升（NLI 降低）。
    assert float(np.mean(corrected.snr_db)) > float(np.mean(base.snr_db))


def test_explicit_egn_is_rejected() -> None:
    """与显式 ``egn=True`` 同时开启会被拒绝（避免双重修正）。"""
    with pytest.raises(ValueError, match="egn"):
        make_model(
            "pcfm_h3_raman",
            CASE_C,
            egn=True,
            modulation_format_correction=True,
            excess_kurtosis=0.2,
        )


def test_auto_egn_on_an_unpumped_link_is_rejected_too() -> None:
    """``egn="auto"`` 在无泵链路上解析为 True，因此同样必须被拒绝。"""
    with pytest.raises(ValueError, match="egn"):
        make_model("pcfm_h3_raman", CASE_C, modulation_format_correction=True, excess_kurtosis=0.2)


def test_zero_kurtosis_is_allowed_next_to_egn() -> None:
    """``Φ=0`` 是恒等修正，不算"双重修正"，所以不拒绝。"""
    model = make_model(
        "pcfm_h3_raman",
        CASE_C,
        egn=True,
        modulation_format_correction=True,
        excess_kurtosis=0.0,
    )
    assert model.egn is True


def test_kurtosis_without_the_flag_warns() -> None:
    """只给 ``excess_kurtosis`` 但没开开关时警告，并且结果不变。"""
    with pytest.warns(UserWarning, match="modulation_format_correction"):
        model = make_model("pcfm_h3_raman", CASE_C, egn=False, excess_kurtosis=0.2)
    assert model._mf_active is False
