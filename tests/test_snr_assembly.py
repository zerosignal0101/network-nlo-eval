"""ROADM 常量折叠、单信道 SNR 累积与链路噪声装配的行为测试。"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.rwa.qot_checker import RoadmTerms, accumulate_service_snr, fold_roadm_terms

CHANNEL_SPACING_HZ = 50e9


def _roadm(**overrides: object) -> ROADMConfig:
    fields = {
        "insertion_loss_db": 12.0,
        "drop_loss_db": 18.0,
        "equalize_output_power": False,
        "booster_noise_figure_db": 6.0,
        "filtering_penalty_db": 1.5,
    }
    fields.update(overrides)
    return ROADMConfig(**fields)


def _cache(spm: float, xpm: float, ase: float, span_count: int = 2) -> LinkNoiseCache:
    values = np.array([spm, xpm, ase], dtype=np.float64)
    return LinkNoiseCache(spm=values[:1], xpm=values[1:2], ase=values[2:], span_count=span_count)


def test_fold_uses_port_dependent_loss() -> None:
    """Drop 与 express 取不同损耗，且 loss_db 与 loss 出自同一次求值。"""
    roadm = _roadm()
    express = fold_roadm_terms(roadm, is_last_hop=False)
    drop = fold_roadm_terms(roadm, is_last_hop=True)

    assert express.loss_db == roadm.path_loss_db("express")
    assert drop.loss_db == roadm.path_loss_db("drop")
    assert drop.loss > express.loss
    assert express.loss == 10 ** (express.loss_db / 10)


def test_fold_keeps_filtering_penalty_out_of_loss() -> None:
    """滤波惩罚是噪声域乘子，不并进光功率损耗。"""
    roadm = _roadm()
    terms = fold_roadm_terms(roadm, is_last_hop=False)
    assert terms.loss == 10 ** (roadm.path_loss_db("express") / 10)
    assert terms.filtering_penalty == 10 ** (roadm.filtering_penalty_db / 10)


def test_booster_exists_only_when_equalizing() -> None:
    """Booster 只在 equalize_output_power 下存在。"""
    plain = fold_roadm_terms(_roadm(), is_last_hop=False)
    equalized = fold_roadm_terms(_roadm(equalize_output_power=True), is_last_hop=False)

    assert plain.booster_nf is None
    assert equalized.booster_nf is not None
    assert equalized.booster_nf_db == 6.0


def test_accumulate_attenuates_signal_and_noise_together() -> None:
    """无 booster 时信号与噪声同乘 1/loss，SNR 与无损耗 ROADM 相同。"""
    lossy = fold_roadm_terms(_roadm(), is_last_hop=True)
    lossless = fold_roadm_terms(_roadm(insertion_loss_db=0.0, add_drop_loss_db=0.0), is_last_hop=True)

    def snr(terms: RoadmTerms) -> float:
        return accumulate_service_snr(
            [(terms, _cache(1e-6, 1e-6, 1e-6))],
            channel_idx=0,
            launch_power_w=1e-3,
            frequency_hz=193.1e12,
            channel_bandwidth_hz=50e9,
            is_blocked=lambda _terms: False,
        )

    assert lossless.loss == 1.0
    assert np.isclose(snr(lossy), snr(lossless))
    assert snr(lossy) < float("inf")


def test_accumulate_adds_booster_ase_and_keeps_signal_power() -> None:
    """Equalize 下信号不被衰减，代价是 booster ASE 与滤波惩罚。"""
    terms = fold_roadm_terms(_roadm(equalize_output_power=True), is_last_hop=True)
    snr = accumulate_service_snr(
        [(terms, _cache(0.0, 0.0, 0.0, span_count=1))],
        channel_idx=0,
        launch_power_w=1e-3,
        frequency_hz=193.1e12,
        channel_bandwidth_hz=50e9,
        is_blocked=lambda _terms: False,
    )
    booster_ase = terms.booster_base * 193.1e12 * 50e9 * (terms.loss - 1)
    assert np.isclose(snr, 10 * np.log10(1e-3 / (booster_ase * terms.filtering_penalty)))


def test_accumulate_returns_infinity_for_zero_noise_and_negative_for_blocked() -> None:
    """噪声为 0 给 +inf；任一跳被阻断给 -inf。"""
    terms = fold_roadm_terms(_roadm(), is_last_hop=True)
    hops = [(terms, LinkNoiseCache(spm=np.zeros(1), xpm=np.zeros(1), ase=np.zeros(1), span_count=0))]
    assert accumulate_service_snr(
        hops,
        channel_idx=0,
        launch_power_w=1e-3,
        frequency_hz=193.1e12,
        channel_bandwidth_hz=50e9,
        is_blocked=lambda _terms: False,
    ) == float("inf")
    assert accumulate_service_snr(
        hops,
        channel_idx=0,
        launch_power_w=1e-3,
        frequency_hz=193.1e12,
        channel_bandwidth_hz=50e9,
        is_blocked=lambda _terms: True,
    ) == float("-inf")


def test_assemble_link_noise_splits_full_spans_and_remainder() -> None:
    """整跨与余段各自成段，噪声按整跨数放大。"""
    evaluator = MultiBandISRSGN(SpectrumGrid(4, CHANNEL_SPACING_HZ, 193.1e12), FiberSpanConfig(length_km=150.0), 50e9)
    power = np.full(4, 1e-3)
    cache = assemble_link_noise(
        evaluator=evaluator,
        edfa_config=None,
        power_profile=power,
        span_length_m=80e3,
        num_full_spans=1,
        remainder_length_m=70e3,
        num_channels=4,
    )
    _, spm_full, _, _ = evaluator._calc_span_noise_and_power(power, 80e3, None)
    _, spm_rest, _, _ = evaluator._calc_span_noise_and_power(power, 70e3, None)
    assert cache.span_count == 2
    assert np.allclose(cache.spm, spm_full + spm_rest)


def test_assemble_link_noise_ignores_short_remainder() -> None:
    """余段 <= 1 m 不成段；全零情形返回三段独立的全零数组。"""
    evaluator = MultiBandISRSGN(SpectrumGrid(2, CHANNEL_SPACING_HZ, 193.1e12), FiberSpanConfig(length_km=80.0), 50e9)
    cache = assemble_link_noise(
        evaluator=evaluator,
        edfa_config=None,
        power_profile=np.full(2, 1e-3),
        span_length_m=80e3,
        num_full_spans=0,
        remainder_length_m=0.5,
        num_channels=2,
    )
    assert cache.span_count == 0
    assert np.array_equal(cache.spm, np.zeros(2))
    assert cache.spm is not cache.xpm


def test_fold_results_are_hashable_value_objects() -> None:
    """RoadmTerms 是冻结值对象，同一配置折叠两次等价。"""
    roadm = _roadm()
    assert fold_roadm_terms(roadm, is_last_hop=True) == fold_roadm_terms(roadm, is_last_hop=True)
    assert isinstance(fold_roadm_terms(roadm, is_last_hop=True), RoadmTerms)
