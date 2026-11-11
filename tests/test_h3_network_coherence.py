"""任务 BH：网络层跨间相干装配的验收测试。

覆盖契约：

* ``assemble_link_noise`` 默认 ``coherent=False`` 逐位保持"整跨 × N + 余段"
  的功率装配（与直接调用评估器单跨入口的算术顺序逐位相同）；
* 单跨时相干/非相干等价（模型在 ``total_spans <= 1`` 时本来就只给功率）；
* 多跨相干装配与 ``evaluate_path`` 的 SNR 差 ≤0.01 dB（1/5/10 跨，
  含余段与 additive FWM 拆分）；
* 无复振幅的模型（``ecz``、有活跃泵的 ``pcfm_h3_raman``）在
  ``coherent=True`` 下保持原路径，数组逐位不变；
* ROADM（均衡 booster ASE 与普通损耗）路径无 NaN。

参考只用仓库内的 ``evaluate_path`` 与同一评估器的单跨入口，不读
``data/`` 或 ``results/``。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, RamanPumpConfig, ROADMConfig
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.rwa.qot_checker import accumulate_service_snr_detail, fold_roadm_terms

_EDFA = EDFAConfig(noise_figure_db=5.5)
_SPAN_KM = 100.0
_SYMBOL_RATE_HZ = 64e9


def _fiber(length_km: float):
    """C 波段 SMF，跨长按需要缩放；参考光纤与基准算例同源。"""
    return SMF_100KM.model_copy(
        update={"length_km": length_km, "max_span_length_km": _SPAN_KM}
    )


def _model(name: str, grid: SpectrumGrid, **options: object):
    """按注册表名构造模型；参考光纤固定为单跨 100 km。"""
    return create_model(name, grid, _fiber(_SPAN_KM), symbol_rate_hz=_SYMBOL_RATE_HZ, **options)


def _path(spans: int) -> MultiSpanOpticalPath:
    return MultiSpanOpticalPath([_fiber(_SPAN_KM)] * spans, [_EDFA] * spans)


def _assemble(
    model,
    power: np.ndarray,
    *,
    spans: int,
    remainder_km: float = 0.0,
    coherent: bool = False,
):
    """单链路噪声缓存；整跨数与余段长度由调用方给出。"""
    link_km = spans * _SPAN_KM + remainder_km
    return assemble_link_noise(
        evaluator=model.for_fiber(_fiber(link_km)),
        edfa_config=_EDFA,
        power_profile=power,
        span_length_m=_SPAN_KM * 1e3,
        num_full_spans=spans,
        remainder_length_m=remainder_km * 1e3,
        num_channels=power.size,
        coherent=coherent,
    )


def _snr_db(power: np.ndarray, spm: np.ndarray, xpm: np.ndarray, ase: np.ndarray) -> np.ndarray:
    """与 ``evaluate_path`` 相同的单链路 SNR 定义（无 SPM 跨数惩罚）。"""
    return 10.0 * np.log10(power / (spm + xpm + ase))


# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["pcfm_h3_raman", "ecz"])
def test_default_assembly_keeps_full_span_times_n(model_name: str) -> None:
    """默认 ``coherent=False`` 与"整跨 × N + 余段"的功率算术逐位相同。

    预期值用同一评估器的单跨入口按文档顺序累加：从三个全零数组出发
    ``((0 + s*N) + r)``。这就是抽取前的写法，任何改动都会在这里显形。
    """
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    model = _model(model_name, grid)
    power = np.full(6, 1e-3)
    evaluator = model.for_fiber(_fiber(250.0))

    _, s_spm, s_xpm, s_ase = evaluator._calc_span_noise_and_power(
        power, _SPAN_KM * 1e3, _EDFA, None, n_effective_spans=1
    )
    _, r_spm, r_xpm, r_ase = evaluator._calc_span_noise_and_power(
        power, 50e3, _EDFA, n_effective_spans=1
    )
    expected = []
    for s, r in ((s_spm, r_spm), (s_xpm, r_xpm), (s_ase, r_ase)):
        total = np.zeros(6, dtype=np.float64)
        total += s * 2
        total += r
        expected.append(total)

    cache = _assemble(model, power, spans=2, remainder_km=50.0)
    assert cache.span_count == 3
    assert np.array_equal(cache.spm, expected[0])
    assert np.array_equal(cache.xpm, expected[1])
    assert np.array_equal(cache.ase, expected[2])


def test_single_span_coherent_equals_incoherent() -> None:
    """单跨时模型不返回振幅：相干开关与默认路径逐位相同。"""
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    power = np.full(6, 1e-3)
    for model_name in ("pcfm_h3_raman", "ecz_coherent"):
        model = _model(model_name, grid)
        incoherent = _assemble(model, power, spans=1)
        coherent = _assemble(model, power, spans=1, coherent=True)
        assert coherent.span_count == 1
        assert np.array_equal(coherent.spm, incoherent.spm)
        assert np.array_equal(coherent.xpm, incoherent.xpm)
        assert np.array_equal(coherent.ase, incoherent.ase)


# ---------------------------------------------------------------------------
# 多跨：相干装配 vs 路径级 evaluate_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", [100e9, 37.5e9])
@pytest.mark.parametrize(
    "model_name,options",
    [
        ("pcfm_h3_raman", {}),
        ("pcfm_h3_raman", {"dense_grid_fwm_fix": True}),
        ("pcfm_h3_raman", {"dense_grid_fwm_fix": True, "dense_grid_fwm_coherence": "incoherent"}),
        ("ecz_coherent", {}),
    ],
)
@pytest.mark.parametrize("spans", [1, 5, 10])
def test_coherent_assembly_matches_evaluate_path(
    model_name: str, options: dict[str, object], spans: int, spacing_hz: float
) -> None:
    """多跨相干装配与 ``evaluate_path`` 的 SNR 差 ≤0.01 dB。

    6 dBm/信道是偏差最大的工作点（非线性项主导），比 0 dBm 更能暴露口径
    错误；参考值是路径级的 ``evaluate_path``，两只脚都走同一份单跨物理。
    """
    num_channels = 8 if spacing_hz >= 100e9 else 10
    grid = SpectrumGrid(num_channels, spacing_hz, 193.1e12)
    model = _model(model_name, grid, **options)
    power = np.full(num_channels, 1e-3 * 10 ** (6.0 / 10.0))

    path_result = model.evaluate_path(_path(spans), power)
    cache = _assemble(model, power, spans=spans, coherent=True)

    snr_path = _snr_db(
        power, path_result.spm_noise_w, path_result.xpm_noise_w, path_result.ase_noise_w
    )
    snr_cache = _snr_db(power, cache.spm, cache.xpm, cache.ase)
    assert np.all(np.isfinite(snr_cache))
    assert float(np.max(np.abs(snr_cache - snr_path))) <= 0.01

    # 相干装配逐项复现路径级数组（允许跨平台末位差）。
    assert np.allclose(cache.spm, path_result.spm_noise_w, rtol=1e-10, atol=0.0)
    assert np.allclose(cache.xpm, path_result.xpm_noise_w, rtol=1e-10, atol=0.0)
    assert np.allclose(cache.ase, path_result.ase_noise_w, rtol=1e-10, atol=0.0)


def test_coherent_assembly_handles_remainder_span() -> None:
    """整跨 + 余段：与 ``evaluate_path`` 的跨切分一致（100 + 50 km）。"""
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(6, 1e-3 * 10 ** (6.0 / 10.0))

    optical_path = MultiSpanOpticalPath([_fiber(150.0)], [_EDFA])
    path_result = model.evaluate_path(optical_path, power)
    cache = _assemble(model, power, spans=1, remainder_km=50.0, coherent=True)

    assert cache.span_count == 2
    snr_path = _snr_db(
        power, path_result.spm_noise_w, path_result.xpm_noise_w, path_result.ase_noise_w
    )
    snr_cache = _snr_db(power, cache.spm, cache.xpm, cache.ase)
    assert float(np.max(np.abs(snr_cache - snr_path))) <= 0.01


@pytest.mark.parametrize(
    "model_name,options",
    [
        ("ecz_coherent", {}),
        ("pcfm_h3_raman", {"signal_polarizations": 2}),
    ],
)
def test_coherent_assembly_matches_evaluate_path_in_per_pol(
    model_name: str, options: dict[str, object]
) -> None:
    """``per_pol`` 坐标下振幅与噪声同坐标：相干装配同样复现路径级结果。"""
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    model = create_model(
        model_name,
        grid,
        _fiber(_SPAN_KM),
        symbol_rate_hz=_SYMBOL_RATE_HZ,
        launch_power_convention="per_pol",
        **options,
    )
    power = np.full(6, 0.5e-3 * 10 ** (6.0 / 10.0))

    path_result = model.evaluate_path(_path(5), power)
    cache = _assemble(model, power, spans=5, coherent=True)

    snr_path = _snr_db(
        power, path_result.spm_noise_w, path_result.xpm_noise_w, path_result.ase_noise_w
    )
    snr_cache = _snr_db(power, cache.spm, cache.xpm, cache.ase)
    assert float(np.max(np.abs(snr_cache - snr_path))) <= 0.01
    assert np.allclose(cache.ase, path_result.ase_noise_w, rtol=1e-12, atol=0.0)


def test_incoherent_assembly_differs_when_coherence_matters() -> None:
    """10 跨 6 dBm：非相干"整跨 × N"与路径级结果不可忽略地偏离。

    这条断言守住审计的核心结论：默认网络口径不是路径级相干口径，差值来自
    跨间相位而不是单跨物理。
    """
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(8, 1e-3 * 10 ** (6.0 / 10.0))

    path_result = model.evaluate_path(_path(10), power)
    incoherent = _assemble(model, power, spans=10)
    coherent = _assemble(model, power, spans=10, coherent=True)

    snr_path = _snr_db(
        power, path_result.spm_noise_w, path_result.xpm_noise_w, path_result.ase_noise_w
    )
    snr_incoherent = _snr_db(power, incoherent.spm, incoherent.xpm, incoherent.ase)
    snr_coherent = _snr_db(power, coherent.spm, coherent.xpm, coherent.ase)
    assert float(np.max(np.abs(snr_incoherent - snr_path))) > 0.05
    assert float(np.max(np.abs(snr_coherent - snr_path))) <= 0.01


# ---------------------------------------------------------------------------
# 无复振幅的模型
# ---------------------------------------------------------------------------


def test_power_only_model_ignores_coherent_flag() -> None:
    """冻结基线 ``ecz`` 没有跨间相干状态：``coherent=True`` 保持原数组。"""
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    model = _model("ecz", grid)
    power = np.full(6, 1e-3)
    assert not getattr(model.for_fiber(_fiber(1000.0)), "supports_span_context", False)

    default = _assemble(model, power, spans=10)
    coherent = _assemble(model, power, spans=10, coherent=True)
    assert np.array_equal(coherent.spm, default.spm)
    assert np.array_equal(coherent.xpm, default.xpm)
    assert np.array_equal(coherent.ase, default.ase)


def test_pumped_h3_is_power_only_and_ignores_coherent_flag() -> None:
    """有活跃泵的 H3 解析为 ``span_coherence="incoherent"``，网络层不切换口径。"""
    grid = SpectrumGrid(4, 100e9, 193.1e12)
    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",
        attenuation_db_per_km=0.25,
    )
    model = _model("pcfm_h3_raman", grid, raman_pumps=[pump])
    power = np.full(4, 1e-3)
    evaluator = model.for_fiber(_fiber(200.0))
    assert evaluator.supports_span_context is False

    default = _assemble(model, power, spans=2)
    coherent = _assemble(model, power, spans=2, coherent=True)
    assert np.array_equal(coherent.spm, default.spm)
    assert np.array_equal(coherent.xpm, default.xpm)
    assert np.array_equal(coherent.ase, default.ase)


def test_zero_span_link_returns_independent_zero_arrays() -> None:
    """零跨段：相干开关不改变"三个独立零数组 + 0 跨"的语义。"""
    grid = SpectrumGrid(3, 100e9, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(3, 1e-3)
    for flag in (False, True):
        cache = assemble_link_noise(
            evaluator=model.for_fiber(_fiber(50.0)),
            edfa_config=None,
            power_profile=power,
            span_length_m=100e3,
            num_full_spans=0,
            remainder_length_m=0.0,
            num_channels=3,
            coherent=flag,
        )
        assert cache.span_count == 0
        assert not np.any(cache.spm) and not np.any(cache.xpm) and not np.any(cache.ase)


# ---------------------------------------------------------------------------
# ROADM / EDFA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("equalize", [False, True])
def test_roadm_path_has_no_nan(equalize: bool) -> None:
    """ROADM 损耗 / 均衡 booster ASE 在相干缓存上仍给出有限 SNR。"""
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(6, 1e-3 * 10 ** (6.0 / 10.0))
    cache = _assemble(model, power, spans=5, coherent=True)

    roadm = ROADMConfig(
        insertion_loss_db=12.0,
        drop_loss_db=18.0,
        equalize_output_power=equalize,
        booster_noise_figure_db=6.0,
        filtering_penalty_db=1.5,
    )
    terms = fold_roadm_terms(roadm, is_last_hop=True)
    for index in range(6):
        qot = accumulate_service_snr_detail(
            [(terms, cache)],
            channel_idx=index,
            launch_power_w=float(power[index]),
            frequency_hz=193.1e12 + (index - 3) * 100e9,
            channel_bandwidth_hz=_SYMBOL_RATE_HZ,
            is_blocked=lambda _terms: False,
        )
        assert np.isfinite(qot.snr_db)

    assert np.all(np.isfinite(cache.spm))
    assert np.all(np.isfinite(cache.xpm))
    assert np.all(np.isfinite(cache.ase))
