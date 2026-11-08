"""``resolution`` 显式档位的回归测试。

档位是**opt-in**：默认（``None``/``"default"``）必须逐位复现既有的每选项
默认值，只有调用方写明 ``"fast"``/``"accurate"`` 时才改变分辨率旋钮。
因此这里的第一组测试钉住"不传"与"传 None"完全等价，后面的测试才是档位
自身的语义。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm import XCI_EDGE_QUADRATURE_POINTS
from network_nlo_eval.network.elements import RamanPumpConfig

#: 逐位比较覆盖的输出数组。
OUTPUT_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")

#: E 波段算例是唯一有 XCI gate 行的基准：零色散点在梳内，边缘行需要精确
#: 积分，所以它才是 quadrature 档位该被检验的地方。
ESCL = build_case("escl_479ch_1span")
#: 小算例：只用来验证档位能跑通，不参与精度结论。
SMALL = build_case("scaling_40ch")


def _run(case, model_name: str, **options):
    """构造模型并跑一次路径。"""
    model = make_model(model_name, case, **options)
    return model, model.evaluate_path(build_path(case), launch_profile(case, 3.0))


def _assert_bit_identical(left, right) -> None:
    """全部输出数组逐位相等。"""
    for field in OUTPUT_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, field)), np.asarray(getattr(right, field))), field


def test_absent_resolution_is_the_documented_default() -> None:
    """不传 = ``None`` = ``"default"``：步数、节点数与输出都逐位一致。"""
    base, base_result = _run(SMALL, "pcfm", poly_degree=3)
    explicit, explicit_result = _run(SMALL, "pcfm", poly_degree=3, resolution=None)
    alias, alias_result = _run(SMALL, "pcfm", poly_degree=3, resolution="default")

    for model in (base, explicit, alias):
        assert model.resolution is None
        assert model.spp_steps == 32
        assert model.xci_edge_quadrature_points == XCI_EDGE_QUADRATURE_POINTS
    _assert_bit_identical(base_result, explicit_result)
    _assert_bit_identical(base_result, alias_result)


def test_fast_profile_lowers_only_the_xci_quadrature() -> None:
    """``"fast"`` 只动 XCI 边缘积分的节点数，NLI 在该算例上不变。

    节点数从 24 降到 12 只是少采几个 Gauss 点：被 gate 的行（这里是 E 波段
    零色散边缘的 71 行）是唯一用到它的地方，而 12 点已经把积分收敛到
    1e-10 相对量级。任何更大的差都说明档位碰到了不该碰的旋钮。
    """
    default_model, default_result = _run(ESCL, "pcfm_h3_raman", poly_degree=3, egn="auto")
    fast_model, fast_result = _run(ESCL, "pcfm_h3_raman", poly_degree=3, egn="auto", resolution="fast")

    assert fast_model.resolution == "fast"
    assert fast_model.xci_edge_quadrature_points == 12
    # SPP/粗 ODE 步数不由该档位设定。
    assert fast_model.spp_steps == default_model.spp_steps
    assert fast_model._h3_raman_steps == default_model._h3_raman_steps

    kernel = next(iter(default_model._kernel_cache.values()))
    gated, fraction = kernel.xci_edge_audit(ESCL.span_length_km * 1e3)
    assert np.count_nonzero(gated) > 0
    assert float(np.max(fraction)) > 0.5

    for field in ("spm_noise_w", "xpm_noise_w"):
        reference = np.asarray(getattr(default_result, field))
        value = np.asarray(getattr(fast_result, field))
        assert np.max(np.abs(value - reference) / np.abs(reference)) < 1e-9, field
    # 总功率口径更严：逐信道差可以到 1e-9，总量应当低两个数量级。
    total_reference = np.sum(default_result.spm_noise_w) + np.sum(default_result.xpm_noise_w)
    total_value = np.sum(fast_result.spm_noise_w) + np.sum(fast_result.xpm_noise_w)
    assert abs(total_value - total_reference) / total_reference < 1e-9
    snr_reference = np.asarray(default_result.snr_db)
    snr_value = np.asarray(fast_result.snr_db)
    assert np.max(np.abs(snr_value - snr_reference)) < 1e-6


def test_accurate_profile_sets_the_sweep_reference_steps() -> None:
    """``"accurate"`` 就是 Pareto 报告的收敛参考：spp=96（H3 再加 raman=96）。"""
    pcfm, pcfm_result = _run(SMALL, "pcfm", poly_degree=3, resolution="accurate")
    assert pcfm.resolution == "accurate"
    assert pcfm.spp_steps == 96
    assert pcfm._active_spp_steps == 96
    assert np.all(np.isfinite(np.asarray(pcfm_result.snr_db)))

    pump = RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="backward", attenuation_db_per_km=0.25)
    h3, h3_result = _run(SMALL, "pcfm_h3_raman", poly_degree=3, egn="auto", raman_pumps=[pump], resolution="accurate")
    assert h3.resolution == "accurate"
    assert h3.spp_steps == 96
    assert h3._h3_raman_steps == 96
    kernel = next(iter(h3._kernel_cache.values()))
    assert kernel.raman_steps == 96
    assert np.all(np.isfinite(np.asarray(h3_result.snr_db)))


def test_profiles_do_not_silently_override_explicit_steps() -> None:
    """档位与显式步数冲突时报错，而不是悄悄覆盖调用方的定标。"""
    with pytest.raises(ValueError, match="spp_steps"):
        make_model("pcfm", SMALL, resolution="accurate", spp_steps=12)
    with pytest.raises(ValueError, match="spp_steps"):
        make_model("pcfm_h3_raman", SMALL, resolution="accurate", spp_steps=12)
    with pytest.raises(ValueError, match="raman_steps"):
        make_model("pcfm_h3_raman", SMALL, resolution="accurate", raman_steps=8)


@pytest.mark.parametrize("model_name", ["pcfm", "pcfm_h3_raman"])
def test_unknown_resolution_profile_is_rejected(model_name: str) -> None:
    """未知档位直接报错；拼错的档位不能被当成"默认"。"""
    with pytest.raises(ValueError, match="resolution"):
        make_model(model_name, SMALL, resolution="turbo")
