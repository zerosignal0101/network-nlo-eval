"""Reference and invariant tests for the Semrau 2019 kernels."""

from typing import TypedDict

import numpy as np
import pytest

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.semrau2019_kernels import (
    SemrauMatrices,
    build_semrau_matrices,
    gn_weighted_spm_visibility,
    semrau_span_jit,
)


class VisibilityInputs(TypedDict):
    bandwidth_hz: NDArrayFloat
    alpha_npm: NDArrayFloat
    span_length_m: float
    span_separations_m: NDArrayFloat


def kernel_parameters() -> tuple[
    NDArrayFloat, NDArrayFloat, NDArrayFloat, SemrauMatrices
]:
    relative = np.array([-50e9, 0, 50e9])
    bandwidth = np.array([32e9, 48e9, 64e9])
    attenuation = np.full(3, 0.2 / 4.342944819032518 / 1000)
    matrices = build_semrau_matrices(
        f_rel_hz=relative,
        lambdas_m=C_LIGHT / (193.1e12 + relative),
        b_hz=bandwidth,
        alpha_npm=attenuation,
        a_eff_m2=np.array([80e-12, 81e-12, 82e-12]),
        beta2_s2_m=-21.7e-27,
        beta3_s3_m=0.12e-39,
        n2_m2_w=2.6e-20,
        with_egn=True,
    )
    return relative, bandwidth, attenuation, matrices


def test_multispan_kernel_matches_reference_and_per_span_normalization() -> None:
    relative, bandwidth, attenuation, mats = kernel_parameters()
    args = (
        np.array([1e-3, 2e-3, 0.5e-3]),
        80e3,
        5,
        0.028 / 1e3 / 1e12,
        -0.68,
        0.1,
    )
    remaining = (
        attenuation,
        relative,
        bandwidth,
        mats.spm_u,
        mats.spm_v,
        mats.spm_coef,
        mats.mat_u,
        mats.mat_v,
        mats.mat_w,
    )
    total = semrau_span_jit(*args, False, *remaining)
    per_span = semrau_span_jit(*args, True, *remaining)
    expected = (
        [2.5121097701423184e-5, 5.023698407799246e-5, 1.2557943323419592e-5],
        [1.459012028773238e-6, 7.5837372406777165e-6, 8.10132926198528e-8],
        [7.844474667275055e-7, 6.536840960825103e-7, 4.598773173884558e-7],
    )
    for actual, reference in zip(total, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(total[0], per_span[0])
    np.testing.assert_allclose(total[1], per_span[1] * 5, rtol=1e-12)
    np.testing.assert_allclose(total[2], per_span[2] * 5, rtol=1e-12)
    for matrix in (mats.mat_u, mats.mat_v, mats.mat_w):
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(3))


def test_gn_visibility_reference_and_zero_dispersion() -> None:
    _, bandwidth, attenuation, _ = kernel_parameters()
    kwargs: VisibilityInputs = {
        "bandwidth_hz": bandwidth,
        "alpha_npm": attenuation,
        "span_length_m": 80e3,
        "span_separations_m": np.array([0, 80e3, 160e3]),
    }
    visibility = gn_weighted_spm_visibility(beta2_abs_s2_m=21.7e-27, **kwargs)
    expected = [
        [1.0, 1.0, 1.0],
        [0.20549556628420695, 0.15155143178231636, 0.12845647182710188],
        [0.08464600843949929, 0.059146241916821586, 0.04835734735031591],
    ]
    np.testing.assert_allclose(visibility, expected, rtol=1e-12)
    np.testing.assert_array_equal(
        gn_weighted_spm_visibility(beta2_abs_s2_m=0, **kwargs), np.ones((3, 3))
    )
    kwargs["span_length_m"] = 0
    with pytest.raises(ValueError, match="span_length_m must be positive"):
        gn_weighted_spm_visibility(beta2_abs_s2_m=0, **kwargs)
