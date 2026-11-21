"""Focused integration tests for the ISRS-GN physical model."""

import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    available_models,
    create_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
    ROADMConfig,
    ROADMOutputTarget,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "ecz",
        "ecz_coherent",
        "cgn",
        "semrau2019",
        "buglia2023",
        "damico2022",
        "carena2014",
        "numerical",
        "pcfm",
        "pcfm_original",
        "raman_lumped",
    ],
)
@pytest.mark.parametrize("pumped_reference", [True, False])
def test_ecz_front_ends_reject_pumps_in_reference_and_route_fibers(
    model_name: str, pumped_reference: bool
) -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=80)
    pumped = fiber.model_copy(
        update={"raman_pumps": [RamanPumpConfig(wavelength_nm=1450, power_w=0.2)]}
    )
    if pumped_reference:
        with pytest.raises(ValueError, match="active Raman pumps"):
            create_model(model_name, grid, pumped, 32e9)
    else:
        model = create_model(model_name, grid, fiber, 32e9)
        with pytest.raises(ValueError, match="active Raman pumps"):
            model.evaluate_path(
                MultiSpanOpticalPath([pumped], [EDFAConfig()]), np.full(2, 1e-3)
            )


@pytest.mark.parametrize("model_name", ["ecz", "ecz_coherent", "cgn"])
def test_ecz_front_ends_allow_inactive_pumps(model_name: str) -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=80)
    inactive = fiber.model_copy(
        update={"raman_pumps": [RamanPumpConfig(wavelength_nm=1450, power_w=0.0)]}
    )
    power = np.full(2, 1e-3)
    a = create_model(model_name, grid, fiber, 32e9).evaluate_path(
        MultiSpanOpticalPath([fiber], [EDFAConfig()]), power
    )
    b = create_model(model_name, grid, inactive, 32e9).evaluate_path(
        MultiSpanOpticalPath([inactive], [EDFAConfig()]), power
    )
    for field in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "snr_db"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_multispan_path_matches_reference_vector() -> None:
    fiber = FiberSpanConfig(length_km=120, max_span_length_km=80)
    model = MultiBandISRSGN(
        SpectrumGrid(4, 50e9, 193.1e12), fiber, 32e9, ode_accel="off"
    )
    path = MultiSpanOpticalPath(
        [fiber], [EDFAConfig(target_gain_db=17, noise_figure_db=5.5)], [ROADMConfig()]
    )
    result = model.evaluate_path(path, np.array([0.001, 0.002, 0.0, 0.0005]))
    expected = {
        "signal_power_w": [
            0.010002164551407077,
            0.01999999959844569,
            0.0,
            0.004997835850147171,
        ],
        "spm_noise_w": [
            6.664562497431326e-06,
            5.3388864420362666e-05,
            0.0,
            8.364677879757955e-07,
        ],
        "xpm_noise_w": [
            1.0624168882882526e-05,
            5.869639910858695e-06,
            0.0,
            3.1168248964237126e-06,
        ],
        "ase_noise_w": [
            6.720871581488242e-06,
            6.721943533669781e-06,
            0.0,
            6.7240866248934956e-06,
        ],
        "snr_db": [26.197090167564603, 24.816147278024484, -np.inf, 26.703173168540676],
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(getattr(result, name), reference, rtol=1e-12, atol=0)
    assert result.span_count == 2
    assert result.roadm_count == 1
    assert "ecz" in available_models()
    assert isinstance(create_model("ecz", model.grid, fiber), MultiBandISRSGN)


def test_total_and_per_polarization_are_physically_equivalent() -> None:
    fiber = FiberSpanConfig(length_km=80)
    grid = SpectrumGrid(4, 50e9, 193.1e12)
    path = MultiSpanOpticalPath(
        [fiber], [EDFAConfig()], [ROADMConfig(equalize_output_power=True)]
    )
    total = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention="total")
    per_pol = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention="per_pol")
    a = total.evaluate_path(path, np.full(4, 1e-3))
    b = per_pol.evaluate_path(path, np.full(4, 0.5e-3))
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"):
        np.testing.assert_array_equal(getattr(a, name), 2 * getattr(b, name))
    np.testing.assert_array_equal(a.snr_db, b.snr_db)


def test_flexible_bandwidth_blocking_and_fiber_cache() -> None:
    grid = FlexibleGrid.from_legacy_grid(2, 50e9, 193.1e12)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=i,
            first_slot=i * 4,
            n_slots=4,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=25e9,
            roll_off=0,
            launch_power_w=1e-3,
        )
        for i in range(2)
    ]
    fiber = FiberSpanConfig(length_km=80)
    model = MultiBandISRSGN(
        grid,
        fiber,
        50e9,
        connections=connections,
        slots_per_channel=4,
        channel_bandwidth_source="occupied",
    )
    np.testing.assert_array_equal(model.channel_bandwidth_hz_channels, [50e9, 50e9])
    assert model.blocked_nodes(ROADMConfig(blocked_slots=[1])) == [0]
    assert model.blocked_nodes(ROADMConfig(blocked_channels=[1])) == [1]
    child = model.for_fiber(fiber)
    assert child is model.for_fiber(fiber)
    assert child.connections == connections


@pytest.mark.parametrize(
    "edfa",
    [
        EDFAConfig(target_gain_db=17, noise_figure_db=5.5),
        EDFAConfig(
            target_gain_db=17, noise_figure_db=5.5, max_output_signal_power_w=0.025
        ),
        EDFAConfig.model_validate(
            {
                "target_gain_db": 17,
                "noise_figure_db": 5.5,
                "max_output_signal_power_w": 0.025,
                "compression_curve": [
                    {"backoff_db": 0},
                    {
                        "backoff_db": 5,
                        "noise_figure_penalty_db": 1,
                        "tilt_db_per_thz": -0.5,
                    },
                ],
            }
        ),
        EDFAConfig(
            target_gain_db=17,
            passbands_hz=((191.9e12, 192.3e12), (193.2e12, 194.3e12)),
            input_voa_db=0.5,
            output_voa_db=2,
            max_output_signal_power_w=0.025,
        ),
    ],
)
def test_cached_and_batched_models_match_pairwise(edfa: EDFAConfig) -> None:
    fiber = FiberSpanConfig(length_km=80)
    grid = SpectrumGrid(48, 50e9, 193.1e12)
    power = np.full(48, 1e-3)
    off = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off")
    cached = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="cached")
    matvec = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="matvec")
    reference = off._calc_span_noise_and_power(power, 80e3, edfa)
    cached._calc_span_noise_and_power(power, 80e3, edfa)
    hit = cached._calc_span_noise_and_power(power, 80e3, edfa)
    for array in hit:
        array[:] = -1
    again = cached._calc_span_noise_and_power(power, 80e3, edfa)
    batch = matvec.calc_span_noise_and_power_batch(
        np.stack([power, np.zeros(48), power / 4]), 80e3, edfa
    )
    sparse_reference = off._calc_span_noise_and_power(power / 4, 80e3, edfa)
    for expected, actual, accelerated, sparse in zip(
        reference, again, batch, sparse_reference, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(accelerated[0], expected, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(accelerated[1], np.zeros(48))
        np.testing.assert_allclose(accelerated[2], sparse, rtol=1e-12, atol=0)
    if (output_limit := edfa.max_output_signal_power_w) is not None:
        stage_limit = output_limit * 10 ** (-edfa.output_voa_db / 10)
        assert batch[0][0].sum() == pytest.approx(stage_limit)
        assert batch[0][2].sum() < stage_limit
    assert matvec._p1_xpm_matrices is not None
    with pytest.raises(ValueError, match="requires ode_accel"):
        off.calc_span_noise_and_power_batch(power[None, :], 80e3)


def test_batched_ecz_applies_passive_roadm_psd_target_per_channel() -> None:
    fiber = FiberSpanConfig(length_km=80)
    model = MultiBandISRSGN(SpectrumGrid(2, 50e9, 193.1e12), fiber, 32e9)
    edfa = EDFAConfig()
    roadm = ROADMConfig(
        insertion_loss_db=10, output_target=ROADMOutputTarget(psd_w_per_hz=8e-15)
    )
    power = np.array([[4e-3, 1e-3], [2.5e-4, 0]])
    batch = model.calc_span_noise_and_power_batch(power, 80e3, edfa, roadm)
    np.testing.assert_allclose(batch[0], [[2.56e-4, 1e-4], [2.5e-5, 0]])
    for index, launch in enumerate(power):
        serial = model._calc_span_noise_and_power(launch, 80e3, edfa, roadm)
        for row, expected in zip(batch, serial, strict=True):
            np.testing.assert_allclose(row[index], expected, rtol=1e-12, atol=0)
