# Coherent PCFM2 and optional model selection

`pcfm2` implements the polynomial span-pair calculation in Y. Gao, Y. Jiang and
P. Poggiolini, **The Coherent Polynomials Closed-Form Model for Evaluating
Nonlinear Interference in Any Island**, arXiv:2602.03860v2. The implementation
uses the supplied author LaTeX and original paper. It implements GN, with no
machine-learned EGN correction.

Use `pcfm2` when full island support, in-band PSD variation, low dispersion, or
frequency-dependent coherence matters enough to justify its cost. Retain
`pcfm_h3_raman` for large routine workloads where its approximations have been
checked. `pcfm_auto` is an opt-in computational policy between these models; it
does not change any application default.

## What changes relative to existing models

| Property                      | H3 Raman                                          | PCFM2                                                         | Existing cGN                                                    |
| ----------------------------- | ------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- |
| Main NLI kernel               | Specialized polynomial SCI/XCI expressions        | Polynomial overlap and analytical span-pair moments           | Correlation-domain numerical integration                        |
| Receiver-frequency dependence | Primarily channel-level approximations            | Arbitrary CUT observation frequencies and in-band integration | Optional in-band integration                                    |
| General MCI islands           | Optional repository dense-grid/FWM approximations | Every supported channel triple                                | Every supported channel triple                                  |
| General-route coherence       | Visibility/equivalent scalar amplitudes           | Frequency-dependent span pairs                                | Frequency-dependent span pairs                                  |
| Exactly zero dispersion       | Legacy divisions make this unsupported            | Explicit finite limiting kernels                              | Supported                                                       |
| Pumped profiles and ASE       | Shared coupled Raman solver                       | Same solver, integrated as a repository extension             | Active pumps unsupported                                        |
| Discontinuous in-span loss    | Segmented profiles                                | Unsupported                                                   | Unsupported                                                     |
| EGN correction                | Optional fitted correction                        | GN only                                                       | GN only                                                         |
| Typical computational role    | Large repeated workloads                          | Analytical GN reference and bounded detailed studies          | Fast independent full-island alternative in supported scenarios |

The paper's contribution implemented here is the closed-form polynomial
span-pair kernel for arbitrary island rectangles, including coherent buildup and
the primitive-moment machinery. Full-island support and general-route coherence
already exist in the repository's cGN model; those capabilities are not new to
this repository. Adaptive precision, cache organization, route integration,
Raman front end, and budget dispatch are implementation choices. This work does
not assert a new research method beyond the cited paper.

## Implemented calculation

For every observation frequency in a CUT, the three rectangular channel supports
define an SCI, XCI, or MCI island. The implementation enumerates every supported
channel triple, including permutations through the first/second-wave symmetry
factor. A midpoint staircase covers each island. Additional splits at support
breakpoints make its total area exact; the response-weighted integral still
requires convergence with `island_cells`.

For each triple and each physical span, the code fits the effective profile

```text
p_x(u) = sqrt(p_first(u) * p_second(u) * p_third(u) / p_CUT(u))
```

using normalized position `u` in `[0, 1]`. It fits this ratio directly, rather
than forming it from separately fitted channel polynomials. The nonlinear
coefficient uses the arithmetic mean of the four channel effective areas.
Dispersion includes beta2 and beta3 about the fiber reference wavelength and is
frozen at each rectangle's center. Beta4 is zero.

The analytical rectangle response is

```text
F(t) = integral_rectangle cos(4*pi**2*x*y*t) dx dy
     = [-Si(4*pi**2*a*d*t) + Si(4*pi**2*a*c*t)
        -Si(4*pi**2*b*c*t) + Si(4*pi**2*b*d*t)] / (4*pi**2*t).
F(0) = (b-a)*(d-c).
```

The polynomial span-pair integral uses the paper's piecewise polynomial overlap
and primitive moments. `J(X)` uses the generalized hypergeometric expression for
moderate arguments and an asymptotic remainder for large arguments. Small
arguments use convergent series. A private 50-digit `mpmath` context protects
cancellations before conversion to float64. Precision increases automatically
with the dispersion-scale ratio and polynomial degree for nearly degenerate span
pairs. The context is local to the call, and cache keys include precision. There
is no longitudinal or frequency quadrature inside a rectangle. Overlap
polynomials, primitive moments, and pair results have bounded caches.

Signed dispersion uses ordered transformed limits and absolute Jacobians.
Exactly zero dispersion uses direct limiting expressions. Self terms appear
once, and each cross-span real term appears with factor two. The accumulated
dispersion includes every traversed span, including spans with no local
stimulus. An incoherent option retains only self terms.

Full-route evaluation collects the actual launches after shared amplification
and ROADM processing. Every span contribution uses its actual transfer to the
receiver and downstream filtering penalty. This retains heterogeneous fiber,
unequal launch, fixed gain, passive propagation, and ROADM effects in the
span-pair calculation.

`spm_noise_w` contains SCI. `xpm_noise_w` contains all non-SCI NLI, including
MCI, under the existing result contract. Noise is integrated through a
rectangular matched receiver filter; `band_points=1` instead uses center PSD
times bandwidth.

## Usage and configuration

```python
import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

grid = SpectrumGrid(3, 50e9, 193.1e12)
fiber = FiberSpanConfig(length_km=80)
path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
model = create_model(
    "pcfm2",
    grid,
    fiber,
    symbol_rate_hz=32e9,
    poly_degree=5,
    island_cells=8,
    band_points=5,
    span_coherence="coherent",
    spp_steps=64,
    egn=False,
)
result = model.evaluate_path(path, np.full(3, 1e-3))
```

The same shared route options support flexible grids, connections,
`channel_bandwidth_source`, and `power_width_mode`. Spectra must be
non-overlapping rectangles. An arbitrary-frequency single-span diagnostic is
available through
`PCFM2SpanKernel.nli_psd(power, length_m, cut, observations_hz)`; its outputs
are SCI and non-SCI PSDs in W/Hz, with exact span-loss compensation. Every
observation must lie inside the selected CUT.

| Control            | Default      | Purpose                                                           |
| ------------------ | ------------ | ----------------------------------------------------------------- |
| `poly_degree`      | `5`          | Degree of the effective triple-profile fit; supported range 0–9   |
| `island_cells`     | `8`          | Uniform first-axis slices before support-breakpoint splits        |
| `band_points`      | `5`          | Gauss nodes across each CUT; 1 enables the flat-PSD approximation |
| `span_coherence`   | `"coherent"` | Explicit span pairs; `"incoherent"` keeps self terms              |
| `spp_steps`        | `64`         | RK4 steps of the shared coupled Raman profile solver              |
| `include_isrs`     | `True`       | Signal-to-signal Raman transfer                                   |
| `raman_ase`        | `True`       | Newly generated spontaneous Raman noise                           |
| `raman_gain_model` | `"gaussian"` | Shared Raman gain spectrum name or table-file path                |
| `raman_iterations` | `64`         | Backward-pump iteration limit; non-convergence raises             |
| `raman_backend`    | `"python"`   | Shared solver backend; also accepts `"numba"` and `"auto"`        |
| `egn`              | `False`      | GN only; `True` raises                                            |

For attenuation-only controls, `include_isrs=False` with no active pumps uses
the exact exponential profile before polynomial fitting. Distributed pumps use
the existing coupled signal/pump/ASE solver. This pump integration is a
repository extension: the PCFM2 paper supplies the NLI kernel, rather than these
pump and ASE equations. Raman ASE follows the shared amplifier and route
conventions. Pumps overlapping signal channels are rejected.

In-span lumped losses are rejected because a single global polynomial cannot
represent their discontinuities. Use H3's segmented profile implementation for
that scenario within its applicability limits. This version does not claim an
optimal polynomial fit or a universal error bound for strongly shaped Raman
profiles. `kernel.last_profile_error` records the largest sampled absolute fit
residual divided by that effective profile's sampled peak. Refine profiles
independently of island and receiver integration.

| Scenario                                        | Starting configuration                                  | Required refinement check                                                     |
| ----------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Few channels, one ordinary span                 | Defaults                                                | Double island and band resolution                                             |
| Low dispersion, SCI PSD or adjacent subcarriers | Degree 5, at least 8 island slices and 5 receiver nodes | Refine island and receiver grids; preserve MCI                                |
| Many coherent spans or dispersion reversal      | Coherent mode, more receiver nodes                      | Increase `band_points` and `island_cells` until oscillatory buildup converges |
| Strong Raman profile shaping                    | Degree 5–7, at least 64 SPP steps                       | Double SPP steps; compare degrees and sampled profile residual                |
| Large ordinary WDM comb                         | H3 with explicit resolution, or `pcfm_auto`             | Check H3 against a tractable PCFM2/reference subset                           |
| In-span lumped loss                             | H3 with its supported segmented solver                  | PCFM2 is unsupported                                                          |

There is no universally optimal configuration. For an accuracy target, refine
one axis at a time and then refine them together. A change below 0.01 dB in NLI
power can be a useful numerical convergence target; it is not a bound on GN
physical-model error.

## Optional system selection

```python
model = create_model(
    "pcfm_auto",
    grid,
    fiber,
    symbol_rate_hz=32e9,
    max_pcfm2_work=20000,
    pcfm2_options={"poly_degree": 5, "island_cells": 8, "band_points": 5},
    h3_options={"egn": False, "spp_steps": 32},
)
result = model.evaluate_path(path, np.full(3, 1e-3))
print(model.last_selection)
```

The policy estimates

```text
work = active_channels**4 * physical_spans**2 * band_points * island_cells
```

and chooses PCFM2 when this is within budget. This conservative operation proxy
is not a time prediction and omits polynomial degree, cache state, sparsity of
supported triples, and support-breakpoint splits. H3 handles larger routine
cases. Its policy default explicitly disables EGN and pins `spp_steps=32`;
callers may supply other H3 settings through `h3_options`. The two delegates
retain their own coherence approximations.

The policy checks the beta2/beta3 dispersion range across the spectrum. A zero
crossing or effective dispersion magnitude below 5 ps²/km requires PCFM2. If
such a case exceeds the budget, it raises with instructions to increase the
budget or select PCFM2 explicitly. In-span losses select H3, unless low
dispersion makes that combination unsupported by the policy. Pump/signal overlap
raises. Selection happens once per path.

`last_selection` contains `model`, `reason`, and `estimated_work`. It describes
a computational choice, not a measured winner or an accuracy certificate.
`pcfm_auto.kernel_for` is intentionally unavailable because a kernel depends on
the selected delegate. Direct `span_step` dispatches a standalone span.

Use `QoTValidator(..., coherent_assembly=True)` for route-wide span pairs when
the same carriers and nominal launch powers occupy every link. For varying
loads, the validator evaluates all spans of each link together and combines
their noise powers across the route. Default link assembly uses separate
one-span evaluations on implicitly split, gain-compensated links.

### Mixed-load coherence

Set `QoTValidator(..., mixed_load_coherence=True)` or run
`network-nlo-eval simulate --qot-model pcfm2 --mixed-load-coherence` to enable
cross-link span pairs for fixed- and flexible-grid admission. The option
requires `pcfm2` with `span_coherence="coherent"` and defaults to off. It can be
used independently of `coherent_assembly`.

The calculation uses each link's actual carrier spectrum and span launches. SCI,
XCI, and MCI terms accumulate coherently over consecutive links where the same
participating services retain their frequencies and bandwidths. A service that
joins, leaves and later rejoins, or replaces another service at the same
frequency begins a separate contribution. Power injected without a service
identity contributes within its link. Generated NLI follows the victim through
downstream gain, loss, and filtering, including after an interferer drops. ASE
and link validity retain their separate calculations.

PCFM2 evaluates additional span pairs for every affected receiver. Cost grows
with the number of supported mixing islands and spans over which they remain
coherent. The option is intended for workloads that can accommodate this
additional calculation.

## Validation and comparison scope

Independent tests check rectangle frequency integration, polynomial pair
integration, signed and zero dispersion, actual SCI strip area and in-band PSD,
eight-channel MCI, sparse loading, coherent/incoherent buildup, dispersion
reversal, passive segmentation, ROADM filtering/blocking, and agreement of
shared Raman signal/ASE conventions. A full-island exponential GN integral
independently checks single-span normalization and response. Network cache tests
exercise both registered models.

Temporary evaluation scripts and raw results are in `scratch/`. The
`scratch/evaluate_pcfm2.py` control study integrates the true frequency strips
with Gauss quadrature and the exact exponential longitudinal link. It increases
both observation and internal frequency resolution to check the reference. It
compares H3 with `egn=False`; a higher predicted SNR is not treated as an
accuracy improvement.

The controls use constant attenuation, constant beta2, zero beta3, 100 km spans,
0 dBm per channel, and no Raman transfer. They include a 140 GBaud SCI channel,
ten-span coherent SCI, and eight adjacent 18 GBaud subcarriers. Only the
leftmost subcarrier is evaluated in the eight-channel timing study. These
scenarios are inspired by the paper; its numerical Raman profiles are not
bundled, so these are not exact reproductions of its figures.

No waveform comparison has been established: the optional external
`physics_nlo_eval` package is unavailable in the evaluation environment. Results
against a converged GN reference assess implementation and approximation error,
not experimental transmission accuracy or EGN accuracy.

## Measured controls, November 16, 2026

The following values are signed **NLI power errors in dB**, relative to the
independent GN integral. They are not SNR errors. The largest change in that
reference between its two resolutions was 0.000152 dB. Default PCFM2 uses degree
5, 8 island slices, and 5 receiver nodes. Refined PCFM2 uses degree 7, 24
slices, and 16 receiver nodes, increased to 24 for the ten-span cases. H3 uses
GN, 32 SPP steps, and its PCFM coherence setting.

| Control                                     | Default PCFM2 | Refined PCFM2 |       H3 GN |
| ------------------------------------------- | ------------: | ------------: | ----------: |
| 140 GBaud, one span, zero dispersion        |     +0.000521 |     +0.000013 | Unsupported |
| 140 GBaud, one span, 1 ps²/km               |     −0.008510 |     −0.001053 |   +1.286461 |
| 140 GBaud, one span, 5 ps²/km               |     −0.004621 |     −0.001539 |   +0.854877 |
| 140 GBaud, one span, 22 ps²/km              |     −0.016757 |     −0.002317 |   +0.603367 |
| 140 GBaud, ten coherent spans, 1 ps²/km     |     −0.025741 |     −0.002640 |   +1.286059 |
| 140 GBaud, ten coherent spans, 5 ps²/km     |     −0.038246 |     −0.004865 |   +0.855105 |
| 140 GBaud, ten coherent spans, 22 ps²/km    |     −0.015344 |     −0.005141 |   +0.602653 |
| Eight 18 GBaud subcarriers, zero dispersion |     +0.000521 |     +0.000013 | Unsupported |
| Eight 18 GBaud subcarriers, 5 ps²/km        |     −0.000861 |     −0.000131 |   −0.356725 |
| Eight 18 GBaud subcarriers, 22 ps²/km       |     +0.000589 |     −0.000322 |   +0.126898 |

For the single-span SCI controls, these default PCFM2 settings were sufficient
for a 0.02 dB numerical target. Across the ten-span controls, default PCFM2
stayed within 0.04 dB and the refined configuration stayed within 0.006 dB. The
eight-subcarrier controls did not need the refined settings for a 0.001 dB
target on total NLI. These are measured starting configurations for these
controls, not prescriptions for other baud rates, profiles, or routes.

Runtime is the cost of retaining this detail. Observed single-span, nonzero
dispersion SCI evaluations took 0.15–0.31 seconds with default PCFM2 and 2.8–4.2
seconds when refined. Ten-span evaluations took 5.2–9.2 seconds and 106–186
seconds respectively. H3 took approximately 0.001–0.016 seconds for the
single-span cases and 0.039–0.079 seconds for ten spans. These are exploratory
timings including construction, with mixed JIT/cache state, not statistically
controlled throughput benchmarks.

Additional GN baselines on the same one-span, 140 GBaud controls give:

| Model/settings                                | Error at 5 ps²/km | Error at 22 ps²/km |
| --------------------------------------------- | ----------------: | -----------------: |
| cGN, 64 correlation nodes / 16 receiver nodes |         +0.002650 |          −0.008420 |
| Historical PCFM degree-five preset            |         +0.851496 |          +0.609604 |
| Semrau 2019, GN / Raman slope zero            |         +0.753147 |          +0.564061 |
| Buglia 2023, GN                               |         +0.803597 |          +0.637227 |
| ECZ                                           |         +0.753147 |          +0.564061 |

cGN also gave +0.000652 and −0.000186 dB in the eight-subcarrier controls, with
about 0.16 seconds to evaluate all eight channels. PCFM2 default took 8.7–13.7
seconds to evaluate just the leftmost subcarrier. Thus PCFM2 does not establish
a performance advantage over cGN in these unpumped controls. The analytical
polynomial span-pair representation and integration with active Raman profiles
are reasons to use PCFM2; cGN remains a useful independent alternative where
supported.

H3's `dense_grid_fix` and separate `dense_grid_fwm_fix` options did not change
the eight-subcarrier outputs in these configurations. Increasing H3 to degree
five and 64 SPP steps gave −0.326640 and +0.154618 dB, respectively, so profile
refinement alone did not remove that discrepancy.

Inspect SCI and non-SCI separately: their errors can cancel. At 5 ps²/km in the
eight-subcarrier control, H3's SCI error was +1.740 dB while non-SCI was −0.582
dB, producing the smaller −0.357 dB total error. PCFM2's arbitrary
observation-frequency API also exposes the in-band PSD shape directly, instead
of treating integrated power divided by bandwidth as the PSD.

For the 140 GBaud, 5 ps²/km SCI shape check, degree-five PCFM2 with 16 island
slices was within 0.20% of independently integrated PSD at all 41 sampled
observation frequencies. It resolves the substantial decrease toward the channel
edges that a flat-PSD power convention does not represent.

Reproducible local evidence is generated by:

```bash
.venv/bin/python scratch/evaluate_pcfm2.py
.venv/bin/python scratch/compare_pcfm2_baselines.py
.venv/bin/python scratch/pcfm2_figures.py --data
```

Scripts and raw artifacts are intentionally temporary and ignored by Git. The
quantitative tables above preserve the findings in the tracked documentation.
The raw files include `comparisons.json`, `other_baselines.json`, and
`psd_shape.json` under `scratch/pcfm2_evaluation/`. The plotting mode of
`pcfm2_figures.py` needs Matplotlib and produces standalone PNG/SVG files in the
same directory.
