# QoT model correspondence to the literature

Audit date: November 16, 2026. This inventory covers all 18 registered model
entries, their primary equations, their implemented variants, and their
supported scenarios. The implementation and its public descriptions were checked
against original papers, with the PDF or author LaTeX taking precedence over
extracted text and translations. This is a formula and implementation audit; it
does not reproduce every transmission experiment or accuracy claim in the
papers.

The literature collection contains 14 relevant supplied papers and one
supplementary primary cGN paper, with PDFs, original text, figures, and Chinese
translations. The collection and its source metadata are maintained outside the
repository. No workstation-specific literature location is required by the code.

## Registered models

| Model key            | Primary sources               | Implemented calculation and correspondence                                                                                         |
| -------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `ecz`                | R1, R2                        | Lasagni's generalized Raman shaping profile and single-span SPM/XPM closed forms; route NLI accumulates incoherently.              |
| `ecz_coherent`       | R1, R2, R4                    | ECZ single-span physics with the repository's GN-weighted visibility and equivalent scalar phase ramp.                             |
| `semrau2019`         | R2, R3                        | Closed-form ISRS-GN with the optional 2019 modulation-format correction; repository visibility controls route coherence.           |
| `buglia2023`         | R5                            | GN contribution of the finite-loss, finite-length closed forms; attenuation-based profile parameters and repository coherence.     |
| `cgn`                | R6; R1 for the ISRS front end | Semi-analytical correlation GN, SCI/XCI/MCI island covers, and coherent span-pair accumulation.                                    |
| `damico2022`         | R7; R1 for the ISRS front end | Disaggregated single-channel/cross-channel calculation, segmented XC approximation, and a repository SC coherence envelope.        |
| `carena2014`         | R8; R1 for the GN baseline    | Numerically integrated EGN corrections transferred as EGN/GN ratios to the SRS-aware GN baseline.                                  |
| `pcfm`               | R9, R10                       | Polynomial spatial-power-profile GN kernels, optional fitted Zefreh format correction, and repository coherence/edge extensions.   |
| `pcfm_original`      | R9                            | Historical degree-five GN preset without the later edge correction or span cache; inherits the parent normalization and coherence. |
| `raman_lumped`       | R9                            | Retained unpumped PCFM compatibility preset. Active pumps require a distributed-Raman model.                                       |
| `buglia2024_ra_gn`   | R11                           | Five-parameter Raman profile fitted to a coupled ODE, closed SPM/XPM terms, and paper-style power accumulation.                    |
| `hybrid_raman_paper` | R12                           | Coupled signal/pump/ASE propagation, fitted longitudinal profiles, and hybrid-amplified GN formulas.                               |
| `pcfm_h3_raman`      | R9, R10, R12                  | Repository extension: degree-three PCFM/H3 kernels driven by coupled Raman profiles, with optional dense-grid corrections.         |
| `pcfm2`              | R15; R12 for Raman profiles   | Analytical polynomial span-pair moments over SCI/XCI/MCI island covers, with receiver-band integration and actual route transfer.  |
| `pcfm_auto`          | Repository policy             | Opt-in dispatch between PCFM2 and H3 by supported physics and an explicit work budget; no accuracy guarantee.                      |
| `zhang2024_dra_egn`  | R13                           | Zhang-inspired coherent **GN** calculation with fitted profiles, numerical cross-span integration, and repository calibration.     |
| `numerical`          | R4                            | Approximate numerical ISRS-GN: ODE profiles, a band-averaged frequency response, and repository coherence.                         |
| `numerical_dra`      | R4, R11, R12                  | Independent adaptive Raman ODE and numerical longitudinal/frequency integration with incoherent route accumulation.                |

These are 18 selectable entries, rather than 18 independent paper derivations.
Several entries are presets or repository extensions. The historical `_egn`
suffix of the Zhang key does not enable a complete EGN calculation.

## Common physical conventions and scenario checks

`SpanFiberOutput.power_out_w` is the fiber-output signal. Its NLI powers and
complex amplitudes use exact per-channel span-loss compensation as their
normalization. The element layer converts NLI to the actual amplifier output
using

```text
T_NLI = Gamma * P_CUT(L) / P_CUT(0)
NLI_output = T_NLI * NLI_compensated
amplitude_output = sqrt(T_NLI) * amplitude_compensated
```

This supplies the outer transfer factor in PCFM Eqs. (21) and (26). Automatic
per-channel loss compensation gives a unit factor. Fixed gain, gain ripple, and
an unamplified span use their actual transfer. Subsequent elements transfer
previously generated noise and newly generated noise to the same receiver plane.
The ECZ batch interface follows the same convention. Complex amplitudes retain
squared magnitudes equal to their corresponding NLI powers.

Fiber `D` and `S` refer to `reference_wavelength_nm`. The ECZ, Semrau, and
Buglia implementations translate their beta2/beta3 Taylor expansion to the
spectrum's frequency origin. Buglia builds attenuation, effective area,
nonlinearity, and dispersion from each actual route fiber. cGN retains the
reference-frequency shift explicitly in each island's phase calculation.

Active Raman pumps require a distributed-Raman implementation. Unpumped models
reject them both in their reference fiber and in fibers encountered later in a
route. The supported pumped model families are `buglia2024_ra_gn`,
`hybrid_raman_paper`, `pcfm_h3_raman`, `pcfm2`, `zhang2024_dra_egn`, and
`numerical_dra`, subject to their individual pump-direction and solver
restrictions. ECZ launch pre-emphasis is dimensionless and is set to zero.

In-span discrete losses require `pcfm` or `pcfm_h3_raman`; the H3 Raman backend
must also support the configured losses. Other registered models reject such
fiber configurations. ROADM losses and amplifier settings continue to use the
shared element and route interfaces.

The `pcfm_auto` wrapper delegates supported in-span loss cases to H3 and
otherwise follows the selected implementation's restrictions.

## Formula correspondence and model variants

### ECZ and coherent ECZ

Lasagni R1 Eq. (18) supplies the piecewise Raman shaping function, Eq. (20) the
channel power transfer, and Eqs. (29)-(32) the SPM/XPM terms. R2 provides their
closed-form ISRS-GN background. The shaping approximation assumes uniform
launched spectral density; sparse loading and unequal channel powers extend that
approximation beyond the paper's uniform-PSD derivation.

The `ecz` route evaluates each span and sums transferred noise powers. The
`ecz_coherent` variant averages SPM coherence with GN noise power and maps its
total buildup to an equivalent phase ramp. XPM uses a band visibility
approximation. That scalar representation is a repository model: its exact
matching of an integrated homogeneous-link total does not preserve every
frequency-dependent pairwise phase in a heterogeneous or unequally weighted
link. It is not the full coherent ISRS-GN integral in R4.

### Semrau 2019

The GN terms follow R2 Eqs. (10)/(11), with the implemented choice
`alpha_bar = alpha`. The optional modulation correction follows R3 Eq. (16),
including its single-span and asymptotic multi-span terms. The total span count
is available to that correction. The separate `paper_eq16_total` calculation
provides the power-only paper expression for comparison.

Route accumulation adds the repository visibility model to these single-span
terms, including its treatment of corrected XPM. The scalar coherence treatment
is an extension to the paper calculation. The no-Raman limit is checked against
ECZ with its Raman slope disabled.

### Buglia 2023

The implemented GN XPM and SPM terms correspond to R5 Eqs. (17)/(18), with the
matched longitudinal expression of Eq. (30). The paper also derives
modulation-format corrections in Eqs. (2), (8), and (19); this registered entry
selects the GN contribution and rejects `egn=True`.

Profile parameters use channel attenuation directly rather than a fitted Raman
power profile. Output power uses the ECZ shaping approximation. The route uses
individual-span evaluation and the repository GN-weighted phase ramp. These
choices define a GN approximation based on R5, rather than a reproduction of all
of its fitted-profile and modulation-correction results.

### Correlation GN

R6 Eq. (5) defines the propagation-constant expansion; the implemented physical
default keeps beta2 and beta3 about the fiber reference and sets beta4 to zero.
Eq. (6) defines the nonlinear stimulus profile (NSP): the square root of the
product of the three interacting power profiles divided by the CUT profile. Both
CUT-first and MCI islands use this convention, including the segmented ISRS
residuals.

The frequency integrals use midpoint rectangular covers and one-dimensional
frequency-product integration. SCI, XCI, and MCI are included in both center-PSD
and in-band modes, including zero dispersion. MCI slot enumeration retains its
geometrically valid candidates without a fixed slot-count cutoff. The
single-span correlation kernels retain a finite attenuation-based integration
window. `correlation_points`, `island_cells`, `isrs_segments`, and `band_points`
control different numerical approximations and should be refined separately.

For identical, power-balanced spans, the pointwise Dirichlet factor gives the
coherent total efficiently. General routes instead integrate the coherent sum of
the span fields inside each frequency-product integral. This retains the SS and
XS contributions of the span-pair decomposition, each span's own phase and
launch PSD, and downstream signal transfer and ROADM noise penalties. Adaptive
integration makes this general calculation more expensive.

The ISRS front end remains the repository's segmented unpumped profile
approximation. General Raman-amplified profiles, lumped in-span elements, beta4,
and arbitrary subcarrier spectra of the full paper framework are outside this
implementation's supported scenario. Public diagnostic convention options other
than `dispersion_convention="paper"` and `nsp_convention="gn"` are comparison
calculations; the physical correspondence above applies to those defaults.

### D'Amico 2022

R7 Eqs. (24)/(25) split SC and XC noise; Eqs. (28)/(29) supply the segmented XC
approximation. The paper itself uses a separate SC estimator, so the use of a
standard closed-form SC term is consistent with that architecture.

This implementation uses ECZ power propagation instead of the paper's coupled
SRS solver, and constructs a band-averaged phase-mismatch envelope for SC
coherence rather than importing a fitted `C_inf` result. The default pairwise XC
aggregation and optional band-centroid aggregation are distinct approximation
levels. Four-wave mixing is outside this model's calculation. Its scalar phase
ramp has the same homogeneous-link interpretation as the other visibility
extensions.

### Carena 2014

The SCI correction retains the distinct fourth-order integration domains with
coefficients `80/81` and `16/81`. These are separate integrals, rather than
interchangeable prefactors. XCI and adjacent three-channel MCI corrections are
also calculated numerically.

R8 assumes equally spaced channels, uniform channel properties, identical spans,
and no Raman power transfer. The repository applies ratios of its EGN and GN
calculations to an ECZ SRS-aware baseline. It also uses finite neighbor windows,
a limited adjacent-channel MCI selection, representative offsets for irregular
grids, and scalar route coherence. It therefore implements an adapted EGN
correction, not every MCI domain in the paper's Appendix B.

### PCFM and its presets

R9 Sections IV-B/IV-C supply the normalized spatial power profile, its
polynomial fit in Eq. (23), and the XCI/SCI kernels in Eqs. (24)/(30). The
kernel and route together apply the outer amplifier/CUT transfer of Eqs.
(21)/(26). Beta4 is zero, as no fourth-order fiber coefficient is exposed by the
current configuration. The associated error depends on fiber and frequency
range.

The optional fitted EGN factor uses all 24 coefficients from R10 Table IV,
consistent with R9's reuse of that fit. The implemented preset fixes Gaussian
modulation parameter `Phi=0` and zero roll-off for that factor; it is not a
general modulation-format selector. Repository visibility, edge corrections,
dense-grid/FWM options, and validity/error-budget heuristics are additional
choices, each controlled by the corresponding option.

`pcfm_original` fixes degree five, GN, no edge correction, and no span cache. It
inherits the parent's normalization and coherence. `raman_lumped` is retained as
an unpumped compatibility preset and rejects active pump scenarios. Coupled
Raman propagation is supplied by the separate distributed-Raman models.

### Coherent PCFM2

R15 Sections II/III and Appendix A supply the arbitrary-island GN integral,
effective triple profile, polynomial overlap reduction, and primitive moments
for every self/cross span pair. `pcfm2` applies the 16/27 GN prefactor, single
self terms and doubled real cross terms, signed-dispersion Jacobians, and exact
zero-dispersion limits. Private adaptive precision, starting at 50 digits,
protects primitive cancellation. Island tiling, polynomial fitting, and
receiver-frequency quadrature remain controlled approximations. Beta3 is frozen
at each rectangle center; beta4 is zero.

The shared R12 Raman profile/ASE solver is a repository integration.
Discontinuous in-span losses are unsupported. Non-SCI output includes XCI and
MCI. Full-route evaluation retains actual launches, receiver transfer, and
downstream ROADM filtering. No fitted EGN factor is applied.

`pcfm_auto` is an explicit computational policy, with an inspectable decision
and work estimate. It selects PCFM2 for bounded workloads and H3 for larger
routine ones, and refuses an over-budget low-dispersion fallback. This is not an
accuracy certification. See [the PCFM2 guide](pcfm2.md) for implementation
limits, controls, and independent comparison evidence.

### Raman-amplified analytical models

`buglia2024_ra_gn` fits the five-parameter profile of R11 Eq. (2) to coupled
Raman propagation, evaluates the Eq. (9)/(10) link and Eq. (11)/(12) SPM/XPM
terms, and uses Eq. (4) power accumulation. The sine sign follows the complex
longitudinal integral. The separated terms of Eq. (12) are consistent with the
Appendix D Eq. (56) derivation. Independent numerical integration checks these
algebraic forms. The span coherence exponent is configurable; its default zero
is a chosen incoherent setting, not a new fit to the paper's experiments.

`hybrid_raman_paper` uses the signal/pump/ASE equations of R12 Eqs. (2)/(4), the
profile representation in Eq. (10), the NLI expressions in Eqs. (8)/(9), and the
power accumulation of Eq. (6). Its default `nli_form="paper"` is the paper-style
calculation. Its numerical-integral and legacy PCFM choices are separate
controls. Its default coherence exponent is also zero.

`pcfm_h3_raman` combines a degree-three/H3 polynomial calculation with coupled
signal, pump, and ASE propagation. It is a repository extension based on R9,
R10, and R12, rather than a separately published analytical model. Dense overlap
handling, displaced-third-wave terms, EGN auto-selection, backend choices, and
uncertainty margins belong to this implementation. Their empirical margins are
not universal error bounds from the source papers.

The Raman ODE uses frequency ratios consistent with photon conservation for
stimulated power transfer. Tests check lossless forward photon balance,
polarization invariance, and Python/Numba agreement.

### Zhang 2024

R13 Eq. (1) specifies the three-exponential backward-Raman profile and Eq. (2)
its long-span link approximation. Eq. (3) contains GN terms and an XPM format
correction; Eqs. (4)/(5) retain explicit coherent inter-span phase terms.

The paper does not publish all derived closed expressions or its calibrated
coefficients/data. The repository fits its own profiles, evaluates analogous
same-span GN terms and numerical coherent cross-span terms, and loads its own
calibration JSON. This is a Zhang-inspired coherent GN implementation;
`egn=True` is rejected. The calibration's provenance identifies it as a
repository fit. The original fitting script is not bundled, and the paper's
reported accuracy cannot be inferred from that JSON or the unit tests.

### Numerical controls

`numerical` numerically solves an unpumped SRS profile with an analytic Raman
spectrum surrogate. Its NLI calculation restricts one interacting wave to the
CUT, uses an attenuation response with a Gaussian band average and a
profile-energy scaling, and accumulates route coherence with a scalar visibility
model. Its displaced-third-wave FWM output is a diagnostic subset of SPM/XPM and
is not added again. Refining quadrature improves that calculation's numerical
resolution; it does not turn its band/profile approximations into the complete
R4 integral.

`numerical_dra` independently solves the coupled propagation equations with
DOP853 and uses raw profiles for numerical longitudinal and frequency
integration. It is a useful check on fitted Raman analytical kernels, with its
own interpolation, gain-spectrum, tail truncation, and incoherent-route
assumptions. It is not a waveform simulation. The optional waveform comparison
uses an external package; R14 is background for that numerical simulation
methodology.

## Validation evidence

The audit adds independent checks for the beta2/beta3 propagation-constant
polynomial, invariance under equivalent dispersion reference wavelengths,
actual-route fiber properties, output NLI transfer, and power/amplitude
consistency. Synthetic exponential profiles check the segmented cGN NSP against
its exact exponential longitudinal response. An analytic island-area sum checks
zero-dispersion SCI/XCI/MCI. Heterogeneous-link checks cover a lossless coherent
field sum, opposite dispersion, and a segmented unamplified fiber against one
continuous fiber. Unsupported pump and in-span loss configurations are tested
both at construction and on later route fibers.

The existing suite additionally checks the Raman closed link against an exact
complex longitudinal integral, SPM/XPM frequency integrals, distinct Carena SCI
domains, independent Raman ODE agreement, and coherent Zhang profile/phase
calculations. Regression vectors were updated where the physical transfer or
dispersion reference changes the output; the independent invariants substantiate
those changes.

The earlier audit completed with 219 passed tests and one optional test skipped.
Ruff lint/format checks, all applicable pre-commit hooks, and Mypy across 109
source files passed. An optional waveform test requires `physics_nlo_eval`; a
skipped test does not establish waveform agreement. No paper's broad sub-decibel
accuracy claim is promoted to a project-wide guarantee by this audit.

The PCFM2 addition passed 271 tests with the same optional waveform skip. Ruff
lint/format and Mypy across 115 source/test files passed. Its independent GN
controls, configuration comparisons, timings, and scope limits are recorded in
[the PCFM2 guide](pcfm2.md).

## Primary references

| ID  | Original title                                                                                                                                                  | Identifier                            |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| R1  | A Generalized Raman Scattering Model for Real-Time SNR Estimation of Multi-Band Systems                                                                         | DOI `10.1109/JLT.2023.3250751`        |
| R2  | A Closed-Form Approximation of the Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman Scattering                                            | DOI `10.1109/JLT.2019.2895237`        |
| R3  | A Modulation Format Correction Formula for the Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman Scattering                                | DOI `10.1109/JLT.2019.2929461`        |
| R4  | The Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman Scattering                                                                           | DOI `10.1109/JLT.2018.2830973`        |
| R5  | A Closed-Form Expression for the Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman Scattering Extended for Arbitrary Loss and Fibre Length | DOI `10.1109/JLT.2023.3256185`        |
| R6  | An Efficient Correlation-based Evaluation of the GN-Model for General Multi-Span Optical Systems                                                                | `arXiv:2610.11624v1`, October 8, 2026 |
| R7  | Scalable and Disaggregated GGN Approximation Applied to a C+L+S Optical Network                                                                                 | DOI `10.1109/JLT.2022.3162134`        |
| R8  | EGN model of non-linear fiber propagation                                                                                                                       | DOI `10.1364/OE.22.016335`            |
| R9  | Polynomial Closed Form Model for Ultra-Wideband Transmission Systems                                                                                            | DOI `10.1109/JLT.2026.3678322`        |
| R10 | Accurate Closed-Form Real-Time EGN Model Formula Leveraging Machine-Learning Over 8500 Thoroughly Randomized Full C-Band Systems                                | DOI `10.1109/JLT.2020.2997395`        |
| R11 | A Closed-Form Expression for the Gaussian Noise Model in the Presence of Raman Amplification                                                                    | DOI `10.1109/JLT.2023.3315127`        |
| R12 | Ultra-Wideband Modeling of Optical Fibre Nonlinearity in Hybrid-Amplified Links                                                                                 | DOI `10.1109/JLT.2024.3439999`        |
| R13 | Accurate Closed-Form Coherent EGN Model Supporting Distributed Raman Amplification                                                                              | DOI `10.1109/OECC54135.2024.10975528` |
| R14 | On Numerical Simulations of Ultra-Wideband Long-Haul Optical Communication Systems                                                                              | DOI `10.1109/JLT.2019.2938580`        |
| R15 | The Coherent Polynomials Closed-Form Model for Evaluating Nonlinear Interference in Any Island                                                                  | `arXiv:2602.03860v2`, May 4, 2026     |
