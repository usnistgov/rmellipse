"""Regression tests based on JCGM 101:2008 and JCGM 102:2011.

Primary references
------------------
* JCGM 101:2008, *Evaluation of measurement data -- Supplement 1 to the
  "Guide to the expression of uncertainty in measurement" -- Propagation of
  distributions using a Monte Carlo method*.
* JCGM 102:2011, *Evaluation of measurement data -- Supplement 2 to the
  "Guide to the expression of uncertainty in measurement" -- Extension to
  any number of output quantities*.

Every reference below gives the document's printed page number followed by the
page number in the supplied PDF.  The source tables intentionally report only a
few significant digits, and their Monte Carlo rows are themselves stochastic.
Assertions therefore use absolute tolerances that reflect both the published
rounding and sampling variation.

Covered worked examples
-----------------------
* JCGM 101 section 9.2: additive scalar model.
* JCGM 101 section 9.3: mass calibration.
* JCGM 101 section 9.4: comparison loss in power-meter calibration.
* JCGM 101 section 9.5: gauge-block calibration.
* JCGM 102 section 9.2: additive two-output model and joint coverage regions.
* JCGM 102 section 9.3: nonlinear Cartesian-to-polar transformation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest
import xarray as xr

from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas


MC_TRIALS = 200_000


def _additive_model(x1, x2, x3, x4):
    """JCGM 101:2008, 9.2.1, p. 35 (PDF p. 43), equation (21)."""
    return x1 + x2 + x3 + x4


def _mass_calibration_model(m_r_c, delta_m_r_c, rho_a, rho_w, rho_r):
    """JCGM 101:2008, 9.3.1.3, p. 40 (PDF p. 48), equation (24)."""
    return (m_r_c + delta_m_r_c) * (
        1 + (rho_a - 1.2) * (1 / rho_w - 1 / rho_r)
    ) - 100_000


def _normal_affine_model(standard_normal, location, scale):
    """Construct a Gaussian component used by both supplements.

    See JCGM 101:2008, 9.4.1.4, p. 42 (PDF p. 50), equation (27),
    and JCGM 102:2011, 9.3.1.2, p. 43 (PDF p. 51).
    """
    return location + scale * standard_normal


def _correlated_normal_model(
    shared_standard_normal,
    independent_standard_normal,
    correlation,
    scale,
):
    """Construct a correlated Gaussian component.

    This is the standard-normal representation of the bivariate Gaussian PDFs
    used by JCGM 101:2008, 9.4.1.4, p. 42 (PDF p. 50), equation (27), and
    JCGM 102:2011, 9.3.1.2, p. 43 (PDF p. 51).
    """
    return scale * (
        correlation * shared_standard_normal
        + np.sqrt(1.0 - correlation**2) * independent_standard_normal
    )


def _comparison_loss_model(reflection_real, reflection_imaginary):
    """JCGM 101:2008, 9.4.1.5, p. 43 (PDF p. 51), equation (28)."""
    return reflection_real**2 + reflection_imaginary**2


def _multivariate_additive_model(x1, x2, x3):
    """JCGM 102:2011, 9.2.1, p. 36 (PDF p. 44), equation (24)."""
    return x1 + x3, x2 + x3


def _cartesian_to_polar_model(x1, x2):
    """JCGM 102:2011, 9.3.1.1, p. 43 (PDF p. 51), equation (25)."""
    return np.sqrt(x1**2 + x2**2), np.arctan2(x2, x1)


def _gauge_block_approximate_model(
    length_standard,
    average_difference,
    comparator_random,
    comparator_systematic,
    alpha_standard,
    average_temperature,
    cyclic_temperature,
    delta_alpha,
    delta_temperature,
):
    """JCGM 101:2008, 9.5.1.5, p. 51 (PDF p. 59), equation (37)."""
    return (
        length_standard
        + average_difference
        + comparator_random
        + comparator_systematic
        - length_standard
        * (
            delta_alpha * (average_temperature + cyclic_temperature)
            + alpha_standard * delta_temperature
        )
        - 50_000_000
    )


def _gauge_block_exact_model(
    length_standard,
    average_difference,
    comparator_random,
    comparator_systematic,
    alpha_standard,
    average_temperature,
    cyclic_temperature,
    delta_alpha,
    delta_temperature,
):
    """JCGM 101:2008, 9.5.1.5, p. 51 (PDF p. 59), equation (36)."""
    temperature = average_temperature + cyclic_temperature
    return (
        length_standard
        * (1 + alpha_standard * (temperature - delta_temperature))
        + average_difference
        + comparator_random
        + comparator_systematic
    ) / (1 + (alpha_standard + delta_alpha) * temperature) - 50_000_000


def _symmetric_interval(
    values: xr.DataArray | np.ndarray,
    coverage: float,
) -> tuple[float, float]:
    """JCGM 101:2008, 7.7.2, p. 30 (PDF p. 38), symmetric interval."""
    alpha = (1.0 - coverage) / 2.0
    lower, upper = np.quantile(np.asarray(values), [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _shortest_interval(
    values: xr.DataArray | np.ndarray,
    coverage: float,
) -> tuple[float, float]:
    """Return the shortest interval from the ordered Monte Carlo sample.

    See JCGM 101:2008, 7.7.2, p. 30 (PDF p. 38), and D.8, p. 65
    (PDF p. 73), for the ordered-sample construction.
    """
    ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
    covered_samples = int(np.floor(coverage * ordered.size))
    widths = ordered[covered_samples:] - ordered[: ordered.size - covered_samples]
    index = int(np.argmin(widths))
    return float(ordered[index]), float(ordered[index + covered_samples])


def _order_statistic(values: np.ndarray, coverage: float) -> float:
    """Return the empirical ``coverage`` quantile without interpolation."""
    flattened = np.asarray(values, dtype=float).reshape(-1)
    rank = int(np.ceil(coverage * flattened.size)) - 1
    rank = min(max(rank, 0), flattened.size - 1)
    return float(np.partition(flattened, rank)[rank])


def _joint_coverage_factors(
    samples: np.ndarray,
    coverage: float = 0.95,
) -> tuple[float, float]:
    """Return JCGM 102 hyper-ellipsoid and rectangle coverage factors.

    ``k_p`` follows JCGM 102:2011, 7.7.2, pp. 28-29 (PDF pp. 36-37),
    equations (20) and (21): samples are ranked by Mahalanobis distance from
    the Monte Carlo mean.  ``k_q`` follows 7.7.3, pp. 29-30
    (PDF pp. 37-38), equation (23): samples are ranked by the largest
    componentwise standardized distance.
    """
    sample_array = np.asarray(samples, dtype=float)
    centered = sample_array - sample_array.mean(axis=0)
    covariance = np.cov(sample_array, rowvar=False, ddof=1)

    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, centered.T).T
    ellipsoid_distances = np.linalg.norm(whitened, axis=1)

    standard_uncertainties = np.sqrt(np.diag(covariance))
    rectangle_distances = np.max(
        np.abs(centered) / standard_uncertainties,
        axis=1,
    )
    return (
        _order_statistic(ellipsoid_distances, coverage),
        _order_statistic(rectangle_distances, coverage),
    )


def _lpu_covariance_matrix(results: Sequence[RMEMeas]) -> np.ndarray:
    """Reconstruct output covariance from RME's signed one-u changes."""
    components: list[dict[str, float]] = []
    all_mechanisms: set[str] = set()

    for result in results:
        nominal = _nominal(result)
        result_components = {
            str(mechanism): float(
                (result.cov.sel(umech_id=mechanism).item() - nominal)
            )
            for mechanism in result.umech_id
        }
        components.append(result_components)
        all_mechanisms.update(result_components)

    return np.asarray(
        [
            [
                sum(
                    left.get(mechanism, 0.0) * right.get(mechanism, 0.0)
                    for mechanism in all_mechanisms
                )
                for right in components
            ]
            for left in components
        ],
        dtype=float,
    )


def _mc_vector_statistics(
    results: Sequence[RMEMeas],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return samples, means, standard uncertainties, and correlation."""
    samples = np.column_stack([np.asarray(result.mc) for result in results])
    means = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False, ddof=1)
    standard_uncertainties = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(
        standard_uncertainties,
        standard_uncertainties,
    )
    return samples, means, standard_uncertainties, correlation


def _sampled_scalar_root(
    *,
    name: str,
    nominal: float,
    lpu_standard_uncertainty: float,
    draws: Sequence[float] | np.ndarray,
    degrees_of_freedom: float = np.inf,
) -> RMEMeas:
    samples = np.asarray(draws, dtype=float)
    covariance = xr.DataArray(
        [nominal, nominal + lpu_standard_uncertainty],
        dims=("umech_id",),
        coords={"umech_id": ["nominal", name]},
    )
    monte_carlo = xr.DataArray(
        samples,
        dims=("sample_id",),
        coords={"sample_id": np.arange(samples.size)},
    )
    covariance_dofs = xr.DataArray(
        [degrees_of_freedom],
        dims=("umech_id",),
        coords={"umech_id": [name]},
    )
    return RMEMeas(
        name=name,
        cov=covariance,
        mc=monte_carlo,
        covdofs=covariance_dofs,
    )


def _normal_root(
    *, name: str, nominal: float, standard_uncertainty: float, rng: np.random.Generator
) -> RMEMeas:
    return _sampled_scalar_root(
        name=name,
        nominal=nominal,
        lpu_standard_uncertainty=standard_uncertainty,
        draws=rng.normal(nominal, standard_uncertainty, MC_TRIALS),
    )


def _uniform_root(
    *, name: str, lower: float, upper: float, rng: np.random.Generator
) -> RMEMeas:
    return _sampled_scalar_root(
        name=name,
        nominal=(lower + upper) / 2,
        lpu_standard_uncertainty=(upper - lower) / np.sqrt(12.0),
        draws=rng.uniform(lower, upper, MC_TRIALS),
    )


def _propagate(model: Callable, *arguments) -> RMEMeas:
    propagator = RMEProp(
        sensitivity=True,
        montecarlo_sims=MC_TRIALS,
        vectorize=True,
        set_active=False,
    )
    return propagator.propagate(model)(*arguments)


def _nominal(result: RMEMeas) -> float:
    return float(result.nom.item())


def _lpu_standard_uncertainty(result: RMEMeas) -> float:
    return float(result.stdunc().cov.item())


def _mc_standard_uncertainty(result: RMEMeas) -> float:
    return float(np.asarray(result.mc).std(ddof=1))


def test_jcgm_101_9_2_additive_normal_inputs():
    """Reproduce JCGM 101:2008 Table 2 for equation (21).

    Reference: 9.2.2, pp. 35-36 (PDF pp. 43-44), especially Table 2 on
    p. 36 (PDF p. 44).  The asserted 95 % interval is the symmetric interval
    described in 7.7.2, p. 30 (PDF p. 38).
    """
    rng = np.random.default_rng(9201)
    inputs = [
        _normal_root(
            name=f"X{index}", nominal=0.0, standard_uncertainty=1.0, rng=rng
        )
        for index in range(1, 5)
    ]

    result = _propagate(_additive_model, *inputs)
    lower, upper = _symmetric_interval(result.mc, 0.95)

    assert _nominal(result) == pytest.approx(0.0, abs=1e-14)
    assert _lpu_standard_uncertainty(result) == pytest.approx(2.00, abs=1e-12)
    assert result.mc.mean().item() == pytest.approx(0.00, abs=0.02)
    assert _mc_standard_uncertainty(result) == pytest.approx(2.00, abs=0.02)
    assert lower == pytest.approx(-3.92, abs=0.06)
    assert upper == pytest.approx(3.92, abs=0.06)


def test_jcgm_101_9_2_additive_rectangular_inputs_same_width():
    """Reproduce JCGM 101:2008 Table 3 for equation (21).

    Reference: 9.2.3, pp. 37-38 (PDF pp. 45-46), especially Table 3 on
    p. 37 (PDF p. 45).  The exact endpoint is also given by Annex E,
    equation (E.1), p. 66 (PDF p. 74).
    """
    rng = np.random.default_rng(9202)
    half_width = np.sqrt(3.0)
    inputs = [
        _uniform_root(
            name=f"X{index}", lower=-half_width, upper=half_width, rng=rng
        )
        for index in range(1, 5)
    ]

    result = _propagate(_additive_model, *inputs)
    lower, upper = _symmetric_interval(result.mc, 0.95)
    analytic_endpoint = 2 * np.sqrt(3.0) * (2 - (3 / 5) ** 0.25)

    assert _nominal(result) == pytest.approx(0.0, abs=1e-14)
    assert _lpu_standard_uncertainty(result) == pytest.approx(2.00, abs=1e-12)
    assert result.mc.mean().item() == pytest.approx(0.00, abs=0.02)
    assert _mc_standard_uncertainty(result) == pytest.approx(2.00, abs=0.02)
    assert lower == pytest.approx(-analytic_endpoint, abs=0.06)
    assert upper == pytest.approx(analytic_endpoint, abs=0.06)


def test_jcgm_101_9_2_additive_rectangular_dominant_input():
    """Reproduce JCGM 101:2008 Table 4 for equation (21).

    Reference: 9.2.4, pp. 38-39 (PDF pp. 46-47), especially Table 4 on
    p. 38 (PDF p. 46).
    """
    rng = np.random.default_rng(9203)
    standard_uncertainties = (1.0, 1.0, 1.0, 10.0)
    inputs = [
        _uniform_root(
            name=f"X{index}",
            lower=-np.sqrt(3.0) * standard_uncertainty,
            upper=np.sqrt(3.0) * standard_uncertainty,
            rng=rng,
        )
        for index, standard_uncertainty in enumerate(
            standard_uncertainties, start=1
        )
    ]

    result = _propagate(_additive_model, *inputs)
    lower, upper = _symmetric_interval(result.mc, 0.95)
    expected_standard_uncertainty = np.sqrt(103.0)

    assert _lpu_standard_uncertainty(result) == pytest.approx(
        expected_standard_uncertainty, abs=1e-12
    )
    assert _mc_standard_uncertainty(result) == pytest.approx(
        expected_standard_uncertainty, abs=0.06
    )
    # Table 4 reports only one decimal place for u(y) and its interval endpoints.
    assert result.mc.mean().item() == pytest.approx(0.0, abs=0.09)
    assert lower == pytest.approx(-17.0, abs=0.15)
    assert upper == pytest.approx(17.0, abs=0.15)


def test_jcgm_101_9_3_mass_calibration():
    """Reproduce the principal JCGM 101:2008 mass-calibration results.

    Model: 9.3.1.3, p. 40 (PDF p. 48), equation (24).  Input PDFs: 9.3.1.4,
    Table 5, p. 40 (PDF p. 48).  Assertions: 9.3.2.1 and 9.3.2.6, Table 6,
    p. 41 (PDF p. 49).  The interval is the shortest 95 % interval from the
    ordered sample, as specified in 7.7.2 and Annex D.8.
    """
    rng = np.random.default_rng(9301)
    result = _propagate(
        _mass_calibration_model,
        _normal_root(
            name="m_R,c",
            nominal=100_000.000,
            standard_uncertainty=0.050,
            rng=rng,
        ),
        _normal_root(
            name="delta m_R,c",
            nominal=1.234,
            standard_uncertainty=0.020,
            rng=rng,
        ),
        _uniform_root(name="rho_a", lower=1.10, upper=1.30, rng=rng),
        _uniform_root(name="rho_W", lower=7_000.0, upper=9_000.0, rng=rng),
        _uniform_root(name="rho_R", lower=7_950.0, upper=8_050.0, rng=rng),
    )
    lower, upper = _shortest_interval(result.mc, 0.95)

    assert _nominal(result) == pytest.approx(1.2340, abs=1e-10)
    assert _lpu_standard_uncertainty(result) == pytest.approx(0.0539, abs=0.0001)
    assert result.mc.mean().item() == pytest.approx(1.2341, abs=0.0008)
    assert _mc_standard_uncertainty(result) == pytest.approx(0.0754, abs=0.0008)
    # The shortest interval location is noticeably noisier than its width.
    assert lower == pytest.approx(1.0834, abs=0.010)
    assert upper == pytest.approx(1.3825, abs=0.010)


@pytest.mark.parametrize(
    ("correlation", "x1", "reported_mean", "reported_std", "reported_interval"),
    [
        (0.0, 0.000, 50.0, 50.0, (0.0, 150.0)),
        (0.0, 0.010, 150.0, 112.0, (0.0, 367.0)),
        (0.0, 0.050, 2551.0, 502.0, (1590.0, 3543.0)),
        (0.9, 0.000, 50.0, 67.0, (0.0, 185.0)),
        (0.9, 0.010, 150.0, 121.0, (13.0, 398.0)),
        (0.9, 0.050, 2551.0, 504.0, (1628.0, 3555.0)),
    ],
)
def test_jcgm_101_9_4_comparison_loss_monte_carlo(
    correlation: float,
    x1: float,
    reported_mean: float,
    reported_std: float,
    reported_interval: tuple[float, float],
):
    """Reproduce the six MCM rows in JCGM 101 Tables 8 and 9.

    Model and cases: JCGM 101:2008, 9.4.1.5-9.4.1.7, p. 43
    (PDF p. 51), equation (28).  Zero-correlation results: 9.4.2,
    Table 8, p. 45 (PDF p. 53).  Correlation 0.9 results: 9.4.3.2.1,
    Table 9, p. 48 (PDF p. 56).
    """
    case = int(round(correlation * 10_000 + x1 * 1_000_000))
    rng = np.random.default_rng(9400 + case)
    z1 = _normal_root(
        name="z1", nominal=0.0, standard_uncertainty=1.0, rng=rng
    )
    z2 = _normal_root(
        name="z2", nominal=0.0, standard_uncertainty=1.0, rng=rng
    )
    reflection_real = _propagate(_normal_affine_model, z1, x1, 0.005)
    reflection_imaginary = _propagate(
        _correlated_normal_model, z1, z2, correlation, 0.005
    )
    result = _propagate(
        _comparison_loss_model, reflection_real, reflection_imaginary
    )
    draws_in_micro_units = np.asarray(result.mc) * 1e6
    lower, upper = _shortest_interval(draws_in_micro_units, 0.95)

    assert draws_in_micro_units.mean() == pytest.approx(reported_mean, abs=4.0)
    assert draws_in_micro_units.std(ddof=1) == pytest.approx(reported_std, abs=5.0)
    # These are rounded endpoints from one 10^6-trial run in the source.
    assert lower == pytest.approx(reported_interval[0], abs=22.0)
    assert upper == pytest.approx(reported_interval[1], abs=22.0)


def test_jcgm_101_9_4_lpu_diagnoses_macroscopic_forward_difference():
    """Record why RME LPU does not equal either GUM result at the origin.

    JCGM 101:2008, 9.4.2.2.1, p. 43 (PDF p. 51), explains that all
    first derivatives of equation (28) vanish there.  Table 8, p. 45
    (PDF p. 53), gives zero for first-order GUF; 9.4.2.2.8 on the same page
    gives 50 x 10^-6 for the higher-order treatment.  RME instead propagates
    finite one-standard-uncertainty changes, which this diagnostic preserves.
    """
    rng = np.random.default_rng(9499)
    z1 = _normal_root(
        name="z1", nominal=0.0, standard_uncertainty=1.0, rng=rng
    )
    z2 = _normal_root(
        name="z2", nominal=0.0, standard_uncertainty=1.0, rng=rng
    )
    reflection_real = _propagate(_normal_affine_model, z1, 0.0, 0.005)
    reflection_imaginary = _propagate(
        _correlated_normal_model, z1, z2, 0.0, 0.005
    )
    result = _propagate(
        _comparison_loss_model, reflection_real, reflection_imaginary
    )

    # Each one-u forward perturbation contributes (0.005)^2.  RME combines
    # those two finite changes in quadrature: sqrt(2) * 25e-6 = 35.36e-6.
    expected_forward_difference = np.sqrt(2.0) * 25e-6
    assert _nominal(result) == pytest.approx(0.0, abs=1e-18)
    assert _lpu_standard_uncertainty(result) == pytest.approx(
        expected_forward_difference, rel=1e-12
    )
    # JCGM 101 Table 8 instead gives 0 for first-order GUF and 50e-6 for its
    # higher-order treatment.  This assertion makes the diagnosed distinction explicit.
    assert _lpu_standard_uncertainty(result) != pytest.approx(0.0, abs=1e-12)
    assert _lpu_standard_uncertainty(result) != pytest.approx(50e-6, abs=1e-12)


def _gauge_block_inputs() -> tuple[RMEMeas, ...]:
    """Construct JCGM 101:2008 Table 10 inputs with unrounded parameters.

    See 9.5.2.1, Table 10, p. 51 (PDF p. 59), and the detailed assignments
    in 9.5.2.2-9.5.2.10, pp. 52-54 (PDF pp. 60-62).
    """
    rng = np.random.default_rng(9501)

    def scaled_t(
        name: str,
        nominal: float,
        scale: float,
        degrees_of_freedom: int,
    ) -> RMEMeas:
        # In JCGM's t_nu(mu, sigma^2), sigma is the scale parameter used by
        # GUF; the PDF's actual standard deviation is larger for finite nu.
        draws = nominal + scale * rng.standard_t(degrees_of_freedom, MC_TRIALS)
        return _sampled_scalar_root(
            name=name,
            nominal=nominal,
            lpu_standard_uncertainty=scale,
            draws=draws,
            degrees_of_freedom=float(degrees_of_freedom),
        )

    def curvilinear_trapezoid(
        name: str,
        lower: float,
        upper: float,
        limit_uncertainty: float,
        gum_degrees_of_freedom: int,
    ) -> RMEMeas:
        r1 = rng.random(MC_TRIALS)
        r2 = rng.random(MC_TRIALS)
        sampled_lower = lower - limit_uncertainty + 2 * limit_uncertainty * r1
        sampled_upper = lower + upper - sampled_lower
        draws = sampled_lower + (sampled_upper - sampled_lower) * r2
        # GUF uses the original rectangular standard uncertainty and the
        # stated reliability as degrees of freedom; MCM uses CTrap(a, b, d).
        return _sampled_scalar_root(
            name=name,
            nominal=(lower + upper) / 2,
            lpu_standard_uncertainty=(upper - lower) / np.sqrt(12.0),
            draws=draws,
            degrees_of_freedom=float(gum_degrees_of_freedom),
        )

    # Use the unrounded quantities stated in 9.5.2, rather than the rounded
    # 6 nm, 4 nm, and 7 nm displayed in Table 10.
    length_standard = scaled_t("Ls", 50_000_623.0, 75.0 / 3.0, 18)
    average_difference = scaled_t("D", 215.0, 13.0 / np.sqrt(5.0), 24)
    comparator_random = scaled_t("d1", 0.0, 10.0 / 2.57, 5)
    comparator_systematic = scaled_t("d2", 0.0, 20.0 / 3.0, 8)

    alpha_standard = _uniform_root(
        name="alpha_s", lower=9.5e-6, upper=13.5e-6, rng=rng
    )
    average_temperature = _normal_root(
        name="theta_0", nominal=-0.1, standard_uncertainty=0.2, rng=rng
    )
    phase = rng.random(MC_TRIALS)
    cyclic_temperature = _sampled_scalar_root(
        name="Delta",
        nominal=0.0,
        lpu_standard_uncertainty=1.0 / np.sqrt(8.0),
        draws=0.5 * np.sin(2 * np.pi * phase),
    )
    delta_alpha = curvilinear_trapezoid(
        "delta_alpha", -1.0e-6, 1.0e-6, 0.1e-6, 50
    )
    delta_temperature = curvilinear_trapezoid(
        "delta_temperature", -0.050, 0.050, 0.025, 2
    )
    return (
        length_standard,
        average_difference,
        comparator_random,
        comparator_systematic,
        alpha_standard,
        average_temperature,
        cyclic_temperature,
        delta_alpha,
        delta_temperature,
    )


def test_jcgm_101_9_5_gauge_block_calibration():
    """Reproduce JCGM 101:2008 Table 11 with both model equations.

    The exact and approximate models are equations (36) and (37), 9.5.1.5,
    p. 51 (PDF p. 59).  Assertions come from 9.5.4.1-9.5.4.3, Table 11,
    p. 55 (PDF p. 63), including the stated agreement of the two models at
    the reported precision.
    """
    inputs = _gauge_block_inputs()
    approximate = _propagate(_gauge_block_approximate_model, *inputs)
    exact = _propagate(_gauge_block_exact_model, *inputs)
    lower, upper = _shortest_interval(approximate.mc, 0.99)

    assert _nominal(approximate) == pytest.approx(838.0, abs=0.01)
    assert _lpu_standard_uncertainty(approximate) == pytest.approx(32.0, abs=0.5)
    assert approximate.mc.mean().item() == pytest.approx(838.0, abs=0.5)
    assert _mc_standard_uncertainty(approximate) == pytest.approx(36.0, abs=0.8)
    assert lower == pytest.approx(745.0, abs=5.0)
    assert upper == pytest.approx(932.0, abs=5.0)

    # Section 9.5.4.3 reports identical results, at the shown precision, for
    # the exact nonlinear model and the approximation.
    assert _nominal(exact) == pytest.approx(_nominal(approximate), abs=0.01)
    assert _lpu_standard_uncertainty(exact) == pytest.approx(
        _lpu_standard_uncertainty(approximate), abs=0.01
    )
    assert exact.mc.mean().item() == pytest.approx(
        approximate.mc.mean().item(), abs=0.01
    )
    assert _mc_standard_uncertainty(exact) == pytest.approx(
        _mc_standard_uncertainty(approximate), abs=0.01
    )


@pytest.mark.parametrize(
    (
        "x3_pdf",
        "x3_standard_uncertainty",
        "reported_ellipsoid_factor",
        "reported_rectangle_factor",
        "seed",
    ),
    [
        pytest.param("normal", 1.0, 2.45, 2.21, 102_921, id="table-3-normal"),
        pytest.param(
            "rectangular",
            1.0,
            2.38,
            2.15,
            102_922,
            id="table-4-equal-rectangular-component",
        ),
        pytest.param(
            "rectangular",
            3.0,
            2.28,
            1.87,
            102_923,
            id="table-5-dominant-rectangular-component",
        ),
    ],
)
def test_jcgm_102_9_2_multivariate_additive_model(
    x3_pdf: str,
    x3_standard_uncertainty: float,
    reported_ellipsoid_factor: float,
    reported_rectangle_factor: float,
    seed: int,
):
    """Reproduce JCGM 102:2011 Tables 3-5 for equation (24).

    Model and inputs: 9.2.1, p. 36 (PDF p. 44), equation (24), followed by
    the three cases in 9.2.2-9.2.4, Tables 3-5 on pp. 36, 39, and 41
    (PDF pp. 44, 47, and 49).  The joint 95 % hyper-ellipsoid and rectangle
    factors follow 7.7.2-7.7.3, pp. 28-30 (PDF pp. 36-38), equations
    (20), (21), and (23).
    """
    rng = np.random.default_rng(seed)
    x1 = _normal_root(
        name=f"jcgm102-9.2-{seed}-x1",
        nominal=0.0,
        standard_uncertainty=1.0,
        rng=rng,
    )
    x2 = _normal_root(
        name=f"jcgm102-9.2-{seed}-x2",
        nominal=0.0,
        standard_uncertainty=1.0,
        rng=rng,
    )
    if x3_pdf == "normal":
        x3 = _normal_root(
            name=f"jcgm102-9.2-{seed}-x3",
            nominal=0.0,
            standard_uncertainty=x3_standard_uncertainty,
            rng=rng,
        )
    else:
        half_width = np.sqrt(3.0) * x3_standard_uncertainty
        x3 = _uniform_root(
            name=f"jcgm102-9.2-{seed}-x3",
            lower=-half_width,
            upper=half_width,
            rng=rng,
        )

    outputs = _propagate(_multivariate_additive_model, x1, x2, x3)
    assert isinstance(outputs, tuple)
    samples, means, mc_standard_uncertainties, mc_correlation = (
        _mc_vector_statistics(outputs)
    )

    shared_variance = x3_standard_uncertainty**2
    expected_covariance = np.asarray(
        [
            [1.0 + shared_variance, shared_variance],
            [shared_variance, 1.0 + shared_variance],
        ]
    )
    expected_standard_uncertainty = np.sqrt(1.0 + shared_variance)
    expected_correlation = shared_variance / (1.0 + shared_variance)

    # Equations (24) and the independent input assignments make this LPU
    # covariance analytical; Tables 3-5 display the same matrices.
    np.testing.assert_allclose(
        [_nominal(output) for output in outputs],
        [0.0, 0.0],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        _lpu_covariance_matrix(outputs),
        expected_covariance,
        atol=1e-12,
    )

    # Tables 3-5 report moments and correlation to three decimals and
    # coverage factors to two.  This deterministic 200 000-trial test also
    # allows the expected Monte Carlo sampling variation.
    np.testing.assert_allclose(
        means,
        [0.0, 0.0],
        atol=0.01 * expected_standard_uncertainty,
    )
    np.testing.assert_allclose(
        mc_standard_uncertainties,
        [expected_standard_uncertainty, expected_standard_uncertainty],
        atol=0.01 * expected_standard_uncertainty,
    )
    assert mc_correlation[0, 1] == pytest.approx(
        expected_correlation,
        abs=0.006,
    )

    ellipsoid_factor, rectangle_factor = _joint_coverage_factors(samples)
    assert ellipsoid_factor == pytest.approx(
        reported_ellipsoid_factor,
        abs=0.025,
    )
    assert rectangle_factor == pytest.approx(
        reported_rectangle_factor,
        abs=0.025,
    )


@pytest.mark.parametrize(
    (
        "input_correlation",
        "x1_nominal",
        "reported_means",
        "reported_standard_uncertainties",
        "reported_output_correlation",
        "seed",
    ),
    [
        pytest.param(
            0.0,
            0.001,
            (0.013, -0.001),
            (0.007, 1.744),
            0.000,
            102_931,
            id="table-6-x1-0.001",
        ),
        pytest.param(
            0.0,
            0.010,
            (0.015, 0.000),
            (0.008, 1.117),
            0.000,
            102_932,
            id="table-6-x1-0.010",
        ),
        pytest.param(
            0.0,
            0.100,
            (0.101, 0.000),
            (0.010, 0.101),
            0.000,
            102_933,
            id="table-6-x1-0.100",
        ),
        pytest.param(
            0.9,
            0.001,
            (0.012, -0.556),
            (0.008, 1.599),
            -0.070,
            102_934,
            id="table-7-x1-0.001",
        ),
        pytest.param(
            0.9,
            0.010,
            (0.015, -0.343),
            (0.008, 0.903),
            0.352,
            102_935,
            id="table-7-x1-0.010",
        ),
        pytest.param(
            0.9,
            0.100,
            (0.101, -0.009),
            (0.010, 0.102),
            0.882,
            102_936,
            id="table-7-x1-0.100",
        ),
    ],
)
def test_jcgm_102_9_3_cartesian_to_polar_monte_carlo(
    input_correlation: float,
    x1_nominal: float,
    reported_means: tuple[float, float],
    reported_standard_uncertainties: tuple[float, float],
    reported_output_correlation: float,
    seed: int,
):
    """Reproduce the MCM columns in JCGM 102:2011 Tables 6 and 7.

    Model and six cases: 9.3.1.1-9.3.1.4, p. 43 (PDF p. 51), equation (25).
    Results for uncorrelated inputs: 9.3.2, Table 6, p. 44 (PDF p. 52).
    Results for input correlation 0.9: 9.3.3, Table 7, p. 45
    (PDF p. 53).  The source used 10^7 trials and reports three decimals.
    """
    rng = np.random.default_rng(seed)
    z1 = _normal_root(
        name=f"jcgm102-9.3-{seed}-z1",
        nominal=0.0,
        standard_uncertainty=1.0,
        rng=rng,
    )
    z2 = _normal_root(
        name=f"jcgm102-9.3-{seed}-z2",
        nominal=0.0,
        standard_uncertainty=1.0,
        rng=rng,
    )
    x1 = _propagate(_normal_affine_model, z1, x1_nominal, 0.010)
    x2 = _propagate(
        _correlated_normal_model,
        z1,
        z2,
        input_correlation,
        0.010,
    )
    outputs = _propagate(_cartesian_to_polar_model, x1, x2)
    assert isinstance(outputs, tuple)
    _, means, standard_uncertainties, output_correlation = (
        _mc_vector_statistics(outputs)
    )

    # Near the origin this model is highly nonlinear and the angular output
    # crosses the +/-pi branch cut.  RME's covariance path stores full
    # one-standard-uncertainty finite changes, rather than infinitesimal GUF
    # derivatives, so comparing that path to Tables 6 and 7 would conflate two
    # different methods.  These assertions intentionally target MCM only.
    # Absolute (not relative) tolerances are required for the reported zero
    # means/correlations.  They cover the source's three-decimal rounding,
    # the larger sampling error of Y2 with 200 000 rather than 10^7 trials,
    # and finite-precision accumulation through staged RME propagation.
    assert means[0] == pytest.approx(reported_means[0], abs=0.0015)
    assert means[1] == pytest.approx(reported_means[1], abs=0.012)
    assert standard_uncertainties[0] == pytest.approx(
        reported_standard_uncertainties[0],
        abs=0.0012,
    )
    assert standard_uncertainties[1] == pytest.approx(
        reported_standard_uncertainties[1],
        abs=0.012,
    )
    assert output_correlation[0, 1] == pytest.approx(
        reported_output_correlation,
        abs=0.020,
    )
