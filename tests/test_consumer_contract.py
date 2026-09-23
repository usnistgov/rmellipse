"""Contract tests for public uncertainty storage and propagation behavior.

Each test states an observable guarantee and describes the probe used to
separate the expected behavior from a plausible regression.  The suite uses
only public measurement and propagator objects so it can serve as an
integration boundary for independently released consumer layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import xarray as xr

from rmellipse.propagators import RMEProp
from rmellipse.uobjects import RMEMeas, RMEMeasFormatError


def _scalar_measurement(
    *,
    name: str,
    nominal: complex | float,
    components: Mapping[str, complex | float],
    draws: Sequence[complex | float] | None = None,
    degrees_of_freedom: Mapping[str, float] | None = None,
    categories: Mapping[str, Mapping[str, str]] | None = None,
    attrs: Mapping | None = None,
) -> RMEMeas:
    """Build a scalar measurement with explicit signed components."""
    mechanism_ids = list(components)
    cov = xr.DataArray(
        [nominal, *(nominal + components[key] for key in mechanism_ids)],
        dims='umech_id',
        coords={'umech_id': ['nominal', *mechanism_ids]},
    )
    mc = None
    if draws is not None:
        mc = xr.DataArray(
            np.asarray(draws),
            dims='sample_id',
            coords={'sample_id': np.arange(len(draws))},
        )

    covdofs = None
    if degrees_of_freedom is not None:
        covdofs = xr.DataArray(
            [degrees_of_freedom[key] for key in mechanism_ids],
            dims='umech_id',
            coords={'umech_id': mechanism_ids},
        )

    covcats = None
    if categories is not None:
        category_names = list(
            dict.fromkeys(
                category
                for mechanism_id in mechanism_ids
                for category in categories.get(mechanism_id, {})
            )
        )
        covcats = xr.DataArray(
            [
                [categories.get(mechanism_id, {}).get(category, '') for category in category_names]
                for mechanism_id in mechanism_ids
            ],
            dims=('umech_id', 'categories'),
            coords={
                'umech_id': mechanism_ids,
                'categories': category_names,
            },
        )

    return RMEMeas(
        name=name,
        cov=cov,
        mc=mc,
        covdofs=covdofs,
        covcats=covcats,
        attrs=dict(attrs or {}),
    )


def _component(measurement: RMEMeas, mechanism_id: str):
    """Return one signed finite-difference component."""
    return measurement.cov.sel(umech_id=mechanism_id) - measurement.nom


def test_complete_storage_separates_estimate_components_and_draws():
    """A stored draw must never be interpreted as the nominal estimate.

    The probe deliberately makes stochastic row zero very different from the
    estimate, then checks the nominal accessor, mechanism metadata, and first
    sample independently.  This also verifies that default metadata describes
    mechanisms only and does not create a synthetic nominal metadata row.
    """
    measurement = _scalar_measurement(
        name='separate representations',
        nominal=10.0,
        components={'calibration': 0.5},
        draws=[-7.0, 12.0],
    )

    assert measurement.validate() is measurement
    assert measurement.nom.item() == pytest.approx(10.0)
    assert measurement.mc.isel(sample_id=0).item() == pytest.approx(-7.0)
    assert measurement.mc.sizes['sample_id'] == 2
    np.testing.assert_array_equal(measurement.mc.sample_id, [0, 1])
    assert measurement.umech_id == ['calibration']
    np.testing.assert_array_equal(measurement.covdofs.umech_id, ['calibration'])
    assert measurement.covdofs.item() == np.inf
    np.testing.assert_array_equal(measurement.covcats.umech_id, ['calibration'])
    assert measurement.covcats.sel(categories='Type').item() == 'B'


def test_validation_rejects_noncanonical_draw_indices_without_nominal_checks():
    """Draw indices must be integer, zero-based, and consecutive.

    Two malformed coordinate grids instrument the index checks.  A separate
    valid object uses a first draw unequal to the estimate, demonstrating that
    validation constrains structure rather than requiring a nominal-prefixed
    sample population.
    """
    cov = xr.DataArray(
        [3.0],
        dims='umech_id',
        coords={'umech_id': ['nominal']},
    )

    for sample_ids in ([0, 2], [0.0, 1.0]):
        measurement = RMEMeas(
            name='invalid draw grid',
            cov=cov,
            mc=xr.DataArray(
                [20.0, 21.0],
                dims='sample_id',
                coords={'sample_id': sample_ids},
            ),
        )
        with pytest.raises(RMEMeasFormatError, match='consecutive integers'):
            measurement.validate()

    valid = RMEMeas(
        name='valid stochastic row zero',
        cov=cov,
        mc=xr.DataArray(
            [20.0, 21.0],
            dims='sample_id',
            coords={'sample_id': [0, 1]},
        ),
    )
    assert valid.validate() is valid


def test_monte_carlo_propagation_pairs_draws_without_inserting_an_estimate_row():
    """Joint propagation must preserve one-to-one stochastic trial pairing.

    The two inputs use visibly asymmetric draw sequences.  Their sum identifies
    whether rows were paired by trial and whether an extra nominal row was
    inserted before or after evaluation.
    """
    left = _scalar_measurement(
        name='left',
        nominal=100.0,
        components={'left source': 1.0},
        draws=[1.0, 2.0, 4.0],
    )
    right = _scalar_measurement(
        name='right',
        nominal=200.0,
        components={'right source': 2.0},
        draws=[10.0, 20.0, 40.0],
    )
    propagator = RMEProp(
        sensitivity=True,
        montecarlo_sims=3,
        vectorize=True,
        set_active=False,
    )

    @propagator.propagate
    def add(first, second):
        return first + second

    result = add(left, right)

    assert result.nom.item() == pytest.approx(300.0)
    np.testing.assert_allclose(result.mc.values, [11.0, 22.0, 44.0])
    np.testing.assert_array_equal(result.mc.sample_id, [0, 1, 2])
    assert result.mc.sizes['sample_id'] == 3


def test_mechanism_identity_controls_correlation_and_stable_row_order():
    """Equal mechanism IDs must act as one correlated latent contribution.

    Both inputs carry one shared ID plus one private ID.  Adding them reveals
    whether the shared perturbations are evaluated together, while the output
    row order verifies deterministic first-seen deduplication for consumers
    that attach metadata by mechanism coordinate.
    """
    left = _scalar_measurement(
        name='left',
        nominal=10.0,
        components={'shared source': 1.0, 'left source': 0.5},
    )
    right = _scalar_measurement(
        name='right',
        nominal=20.0,
        components={'shared source': 2.0, 'right source': 4.0},
    )
    propagator = RMEProp(sensitivity=True, set_active=False)

    @propagator.propagate
    def add(first, second):
        return first + second

    result = add(left, right)

    assert result.umech_id == ['shared source', 'left source', 'right source']
    assert _component(result, 'shared source').item() == pytest.approx(3.0)
    assert _component(result, 'left source').item() == pytest.approx(0.5)
    assert _component(result, 'right source').item() == pytest.approx(4.0)
    assert result.stdunc().cov.item() == pytest.approx(
        np.sqrt(3.0**2 + 0.5**2 + 4.0**2)
    )


def test_nonlinear_lpu_uses_signed_finite_difference_evaluations():
    """Nonlinear propagation must retain the macroscopic one-step change.

    Squaring a value of ten with a positive unit perturbation produces a change
    of twenty-one under direct finite-difference evaluation, rather than the
    derivative-only value of twenty.  The chosen numbers distinguish the two
    algorithms exactly.
    """
    value = _scalar_measurement(
        name='nonlinear input',
        nominal=10.0,
        components={'source': 1.0},
    )
    propagator = RMEProp(sensitivity=True, set_active=False)

    @propagator.propagate
    def square(argument):
        return argument**2

    result = square(value)

    assert result.nom.item() == pytest.approx(100.0)
    assert _component(result, 'source').item() == pytest.approx(21.0)


@pytest.mark.parametrize('vectorize', [True, False])
def test_keyword_inputs_and_tuple_outputs_work_in_both_execution_modes(vectorize):
    """Uncertain keyword arguments and multiple outputs must be first-class.

    The probe passes the only uncertain value by keyword and returns an
    asymmetric pair.  Running both execution modes checks row-wise and batched
    paths, while the expected covariance and draw arrays detect tuple coercion
    or lost structural dimensions.
    """
    offset = _scalar_measurement(
        name='keyword offset',
        nominal=2.0,
        components={'offset source': 0.5},
        draws=[1.8, 2.3],
    )
    propagator = RMEProp(
        sensitivity=True,
        montecarlo_sims=2,
        vectorize=vectorize,
        set_active=False,
    )

    @propagator.propagate
    def pair(base, *, shift):
        return base + shift, base - shift

    increased, decreased = pair(1.0, shift=offset)

    for result in (increased, decreased):
        assert isinstance(result, RMEMeas)
        assert result.cov.dims == ('umech_id',)
        assert result.mc.dims == ('sample_id',)
        assert result.dtype != np.dtype(object)

    np.testing.assert_allclose(increased.cov.values, [3.0, 3.5])
    np.testing.assert_allclose(increased.mc.values, [2.8, 3.3])
    np.testing.assert_allclose(decreased.cov.values, [-1.0, -1.5])
    np.testing.assert_allclose(decreased.mc.values, [-0.8, -1.3])


def test_public_metadata_mergers_align_missing_fields_and_shared_ids():
    """Public metadata helpers must produce a complete mechanism table.

    Inputs intentionally use different mechanism orders, category columns, and
    degrees of freedom.  Direct calls to the public helpers verify first-seen
    mechanism ordering, minimum shared degrees of freedom, blank-fill behavior,
    and string-preserving category storage without relying on private helpers.
    """
    first = _scalar_measurement(
        name='first metadata source',
        nominal=1.0,
        components={'shared': 0.1, 'left': 0.2},
        degrees_of_freedom={'shared': 12.0, 'left': 8.0},
        categories={
            'shared': {'Type': 'B', 'Origin': 'certificate'},
            'left': {'Type': 'A', 'Origin': 'repeatability'},
        },
    )
    second = _scalar_measurement(
        name='second metadata source',
        nominal=2.0,
        components={'right': 0.3, 'shared': 0.4},
        degrees_of_freedom={'right': 5.0, 'shared': 6.0},
        categories={
            'right': {'Type': 'B', 'Owner': 'laboratory'},
            'shared': {'Type': 'B', 'Owner': 'laboratory'},
        },
    )
    mechanism_ids = ['shared', 'left', 'right']

    merged_dofs = RMEProp.merge_covdofs(mechanism_ids, [first, second], {})
    merged_categories = RMEProp.merge_categories(
        mechanism_ids, [first, second], {}
    ).sel(umech_id=mechanism_ids)

    assert merged_dofs.dims == ('umech_id',)
    np.testing.assert_array_equal(merged_dofs.umech_id, mechanism_ids)
    np.testing.assert_allclose(merged_dofs.values, [6.0, 8.0, 5.0])
    assert merged_dofs.dtype.kind == 'f'

    assert merged_categories.dims == ('umech_id', 'categories')
    assert merged_categories.dtype == np.dtype('T')
    assert set(merged_categories.categories.values.tolist()) == {
        'Type',
        'Origin',
        'Owner',
    }
    assert (
        merged_categories.sel(umech_id='shared', categories='Origin').item()
        == 'certificate'
    )
    assert (
        merged_categories.sel(umech_id='shared', categories='Owner').item()
        == 'laboratory'
    )
    assert (
        merged_categories.sel(umech_id='left', categories='Owner').item() == ''
    )
    assert (
        merged_categories.sel(umech_id='right', categories='Origin').item() == ''
    )


def test_public_category_merge_warns_and_keeps_first_explicit_value():
    """Conflicting nonblank category values require visible deterministic handling.

    Two records assign different values to the same mechanism/category cell.
    The warning instruments conflict visibility, and the retained first value
    fixes precedence so results cannot depend on an undocumented overwrite.
    """
    first = _scalar_measurement(
        name='first category source',
        nominal=1.0,
        components={'shared': 0.1},
        categories={'shared': {'Type': 'A'}},
    )
    second = _scalar_measurement(
        name='second category source',
        nominal=2.0,
        components={'shared': 0.2},
        categories={'shared': {'Type': 'B'}},
    )

    with pytest.warns(UserWarning, match='CATEGORY MISMATCH'):
        merged = RMEProp.merge_categories(['shared'], [first, second], {})

    assert merged.sel(umech_id='shared', categories='Type').item() == 'A'


def test_propagation_metadata_matches_the_public_merge_contract():
    """Propagated rows must carry the same metadata exposed by merge helpers.

    The inputs combine shared and independent mechanisms with partial category
    tables.  Comparing the propagated result to explicit public merges catches
    metadata loss, reordering, or a divergence between assembly and propagation
    behavior.
    """
    first = _scalar_measurement(
        name='first propagated metadata',
        nominal=4.0,
        components={'shared': 0.1, 'first only': 0.2},
        degrees_of_freedom={'shared': 20.0, 'first only': 7.0},
        categories={
            'shared': {'Type': 'B', 'Origin': 'certificate'},
            'first only': {'Type': 'A'},
        },
    )
    second = _scalar_measurement(
        name='second propagated metadata',
        nominal=5.0,
        components={'second only': 0.3, 'shared': 0.4},
        degrees_of_freedom={'second only': 9.0, 'shared': 11.0},
        categories={
            'second only': {'Type': 'B', 'Owner': 'laboratory'},
            'shared': {'Type': 'B', 'Owner': 'laboratory'},
        },
    )
    mechanism_ids = ['shared', 'first only', 'second only']
    expected_dofs = RMEProp.merge_covdofs(mechanism_ids, [first, second], {})
    expected_categories = RMEProp.merge_categories(
        mechanism_ids, [first, second], {}
    ).sel(umech_id=mechanism_ids)
    propagator = RMEProp(sensitivity=True, set_active=False)

    @propagator.propagate
    def add(left, right):
        return left + right

    result = add(first, second)

    xr.testing.assert_identical(result.covdofs, expected_dofs)
    xr.testing.assert_identical(result.covcats, expected_categories)


def test_physical_selection_preserves_all_uncertainty_representations():
    """Selecting physical labels must transform every numerical plane together.

    A two-point value with LPU rows, stochastic draws, and mechanism metadata is
    selected by label and position.  Expected xarray selections instrument the
    numerical result, while unchanged metadata verifies that physical slicing
    does not silently alter mechanism semantics.
    """
    nominal = xr.DataArray(
        [10.0, 20.0],
        dims='point',
        coords={
            'point': ['a', 'b'],
            'quality': ('point', ['reference', 'working']),
        },
    )
    cov = xr.concat(
        [nominal, nominal + xr.DataArray([0.1, 0.2], dims='point')],
        dim=xr.IndexVariable('umech_id', ['nominal', 'source']),
    )
    mc = xr.DataArray(
        [[9.0, 19.0], [11.0, 21.0]],
        dims=('sample_id', 'point'),
        coords={
            'sample_id': [0, 1],
            'point': ['a', 'b'],
            'quality': ('point', ['reference', 'working']),
        },
    )
    covdofs = xr.DataArray(
        [12.0], dims='umech_id', coords={'umech_id': ['source']}
    )
    covcats = xr.DataArray(
        [['B', 'certificate']],
        dims=('umech_id', 'categories'),
        coords={
            'umech_id': ['source'],
            'categories': ['Type', 'Origin'],
        },
    )
    measurement = RMEMeas(
        name='selectable',
        cov=cov,
        mc=mc,
        covdofs=covdofs,
        covcats=covcats,
    )

    by_label = measurement.sel(point=['b'])
    by_position = measurement.isel(point=[0])

    xr.testing.assert_identical(by_label.cov, measurement.cov.sel(point=['b']))
    xr.testing.assert_identical(by_label.mc, measurement.mc.sel(point=['b']))
    xr.testing.assert_identical(by_position.cov, measurement.cov.isel(point=[0]))
    xr.testing.assert_identical(by_position.mc, measurement.mc.isel(point=[0]))
    for selected in (by_label, by_position):
        assert selected.cov.dims == ('umech_id', 'point')
        assert selected.mc.dims == ('sample_id', 'point')
        xr.testing.assert_identical(selected.covdofs, measurement.covdofs)
        xr.testing.assert_identical(selected.covcats, measurement.covcats)


def test_propagation_does_not_mutate_borrowed_input_state():
    """Evaluation must treat caller-owned measurements as read-only inputs.

    Deep snapshots of numerical arrays, metadata, and nested attributes are
    compared after a propagation call.  This catches in-place alignment,
    coordinate rewriting, metadata merging, or attribute mutation that would
    invalidate an independently managed identity or persistence record.
    """
    measurement = _scalar_measurement(
        name='borrowed input',
        nominal=6.0,
        components={'source': 0.4},
        draws=[5.5, 6.5],
        degrees_of_freedom={'source': 9.0},
        categories={'source': {'Type': 'B', 'Origin': 'certificate'}},
        attrs={'nested': {'revision': 3}},
    )
    cov_before = measurement.cov.copy(deep=True)
    mc_before = measurement.mc.copy(deep=True)
    dofs_before = measurement.covdofs.copy(deep=True)
    categories_before = measurement.covcats.copy(deep=True)
    attrs_before = {'nested': {'revision': 3}}
    propagator = RMEProp(
        sensitivity=True,
        montecarlo_sims=2,
        vectorize=True,
        set_active=False,
    )

    @propagator.propagate
    def scale(value):
        return 3.0 * value

    result = scale(measurement)

    assert result.nom.item() == pytest.approx(18.0)
    xr.testing.assert_identical(measurement.cov, cov_before)
    xr.testing.assert_identical(measurement.mc, mc_before)
    xr.testing.assert_identical(measurement.covdofs, dofs_before)
    xr.testing.assert_identical(measurement.covcats, categories_before)
    assert measurement.attrs['nested'] == attrs_before['nested']


def test_nonactivating_propagator_preserves_existing_operator_dispatch(monkeypatch):
    """An explicitly isolated propagator must not replace global dispatch state.

    A sentinel is installed as the active operator engine before constructing a
    new evaluator with activation disabled.  Identity comparison detects any
    unintended global registration during construction.
    """
    sentinel = object()
    monkeypatch.setattr(RMEProp, '_active', sentinel)

    created = RMEProp(set_active=False)

    assert created is not sentinel
    assert RMEProp._active is sentinel
