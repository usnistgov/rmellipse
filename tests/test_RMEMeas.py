from rmellipse.uobjects import RMEMeas, RMEMeasFormatError
from rmellipse.propagators import RMEProp
from rmellipse._test_collections.rmemeas import make_example_meas
from pathlib import Path
import numpy as np
import h5py
import xarray as xr
import pytest
from rmellipse._test_collections.rmemeas import from_dist
import rmellipse as rme

# Local directory for finding tests (the directory of this file)
# Pytest is annoying on what the root actually is
# so this gives a consisten, known location to build paths from
# that is OS independent
LOCAL = Path(__file__).parents[0]


def test_from_dist():
    nom = 1
    std = 1
    with pytest.raises(ValueError):
        a = from_dist(name='dummy', nom=1, std=1, dist='not supported')

    a = from_dist(
        name='dummy', nom=1, std=1, dist='gaussian', use_sample_mean=False, samples=1000
    )
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    assert a.mc.sizes['sample_id'] == 1000

    a = from_dist(name='dummy', nom=1, std=1, dist='normal', use_sample_mean=False)
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)

    a = from_dist(name='dummy', nom=1, std=1, dist='normal', use_sample_mean=False)
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)

    a = from_dist(name='dummy', nom=1, std=1, dist='rectangular', use_sample_mean=False)
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)

    a = from_dist(name='dummy', nom=1, std=1, dist='rectangular', use_sample_mean=False)
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)


def test_interp():
    # makes coordinate between 0 and 1
    m1 = make_example_meas(
        nom_shape=(3, 3), rand_nom=True, rand_coords=True, N_mc_samples=10
    )
    out_of_bounds_coords = np.array([-0.5, 0.5, 1.5])
    m1_oob = m1.interp(d1=out_of_bounds_coords)
    assert all(m1_oob.cov.d1 == m1.cov.d1)


def test_validate_conventions():
    # this should fail because first label is wrong
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['not nominal', 'dummy'], 'd1': [0]},
            ),
            mc=None,
        )
        bad_cov._validate_conventions()

    # this should fail because umech_id is in the wrong spot
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0, 1]],
                dims=('d1', 'umech_id'),
                coords={'umech_id': ['nominal', 'dummy'], 'd1': [0]},
            ),
            mc=None,
        )
        bad_cov._validate_conventions()

    # this should fail because covdofs plocs are named inproperly
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['nominal', 'dummy'], 'd1': [0]},
            ),
            covdofs=xr.DataArray(
                [[]],
                dims=('not umech_id', 'category'),
                coords={'not umech_id': ['not dummy']},
            ),
        )

        bad_cov._validate_conventions()

    # this should fail because covdofs plocs don't match cov
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['nominal', 'dummy'], 'd1': [0]},
            ),
            covdofs=xr.DataArray(
                [[]], dims=('umech_id', 'category'), coords={'umech_id': ['not dummy']}
            ),
        )

        bad_cov._validate_conventions()

    # this should fail because covdofs ploc dim is in the wrong spot
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['nominal', 'dummy'], 'd1': [0]},
            ),
            covdofs=xr.DataArray(
                [
                    [0],
                ],
                dims=('category', 'umech_id'),
                coords={'umech_id': ['not dummy']},
            ),
        )

        bad_cov._validate_conventions()

    # this should fail because first label is wrong
    with pytest.raises(rme.ValidationError):
        bad_cov = RMEMeas(
            name='bad',
            mc=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['not nominal', 'dummy'], 'd1': [0]},
            ),
            cov=None,
        )
        bad_cov._validate_conventions()

    # this should fail because umech_id doesn't exist in mc
    with pytest.raises(rme.ValidationError):
        bad_cov = RMEMeas(
            name='bad',
            mc=xr.DataArray([[0, 1]]),
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['nominal', 'dummy']},
            ),
        )
        print(bad_cov)
        bad_cov._validate_conventions()

    # this should fail because umech_id doesn't exist in cov
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            mc=xr.DataArray([[0, 1]]),
            cov=xr.DataArray(
                [[0], [1]],
                dims=('not umech_id', 'd1'),
                coords={'not umech_id': ['nominal', 'dummy']},
            ),
            covdofs=0,
            covcats=0,
        )
        print(bad_cov)
        bad_cov._validate_conventions()

    # this should fail because first label is wrong
    with pytest.raises(rme.ValidationError):
        bad_cov = RMEMeas(
            name='bad',
            mc=xr.DataArray(
                [[0, 1]],
                dims=('d1', 'umech_id'),
                coords={'umech_id': ['not nominal', 'dummy'], 'd1': [0]},
            ),
            cov=None,
        )
        bad_cov._validate_conventions()

        # this should fail because covdofs plocs are named inproperly
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['nominal', 'dummy'], 'd1': [0]},
            ),
            covcats=xr.DataArray(
                [[]],
                dims=('not umech_id', 'category'),
                coords={'not umech_id': ['not dummy']},
            ),
        )

        bad_cov._validate_conventions()

    # this should fail because covdofs plocs don't match cov
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['nominal', 'dummy'], 'd1': [0]},
            ),
            covcats=xr.DataArray(
                [[]], dims=('umech_id', 'category'), coords={'umech_id': ['not dummy']}
            ),
        )

        bad_cov._validate_conventions()

    # this should fail because plocs is in the wrong spot
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name='bad',
            cov=xr.DataArray(
                [[0], [1]],
                dims=('umech_id', 'd1'),
                coords={'umech_id': ['nominal', 'dummy'], 'd1': [0]},
            ),
            covcats=xr.DataArray(
                [
                    [0],
                ],
                dims=('category', 'umech_id'),
                coords={'umech_id': ['not dummy']},
            ),
        )

        bad_cov._validate_conventions()


def test_overload():
    prop = RMEProp(sensitivity=True)
    prop.set_active()
    m1 = make_example_meas(nom_shape=(1,), rand_nom=True)
    m2 = make_example_meas(nom_shape=(1,), rand_nom=True)
    # these are ndarrys, they will overload into __array_ufunc__
    nd1 = m1.nom[0].values
    nd2 = m2.nom[0].values
    # floats
    f1 = float(nd1)
    f2 = float(nd2)

    # __array_unfunc__
    assert np.isclose((nd1 * m2).nom[0], f1 * f2)
    assert np.isclose(np.sin(m1).nom[0], np.sin(f1))

    # __add__
    assert np.isclose((m1 + m2).nom[0], f1 + f2)

    # __radd__
    assert np.isclose((f1 + m2).nom[0], f1 + f2)

    # __mul__
    assert np.isclose((m1 * m2).nom[0], f1 * f2)

    # __rmul__
    assert np.isclose((f1 * m2).nom[0], f1 * f2)

    # __sub__
    assert np.isclose((m1 - m2).nom[0], f1 - f2)

    # __rsub__
    assert np.isclose((f1 - m2).nom[0], f1 - f2)

    # __trudiv__
    assert np.isclose((m1 / m2).nom[0], f1 / f2)

    # __rtruediv__
    assert np.isclose((f1 / m2).nom[0], f1 / f2)

    # __pow__
    assert np.isclose((m1**m2).nom[0], f1**f2)

    # __rtruediv__
    assert np.isclose((f1 / m2).nom[0], f1 / f2)


def test_xml():
    m1 = make_example_meas(nom_shape=(2, 2), N_mc_samples=100)
    m1.name = 'test_name'

    def to_txt(data, path):
        np.savetxt(path, data.values, delimiter=',')

    def from_txt(path):
        values = xr.DataArray(np.loadtxt(path, float, delimiter=','))
        return values

    target = LOCAL / 'mutable'
    print(m1.mc)
    m1.to_xml(
        str(target.resolve()), to_txt, data_extension='.csv', header_extension='.meas'
    )

    m2 = RMEMeas.from_xml(str(str(target / 'test_name.meas')), from_csv=from_txt)
    assert (m2.cov.values == m1.cov.values).all()
    assert (m2.mc.values == m1.mc.values).all()
    np.testing.assert_array_equal(m2.mc.sample_id, np.arange(m1.mc.shape[0]))


def test_copy():
    m1 = make_example_meas(nom_shape=(2, 2), N_mc_samples=100)

    test = RMEMeas(m1.name, m1.cov, m1.mc)
    copy = test.copy()
    assert (test.cov == m1.cov).all()
    assert (test.mc == copy.mc).all()
    assert (test.covdofs == copy.covdofs).all()
    assert (test.covcats == copy.covcats).all()

    test = RMEMeas(m1.name, m1.cov, None)
    copy = test.copy()
    assert (test.cov == m1.cov).all()
    assert test.mc == copy.mc
    assert (test.covdofs == copy.covdofs).all()
    assert (test.covcats == copy.covcats).all()


def test_repr():
    m1 = make_example_meas()
    assert str(m1) == m1.__repr__()


def test_stdunc():
    m1 = make_example_meas(
        lin_unc=1, mc_unc=1, nom_shape=(1,), rand_nom=False, N_mc_samples=100
    )

    m1._validate_conventions()

    unc = m1.stdunc(k=2)
    # check that the first index is the cov
    # expecting a named tuple like object
    assert unc.cov == unc[0]
    assert unc.mc == unc[1]
    assert unc.cov[0] == 2
    # this is a random number, but it should be close to 2
    assert unc.mc[0] > 1.5 and unc.mc[0] < 2.5

    # check for angles
    m1 = RMEMeas.from_nom('angle', xr.DataArray([np.pi]))
    m1.add_umech('test', np.pi + 2 * np.pi)

    # normal call thinks the phase difference here is > 0
    unc = m1.stdunc().cov
    assert not np.isclose(unc, 0)

    # passing rad should yield 0 or deg zero if uncertainty circles arround
    unc = m1.stdunc(rad=True).cov
    assert np.isclose(unc, 0)

    m1 = RMEMeas.from_nom('angle', xr.DataArray([180]))
    m1.add_umech('test', 180 + 360)
    unc = m1.stdunc(deg=True).cov
    assert np.isclose(unc, 0)

    with pytest.raises(ValueError):
        m1.stdunc(deg=True, rad=True)
    m1.cov = None
    m1.stdunc()


def test_cull_cov():
    """
    Tests the autoculling functionality of the RMEmeas object.

    Returns
    -------
    None.

    """
    prop = RMEProp(sensitivity=True)

    test = from_dist('test', [1, 1, 1], [0, 0, 0])
    test.cull_cov()
    assert len(test.umech_id) == 0
    test = test + 2

    # check that setting tolerance above 0.1 keeps mechanisms with tolerance above 0.1
    test = from_dist('test', [1, 1, 1], [0.1, 0.1, 0.1], use_sample_mean=False)
    test.add_umech('thing', test.nom + np.array([0.1, 0.01, 0.01]))
    test.cull_cov(tolerance=0.3)
    assert len(test.umech_id) == 1


# def test_h5_encoding():
#     test = make_example_meas()
#     print('the local path is ', LOCAL)
#     with h5py.File(LOCAL / 'mutable/tests.hdf5', 'w') as hf:
#         test.to_h5(hf)
#         # saving an existing object should fail
#         # with an error message saying it already exists
#         try:
#             test.to_h5(hf)
#         except ValueError as e:
#             assert 'exists' in str(e)
#         test.to_h5(hf, override=True)
#         print(test.name)
#         read = RMEMeas.from_h5(hf[test.name])
#         read_nom = RMEMeas.from_h5(hf[test.name], nominal_only=True)
#     assert (read.cov == test.cov).all()
#     assert read_nom.cov.shape[0] == 1
#     assert (read_nom.nom == read.nom).all()

#     # check tthe name functon
#     with h5py.File(LOCAL / 'mutable/tests.hdf5', 'w') as hf:
#         oldname = test.name
#         test.to_h5(hf, name='testname')
#         # saving an existing object should fail
#         # with an error message saying it already exists
#         assert test.name == oldname
#         with pytest.raises(AttributeError):
#             test.to_h5('asf', override=True)
#             assert test.name == oldname
#             print(test.name)
#         read = RMEMeas.from_h5(hf['testname'])
#         read_nom = RMEMeas.from_h5(hf['testname'], nominal_only=True)
#     assert (read.cov == test.cov).all()
#     assert read_nom.cov.shape[0] == 1
#     assert (read_nom.nom == read.nom).all()

#     # check that MUFmeas objects read work
#     with h5py.File(LOCAL / 'const' / '24splitter_proto.h5', 'r') as hf:
#         old = RMEMeas.from_h5(hf['24mm_clrmproto_splitter'])
#         with h5py.File(LOCAL / 'mutable/MUFmeas_write_test.hdf5', 'w') as hf2:
#             old.to_h5(hf2, override=True)


def test_indexing():
    test = from_dist('test', [1, 1, 1], [0, 0, 0])
    getitem = test[0]
    loc = test.loc[1]
    sel = test.sel({3: 0})
    isel = test.isel({3: 0})

    for thing in [loc, sel, isel]:
        assert thing.nom == getitem.nom

    # check that sel/isel blocks umech_id
    raised = False
    try:
        test.sel({'umech_id': 'nominal'})
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        test.sel(umech_id='nominal')
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        test.isel({'umech_id': 0})
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        test.isel(umech_id=0)
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        test.usel(umech_id=0)
    except ValueError:
        raised = True
    assert raised

    first_sample = test.usel(sample_id=[0])
    assert first_sample.mc.sizes['sample_id'] == 1
    np.testing.assert_array_equal(first_sample.mc.sample_id, [0])
    xr.testing.assert_equal(
        first_sample.mc.isel(sample_id=0, drop=True),
        test.mc.isel(sample_id=0, drop=True),
    )

    raised = False
    try:
        test.usel(umech_id=['nominal'])
    except ValueError:
        raised = True
    assert raised

    usel = test.usel(umech_id=[test.umech_id[0]])
    assert len(usel.umech_id) == 1
    # no covariance mechanisms
    usel = test.usel(umech_id=[])
    assert len(usel.umech_id) == 0
    usel = test.usel(sample_id=[2, 3])
    assert usel.mc.shape[0] == 2
    np.testing.assert_array_equal(usel.mc.sample_id, [0, 1])
    np.testing.assert_allclose(usel.mc, test.mc.isel(sample_id=[2, 3]))
    usel = test.usel(umech_id=[], sample_id=[2, 3])
    assert usel.mc.shape[0] == 2 and len(usel.umech_id) == 0
    assert 'nominal' not in usel.umech_id
    assert test.usel(sample_id=[]).mc is None


def test_make_umechs_unique():
    m = make_example_meas()
    m2 = m.copy()
    m2.make_umechs_unique(same_uid=False)
    assert not (np.array(m.umech_id) == np.array(m2.umech_id)).any()

    m = make_example_meas()
    m2 = m.copy()
    m2.make_umechs_unique(same_uid=True)
    added = m2.umech_id[0].replace(m.umech_id[0], '')
    print(added)
    assert all([added in p for p in m2.umech_id])


def test_add_umech():
    import uuid

    uid = str(uuid.uuid4())
    m = make_example_meas()
    m.add_umech('tes1', m.nom)
    with pytest.raises(ValueError):
        m.add_umech('tes1', m.nom)
    m.add_umech('test2', m.cov[0, ...])
    m.add_umech('test3', m.cov[[0], ...])


def test_nom():
    m = make_example_meas(N_mc_samples=10)
    test = m.copy()
    test.cov = None
    xr.testing.assert_allclose(test.nom, m.mc.mean(dim='sample_id'))

    test = m.copy()
    test.mc = None
    test.nom

    test = m.copy()
    test.mc = 1
    test.nom

    with pytest.raises(RMEMeasFormatError):
        test = m.copy()
        test.cov = 1
        test.mc = xr.DataArray([0])
        test.nom

    with pytest.raises(RMEMeasFormatError):
        test = m.copy()
        test.mc = None
        test.cov = None
        test.nom


def test_umech_id_attr():
    with pytest.raises(RMEMeasFormatError):
        test = make_example_meas(N_mc_samples=10)
        test.cov = None
        test.umech_id


def test_confint(capsys):
    m = from_dist('dummy', 0, 1.0, dist='gaussian')
    cl, cu = m.confint(0.95)
    assert np.isclose(cu, 1.96, atol=0.01)

    cl, cu = m.confint(0.95, rad=True)
    assert np.isclose(cu, 1.96, atol=0.01)

    with pytest.raises(ValueError):
        m.confint(0.95, rad=True, deg=True)

    assert capsys.readouterr().out == ''


def test_dof_fails():
    m = from_dist('dummy', 0, 1.0, dist='gaussian')
    with pytest.raises(ValueError):
        m.dof(rad=True, deg=True)


def test_uncbounds():
    m = from_dist('dummy', 0, 1.0, dist='gaussian')
    ub = m.uncbounds(k=1)
    lb = m.uncbounds(k=-1)
    assert np.isclose(ub.cov, 1)

    m2 = m.copy()
    m2.cov = None
    ub = m2.uncbounds(k=1)
    assert ub.cov is None

    m2 = m.copy()
    m2.mc = None
    ub = m2.uncbounds(k=1)
    assert ub.mc is None


def test_assign_categories_to_all():
    m = from_dist('dummy', 0, 1.0, dist='gaussian')
    m.assign_categories_to_all(**{'Type': 'DD'})
    assert (m.covcats.sel(categories='Type') == 'DD').all()


def test_assign_categories():
    m = from_dist('dummy', 0, 1.0, dist='gaussian')
    uid = m.add_umech('umech', m.nom)
    m.assign_categories([uid], ['category'], ['thing'])
    assert m.covcats.sel(umech_id=uid, categories='category') == 'thing'


def test_create_empty_categories():
    m = from_dist('dummy', 0, 1.0, dist='gaussian')
    m.create_empty_categories('a')
    assert 'a' in m.covcats.categories
    m.create_empty_categories(['a', 'b', 'c'])
    assert all([ci in m.covcats.categories for ci in ['a', 'b', 'c']])


def test_add_mc_sample_stores_the_first_stochastic_trial_at_zero():
    measurement = RMEMeas.from_nom('samples', xr.DataArray([100.0], dims='point'))
    first_draw = xr.DataArray([1.5], dims='point')
    second_draw = xr.DataArray([2.5], dims='point')

    measurement.add_mc_sample(first_draw)
    measurement.add_mc_sample(second_draw)

    assert measurement.mc.sizes['sample_id'] == 2
    np.testing.assert_array_equal(measurement.mc.sample_id, [0, 1])
    xr.testing.assert_equal(measurement.mc.isel(sample_id=0, drop=True), first_draw)
    xr.testing.assert_equal(measurement.mc.isel(sample_id=1, drop=True), second_draw)
    xr.testing.assert_equal(measurement.nom, xr.DataArray([100.0], dims='point'))


def test_mc_standard_uncertainty_uses_every_stored_stochastic_trial():
    measurement = RMEMeas(
        name='stochastic-only uncertainty',
        cov=xr.DataArray(
            [100.0], dims='umech_id', coords={'umech_id': ['nominal']}
        ),
        mc=xr.DataArray(
            [0.0, 2.0], dims='sample_id', coords={'sample_id': [0, 1]}
        ),
    )

    assert measurement.stdunc().mc.item() == pytest.approx(1.0)


def test_from_dist_sample_statistics_use_all_stochastic_trials():
    np.random.seed(1234)
    measurement = from_dist(
        name='sample statistics',
        nom=10.0,
        std=2.0,
        dist='normal',
        samples=12,
        use_sample_mean=True,
    )

    assert measurement.mc.sizes['sample_id'] == 12
    assert measurement.nom.item() == pytest.approx(
        measurement.mc.mean(dim='sample_id').item()
    )
    assert (measurement.cov.isel(umech_id=1) - measurement.nom).item() == pytest.approx(
        measurement.mc.std(dim='sample_id', ddof=1).item()
    )


def test_curvefit_uses_the_independent_nominal_to_seed_mc_fits(monkeypatch):
    calls = []

    def fake_curvefit(array, *args, **kwargs):
        calls.append(array.copy())
        leading = next(
            (dim for dim in ('umech_id', 'sample_id') if dim in array.dims),
            None,
        )
        if leading is None:
            coefficients = xr.DataArray(
                [array.mean().item()], dims='param', coords={'param': ['offset']}
            )
        else:
            coefficients = array.mean(dim='x').expand_dims(param=['offset'])
            coefficients = coefficients.transpose(leading, 'param')
        return xr.Dataset({'curvefit_coefficients': coefficients})

    monkeypatch.setattr(xr.DataArray, 'curvefit', fake_curvefit)
    measurement = RMEMeas(
        name='fit',
        cov=xr.DataArray(
            [[10.0, 10.0], [11.0, 11.0]],
            dims=('umech_id', 'x'),
            coords={'umech_id': ['nominal', 'source'], 'x': [0.0, 1.0]},
        ),
        mc=xr.DataArray(
            [[100.0, 100.0], [101.0, 101.0]],
            dims=('sample_id', 'x'),
            coords={'sample_id': [0, 1], 'x': [0.0, 1.0]},
        ),
    )

    measurement.curvefit('x', lambda x, offset: x + offset)

    assert len(calls) == 4
    np.testing.assert_allclose(calls[0], [10.0, 10.0])
    np.testing.assert_allclose(calls[2], [10.0, 10.0])
    np.testing.assert_allclose(calls[3].isel(sample_id=0), [100.0, 100.0])


def test_constructor_defaults_name_without_mutating_attrs():
    cov = xr.DataArray(
        [1.0],
        dims='umech_id',
        coords={'umech_id': ['nominal']},
    )
    attrs = {'nested': {'value': 1}}

    measurement = RMEMeas(cov=cov, attrs=attrs)
    measurement.attrs['nested']['value'] = 2

    assert measurement.name == 'RMEMeas'
    assert attrs == {'nested': {'value': 1}}


def test_validate_rejects_noncanonical_monte_carlo_structure():
    cov = xr.DataArray(
        [1.0],
        dims='umech_id',
        coords={'umech_id': ['nominal']},
    )
    bad_ids = RMEMeas(
        name='bad ids',
        cov=cov,
        mc=xr.DataArray(
            [1.0, 1.1],
            dims='sample_id',
            coords={'sample_id': [0, 2]},
        ),
    )
    with pytest.raises(RMEMeasFormatError, match='consecutive integers'):
        bad_ids.validate()

    stochastic_first_row = RMEMeas(
        name='stochastic first row',
        cov=cov,
        mc=xr.DataArray(
            [9.0, 1.1],
            dims='sample_id',
            coords={'sample_id': [0, 1]},
        ),
    )
    stochastic_first_row.validate()

    layout_cov = xr.DataArray(
        [[1.0, 2.0]],
        dims=('umech_id', 'point'),
        coords={'umech_id': ['nominal'], 'point': [0, 1]},
    )
    bad_layout = RMEMeas(
        name='bad layout',
        cov=layout_cov,
        mc=xr.DataArray(
            [[0.9, 2.1]],
            dims=('sample_id', 'point'),
            coords={'sample_id': [0], 'point': [0, 2]},
        ),
    )
    with pytest.raises(RMEMeasFormatError, match='dimensions and coordinates'):
        bad_layout.validate()


def test_validate_rechecks_metadata_after_in_place_edits():
    measurement = RMEMeas(
        name='metadata',
        cov=xr.DataArray(
            [1.0, 1.2],
            dims='umech_id',
            coords={'umech_id': ['nominal', 'source']},
        ),
    )
    measurement.covdofs = measurement.covdofs.assign_coords(
        umech_id=['different source']
    )

    with pytest.raises(RMEMeasFormatError, match='covdofs umech_id coordinates'):
        measurement.validate()


def test_stdunc_uses_complex_perturbation_magnitude():
    measurement = RMEMeas(
        name='complex uncertainty',
        cov=xr.DataArray(
            [1.0 + 2.0j, 1.3 + 2.4j],
            dims='umech_id',
            coords={'umech_id': ['nominal', 'complex source']},
        ),
    )

    assert measurement.stdunc().cov.item() == pytest.approx(0.5)


def test_empty_construction_is_valid():
    empty = RMEMeas()

    assert empty.name == 'RMEMeas'
    assert empty.cov is None
    assert empty.mc is None
    assert empty.validate() is empty


def test_legacy_monte_carlo_axis_is_validated_after_rename():
    legacy = RMEMeas(
        name='legacy Monte Carlo',
        mc=xr.DataArray(
            [1.0, 1.1],
            dims='umech_id',
            coords={'umech_id': [0, 1]},
        ),
    )

    assert legacy.mc.dims == ('sample_id',)
    assert legacy.validate() is legacy


if __name__ == '__main__':
    test_interp()
    test_indexing()
    test_nom()
    test_add_umech()
    test_assign_categories()
    test_from_dist()
    test_overload()
    # test_h5_encoding()
    test_validate_conventions()
    # a = np.array(1)s
    # print(a.size)
