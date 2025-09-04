from rmellipse.uobjects import RMEMeas, RMEMeasFormatError
from rmellipse.propagators import RMEProp
from rmellipse.utils import save_object
from rmellipse._test_collections.rmemeas import make_example_meas
from pathlib import Path
import numpy as np
import h5py
import xarray as xr
import pytest
# Local directory for finding tests (the directory of this file)
# Pytest is annoying on what the root actually is
# so this gives a consisten, known location to build paths from
# that is OS independent
LOCAL = Path(__file__).parents[0]

def test_from_dist():
    nom = 1
    std = 1
    with pytest.raises(ValueError):
        a = RMEMeas.from_dist(
            name = 'dummy',
            nom = 1,
            std = 1,
            dist = 'not supported'
        )

    a = RMEMeas.from_dist(
        name = 'dummy',
        nom = 1,
        std = 1,
        dist = 'gaussian',
        use_sample_mean = False,
        samples = 1000
        )
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)

    a = RMEMeas.from_dist(
        name = 'dummy',
        nom = 1,
        std = 1,
        dist = 'normal',
        use_sample_mean = False
        )
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)

    a = RMEMeas.from_dist(
        name = 'dummy',
        nom = 1,
        std = 1,
        dist = 'normal',
        use_sample_mean = False
        )
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)

    a = RMEMeas.from_dist(
        name = 'dummy',
        nom = 1,
        std = 1,
        dist = 'rectangular',
        use_sample_mean = False
        )
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)

    a = RMEMeas.from_dist(
        name = 'dummy',
        nom = 1,
        std = 1,
        dist = 'rectangular',
        use_sample_mean = False
        )
    assert np.isclose(a.nom, nom)
    assert np.isclose(a.stdunc().cov, std)
    # assert np.isclose(a.stdunc().mc, std, atol = .15)

def test_interp():
    # makes coordinate between 0 and 1
    m1 = make_example_meas(nom_shape=(3,3), rand_nom= True, rand_coords=True,N_mc_samples=10)
    out_of_bounds_coords = np.array([-.5,0.5,1.5])
    m1_oob = m1.interp(d1 = out_of_bounds_coords)
    assert all(m1_oob.cov.d1 == m1.cov.d1)

def test_validate_conventions():
    # this should fail because first label is wrong
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['not nominal','dummy'],
                                        'd1':[0]}),
            mc = None
        )
        bad_cov._validate_conventions()

    # this should fail because umech_id is in the wrong spot
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0,1]], dims = ('d1','umech_id'),
                               coords ={'umech_id':['nominal','dummy'],
                                        'd1':[0]}),
            mc = None
        )
        bad_cov._validate_conventions()

    # this should fail because covdofs plocs are named inproperly
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['nominal','dummy'],
                                        'd1':[0]}),
            covdofs = xr.DataArray([[]],dims = ('not umech_id', 'category'),coords = {'not umech_id':['not dummy']})
        )

        bad_cov._validate_conventions()

    # this should fail because covdofs plocs don't match cov
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['nominal','dummy'],
                                        'd1':[0]}),
            covdofs = xr.DataArray([[]],dims = ('umech_id', 'category'),coords = {'umech_id':['not dummy']})
        )

        bad_cov._validate_conventions()

    # this should fail because covdofs ploc dim is in the wrong spot
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['nominal','dummy'],
                                        'd1':[0]}),
            covdofs = xr.DataArray([[0],],dims = ( 'category','umech_id'),coords = {'umech_id':['not dummy']})
        )

        bad_cov._validate_conventions()

    # this should fail because first label is wrong
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            mc = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['not nominal','dummy'],
                                        'd1':[0]}),
            cov = None
        )
        bad_cov._validate_conventions()

    # this should fail because umech_id doesn't exist in mc
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            mc = xr.DataArray([[0,1]]),
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['nominal','dummy']})
        )
        print(bad_cov)
        bad_cov._validate_conventions()

    # this should fail because umech_id doesn't exist in cov
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            mc = xr.DataArray([[0,1]]),
            cov = xr.DataArray([[0],[1]], dims = ('not umech_id','d1'),
                               coords ={'not umech_id':['nominal','dummy']}),
            covdofs = 0,
            covcats = 0
        )
        print(bad_cov)
        bad_cov._validate_conventions()

    # this should fail because first label is wrong
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            mc = xr.DataArray([[0,1]], dims = ('d1','umech_id'),
                               coords ={'umech_id':['not nominal','dummy'],
                                        'd1':[0]}),
            cov = None
        )
        bad_cov._validate_conventions()


        # this should fail because covdofs plocs are named inproperly
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['nominal','dummy'],
                                        'd1':[0]}),
            covcats = xr.DataArray([[]],dims = ('not umech_id', 'category'),coords = {'not umech_id':['not dummy']})
        )

        bad_cov._validate_conventions()

    # this should fail because covdofs plocs don't match cov
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['nominal','dummy'],
                                        'd1':[0]}),
            covcats = xr.DataArray([[]],dims = ('umech_id', 'category'),coords = {'umech_id':['not dummy']})
        )

        bad_cov._validate_conventions()

    # this should fail because plocs is in the wrong spot
    with pytest.raises(RMEMeasFormatError):
        bad_cov = RMEMeas(
            name = 'bad',
            cov = xr.DataArray([[0],[1]], dims = ('umech_id','d1'),
                               coords ={'umech_id':['nominal','dummy'],
                                        'd1':[0]}),
            covcats = xr.DataArray([[0],],dims = ( 'category','umech_id'),coords = {'umech_id':['not dummy']})
        )

        bad_cov._validate_conventions()


def test_overload():
    prop = RMEProp(sensitivity=True)
    prop.set_active()
    m1 = make_example_meas(nom_shape=(1,), rand_nom = True)
    m2 = make_example_meas(nom_shape=(1,), rand_nom = True)
    # these are ndarrys, they will overload into __array_ufunc__
    nd1 = m1.nom[0].values
    nd2 = m2.nom[0].values
    # floats
    f1 = float(nd1)
    f2 = float(nd2)

    #__array_unfunc__
    assert np.isclose((nd1*m2).nom[0], f1*f2)
    assert np.isclose(np.sin(m1).nom[0], np.sin(f1))

    # __add__
    assert np.isclose((m1+m2).nom[0], f1+f2)

    # __radd__
    assert np.isclose((f1+m2).nom[0], f1+f2)

    # __mul__
    assert np.isclose((m1*m2).nom[0], f1*f2)

    # __rmul__
    assert np.isclose((f1*m2).nom[0], f1*f2)

    # __sub__
    assert np.isclose((m1-m2).nom[0], f1-f2)

    # __rsub__
    assert np.isclose((f1-m2).nom[0], f1-f2)

    # __trudiv__
    assert np.isclose((m1/m2).nom[0], f1/f2)

    # __rtruediv__
    assert np.isclose((f1/m2).nom[0], f1/f2)

    # __pow__
    assert np.isclose((m1**m2).nom[0], f1**f2)

    # __rtruediv__
    assert np.isclose((f1/m2).nom[0], f1/f2)

def test_xml():
    m1 = make_example_meas(nom_shape=(2, 2),N_mc_samples=100)
    m1.name = 'test_name'

    def to_txt(data, path):
        np.savetxt(path, data.values, delimiter=',')

    def from_txt(path):
        values = xr.DataArray(np.loadtxt(path, float, delimiter=','))
        return values
    target = LOCAL / 'mutable'
    print(m1.mc)
    m1.to_xml(
        str(target.resolve()),
        to_txt,
        data_extension='.csv',
        header_extension='.meas'
    )

    m2 = RMEMeas.from_xml(
        str(str(target / 'test_name.meas')),
        from_csv=from_txt
    )
    assert (m2.cov.values == m1.cov.values).all()

def test_copy():
    m1 = make_example_meas(nom_shape=(2, 2),N_mc_samples=100)

    test = RMEMeas(m1.name,m1.cov,m1.mc)
    copy = test.copy()
    assert (test.cov == m1.cov).all()
    assert (test.mc == copy.mc).all()
    assert (test.covdofs == copy.covdofs).all()
    assert (test.covcats == copy.covcats).all()

    test = RMEMeas(m1.name,m1.cov,None)
    copy = test.copy()
    assert (test.cov == m1.cov).all()
    assert (test.mc == copy.mc)
    assert (test.covdofs == copy.covdofs).all()
    assert (test.covcats == copy.covcats).all()

def test_repr():
    m1 = make_example_meas()
    assert str(m1) == m1.__repr__()


def test_stdunc():

    m1 = make_example_meas(
        lin_unc=1,
        mc_unc=1,
        nom_shape=(1,),
        rand_nom=False,
        N_mc_samples=100
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
        m1.stdunc(deg=True, rad = True)
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

    test = RMEMeas.from_dist('test', [1, 1, 1], [0, 0, 0])
    test.cull_cov()
    assert len(test.umech_id) == 0
    test = test + 2

    # check that setting tolerance above 0.1 keeps mechanisms with tolerance above 0.1
    test = RMEMeas.from_dist('test', [1, 1, 1], [.1, .1, .1], use_sample_mean=False)
    test.add_umech('thing', test.nom + np.array([.1, .01, .01]))
    test.cull_cov(tolerance=0.3)
    assert len(test.umech_id) == 1


def test_h5_encoding():
    test = make_example_meas()
    print('the local path is ', LOCAL)
    with h5py.File(LOCAL / 'mutable/tests.hdf5', 'w') as hf:
        test.to_h5(hf)
        # saving an existing object should fail
        # with an error message saying it already exists
        try:
            test.to_h5(hf)
        except ValueError as e:
            assert 'exists' in str(e)
        test.to_h5(hf, override=True)
        print(test.name)
        read = RMEMeas.from_h5(hf[test.name])
        read_nom = RMEMeas.from_h5(hf[test.name], nominal_only = True)
    assert (read.cov == test.cov).all()
    assert read_nom.cov.shape[0] == 1
    assert (read_nom.nom == read.nom).all()

    # check tthe name functon
    with h5py.File(LOCAL / 'mutable/tests.hdf5', 'w') as hf:
        oldname = test.name
        test.to_h5(hf, name = 'testname')
        # saving an existing object should fail
        # with an error message saying it already exists
        assert test.name == oldname
        with pytest.raises(AttributeError):
            test.to_h5('asf', override=True)
            assert test.name == oldname
            print(test.name)
        read = RMEMeas.from_h5(hf['testname'])
        read_nom = RMEMeas.from_h5(hf['testname'], nominal_only = True)
    assert (read.cov == test.cov).all()
    assert read_nom.cov.shape[0] == 1
    assert (read_nom.nom == read.nom).all()

    # check that MUFmeas objects read work
    with h5py.File(LOCAL/'const'/'24splitter_proto.h5','r') as hf:
        old = RMEMeas.from_h5(hf['24mm_clrmproto_splitter'])
        with h5py.File(LOCAL / 'mutable/MUFmeas_write_test.hdf5', 'w') as hf2:
            old.to_h5(hf2,override = True)




def test_h5_group_to_dict():
    test = make_example_meas()
    print('the local path is ', LOCAL)
    names = ['n1','n2','n3']
    path = LOCAL / 'mutable/tests.hdf5'

    # should return anything saved in the base directory
    with h5py.File(path, 'w') as hf:
        for n in names:
            test.name = n
            test.to_h5(hf)
    d = RMEMeas.dict_from_group(path)
    for (k,v),n in zip(d.items(), names):
        assert k == n

    # should work with groups, and should ignore something thats
    # not a MUFmeas
        with h5py.File(path, 'w') as hf:
            grp = hf.require_group('mygroup')
            # make something that isn't an RME obejct
            dset = grp.create_dataset("default", (100,))
            for n in names:
                test.name = n
                test.to_h5(grp)
    d = RMEMeas.dict_from_group(path,group_path = 'mygroup')
    for (k,v),n in zip(d.items(), names):
        assert k == n

def test_indexing():
    test = RMEMeas.from_dist('test', [1, 1, 1], [0, 0, 0])
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

    raised = False
    try:
        test.usel(mcsamples=[0])
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        test.usel(umech_id=['nominal'])
    except ValueError:
        raised = True
    assert raised

    usel = test.usel(umech_id=[test.umech_id[0]])
    assert len(usel.umech_id) == 1
    # no cov samples
    usel = test.usel(umech_id=[])
    assert len(usel.umech_id) == 0
    usel = test.usel(mcsamples=[2, 3])
    assert usel.mc.shape[0] == 3  # nominal plus 2 samples
    usel = test.usel(umech_id=[], mcsamples=[2, 3])
    assert usel.mc.shape[0] == 3 and len(usel.umech_id) == 0

def test_make_umechs_unique():
    m = make_example_meas()
    m2 = m.copy()
    m2.make_umechs_unique(same_uid = False)
    assert not (np.array(m.umech_id) == np.array(m2.umech_id)).any()

    m = make_example_meas()
    m2 = m.copy()
    m2.make_umechs_unique(same_uid = True)
    added = m2.umech_id[0].replace(m.umech_id[0],'')
    print(added)
    assert all([added in p for p in m2.umech_id])

def test_add_umech():
    m = make_example_meas()
    m.add_umech('tes1',m.nom)
    with pytest.raises(ValueError):
         m.add_umech('tes1',m.nom)
    m.add_umech('test2',m.cov[0,...])
    m.add_umech('test3',m.cov[[0],...])

def test_nom():
    m = make_example_meas(N_mc_samples=10)
    test = m.copy()
    test.cov = None
    test.nom

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

def test_confint():
    m = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian')
    cl,cu = m.confint(0.95)
    assert np.isclose(cu,1.96,atol = 0.01)

    cl,cu = m.confint(0.95, rad = True)
    assert np.isclose(cu,1.96,atol = 0.01)

    with pytest.raises(ValueError):
        m.confint(.95,rad = True, deg = True)

def test_dof_fails():
    m = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian')
    with pytest.raises(ValueError):
        m.dof(rad = True, deg = True)

def test_uncbounds():
    m = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian')
    ub = m.uncbounds(k = 1)
    lb = m.uncbounds(k = -1)
    assert np.isclose(ub.cov,1)

    m2 = m.copy()
    m2.cov = None
    ub = m2.uncbounds(k = 1)
    assert ub.cov is None

    m2 = m.copy()
    m2.mc = None
    ub = m2.uncbounds(k = 1)
    assert ub.mc is None

def test_assign_categories_to_all():
    m = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian')
    m.assign_categories_to_all(**{'Type':'DD'})
    assert (m.covcats.sel(categories = 'Type') == 'DD').all()

def test_assign_categories():
    m = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian')
    m.add_umech('umech',m.nom)
    m.assign_categories(['umech'],['category'],['thing'])
    assert m.covcats.sel(umech_id = 'umech', categories = 'category') == 'thing'

def test_create_empty_categories():
    m  = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian')
    m.create_empty_categories('a')
    assert 'a' in m.covcats.categories
    m.create_empty_categories(['a','b','c'])
    assert all([ci in m.covcats.categories for ci in ['a','b','c']] )

def test_grouping():
    m1 = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian',mechanism_name = 'a')
    m2 = RMEMeas.from_dist('dummy',0,1.0,dist = 'gaussian',mechanism_name = 'b')
    prop = RMEProp(sensitivity = True)
    m3 = prop.combine(m1,m2)
    m3.assign_categories(['a','b'],['type','type'],['a','b'])
    m3.assign_categories(['a'],['something','other'],['a','b'])
    m3.dof()
    a = m3.stdunc().cov
    b = m3.categorize_by('type').stdunc().cov
    c = m3.categorize_by('something').stdunc().cov
    assert (a == b).all()
    assert (a == c).all()
    m4 = m3.copy()
    m4.mc = None
    a = m4.stdunc().cov
    b = m4.categorize_by('type').stdunc().cov
    c = m4.categorize_by('something').stdunc().cov
    assert (a == b).all()
    assert (a == c).all()

if __name__ == '__main__':
    test_overload()
    test_h5_encoding()
    test_validate_conventions()
    a = np.array(1)
    print(a.size)