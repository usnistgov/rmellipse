"""
Unit test for propagator-related thing, such as the handle grid method, the uid-ducks, and output

If something here failed, it needs to be fixed :()

"""

from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
from rmellipse._test_collections.rmemeas import make_example_meas
import numpy as np
import xarray as xr
import pytest
# import h5py


def test_magic():
    a = RMEProp()
    assert a.__repr__() == a.__str__()

def test_sample_distributions():
    m = make_example_meas(N_mc_samples=100)
    flt = 1.0
    sampled = RMEProp._sample_distribution(10,flt)
    assert sampled == flt
    sampled = RMEProp._sample_distribution(10,m)
    # shoul return 1 + number of samples (acounting for nominal)
    assert len(sampled.umech_id) == 11

def test_MIMO_vectorized():
    m1 = make_example_meas(N_mc_samples=100)
    m2 = make_example_meas(N_mc_samples=100)
    prop = RMEProp(sensitivity = True, montecarlo_sims=100)
    @prop.propagate
    def passthru(*args):
        return args
    r1,r2 = passthru(m1,m2)
    assert (r1.nom == m1.nom).all()
    assert (r2.cov == m2.cov).all()

def test_MIMO_unvectorized():
    def check_same(in1,out1):
        if out1.cov is not None:    
            assert np.isclose(out1.cov[0,...],  in1.cov[0,...], atol = 1e-10).all()
        # all the samples in r1.mc should be in m1.mc
        if out1.mc is not None:
            for val in out1.mc.values.flatten():
                assert np.isin(val,in1.mc.values) 

    m1 = make_example_meas(nom_shape=(1,),N_mc_samples=2)
    m2 = make_example_meas(nom_shape=(1,),N_mc_samples=2)
    prop = RMEProp(sensitivity = True, montecarlo_sims=2, vectorize= False)

    # both at the same time
    def passthru(*args):
        return args
    pass2 = prop.propagate(passthru)
    # print(pass2(m1))
    r1, = pass2(m1)
    check_same(m1,r1)
    r1,r2 = pass2(m1,m2)
    check_same(m1,r1)
    check_same(m2,r2)

    # sensitivity only
    prop.settings['montecarlo_sims'] = 0
    prop.settings['sensitivity'] = True

    # print(pass2(m1))
    r1, = pass2(m1)
    check_same(m1,r1)
    assert r1.mc is None
    r1,r2 = pass2(m1,m2)
    check_same(m1,r1)
    check_same(m2,r2)

    # montecarlo only
    prop.settings['montecarlo_sims'] = 2
    prop.settings['sensitivity'] = False
    
    # print(pass2(m1))
    r1, = pass2(m1)
    check_same(m1,r1)
    r1,r2 = pass2(m1,m2)
    check_same(m1,r1)
    check_same(m2,r2)


def test_covcats_collisions():
    m1 = make_example_meas()
    m1.assign_categories([m1.umech_id[0]],['Type'],['A'])
    m2 = m1.copy()
    m1.assign_categories([m1.umech_id[0]],['Type'],['B'])

    prop = RMEProp(sensitivity = True, montecarlo_sims=2, vectorize= False)
    # both at the same time
    @prop.propagate
    def passthru(*args):
        return args

    # this should raise a warning because we have different category information
    # for the same mechanism
    with pytest.warns(UserWarning):
        r1,r2 = passthru(m1,m2)

    #callikng get_new_categories with not sensitivity (False in the func) should return 
    # None
    should_be_none = RMEProp._get_new_categories('a',[],{},False,True)
    assert should_be_none is None
def test_common_grid_interp_common():
    m1 = make_example_meas(rand_coords=True)
    m2 = make_example_meas(rand_coords=True)
    mref = RMEMeas.from_nom(name = 'empty', nom = xr.DataArray([0]))
    interp_grid = np.array([0, 0.1, 0.2])

    prp = RMEProp(
        sensitivity=True,
        common_grid='d1',
        common_coords={'d1': interp_grid},
        handle_common_grid_method='interp_common'
    )

    @prp.propagate
    def passthru(*args):
        return args

    r1, r2, rref = passthru(m1, m2, mref)

    # check reference didn't change:
    assert all(rref.nom == mref.nom)

    # check that we interpolated properly
    for r,m in zip([r1, r2],[m1,m2]):
        # we should be on the correct grid
        assert all(r.cov.d1.values == interp_grid)
        # we should have the same dimension order
        assert r.nom.dims == m.nom.dims
    pass


def setup_VI(N=5):
    """Get 2 measurements to test things with"""
    V = RMEMeas.from_dist(
        name='voltage',
        nom=2,
        std=0.01,
        samples=N,
        dist='gaussian',
        use_sample_mean=False
    )

    I = RMEMeas.from_dist(
        name='current',
        nom=1.5,
        std=0.01,
        samples=N,
        dist='gaussian',
        use_sample_mean=False
    )
    return V, I


def test_mc_linear_propagate_options():
    """
    Test the sensitivit and linear options for a MUF propagator.


    """
    np.random.seed = 0
    N = 5

    myprop = RMEProp(
        montecarlo_sims=N,
        sensitivity=False,
        vectorize=False,
        verbose=False)

    V, I = setup_VI()

    def multiply(v, i):
        return v * i

    mult = myprop.propagate(multiply)

    # check that passing in normal numbers works
    test1 = mult(2, 1.5)
    assert test1 == 3
    # %% With vectorization
    myprop.settings['vectorize'] = True

    # check that sensitivity and montecarlo sims together works
    myprop.settings['sensitivity'] = True
    myprop.settings['montecarlo_sims'] = N

    test1 = mult(V, I)
    assert test1.mc.shape == (N + 1,)
    assert test1.cov.shape == (3,)
    assert test1.nom == 3
    test1.umech_id

    # check that sensitivity and montecarlo off behaves normally
    # shoulbe be a nominal in the cov att, no sensitivity info
    myprop.settings['sensitivity'] = False
    myprop.settings['montecarlo_sims'] = N

    test1 = mult(V, I)
    assert test1.mc.shape == (N + 1,)
    assert test1.cov.shape == (1,)
    assert test1.nom == 3
    test1.umech_id

    # check thatturning sensitivity on and montecarlo off gives nothing
    # in the mc data and leaves the sensitvity alone
    myprop.settings['sensitivity'] = True
    myprop.settings['montecarlo_sims'] = 0

    test1 = mult(V, I)
    assert test1.mc is None
    assert test1.cov.shape == (3,)
    assert test1.nom == 3
    test1.umech_id

    # turn both off, should be None in mc and (1,) in cov
    myprop.settings['sensitivity'] = False
    myprop.settings['montecarlo_sims'] = 0

    test1 = mult(V, I)
    assert test1.mc is None
    assert test1.cov.shape == (1,)
    assert test1.nom == 3
    test1.umech_id
    # %% Without vectorization
    myprop.settings['vectorize'] = False

    # check that sensitivity and montecarlo sims together works
    myprop.settings['sensitivity'] = True
    myprop.settings['montecarlo_sims'] = N

    test1 = mult(V, I)
    assert test1.mc.shape == (N + 1,)
    assert test1.cov.shape == (3,)
    assert test1.nom == 3
    test1.umech_id

    # check that sensitivity and montecarlo off behaves normally
    # shoulbe be a nominal in the cov att, no sensitivity info
    myprop.settings['sensitivity'] = False
    myprop.settings['montecarlo_sims'] = N

    test1 = mult(V, I)
    assert test1.mc.shape == (N + 1,)
    assert test1.cov.shape == (1,)
    assert test1.nom == 3
    test1.umech_id

    # check thatturning sensitivity on and montecarlo off gives nothing
    # in the mc data and leaves the sensitvity alone
    myprop.settings['sensitivity'] = True
    myprop.settings['montecarlo_sims'] = 0

    test1 = mult(V, I)
    assert test1.mc is None
    assert test1.cov.shape == (3,)
    assert test1.nom == 3
    test1.umech_id

    # turn both off, should be None in mc and (1,) in cov
    myprop.settings['sensitivity'] = False
    myprop.settings['montecarlo_sims'] = 0

    test1 = mult(V, I)
    assert test1.mc is None
    assert test1.cov.shape == (1,)
    assert test1.nom == 3
    test1.umech_id

def test_null():
    prop = RMEProp(sensitivity = False, montecarlo_sims=0, verbose = True)
    m1 = None
    @prop.propagate
    def passthru(a):
        return None
    assert passthru(m1) is None

    prop.settings['sensitivity'] = True
    assert passthru(m1) is None

    prop.settings['sensitivity'] = False
    prop.settings['montecarlo_trials'] = 10
    assert passthru(m1) is None

def test_common_grid_handling():
    m1 = make_example_meas(nom_shape=(3,3), rand_nom= True, rand_coords=False ,N_mc_samples=10)
    m2 = m1.copy()[1:]

    
    prop = RMEProp(
        sensitivity = True,
        montecarlo_sims = 1,
        common_grid='d1',
        handle_common_grid_method = 'common'
        )
    @prop.propagate
    def passthru(*args):
        return args

    # test the common mode, this should be the
    # common mode
    coord_ref = m2.cov.coords['d1'] 
    r1,r2 = passthru(m1,m2)
    assert (r1.cov.d1 == coord_ref).all()
    assert (r2.cov.d1 == coord_ref).all()

    # test the interp mode
    prop.settings['common_coords'] = {'d1':coord_ref}
    prop.settings['handle_common_grid_method'] = 'interp_common'
    r1,r2 = passthru(m1,m2)
    assert (r1.cov.d1 == coord_ref).all()
    assert (r2.cov.d1 == coord_ref).all()

def test_combine():
    def make():
        return make_example_meas(nom_shape=(3,3), rand_nom= True, rand_coords=False ,N_mc_samples=10)
    m1 = make()
    m2 = make()

    prop = RMEProp(
        sensitivity = True,
        montecarlo_sims = 10,
        verbose = True
        )
    
    # check the average is right, and check that combine IDs are assigned properly
    basename = 'test combine'
    m12 = prop.combine(m1,m2, combine_basename=basename, combine_categories={'Origin':'Test'},add_uuid=False)
    mean = np.mean([m1.nom,m2.nom], axis = 0)
    assert np.isclose(m12.nom,mean).all()
    # check that all the umech_id with the declared basename share a combine_id
    cid = []
    for p in m12.umech_id:
        if basename in p:
            cid.append(m12.covcats.sel(categories = 'combine_id',umech_id = p))
    assert all([cid[0] == ci for ci in cid])


    # check extra features
    def make2():
        return make_example_meas(nom_shape=(10,), rand_nom= True, rand_coords=False ,N_mc_samples=10)
    measurements = [make2() for i in range(10)]
    m12 = prop.combine(*measurements,n_single_values = .9, error_of_mean = True, combine_basename = basename)
    err_count = np.sum([basename in p for p in m12.umech_id])
    # we are only keeping 90% of variance, so we should have less erro vectors
    # than their are measurands (10 in this case)
    assert err_count < 10

    measurements = [make2() for i in range(10)]
    m12 = prop.combine(*measurements,n_single_values = 3, error_of_mean = True, combine_basename = basename)
    err_count = np.sum([basename in p for p in m12.umech_id])
    # in this case we are specifying to only keep 3 error vectors
    assert err_count ==3

    # this should fail because of the -1
    with pytest.raises(ValueError):
        prop.combine(*measurements,n_single_values = -1, error_of_mean = True, combine_basename = basename)
    
    # testing that the algorithm can handle when one of the
    # dimensions it creatas/then reduces across when concatenating
    # already exists. The generate error vector functions
    # creats a dim called 'measdim' to average across and
    # generate PCA error vectors. Just checking it 
    # can handle someone else using that dim name
    @prop.propagate
    def rename(m):
        return m.rename({'d1':'measdim'})
    measurements = [rename(make2()) for m in measurements]
    prop.combine(*measurements,n_single_values = 3, error_of_mean = True, combine_basename = basename)
    assert err_count ==3

    # passing a single object just returns it
    a = RMEMeas.from_nom('name', xr.DataArray([0]))
    b = prop.combine(a)
    assert (a.cov == b.cov).all()

    b = prop.combine(a,a)

    with pytest.raises(ValueError):
        prop.combine()

if __name__ == '__main__':
    test_combine()
    pass