from rmellipse.utils import save_object, load_object, GroupSaveable
from pathlib import Path
import pytest
import h5py as h5
import xarray as xr
from rmellipse.uobjects import RMEMeas

LOCAL = Path(__file__).parents[0]
MUTABLE_DIR= LOCAL/'mutable'
TEST_FILE = MUTABLE_DIR/'group_saveable_test.hdf5'
def myfunc():
    return 1

class Saveable(GroupSaveable):
    def __init__(self, name: str):
        GroupSaveable.__init__(       
            self,     
            name=name,
            parent=None,
            attrs={'name': name, 'is_big_object': False}
        )
        self.vals = [1,2,3]
        self.add_child('vals',self.vals)


        # no key, not inferred
        with pytest.raises(ValueError):
            self.add_child(data = 1)        

        # key doesn't match
        with pytest.raises(ValueError):
            self.add_child(key = 'bad',data = SaveableIsh('name'))
        self.kid = SaveableIsh('kid')
        self.add_child(key = 'kid', data = self.kid )
    def __eq__(self, other):
        return all([si == oi for si,oi in zip(self.vals,other.vals)])

class Childless(GroupSaveable):
    def __init__(self, parent = None, attrs = None):
        if attrs is None: 
            attrs = {'name':'i_am_bereft', 'is_big_object':False,'my_metadata':'filepathprobably'}
        GroupSaveable.__init__(       
            self,     
            parent=None,
            attrs=attrs
        )


class SaveableIsh(GroupSaveable):
    def __init__(self, name: str):
        GroupSaveable.__init__(       
            self,     
            name=name,
            parent=None,
            attrs={'name': name, 'is_big_object': False}
        )  
        self.int = 1
        self.add_child('int', self.int)


def test_primatives():
    prims = [
        'asd',1, 1.0, 'asf', bool
    ]

    with h5.File(TEST_FILE,'w') as f:
        for p in prims:
            print('saving ', p)
            group = save_object(f, 'myobj', p, verbose = True)
            read = load_object(group)
            print(read)
            assert p == read
            del f['myobj']

def test_lists():
    prims = [
        [1,2,3], ['True', True]
    ]

    with h5.File(TEST_FILE,'w') as f:
        for p in prims:
            # print('saving ', p)
            group = save_object(f, 'myobj', p, verbose = True)
            read = load_object(group)
            
            for pi,ri in zip(p,read):
                assert pi == ri 
            del f['myobj']

def test_datasets():
    import numpy as np
    prims = [
        np.array([1,2,3]), np.array(['1','2','3']), np.array([True,False])
    ]

    with h5.File(TEST_FILE,'w') as f:
        for p in prims:
            # print('saving ', p)
            group = save_object(f, 'myobj', p, verbose = True)
            read = load_object(group)
            
            for pi,ri in zip(p,read):
                assert pi == ri 
            del f['myobj']

def test_dicts():
    prims = [
        {'1':1,'2':2},{1:1,2:2},{1:1,'1':2}
    ]
    with h5.File(TEST_FILE,'w') as f:
        grp = f.require_group('dicts')
        for i,p in enumerate(prims):
            print('saving ', p)
            group = save_object(grp, str(i), p, verbose = True)
            read = load_object(group)
            print(read)
            assert all([kp == kr and vp == vr for (kp,vp),(kr,vr) in zip(p.items(),read.items())])

def test_functions():
    def get():
        import numpy as np
        return np.sin
    p = get()
    testval = p(0)
    with h5.File(TEST_FILE,'w') as f:
        #save and load normally
        group = save_object(f, 'myobj', p, verbose = True)
        read = load_object(group)
        print(read)
        assert read(0) == testval
        del f['myobj']

def test_DataArrays():
    import numpy as np
    prims = [
        xr.DataArray([1,2,3], dims = ('d1'), coords = {'d1':[0.1,0.2,0.3]}),
        xr.DataArray([1,2,3], dims = ('d1'), coords = {'d1':['1','2','3']})
    ]

    with h5.File(TEST_FILE,'w') as f:
        for p in prims:
            # print('saving ', p)
            group = save_object(f, 'myobj', p, verbose = True)
            read = load_object(group)
            
            for pi,ri in zip(p,read):
                assert pi == ri 
            assert p.dims == read.dims
            assert (p.d1 == read.d1).all()
            del f['myobj']

def test_class():
    test = Saveable('a')
    assert test['kid'].int == test.kid.int
    test2 = Saveable('b')
    assert test == test2
    with h5.File(TEST_FILE,'w') as f:
        test.save(f, verbose = True)
        test2.save(f)
    # I don't know how to make this load properly.

def test_slice():
    prims = [
        'asd',slice(2)
    ]

    with h5.File(TEST_FILE,'w') as f:
        for p in prims:
            print('saving ', p)
            group = save_object(f, 'myobj', p, verbose = True)
            read = load_object(group)
            print(read)
            assert p == read
            del f['myobj']

def test_rmemeas():
    import rmellipse.uobjects as obj
    a0 = obj.RMEMeas.from_dist('a',1,1)
    with h5.File(TEST_FILE,'a') as f:
        save_object(f,a0.name,a0)
        a1 = load_object(f['a'], load_big_objects=True)
    assert (a1.cov == a0.cov.data).all()
    assert (a1.mc == a0.mc.data).all()

def test_rmemeas_list():
    import rmellipse.uobjects as obj
    a0 = obj.RMEMeas.from_dist('a',1,1)
    b0 = obj.RMEMeas.from_dist('a',1,1)

    my_list = [a0, b0]
    with h5.File(TEST_FILE,'a') as f:
        save_object(f,'my_list',my_list)
        ml1 = load_object(f['my_list'], load_big_objects=True)
    
    for ai,bi in zip(my_list, ml1):
        print(bi.name, ai.name)
        assert ai.name == bi.name
        assert (ai.cov == bi.cov.data).all()
        assert (ai.mc == bi.mc.data).all()

def test_rmemeas_dict():
    import rmellipse.uobjects as obj
    a0 = obj.RMEMeas.from_dist('a',1,1)
    b0 = obj.RMEMeas.from_dist('b',1,1)

    my_dict = {a0.name: a0, b0.name: b0}
    with h5.File(TEST_FILE,'a') as f:
        save_object(f,'my_d',my_dict)
        myd = load_object(f['my_d'], load_big_objects=True)
        a1 = RMEMeas.from_h5(f['my_d']['a'])
    
    for ai,bi in zip(my_dict.values(), myd.values()):
        print(bi.name, ai.name)
        assert ai.name == bi.name
        assert (ai.cov == bi.cov.data).all()
        assert (ai.mc == bi.mc.data).all()
    
    for ai,bi in zip(my_dict.keys(), myd.keys()):
        assert ai == bi

def test_childless():
    widow = Childless()
    with h5.File(TEST_FILE,'a') as f:
        save_object(f,'widow',widow)
        load_object(f['widow'])

if __name__ == '__main__':
    with h5.File(TEST_FILE,'w') as f:
        pass
    test_rmemeas()
    test_rmemeas_list()
    test_rmemeas_dict()
    test_childless()