import rmellipse.arrschema as arrschema
from rmellipse.utils import load_object, save_object
import xarray as xr
import numpy as np
from rmellipse.uobjects import RMEMeas
import pytest
import h5py
from pathlib import Path

LOCALS = Path(__file__).parents[0]
ARRAY_SAMPLES = LOCALS / 'arrsamples'
REGISTRY = arrschema.ArrayClassRegistry()
MUTABLE = LOCALS / 'mutable'

empty_float = arrschema.ArraySchema('empty', (...,), (...,), float)
# basic float
with pytest.raises(Exception):
    mismatched_dims = arrschema.ArraySchema('empty', (..., 'N'), (...,), float)

s2p_ri_schema = arrschema.ArraySchema(
    name='s2p_ri',
    shape=(..., 'N', 8),
    dims=(..., 'frequency', 'col'),
    dtype=float,
    coords={
        'frequency': {
            'units': 'GHz',
            'dtype': float,
        },
        'col': {
            'values': [
                'Re(S11)',
                'Im(S11)',
                'Re(S12)',
                'Im(S12)',
                'Re(S21)',
                'Im(S21)',
                'Re(S22)',
                'Im(S22)',
            ],
            'dtype': 'U8',
        },
    },
    attrs_schema={
        'type': 'object',
        'properties': {
            'frequency_units': {
                'enum': ['GHz', 'Hz'],
            }
        },
        'required': ['frequency_units'],
    },
)

zeros_schema = arrschema.ArraySchema(
    name='zeros', shape=(...,), dims=(...,), dtype=float
)

zeros_complex_schema = arrschema.ArraySchema(
    name='ZEROS_COMPLEX', shape=(...,), dims=(...,), dtype=complex
)

float_2by2_schema = arrschema.ArraySchema(
    name='float_2by2',
    shape=(3, ..., 2, 2),
    dims=('d0', ..., 'd1', 'd2'),
    dtype=float,
    coords={
        'd0': {'values': [0, 1, 2], 'dtype': float},
        'd1': {'values': [0, 1], 'dtype': int},
        'd2': {'dtype': int},
    },
)


# S2P_RI = REGISTRY.build_and_add_class(s2p_ri)
class S2PRI(arrschema.AnnotatedArray):
    schema = s2p_ri_schema
    registry = REGISTRY


class Zeros(arrschema.AnnotatedArray):
    schema = zeros_schema
    registry = REGISTRY


class ZerosComplex(arrschema.AnnotatedArray):
    schema = zeros_complex_schema
    registry = REGISTRY


class Float2By2(arrschema.AnnotatedArray):
    schema = float_2by2_schema
    registry = REGISTRY


REGISTRY.add_loader(
    'rmellipse.arrschema.examples:load_csv_like_s2p_ri',
    '.s2p',
    loader_type='csv',
    schema=s2p_ri_schema,
)

REGISTRY.add_loader(
    'rmellipse.arrschema.examples:load_group_saveable',
    ['.h5', '.hdf5'],
    loader_type='group_saveable',
    schema=s2p_ri_schema,
)

REGISTRY.add_saver(
    'rmellipse.arrschema.examples:save_group_saveable',
    ['.h5', '.hdf5'],
    saver_type='group_saveable',
    schema=s2p_ri_schema,
)


REGISTRY.add_converter(
    'rmellipse.arrschema.examples:convert_zeros_to_s2p_ri',
    input_schema=s2p_ri_schema,
    output_schema=zeros_schema,
)


def test_load_and_save():
    print('trying to load s2p_ri')
    s2p_path = ARRAY_SAMPLES / 'load.s2p'
    data1 = S2PRI.load(s2p_path, loader_type='csv', verbose=True)

    data2 = S2PRI.load(ARRAY_SAMPLES / 'load.s2p', verbose=True)

    h5_path = MUTABLE / 'loadarr.h5'
    group = 'sample'

    # this should fail because this
    # schema requires Hz or GHz
    # on the frequency units
    with pytest.raises(Exception):
        data1.attrs['frequency_units'] = 'MHz'
        data1.save(h5_path, 'sample')

    data2.attrs['frequency_units'] = 'GHz'

    data2.save(h5_path, group)

    data3 = S2PRI.load(h5_path, group=group)

    return data3


def test_convert():
    data = S2PRI.load(
        ARRAY_SAMPLES / 'load.s2p',
        loader_type='csv',
        verbose=True,
    )
    S2PRI.validate(data)
    new = data.convert_to(Zeros)
    print(new)


def test_as_xr_schema():
    # schema with 2by2
    print('zeros arbitrary 3,2,2')
    output = Float2By2.from_dataarray(
        xr.DataArray(np.zeros((3, 2, 2), dtype='f4')),
    )
    print(output)

    # basic arbitrary array with changing types
    print('zeros arbitrary')
    output = ZerosComplex.from_dataarray(
        xr.DataArray(np.zeros((4, 4), dtype='f8')),
    )
    print(output)


def test_arrschema_groupsaveable():
    zeros = ZerosComplex.from_dataarray(
        xr.DataArray(np.zeros((4, 4), dtype='f8')),
    )
    with h5py.File(MUTABLE / 'arrschema_groupsaveable.h5', 'w') as f:
        save_object(f, 'zeros', zeros)
        read = load_object(f['zeros'])
        ...


if __name__ == '__main__':
    test_arrschema_groupsaveable()
    data = test_load_and_save()

    print(data)
    test_convert()
    print('ZEROS LIKE \n ============')
    test_as_xr_schema()
    import json

    # print(json.dumps(ZerosComplex.schema, indent=True))
