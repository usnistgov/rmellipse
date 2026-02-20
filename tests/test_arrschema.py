import rmellipse.arrschema as arrschema
import xarray as xr
import numpy as np
from rmellipse.uobjects import RMEMeas
import pytest
from pathlib import Path

LOCALS = Path(__file__).parents[0]
ARRAY_SAMPLES = LOCALS / 'arrsamples'
REGISTRY = arrschema.ArrayClassRegistry()
MUTABLE = LOCALS / 'mutable'

empty_float = arrschema.ArraySchema('empty', (...,), (...,), float)
# basic float
with pytest.raises(Exception):
    mismatched_dims = arrschema.ArraySchema('empty', (..., 'N'), (...,), float)

s2p_ri = arrschema.ArraySchema(
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

zeros = arrschema.ArraySchema(name='zeros', shape=(...,), dims=(...,), dtype=float)

zeros_complex = arrschema.ArraySchema(
    name='zeros_complex', shape=(...,), dims=(...,), dtype=complex
)

float_2by2 = arrschema.ArraySchema(
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

S2P_RI = REGISTRY.build_and_add_class(s2p_ri)
ZEROS = REGISTRY.build_and_add_class(zeros)
ZEROS_COMPLEX = REGISTRY.build_and_add_class(zeros_complex)
FLOAT_2BY2 = REGISTRY.build_and_add_class(float_2by2)

REGISTRY.add_loader(
    'rmellipse.arrschema.examples:load_csv_like_s2p_ri',
    '.s2p',
    loader_type='csv',
    schema=s2p_ri,
)

REGISTRY.add_loader(
    'rmellipse.arrschema.examples:load_group_saveable',
    ['.h5', '.hdf5'],
    loader_type='group_saveable',
    schema=s2p_ri,
)

REGISTRY.add_saver(
    'rmellipse.arrschema.examples:save_group_saveable',
    ['.h5', '.hdf5'],
    saver_type='group_saveable',
    schema=s2p_ri,
)


REGISTRY.add_converter(
    'rmellipse.arrschema.examples:convert_zeros_to_s2p_ri',
    input_schema=s2p_ri,
    output_schema=zeros,
)


def test_with_RMEMeas():
    data = S2P_RI.load(
        ARRAY_SAMPLES / 'load.s2p',
        loader_type='csv',
        verbose=True,
    )

    data = RMEMeas.from_nom('mysample', data)

    # make a class method version of validate?
    s2p_ri.validate(data)


def test_load_and_save():
    print('trying to load s2p_ri')
    s2p_path = ARRAY_SAMPLES / 'load.s2p'
    data1 = S2P_RI.load(s2p_path, loader_type='csv', verbose=True)

    data2 = S2P_RI.load(ARRAY_SAMPLES / 'load.s2p', verbose=True)

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

    data3 = S2P_RI.load(h5_path, group=group)

    return data3


def test_convert():
    data = S2P_RI.load(
        ARRAY_SAMPLES / 'load.s2p',
        loader_type='csv',
        verbose=True,
    )
    S2P_RI.validate(data)
    new = data.convert_to(ZEROS)
    print(new)


def test_as_xr_schema():
    # schema with 2by2
    print('zeros arbitrary 3,2,2')
    output = FLOAT_2BY2.from_dataarray(
        xr.DataArray(np.zeros((3, 2, 2), dtype='f4')),
    )
    print(output)

    # basic arbitrary array with changing types
    print('zeros arbitrary')
    output = ZEROS_COMPLEX.from_dataarray(
        xr.DataArray(np.zeros((4, 4), dtype='f8')),
    )
    print(output)


def test_arrschema_collisions():
    registry = arrschema.ArrayClassRegistry()

    s1_schema = arrschema.ArraySchema(
        name='schema1', dtype='float', shape=(..., 'N'), dims=(..., 'col2')
    )

    s2_schema = arrschema.ArraySchema(
        name='schema1', dtype='float', shape=(..., 'N'), dims=(..., 'col')
    )

    S1 = registry.build_and_add_class(s1_schema)
    S2 = registry.build_and_add_class(s2_schema)

    ...


if __name__ == '__main__':
    data = test_load_and_save()

    test_with_RMEMeas()
    print(data)
    test_convert()
    print('ZEROS LIKE \n ============')
    test_as_xr_schema()
    import json

    print(json.dumps(zeros_complex, indent=True))
    test_arrschema_collisions()
