import rmellipse.arrschema as arrschema
from rmellipse.utils import load_object, save_object
import xarray as xr
import numpy as np
from rmellipse.uobjects import RMEMeas
import pytest
import json
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


def test_zeros_like():
    # schema with 2by2
    print('zeros arbitrary 3,2,2')
    # should work
    output = Float2By2.zeros_from(
        xr.DataArray(np.zeros((3, 2, 2), dtype='f4')),
        rename_dims={'dim_0': 'd0', 'dim_1': 'd1', 'dim_2': 'd2'},
        use_coords={'d0': [1, 2, 3], 'd1': [1, 2], 'd2': [1, 2]},
    )
    output.validate()

    # failed for missing coordinate that isn't in proto array

    output = Float2By2.zeros_from(
        xr.DataArray(np.zeros((3, 2, 2), dtype='f4')),
        rename_dims={'dim_0': 'd0', 'dim_1': 'd1', 'dim_2': 'd2'},
        use_coords={
            'd0': [1, 2, 3],
            'd1': [1, 2],
        },
    )
    output.validate()

    out = S2PRI.zeros_from(
        xr.DataArray(np.zeros((3, 2, 8), dtype='f4')),
        rename_dims={'dim_0': 'blarg', 'dim_1': 'frequency', 'dim_2': 'col'},
        attrs={'frequency_units': 'Hz'},
    )

    out.validate()


def test_from_dataarray():
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


def test__allowed_shape_spec():
    # _allowed_shape_spec returns True if it is given a lower case letter
    assert arrschema._arrschema._allowed_shape_spec('a')
    # _allowed_shape_spec returns True if it is given an ellipse
    assert arrschema._arrschema._allowed_shape_spec(...)
    # _allowed_shape_spec returns False if it is given a string of length > 1
    assert not arrschema._arrschema._allowed_shape_spec('aa')


def test_convert_h5attrs_to_json_types():
    py_list = [0, 1, 2]
    py_float = 3.14
    py_str = 'hello world'
    d = {
        'np_array': np.array(py_list),
        'np_float': np.float64(py_float),
        'py_str': py_str,
    }
    d_out = arrschema._arrschema.convert_h5attrs_to_json_types(d)
    assert d_out['np_array'] == py_list  # np arrays turn into python lists
    assert d_out['np_float'] == py_float  # np generics turn into python generics
    assert d_out['py_str'] == py_str  # python objects stay as python objects


def convert_float_to_int(zeros):
    return zeros.astype(int)


def convert_int_to_float(zeros):
    return zeros.astype(float)


def test_convert_from():
    TEST_REGISTRY = arrschema.ArrayClassRegistry()

    class FloatZeros(arrschema.AnnotatedArray):
        registry = TEST_REGISTRY
        schema = arrschema.ArraySchema(
            name='float_zeros', shape=(...,), dims=(...,), dtype=float
        )

    class IntZeros(arrschema.AnnotatedArray):
        registry = TEST_REGISTRY
        schema = arrschema.ArraySchema(
            name='int_zeros', shape=(...,), dims=(...,), dtype=int
        )

    float_zeros_data = FloatZeros(xr.DataArray(np.zeros((4, 4), dtype=float)))
    float_zeros_data.validate()
    int_zeros_data = IntZeros(xr.DataArray(np.zeros((4, 4), dtype=int)))
    int_zeros_data.validate()

    TEST_REGISTRY.add_converter(
        'test_arrschema:convert_float_to_int',
        input_schema=FloatZeros.schema,
        output_schema=IntZeros.schema,
    )

    TEST_REGISTRY.add_converter(
        'test_arrschema:convert_int_to_float',
        input_schema=IntZeros.schema,
        output_schema=FloatZeros.schema,
    )

    converted = IntZeros.convert_from(float_zeros_data)
    assert (
        json.loads(converted.attrs[arrschema._arrschema.SCHEMA_ATTRS_KEY])['name']
        == 'int_zeros'
    )
    assert (
        json.loads(converted.attrs[arrschema._arrschema.SCHEMA_ATTRS_KEY])['dtype']
        == 'int64'
    )
    converted = FloatZeros.convert_from(int_zeros_data)
    assert (
        json.loads(converted.attrs[arrschema._arrschema.SCHEMA_ATTRS_KEY])['name']
        == 'float_zeros'
    )
    assert (
        json.loads(converted.attrs[arrschema._arrschema.SCHEMA_ATTRS_KEY])['dtype']
        == 'float64'
    )


if __name__ == '__main__':
    test_arrschema_groupsaveable()
    data = test_load_and_save()
    test_zeros_like()
    # test_with_RMEMeas()
    # print(data)
    # test_convert()
    # print('ZEROS LIKE \n ============')
    # test_as_xr_schema()
    # import json

    print(data)
    test_convert()
    print('ZEROS LIKE \n ============')
    import json

    # print(json.dumps(ZerosComplex.schema, indent=True))
