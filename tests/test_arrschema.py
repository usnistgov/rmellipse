import json
import pytest

from pathlib import Path

import xarray as xr
import numpy as np
import h5py
from pydantic import BaseModel
from typing import Literal
import rmellipse.arrschema as arrschema
from rmellipse.utils import load_object, save_object, save_file, load_file


LOCALS = Path(__file__).parents[0]
ARRAY_SAMPLES = LOCALS / 'arrsamples'
MUTABLE = LOCALS / 'mutable'
TEST_FILES = LOCALS / 'const'


class TouchStoneMeta(BaseModel):
    frequency_units: Literal['GHz', 'Hz']


class S2PRI(arrschema.AnnotatedArray):
    schema = arrschema.ArraySchema(
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
        attrs=TouchStoneMeta,
    )


class Zeros(arrschema.AnnotatedArray):
    schema = arrschema.ArraySchema(shape=(...,), dims=(...,), dtype=float)


class ZerosComplex(arrschema.AnnotatedArray):
    schema = arrschema.ArraySchema(shape=(...,), dims=(...,), dtype=complex)


class Float2By2(arrschema.AnnotatedArray):
    schema = arrschema.ArraySchema(
        shape=(3, ..., 2, 2),
        dims=('d0', ..., 'd1', 'd2'),
        dtype=float,
        coords={
            'd0': {'values': [0, 1, 2], 'dtype': float},
            'd1': {'values': [0, 1], 'dtype': int},
            'd2': {'dtype': int},
        },
    )


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
    # using load 0bject
    with h5py.File(MUTABLE / 'arrschema_groupsaveable.h5', 'w') as f:
        save_object(f, 'zeros', zeros)
        read = load_object(f['zeros'])
        ...

    save_file(MUTABLE / 'arrschema_groupsaveable_file.h5', zeros)
    new_zeros = load_file(MUTABLE / 'arrschema_groupsaveable_file.h5')


def test__allowed_shape_spec():
    # _allowed_shape_spec returns True if it is given a lower case letter
    assert arrschema._arrschema._allowed_shape_spec('a')
    # _allowed_shape_spec returns True if it is given an ellipse
    assert arrschema._arrschema._allowed_shape_spec(...)
    # _allowed_shape_spec returns False if it is given a string of length > 1
    assert not arrschema._arrschema._allowed_shape_spec('aa')


# testing ArraySchema class
def test_array_schema_init():
    basic_schema = arrschema.ArraySchema(shape=(...,), dims=(...,), dtype=float)

    assert basic_schema['shape'] == ('...',)
    assert basic_schema['dims'] == ('...',)
    assert basic_schema['dtype'] is float
    assert basic_schema['coords'] == {}

    basic_schema = arrschema.ArraySchema(shape=('...',), dims=(...,), dtype=float)

    assert basic_schema['shape'] == ('...',)
    assert basic_schema['dims'] == ('...',)
    assert basic_schema['dtype'] is float
    assert basic_schema['coords'] == {}

    with pytest.raises(Exception):
        # mismatched shape and dims length
        arrschema.ArraySchema(shape=('N', 'M'), dims=('M',), dtype=float)
    with pytest.raises(Exception):
        # shape needs to be a letter or a number
        arrschema.ArraySchema(shape=('NM', 'M'), dims=('N', 'M'), dtype=float)
    with pytest.raises(Exception):
        # dims needs to be an ellipse or string (typechecking should handle this)
        arrschema.ArraySchema(shape=('N', 'M'), dims=(4, 'M'), dtype=float)
    with pytest.raises(Exception):
        # shape and dims have to have ... in same place
        arrschema.ArraySchema(shape=(...,), dims=('M',), dtype=float)

    with pytest.raises(Exception):
        arrschema.ArraySchema(
            shape=(..., 'N', 2),
            dims=(..., 'frequency', 're_im'),
            dtype=float,
            coords={
                'frequency': {
                    'units': 'GHz',
                    'dtype': float,
                },
                're_im': {
                    'units': 'arb',
                    'dtype': float,
                    'values': ['Re', 'Im', 'Extra'],
                },
            },
        )
    with pytest.raises(Exception):
        arrschema.ArraySchema(
            shape=(..., 'N', 3),  # should be 2
            dims=(..., 'frequency', 're_im'),
            dtype=float,
            coords={
                'frequency': {
                    'units': 'GHz',
                    'dtype': float,
                },
                're_im': {
                    'units': 'arb',
                    'dtype': float,
                    'values': ['Re', 'Im'],
                },
            },
        )


def test_array_schema_validate():
    s2p_ri = arrschema.ArraySchema(
        shape=(..., 2, 8),
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
    )

    with pytest.raises(arrschema.ValidationError):
        data = S2PRI.from_dataarray(
            xr.DataArray(
                np.random.randn(3, 8),
                dims=('frequency', 'col'),
                coords={'frequency': [10, 20, 20]},
            )
        )
        data.validate()


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


def test_from_dataarray():
    class FloatE3E22(arrschema.AnnotatedArray):
        schema = arrschema.ArraySchema(
            shape=(..., 3, ..., 2, 2),
            dims=(..., 'd0', ..., 'd1', 'd2'),
            dtype=float,
            coords={
                'd0': {'values': [0, 1, 2], 'dtype': float},
                'd1': {'values': [0, 1], 'dtype': int},
                'd2': {'dtype': int},
            },
        )

    data = xr.DataArray(np.random.randn(1, 3, 1, 2, 2))

    with pytest.raises(ValueError):
        FloatE3E22.from_dataarray(data)

    class Float3E22(arrschema.AnnotatedArray):
        schema = arrschema.ArraySchema(
            shape=(3, ..., 2, 2),
            dims=('d0', ..., 'd1', 'd2'),
            dtype=float,
            coords={
                'd0': {'dtype': float},
                'd1': {'values': [0, 1], 'dtype': int},
                'd2': {'dtype': int},
            },
        )

    data = xr.DataArray(
        np.random.randn(3, 1, 2, 2),
        dims=('d0', ..., 'd1', 'd2'),
        coords={'d0': [0, 1, 2]},
    )
    data_annotated = Float3E22.from_dataarray(data)


def test_no_dtype():
    class Generic(arrschema.AnnotatedArray):
        schema = arrschema.ArraySchema(shape=(...,), dims=(...,))

    # these should both be allowed since
    # no type was specified
    a_str = Generic('asdasd')
    a_flt = Generic(1)
    a_str.validate()
    a_flt.validate()

    a = Generic.zeros_from(a_str)
    print(a)


def test_empty():
    # mixed required an dunrequired dimensions
    class Float3E22(arrschema.AnnotatedArray):
        schema = arrschema.ArraySchema(
            shape=(3, ..., 2, 2),
            dims=('d0', ..., 'd1', 'd2'),
            dtype=float,
            coords={
                'd0': {'dtype': float},
                'd1': {'values': [0, 1], 'dtype': int},
                'd2': {'dtype': int},
            },
        )

    new = Float3E22.zeros(d0=[1, 2, 3], d2=[1, 2])

    # add an extra dimension
    new = Float3E22.zeros(d0=[1, 2, 3], e11=[1], d2=[1, 2])
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


def test_string_dtype():
    class Strings(arrschema.AnnotatedArray):
        schema = arrschema.ArraySchema(shape=(...,), dims=(...,), dtype=str)

    # these should both be allowed since
    # no type was specified
    Strings('a').validate()
    Strings('aaasdasdasdasd').validate()

    # save_file(MUTABLE / 'dummy.h5', a_str)
    # load_file(MUTABLE / 'dummy.h5')

    # a  = Generic.zeros_from(a_str)
    # print(a)


if __name__ == '__main__':
    test_empty()
    test_string_dtype()
    test_array_schema_init()
    test_from_dataarray()
    # test_arrschema_groupsaveable()
    # test_zeros_like()
    # test_no_dtype()
    # print('ZEROS LIKE \n ============')
    # test_as_xr_schema()
    # import json

    # # print(json.dumps(ZerosComplex.schema, indent=True))
    # # This saves an annotated array using a schema defined
    # # in this module, so if I were to try and read it in from anywhere else it should fail.
    # zeros = ZerosComplex.from_dataarray(
    #     xr.DataArray(np.zeros((4, 4), dtype='f8')),
    # )
    # with h5py.File(TEST_FILES / 'arrschema_groupsaveable.h5', 'w') as f:
    #     save_object(f, 'zeros', zeros)
    #     read = load_object(f['zeros'])
    #     ...
