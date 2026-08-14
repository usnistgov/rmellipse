# -*- coding: utf-8 -*-
"""
Defining ArraySchema and AnnotatedArray's
=========================================
ArraySchema are dictionary objects that describe the requirements of
an array structure. AnnotatedArray are xarray.DataArray's that conform
to a given ArraySchema.

"""

import rmellipse.arrschema as arrschema
import rmellipse.arrschema.examples as examples
import xarray as xr
import numpy as np

# %%
# For a simple example, we will define a basic schema of an array with arbitrary shape
# made of floats called "float_zeros".


class FloatZeros(arrschema.AnnotatedArray):
    schema = arrschema.ArraySchema(
        name='float_zeros', shape=(...,), dims=(...,), dtype=float
    )


# %%
# Validation
# ----------
#
# ArraySchema aren't enforced at runtime, but you
# can check manually by calling the validate() method
# on something you've cast into a Schema.

# cast an array into the type to check it conforms
# to the specification,
my_data = xr.DataArray(np.zeros((4, 4), dtype=float))
my_data = FloatZeros(my_data)
my_data.validate()

# this will fail because the dtype isn't correct
my_data_fails = xr.DataArray(np.zeros((4, 4), dtype=complex))
try:
    my_data_fails = FloatZeros(my_data_fails)
    my_data_fails.validate()
except arrschema.ValidationError as e:
    print('caught error: \n', e)

# %%
# Schema with Coordinates
# -----------------------
#
# You can specifiy the requirements of the
# coordinates as well. This example
# defines the structure of a 2 x 2 matrix
# with a time coordinate. Using letters to label
# shapes of dimensions indicate an arbitrary length,
# using integers to label shapes indicate a fixed length.
# For coordinates, if specific values are expected, those can
# be set with the a list of values.


class TimeDomain2x2(arrschema.AnnotatedArray):
    schema = arrschema.ArraySchema(
        name='TimeDomain2x2',
        shape=('N', 2, 2),
        dims=('time', 'row', 'col'),
        dtype=float,
        coords={
            'time': {'dtype': float},
            'row': {'values': [0, 1], 'dtype': int},
            'col': {'values': [0, 1], 'dtype': int},
        },
    )


# %%
# Casting and Validating
# ----------------------
#
# To use a schema cast a DataArray in the AnnotatedArray
# class. Arrays are not validated at runtime, and can
# be manually checked with the validate method after
# casting.

# cast an array into the type to check it conforms
# to the specification,
my_data = xr.DataArray(np.zeros((4, 4), dtype=float))
my_data = FloatZeros(my_data)

try:
    my_data.validate()
except arrschema.ValidationError as e:
    print('caught error: \n', e)

# %%
# Annotating Functions
# --------------------
#
# Once the schema has been defined, you can use it
# to annotate functions for clear documentation.


class Matrix2x2(arrschema.AnnotatedArray):
    schema = arrschema.ArraySchema(
        name='Matrix2x2',
        shape=(2, 2),
        dims=('row', 'col'),
        dtype=float,
        coords={
            'row': {'values': [0, 1], 'dtype': int},
            'col': {'values': [0, 1], 'dtype': int},
        },
    )


def timedomain_average_(data: TimeDomain2x2) -> Matrix2x2:
    time_average = data.mean(dim='time')
    out = Matrix2x2(time_average)
    out.validate()
    return out
