# -*- coding: utf-8 -*-
"""
Defining Array Schema
=====================

Array schema are json documents that act as metadata describing the
structure of array-like data. The submodule provides some python bindings
for generating the schema in python.

See
"""

# %%
# Make a Registry
# ---------------
#
# First make a registry to store a collection of schema you want to
# use. We will also import some example functions including in the
# rmellipse package, as well as the json package to format the
# json documents.

import rmellipse.arrschema as arrschema
import rmellipse.arrschema.examples as examples
import xarray as xr
import numpy as np
import json

REGISTRY = arrschema.ArrayClassRegistry()

# %%
# We will define a basic schema of an arry with arbitrary shape
# made of floats called "float_zeros". When the schema is made with
# the arrschema function a uid is automatically added. Then we
# add it to the registry.

# a schema is just a dictionary that specifies
# what the expected structure of a dataset is
float_zeros_schema = arrschema.ArraySchema(
    name='float_zeros', shape=(...,), dims=(...,), dtype=float
)

# print(json.dumps(FLOAT_ZEROS, indent=True))
# you can use that dictionary to generate a
# python class object within the context of a registry
FLOAT_ZEROS = REGISTRY.build_and_add_class(float_zeros_schema)
print(FLOAT_ZEROS)


# %%
# Validation
# ----------
#
# Requires that the array conforms to the
# AnnotatedArray subclass specification.

# cast an array into the type to check it conforms
# to the specification,
my_data = xr.DataArray(np.zeros((4, 4), dtype=float))
my_data = FLOAT_ZEROS(my_data)
my_data.validate()

# this will fail because the dtype isn't correct
my_data_fails = xr.DataArray(np.zeros((4, 4), dtype=complex))
try:
    my_data_fails = FLOAT_ZEROS(my_data_fails)
    my_data_fails.validate()
except arrschema.ValidationError as e:
    print('caught error: \n', e)

# succesfully validated datasets store the associated schema in the metadata
print(my_data.attrs['ARRSCHEMA'])

# %%
# Loaders and Savers
# ------------------
#
# Array schema provide a system for organizing and envoking different
# encoding and decoding functions for various schema. Often times
# we work with a single in memory representation of a particular object,
# but may need to be able to read/write to and from multiple different
# methods of storing that data on disc. We do this by associating
# a loader/saver with a function using a module spec, and related
# file extensions.

# supply the module pathspec to the function
REGISTRY.add_loader(
    'rmellipse.arrschema.examples:load_group_saveable',
    ['.h5', '.hdf5'],
    loader_type='group_saveable',
    schema=float_zeros_schema,
)

REGISTRY.add_saver(
    'rmellipse.arrschema.examples:save_group_saveable',
    ['.h5', '.hdf5'],
    saver_type='group_saveable',
    schema=float_zeros_schema,
)

# %%

# Encoding and decoding functions are expected have function signatures
# that look like ``fun(path, data, *args, **kwargs)``

my_data.save('example.h5', 'my_data_name')
my_data_read = FLOAT_ZEROS.load('example.h5', group='my_data_name')
print(my_data_read)

# %%
# Conversion
# ----------
#
# Converters can be assigned as well. Converters are functions that take in
# a single data set with an associated schema, and returns a new data set
# with an associated schema.

# define a new format we care about

int_zeros_schema = arrschema.ArraySchema(
    name='int_zeros', shape=(...,), dims=(...,), dtype=int
)

INT_ZEROS = REGISTRY.build_and_add_class(int_zeros_schema)

REGISTRY.add_converter(
    'rmellipse.arrschema.examples:convert_float_to_int',
    input_schema=float_zeros_schema,
    output_schema=int_zeros_schema,
)

converted = my_data.convert_to(INT_ZEROS)

print(converted)
