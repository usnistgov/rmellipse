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

registry = arrschema.ArrSchemaRegistry()

# %%
# We will define a basic schema of an arry with arbitrary shape
# made of floats called "float_zeros". When the schema is made with
# the arrschema function a uid is automatically added. Then we
# add it to the registry.

float_zeros = arrschema.arrschema(
	name='float_zeros', shape=(...,), dims=(...,), dtype=float
)

print(json.dumps(float_zeros, indent=True))

registry.add_schema(float_zeros)

# %%
# Validation
# ----------
#
# Currently validation is only implemented for xarray.DataArray objects
# and RMEmeas objects.

# the schema can be provided directly
my_data = xr.DataArray(np.zeros((4, 4), dtype=float))
arrschema.validate(my_data, schema=float_zeros)

# reffered to by a registry and uid
my_data = xr.DataArray(np.zeros((4, 4), dtype=float))
arrschema.validate(my_data, schema=float_zeros)

# reffered to by a registry and name
# Names are not unique, this may fail if multiple
# schemas share a name and is not recommended.
my_data = xr.DataArray(np.zeros((4, 4), dtype=float))
arrschema.validate(my_data, schema=float_zeros)

# this will fail because the dtype isn't correct
my_data_fails = xr.DataArray(np.zeros((4, 4), dtype=complex))
try:
	arrschema.validate(my_data_fails, schema=float_zeros)
except arrschema.ValidationError as e:
	print(e)

# validated datasets store the associated schema in the metadata
print(my_data.attrs)

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

registry.add_loader(
	'rmellipse.arrschema.examples:load_group_saveable',
	['.h5', '.hdf5'],
	loader_type='group_saveable',
	schema=float_zeros,
)

registry.add_saver(
	'rmellipse.arrschema.examples:save_group_saveable',
	['.h5', '.hdf5'],
	saver_type='group_saveable',
	schema=float_zeros,
)

# %%

# Encoding and decoding functions are expected have function signatures
# that look like ``fun(path, data, *args, **kwargs)``

arrschema.save('example.h5', my_data, 'my-name', registry=registry)
my_data_read = arrschema.load(
	'example.h5', group='my-name', registry=registry, schema=float_zeros
)
print(my_data_read)

# %%
#
# Otherwise, you will have to explicitly declare what schema your
# data corresponds to.

arrschema.save('example.h5', my_data, 'my-name', registry=registry, schema=float_zeros)

# %%
# Conversion
# ----------
#
# Converters can be assigned as well. Converters are functions that take in
# a single data set with an associated schema, and returns a new data set
# with an associated schema.

# define a new format we care about

int_zeros = arrschema.arrschema(name='int_zeros', shape=(...,), dims=(...,), dtype=int)

registry.add_schema(int_zeros)

registry.add_converter(
	'rmellipse.arrschema.examples:convert_float_to_int',
	input_schema=float_zeros,
	output_schema=int_zeros,
)

converted = arrschema.convert(my_data, registry=registry, output_schema=int_zeros)
arrschema.validate(converted, schema=int_zeros)
print(converted)
