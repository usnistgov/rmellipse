# -*- coding: utf-8 -*-
"""
Defining ArraySchema and AnnotatedArray's
=========================================
ArraySchema are dictionary objects that describe the requirements of
an array structure. AnnotatedArray are xarray.DataArray's that are expected
to conform to a given ArraySchema - subclasses of xarray.DataArray.

"""

import rmellipse as rme


# %%
# Defining AnnotatedArrays
# ------------------------
#
# There are 2 required fields in a schema. The first is a shape and the second
# is a dimension specification.


class Array2x2(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=(2, 2),
        dims=('row', 'column'),
    )


# %%
# Dimension and shape specifiers can also be alphabetical letters (lower or
# upper case) that indicate arbitrary dimensionality. If two specifiers
# are the same letter, that indicates they are the same shape. For example,
# a stack of square matrices might be defined as


class ArrayMxNxN(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=('M', 'N', 'N'),
        dims=('page', 'row', 'column'),
    )


# %%
# It's possible to define arbitary dimensionality or shape with ellipses.
# For example, the following schema is an array of arbitary leading dimensions
# ending in an NxN shape of rows and columns. If the shape is arbitary, then
# so are the dimension names in the same position. There can be only a single
# arbitrary specifier. So (...,'N') is okay but (...,'N',...) is not.


class ArrayLeadingNxN(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=(..., 'N', 'N'),
        dims=(..., 'row', 'column'),
    )


# %%
# Data Types
# ----------
#
# By default, an ArraySchema has no dtype specifier (None) but one can be
# provided to identity what kind of data is expected in the values of the
# DataArray. An array is considered valid to a schema it it can be casted
# into that schema's dtype using the numpy.can_cast() function.


class Array2x2Float(rme.AnnotatedArray):
    schema = rme.ArraySchema(shape=(2, 2), dims=('row', 'column'), dtype=float)


# %%
# Coordinates
# -----------
#
# You can specify the expected coordinates of an AnnotatedArray
# by including a coords field with a CoordinateSchema.


class TimeDomainArray(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=('N',), dims=('time',), coords={'time': rme.CoordinateSchema(dtype=float)}
    )


# %%
# If an array has a fixed dimension shape and coordinates, you can
# define the coordinate values by supplying values to the coordinate field.


class TimeDomainArray2x2(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=('N', 2, 2),
        dims=('time', 'row', 'col'),
        coords={
            'time': rme.CoordinateSchema(dtype=float),
            'row': rme.CoordinateSchema(dtype=int, values=[0, 1]),
            'col': rme.CoordinateSchema(dtype=int, values=[0, 1]),
        },
    )


# %%
# Units
# -----
#
# By default the units field is None (which means no specified units, not
# unitless). If your array contains values of a physical unit, it can be
# supplied as a string in the units field. This can be provided to both
# the array values themselves, and to individual coordinates.


class TimeDomainVoltage(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=('N',),
        dims=('time',),
        units='V',
        coords={'time': rme.CoordinateSchema(dtype=float, units='s')},
    )


# %%
# Metadata
# --------
#
# Metadata on xarray.DataArrays (and consequently AnnotatedArrays) are stored
# in the attrs attribute as a dictionary. If your data model is expecting
# specific structures of metadata, those can be defined using Pydantic datamodels.
# For example, you may be making a system that records DC measurments, and want
# to require that operator, temperature, and source current metadata
# fields are always present. It's strongly recommended that extra metadata
# fields be allowed as well.

from pydantic import BaseModel, ConfigDict


class DCMeasurementMetadata(BaseModel):
    # Enable extra fields
    model_config = ConfigDict(extra='allow')
    operator: str
    temperature_celcius: float
    source_current_amps: float


class TimeDomainVoltageWithMetadata(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=('N',),
        dims=('time',),
        units='V',
        coords={'time': rme.CoordinateSchema(dtype=float, units='s')},
        attrs=DCMeasurementMetadata,
    )
