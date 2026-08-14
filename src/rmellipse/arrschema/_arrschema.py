"""
Tools for annotating, saving, loading, converting array-like data.
"""

import xarray as xr
import numpy as np
import uuid
import copy
from types import EllipsisType
from typing import Mapping, Any, Tuple, Self, Callable
from pathlib import Path
import yaml
import json
import importlib.util
import sys
import jsonschema
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
import inspect
import os


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray


ALLOWED_SHAPE_SPECS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALLOWED_SHAPE_SPECS = list(ALLOWED_SHAPE_SPECS) + ['...', ...]
UNSPECIFIED_SPECS = ('...', ...)

SCHEMA_ATTRS_KEY = 'ARRSCHEMA'
ANNOTATION_ATTRS_KEY = 'ARRANNOTATION'


__all__ = [
    'ValidationError',
    'ArraySchema',
    'AnnotatedArray',
]


def _allowed_shape_spec(s: object):
    if isinstance(s, int):
        return True
    if isinstance(s, type(...)):
        return True
    elif isinstance(s, str) and s in ALLOWED_SHAPE_SPECS:
        return True
    else:
        return False


def convert_h5attrs_to_json_types(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = copy.copy(v)
    return out


def _lazy_import_module(name):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError()
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


class ValidationError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class AnnotatedArray(xr.DataArray):
    """
    Interface for an array structure with annotated dimensions and coordinates.

    Inspired by xarray DataArray. Classes conforming to this specification
    can be registered in an ArrayClassRegister.
    """

    __slots__ = ()
    schema: 'ArraySchema'

    def __init__(self, *args, **kwargs):
        xr.DataArray.__init__(self, *args, **kwargs)

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        # add this class to the registry
        xr.DataArray.__init_subclass__(*args, **kwargs)

    @classmethod
    def __subclasshook__(cls, C):
        has_shape = hasattr(C, 'shape')
        has_dims = hasattr(C, 'dims')
        has_coords = hasattr(C, 'coords')
        has_dtype = hasattr(C, 'dtype')
        has_attrs = hasattr(C, 'attrs')
        test = has_shape and has_dims and has_coords and has_dtype and has_attrs
        return test

    def validate(self):
        """
        Validate array data against the schema of this type.
        """
        self.schema.validate(self)

    @classmethod
    def from_dataarray(
        cls,
        array: xr.DataArray,
    ) -> Self:
        """
        Cast an array into an annotated array.

        Parameters
        ----------
        array : xr.DataArray
            Array that can be cast into this format.

        Returns
        -------
        Self
            New array.

        """
        new_shape_spec = cls.schema['shape']
        new_dims_spec = cls.schema['dims']
        new_dtype = cls.schema['dtype']

        # require that specified dimensions be uninterrupted
        # i.e. at most 1 unspecified, arbitrary dimensions
        unq_vals, unq_counts = np.unique(new_shape_spec, return_counts=True)
        if '...' in unq_vals:
            unspecified_count = unq_counts[unq_vals == '...'][0]
            if unspecified_count > 1:
                raise ValueError(
                    f'Schema with >1 arbitrary dimension specifications (...) can not be intialized as xarrays from a schema: \n {json.dumps(cls.schema, indent=True)}'
                )

        # instantiate new array
        new = array
        old_dims = array.dims
        for i, (si, di) in enumerate(zip(new_shape_spec, new_dims_spec)):
            if si == '...':
                break
            # print('forward', si, di)
            new = new.rename({old_dims[i]: di})
            if di in cls.schema['coords']:
                crd_schema = cls.schema['coords'][di]
                crd_dtype = crd_schema['dtype']
                if 'values' not in crd_schema:
                    new = new.assign_coords({di: new.coords[di].astype(crd_dtype)})
                else:
                    new_crd_vals = crd_schema['values']
                    new = new.assign_coords(
                        {di: np.array(new_crd_vals).astype(crd_dtype)}
                    )

        # recast coordinate and dimension names
        rev_new_shape_spec = list(new_shape_spec)[::-1]
        rev_new_dims_spec = list(new_dims_spec)[::-1]
        for i, (si, di) in enumerate(zip(rev_new_shape_spec, rev_new_dims_spec)):
            if si == '...':
                break
            # print('backward', si, di, old_dims[-(i + 1)])
            new = new.rename({old_dims[-(i + 1)]: di})
            if di in cls.schema['coords']:
                crd_schema = cls.schema['coords'][di]
                crd_dtype = crd_schema['dtype']
                if 'values' not in crd_schema:
                    new = new.assign_coords({di: new.coords[di].astype(crd_dtype)})
                else:
                    new_crd_vals = crd_schema['values']
                    new = new.assign_coords(
                        {di: np.array(new_crd_vals).astype(crd_dtype)}
                    )

        new = new.astype(new_dtype)

        kwargs = {
            'data': new.data,
            'coords': new.coords,
            'dims': new.dims,
            'name': new.name,
            'attrs': new.attrs,
        }

        return cls(**kwargs)


class CoordinateSchema(dict):
    def __init__(
        self,
        values: list | None,
        dtype: str | type[float] | None = None,
        units: str | None = None,
    ):
        self.update(
            {
                k: v
                for k, v in zip(
                    ('values', 'dtype', 'units'), (values, np.dtype(dtype).str, units)
                )
                if v
            }
        )


class ArraySchema(dict):
    """
    Specialized dict subclass to describe the shape an array.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[str | int | EllipsisType, ...],
        dims: tuple[str | EllipsisType, ...],
        dtype: str | type[float] | type[complex],
        units: str | None = None,
        coords: Mapping = {},
        attrs_schema: Mapping | None = None,
    ):
        """
        Initialize an ArraySchema.

        Parameters
        ----------
        name : str
            Name of the array structure.
        shape : tuple[str | int | EllipsisType, ...]
            Shape of structure. Ellipses indicate arbitrary dimensions,
            letters indicate a required dimension of unknown length, and
            integers indicate a required dimension of a required length.
        dims : tuple[str | EllipsisType, ...]
            Names assigned to dimensions specified by shape. Any required
            dimension must be names, and arbitrary dimensions must also be
            ellipses.
        dtype : str | type[float]
            Type must be parseable by numpy's dtype (e.g. f8, c8, u8, etc)
        units : Mapping, optional
            Mapping of units to the array structure. If the whole structure
            has a single unit, then a string can be passed. Optionally, a
            single required dimension can be mapped to a 1-d array of units.
            For example, if a dimension called "col" corresponds to columns in spread-sheet
            like data and each column has its own unit, you could specify that
            as {"col":["unit 1", "unit 2"]}.
        coords : Mapping, optional
            Mapping of required dimensions to a coordinate space. Must provide
            at least a dtype and a single unit as a string. Optionally,
            if the coordinates are fixed (i.e. the row and column indices of
            stacks of 2-d matrices) then you may specify those coordinates
            here.
        attrs_schema : mapping, optional
            JSON Schema for validating metadata attributes.

        Returns
        -------
        dict
            Dictionary conforming to an arrschema specification.

        Raises
        ------
        Exception
            If some logical inconsistency or is found, or the provided
            schema doesn't follow the specification for an array schema.
        """
        dict.__init__(self)

        # replace Ellipsis with "..." to ensure compatibility with sorting and JSON
        shape = tuple('...' if s == Ellipsis else s for s in shape)
        dims = tuple('...' if d == Ellipsis else d for d in dims)

        # validate shape/dims/dtype
        if len(shape) != len(dims):
            raise Exception('shape and dims length must match')
        if any([not _allowed_shape_spec(s) for s in shape]):
            raise Exception('shape must be a letter or ...')
        if any(
            [
                not isinstance(
                    d,
                    (
                        str,
                        type(Ellipsis),
                    ),
                )
                for d in dims
            ]
        ):
            raise Exception('dims must be str or ...')
        if any([(s == '...') ^ (d == '...') for s, d in zip(shape, dims)]):
            raise Exception(
                'unspecified shapes specs (...) must have unspecified dimension names. Specified shapes must have specified dimension names.'
            )
        dtype = np.dtype(dtype).str

        # validate coords and make CoordinateSchemas for each
        shape_lookup = dict(zip(dims, shape))
        out_coords = {}
        for coord_name, coord in coords.items():
            coord_schema = CoordinateSchema(
                coord.get('values'),
                np.dtype(coord.get('dtype')).str,
                coord.get('units'),
            )
            if (
                coord_schema.get('values', False)
                and isinstance(shape_lookup[coord_name], int)
                and len(coord_schema.get('values') or []) != shape_lookup[coord_name]
            ):
                raise ValueError(
                    f'Coord values {coord_name} length do not match dimension spec {shape_lookup[coord_name]}'
                )
            out_coords[coord_name] = coord_schema

        # units should be specified.
        if not isinstance(units, str):
            Exception('Units must be specified for top level data, if unitless: "arb"')

        if attrs_schema is None:
            attrs_schema = {}

        out: dict[str, Any] = {
            'name': name,
            'shape': tuple(shape),
            'dims': tuple(dims),
            'dtype': dtype,
            'units': units,
            'coords': out_coords,
            'attrs_schema': attrs_schema,
        }
        self.update(out)

    def validate(
        self,
        arr: AnnotatedArray,
        attach_schema: bool = True,
    ):
        """
        Test if array conforms to schema.

        Parameters
        ----------
        arr : AnnotatedArray
            Array to check.
        attach_schema : bool, optional
            If True, the schema is dumped into a string and
            attatched to the attrs of the input data array.
            The default is True.

        Raises
        ------
        ValidationError
            If a discrepancy is found between the data and the schema.

        Returns
        -------
        None.

        """
        # get the schema
        # compare dimensions to the actual shape,
        # map symbolic dimensions to actual dimensions
        sym_map = {}
        for d, s in zip(self['dims'], self['shape']):
            # if its a symbolic dimension size
            # according to first appearence of that symbol
            # and check it against the existing sym_map
            if d != '...' and isinstance(s, str):
                if s not in sym_map:
                    sym_map[s] = len(arr.coords[d])
                if len(arr.coords[d]) != sym_map[s]:
                    raise ValidationError(
                        f'Coord {d} length {len(arr.coords[d])} doesnt match shape symbolic spec {s}={sym_map[s]}'
                    )
            # if its a statically sized dimension
            # then just check that it matches
            elif d != '...' and isinstance(s, int):
                if len(arr.coords[d]) != s:
                    raise ValidationError(
                        f'Coord {d} length {len(arr.coords[d])} doesnt match shape spec {s}'
                    )

        # check that the static types match
        if self['dtype'] != np.dtype(arr.dtype):
            raise ValidationError(
                f'dtype {arr.dtype} doesnt match expected {self["dtype"]} for : \n {arr}'
            )

        # check that the coordinate dimensions are acceptable
        for cname, coord in self['coords'].items():
            # check values match for static coordinates
            if 'values' in coord:
                if not (arr.coords[cname] == coord['values']).all():
                    raise ValidationError(
                        f'not all coordinates in self match. \n Got: {arr.coords[cname]} \n Expected {coord["values"]} for : \n {arr}'
                    )

            # check that dtypes match for coordinates
            if coord['dtype'] != np.dtype(arr.coords[cname].dtype):
                raise ValidationError(
                    f'dtype {arr.coords[cname].dtype} doesnt match expected {coord["dtype"]} for : \n {arr}'
                )

        # validate the metadata schema
        try:
            jsonschema.validate(dict(arr.attrs), schema=self['attrs_schema'])
        except JsonSchemaValidationError as e:
            msg = 'Failed to validate attrs schema : \n ' + str(e)
            raise ValidationError(msg) from e

        if attach_schema:
            arr.attrs[SCHEMA_ATTRS_KEY] = json.dumps(self)
