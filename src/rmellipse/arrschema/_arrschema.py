"""
Tools for annotating array structures.
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
import pydantic
import sys
import jsonschema
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
import inspect
import os
from typing import Type
from enum import Enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray


ALLOWED_SHAPE_SPECS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALLOWED_SHAPE_SPECS = list(ALLOWED_SHAPE_SPECS) + ['...', ...]
UNSPECIFIED_SPECS = ('...', ...)

__all__ = ['AnnotatedArray', 'ArraySchema', 'CoordinateSchema', 'ValidationError']


def can_cast_dtype(test: 'AnnotatedArray', dtype) -> bool:
    return np.can_cast(test.dtype, dtype, casting='unsafe')


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


class ValidationError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class AnnotatedArray(xr.DataArray):
    """
    Extension of xr.DataArray that is expected to conform to a specific schema.

    Schema is defined by an ArraySchema class.
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
        if 'dtype' in cls.schema:
            new_dtype = cls.schema['dtype']
        else:
            new_dtype = None
        # require that specified dimensions be uninterrupted
        # i.e. at most 1 unspecified, arbitrary dimensions
        unq_vals, unq_counts = np.unique(new_shape_spec, return_counts=True)
        if '...' in unq_vals:
            unspecified_count = unq_counts[unq_vals == '...'][0]
            if unspecified_count > 1:
                raise ValueError(
                    'Schema has >1 arbitrary dimension specification (...) and can not be initialized from an array.'
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
                # cast into new dtype if available, other wise
                # maintain original
                if 'dtype' in crd_schema:
                    crd_dtype = crd_schema['dtype']
                else:
                    crd_dtype = new.coords[di].dtype
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
                # use schema's dtype
                if 'dtype' in crd_schema:
                    crd_dtype = crd_schema['dtype']
                else:
                    crd_dtype = new.coords[di].dtype
                if 'values' not in crd_schema:
                    new = new.assign_coords({di: new.coords[di].astype(crd_dtype)})
                else:
                    new_crd_vals = crd_schema['values']
                    new = new.assign_coords(
                        {di: np.array(new_crd_vals).astype(crd_dtype)}
                    )

        # cast to new dtype only if
        # a new dtype has been specified
        # by the array schema
        if new_dtype is not None:
            new = new.astype(new_dtype)

        kwargs = {
            'data': new.data,
            'coords': new.coords,
            'dims': new.dims,
            'attrs': new.attrs,
        }

        return cls(**kwargs)

    @classmethod
    def zeros(
        cls,
        attrs: Mapping | None = None,
        **coords: np.ndarray | xr.DataArray,
    ) -> 'AnnotatedArray':
        """
        Generate an empty array of zeros based on the schema.

        If extra coordinates are
        supplied they will be inserted at the first arbitrary dimension
        specificier in the AnnotatedArray's schema ('...'). Coordinates
        with set values can be ignored, and will be automatically inserted.

        Zero array is initialized with numpy.zeros.

        Parameters
        ----------
        attrs : Mapping | None = None
            Provided metadata to instantiate the AnnotatedArray with. The
            values in the supplied metadata are
            shallow copied onto the instantiated AnnotatedArrays's attrs.

        **coords : np.ndarray | xr.DataArray
            KeyValue pairs of coordinates. Must include the required
            coordinates of the AnnotatedArray.

        Returns
        -------
        AnnotatedArray
            Array with supplied coordinates that conforms to the schema.

        Raises
        ------
        KeyError
            DESCRIPTION.
        """
        extras_inserted = False
        dims = []
        for d in cls.schema['dims']:
            if '...' != d:
                dims.append(d)
            elif not extras_inserted:
                extras_inserted = True
                dims += [extra for extra in coords if extra not in cls.schema['dims']]

        shape = []
        new_coords = {}
        for d in dims:
            if d in cls.schema['coords'] and 'values' in cls.schema['coords'][d]:
                coord_schema = cls.schema['coords'][d]
                shape.append(len(coord_schema['values']))
                new_coords[d] = np.array(coord_schema['values'], coord_schema['dtype'])
            else:
                try:
                    shape.append(len(coords[d]))
                    new_coords[d] = coords[d]
                except KeyError as e:
                    raise KeyError(
                        f'Missing required coordinate {d}:{cls.schema["coords"][d]}'
                    ) from e
        out = cls(
            data=np.zeros(shape, dtype=cls.schema['dtype']),
            dims=dims,
            coords=new_coords,
        )
        if attrs is not None:
            for k in attrs:
                out.attrs[k] = attrs[k]
        out.validate()
        return out

    @classmethod
    def zeros_from(
        cls,
        prototype: 'AnnotatedArray',
        drop_dims: list[str] | None = None,
        rename_dims: Mapping | None = None,
        use_coords: Mapping | None = None,
        reorder: bool = True,
        validate: bool = True,
        attrs: Mapping | None = None,
        **coords,
    ) -> 'AnnotatedArray':
        """
        Generate a new zeros array based on a prototype array.

        Dimensions that are mapped from the prototype array to the output
        array are cast into the correct type. Otherwise, dimensions are
        inserted in the expected place.

        Parameters
        ----------
        prototype : AnnotatedArray
            Array to base the new array off of.
        drop_dims : list[str], optional
            Drop these dimensions. The default is None.
        rename_dims : Mapping | dict, optional
            Mapping of dimensions on the prototype array that should be
            converted to dimensions of this type of array. The default is None.
        use_coords : Mapping | dict, optional
            Additional dimensions required for the new type, key is
            the dimension name and value is the new coordinate to use for
            that dimension.
        reorder : bool, optional
            Automatically try to reorder dimensions to conform
            to the specification.
        validate : bool, optional
            If true, validate after creation. Default is False
        attrs : Mapping | dict, optional
            If provided, supply metadata to be used as attributes.
        **coords : Mapping | None, optional
            Keyword version of use_coords. Is merged with ontop of
            use_coords.

        Returns
        -------
        zeros : AnnotatedArray
            Zeros array in the new format.

        """
        if use_coords is None:
            use_coords = {}
        use_coords = use_coords | coords

        # make a shallow copy of the add dims
        use_coords = {k: v for k, v in use_coords.items()}

        # initialize as zeros from the prototype
        # use the schemas dtype unless none is specified
        if 'dtype' in cls.schema and cls.schema['dtype'] is not None:
            out = xr.zeros_like(prototype, dtype=cls.schema['dtype'])
        else:
            out = xr.zeros_like(prototype)

        # drop the dimensions no longer needed
        if drop_dims:
            sel_dict = {k: 0 for k in drop_dims}
            out = out.isel(sel_dict, drop=True)

        # rename dimensions as requested
        if rename_dims:
            out = out.rename(rename_dims)

        # see what dimensions are missing and create them.
        # If they have fixed values use those, otherwise get them from the
        # add coords field.
        for d in cls.schema['dims']:
            if d == '...':
                continue
            # if it already exists, make it match the schema
            crd_schema = cls.schema['coords'][d]
            crd_dtype = crd_schema['dtype']
            if d in out.dims and 'values' in crd_schema:
                # if dimension is already present
                # assign the expected fixed coordinates
                fixed_crd_values = np.array(crd_schema['values'], dtype=crd_dtype)
                assign_coords = {d: fixed_crd_values}
                out = out.assign_coords(assign_coords)

            # it doesn't exist and has a set value, make it
            elif d not in out.dims and 'values' in crd_schema:
                # if dimension is already present
                # assign the expected fixed coordinates
                fixed_crd_values = np.array(crd_schema['values'], dtype=crd_dtype)
                expand_input = {d: fixed_crd_values}
                out = out.expand_dims(expand_input)

            # if the dimension doesnt exist, create it with coordinates
            elif d not in out.dims and d in use_coords:
                err_msg = f'coordinate for dimension {d} is required for new {cls.__name__} and coordinate wasnt supplied in use_coords or present in prototype array.'
                if not use_coords:
                    raise ValueError(err_msg)
                try:
                    expand_input = {d: use_coords[d]}
                except KeyError as e:
                    raise ValueError(err_msg) from e

                out = out.expand_dims(expand_input)
                use_coords.pop(d)

            # if the dimension already exists, use it and assign coords
            elif d not in out.coords and d in use_coords:
                out = out.assign_coords({d: use_coords[d]})

        # sort the dimensions into the spec
        if reorder:
            spec = [d if d != '...' else ... for d in cls.schema['dims']]
            out = out.transpose(*spec)

        out = cls.from_dataarray(out)

        if attrs:
            for a in attrs:
                out.attrs[a] = attrs[a]

        if validate:
            out.validate()

        return out


class CoordinateSchema(dict):
    def __init__(
        self,
        values: list | None = None,
        dtype: type | None = None,
        units: str | None = None,
    ):
        self.update(
            {
                k: v
                for k, v in zip(('values', 'dtype', 'units'), (values, dtype, units))
                if v
            }
        )


class ArraySchema(dict):
    """
    Specialized dict subclass to describe the shape an array.
    """

    def __init__(
        self,
        shape: tuple[str | int | EllipsisType, ...],
        dims: tuple[str | EllipsisType, ...],
        dtype: type | None = None,
        units: str | None = None,
        coords: Mapping = {},
        attrs: pydantic.BaseModel = None,
    ):
        """
        Initialize an ArraySchema.

        Parameters
        ----------
        shape : tuple[str | int | EllipsisType, ...]
            Shape of structure. Ellipses indicate arbitrary dimensions,
            letters indicate a required dimension of unknown length, and
            integers indicate a required dimension of a required length.
        dims : tuple[str | EllipsisType, ...]
            Names assigned to dimensions specified by shape. Any required
            dimension must be names, and arbitrary dimensions must also be
            ellipses.
        dtype : type | None
            Type must be parseable by numpy's dtype (e.g. f8, c8, u8, etc).
            None means no datatype requriement, can be Any.
        units : Mapping, optional
            Mapping of units to the array structure.
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

        # validate coords and make CoordinateSchemas for each
        shape_lookup = dict(zip(dims, shape))
        out_coords = {}
        for coord_name, coord in coords.items():
            coord_schema = CoordinateSchema(
                coord.get('values'),
                coord.get('dtype'),
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

        out: dict[str, Any] = {
            'shape': tuple(shape),
            'dims': tuple(dims),
            'coords': out_coords,
        }

        if units is not None:
            # units should be specified.
            if not isinstance(units, str):
                Exception(
                    'Units must be specified for top level data, if unitless: "arb"'
                )
            self.update(units=units)

        if dtype is not None:
            self.update(dtype=dtype)

        if attrs is not None:
            self.update(attrs=attrs)

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
                    try:
                        sym_map[s] = len(arr.coords[d])
                    except KeyError as e:
                        raise ValidationError(f'missing coordinate {d}') from e
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
        if 'dtype' in self and not can_cast_dtype(arr, self['dtype']):
            raise ValidationError(
                f'dtype {arr.dtype} not castable to  {self["dtype"]} for : \n {arr}'
            )

        # check that the coordinate dimensions are acceptable if a type is specified
        for cname, coord in self['coords'].items():
            # check values match for static coordinates
            if 'values' in coord:
                if not (arr.coords[cname] == coord['values']).all():
                    raise ValidationError(
                        f'not all coordinates in self match. \n Got: {arr.coords[cname]} \n Expected {coord["values"]} for : \n {arr}'
                    )

            # check that dtypes match for coordinates if a type is specified
            if 'dtype' in coord and not can_cast_dtype(
                arr.coords[cname], coord['dtype']
            ):
                raise ValidationError(
                    f'dtype {arr.coords[cname].dtype} not castable to {coord["dtype"]} for : \n {arr}'
                )

        # validate the metadata schema
        if 'attrs' in self:
            try:
                self['attrs'].model_validate(arr.attrs)
            except pydantic.ValidationError as e:
                msg = 'Failed to validate attributes: \n ' + str(e)
                raise ValidationError(msg) from e
