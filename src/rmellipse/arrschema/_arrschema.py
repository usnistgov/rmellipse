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
import dict_hash
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
    'ArrayClassRegistry',
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
    Interface for an array structure with annotate dimensions and coordinates.

    Inspired by xarray DataArray. Classes conforming to this specification
    can be registered in an ArrayClassRegister.
    """

    __slots__ = ()
    schema: 'ArraySchema'
    registry: 'ArrayClassRegistry'

    def __init__(self, *args, **kwargs):
        xr.DataArray.__init__(self, *args, **kwargs)

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        # add this class to the registry
        xr.DataArray.__init_subclass__(*args, **kwargs)
        cls.registry.add_class(cls, cls.schema)

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

    def save(
        self,
        path: str | Path,
        *saver_args,
        saver_type: str | None = None,
        validate_schema: bool = True,
        verbose: bool = False,
        **saver_kwargs,
    ) -> object:
        """
        Save array using specified saver.

        This function dispatches to saver functions supplied
        to the registry through add_saver.

        Parameters
        ----------
        path : str | Path
            Path to location where data should will be saved.
        *saver_args : positional arguments
            Passed to saver as positional args.
        saver_type : str, optional
            Used to pick which saver to use.
            The default is None.
        validate_schema : bool, optional
            If True, raise an exception if data does not conform to schema.
            The default is True.
        verbose : bool, optional
            DESCRIPTION. The default is False.
        **saver_kwargs : keyword arguments
            Passed to saver as keyword arguments.

        Returns
        -------
        object
            Object returned by saver fuction.

        """
        # lookup by the schema if provided
        extension = ''.join(Path(path).suffixes)

        # grab the right saver function
        saver = self.registry.import_saver(
            extension=extension,
            saver_type=saver_type,
            verbose=verbose,
            schema_uid=self.schema['uid'],
        )

        # validate on the way in to the saver
        if validate_schema:
            self.schema.validate(self)

        # save it
        return saver(path, self, *saver_args, **saver_kwargs)

    def convert_to(self, output_type: type) -> 'AnnotatedArray':
        """
         Convert self to a specified type.

        This function dispatches to converter functions supplied
        to the registry through add_converter.

         Parameters
         ----------
         output_schema : ArraySchema | type
             Schema for output array type or an array schema
             class built by a registry.

         Returns
         -------
         AnnotatedArray
             Array matching output schema.

        """
        # allow someoine to pass in a class object
        converter_fun = self.registry.import_converter(
            input_schema_uid=self.schema['uid'],
            output_schema_uid=output_type.schema['uid'],
        )
        out = output_type(converter_fun(self))
        out.validate()
        return out

    @classmethod
    def convert_from(cls, input_array: 'AnnotatedArray') -> Self:
        """
         Initialize from an input array.

        This function dispatches to converter functions supplied
        to the registry through add_converter.

         Parameters
         ----------
         input_array : AnnotatedArray
             Input data.

         Returns
         -------
         Self
             New array.

        """
        converter_fun = cls.registry.import_converter(
            input_schema_uid=input_array.schema['uid'],
            output_schema_uid=cls.schema['uid'],
        )
        new_data = converter_fun(input_array)
        out = cls(new_data)
        out.validate()
        return out

    @classmethod
    def load(
        cls,
        path: Path | str,
        *load_args,
        loader_type: str | None = None,
        validate_schema: bool = True,
        verbose=False,
        **load_kwargs,
    ) -> Self:
        """
         Save a dataset.

        This function dispatches to loader functions supplied
        to the registry through add_loader.

         Parameters
         ----------
         path : Path | str
                 Path to data to load.
         loader_type : str, optional
                 If there are mulitple loaders, specify which one.
         validate_schema : bool, optional
                 If True, raise an exception if data do not conform to schema.
                 The default is True.
         verbose : bool, optional
                 If True, print debugging messages.
                 The default is False.

         Returns
         -------
         Self
                 Loaded data.
        """
        # lookup by the schema if provided
        extension = ''.join(Path(path).suffixes)
        loader = cls.registry.import_loader(
            extension=extension,
            loader_type=loader_type,
            verbose=verbose,
            schema_uid=cls.schema['uid'],
        )
        read = loader(path, *load_args, **load_kwargs)
        out_array = cls.from_dataarray(read)
        if validate_schema:
            cls.schema.validate(out_array)

        return out_array

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
        uid: str | None = None,
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
        uid : str, optional
            The uid of a schema can be provided here, it is created using
            uuid4 if it is not provided, by default None.
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
            'uid': uid,
            'name': name,
            'shape': tuple(shape),
            'dims': tuple(dims),
            'dtype': dtype,
            'units': units,
            'coords': out_coords,
            'attrs_schema': attrs_schema,
        }

        if uid is None:
            out.pop('uid')
            out['uid'] = dict_hash.sha256(out)
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


class ArrayClassRegistry:
    """
    Organizes schema, classes, loaders, savers, and converters.

    All of these objects are stored in dictionaries, indexed by unique ids.
    The unique ids correspond to the schema's unique id attribute.
    """

    def __init__(self):
        # array schema by uid
        self.classes = {}
        self.schema = {}
        self.loaders = {}
        self.savers = {}

        # dictionary of [input_uid][output_uid][function]
        self.converters = {}

        self._named_lookup = {}
        self._imported_modules = {}
        self._extension_lookup = {'loaders': {}, 'savers': {}}

    def __getitem__(self, key: str):
        return getattr(self, key)

    def find_schema(
        self, schema_name: str | None = None, schema_uid: str | None = None
    ) -> ArraySchema:
        """
        Find a schema.

        Name of uid must be provided. Will throw an error if only
        the name is provided and multiple schema in the registry
        share a name.

        Parameters
        ----------
        name : str
            Name of the schema to look for.
        uid: str
            uid of the schema to find.

        Returns
        -------
        ArraySchema
            Requested schema

        Raises
        ------
        ValueError
            If multiple schemas in the regsitry have the same name.
        """
        if schema_uid is None and schema_name is not None:
            schemas = self._named_lookup[schema_name]
            if len(schemas) > 1:
                raise ValueError(
                    'Multiple schemas with the same name found. Try looking up by UID'
                )
            else:
                return schemas[0]
        elif schema_uid is not None and schema_name is None:
            schema = self.schema[schema_uid]
        else:
            raise ValueError('Either schema_name or schema_uid must be provided.')
        return schema

    def show_schema(self):
        """
        Show schema.

        Returns
        -------
        None.

        """
        for name, schema in self.schema.items():
            print(schema['name'])
            print('---------------')
            print(json.dumps(schema, sort_keys=True, indent=4))

    def import_saver(
        self,
        schema_name=None,
        schema_uid: str | None = None,
        extension: str | None = None,
        saver_type: str | None = None,
        verbose: bool = False,
    ):
        """
        Get a loader function associated
        with a schema.

        Parameters
        ----------
        schema_name : str, optional
            Name of the schema being saved, by default None
        schema_uid : str, optional
            Unique id of the schema being saved, by default None
        extension : str, optional
            File extension, by default None
        saver_type : str, optional
            Type of saver to use, by default None
        verbose : bool, optional
            Print info, by default False

        Returns
        -------
        callable
            Saver function imported from the registry
        """
        return self._import_serializer(
            'saver',
            schema_name=schema_name,
            schema_uid=schema_uid,
            extension=extension,
            serializer_type=saver_type,
            verbose=verbose,
        )

    def import_loader(
        self,
        schema_name=None,
        schema_uid: str | None = None,
        extension: str | None = None,
        loader_type: str | None = None,
        verbose: bool = False,
    ):
        """
        Get a loader function associated
        with a schema.

        Parameters
        ----------
        schema_name : str, optional
            Name of the schema being saved, by default None
        schema_uid : str, optional
            Unique id of the schema being saved, by default None
        extension : str, optional
            File extension, by default None
        loader_type : str, optional
            Type of loader to use, by default None
        verbose : bool, optional
            Print info, by default False
        Returns
        -------
        _type_
            _description_
        """
        return self._import_serializer(
            'loader',
            schema_name=schema_name,
            schema_uid=schema_uid,
            extension=extension,
            serializer_type=loader_type,
            verbose=verbose,
        )

    def _import_serializer(
        self,
        loader_or_saver: str,
        schema_name: str | None = None,
        schema_uid: str | None = None,
        extension: str | None = None,
        serializer_type: str | None = None,
        verbose: bool = False,
    ) -> Callable:
        """
        Get a loader function for a particular schema.

        Can provide (in order of lookup priority) schema_uid, schema_name, or
        the file extension.

        Parameters
        ----------
        loader_or_saver : str,
            Specified if this is a loading or saving
            function.
        schema_name : str, optional
            Name of the schema, by default None.
        schema_uid : str, optional
            Unique id of the schema, by default None.
        extension : str, optional
            File extension, by default None.
        serializer_type: str, optional
            Categorical string to match against
            when selecting a serializer.
        verbose: bool, optional
            Prints information which serializer was grabbed.

        Returns
        -------
        Callable
            Imported loader or saver.
        """
        use_importmap = None

        # if a schema wasnt provided
        if schema_name is not None or schema_uid is not None:
            schema = self.find_schema(schema_name=schema_name, schema_uid=schema_uid)
            importmap = self[f'{loader_or_saver}s'][schema['uid']]
            # if there are multiple loadmaps
            # and one wasnt specified, use the
            # first one
            if serializer_type is not None:
                if verbose:
                    print(f'using {serializer_type}')
                use_importmap = importmap[serializer_type]

            # use the default
            elif serializer_type is None and len(importmap) == 1:
                if verbose:
                    print('using only {loader_or_saver} available')
                key = list(importmap.keys())[0]
                use_importmap = importmap[key]

            # try to match against the file extension if a load type wasnt provided.
            elif serializer_type is None and len(importmap) > 1:
                if verbose:
                    print('trying to infer loader from file extension')
                matching = {
                    ltype: items['extension']
                    for ltype, items in importmap.items()
                    if extension in items['extension']
                }
                if len(matching) > 1:
                    raise ValueError(
                        f'Multiple loaders with the same extension for the same schema. Specify one of {list(importmap.keys())}'
                    )
                else:
                    key = list(matching.keys())[0]
                    use_importmap = importmap[key]
            else:
                raise Exception

        # if a schema wasnt specified in any way
        # try to infer by the file extension
        elif extension is not NotImplemented:
            importmap = self._extension_lookup[f'{loader_or_saver}s'][extension]
            if len(importmap) > 1:
                raise ValueError(
                    f'Multiple loaders with the same extension for the same extension {extension}. Specify the schema you are trying to load.'
                )
            use_importmap = importmap[0]
            schema = self.find_schema(schema_uid=use_importmap['schema_uid'])

        if use_importmap is None:
            raise ValueError(
                'Never found a value for use_importmap inside _arrchema.py::ArrayClassRegistry::_import_serializer'
            )

        # we figured out what function we should be using
        # import it in, cache it for later, and return a
        # pointer to the function
        modstr = use_importmap['funspec'].split(':')[0]
        funstr = use_importmap['funspec'].split(':')[1]
        if modstr not in self._imported_modules:
            self._imported_modules[modstr] = _lazy_import_module(modstr)
        return getattr(self._imported_modules[modstr], funstr)

    def _find_loadmap_by_extension(self, extension: str):
        loadmap = self._extension_lookup[extension]
        if len(loadmap) > 1:
            raise ValueError(
                f'More then one loader asociated with extension {extension}. Cant find a loader.'
            )
        return loadmap[0]

    def _add_serialization(
        self,
        loader_or_saver: str,
        funspec: str,
        extension: str | list[str],
        serial_type: str,
        schema_name: str | None = None,
        schema_uid: str | None = None,
    ):
        """
        Add a serializing function (loader or saver)

        Parameters
        ----------
        loader_or_saver : str, {loader, saver}
            State if this is a loader or a saver function.
        funspec : str
            _description_
        extension : list[str], optional
            _description_, by default None
        serial_type : str, optional
            Specify the type of serializer (e.g. csv like, HDF5, group_saveable).
            If notprovided, '' is used. Used for identifing
            methods when calling load or save.
        schema_name : str, optional
            _description_, by default None
        schema_uid : str, optional
            _description_, by default None

        Raises
        ------
        ValueError
            If invalid input.
        """
        loaders_or_savers_dict = None
        if loader_or_saver == 'loader':
            loaders_or_savers_dict = self.loaders
        if loader_or_saver == 'saver':
            loaders_or_savers_dict = self.savers
        if loaders_or_savers_dict is None:
            raise ValueError(
                'Expected "loader" or "saver", found "{}"'.format(loader_or_saver)
            )

        # add to the extension lookup
        if isinstance(extension, str):
            extension_list = [extension]
        else:
            extension_list = extension

        if schema_name is None and schema_uid is None:
            raise ValueError('Must provide either name or uid.')

        if schema_uid is not None:
            schema = self.schema[schema_uid]
        else:
            schema = self.find_schema(schema_name)

        if serial_type is None:
            serial_type = ''

        load_map = {
            'funspec': funspec,
            'schema_uid': schema['uid'],
            f'{loader_or_saver}_type': serial_type,
            'extension': extension_list,
        }

        if schema['uid'] in loaders_or_savers_dict.keys():
            if serial_type in loaders_or_savers_dict[schema['uid']]:
                raise ValueError('Each load type can have 1 loader per schema.')
            loaders_or_savers_dict[schema['uid']][serial_type] = load_map
        else:
            loaders_or_savers_dict[schema['uid']] = {serial_type: load_map}

        # add to the extension lookup table
        for e in extension_list:
            if e in self._extension_lookup:
                self._extension_lookup[f'{loader_or_saver}s'][e].append(load_map)
            else:
                self._extension_lookup[f'{loader_or_saver}s'][e] = [load_map]

    def add_converter(
        self,
        funspec: str,
        input_schema: dict,
        output_schema: dict,
    ):
        """
        Add a converting functiom between two schema.

        Converting functions take in exactly one argument
        and output exactly 1

        Parameters
        ----------
        funspec : str
            Path spec of function in dot-notation
            (module.submodule:function)
        input_schema : dict, optional
            name of schema (or provide the uid) for converter
        output_schema : dict, optional
            output_schema for converter
        """
        # get the actual schema
        input_uid = input_schema['uid']
        output_uid = output_schema['uid']

        if input_uid not in self.converters:
            self.converters[input_uid] = {}
        if output_uid not in self.converters[input_uid]:
            self.converters[input_uid][output_uid] = {}

        self.converters[input_uid][output_uid] = funspec

    def import_converter(
        self,
        input_schema_uid: str | None = None,
        output_schema_uid: str | None = None,
    ) -> Callable:
        """
        Add a converting functiom between two schema.

        Converting functions take in exactly one argument
        and output exactly 1

        Parameters
        ----------
        input_schema_uid : str, optional
            uid of input schema (or provide the name), by default None
        output_schema_uid : str, optional
            _description_, by default None
        """

        try:
            fspec = self.converters[input_schema_uid][output_schema_uid]
        except KeyError as e:
            msg = f'conversion from {input_schema_uid} to {output_schema_uid} not defined in registry.'
            raise ValueError(msg) from e

        modstr = fspec.split(':')[0]
        funstr = fspec.split(':')[1]
        if modstr not in self._imported_modules:
            self._imported_modules[modstr] = _lazy_import_module(modstr)
        return getattr(self._imported_modules[modstr], funstr)

    def add_saver(
        self,
        funspec: str,
        extension: str,
        saver_type: str,
        schema: dict,
    ):
        """
        Add a saving function to the registry.

        Parameters
        ----------
        funspec : str
            _description_
        extension : list[str], optional
            _description_, by default None
        saver_type : str, optional
            Specify the type of saver (e.g. csv like, HDF5, group_saveable).
            If notprovided, '' is used.
        schema : dict, optional
            Schema, by default None

        Raises
        ------
        ValueError
            _description_
        """
        self._add_serialization(
            'saver',
            funspec=funspec,
            extension=extension,
            serial_type=saver_type,
            schema_uid=schema['uid'],
        )

    def add_loader(
        self, funspec: str, extension: str, loader_type: str, schema: dict | Mapping
    ):
        """


        Parameters
        ----------
        funspec : str
            _description_
        extension : list[str], optional
            _description_, by default None
        loader_type : str, optional
            Specify the type of loader (e.g. csv like, HDF5, group_saveable).
            If notprovided, '' is used.
        schema : dict | Mapping
                Schema to use

        Raises
        ------
        ValueError
            _description_
        """
        self._add_serialization(
            'loader',
            funspec=funspec,
            extension=extension,
            serial_type=loader_type,
            schema_uid=schema['uid'],
        )

    def add_schema(
        self,
        schema: dict | Mapping | Path,
    ):
        # if its a path, load it in
        if isinstance(schema, Path) or isinstance(schema, str):
            path = Path(schema)
            if path.suffix == '.json':
                loader = json.load
            elif path.suffix == '.yaml' or path.suffix == '.yml':
                loader = yaml.safe_load
            else:
                raise ValueError(
                    'path suffix was not .json .yaml or .yml for ArrayClassRegistry::add_schema'
                )
            with open(path, 'r') as f:
                schema_dictlike = loader(f)
        else:
            schema_dictlike = schema
        # make it a real schema
        schema = ArraySchema(**schema_dictlike)

        # add it to the registry
        uid = schema['uid']
        self.schema[uid] = schema

        # add it to the named lookup
        if schema['name'] in self._named_lookup:
            self._named_lookup[schema['name']].append(schema)
        else:
            self._named_lookup[schema['name']] = [schema]

    def add_class(self, class_to_add: type[AnnotatedArray], schema: ArraySchema):
        self.add_schema(schema)
        self.classes[schema['uid']] = class_to_add
