"""
Definitions for defining dataformats in xarrays.

Data formats are instructions for how to define the coordinates and labelling
systes for xarray objects for types of data sets.

Dataformats often relate the xarray format (i.e coordinat set and dimensions) to
a csv-like file format, and include instructions for reading that format
and converting them to others.

Some analysis functions will expect inputs to have specific dimensions, and labels,
and will refer to a specific DataFormat in the documentation when describing
the input.
"""

import xarray as xr
import numpy as _np
import uuid
import numpy as np
import copy
from typing import Union
from pathlib import Path
from typing import Mapping
import yaml
import json
import importlib
import sys
import jsonschema
# delete accessors before redefining, avoids a warning
try:
    del xr.DataArray.dfm
except AttributeError:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray


ALLOWED_SHAPE_SPECS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALLOWED_SHAPE_SPECS = list(ALLOWED_SHAPE_SPECS) + ["...", ...]
UNSPECIFIED_SPECS = ("...", ...)

SCHEMA_ATTRS_KEY = 'ARRSCHEMA_UID'


def _allowed_shape_spec(s: object):
    if isinstance(s, int):
        return True
    elif s in ALLOWED_SHAPE_SPECS:
        return True
    else:
        return False


__all__ = ["ArrSchemaRegistry", "stdreg", "arrschema", "load", "validate","save"]


def lazy_import_module(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


class ArrSchemaRegistry(dict):
    """Dict subclass that stores schema information."""

    def __init__(self):
        dict.__init__(self)
        self["schema"] = {}
        self["loaders"] = {}
        self["savers"] = {}
        self["converters"] = {}
        self._named_lookup = {}
        self._imported_modules = {}
        self._extension_lookup = {
            "loaders":{},
            "savers":{}
        }

    def show_schema(self):
        for name, schema in self["schema"].items():
            print(schema["name"])
            print("---------------")
            print(json.dumps(schema, sort_keys=True, indent=4))

    def show_loaders(self):
        for name, schema in self["loaders"].items():
            print(name)
            print("---------------")
            print(json.dumps(schema, sort_keys=True, indent=4))

    def find_schema(self, schema_name: str = None, schema_uid: str = None):
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
        dict
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
                    "Multiple schemas with the same name found. Try looking up by UID"
                )
            else:
                return schemas[0]
        elif schema_uid is not None and schema_name is None:
            schema = self["schema"][schema_uid]
        else:
            raise ValueError("Either schema_name or schema_uid must be provided.")
        return schema

    def import_saver(
        self,
        schema_name=None,
        schema_uid: str = None,
        extension: str = None,
        saver_type: str = None,
        verbose: bool = False,
    ):
        """
        Get a loader functiona associated
        with a schema.

        Parameters
        ----------
        schema_name : str, optional
            Name of the schema being saved, by default None
        schema_uid : str, optional
            UID of the schema being saved, by default None
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
        schema_uid = schema_uid,
        extension = extension,
        serializer_type = saver_type,
        verbose = verbose,
        )

    def import_loader(
        self,
        schema_name=None,
        schema_uid: str = None,
        extension: str = None,
        loader_type: str = None,
        verbose: bool = False,
    ):
        """
        Get a loader functiona associated
        with a schema.

        Parameters
        ----------
        schema_name : str, optional
            _description_, by default None
        schema_uid : str, optional
            _description_, by default None
        extension : str, optional
            _description_, by default None
        loader_type : str, optional
            _description_, by default None
        verbose : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        return self._import_serializer(
        'loader',
        schema_name=schema_name,
        schema_uid = schema_uid,
        extension = extension,
        serializer_type = loader_type,
        verbose = verbose,
        )

    def _import_serializer(
        self,
        loader_or_saver: str,
        schema_name=None,
        schema_uid: str = None,
        extension: str = None,
        serializer_type: str = None,
        verbose: bool = False,
    ) -> tuple[callable, dict]:
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
            Name of the schema, by default None
        schema_uid : str, optional
            UID of the schema, by default None
        extension : str, optional
            File extension, by default None
        serializer_type: str, optional
            Categorical string to match against
            when selecting a serializer.
        verbose: bool, optional
            Prints information which serializer was grabbed.

        Returns
        -------
        tuple[callable, dict]
            _description_
        """
        use_importmap = None

        # if a schema wasnt provided
        if schema_name is not None or schema_uid is not None:
            schema = self.find_schema(schema_name=schema_name, schema_uid=schema_uid)
            importmap = self[f"{loader_or_saver}s"][schema["uid"]]
            # if there are multiple loadmaps
            # and one wasnt specified, use the
            # first one
            if serializer_type is not None:
                if verbose:
                    print(f"using {serializer_type}")
                use_importmap = importmap[serializer_type]

            # use the default
            elif serializer_type is None and len(importmap) == 1:
                if verbose:
                    print("using only {loader_or_saver} available")
                key = list(importmap.keys())[0]
                use_importmap = importmap[key]

            # try to match against the file extension if a load type wasnt provided.
            elif serializer_type is None and len(importmap) > 1:
                if verbose:
                    print("trying to infer loader from file extension")
                matching = {
                    ltype: items["extension"]
                    for ltype, items in importmap.items()
                    if extension in items["extension"]
                }
                if len(matching) > 1:
                    raise ValueError(
                        f"Multiple loaders with the same extension for the same schema. Specify one of {list(importmap.keys())}"
                    )
                else:
                    key = list(matching.keys())[0]
                    use_importmap = importmap[key]
            else:
                raise Exception

        # if a schema wasnt specified in any way
        # try to infer by the file extension
        elif extension is not NotImplemented:
            importmap = self._extension_lookup[f"{loader_or_saver}s"][extension]
            if len(importmap) > 1:
                raise ValueError(
                    f"Multiple loaders with the same extension for the same extension {extension}. Specify the schema you are trying to load."
                )
            use_importmap = importmap[0]
            schema = self.find_schema(schema_uid=use_importmap["schema_uid"])

        # we figured out what function we should be using
        # import it in, cache it for later, and return a
        # pointer to the function
        modstr = use_importmap["funspec"].split(":")[0]
        funstr = use_importmap["funspec"].split(":")[1]
        if modstr not in self._imported_modules:
            self._imported_modules[modstr] = lazy_import_module(modstr)
        return getattr(self._imported_modules[modstr], funstr), schema

    def _find_loadmap_by_extension(self, extension: str):
        loadmap = self._extension_lookup[extension]
        if len(loadmap) > 1:
            raise ValueError(
                f"More then one loader asociated with extension {extension}. Cant find a loader."
            )
        return loadmap[0]

    def _add_serialization(
        self,
        loader_or_saver: str,
        funspec: str,
        extension: str,
        serial_type: str,
        schema_name: str = None,
        schema_uid: str = None,
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
            _description_
        """
                # add to the extension lookup
        if isinstance(extension, str):
            extension = [extension]

        if schema_name is None and schema_uid is None:
            raise ValueError(f"Must provide either name or uid.")

        if schema_uid is not None:
            schema = self["schema"][schema_uid]
        else:
            schema = self.find_schema(schema_name)

        if serial_type is None:
            serial_type = ""

        load_map = {
            "funspec": funspec,
            "schema_uid": schema["uid"],
            f"{loader_or_saver}_type": serial_type,
            "extension": extension,
        }

        if schema["uid"] in self[f"{loader_or_saver}s"]:
            if serial_type in self[f"{loader_or_saver}s"][schema["uid"]]:
                raise ValueError('Each load type can have 1 loader per schema.')
            self[f"{loader_or_saver}s"][schema["uid"]][serial_type] = load_map
        else:
            self[f"{loader_or_saver}s"][schema["uid"]] = {serial_type: load_map}

        # add to the extension lookup table
        for e in extension:
            if e in self._extension_lookup:
                self._extension_lookup[f"{loader_or_saver}s"][e].append(load_map)
            else:
                self._extension_lookup[f"{loader_or_saver}s"][e] = [load_map]

    def add_saver(
        self,
        funspec: str,
        extension: str,
        saver_type: str,
        schema_name: str = None,
        schema_uid: str = None,
    ):
        """


        Parameters
        ----------
        funspec : str
            _description_
        extension : list[str], optional
            _description_, by default None
        saver_type : str, optional
            Specify the type of saver (e.g. csv like, HDF5, group_saveable).
            If notprovided, '' is used.
        schema_name : str, optional
            _description_, by default None
        schema_uid : str, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        self._add_serialization(
            "saver",
            funspec=funspec,
            extension=extension,
            serial_type=saver_type,
            schema_name=schema_name,
            schema_uid=schema_uid
        )

    def add_loader(
        self,
        funspec: str,
        extension: str,
        loader_type: str,
        schema_name: str = None,
        schema_uid: str = None,
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
        schema_name : str, optional
            _description_, by default None
        schema_uid : str, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        self._add_serialization(
            "loader",
            funspec=funspec,
            extension=extension,
            serial_type=loader_type,
            schema_name=schema_name,
            schema_uid=schema_uid
        )

    def add_schema(
        self,
        schema: dict | Mapping | Path,
    ):
        # if its a path, load it in
        if isinstance(schema, Path) or isinstance(schema, str):
            path = Path(schema)
            if path.suffix == ".json":
                loader = json.load
            if path.suffix == ".yaml" or path.suffix == ".yml":
                loader = yaml.safe_load
            with open(path, "r") as f:
                schema = loader(f)
        # make it a real schema
        schema = arrschema(**schema)

        # add it to the registry
        uid = schema["uid"]
        self["schema"][uid] = schema

        # add it to the named lookup
        if schema["name"] in self._named_lookup:
            self._named_lookup[schema["name"]].append(schema)
        else:
            self._named_lookup[schema["name"]] = [schema]


"""Standard registry to load in and define array schemas."""
stdreg = ArrSchemaRegistry()


class ValidationError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

def save(
    path: str | Path,
    arr: object,
    *saver_args,
    registry: ArrSchemaRegistry,
    schema_name: str = None,
    schema_uid: str = None,
    saver_type: str = None,
    verbose=False,
    **saver_kwargs,
) -> object:
    """
    Load a DataArray object with a schema.

    Parameters
    ----------
    data : Path | str
        _description_
    schema_name : str, optional
        _description_, by default None
    schema_uid : str, optional
        _description_, by default None
    registry : ArrSchemaRegistry
        _description_, by default stdreg
    verbose: bool = True

    Returns
    -------
    object
        _description_
    """
    # lookup by the schema if provided
    extension = "".join(Path(path).suffixes)

    # if a specific schema wasnt asked for
    # then try to use one thats already attatched
    if SCHEMA_ATTRS_KEY in arr.attrs and schema_uid is None and schema_name is None:
        schema_uid = arr.attrs[SCHEMA_ATTRS_KEY]

    # grab he right saver function
    saver, schema = registry.import_saver(
        schema_name=schema_name,
        schema_uid=schema_uid,
        extension=extension,
        saver_type=saver_type,
        verbose=verbose,
    )

    # validate on the way in to the saver
    validate(arr, schema=schema, registry=registry)

    # save it
    return saver(path, arr, *saver_args, **saver_kwargs)

def load(
    path: Path | str,
    *load_args,
    registry: ArrSchemaRegistry,
    schema_name: str = None,
    schema_uid: str = None,
    loader_type: str = None,
    verbose=False,
    **load_kwargs,
) -> object:
    """
    Load a DataArray object with a schema.

    Parameters
    ----------
    data : Path | str
        _description_
    schema_name : str, optional
        _description_, by default None
    schema_uid : str, optional
        _description_, by default None
    registry : ArrSchemaRegistry, optional
        _description_, by default stdreg
    verbose: bool = True

    Returns
    -------
    object
        _description_
    """
    # lookup by the schema if provided
    extension = "".join(Path(path).suffixes)
    loader, schema = registry.import_loader(
        schema_name=schema_name,
        schema_uid=schema_uid,
        extension=extension,
        loader_type = loader_type,
        verbose=verbose,
    )
    read = loader(path, *load_args, **load_kwargs)
    validate(read, schema=schema, registry=registry)
    return read


def validate(
    arr: "xarray.DataArray",
    *,
    registry: ArrSchemaRegistry,
    schema_name: str = None,
    schema_uid: str = None,
    schema: Mapping = None,
    attach_schema: bool = True
):
    """
    Check if a DataArray conforms to a particular schema.

    Parameters
    ----------
    arr : xarray.DataArray
        DataArray object to validate
    schema_name : str, optional
        Non-unique name of the schema in the registry, by default None
    schema_uid : str, optional
        Unique ID of the schema, by default None
    schema : Mapping, optional
        A schema dictionary to validate against, by default None
    registry : ArrSchemaRegistry, optional
        _description_, by default stdreg
    attach_schema : bool, optional
        If True, the schema is dumped into a string and
        attatched to the attrs of the input data array.
        The default is True.

    """
    # get the schema
    if schema is None:
        schema = registry.find_schema(schema_name=schema_name, schema_uid=schema_uid)

    # compare dimensions to the actual shape,
    # map symbolic dimensions to actual dimensions
    sym_map = {}
    for d, s in zip(schema["dims"], schema["shape"]):
        # if its a symbolic dimension size
        # according to first appearence of that symbol
        # and check it against the existing sym_map
        if d != "..." and isinstance(s, str):
            if s not in sym_map:
                sym_map[s] = len(arr.coords[d])
            if len(arr.coords[d]) != sym_map[s]:
                raise ValidationError(
                    f"Coord {d} length {len(arr.coords[d])} doesnt match shape symbolic spec {s}={sym_map[s]}"
                )
        # if its a statically sized dimension
        # then just check that it matches
        elif d != "..." and isinstance(s, int):
            if len(arr.coords[d]) != s:
                raise ValidationError(
                    f"Coord {d} length {len(arr.coords[d])} doesnt match shape spec {s}"
                )

    # check that the static types match
    if schema["dtype"] != atomic_str(arr.dtype):
        raise ValidationError(
            f"dtype {arr.dtype} doesnt match expected {schema['dtype']} for : \n {arr}"
        )

    # check that the coordinate dimensions are acceptable
    for cname, coord in schema["coords"].items():
        # check values match for static coordinates
        if "values" in coord:
            if not (arr.coords[cname] == coord["values"]).all():
                raise ValidationError(
                    f"not all coordinates in schema match. \n Got: {arr.coords[cname]} \n Expected {coord['values']} for : \n {arr}"
                )

        # check that dtypes match for coordinates
        if coord["dtype"] != atomic_str(arr.coords[cname].dtype):
            raise ValidationError(
                f"dtype {arr.coords[cname].dtype} doesnt match expected {coord['dtype']} for : \n {arr}"
            )

    # validate the metadata schema
    try:
        jsonschema.validate(dict(arr.attrs),schema = schema['attrs_schema'])
    except jsonschema.exceptions.ValidationError as e:
        msg = 'Failed to validate attrs schema : \n '+str(e)
        raise ValidationError(msg) from e

    if attach_schema:
        arr.attrs[SCHEMA_ATTRS_KEY] = schema['uid']


def atomic_str(dtype: str):
    """
    Validate and return a string according to an atomic type.
    """
    return _np.dtype(dtype)


# %% defines what is in a dataformat
def arrschema(
    name: str,
    shape: tuple[str | int],
    dims: tuple[str],
    dtype: str,
    units: Mapping | str = None,
    coords: Mapping = None,
    uid: str = None,
    attrs_schema: Mapping = None
):
    """
    Generate a dictionary that describes an array structure.

    Parameters
    ----------
    name : str
        Name of the array structure.
    shape : tuple[str  |  int]
        Shape of structure. Ellipses indicate arbitrary dimensions,
        letters indicate a required dimension of unknown length, and
        integers indicate a required dimension of a required length.
    dims : tuple[str]
        Names assigned to dimensions specified by shape. Any required
        dimension must be names, and arbitrary dimensions must also be
        ellipses.
    dtype : str
        Type string, corresponds to numpy's dtype (e.g. f8, c8, u8, etc)
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
    # check that the shape and dims make sense
    if len(shape) != len(dims):
        raise Exception("shape and dims length must match")
    shape = list(copy.copy(shape))
    dims = list(copy.copy(dims))
    # replace ellipses with strings to make it
    # more compatable with json
    shape_lookup = {}
    for spec_tuple in (shape, dims):
        for i, s in enumerate(spec_tuple):
            if s == ...:
                spec_tuple[i] = "..."
    # check that shape and dimensions
    # make sense and agree with eachother
    for i, (s, d) in enumerate(zip(shape, dims)):
        shape_lookup[d] = s
        # can only used valid shape specifications
        if not _allowed_shape_spec(s):
            raise Exception(f"Shape spec {s} must be a letter or an ellipses")
        if not isinstance(d, str) and d != ...:
            raise ValueError("Dimensions names must be strings or ...")

        # if one has an ..., the other must too
        s_unspecd = s in UNSPECIFIED_SPECS
        d_unspecd = d in UNSPECIFIED_SPECS
        if s_unspecd ^ d_unspecd:
            raise ValueError(
                "unspecified shapes specs (...) must have unspecified dimension names. Specified shapes must have specified dimension names."
            )

    # check that the dtype str is valid
    dtype = str(np.dtype(dtype))

    # check that the coordinates are valid
    out_coords = {}
    if coords is not None:
        for c, crd in coords.items():
            try:
                cunits = coords[c]["units"]
            except KeyError:
                cunits = None
            if cunits is not None and not isinstance(cunits, str):
                for cui in cunits:
                    if not isinstance(crd["units"], str):
                        raise ValueError(
                            "Coordinates can have only a single unit (i.e. must be a string.)"
                        )

            cshape = (shape_lookup[c],)
            cdims = (c,)
            cschema = arrschema(
                name=d, dims=cdims, shape=cshape, dtype=crd["dtype"], units=cunits
            )
            if "values" in crd:
                if len(crd["values"]) != shape_lookup[c]:
                    raise ValueError(
                        f"Coord values {c} length do not match dimension spec {shape_lookup[c]}"
                    )
                cschema["values"] = crd["values"]

            # require that the coordinat dimension shapes match
            out_coords[c] = cschema
    else:
        coords = {}
    # assign a uid:
    uid = uuid.uuid4()

    # check that the units mapping is valid
    if isinstance(units, str):
        units = units
    elif units is not None:
        if len(units) > 1:
            raise Exception("Only 1 unit dimension is allowed")
        if len(units) == 0:
            raise ValueError(f"Must supply exactly 1 unit dimension {units}")
        udims = list(units.keys())
        # replace any ... with strings
        # for json compatability
        for ud in udims:
            if ud in UNSPECIFIED_SPECS:
                units["..."] = units[ud]
                units.pop(ud)
        # check that its valied
        for ud in units:
            if ud not in dims:
                raise ValueError(f"Unit dimension {ud} not in dims.")
            # if its a string, then its a global unit
            if isinstance(units[ud], str):
                pass
            else:
                # if ud is defined in coords,
                # make sure the lengths

                if ud not in coords:
                    raise ValueError(
                        "If providing multiple units for a unit dimensin, that dimension must have defined coordinates."
                    )

                if len(units[ud]) != len(coords[ud]["values"]):
                    raise ValueError(
                        f"Unit dimension values {ud} array must match length of the matching coordinates."
                    )

    if attrs_schema is None:
        attrs_schema = {}

    out = {
        "uid": str(uid),
        "name": name,
        "shape": tuple(shape),
        "dims": tuple(dims),
        "dtype": dtype,
        "units": units,
        "coords": out_coords,
        "attrs_schema": attrs_schema
    }

    if units is None:
        out.pop('units')

    return out