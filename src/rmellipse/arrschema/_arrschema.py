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
from typing import Mapping, Any, Tuple
from pathlib import Path
import yaml
import json
import importlib
import sys
import jsonschema
from abc import ABC, abstractmethod
import dict_hash

# delete accessors before redefining, avoids a warning
try:
	del xr.DataArray.dfm
except AttributeError:
	pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import xarray


ALLOWED_SHAPE_SPECS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALLOWED_SHAPE_SPECS = list(ALLOWED_SHAPE_SPECS) + ['...', ...]
UNSPECIFIED_SPECS = ('...', ...)

SCHEMA_ATTRS_KEY = 'ARRSCHEMA'
ANNOTATION_ATTRS_KEY = 'ARRANNOTATION'


def _allowed_shape_spec(s: object):
	if isinstance(s, int):
		return True
	elif s in ALLOWED_SHAPE_SPECS:
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


__all__ = [
	'ArrSchemaRegistry',
	'ValidationError',
	'arrschema',
	'load',
	'validate',
	'save',
	'convert',
	'annotate',
	'zeros',
	'as_schema',
]


class AnnotatedArrayLike(ABC):
	"""
	Interface for an array structure with annotate dimensions and coordinates.

	Inspired by xarray DataArray. Objects conforming to this specification
	can be interacted with by functions in this module to generate annotations
	or validate against schema.
	"""

	@property
	@abstractmethod
	def shape(self) -> tuple[int]:
		"""Tuple of dimension sizes corresponding to dim."""
		pass

	@property
	@abstractmethod
	def dims(self) -> tuple[str]:
		"""Tuple of dimension names corresponding to shape."""
		pass

	@property
	@abstractmethod
	def coords(self) -> Mapping[str, 'AnnotatedArrayLike']:
		"""Mapping of dimension names to cooordinate setsgit ."""
		pass

	@property
	@abstractmethod
	def dtype(self) -> str | Any:
		"""Data type that conforms to numpy dtype_string specs."""
		pass

	@property
	@abstractmethod
	def attrs(self) -> dict:
		"""JSON compatable dictionary of metadata."""
		pass


def annotate(arr: AnnotatedArrayLike):
	"""Generate an array annotation and attatch it as metadata."""
	annotation = {}
	attrs = convert_h5attrs_to_json_types(arr.attrs)
	if SCHEMA_ATTRS_KEY in attrs:
		attrs.pop(SCHEMA_ATTRS_KEY)
	if ANNOTATION_ATTRS_KEY in attrs:
		attrs.pop(ANNOTATION_ATTRS_KEY)

	annotation = {
		'shape': list(arr.shape),
		'dims': list(arr.dims),
		'dtype': str(arr.dtype),
		'attrs': attrs,
		'coords': {},
	}

	# add coordinate annotations
	for k, v in arr.coords.items():
		annotation['coords']['dtype'] = str(v.dtype)
		annotation['coords']['shape'] = list(v.shape)

	arr.attrs[ANNOTATION_ATTRS_KEY] = json.dumps(annotation)


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
		# array schema by
		self['schema'] = {}
		self['loaders'] = {}
		self['savers'] = {}
		# dictionary of [input_uid][output_uid][function]
		self['converters'] = {}
		self._named_lookup = {}
		self._imported_modules = {}
		self._extension_lookup = {'loaders': {}, 'savers': {}}

	@property
	def schema(self):
		return self['schema']

	@property
	def loaders(self):
		return self['loaders']

	@property
	def savers(self):
		return self['savers']

	def show_schema(self):
		for name, schema in self['schema'].items():
			print(schema['name'])
			print('---------------')
			print(json.dumps(schema, sort_keys=True, indent=4))

	def show_loaders(self):
		for name, schema in self['loaders'].items():
			print(name)
			print('---------------')
			print(json.dumps(schema, sort_keys=True, indent=4))

	def find_schema(self, schema_name: str = None, schema_uid: str = None) -> dict:
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
					'Multiple schemas with the same name found. Try looking up by UID'
				)
			else:
				return schemas[0]
		elif schema_uid is not None and schema_name is None:
			schema = self['schema'][schema_uid]
		else:
			raise ValueError('Either schema_name or schema_uid must be provided.')
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
			schema_uid=schema_uid,
			extension=extension,
			serializer_type=saver_type,
			verbose=verbose,
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
			schema_uid=schema_uid,
			extension=extension,
			serializer_type=loader_type,
			verbose=verbose,
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

		# we figured out what function we should be using
		# import it in, cache it for later, and return a
		# pointer to the function
		modstr = use_importmap['funspec'].split(':')[0]
		funstr = use_importmap['funspec'].split(':')[1]
		if modstr not in self._imported_modules:
			self._imported_modules[modstr] = lazy_import_module(modstr)
		return getattr(self._imported_modules[modstr], funstr), schema

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
			raise ValueError('Must provide either name or uid.')

		if schema_uid is not None:
			schema = self['schema'][schema_uid]
		else:
			schema = self.find_schema(schema_name)

		if serial_type is None:
			serial_type = ''

		load_map = {
			'funspec': funspec,
			'schema_uid': schema['uid'],
			f'{loader_or_saver}_type': serial_type,
			'extension': extension,
		}

		if schema['uid'] in self[f'{loader_or_saver}s']:
			if serial_type in self[f'{loader_or_saver}s'][schema['uid']]:
				raise ValueError('Each load type can have 1 loader per schema.')
			self[f'{loader_or_saver}s'][schema['uid']][serial_type] = load_map
		else:
			self[f'{loader_or_saver}s'][schema['uid']] = {serial_type: load_map}

		# add to the extension lookup table
		for e in extension:
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

		if input_uid not in self['converters']:
			self['converters'][input_uid] = {}
		if output_uid not in self['converters'][input_uid]:
			self['converters'][input_uid][output_uid] = {}

		self['converters'][input_uid][output_uid] = funspec

	def import_converter(
		self,
		input_schema_name: str = None,
		input_schema_uid: str = None,
		output_schema_name: str = None,
		output_schema_uid: str = None,
	) -> callable:
		"""
		Add a converting functiom between two schema.

		Converting functions take in exactly one argument
		and output exactly 1

		Parameters
		----------
		funspec : str
		    Path spec of function in dot-notation
		    (module.submodule:function)
		input_schema_name : str, optional
		    name of schema (or provide the uid), by default None
		input_schema_uid : str, optional
		    uid of input schema (or provide the name), by default None
		output_schema_name : str, optional
		    _description_, by default None
		output_schema_uid : str, optional
		    _description_, by default None
		"""
		# get the actual schema
		input_schema = self.find_schema(input_schema_name, input_schema_uid)
		output_schema = self.find_schema(output_schema_name, output_schema_uid)

		input_uid = input_schema['uid']
		output_uid = output_schema['uid']

		try:
			fspec = self['converters'][input_uid][output_uid]
		except KeyError as e:
			msg = f'conversion from {input_schema["name"]} to {output_schema["name"]} not defined in registry.'
			raise ValueError(msg) from e
		modstr = fspec.split(':')[0]
		funstr = fspec.split(':')[1]
		if modstr not in self._imported_modules:
			self._imported_modules[modstr] = lazy_import_module(modstr)
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
			if path.suffix == '.yaml' or path.suffix == '.yml':
				loader = yaml.safe_load
			with open(path, 'r') as f:
				schema = loader(f)
		# make it a real schema
		schema = arrschema(**schema)

		# add it to the registry
		uid = schema['uid']
		self['schema'][uid] = schema

		# add it to the named lookup
		if schema['name'] in self._named_lookup:
			self._named_lookup[schema['name']].append(schema)
		else:
			self._named_lookup[schema['name']] = [schema]


class ValidationError(Exception):
	def __init__(self, *args, **kwargs):
		Exception.__init__(self, *args, **kwargs)


def save(
	path: str | Path,
	arr: AnnotatedArrayLike,
	*saver_args,
	registry: ArrSchemaRegistry,
	schema: dict = None,
	saver_type: str = None,
	validate_schema: bool = True,
	verbose=False,
	**saver_kwargs,
) -> object:
	"""
	Load a dataset using arrschema registry.

	Parameters
	----------
	path : str | Path
		_description_
	arr : object
		_description_
	registry : ArrSchemaRegistry
		_description_
	schema_name : str, optional
		_description_, by default None
	schema_uid : str, optional
		_description_, by default None
	saver_type : str, optional
		_description_, by default None
	validate_schema : bool, optional
		_description_, by default True
	verbose : bool, optional
		_description_, by default False

	Returns
	-------
	object
		_description_
	"""
	# lookup by the schema if provided
	extension = ''.join(Path(path).suffixes)

	# if a specific schema wasnt asked for
	# then try to use one thats already attatched

	if schema is None and SCHEMA_ATTRS_KEY in arr.attrs:
		schema_uid = json.loads(arr.attrs[SCHEMA_ATTRS_KEY])['uid']
	elif schema is not None:
		schema_uid = schema['uid']
	else:
		raise ValueError(
			'Arr must have a schema attatched to attrs OR schema must be provided as a key word argument.'
		)

	# grab he right saver function
	saver, schema = registry.import_saver(
		schema_uid=schema_uid,
		extension=extension,
		saver_type=saver_type,
		verbose=verbose,
	)

	# validate on the way in to the saver
	if validate_schema:
		validate(arr, schema=schema)

	# save it
	return saver(path, arr, *saver_args, **saver_kwargs)


def as_schema(
	array: xr.DataArray,
	registry: ArrSchemaRegistry = None,
	schema: dict | Mapping = None,
	schema_uid: str = None,
	schema_name: str = None,
):
	"""
	Cast an array into a schema.

	Casts the correct data_type of the data
	as well as the coordinates. If coordinates
	are fixed, applies those as the values.


	Parameters
	----------
	array : xr.DataArray
		_description_
	registry : ArrSchemaRegistry, optional
		_description_, by default None
	schema : dict | Mapping, optional
		_description_, by default None
	schema_uid : str, optional
		_description_, by default None
	schema_name : str, optional
		_description_, by default None

	Returns
	-------
	_type_
		_description_

	Raises
	------
	ValueError
		_description_
	"""

	# get schema if not given
	if schema is None:
		schema = registry.find_schema(schema_name=schema_name, schema_uid=schema_uid)

	new_shape_spec = schema['shape']
	new_dims_spec = schema['dims']
	new_dtype = schema['dtype']

	# require that specified dimensions be uninterrupted
	# i.e. at most 1 unspecified, arbitrary dimensions
	unq_vals, unq_counts = np.unique(new_shape_spec, return_counts=True)
	unspecified_count = unq_counts[unq_vals == '...'][0]
	if unspecified_count > 1:
		raise ValueError(
			f'Schema with >1 arbitrary dimension specifications (...) can not be intialized as xarrays from a schema: \n {json.dumps(schema, indent=True)}'
		)

	# instantiate new array
	new = array
	old_dims = array.dims
	for i, (si, di) in enumerate(zip(new_shape_spec, new_dims_spec)):
		if si == '...':
			break
		# print('forward', si, di)
		new = new.rename({old_dims[i]: di})
		if di in schema['coords']:
			crd_schema = schema['coords'][di]
			crd_dtype = crd_schema['dtype']
			if 'values' not in crd_schema:
				new = new.assign_coords({di: new.coords[di].astype(crd_dtype)})
			else:
				new_crd_vals = crd_schema['values']
				new = new.assign_coords({di: np.array(new_crd_vals).astype(crd_dtype)})

	# recast coordinate and dimension names
	rev_new_shape_spec = list(new_shape_spec)[::-1]
	rev_new_dims_spec = list(new_dims_spec)[::-1]
	for i, (si, di) in enumerate(zip(rev_new_shape_spec, rev_new_dims_spec)):
		if si == '...':
			break
		# print('backward', si, di, old_dims[-(i + 1)])
		new = new.rename({old_dims[-(i + 1)]: di})
		if di in schema['coords']:
			crd_schema = schema['coords'][di]
			crd_dtype = crd_schema['dtype']
			if 'values' not in crd_schema:
				new = new.assign_coords({di: new.coords[di].astype(crd_dtype)})
			else:
				new_crd_vals = crd_schema['values']
				new = new.assign_coords({di: np.array(new_crd_vals).astype(crd_dtype)})

	new = new.astype(new_dtype)

	validate(new, schema=schema)
	return new


def zeros(
	schema: dict | Mapping,
	like: xr.DataArray = None,
	drop_mismatched_dims: bool = True,
	attrs: dict = None,
	**with_coords,
):
	shape = []
	dims = []
	coords = {}
	extras_appended = False
	like_schema_dims = []
	# try to get extra dimensions
	# frmo the old schema
	try:
		like_schema_dims = json.loads(like.attrs[SCHEMA_ATTRS_KEY])['dims']
	except (KeyError, AttributeError):
		pass

	if attrs is None:
		attrs = {}

	# try to copy metadata
	new_attrs = {}
	try:
		new_attrs = copy.deepcopy(like.attrs)
		if SCHEMA_ATTRS_KEY in new_attrs:
			new_attrs.pop(SCHEMA_ATTRS_KEY)
		if ANNOTATION_ATTRS_KEY in new_attrs:
			new_attrs.pop(ANNOTATION_ATTRS_KEY)
		new_attrs.update(attrs)
	except (KeyError, AttributeError):
		new_attrs.update(attrs)

	for s, d in zip(schema['shape'], schema['dims']):
		# insert any extra dimensions
		if s == '...' and not extras_appended:
			if like is not None:
				for d in like.dims:
					# matching dims go in location of new schema
					# so only look for extra dims here
					if d not in schema['dims']:
						# if its not a required dimension OR requested
						# not to drop old required dimensions
						if d not in like_schema_dims or not drop_mismatched_dims:
							shape.append(len(like.coords[d]))
							dims.append(d)
							coords[d] = like.coords[d].copy()
			extras_appended = True

		# coordinate is required and has predefined labels
		elif type(s) is int:
			crd_schema = schema['coords'][d]
			shape.append(s)
			dims.append(d)
			coords[d] = np.array(
				schema['coords'][d]['values'], dtype=crd_schema['dtype']
			)
		# coordinate is required and doesn't have predefined labels
		else:
			crd_schema = schema['coords'][d]
			dims.append(d)
			if like is not None:
				shape.append(len(like.coords[d]))
				coords[d] = like.coords[d].astype(crd_schema['dtype'])

			else:
				try:
					shape.append(len(with_coords[d]))
				except KeyError as e:
					msg = f'Missing required coordinates {d} with schema: \n{json.dumps(crd_schema, indent=True)}'
					raise ValueError(msg) from e
				coords[d] = np.array(with_coords[d]).astype(crd_schema['dtype'])

	new = np.zeros(shape, dtype=schema['dtype'])
	new = xr.DataArray(new, dims=dims, coords=coords)
	new.attrs = new_attrs
	validate(new, schema=schema, attach_schema=True)
	return new


def load(
	path: Path | str,
	*load_args,
	registry: ArrSchemaRegistry,
	schema: dict,
	loader_type: str = None,
	validate_schema: bool = True,
	verbose=False,
	**load_kwargs,
) -> object:
	"""
	Save a dataset using an array schema registry.

	Parameters
	----------
	path : Path | str
		_description_
	registry : ArrSchemaRegistry
		_description_
	schema : str, optional
		_description_, by default None
	loader_type : str, optional
		_description_, by default None
	validate_schema : bool, optional
		_description_, by default True
	verbose : bool, optional
		_description_, by default False

	Returns
	-------
	object
		_description_
	"""
	# lookup by the schema if provided
	extension = ''.join(Path(path).suffixes)
	loader, schema = registry.import_loader(
		schema_uid=schema['uid'],
		extension=extension,
		loader_type=loader_type,
		verbose=verbose,
	)
	read = loader(path, *load_args, **load_kwargs)
	if validate_schema:
		validate(read, schema=schema)
	return read


def convert(
	input: AnnotatedArrayLike,
	registry: ArrSchemaRegistry,
	output_schema: dict,
	input_schema: dict = None,
) -> Any:
	"""
	Convert an input data to a new schema.

	Convert functions take in a single input and have
	a single output (SISO).

	Parameters
	----------
	input : Any
	    The input array with a known schema.
	registry : ArrSchemaRegistry
	    The registry containing the converters.
	output_schema_name : str, optional
	    Schema name of the output (or the uid). by default None
	output_schema_uid : str, optional
	    Schema uid of the output (or the name), by default None
	input_schema_name : str, optional
	    Name of the input schema (if it can't be inferred), by default None
	input_schema_uid : str, optional
	    Schema uid of the input (if it can't be inferred), by default None

	Returns
	-------
	Any
	    Converted input to the output schema.

	Raises
	------
	AttributeError
	    If the input schema can't be inferred.
	"""
	if input_schema is None:
		try:
			input_attrs = input.attrs
		except AttributeError as e:
			msg = "Couldn't infer input schema (no attrs). Specify input schema on convert call."
			raise AttributeError(msg) from e
		try:
			input_schema_uid = json.loads(input_attrs[SCHEMA_ATTRS_KEY])['uid']
		except KeyError:
			msg = f"Couldn't infer input schema ({SCHEMA_ATTRS_KEY} isn't present on attrs). Specify input schema on convert call."
	else:
		input_schema_uid = input_schema['uid']

	converter_fun = registry.import_converter(
		input_schema_uid=input_schema_uid,
		output_schema_uid=output_schema['uid'],
	)
	return converter_fun(input)


def validate(
	arr: AnnotatedArrayLike,
	*,
	schema: Mapping,
	attach_schema: bool = True,
):
	"""
	Check if a DataArray conforms to a particular schema.

	Parameters
	----------
	arr : xarray.DataArray
	    DataArray object to validate
	schema : Mapping, optional
	    A schema dictionary to validate against, by default None
	attach_schema : bool, optional
	    If True, the schema is dumped into a string and
	    attatched to the attrs of the input data array.
	    The default is True.

	"""
	# get the schema
	# compare dimensions to the actual shape,
	# map symbolic dimensions to actual dimensions
	sym_map = {}
	for d, s in zip(schema['dims'], schema['shape']):
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
	if schema['dtype'] != atomic_str(arr.dtype):
		raise ValidationError(
			f'dtype {arr.dtype} doesnt match expected {schema["dtype"]} for : \n {arr}'
		)

	# check that the coordinate dimensions are acceptable
	for cname, coord in schema['coords'].items():
		# check values match for static coordinates
		if 'values' in coord:
			if not (arr.coords[cname] == coord['values']).all():
				raise ValidationError(
					f'not all coordinates in schema match. \n Got: {arr.coords[cname]} \n Expected {coord["values"]} for : \n {arr}'
				)

		# check that dtypes match for coordinates
		if coord['dtype'] != atomic_str(arr.coords[cname].dtype):
			raise ValidationError(
				f'dtype {arr.coords[cname].dtype} doesnt match expected {coord["dtype"]} for : \n {arr}'
			)

	# validate the metadata schema
	try:
		jsonschema.validate(dict(arr.attrs), schema=schema['attrs_schema'])
	except jsonschema.exceptions.ValidationError as e:
		msg = 'Failed to validate attrs schema : \n ' + str(e)
		raise ValidationError(msg) from e

	if attach_schema:
		arr.attrs[SCHEMA_ATTRS_KEY] = json.dumps(schema)


def atomic_str(dtype: str):
	"""
	Validate and return a string according to an atomic type.
	"""
	return _np.dtype(dtype)


def from_dict(d: Mapping | dict) -> dict:
	"""
	Generate a schema from a dictionary.

	Parameters
	----------
	d : dict | Mapping
		Schema object
	"""
	return arrschema(**d)


# %% defines what is in a dataformat
def arrschema(
	name: str,
	shape: tuple[str | int],
	dims: tuple[str],
	dtype: str,
	units: Mapping | str = None,
	coords: Mapping = None,
	uid: str = None,
	attrs_schema: Mapping = None,
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
		raise Exception('shape and dims length must match')
	shape = list(copy.copy(shape))
	dims = list(copy.copy(dims))
	# replace ellipses with strings to make it
	# more compatable with json
	shape_lookup = {}
	for spec_tuple in (shape, dims):
		for i, s in enumerate(spec_tuple):
			if s == ...:
				spec_tuple[i] = '...'
	# check that shape and dimensions
	# make sense and agree with eachother
	for i, (s, d) in enumerate(zip(shape, dims)):
		shape_lookup[d] = s
		# can only used valid shape specifications
		if not _allowed_shape_spec(s):
			raise Exception(f'Shape spec {s} must be a letter or an ellipses')
		if not isinstance(d, str) and d != ...:
			raise ValueError('Dimensions names must be strings or ...')

		# if one has an ..., the other must too
		s_unspecd = s in UNSPECIFIED_SPECS
		d_unspecd = d in UNSPECIFIED_SPECS
		if s_unspecd ^ d_unspecd:
			raise ValueError(
				'unspecified shapes specs (...) must have unspecified dimension names. Specified shapes must have specified dimension names.'
			)

	# check that the dtype str is valid
	dtype = str(np.dtype(dtype))

	# check that the coordinates are valid
	out_coords = {}
	if coords is not None:
		for c, crd in coords.items():
			try:
				cunits = coords[c]['units']
			except KeyError:
				cunits = None
			if cunits is not None and not isinstance(cunits, str):
				for cui in cunits:
					if not isinstance(crd['units'], str):
						raise ValueError(
							'Coordinates can have only a single unit (i.e. must be a string.)'
						)

			cshape = (shape_lookup[c],)
			cdims = (c,)
			cschema = arrschema(
				name=d, dims=cdims, shape=cshape, dtype=crd['dtype'], units=cunits
			)
			if 'values' in crd:
				if len(crd['values']) != shape_lookup[c]:
					raise ValueError(
						f'Coord values {c} length do not match dimension spec {shape_lookup[c]}'
					)
				cschema['values'] = crd['values']

			# require that the coordinat dimension shapes match
			out_coords[c] = cschema
	else:
		coords = {}

	# check that the units mapping is valid
	if isinstance(units, str):
		units = units
	elif units is not None:
		if len(units) > 1:
			raise Exception('Only 1 unit dimension is allowed')
		if len(units) == 0:
			raise ValueError(f'Must supply exactly 1 unit dimension {units}')
		udims = list(units.keys())
		# replace any ... with strings
		# for json compatability
		for ud in udims:
			if ud in UNSPECIFIED_SPECS:
				units['...'] = units[ud]
				units.pop(ud)
		# check that its valied
		for ud in units:
			if ud not in dims:
				raise ValueError(f'Unit dimension {ud} not in dims.')
			# if its a string, then its a global unit
			if isinstance(units[ud], str):
				pass
			else:
				# if ud is defined in coords,
				# make sure the lengths

				if ud not in coords:
					raise ValueError(
						'If providing multiple units for a unit dimensin, that dimension must have defined coordinates.'
					)

				if len(units[ud]) != len(coords[ud]['values']):
					raise ValueError(
						f'Unit dimension values {ud} array must match length of the matching coordinates.'
					)

	if attrs_schema is None:
		attrs_schema = {}

	out = {
		'uid': str(uid),
		'name': name,
		'shape': tuple(shape),
		'dims': tuple(dims),
		'dtype': dtype,
		'units': units,
		'coords': out_coords,
		'attrs_schema': attrs_schema,
	}

	if units is None:
		out.pop('units')

	if uid is None:
		out.pop('uid')
		hash = dict_hash.sha256(out)
		out['uid'] = hash

	return out
