"""
Helper functions for arrschema specific to xarray objects.
"""

from rmellipse.arrschema import ArrSchemaRegistry, ValidationError, validate
from typing import Mapping
import xarray as xr
import json
import numpy as np
from rmellipse.arrschema._arrschema import atomic_str

__all__ = ['as_schema']


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
	unq_vals, unq_counts = np.unique(new_shape_spec, return_counts=True, sorted=False)
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
		print('forward', si, di)
		new = new.rename({old_dims[i]: di})
		if di in schema['coords']:
			crd_schema = schema['coords'][di]
			crd_dtype = crd_schema['dtype']
			if 'values' not in crd_schema:
				new = new.assign_coords({di: new.coords[di].astype(crd_dtype)})
			else:
				new_crd_vals = crd_schema['values']
				new = new.assign_coords({di: np.array(new_crd_vals).astype(crd_dtype)})

	rev_new_shape_spec = list(new_shape_spec)[::-1]
	rev_new_dims_spec = list(new_dims_spec)[::-1]
	for i, (si, di) in enumerate(zip(rev_new_shape_spec, rev_new_dims_spec)):
		if si == '...':
			break
		print('backward', si, di, old_dims[-(i + 1)])
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
