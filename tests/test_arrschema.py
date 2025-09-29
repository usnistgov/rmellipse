import rmellipse.arrschema as arrschema
import rmellipse.arrschema.xr as arrschema_xr
import xarray as xr
import numpy as np
from rmellipse.uobjects import RMEMeas
import pytest
from pathlib import Path

LOCALS = Path(__file__).parents[0]
ARRAY_SAMPLES = LOCALS / 'arrsamples'
REGISTRY = arrschema.ArrSchemaRegistry()
MUTABLE = LOCALS / 'mutable'

empty_float = arrschema.arrschema('empty', (...,), (...,), float)
# basic float
with pytest.raises(Exception):
	mismatched_dims = arrschema.arrschema('empty', (..., 'N'), (...,), float)

s2p_ri = arrschema.arrschema(
	name='s2p_ri',
	shape=(..., 'N', 8),
	dims=(..., 'frequency', 'col'),
	dtype=float,
	coords={
		'frequency': {
			'units': 'GHz',
			'dtype': float,
		},
		'col': {
			'values': [
				'Re(S11)',
				'Im(S11)',
				'Re(S12)',
				'Im(S12)',
				'Re(S21)',
				'Im(S21)',
				'Re(S22)',
				'Im(S22)',
			],
			'dtype': 'U8',
		},
	},
	attrs_schema={
		'type': 'object',
		'properties': {
			'frequency_units': {
				'enum': ['GHz', 'Hz'],
			}
		},
		'required': ['frequency_units'],
	},
)

zeros = arrschema.arrschema(name='zeros', shape=(...,), dims=(...,), dtype=float)

REGISTRY.add_schema(s2p_ri)
REGISTRY.add_schema(zeros)

REGISTRY.add_loader(
	'rmellipse.arrschema.examples:load_csv_like_s2p_ri',
	'.s2p',
	loader_type='csv',
	schema_name='s2p_ri',
)

REGISTRY.add_loader(
	'rmellipse.arrschema.examples:load_group_saveable',
	['.h5', '.hdf5'],
	loader_type='group_saveable',
	schema_name='s2p_ri',
)

REGISTRY.add_saver(
	'rmellipse.arrschema.examples:save_group_saveable',
	['.h5', '.hdf5'],
	saver_type='group_saveable',
	schema_name='s2p_ri',
)


REGISTRY.add_converter(
	'rmellipse.arrschema.examples:convert_zeros_to_s2p_ri',
	input_schema_name='s2p_ri',
	output_schema_uid=zeros['uid'],
)

# some more stuff
zeros_complex = arrschema.arrschema(
	name='zeros_complex', shape=(...,), dims=(...,), dtype=complex
)
float_2by2 = arrschema.arrschema(
	name='float_2by2',
	shape=(3, ..., 2, 2),
	dims=('d0', ..., 'd1', 'd2'),
	dtype=float,
	coords={
		'd0': {'values': [0, 1, 2], 'dtype': float},
		'd1': {'values': [0, 1], 'dtype': int},
		'd2': {'dtype': int},
	},
)
REGISTRY.add_schema(zeros_complex)
REGISTRY.add_schema(float_2by2)


def test_with_RMEMeas():
	path = ARRAY_SAMPLES / 'load.s2p'
	data = arrschema.load(
		ARRAY_SAMPLES / 'load.s2p',
		schema_name='s2p_ri',
		loader_type='csv',
		verbose=True,
		registry=REGISTRY,
	)
	data = RMEMeas.from_nom('mysample', data)

	arrschema.validate(data, schema=s2p_ri, registry=REGISTRY)


def test_load_and_save():
	print('trying to load s2p_ri')
	path = ARRAY_SAMPLES / 'load.s2p'
	data = arrschema.load(
		ARRAY_SAMPLES / 'load.s2p',
		schema_name='s2p_ri',
		loader_type='csv',
		verbose=True,
		registry=REGISTRY,
	)

	data = arrschema.load(ARRAY_SAMPLES / 'load.s2p', verbose=True, registry=REGISTRY)

	h5_path = MUTABLE / 'loadarr.h5'
	group = 'sample'

	# this should fail because this
	# schema requires Hz or GHz
	# on the frequency units
	with pytest.raises(Exception):
		data.attrs['frequency_units'] = 'MHz'
		arrschema.save(h5_path, data, 'sample', registry=REGISTRY, schema_name='s2p_ri')

	data.attrs['frequency_units'] = 'GHz'

	arrschema.save(h5_path, data, 'sample', registry=REGISTRY)

	data = arrschema.load(
		h5_path, group='sample', schema_name='s2p_ri', registry=REGISTRY
	)

	return data


def test_convert():
	path = ARRAY_SAMPLES / 'load.s2p'
	data = arrschema.load(
		ARRAY_SAMPLES / 'load.s2p',
		schema_name='s2p_ri',
		loader_type='csv',
		verbose=True,
		registry=REGISTRY,
	)
	arrschema.validate(data, registry=REGISTRY, schema_name='s2p_ri')
	new = arrschema.convert(data, registry=REGISTRY, output_schema_name='zeros')
	print(new)


def test_as_xr_schema():
	# schema with 2by2
	print('zeros arbitrary 3,2,2')
	output = arrschema_xr.as_schema(
		xr.DataArray(np.zeros((3, 2, 2), dtype='f4')),
		0,
		schema=float_2by2,
	)
	print(output)

	# basic arbitrary array with changing types
	print('zeros arbitrary')
	output = arrschema_xr.as_schema(
		xr.DataArray(np.zeros((4, 4), dtype='f8')),
		0,
		schema=zeros_complex,
	)
	print(output)


if __name__ == '__main__':
	data = test_load_and_save()

	test_with_RMEMeas()
	print(data)
	test_convert()
	print('ZEROS LIKE \n ============')
	test_as_xr_schema()
	import json

	print(json.dumps(zeros_complex, indent=True))
