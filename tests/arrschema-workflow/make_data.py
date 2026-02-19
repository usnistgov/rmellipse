from rmellipse.utils import save_object
import rmellipse.arrschema as arrschema
import h5py
import click
import xarray as xr
import numpy as np
from pathlib import Path


registry = arrschema.ArrayClassRegistry()


s2p_ri = arrschema.ArraySchema(
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

expected_hash = 'bf664787135f2dffba08cf8316281f3bea95a5bf3cf1ec56088e316dbef243dd'

zeros = arrschema.ArraySchema(name='zeros', shape=(...,), dims=(...,), dtype=float)

registry.add_schema(s2p_ri)
registry.add_schema(zeros)


@click.command()
@click.argument('h5-file', type=Path)
@click.argument('blob-file', type=Path)
def cli(*args, **kwargs):
    main(*args, **kwargs)


def main(h5_file, blob_file):
    if s2p_ri['uid'] != expected_hash:
        raise Exception("Hash for schema doesn't match.")

    zeros = np.zeros((4, 4, 8))
    data = xr.DataArray(zeros)
    data.attrs['frequency_units'] = 'GHz'
    data = arrschema.as_schema(data, registry=registry, schema=s2p_ri)
    print('Made new data', data)
    arrschema.annotate(data)
    print(data.attrs)
    with h5py.File(h5_file, 'w') as f:
        save_object(f, 'my_data', data, verbose=True)

    pass


if __name__ == '__main__':
    # main('a','b')
    cli()
