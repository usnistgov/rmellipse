import rmellipse.arrschema as arrschema
import xarray as xr
import numpy as np
import pandas as pd

REGISTRY = arrschema.ArrayClassRegistry()

s2p_ri_schema = arrschema.ArraySchema(
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

s2p_c_schema = arrschema.ArraySchema(
    name='s2p_c',
    shape=(..., 'N', 4),
    dims=(..., 'frequency', 'col'),
    dtype=float,
    coords={
        'frequency': {
            'units': 'GHz',
            'dtype': float,
        },
        'col': {
            'values': [
                'S11',
                'S12',
                'S21',
                'S22',
            ],
            'dtype': 'U8',
        },
    },
)

RLGC_schema = arrschema.ArraySchema(
    name='RLGC',
    shape=(..., 'N', 4),
    dims=(..., 'frequency', 'col'),
    dtype=float,
    units={'col': ['Ohm/m', 'H/m', 'S/m', 'F/m']},
    coords={
        'frequency': {
            'units': 'GHz',
            'dtype': float,
        },
        'col': {
            'values': ['R', 'L', 'G', 'C'],
            'dtype': 'U8',
        },
    },
)


class S2P_RI(arrschema.AnnotatedArray):
    registry = REGISTRY
    schema = s2p_ri_schema


class S2P_C(arrschema.AnnotatedArray):
    registry = REGISTRY
    schema = s2p_c_schema


class RLGC(arrschema.AnnotatedArray):
    registry = REGISTRY
    schema = RLGC_schema


RLGC_read_cols = (
    'Freq [GHz]',
    'C(Sig,Sig) [pF]',
    'G(Sig,Sig) [sie]',
    'R(Sig,Sig) [ohm]',
    'L(Sig,Sig) [nH]',
)

RLGC_cols = ['R', 'L', 'G', 'C']


def load_RLGC(filename):
    data = pd.read_csv(filename, names=RLGC_read_cols, header=1)
    frequency = np.array(data['Freq [GHz]']) * 1e9
    R = np.array(data['R(Sig,Sig) [ohm]'])
    L = np.array(data['L(Sig,Sig) [nH]']) * 1e-9
    C = np.array(data['C(Sig,Sig) [pF]']) * 1e-12
    G = np.array(data['G(Sig,Sig) [sie]'])

    shape = (len(frequency), 4)
    out_data = np.zeros(shape=shape, dtype=float)
    out_data[:, 0] = R
    out_data[:, 1] = L
    out_data[:, 2] = G
    out_data[:, 3] = C
    coords = {'frequency': frequency, 'col': RLGC_cols}

    out_array = xr.DataArray(data=out_data, coords=coords)
    return out_array


REGISTRY.add_loader(
    'model:load_RLGC',
    '.csv',
    loader_type='csv',
    schema=RLGC_schema,
)


def initialize_output(prototype, dtype, ignore_dims=[], add_coords={}):
    out_dims = []
    out_shape = []
    out_coords = {}
    for dim in prototype.dims:
        if dim not in ignore_dims:
            out_dims.append(dim)
            coord = prototype.coords[dim]
            out_shape.append(len(coord))
            out_coords[dim] = coord

    for item in add_coords.items():
        dim, coord = item
        out_coords[dim] = coord
        out_shape.append(len(coord))

    out_data = np.zeros(shape=out_shape, dtype=dtype)
    out_array = xr.DataArray(data=out_data, coords=out_coords)
    return out_array


def tune_CG(RLGC_data: RLGC, voltage: float, sensitivity: float) -> RLGC:
    """
    Get RLGC for a transmission line where C, G changes in repsonse to voltage.

    C, G change together in a way that should preserve causality.

    Parameters
    ----------
    RLGC_data : RLGC
        Distributed circuit parameters at 0 voltage.

    voltage : float
        DESCRIPTION.

    sensitivity : float
        Sensitivity in fractional change / Volt.

    Returns
    -------
    RLGC
        RLCG array with perturbed C, G

    """
    R = RLGC_data.sel(col='R')  # real number. units ohms/m
    L = RLGC_data.sel(col='L')  # real number. units H/m
    C = RLGC_data.sel(col='C')  # real number. units F/m
    G = RLGC_data.sel(col='G')  # real number. units S/m

    out_data = initialize_output(RLGC_data, dtype=float)

    min_frequency = np.min(RLGC_data.coords['frequency'])
    C0 = C.sel(frequency=min_frequency)
    print('*' * 10)
    print(voltage)
    print(sensitivity)
    new_C = C0 + (C - C0) * (1.0 + voltage * sensitivity)
    new_G = G * (1.0 + voltage * sensitivity)

    out_data.loc[{'col': 'R'}] = R
    out_data.loc[{'col': 'L'}] = L
    out_data.loc[{'col': 'G'}] = new_G
    out_data.loc[{'col': 'C'}] = new_C
    return out_data


def get_S_parameters(RLGC_data: RLGC, length: float, Zr: float) -> S2P_C:
    """
    Get S-parameters from an RLCG transmission line model.

    Parameters
    ----------
    RLCG_data : RLGC
        Distributed circuit parameters
        (resitance, inductance, capacitace, conductance)

    length : float
        Length of the line in meters.

    Zr : float
        Reference impeadance in Ohms.

    Returns
    -------
    S2P_C
        S-parameters of transmission line.

    """
    R = RLGC_data.sel(col='R')  # real number. units ohms/m
    L = RLGC_data.sel(col='L')  # real number. units H/m
    C = RLGC_data.sel(col='C')  # real number. units F/m
    G = RLGC_data.sel(col='G')  # real number. units S/m
    frequency = RLGC_data.coords['frequency']

    omega = frequency * 2.0 * np.pi  # real number. units rad/s

    Y = G + 1.0j * omega * C
    Z = np.sqrt((R + 1.0j * omega * L) / Y)
    gamma = np.sqrt((R + 1.0j * omega * L) * Y)

    ref = (Z - Zr) / (Z + Zr)
    D = (ref**2) * np.exp(-gamma * length) - np.exp(gamma * length)

    S11 = (ref / D) * (np.exp(-gamma * length) - np.exp(gamma * length))
    S21 = (1.0 / D) * (ref**2 - 1)

    out_data = initialize_output(
        RLGC_data,
        dtype=complex,
        ignore_dims=['col'],
        add_coords={'col': ['S11', 'S21', 'S12', 'S22']},
    )

    out_data.loc[{'col': 'S11'}] = S11
    out_data.loc[{'col': 'S21'}] = S21
    out_data.loc[{'col': 'S12'}] = S21
    out_data.loc[{'col': 'S22'}] = S11

    return out_data
