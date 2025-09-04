import pandas as pd
import xarray as xr
from pathlib import Path
from rmellipse.utils import load_object, save_object
import numpy as np
import h5py

def load_group_saveable(path, group: str = None, load_big_objects=True):
    with h5py.File(path, 'r') as f:
        use = f
        if group is not None:
            use = f[group]
        obj = load_object(use, load_big_objects=load_big_objects)
    return obj

def save_group_saveable(path, object, name: str, group: str = None,  write_mode = 'w'):
    with h5py.File(path, write_mode) as f:
        use = f
        if group is not None:
            use = f.require_group(group)
        save_object(use, name, object)

def load_csv_like_s2p_ri(path, comment="#"):
    d = pd.read_csv(path, delimiter="\t", header=None, comment=comment)
    d = d.to_numpy()
    index = d[:, 0]
    data = d[:, 1:]
    dims = ("frequency", "col")
    column_names = [
                "Re(S11)",
                "Im(S11)",
                "Re(S12)",
                "Im(S12)",
                "Re(S21)",
                "Im(S21)",
                "Re(S22)",
                "Im(S22)",
            ]
    coords = {
        "frequency": index.astype(float),
        "col": np.array(column_names,dtype = 'U8')
    }
    meta = {}
    header = get_header(path, comment)
    meta["header"] = header
    meta['frequency_units'] = 'GHz'
    arr = xr.DataArray(
        data,
        coords=coords,
        dims=dims,
    )
    arr.attrs.update(meta)
    return arr


def get_header(path, comment):
    """
    Returns the header of a file.

    Parameters
    ----------
    path : str
        File path.

    comment : str
        The comment character for the header.

    Returns
    -------
    list
        list of each line of the header.

    """
    with open(path) as file:
        lines = [line.rstrip() for line in file if line[0] == comment]

    return lines
