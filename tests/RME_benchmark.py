# -*- coding: utf-8 -*-
"""
Script to benchmark RME functions
"""

from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
from rmellipse.utils import save_object
import xarray as xr


def add(x, y):
    return x + y


test_file = 'mutable/deleteme.h5'
if __name__ == '__main__':
    # %%
    x = RMEMeas.from_nom('name', xr.DataArray([1], coords={'freq': [0]}))
    N = 50
    for i in range(N):
        x.add_umech('name', x.nom + 1, add_uid=True)
        x.add_mc_sample(x.nom + 0.001)
    for i in range(N):
        x.add_umech('nameasdad', x.nom + 1, add_uid=True)
        x.add_mc_sample(x.nom + 0.001)

    # adding many umech ids
    y = RMEMeas.from_nom(
        'name',
        xr.DataArray([[1, 2], [3, 4]], dims=('freq', 'd'), coords={'freq': [0, 1]}),
    )
