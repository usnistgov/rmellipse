# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:36:50 2025

@author: dcg2
"""

import numpy as np
import xarray as xr
from rmellipse.uobjects import RMEMeas


def make_example_meas(
        nom_shape=(4, 2, 2),
        cov_cats={'Type': 'A', 'Origin': 'Pytest'},
        dtype_nom: object = float,
        rand_coords: bool = False,
        rand_nom: bool = False,
        lin_unc: float = 0.01,
        mc_unc: float = 0.01,
        N_mc_samples: int  = 0,
        seed: int = None
):
    """
    Generate a generic RMEMeas object for testing.

    Parameters
    ----------
    nom_shape : TYPE, optional
        DESCRIPTION. The default is (4, 2, 2).
    cov_cats : TYPE, optional
        DESCRIPTION. The default is {'Type': 'A', 'Origin': 'Pytest'}.
    dtype_nom : TYPE, optional
        DESCRIPTION. The default is float.
    rand_coords : TYPE, optional
        Linspace if false. The default is False.
    rand_nom : TYPE, optional
        Zeros if false. The default is False.
    lin_unc: float,
        What to add to nominal value for sensitivity mechanisms (for all indexes.)
    lin_unc: float,
        Standard deviation of random generated numbers (for all indexes.) that
        are added to n o minal for mc_samples.
    N_mc_samples: int,
        How many mc samples. Default = 0
    seed: int
        Seed all calls to random if provided to make results consistent


    Returns
    -------
    RMEMeas
        MEasurement Object.

    """

    def seed_if():
        if seed:
            np.random.default_rng(seed)

    dims = ['d' + str(i + 1) for i in range(len(nom_shape))]

    def coord(s):
        if rand_coords:
            seed_if()
            a = np.random.random(s)
        else:
            a = np.linspace(0, 1, s).astype(float)
        a.sort()
        return a
    coords = {d: coord(s) for d, s in zip(dims, nom_shape)}
    if rand_nom:
        seed_if()
        nom = np.random.random(nom_shape)
    else:
        nom = np.zeros(nom_shape)
    nom = xr.DataArray(
        nom,
        dims=dims,
        coords=coords
    )

    meas = RMEMeas.from_nom(name='meas', nom=nom)

    meas.add_umech(
        name='mymechanisms',
        value=meas.nom + lin_unc,
        dof=np.inf,
        category={'Type': 'B', 'Origin': 'Pytest'},
        add_uid=True
    )

    seed_if()
    for i in range(N_mc_samples):
        meas.add_mc_sample(meas.nom + np.random.normal(loc = 0, scale = mc_unc, size = meas.nom.shape)-.5)
    meas._validate_conventions()
    return meas
