# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:36:50 2025

@author: dcg2
"""

import numpy as np
import xarray as xr
import warnings
from rmellipse.uobjects._rmemeas import RMEMeas, MC_DIM_NAME


def from_dist(
    name: str,
    nom: float,
    std: float,
    dist='gaussian',
    mechanism_name: str = None,
    samples: float = 100,
    categories: dict[str] = {'Type': 'B'},
    use_sample_mean: bool = False,
) -> 'RMEMeas':
    """
    Generate a RMEMeas object from a probability distribution.

    Will be deprecated in V0.5. Please use RMEmodel instead.

    Parameters
    ----------
    name : str
        Name of the object.
    nom : float
        Expected value of the distribution.
    std : float
        Standard deviation of the distribution.
    dist : str, {'gaussian', 'normal', 'uniform', 'rectangular'}
        Name of the distribution to use. Supports 'gaussian' or 'uniform'.
        The default is 'gaussian'.
    mechanism_name : str, optional
        What to name the linear uncertainty mechanisms associated with the
        distribution. The default is None.
    samples : float, optional
        How many monte-carlo samples to draw from. The default is 100.
    categories : dict[str], optional
        What to categorize the linear uncertainty mechanism as
        Should be {category:value} pairs. The default is {'Type':'B'}.
    use_sample_mean_std : bool, optional
        If true, uses the mean and standard deviation of random samples drawn
        from the defined distribution for the linear sensitivity analysis and
        as the nominal and standard uncertainty values. If False, uses the provided nominal and
        standard deviation of the distribution. The default is True.

    Raises
    ------
    Exception
        Unsupported distribution.

    Returns
    -------
    'RMEMeas'
        RMEMeas based on defined distribution.

    """

    # try:
    #     dummy = nom[0]
    #     std = nom[0]
    #     # raise Exception('Function doesnt currently support arrays as inputs.')
    # except TypeError:
    #      pass
    nom = np.array(nom)
    std = np.array(std)
    if mechanism_name is None:
        mechanism_name = name + '_' + dist

    # make the covariance
    supported = ['gaussian', 'normal', 'uniform', 'rectangular']
    if dist not in supported:
        raise ValueError('Distribution not supported.')

    def choose_mean_nom(vals, nom, std):
        if use_sample_mean:
            nom = np.mean(vals, axis=0)
            std = np.std(vals, axis=0, ddof=1)
        return nom, std

    # make the montecarlo distributions
    if dist == 'gaussian' or dist == 'normal':

        def f(n, s):
            return np.random.normal(loc=n, scale=s)

        f = np.vectorize(f)
        vals = np.array([f(nom, std) for i in range(samples)])
        nom, std = choose_mean_nom(vals, nom, std)
        dims = list(vals.shape)
        dims[0] = MC_DIM_NAME
        mc = xr.DataArray(
            data=vals, dims=dims, coords={MC_DIM_NAME: np.arange(samples)}
        )

    if dist == 'uniform' or dist == 'rectangular':
        diff = np.sqrt(std**2 * 12) / 2
        low = nom - diff
        high = nom + diff

        def f(h, l):
            return np.random.uniform(low=l, high=h)

        f = np.vectorize(f)
        vals = np.array([f(high, low) for i in range(samples)])
        nom, std = choose_mean_nom(vals, nom, std)
        dims = list(vals.shape)
        dims[0] = MC_DIM_NAME
        mc = xr.DataArray(
            data=vals, dims=dims, coords={MC_DIM_NAME: np.arange(samples)}
        )

    cov_dims = [cd for cd in dims]
    cov_dims[0] = 'umech_id'
    cov = xr.DataArray(
        np.array([nom, nom + std]),
        dims=cov_dims,
        coords={'umech_id': ['nominal', mechanism_name]},
    )

    covcats = np.full((1, len(categories)), '').astype('T')
    for i, (c, v) in enumerate(categories.items()):
        covcats[:, i] = str(v)
    coords = {'umech_id': [mechanism_name], 'categories': list(categories.keys())}
    dims = ('umech_id', 'categories')
    covcats = xr.DataArray(covcats, dims=dims, coords=coords)
    # {name:categories}

    return RMEMeas(name, cov, mc, covcats=covcats)


def make_example_meas(
    nom_shape=(4, 2, 2),
    cov_cats={'Type': 'A', 'Origin': 'Pytest'},
    dtype_nom: object = float,
    rand_coords: bool = False,
    rand_nom: bool = False,
    lin_unc: float = 0.01,
    mc_unc: float = 0.01,
    N_mc_samples: int = 0,
    seed: int = None,
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
    nom = xr.DataArray(nom, dims=dims, coords=coords)

    meas = RMEMeas.from_nom(name='meas', nom=nom)

    meas.add_umech(
        name='mymechanisms',
        value=meas.nom + lin_unc,
        dof=np.inf,
        category={'Type': 'B', 'Origin': 'Pytest'},
        add_uid=True,
    )

    seed_if()
    for i in range(N_mc_samples):
        meas.add_mc_sample(
            meas.nom + np.random.normal(loc=0, scale=mc_unc, size=meas.nom.shape) - 0.5
        )
    meas._validate_conventions()
    return meas
