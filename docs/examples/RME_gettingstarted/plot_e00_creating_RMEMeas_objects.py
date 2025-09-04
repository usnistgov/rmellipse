# -*- coding: utf-8 -*-
"""
RMEMeas Objects
===============

This is a basic example demonstrating how to create and use RMEMeas objects. In this
example we go over different ways to initialize one from scratch,
how to create linear uncertainty mechanisms, and how to create probability
distributions for Monte Carlo analysis. Indexing and interpolating are also
briefly discussed.
"""

# %%
# Initializing from a Nominal Data Set
# ------------------------------------
#
# The easiest - and most versatile - way to make a RMEMeas object from scratch is to
# use the :func:`rmellipse.uobjects.RMEMeas.from_nom` function, then
# add the uncertainty mechanisms and Monte Carlo data.
# With this function you just provide a nominal data set and a name for
# your object. The name is used when saving to formats like xml and hdf5.
from rmellipse.uobjects import RMEMeas
import xarray as xr
nom = xr.DataArray(
    [[1, 2, 3],[4, 5, 6]],
    dims=('d1','d2'),
    coords={
        'd1': [0.1, 0.2],
        'd2': ['a', 'b', 'c']
        }
)

meas = RMEMeas.from_nom(name='myval', nom=nom)
print(meas.nom)

# %% Linear Uncertainties
# -----------------------
#
# Then we can manually add linear uncertainty mechanisms, defining the degrees
# of freedom and categorizing the mechanisms as we go. The
# :func:`rmellipse.uobjects.RMEMeas.add_umech` function
# is expecting the value to be the same shape, dimensions, and coordinates of the
# nominal. It is easiest to use the `nom` attribute of your object, then perturb
# it 1 standard uncertainty.
# This example adds a linear uncertainty mechanism with infinite degrees of
# freedom, because
# it comes from an external source we trust, and perturbs each element in the
# nominal data set by 0.001.
# We also add a unique ID to the mechanism name, because we want to be sure
# that this mechanism will be fully independent in the event someone
# uses it with other uncertainty objects they might create in their own
# scripts. We also categorize it as Type B and categorize it's origin from
# a data sheet so we can group this mechanism down the road with mechanisms
# from similar sources.


import numpy as np

meas.add_umech(
    name='My Uncertainty Mechanism',
    value=meas.nom + np.ones(meas.nom.shape) * 0.01,
    dof=np.inf,
    category={'Type': 'B', 'Origin': 'Data Sheet'},
    add_uid=True
)
print(meas.stdunc(k=1).cov)

# %%%
# For linear uncertainties can calculate degrees of freedom
# associated with each measurand.
print(meas.dof())

# Calculate uncertainty bounds for a given expansion factor
# (nominal + k * standard uncertainty).
print(meas.uncbounds(k=1).cov)

# And estimate confidence intervals.
print(meas.confint(0.95))


# %%
# Monte Carlo Distributions
# -------------------------
#
# We also sample from a random distribution to add samples for use
# in Monte Carlo uncertainty analysis using
# :func:`rmellipse.uobjects.RMEMeas.add_mc_sample`. It is expecting a DataArray
# with the same dimensions and coordinates as the nominal.

for i in range(100):
    meas.add_mc_sample(meas.nom + np.random.normal(*meas.nom.shape) * 0.01)

print(meas.stdunc(k=1).mc)

# %%
# We can also calculate uncertainty bounds for a given expansion factor
# (nominal + k * standard uncertainty).
print(meas.uncbounds(k=1).cov)


# %%
# Indexing
# --------
#
# RMEMeas objects support xarray style indexing, which allows for label based
# indexing into arrays (like pandas) and integer based indexing into arrays
# (like numpy).The cov/mc attributes of the newly
# created RMEMeas object will be views on the original. Call RMEMeas.copy()
# to turn the view into it's own measurement.
#
# Importantly, when indexing into a RMEMeas array with xarray like functions (
# __getitem__ , loc, sel, and isel) you index into the RMEMeas object as if you
# were indexing into the nominal value. These functions, when called on the
# RMEMeas object, will always keep all the linear uncertainty mechanisms or
# Monte Carlo samples.
#
# Please refer to the xarray documentation for more details on how to use
# these indexing functions.

print('Inspect the nominal to see the dimensions and coordinates to index')
print(meas.nom.shape)
print(meas.nom.dims)
print(meas.nom.coords, '\n')


print('positional lookup by integer')
print(meas[0, 0], '\n')

print('positional lookup by label')
print(meas.loc[0.1, 'a'], '\n')

print('named lookup by integer')
print(meas.isel(d1=0, d2=0), '\n')

print('named lookup by label')
print(meas.sel(d1=0.1, d2='a'), '\n')

# %%
# Indexing Covariance and Monte Carlo Data
# ----------------------------------------
#
# If you wish to get a view into specific uncertainty mechanisms, or specific
# samples in the Monte Carlo distribution, then use the function `usel`.


# This example throws away the montecarlo samples and looks only at a single
# linear uncertainty mechanism.
mech = meas.usel(
    umech_id=meas.umech_id[0],
    mcsamples=[]
)

linunc, mcunc = mech.stdunc(k=1)
print(linunc)

# %%%
# We can also look a one or more of the Monte Carlo samples
# by throwing away the covariance data and just keeping one of the Monte Carlo
# samples.
sample = meas.usel(
    umech_id=[],
    mcsamples=[1]
)

print(sample.mc[1, ...])

# %%%
# While the xarray indexing methods index across uncertainty mechanisms and
# Monte Carlo samples, the usel method indexes into uncertainty mechanisms
# and Monte Carlo samples. In either case, the nominal values (the 'nominal'
# parameter location and 0th Monte Carlo sample) are always protected and
# always kept regardless of method.
#
# For example, using usel and giving empty lists for umech_id
# and mcsamples throws away all the uncertainty information, and just keeps
# the nominal. This effectively means it no longer has any associated
# uncertainties.

nominal_only = meas.usel(
    umech_id=[],
    mcsamples=[]
)

print(nominal_only.nom)
print(nominal_only.stdunc())

# %%
# Assigning RMEMeas
# -----------------
#
# Currently, RMEMeas objects indexing functions do not support item assignment.
# Values can be reassigned through propagation.

try:
    meas[0] = 1
except TypeError as e:
    print(e)

# %%
# Interpolating
# -------------
#
# RMEMeas supports interpolation by wrapping xarray's built in `interp` function.

print(meas.interp(d1 = [0.125, 0.15, 0.175]))
