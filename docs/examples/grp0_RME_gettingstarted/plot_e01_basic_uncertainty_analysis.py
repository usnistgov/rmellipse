# -*- coding: utf-8 -*-
"""
Basics of RME Propagation
=========================

This is a basic example demonstrating how to propagate RMEMeas objects
through functions.

This example generates a hypothetical voltage and current measurement each
with a Monte Carlo distribution and single linear uncertainty mechanism.

We define a propagator and a propagation function to calculate power and
perform the uncertainty analysis.
"""
# %%
# Importing packages
# ------------------

import rmellipse as rme
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# %%
# Creating a propagator
# ---------------------
#
# The first step is to define a propagator. This is the object that inspects
# functions for uncertainty objects, propagating them according to its settings.

myprop = rme.RMEProp(montecarlo_sims=1000, sensitivity=True)

# %%
# Creating RMEMeas objects
# ------------------------
#
# RMEMeas objects (Rocky Mountain Ellipse Measurements) carry
# along with them 2 key attributes `.cov` and `RMEMeas.mc`, along attributes for
# propagating metadata - `RMEMeas.covcats` and `RMEMeas.covdofs`.
#
# The`RMEMeas.cov`is an xarray.DataArray
# object that stores both the nominal value of the measurement, and copies of the nominal
# value perturbed by 1 standard deviation for each uncertainty mechanism. The
# perturbed copies are stored along the first dimension of the`RMEMeas.cov`attribute,
# which is always called 'umech_id'. Along that dimension
# the first index is always labeled 'nominal' and holds the nominal value, the remaining
# indexes store the names of the linear uncertainty mechanisms. These are labelled
# to be unique. Importantly, if two RMEMeas objects being propagated share an
# uncertainty mechanism with the same umech_id (i.e. they both have a perturbed copy of the nominal
# with the same index label), then they are assume to be fully correlated.
#
# The `RMEMeas.mc` attribute stores Monte Carlo data in an xarray.DataArray.
# Its first dimension is called `sample_id`; every row is a stochastic sample,
# with integer labels starting at zero. The independent nominal value remains in
# the first row of `RMEMeas.cov`.
#
# In this case, we will create a voltage and current measurement, both drawing
# from a gaussian distribution. Because we are creating a measurement using
# a float, the resulting`RMEMeas.cov`attribute will be shape (2,) (index 0 for the nominal
# value and index 1 representing the uncertainty associated with the Gaussian
# we sampled from). The `RMEMeas.mc` attribute will have shape (1000,), with
# every index representing a sample from the Gaussian distribution.


def make_measurement(d1_coords, std=0.1):
    """Make a sample measurement with length 4 coordinate set."""
    nom = xr.DataArray(
        np.ones((4, 2)),
        dims=('d1', 'd2'),
        coords={'d1': d1_coords, 'd2': np.arange(2)},
    )

    meas = rme.RMEMeas.from_nom(name='meas', nom=nom)

    meas.add_umech(
        name='mymechanisms',
        value=meas.nom + np.ones(meas.nom.shape) * std,
        dof=np.inf,
        category={'Type': 'B', 'Origin': 'Data Sheet'},
        add_uid=True,
    )

    for i in range(1000):
        meas.add_mc_sample(meas.nom + np.random.normal(loc=0, scale=std, size=None))
    return meas


V = make_measurement([0, 1, 2, 3])
I = make_measurement([0, 1.1, 2, 2.9])

print(V)
print('            dimensions of V.cov :', V.cov.dims, V.cov.shape)
print('             dimensions of V.mc :', V.mc.dims, V.mc.shape)
print('V linear uncertainty mechanisms :', V.umech_id)

# %%
# Propagating Through a Function
# ------------------------------
#
# Writing a function to be propagated requires a little knowledge about how
# the propagator works, but in trade enables you to take advantage of vectorized
# operations for significantly faster computation.
#
# First we define a function and wrap it the myprop.propagator decorator. This
# is equivalent to calling power = myprop.propagate(power) after defining
# the function power. This will tell the propagator to inspect the arguments of
# power when it is called and propagate any RMEMeas objects that are
# positional or keyword arguments.
#
# The propagator will call the function twice. Once for the linear sensitivity
# analysis where it passes in the`RMEMeas.cov`attribute, and once for the monte-carlo
# analysis where it passes in the`RMEMeas.mc` attribute.
#
# In the case of the sensitivity analysis, the propagator will align the
# linear uncertainty mechanisms of each RMEMeas object so that they can be
# used in math operations. In this example this means that the shape of the v
# and i arguments will have shape (3,) each, with the first axis aligning the
# uncertainty mechanisms that existed in both v and i. Uncertainty mechanisms
# that don't exist in a variable are filled with a copy of the nominal in the
# newly aligned dimension.
#
# For the Monte Carlo analysis, the propagator uses samples from the data stored
# in the `RMEMeas.mc` attribute. We defined 1000 Monte Carlo simulations in our
# propagator, so v and i will have shape (1000,) when passed through our power
# function.
#
# Note that while the propagator is design to encourage vectorization for
# efficiency, if that is not possible simply you can turn off vectorization
# with the vectorize setting. See the example on non-vectorized propagation
# for more details.
#
# The propagator will name your RMEMeas object using the name of the function
# you used to propagate it.


@myprop.propagate
def power(v, i):
    print(v.dims, v.shape, i.shape)
    return v * i


# when we call this, note that the power function is called twice, and the shapes
# of the v and i arguments are changed from the original definitions for the
# `RMEMeas.cov`attribute, now (3,) after the uncertainty mechanisms were aligned.
p = power(V, I)

# %%
# Evaluating Uncertainty
# ----------------------
#
# We can calculate the standard uncertainty of a RMEMeas object easily!

covunc, mcunc = p.stdunc(k=1)


print('sensitivity analysis uncertainty:', covunc)
print('monte-carlo uncertainty:', mcunc)

# %%%
# We can easily calculate the lower/upper uncertainty bounds for a given
# expansion factor on the uncertainty, degrees of freedom based on the linear
# sensitivity analysis, and confidence intervals based on the linear sensitivity
# analysis.

# k = 2 uncertainty bounds estimated by
# linear uncertainty analysis and monte carlo analysis
cov_k2, mc_k2 = p.uncbounds(k=2)

# degrees of freedom
# on the linear uncertaint analysis
dof = p.dof()

# 95% confidence interval
# of the linear uncertainty analysis
lower_ci, upper_ci = p.confint(0.95)
