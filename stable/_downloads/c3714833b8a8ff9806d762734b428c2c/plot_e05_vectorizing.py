"""
Using Non-Vectorized Propagation
================================

Here we go over how to use a propagator with the vectorized setting turned off.
"""

# %%
# Setup
# -----
# First lets create a RMEMeas object we can interact with and initialize
# a propagator.

from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
import xarray as xr
import numpy as np

nom = xr.DataArray(np.zeros((4, 2)),
                   dims=('d1', 'd2'),
                   coords={'d1': [0, 1, 2, 3],
                           'd2': np.arange(2)})

meas = RMEMeas.from_nom(name='meas', nom=nom)

meas.add_umech(
    name='my_mechanism',
    value=meas.nom + np.ones(meas.nom.shape) * 0.01,
    dof=np.inf,
    category={'Type': 'B', 'Origin': 'Data Sheet'}
)

for i in range(100):
    meas.add_mc_sample(meas.nom + np.random.rand(*meas.nom.shape) * 0.01)

myprop = RMEProp(
    sensitivity=True,
    montecarlo_sims=100,
    verbose=True)

# %%%
# Non-Vectorized Propagation
# --------------------------
# Sometimes it isn't possible to vectorize a function. In that case, similar
# to the non-vectorized version, the inputs and outputs of the function need
# to be a DataArray for the propagator to recognize the object as being
# a RMEMeas object. The first dimension of any inputs that were RMEMeas objects will
# be 'umech_id' and the coordinates of that dimension will be the
# names of the uncertainty mechanism. For non vectorized inputs, the
# propagator will loop over that first dimension, so the inputs will have a
# length 1 first dimension called 'umech_id'. The output should
# also have that dimension and coordinate.

myprop.settings['montecarlo_sims'] = 0
myprop.settings['sensitivity'] = True
myprop.settings['vectorize'] = False
myprop.settings['verbose'] = False


@myprop.propagate
def add(x, y):
    """Add 2 numbers."""
    output = x + y
    print(x.umech_id.values, output.umech_id.values)
    return output


added = add(meas, 2)

# %%%
# Vectorized vs. Unvectorized
# ---------------------------
# By default a RME propagator vectorizes all its functions. This is because
# vectorized computation is much faster and
# will save a lot of time, especially with large data sets and large numbers
# of uncertainty mechanisms. For comparison, here is the same function run
# once with the vectorize functionality and once without. Note the orders of
# magnitude increase in run time for the Monte Carlo propagation.


@myprop.propagate
def add2(x):
    return x + 2


myprop.settings['vectorize'] = True
add2(meas)

myprop.settings['vectorize'] = False
add2(meas)
