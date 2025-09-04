"""
Common Grid Handling
====================

A common problem is when multiple inputs into a workflow
share a common dimension like frequency, but those frequency grids aren't
aligned properly. This often results in having to manually
trim down arrays based on a known frequency list.

The RME offers an automatic common grid handling feature to do that for you.
It acts on any RMEMeas input into a propagating function.
"""

# %%
# Creating an Objects
# -------------------
# First lets create 2 RMEMeas objects with slightly different
# coordinates on dimension 'd1'.

from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp
import xarray as xr
import numpy as np


def make_measurement(d1_coords):
    """Make a sample measurement with length 4 coordinate set."""
    nom = xr.DataArray(np.zeros((4, 2)),
                       dims=('d1', 'd2'),
                       coords={'d1': d1_coords,
                               'd2': np.arange(2)})

    meas = RMEMeas.from_nom(name='meas', nom=nom)

    meas.add_umech(
        name='mymechanisms',
        value=meas.nom + np.ones(meas.nom.shape) * 0.01,
        dof=np.inf,
        category={'Type': 'B', 'Origin': 'Data Sheet'},
        add_uid=True
    )

    for i in range(100):
        meas.add_mc_sample(meas.nom + np.random.rand(*meas.nom.shape) * 0.01)
    return meas


m1 = make_measurement([0, 1, 2, 3])
m2 = make_measurement([0, 1.1, 2, 2.9])

# %%%
# Setting up the Propagator
# -------------------------
#
# Next, let's create a propagator. We are going to define the common grid
# as 'd1' and tell it to interpolate that grid to a set of common values. We
# tell it what those common values are with the common_coords argument. The
# verbose argument will print some information about the propagation as we go.
# See the :func:`rmellipse.propagators.RME.handle_common_grid` function for
# what methods are available for the propagator.
#
# When we wrap our add function in the propagator and print the d1 coordinate,
# we see that the d1 coordinate of x and y are now the same because the RME
# interpolated them down to supplied values. Note how the print statement
# happens twice, because a vectorized
# propagator calls the function on the`RMEMeas.cov`attribute then on the`RMEMeas.mc` attribute.

myprop = RMEProp(
    sensitivity=True,
    montecarlo_sims=100,
    common_grid='d1',
    handle_common_grid_method='interp_common',
    common_coords={'d1': [0, .5, 1.5, 2.5]},
    vectorize=True,
    verbose=True)


@myprop.propagate
def add(x, y):
    """Add two data sets."""
    print(x.d1.values, y.d1.values)
    return x + y


m3 = add(m1, m2)
