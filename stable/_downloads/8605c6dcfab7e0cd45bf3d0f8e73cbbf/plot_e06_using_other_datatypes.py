"""
Using Arbitrary Data Types
==========================

The RME was built to work best with complex and float data-types inside
xarray DataArrays. However, the propagation algorithms are agnostic
to what's inside the DataArray so the basic functionality of wrapping a function
in a propagator will work regardless of what data type you put in them.

Basic propagation will work with arbitrary data types. However,
quality of life features like common grid handling, grouping by
categories, combining repeated measurements, calculating standard uncertainty,
and computing confidence intervals may not work depending on the
data type you use. You may have to recreate that functionality manually, and we
strongly recommend testing your data type with the propagator to ensure it
functions as you expect.
"""

# %%
# Setup
# -----
# First lets create a custom class and use it to create a RMEMeas object.
import xarray as xr
from rmellipse.uobjects import RMEMeas
from rmellipse.propagators import RMEProp


class myclass:
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return 'my val is ' + str(self.val)

    def __repr__(self):
        return self.__str__()

    def __add__(self, o):
        newval = self.val + o.val
        return myclass(newval)


nominal = xr.DataArray(myclass(2))
perturbed = xr.DataArray(myclass(2) + myclass(0.1))

meas = RMEMeas.from_nom('test', nominal)
meas.add_umech('sample_meas', perturbed)
print(meas)


# %%
# Propagation
# -----------
# Because our custom class can be added together, we can use a vectorized
# propagator with them.
myprop = RMEProp(sensitivity=True, vectorize=True)


@myprop.propagate
def add(x, y):
    return x + y


print(add(meas, meas))

# %%%
# Sometimes you may need to take the object out of the xarray so you can
# operate on it or pass it through other functions.
# In that case you just need to put it
# back into a DataArray with the parameter locations before you output it.


@myprop.propagate
def add2(x, y):
    # pre-allocate an output
    out = xr.zeros_like(x)
    # take the custom classes out of the xarray object and operate on them
    new_vals = x.values + y.values
    # put them back in the DataArray we pre-allocated
    out.values = new_vals
    return out


new_meas = add2(meas, meas)
print(new_meas)

# %%%
# Next we can convert our data type to a float so we can
# use other built in features, like calculating standard uncertainty.


@myprop.propagate
def to_float(x):
    out = xr.zeros_like(x)
    out.values = [xi.val for xi in x.values]
    return out


new_meas = to_float(new_meas)
lin_unc, mc_un = new_meas.stdunc()
print(new_meas)
print(lin_unc)
