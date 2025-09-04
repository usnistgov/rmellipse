# -*- coding: utf-8 -*-
"""
Reading/Writing RMEMeas Objects
===============================

This is a basic example demonstrating how to read and write a RMEMeas object.
"""

# %%
# Creating an Object
# ------------------
# First lets create an object that we want to save.

from rmellipse.uobjects import RMEMeas

import xarray as xr
import numpy as np

nom = xr.DataArray(np.zeros((10, 2)),
                   dims=('d1', 'd2'),
                   coords={'d1': np.arange(10),
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

# %%
# HDF5 Saving
# -----------
#
# Saving to HDF5 is the recommended format for saving RMEMeas objects.
# Compared to xml, it's much faster and preserves category and degree
# of freedom information about the object. In addition, the HDF5 format doesn't
# require you to define a data format with to_csv and from_csv functions.
#
# To save, open an HDF5 file or group and pass that to the
# :func:`rmellipse.uobjects.RMEMeas.to_h5` function. The RMEMeas object
# will be stored in the group you provide it under a group with it's name.
# The override argument tells the function to delete any pre-existing
# groups with that RMEMeas objects name then try to save it, to avoid
# permission errors.
#
# We recommend downloading the HDF5 viewer software. You can inspect your
# HDF5 object, and you will see that under the 'myval' group this creates, you
# will see the cov,mc,covdofs, and covcats attributes stored. Each one is an
# HDF5 representation of an xarray DataArray and together completely describe
# your RMEMeas object.
import h5py
with h5py.File('meas.hdf5', 'a') as f:
    meas.to_h5(f, override=True)
    print(f[meas.name])

# %%
# HDF5 Reading
# ------------
#
# You can open the HDF5 file you made in read mode,
# then pass in the group with it's name to
# :func:`rmellipse.uobjects.RMEMeas.from_h5` in order to read it.
with h5py.File('meas.hdf5', 'r') as f:
    meas = RMEMeas.from_h5(f['meas'])
    print(meas)


# %%
# XML Saving
# ----------
#
# XML format is a legacy format inherited from the original drag and
# drop menu Microwave Uncertainty Framework. It is included for backwards compatibility purposes, but it is
# not the recommended way to save if you can help it. Specifically be aware
# that saving data to XML you will lose degrees of freedom and category information
# about
# your linear uncertainty mechanisms. The default degrees of freedom for linear uncertainties
# is infinite, and the default category for uncertainties is type B. So
# if you save to XML then read it back in, all your mechanisms will default
# to both of those properties. To preserve categories and degrees of freedom
# you assigned, use HDF5. Additionally, it is much slower than the
# HDF5 format.
#
# To save to XML you need to define a read/write functions.
#
# First we turn our object into the s1p_ri format using the
# :func:`rmellipse.dataformats.as_format`
from rmellipse.propagators import RMEProp

def to_txt(data, path):
        np.savetxt(path, data.values, delimiter=',')

def from_txt(path):
    values = xr.DataArray(np.loadtxt(path, float, delimiter=','))
    return values

m1 = RMEMeas.from_nom('mymeas', xr.DataArray(np.zeros((2,2))))
m1.add_umech('my umech', m1.nom.copy()+0.1)

m1.to_xml(
    '.',
    to_txt,
    data_extension='.csv',
    header_extension='.meas'
)
# %%%
# Once we have our measurement in a format with read/write functions defined
# we can use the :func:`rmellipse.uobjects.RMEMeas.to_xml` and
# :func:`rmellipse.uobjects.RMEMeas.from_xml` functions.
#
# :func:`rmellipse.uobjects.RMEMeas.to_xml` is expecting a target directory
# to save the measurement
# in. It will save a header '.meas' file and a '..._Support'
# in that directory with the covariance and Monte Carlo data next to it. Both will be
# named using the RMEMeas object's name attribute. When a RMEMeas object is
# passed
# through a function, it's name attribute is changed to the name of that
# function.
# So, be sure to change the name attribute to something useful if the function
# name
# is not specific enough.

m2 = RMEMeas.from_xml(
    'mymeas.meas',
    from_csv=from_txt
)
assert (m2.cov.values == m1.cov.values).all()