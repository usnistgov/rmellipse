"""This module contains objects that represent data with uncertainties.

Data with uncertainties correspond to propagators in
``rmellipse.propagators``
"""

from ._uobjs import UObj
from ._rmemeas import RMEMeas, RMEMeasFormatError

try:
    from ._rmemodel import RMEModel, RMEParameter, Norm, Uniform
except Exception as e:
    print(f'Failed to import RMEModel, RMEParameter, Norm, Uniform:  {e}')
    print(
        'Follow steps in rmellipse README to compile the .dll (windows) or .so (linux) if you want to use models.'
    )
