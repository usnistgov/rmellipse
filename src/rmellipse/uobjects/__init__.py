"""This module contains objects that represent data with uncertainties.

Data with uncertainties correspond to propagators in
``rmellipse.propagators``
"""

from ._rmemeas import (
    RMEMeas,
    CovarianceDataArray,
    CovarianceStrMetadata,
    MonteCarloDataArray,
    RMEMeasFormatError,
    RMEUncTuple,
)
