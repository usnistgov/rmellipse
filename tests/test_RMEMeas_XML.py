from rmellipse.uobjects import RMEMeas
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

def read_s2p(path):
    data = pd.read_csv(path, delimiter= '\t')
    return xr.DataArray(data)
header_path = Path(__file__).parents[0] / 'const' / 'Load.meas'

def test_umech_id():
    data = RMEMeas.from_xml(
        str(header_path),
        read_s2p,
        old_dir='dummy_dir',
        new_dir = str(header_path.parents[0].resolve()),
    )

    expected = ['TypeD_S_params_PCA_1', 'TypeD_S_params_PCA_2', 'TypeD_S_params_PCA_0']
    assert all([e == u for e,u in zip(expected,  data.umech_id)])

if __name__ == '__main__':
    test_umech_id()