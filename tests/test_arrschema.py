import rmellipse.arrschema as arrschema
import rmellipse.arrschema._test_collections as arrschema_tests
from rmellipse.uobjects import RMEMeas
from rmellipse.utils import save_object
import h5py
import pytest
import xarray as xr
from pathlib import Path

LOCALS = Path(__file__).parents[0]
ARRAY_SAMPLES = LOCALS / "arrsamples"
REGISTRY = arrschema.ArrSchemaRegistry()
MUTABLE = LOCALS / 'mutable'

empty_float = arrschema.arrschema("empty", (...,), (...,), float)
# basic float
with pytest.raises(Exception):
    mismatched_dims = arrschema.arrschema("empty", (..., "N"), (...,), float)

s2p_ri = arrschema.arrschema(
    name="s2p_ri",
    shape=(..., "N", 8),
    dims=(..., "frequency", "col"),
    dtype=float,
    coords={
        "frequency": {
            "units": "GHz",
            "dtype": float,
        },
        "col": {
            "values": [
                "Re(S11)",
                "Im(S11)",
                "Re(S12)",
                "Im(S12)",
                "Re(S21)",
                "Im(S21)",
                "Re(S22)",
                "Im(S22)",
            ],
            "dtype": "U8"
        },
    },
    attrs_schema={
        "type": "object",
        "properties": {
            "frequency_units": {
                "enum": ["GHz", "Hz"],
            }
        },
        "required": ["frequency_units"],
    },
)

REGISTRY.add_schema(s2p_ri)

REGISTRY.add_loader(
    "rmellipse.arrschema._test_collections:load_csv_like_s2p_ri",
    ".s2p",
    loader_type="csv",
    schema_name="s2p_ri",
)

REGISTRY.add_loader(
    "rmellipse.arrschema._test_collections:load_group_saveable",
    [".h5", ".hdf5"],
    loader_type="group_saveable",
    schema_name="s2p_ri"
)

REGISTRY.add_saver(
    "rmellipse.arrschema._test_collections:save_group_saveable",
    [".h5", ".hdf5"],
    saver_type="group_saveable",
    schema_name="s2p_ri",
)


def test_with_RMEMeas():
    path = ARRAY_SAMPLES / "load.s2p"
    data = arrschema.load(
        ARRAY_SAMPLES / "load.s2p",
        schema_name="s2p_ri",
        loader_type="csv",
        verbose=True,
        registry=REGISTRY,
    )
    data = RMEMeas.from_nom(
        'mysample',
        data
    )

    arrschema.validate(data, schema = s2p_ri, registry = REGISTRY)


def test_load_and_save():
    print("trying to load s2p_ri")
    path = ARRAY_SAMPLES / "load.s2p"
    data = arrschema.load(
        ARRAY_SAMPLES / "load.s2p",
        schema_name="s2p_ri",
        loader_type="csv",
        verbose=True,
        registry=REGISTRY,
    )

    data = arrschema.load(ARRAY_SAMPLES / "load.s2p", verbose=True, registry=REGISTRY)

    h5_path = MUTABLE / 'loadarr.h5'
    group = 'sample'

    # this should fail because this
    # schema requires Hz or GHz
    # on the frequency units
    with pytest.raises(Exception):
        data.attrs['frequency_units'] = 'MHz'
        arrschema.save(
            h5_path,
            data,
            'sample',
            registry=REGISTRY
        )

    data.attrs['frequency_units'] = 'GHz'

    arrschema.save(
        h5_path,
        data,
        'sample',
        registry=REGISTRY
    )

    data = arrschema.load(
        h5_path,
        group="sample",
        schema_name="s2p_ri",
        registry=REGISTRY
    )

    return data


if __name__ == "__main__":
    # test_build_registry()
    data = test_load_and_save()
    test_with_RMEMeas()
    arrschema.stdreg.show_schema()
    # arrschema.stdreg.show_loaders()
    print(data)
