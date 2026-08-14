"""
This module tests trying to load a serialized Annotated Array
from GroupSaveable where the AnnotatedArray class isn't available.
"""

import json
import pytest

from pathlib import Path

import xarray as xr
import numpy as np
import h5py

import rmellipse.arrschema as arrschema
from rmellipse.utils import load_object, save_object, MissingSchemaWarning
from pytest import raises

LOCALS = Path(__file__).parents[0]
TEST_FILES = LOCALS / 'const'


def test_load_without_schemaclass():
    # this file was made by test_arrscheme, so this file
    # doesn't have access to the annotated array class.
    # this hsould throw a warning that the annotated array class isn't
    # available, than return a normal DataArray with an AnnotatedArray that was
    with h5py.File(TEST_FILES / 'arrschema_groupsaveable.h5', 'r') as f:
        read = load_object(f['zeros'])
        assert isinstance(read, xr.DataArray)
        assert not isinstance(read, arrschema.AnnotatedArray)


if __name__ == '__main__':
    test_load_without_schemaclass()
