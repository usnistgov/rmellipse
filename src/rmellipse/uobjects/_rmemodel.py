# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:40:24 2025

@author: amh7
"""

from numbers import Number
import abc as _abc
from abc import ABC, abstractmethod, ABCMeta
import rmellipse.arrschema as arrschema
from typing import Mapping, Any, Tuple, Callable
from functools import partial
from rmellipse.utils import GroupSaveable
import xarray as xr
import hashlib
import numpy as np
import uuid
from inspect import signature
from pathlib import Path
import abc

import os
import ctypes

# Windows (nt) or linux (posix)
_rng_stream_dir = Path(__file__).parent / 'rng_stream'
if os.name == 'nt':
    rngStreamLibFile = _rng_stream_dir / 'libRngStream.dll'
elif os.name == 'posix':
    rngStreamLibFile = _rng_stream_dir / 'libRngStream.so'
    if not rngStreamLibFile.exists():
        raise FileExistsError('cant find compiled libRngStream.dll in module.')

else:
    pass  # Throw error

# __all__ = ['RMEModel', 'RMEParameter']

# Load the library
rngStream = ctypes.cdll.LoadLibrary(os.path.abspath(rngStreamLibFile))

# Set up bindings
rngStream.RngStream_SetPackageSeed.restype = ctypes.c_int
rngStream.RngStream_SetPackageSeed.argtypes = (ctypes.POINTER(ctypes.c_ulong),)

rngStream.RngStream_CreateStream.restype = ctypes.c_void_p
rngStream.RngStream_CreateStream.argtypes = (ctypes.c_char_p,)

rngStream.RngStream_DeleteStream.restype = None
rngStream.RngStream_DeleteStream.argtypes = (ctypes.c_void_p,)

rngStream.RngStream_ResetStartStream.restype = None
rngStream.RngStream_ResetStartStream.argtypes = (ctypes.c_void_p,)

rngStream.RngStream_ResetStartSubstream.restype = None
rngStream.RngStream_ResetStartSubstream.argtypes = (ctypes.c_void_p,)

rngStream.RngStream_ResetNextSubstream.restype = None
rngStream.RngStream_ResetNextSubstream.argtypes = (ctypes.c_void_p,)

rngStream.RngStream_SetAntithetic.restype = None
rngStream.RngStream_SetAntithetic.argtypes = (ctypes.c_void_p, ctypes.c_int)

rngStream.RngStream_IncreasedPrecis.restype = None
rngStream.RngStream_IncreasedPrecis.argtypes = (ctypes.c_void_p, ctypes.c_int)

rngStream.RngStream_SetSeed.restype = None
rngStream.RngStream_SetSeed.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))

rngStream.RngStream_AdvanceState.restype = None
rngStream.RngStream_AdvanceState.argtypes = (
    ctypes.c_void_p,
    ctypes.c_long,
    ctypes.c_long,
)

rngStream.RngStream_GetState.restype = None
rngStream.RngStream_GetState.argtypes = (
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_ulong),
)


rngStream.RngStream_WriteState.restype = None
rngStream.RngStream_WriteState.argtypes = (ctypes.c_void_p,)

rngStream.RngStream_WriteStateFull.restype = None
rngStream.RngStream_WriteStateFull.argtypes = (ctypes.c_void_p,)


rngStream.RngStream_RandU01.restype = ctypes.c_double
rngStream.RngStream_RandU01.argtypes = (ctypes.c_void_p,)

rngStream.RngStream_RandInt.restype = ctypes.c_int
rngStream.RngStream_RandInt.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)


class PuaRandomGen(metaclass=abc.ABCMeta):
    def __init__(self, label: str):
        """
        Initialize a Pua random number generator.

        Parameters
        ----------
        label : str
            Used to seed random number generator. Should be unique.

        Returns
        -------
        None.

        """
        self.label = label

    def getNumbers(self, numVals: int):
        """
        Get random numbers.

        Parameters
        ----------
        numVals : int
            Number of values to fetch.

        Returns
        -------
        numbers : np.ndarray
            1-d array of random numbers.

        """
        numbers = np.zeros((numVals,), dtype=np.float64)
        for i in range(numVals):
            numbers[i] = self.getNumber()

        return numbers

    def __getstate__(self):
        """
        Get state of object.

        This code allows us to serialize the inherited class
        Basically, some of the data in the RandomGenLecSha256 is not serializable
        so we serialize only the label. Then when de-serializing, we re-generate the
        remaining data via the call to setup()
        """
        state = {}
        state['label'] = self.__dict__['label']
        return state

    def __setstate__(self, d):
        """After de-serializing call set up to finish "setting up" the class."""
        self.__dict__ = d
        self.setup()

    @abc.abstractmethod
    def setup(self):
        pass


class RandomGenLecSha256(PuaRandomGen):
    def __init__(self, label: str):
        PuaRandomGen.__init__(self, label)
        self.setup()

    def setup(self):
        """We separated the setup of this class since it includes some data that isn't readily serializable."""
        # Next line creates an array of size 6 of c_ulong to be used later

        self.seedArray = (ctypes.c_ulong * 6)(*([0] * 6))
        self.rngStream = None
        self.chunkBegin = None

        self.genSeedArray()

        inStr = ctypes.c_char_p(b'Uniform')
        cStr = ctypes.cast(ctypes.pointer(inStr), ctypes.c_char_p)

        self.rng = rngStream.RngStream_CreateStream(cStr)

        cOne = ctypes.c_int(1)
        rngStream.RngStream_IncreasedPrecis(self.rng, cOne)

        rngStream.RngStream_SetSeed(self.rng, self.seedArray)

    def genSeedArray(self):
        # Generate 6 ints from SHA256gg
        sha256 = hashlib.sha256()
        sha256.update(self.label.encode())
        bytesFromLabel = sha256.digest()
        for i in range(6):
            self.seedArray[i] = int.from_bytes(
                bytesFromLabel[i * 4 : (i + 1) * 4], byteorder='little', signed=False
            )

    def resetStartStream(self):
        rngStream.RngStream_ResetStartStream(self.rng)

    def setStreamBegin(self, chunkBegin):
        self.chunkBegin = chunkBegin
        [rngE, rngN] = self.offsetToEN(self.chunkBegin)

        self.resetStartStream()
        # print('e, n: {} {}'.format(np.int64(ctypes.c_long(rngE)), np.int64(ctypes.c_long(rngN))))
        rngStream.RngStream_AdvanceState(
            self.rng, ctypes.c_long(rngE), ctypes.c_long(rngN)
        )

    @staticmethod
    def offsetToEN(k):
        """
        Map an offset into the form required for RngStreams.

        # if e > 0, let k = 2^e + c;
        # if e < 0, let k = -2^(-e) + c;
        # if e = 0, let k = c.
        Note: Not sure this works for negative values of k - need to test
        """
        if k < 0:
            sign = -1
            k = -k
        elif k > 0:
            sign = 1
        elif k == 0:
            return 0, -1  # 2^(0) - 1 = 0

        e = np.int64(np.log2(k))
        c = k - 2**e

        e = sign * e
        c = sign * c
        return e, c

    def getNumber(
        self,
    ):
        return rngStream.RngStream_RandU01(self.rng)


class RMEParameter(GroupSaveable):
    """
    RME parameters model scalar random variables.

    Random variables that are a function of a parameter like frequency are
    Represented by RMEModel objects.

    RMEparameters use PuaRandomGen random number generators to
    support paralell Monte-Carlo analysis.

    RME parameters have two attributes that work like names.
    * ``name`` is intended to be human-readable. It does not need to be unique.
    * ``umech_id`` is unique, and serves as a seed for the random number generator.

    umech_id is also used in the linear uncertainty analysis to distinguish
    uncertainty mechanisms. This is another reason it needs to be unique.

    By default, umech_id is generated from the name attribute, plus a
    unique identifier.

    """

    def __init__(self, name: str, umech_id: str = None, dtype: str = None):
        GroupSaveable.__init__(self)

        if umech_id is None:
            umech_id = name + '_' + str(self.attrs['unique_id'])

        else:
            umech_id = umech_id

        # check that the dtype str is valid
        dtype = str(np.dtype(dtype))

        self.add_child('name', name)
        self.add_child('umech_id', umech_id)
        self.add_child('dtype', dtype)
        self.gen = RandomGenLecSha256(umech_id)

    @property
    @_abc.abstractmethod
    def nom(self):
        """
        Return nominal value.

        Returns
        -------
        None.

        """
        pass

    @property
    @_abc.abstractmethod
    def var(self):
        """
        Return variance.

        Returns
        -------
        None.

        """
        pass

    def generate_cov(self):
        """
        Return nominal value perturbed by one standard uncertainty.

        Returns
        -------
        None.

        """
        return self.nom + np.sqrt(self.var)

    @_abc.abstractmethod
    def generate_mc(offset: int, samples: int) -> np.ndarray:
        """
        Generate Monte Carlo samples.

        Parameters
        ----------
        offset : int
            Index of first sample.
        samples : int
            Number of samples.

        Returns
        -------
        np.ndarray
            An array of random numbers.

        """
        pass

    def evaluate(
        self,
        output_coords: dict,
        linear_uncertainty: bool,
        offset: int,
        samples: int,
    ) -> arrschema.AnnotatedArray:
        """
        Generate either Monte Carlo samples or perturbed values.

        Parameters
        ----------
        output_coords : dict
            Coords of output data.
        output_dims : tuple[str]
            Dimensions of output data.
        output_shape : tuple[int]
            Shape of output data.
        linear_uncertainty : bool
            If True, do linear uncertainty analysis.
        offset : int
            Index of first sample.
        samples : int
            Number of samples.

        Returns
        -------
        output : AnnotatedArray
            Array describing either a Monte Carlo or linear uncertainty analysis.

        """
        output_shape = []
        output_dims = []
        for dim, coord_array in output_coords.items():
            output_dims.append(dim)
            output_shape.append(len(coord_array))

        if 'umech_id' in output_coords:
            data = np.zeros(output_shape, dtype=self.dtype)
            output = xr.DataArray(data=data, coords=output_coords, dims=output_dims)
            output += self.nom
            output.loc[dict(umech_id=self.umech_id)] = self.generate_cov()

        if 'sample_id' in output_coords:
            data = self.generate_mc(offset, samples)
            output = xr.DataArray(
                data=data, coords={'sample_id': output_coords['sample_id']}
            )
            other_coords = {}
            for coord in output_coords:
                if coord not in output.coords:
                    other_coords[coord] = output_coords[coord]

            output = output.expand_dims(other_coords)

        return output


class Uniform(RMEParameter):
    """
    A uniform, continuous random variable.

    Following notation in scipy.stats.uniform, the distribution
    is parameterized by ``loc`` and ``scale``.

    The probability density is uniform on the interval
    ``[loc, loc + scale]``

    Implemented by scaling numbers generated by RngStreams library
    (see module-level documentation for citaion).
    """

    def __init__(self, name: str, loc: float, scale: float):
        RMEParameter.__init__(self, name)
        self.loc = loc
        self.scale = scale
        self.nom = self.loc + self.scale / 2.0
        self.var = scale / 12.0

    def nom(self):
        """
        Return nominal value.

        Returns
        -------
        None.

        """
        return self.nom

    def var(self):
        """
        Return variance.

        Returns
        -------
        None.

        """
        return self.var

    def generate_mc(self, offset: int, samples: int) -> np.ndarray:
        """
        Generate Monte Carlo samples.

        Parameters
        ----------
        offset : int
            Index of first sample.
        samples : int
            Number of samples.

        Returns
        -------
        np.ndarray
            An array of random numbers.

        """
        if samples < 1:
            raise (ValueError('Invalid number of samples: {}').format(samples))

        self.gen.setStreamBegin(offset)
        numbers = self.gen.getNumbers(samples)
        return self.loc + numbers * self.scale


class Norm(RMEParameter):
    """
    A Gaussian, continuous random variable.

    Following notation in scipy.stats.norm, the distribution
    is parameterized by ``loc`` and ``scale``.

    ``loc`` is the mean. ``scale`` is standard deviation.

    Implemented with the RngStreams library, which generates uniform
    random numbers. The uniform random numbers are transformed
    to normal by the Box-Muller transform.

    (see module-level documentation for citaion).
    """

    def __init__(self, name: str, loc: float, scale: float):
        RMEParameter.__init__(self, name)
        self.loc = loc
        self.scale = scale
        self.nom = self.loc
        self.var = scale**2
        self.gen = RandomGenLecSha256('parameter')

    def nom(self):
        """
        Return nominal value.

        Returns
        -------
        None.

        """
        return self.nom

    def var(self):
        """
        Return variance.

        Returns
        -------
        None.

        """
        return self.var

    def generate_mc(self, offset: int, samples: int) -> np.ndarray:
        """
        Generate Monte Carlo samples.

        Parameters
        ----------
        offset : int
            Index of first sample.
        samples : int
            Number of samples.

        Returns
        -------
        np.ndarray
            An array of random numbers.

        """

        # the Box-Muller transform works on an even number of samples
        even_number = samples

        if even_number % 2 == 1:
            even_number = samples + 1

        if even_number % 2 != 0 or samples < 1:
            raise (ValueError('Invalid number of samples: {}'.format(samples)))

        # sample random numbers
        self.gen.setStreamBegin(offset)
        numbers = self.gen.getNumbers(even_number)

        # Box-Muller transform
        U1 = numbers[0::2]
        U2 = numbers[1::2]

        R = -2.0 * np.log(U1)
        Theta = 2.0 * np.pi * U2

        Z1 = R * np.cos(Theta)
        Z2 = R * np.sin(Theta)

        unit_gaussian = np.zeros(even_number)
        unit_gaussian[0::2] = Z1
        unit_gaussian[1::2] = Z2
        unit_gaussian = unit_gaussian[:samples]

        return self.loc + self.scale * unit_gaussian


class RMECoord(metaclass=ABCMeta):
    """
    Used to test if objects could be interpreted as coordinates.

    Inspired by np.ndarray.
    """

    @classmethod
    def __subclasshook__(cls, C):
        has_shape = hasattr(C, 'shape')
        has_no_coords = not hasattr(C, 'coords')

        return has_shape and has_no_coords


def _is_number_array(x) -> bool:
    """
    Returns True if x is an array of numbers.

    Internal helper function.

    Parameters
    ----------
    x : RMECOORD
        A coordinate array

    Returns
    -------
    bool
        True if x is an array for numbers.

    """
    is_number = [isinstance(x_i, Number) for x_i in x]
    return np.all(is_number)


class RMEModelError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class RMEModel:
    """
    RME models wrap a function, allowing for:
        * Evaluation on arbitrary coordinate grids,
        * Monte-Carlo Uncertainty analysis,
        * Linear uncertainty analysis.

    Requirements for wrapped function:
        * Function must return an AnnotatedArray as the output.
        * Functions cannot have *args in function signature.

    Requirements for arrays as parameters:
        * Arrays must have a 'coords' attribute.

    Requirements for coords passed as parameters:
        * Coordinates must be 1-dimensional arrays with no 'coords' attribute.
        * Coordinate names are assigned based on the function signature.
        * Coordinates associated with all input data must match.
    """

    def __init__(
        self,
        function: Callable,
        coord_alias: Mapping = {},
    ):
        """
        Initialize an RME model.

        Parameters
        ----------
        function : Callable
            A deterministic mathematical function, which returns an array.

        coord_alias : Mapping, optional
            Used to override names of coordinates in input arrays.
            Keys: coordinate names in input arrays.
            Values: coordinate names RME passes to functions.
            The default is {}.

        Returns
        -------
        None.

        """
        # TODO: add cached value attribute
        self.function = function
        self.fcn_signature = signature(self.function)
        self.partial_bound_args = None
        self.full_bound_args = None
        self.evaluated_args = None

        self._has_var_keyword = False
        # check if function has VAR_KEYWORD args
        for parameter_key, parameter in self.fcn_signature.parameters.items():
            if parameter.kind == parameter.VAR_KEYWORD:
                self._has_var_keyword = True
            if parameter.kind == parameter.VAR_POSITIONAL:
                raise (
                    RMEModelError(
                        'Wrapped function cannot have variable positional arguments (*args).'
                    )
                )

        self.coord_alias = coord_alias

    @staticmethod
    def _get_usage(arg: object) -> str:
        """
        Examine an argument and determine how it is used in model evaluation.

        Helper method, for internal use only.

        Parameters
        ----------
        arg : object
            Object to examine.

        Raises
        ------
        RMEModelError
            If the usage of an object could not be determined.

        Returns
        -------
        usage : str
            Options are: 'unknown', 'coord', 'parameter', 'model'.

        """
        usage = 'unknown'

        if isinstance(arg, arrschema.AnnotatedArray):
            usage = 'array'

        if isinstance(arg, RMECoord):
            usage = 'coord'
            if len(arg.shape) != 1:
                raise RMEModelError(
                    'Tried to interpret {} as a coordinate, but len(shape) was {}. For coordiantes, len(shape) must be 1.'.format(
                        arg, len(arg.shape)
                    )
                )

        if isinstance(arg, RMEParameter):
            usage = 'parameter'

        if isinstance(arg, RMEModel):
            usage = 'model'

        return usage

    def _add_coord(self, name: str, values: RMECoord):
        """
        Add coordinate to self.required_coords.

        Helper method, for internal use only.

        Parameters
        ----------
        name : str
            Name of coordinate.
        values : RMECoord
            Values for coordinate.

        Raises
        ------
        RMEModelError
            If new coordinate is inconsistent with existing coordinates.

        Returns
        -------
        None.

        """
        new_name = name
        if name in self.coord_alias.keys():
            new_name = self.coord_alias[name]

        if new_name in self.required_coords.keys():
            existing_coord_array = self.required_coords[new_name]
            existing_is_numbers = _is_number_array(existing_coord_array)
            values_is_numbers = _is_number_array(values)
            all_numbers = existing_is_numbers and values_is_numbers
            if existing_is_numbers != values_is_numbers:
                raise RMEModelError(
                    'Inconsistent information for coordinate {}'.format(new_name)
                )
            if all_numbers and not np.all(np.isclose(existing_coord_array, values)):
                raise RMEModelError(
                    'Inconsistent information for coordinate {}'.format(new_name)
                )
            if not all_numbers and not np.all(existing_coord_array == values):
                print(existing_coord_array, values)
                raise RMEModelError(
                    'Inconsistent information for coordinate {}'.format(new_name)
                )
        else:
            self.required_coords[new_name] = values

    def get_umech_ids(self):
        """
        Get umech_ids from bound arguments.

        Returns
        -------
        None.

        """
        umech_ids_set = set()

        for key, value in self.partial_bound_args.arguments.items():
            parameter = self.fcn_signature.parameters[key]

            if parameter.kind == parameter.VAR_POSITIONAL:
                continue

            usage = self._get_usage(value)

            if usage == 'array':
                if 'umech_id' in value.coords:
                    umech_ids_set.update(value.coords['umech_id'])

            if usage == 'parameter':
                umech_ids_set.add(value.umech_id)

            if usage == 'model':
                umech_ids_set.update(value.get_umech_ids())

        return umech_ids_set

    def _add_umech_id_to_coords(self, arrays, parameters, models):
        """
        Add umech_id to self.output_coords.

        Helper method, for internal use only.

        Parameters
        ----------
        *arrays : positional arguments.
            Arrays with a umech_id coordinate.

        Returns
        -------
        None.

        """
        umech_id_set = set()
        if 'umech_id' in self.required_coords.keys():
            umech_id_set.update(self.required_coords['umech_id'])

        for parameter in parameters:
            umech_id_set.add(parameter.umech_id)

        for array in arrays:
            try:
                umech_ids = array.coords['umech_id']
                umech_id_set.update(umech_ids)

            except KeyError:
                pass

        for model in models:
            # todo: add model.get_umech_ids().
            # models need to know if sub_models have umech_ids
            # because of bound arguments.
            umech_ids = model.get_umech_ids()
            umech_id_set.update(umech_ids)

        # print("*" * 80)
        # print("umech_id_set", list(umech_id_set))
        # self._add_coord('umech_id', ['nominal'] + list(umech_id_set))
        umech_id_set.discard('nominal')
        self.required_coords['umech_id'] = ['nominal'] + list(umech_id_set)

    def _add_sample_id_to_coords(self, offset: int, samples: int):
        """
        Add sample_id to self.output_coords

        Parameters
        ----------
        offset : int
            Index of first Monte Carlo sample.
        samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        None.

        """
        self._add_coord('sample_id', list(range(offset, offset + samples)))

    def _expand_array(self, array: arrschema.AnnotatedArray):
        """
        Expand array to specified dimesnions.

        Helper method, for internal use only.

        Parameters
        ----------
        array : arrschema.AnnotatedArray
            Array to expand.

        Returns
        -------
        None.

        """
        boring_coords = {}
        for coord_name, coord_array in self.required_coords.items():
            if coord_name != 'umech_id':
                if coord_name not in array.coords.keys():
                    boring_coords[coord_name] = coord_array

        expanded = array.expand_dims(boring_coords)

        if 'umech_id' in self.required_coords and 'umech_id' in array.coords:
            nominal_fill_value = expanded[0, ...]
            expanded = nominal_fill_value.expand_dims(
                {'umech_id': self.required_coords['umech_id']}
            ).copy()
            # assign umech_id locations to original cov data
            intersections, comm1, comm2 = np.intersect1d(
                self.required_coords['umech_id'],
                array.coords['umech_id'],
                assume_unique=True,
                return_indices=True,
            )

            expanded[comm1, ...] = array[comm2, ...]

        if 'umech_id' in self.required_coords and 'umech_id' not in array.coords:
            nominal_fill_value = expanded
            expanded = nominal_fill_value.expand_dims(
                {'umech_id': self.required_coords['umech_id']}
            ).copy()

        return expanded

    def _expand_coord(self, coord_name: str, coord_vals: RMECoord):
        """
        Expand coordinate to specified dimesnions.

        Parameters
        ----------
        coord_name : str
            Name of coordinate
        coord_vals : RMECoord
            Coordinate to expand

        Returns
        -------
        None.

        """
        indexer = []
        for dim in self.required_coords.keys():
            if dim == coord_name:
                indexer.append(slice(None, None, None))
            else:
                indexer.append(np.newaxis)

        return coord_vals[*indexer]

    def bind(self, *args: tuple, **kwargs: dict):
        """
        Bind arguments to the model.

        Not all arguments need to be specified. Unspecified arguments need
        to be supplied when evaluate is called.

        Parameters
        ----------
            args : Positional arguments
            Positional arguments, passed to self.function.

            kwargs : Keyword arguments
            Keyword arguments, passed to self.function.

        """
        self.partial_bound_args = self.fcn_signature.bind_partial(*args, **kwargs)

    # Note, this function exists so that the "__call__" function can
    # have the sampe arguments as the function it wraps.
    def setup(
        self,
        required_coords: Mapping = None,
        linear_uncertainty: bool = False,
        offset: int = 0,
        samples: int = 0,
    ):
        """
        Supply data needed for model evaluation.

        Parameters
        ----------
        required_coords : Mapping, optional
            When a model is evaluated, required coords are added to inputs
            if they do not exist already.
            The default is {}.
        linear_uncertainty : bool, optional
            If True, perform linear uncertainty analysis. The default is False.
        offset : int, optional
            Index of first sample in a Monte Carlo analysis. The default is 0.
        samples : int, optional
            Number of samples in a Monte Carlo analysis. The default is 0.

        Returns
        -------
        None.

        """

        if required_coords is not None:
            self.required_coords = required_coords
        else:
            self.required_coords = {}

        self.linear_uncertainty = linear_uncertainty
        self.offset = offset
        self.samples = samples

        self.full_bound_args = None
        self.evaluated_args = None

        # special logic for sample_id
        if self.samples > 0:
            if linear_uncertainty:
                raise RMEModelError(
                    "Models can't have both linear uncertainty and Monte Carlo analysis."
                )

            self._add_sample_id_to_coords(self.offset, self.samples)

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> arrschema.AnnotatedArray:
        """
        Evaluate function with specified args.

        Evaluating a model entails: determine output grid, evaluating models,
        evaluating parameters, aligning all arrays to a common grid, and
        then feeding all of these arrays to self.function.

        Parameters
        ----------
        *args : Positional arguments
            Positional arguments, passed to function.

        **kwargs : Keyword arguments.
            Keyword arguments, passed to function

        Returns
        -------
        result : AnnotatedArray
            Evaluated on a coordinate grid determined from inputs.

        """
        unevaluated_args = []
        unevaluated_kwargs = {}

        # copy bound arguments into unevaluated_args, unevaluated_kwargs
        if self.partial_bound_args is not None:
            for arg_name in self.partial_bound_args.arguments:
                parameter = self.fcn_signature.parameters[arg_name]
                arg_val = self.partial_bound_args.arguments[arg_name]

                if parameter.kind == parameter.VAR_KEYWORD:
                    unevaluated_kwargs.update(**arg_val)

                elif (
                    parameter.kind == parameter.POSITIONAL_OR_KEYWORD
                    or parameter.kind == parameter.KEYWORD_ONLY
                ):
                    unevaluated_kwargs[arg_name] = arg_val

                elif parameter.kind == parameter.POSITIONAL_ONLY:
                    unevaluated_args.append(arg_val)

        # copy args and kwargs into unevaluated_args, unevaluated_kwargs
        unevaluated_args.extend(args)

        if self._has_var_keyword:
            unevaluated_kwargs.update(**kwargs)

        # if the function_signature has no variable keyword arguments,
        # it will not accept any keyword arguments it does not expect.
        else:
            known_parameters = []
            for name, parameter in self.fcn_signature.parameters.items():
                if (
                    parameter.kind == parameter.KEYWORD_ONLY
                    or parameter.kind == parameter.POSITIONAL_OR_KEYWORD
                ):
                    known_parameters.append(name)

            # we prefer bound arguments over keyword arguments
            # to avoid infinite recursion. I found this problem when
            # model 1 : S-parameters, takes "RLGC_data" as argument
            # model 2 : Perturbed RLGC, takes "RLCG_data" as argument
            # model 1 was using model 2 to generate RLCG data,
            # hence, infinite recursion.
            for key, value in kwargs.items():
                if key in known_parameters and key not in unevaluated_kwargs.keys():
                    unevaluated_kwargs[key] = value

        # print("*" * 80)
        # print("unevaluated_args", unevaluated_args)
        # print("unevaluated_kwargs", unevaluated_kwargs)

        self.full_bound_args = self.fcn_signature.bind(
            *unevaluated_args, **unevaluated_kwargs
        )
        self.full_bound_args.apply_defaults()

        # update coordinates
        arrays = []
        parameters = []
        models = []
        for key, value in self.full_bound_args.arguments.items():
            parameter = self.fcn_signature.parameters[key]

            if parameter.kind == parameter.VAR_POSITIONAL:
                continue

            usage = self._get_usage(value)

            if usage == 'array':
                arrays.append(value)

            if usage == 'parameter':
                parameters.append(value)

            if usage == 'model':
                models.append(value)

        # if linear uncertainty analysis, umech_id needs to be consistent
        if self.linear_uncertainty:
            self._add_umech_id_to_coords(arrays, parameters, models)

        keys = []
        evaluated_args = []
        for key, value in self.full_bound_args.arguments.items():
            parameter = self.fcn_signature.parameters[key]

            keys.append(key)
            evaluated_arg = value

            usage = RMEModel._get_usage(value)

            if usage == 'coord':
                evaluated_arg = self._expand_coord(key, value)

            if usage == 'array':
                evaluated_arg = self._expand_array(value)

            if usage == 'parameter':
                evaluated_arg = value.evaluate(
                    self.required_coords,
                    self.linear_uncertainty,
                    self.offset,
                    self.samples,
                )

            if usage == 'model':
                # print("*"*80)
                # print("evaluating model:", self.name)
                # print("parameter:", key)
                value.setup(
                    self.required_coords,
                    self.linear_uncertainty,
                    self.offset,
                    self.samples,
                )

                evaluated_arg = value.__call__(
                    *unevaluated_args,
                    **unevaluated_kwargs,
                )

            evaluated_args.append(evaluated_arg)

        evaluated_args_dict = dict(zip(keys, evaluated_args))
        self.evaluated_args = self.fcn_signature.bind(**evaluated_args_dict)
        result = self.function(**evaluated_args_dict)

        return result


class RMEMeasModel:
    """
    RMEMeas model is like RMEModel, but generates RMEmeas, instead of arrays.

    Call functions in this order:
    1. __init__: supply a function to generate data.
    2. setup: set parameters of uncertainty analysis.
    3. bind (optional) store parameters.
    4. __call__: evaluate function
    """

    def __init__(
        self,
        function: Callable,
        coord_alias: Mapping = {},
    ):
        """
        Initialize an RMEMeas Model.

        Parameters
        ----------
        function : Callable
            A deterministic mathematical function, which returns an array.

        coord_alias : Mapping, optional
            Used to override names of coordinates in input arrays.
            Keys: coordinate names in input arrays.
            Values: coordinate names RME passes to functions.
            The default is {}.

        Returns
        -------
        None.

        """
        self.function = function
        self.fcn_signature = signature(self.function)
        self.required_coords = None

        self._has_var_keyword = False
        # check if function has VAR_KEYWORD args
        for parameter_key, parameter in self.fcn_signature.parameters.items():
            if parameter.kind == parameter.VAR_KEYWORD:
                self._has_var_keyword = True
            if parameter.kind == parameter.VAR_POSITIONAL:
                raise (
                    RMEModelError(
                        'Wrapped function cannot have variable positional arguments (*args).'
                    )
                )

        self.coord_alias = coord_alias
        self.MC_model = None
        self.linear_model = None

    def setup(
        self,
        required_coords: Mapping = None,
        linear_uncertainty: bool = False,
        offset: int = 0,
        samples: int = 0,
    ):
        """
        Supply data needed for model evaluation.

        Parameters
        ----------
        required_coords : Mapping, optional
            When a model is evaluated, required coords are added to inputs
            if they do not exist already.
            The default is {}.
        linear_uncertainty : bool, optional
            If True, perform linear uncertainty analysis. The default is False.
        offset : int, optional
            Index of first sample in a Monte Carlo analysis. The default is 0.
        samples : int, optional
            Number of samples in a Monte Carlo analysis. The default is 0.

        Returns
        -------
        None.

        """
        if required_coords is not None:
            self.required_coords = required_coords

        else:
            self.required_coords = {}

        self.linear_uncertainty = linear_uncertainty
        self.offset = offset
        self.samples = samples

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> arrschema.AnnotatedArray:
        """
        Evaluate function with specified args.

        Evaluating a model entails: determine output grid, evaluating models,
        evaluating parameters, aligning all arrays to a common grid, and
        then feeding all of these arrays to self.function.

        Parameters
        ----------
        *args : Positional arguments
            Positional arguments, passed to function.

        **kwargs : Keyword arguments.
            Keyword arguments, passed to function

        Returns
        -------
        result : AnnotatedArray
            Evaluated on a coordinate grid determined from inputs.

        """
        pass

    def bind(self, *args: tuple, **kwargs: dict):
        """
        Bind arguments to the model.

        Not all arguments need to be specified. Unspecified arguments need
        to be supplied when evaluate is called.

        Parameters
        ----------
            args : Positional arguments
            Positional arguments, passed to self.function.

            kwargs : Keyword arguments
            Keyword arguments, passed to self.function.

        """
        pass
