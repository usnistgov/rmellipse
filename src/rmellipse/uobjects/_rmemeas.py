"""
Module for creating and interacting with RMEMeas objects.

RMEMeas objects are the
"""

import copy
import warnings

# These need to be imported this way to delay access
# to the underlying classes to avoid a circular import error
import rmellipse.propagators as propagators
from rmellipse.arrschema import AnnotatedArray, ArraySchema, ValidationError
from rmellipse.utils import GroupSaveable, load_object
import h5py
import xarray as xr
import numpy as np  # because math
import os.path
import uuid

# from functools import wraps
from rmellipse.utils import MUFMeasParser

# from rmellipse.utils import console_graphics
from collections.abc import Iterable, Sequence
from collections import namedtuple
from typing import TypeVar, Any


# umechdim = 'umechs'


__all__ = [
    'CovarianceDataArray',
    'CovarianceDOFMetadata',
    'CovarianceStrMetadata',
    'MonteCarloDataArray',
    'RMEMeas',
    'RMEMeasFormatError',
    'RMEUncTuple',
]

UMECHID_DTYPE = np.dtype('object')
RMELLIPSE_NAMESPACE = uuid.UUID('c421f33f-4b1c-4f15-a2a6-896cd39430b1')
MC_DIM_NAME = 'sample_id'


class RMEMeasFormatError(Exception):
    """Error in formatting of data inside RMEMeas

    Parameters
    ----------
    Exception : _type_
        _description_
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _angle_diff(phase1: xr.DataArray, phase2: xr.DataArray, deg=False) -> xr.DataArray:
    """
    Calculate the minimum difference between two angles.

    Parameters
    ----------
    angle1 : xr.DataArray
        Phase array 1 in radians or deg.
    angle2 : xr.DataArray
        Phase array 1 in radians or deg.
    deg: bool, optional
        If true, calculates in degrees

    Returns
    -------
    out : xr.DataArray
        Phase difference between two angles.

    """
    const = np.pi
    if deg:
        const = 180

    diff = ((phase1.values - phase2.values) + const) % (2 * const) - const

    # return the minimum possible phase difference
    out = phase1.copy()
    out.values = diff
    return out


class CovarianceDataArray(AnnotatedArray):
    """
    Store data with linear uncertainty mechanisms.

    The first label of the umech_id coordinate is
    expected to be 'nominal', and represents the
    nominal (i.e. expected value) of the data set.
    Each label of the umech_id coordinate after that
    represents the nominal dataset perturbed by 1 standard
    deviation of that uncertainty mechanism, where the label
    is a universally unique ID that identifies the
    uncertainty mechanism.
    """

    schema = ArraySchema(
        shape=('N', ...),
        dims=('umech_id', ...),
        coords={'umech_id': {'dtype': np.dtype('T')}},
    )


class MonteCarloDataArray(AnnotatedArray):
    """
    Store stochastic Monte Carlo samples.

    Every entry along the ``sample_id`` coordinate is a sample from the
    underlying probability distribution. Labels start at zero and count up by
    one. The nominal value is stored separately in the covariance data.
    """

    schema = ArraySchema(
        shape=('N', ...),
        dims=('sample_id', ...),
        coords={'sample_id': {'dtype': int}},
    )


class CovarianceStrMetadata(AnnotatedArray):
    """
    Stores metadata about linearuncertainty mechanisms.

    The 'nominal' should not be included in the umech_id dim.
    """

    schema = ArraySchema(
        shape=('N', 'M'),
        dims=('umech_id', 'categories'),
        dtype=np.dtype('T'),
        coords={
            'umech_id': {'dtype': np.dtype('T')},
            'categories': {'dtype': np.dtype('T')},
        },
    )


class CovarianceDOFMetadata(AnnotatedArray):
    """
    Stores degrees of freedom of linearuncertainty mechanisms.

    The 'nominal' should not be included in the umech_id dim.
    """

    schema = ArraySchema(
        shape=('N',),
        dims=('umech_id',),
        dtype=float,
        coords={'umech_id': {'dtype': np.dtype('T')}},
    )


RMEUncTuple = namedtuple('RMEUncTuple', 'cov mc')


A = TypeVar('A')


class RMEMeas[A](GroupSaveable):
    """
    Class that stores data along with uncertainties.

    Stores linear uncertainties in the cov attribute and
    montecarlo uncertainties in the cov attribute.

    Used with the rmellipse.RMEProp propagator.
    """

    def __init__(
        self,
        name: str | None = None,
        cov: A | CovarianceDataArray = None,
        mc: A | MonteCarloDataArray | None = None,
        covdofs: xr.DataArray | CovarianceDOFMetadata | None = None,
        covcats: xr.DataArray | CovarianceStrMetadata | None = None,
        parent: GroupSaveable = None,
        attrs: dict | None = None,
    ):
        """
        Initialize a RMEMeas object.

        Please refer to xarray's documentation for information about how to
        use DataArrays, and how to define coordinates and dimensions.

        If covdofs is not provided, distributions of linear uncertainty
        mechanisms are assumed to have infinite degrees of freedom (Gaussian).


        Parameters
        ----------
        name : str, optional
            Name of RMEMeas object. The default is 'RMEMeas'.
        cov : A | CovarianceDataArray, optional
            Covariance data for linear sensitivity analysis. Copies of data set
            are stored along the first dimension (axis 0) of the DataArray, where
            the first index of axis 0 is the nominal data set, and the rest of
            the indexes along that dimension are perturbed by one standard deviation
            (i.e. 1 standard uncertainty).

            The first dimension must be called 'umech_id', and the
            first label of the 'parameter_dimensions' coordinate must be
            'nominal'. The remaining labels for the 'umech_id'
            coordinate should be strings corresponding the the uncertainty
            mechanism. The default is None.
        mc : A | MonteCarloDataArray, optional
            Monte Carlo trials. Every entry along the first dimension (axis 0)
            is a stochastic sample of the data set's distribution; the nominal
            data set is not duplicated in this array.

            The first dimension must be called 'sample_id', and the labels of
            'sample_id' must start at 0 and count up by 1 (i.e. typical integer
            based indexing). The default is None.
        covdofs : xr.DataArray, optional
            DataArray that stores the degrees of freedom for each linear
            uncertainty mechanism in cov. It should be a 1 dimensional
            DataArray with the dimension called 'umech_id' and
            the coordinate set should be identical to the
            'umech_id' coordinate in cov. If covariance data
            is provided and covdofs is not, one will be created that assumes
            all linear mechanisms have infinite degrees of freedom (i.e.
            gaussian distributions). The default is None.
        covcats : xr.DataArray, optional
            DataArray that stores the categories of each uncertainty mechanism.
            Expected to be have dimensions ('umech_id','categories'),
            If not provided, all umech_id are assigned a 'Type' category
            of 'B'. If a mechanism doesn't have a category, it should be an empty
            string.
        """

        # Own metadata independently and provide deterministic identity defaults.
        attrs = copy.deepcopy(dict(attrs or {}))
        if name is None:
            name = attrs.get('name', 'RMEMeas')
        attrs.setdefault('name', name)
        attrs.setdefault('is_big_object', True)

        # initialize as a group saveable thing
        GroupSaveable.__init__(self, name=name, parent=parent, attrs=attrs)

        # assumes first dimension are uncertainty mechanisms
        self.name: str = name  # name of the mechanism
        self.cov: A | CovarianceDataArray = cov  # name of the covariance
        self.mc: A | MonteCarloDataArray | None = mc  # name of the montecarlo data
        self.covdofs = covdofs
        self.covcats = covcats

        # if covariance data is provided but not DOF information is provide
        # default to inf degrees of freedom
        if cov is not None and covdofs is None:
            covdofs = np.full(len(self.umech_id), np.inf)
            covdofs = xr.DataArray(
                covdofs, dims=('umech_id'), coords={'umech_id': self.umech_id}
            )
        self.covdofs: CovarianceDOFMetadata = covdofs

        # default to assigning type b category to anything uncategorized
        if cov is not None and covcats is None:
            dims = ('umech_id', 'categories')
            cats = ['Type']
            coords = {'umech_id': self.umech_id, 'categories': cats}
            values = np.ones((len(self.umech_id), 1)).astype(UMECHID_DTYPE)
            values[...] = 'B'
            covcats = xr.DataArray(values, dims=dims, coords=coords)
        self.covcats: CovarianceStrMetadata = covcats

        # enforce that the umech_id dimension handles strings properly
        self.cast_umechids()
        # enforce the use of string meatadata
        # self.cast_covcats()
        # enforce rule on name of uncertainty/montecarlo dimensions
        # self._validate_conventions()

        # rename the mc dimensiont o support old data sets
        if mc is not None:
            # Legacy Monte Carlo arrays may use 'umech_id' as the sample axis.
            legacy_mc_axis = (
                'sample_id' not in self.mc.dims and 'umech_id' in self.mc.dims
            )
            try:
                self.mc = self.mc.rename({'umech_id': 'sample_id'})
            except ValueError:
                pass
            if legacy_mc_axis and not np.issubdtype(
                self.mc.coords['sample_id'].dtype, np.integer
            ):
                # Integer coordinates distinguish sample indices from
                # covariance mechanism labels.
                raise ValidationError(
                    'legacy Monte Carlo umech_id coordinates must be integers'
                )
            # Validate after normalizing the legacy dimension name.
            MonteCarloDataArray(self.mc).validate()
        # Covariance data is optional for empty or Monte Carlo-only objects.
        if self.cov is not None:
            CovarianceDataArray(self.cov).validate()

        # add attributes as children so they get saved into hdf5 formats
        self.add_child(key='cov', data=self.cov, is_big_object=True)
        self.add_child(key='mc', data=self.mc, is_big_object=True)
        self.add_child(key='covdofs', data=self.covdofs)
        self.add_child(key='covcats', data=self.covcats)

    def __getattr__(self, n):
        try:
            return self.cov.coords[n]
        except KeyError:
            ...
        if self.mc is not None:
            try:
                return self.mc.coords[n]
            except KeyError:
                ...
        raise AttributeError(f'{n} doesnt exists on cov or mc')

    @property
    def covdofs(self):
        """xr.DataArray: The degrees of freedom on linear mechanisms."""
        return self.children['covdofs']

    @covdofs.setter
    def covdofs(self, dataarray):
        """Set the degrees of freedom on linear mechanisms."""
        self.children['covdofs'] = dataarray

    @property
    def covcats(self):
        """xr.DataArray: String metadata for linear uncertainty mechanisms."""
        return self.children['covcats']

    @covcats.setter
    def covcats(self, dictionary):
        """Set the categories of linear mechanisms."""
        self.children['covcats'] = dictionary

    @property
    def mc(self):
        """xr.DataArray: samples of Monte-Carlo distributions."""
        return self.children['mc']

    @mc.setter
    def mc(self, dataarray):
        """Set the montecarlo distributions."""
        self.children['mc'] = dataarray

    @property
    def cov(self):
        """xr.DataArray: Linear uncertainty mechanisms."""
        return self.children['cov']

    @cov.setter
    def cov(self, dataarray):
        """Set the linear uncertainty mechanisms."""
        self.children['cov'] = dataarray

    @property
    def name(self):
        """str: The name of the object."""
        return self.attrs['name']

    @name.setter
    def name(self, value):
        """Set the name of the RMEMeas object."""
        self.attrs['name'] = value

    @property
    def dims(self):
        return self.nom.dims

    @property
    def shape(self):
        return self.nom.shape

    @property
    def coords(self):
        return self.nom.coords

    @property
    def dtype(self):
        return self.nom.dtype

    def __str__(self):
        """Get the string representation of the RMEMeas object."""
        msg = self.name + ' with nominal: \n'
        msg += str(self.nom)
        return msg

    def __repr__(self):
        """Get the representation of the RMEMeas object."""
        return self.__str__()

    @classmethod
    def _if_quacks(cls, obj):
        """Check if obj is a duck type of RMEMeas."""
        if hasattr(obj, 'cov') and hasattr(obj, 'mc') and hasattr(obj, 'name'):
            return True
        else:
            return False

    def cast_umechids(self):
        """
        Cast any umech_id dimensions or other dimensions to the correct type.

        Generally, umech_id and covcats should be 'T' (variable width strings).
        """
        for attr_name in ('covcats', 'covdofs', 'cov'):
            attr = getattr(self, attr_name)
            if attr is not None:
                try:
                    attr_dtype = attr.coords['umech_id'].dtype
                except KeyError as e:
                    raise RMEMeasFormatError(f'{attr} missing umech_id') from e
                except AttributeError as e:
                    raise RMEMeasFormatError(f'{attr} has no coordinates.') from e
                # cast objects of umech_id as string
                # in case they are loaded in as bytes accidentally
                if attr_dtype != np.dtype('T'):
                    setattr(
                        self,
                        attr_name,
                        attr.assign_coords(
                            {'umech_id': attr.coords['umech_id'].astype('T')}
                        ),
                    )
                if self.covcats.dtype != np.dtype('T'):
                    self.covcats = self.covcats.astype('T')

    def _validate_conventions(self):
        """Validate the structural conventions required by RME propagation.

        Call :meth:`validate` after loading, before propagation, or after
        in-place xarray edits that may affect array alignment.
        """
        mechanism_values = []
        if self.cov is not None:
            if not isinstance(self.cov, xr.DataArray):
                raise RMEMeasFormatError('cov must be an xarray.DataArray')
            if not self.cov.dims or self.cov.dims[0] != 'umech_id':
                raise RMEMeasFormatError(
                    'First dimension of cov MUST be called "umech_id"'
                )
            if 'umech_id' not in self.cov.coords:
                raise RMEMeasFormatError('cov is missing its umech_id coordinate')
            mechanism_values = [
                str(value) for value in self.cov.coords['umech_id'].values
            ]
            if not mechanism_values or mechanism_values[0] != 'nominal':
                raise RMEMeasFormatError(
                    'First label of "umech_id" MUST be called "nominal"'
                )
            if len(mechanism_values) != len(set(mechanism_values)):
                raise RMEMeasFormatError('cov umech_id labels must be unique')

        if self.mc is not None:
            if not isinstance(self.mc, xr.DataArray):
                raise RMEMeasFormatError('mc must be an xarray.DataArray')
            if not self.mc.dims or self.mc.dims[0] != 'sample_id':
                raise RMEMeasFormatError(
                    'First dimension of mc MUST be called "sample_id"'
                )
            if 'umech_id' in self.mc.dims:
                raise RMEMeasFormatError(
                    'mc cannot contain the reserved "umech_id" dimension'
                )
            sample_ids = np.asarray(self.mc.coords['sample_id'].values)
            expected_ids = np.arange(self.mc.sizes['sample_id'])
            if not np.issubdtype(sample_ids.dtype, np.integer) or not np.array_equal(
                sample_ids, expected_ids
            ):
                raise RMEMeasFormatError(
                    'mc sample_id coordinates must be consecutive integers starting at 0'
                )
            if self.mc.sizes['sample_id'] == 0:
                raise RMEMeasFormatError(
                    'mc must contain at least one stochastic sample or be None'
                )
            if self.cov is not None:
                sample = self.mc.isel(sample_id=0, drop=True)
                nominal_cov = self.cov.isel(umech_id=0, drop=True)
                same_layout = (
                    sample.dims == nominal_cov.dims
                    and sample.shape == nominal_cov.shape
                    and sample.coords.to_dataset().equals(
                        nominal_cov.coords.to_dataset()
                    )
                )
                if not same_layout:
                    raise RMEMeasFormatError(
                        'mc samples must have the same dimensions and coordinates '
                        'as the cov nominal value'
                    )

        expected_mechanisms = mechanism_values[1:] if mechanism_values else []
        for attribute_name in ('covdofs', 'covcats'):
            metadata = getattr(self, attribute_name)
            if metadata is None:
                continue
            if not isinstance(metadata, xr.DataArray):
                raise RMEMeasFormatError(
                    f'{attribute_name} must be an xarray.DataArray'
                )
            if not metadata.dims or metadata.dims[0] != 'umech_id':
                raise RMEMeasFormatError(
                    f'First dimension of {attribute_name} MUST be called "umech_id"'
                )
            actual = [
                str(value) for value in metadata.coords['umech_id'].values
            ]
            if actual != expected_mechanisms:
                raise RMEMeasFormatError(
                    f'{attribute_name} umech_id coordinates must match cov '
                    'mechanisms, excluding nominal'
                )

    def validate(self):
        """Validate this object in place and return it for fluent use."""
        self._validate_conventions()
        return self

    @classmethod
    def from_nom(cls, name: str, nom: xr.DataArray) -> 'RMEMeas':
        """
        Create a RMEMeas object with just a nominal dataset.

        Parameters
        ----------
        name : str
            The name of the object to be created.
        nom : xr.DataArray
            Nominal data set.

        Returns
        -------
        'RMEMeas'
            A RMEMeas object with only nominal values.

        """
        if not isinstance(nom, xr.DataArray):
            nom = xr.DataArray(nom)
        cov = nom.expand_dims({'umech_id': ['nominal']})
        out = RMEMeas(name=name, cov=cov)
        # these are special GROUP_SAVEABLE keys
        ign = [
            'name',
            'is_big_object',
            '__class__.__module__',
            '__class__.__name__',
            'save_type',
            'unique_id',
        ]
        for k, v in cov.attrs.items():
            if k not in ign:
                out.attrs[k] = v
        return out

    @classmethod
    def from_xml(
        cls,
        path: str,
        from_csv: callable,
        old_dir: str | None = None,
        new_dir: str | None = None,
        verbose: bool = False,
    ) -> 'RMEMeas':
        """
        Read an XML file from the Microwave Uncertainty Framework.

        covcats and covdofs metadata will be loaded as the default values.

        Parameters
        ----------
        path : str
            Path to xml header file.
        from_csv : callable
            Read function, takes a path to a copy of the data file and returns
            an xarray object.
        old_dir : str | None, optional
            Name of old path stored in xml file. Will be swapped with new_dir
            if provided. Old XML format isn't portable, and the paths
            need to be manually swapped when the files are moved around.
            The default is None.
        new_dir : str | None, optional
            Path string to replace old_dir with. The default is None.

        Raises
        ------
        Exception
            If old_dir/new_dir are not both provided, but one is.

        Returns
        -------
        new : MUFmeas
            MUFmeas object from the legacy format.

        """
        dummy = MUFMeasParser(path)
        dummy.open_data(from_csv, old_base_dir=old_dir, new_base_dir=new_dir)

        # The XML format stores nominal and Monte Carlo data separately;
        # retain that separation in RMEMeas.
        cov = [dummy.nominal_data] + dummy.covariance_data
        umechids = ['nominal'] + [
            d['parameter_location'] for d in dummy.covariance_dict
        ]
        cov = xr.concat(cov, dim='umech_id')
        cov = cov.assign_coords(umech_id=umechids)

        mc = None
        if dummy.montecarlo_data:
            mc = xr.concat(dummy.montecarlo_data, dim='sample_id')
            mc = mc.assign_coords(sample_id=np.arange(mc.shape[0]))

        return cls(dummy.name, cov, mc)

    def to_xml(
        self,
        target_directory: str,
        to_csv: callable,
        data_extension: str,
        header_extension: str = '.meas',
        header_directory: str | None = None,
    ):
        """
        Save to a the Microwave Uncertainty Framework Format.

        This does not preserve metadata in covdofs or covcats.

        Parameters
        ----------
        target_directory : str
            Directory in which to save the support folder which stores all
            the copies of the data.
        to_csv : callable
            Dataformat of the underlying cov-data. Will be inferred if left
            as None. The default is None.
        data_extension: str
            What to save the datafile extensions as (e.g. .s1p, .csv, etc)
        header_extension : str, optional
            What to save the xml header file extension as.
            The default is '.meas'.
        header_directory : str | None, optional
            Location in which to store the header file. If None, defaults
            to the target_directory. Default is None.

        Raises
        ------
        Exception
            If no dataformat provided and one cannot be inferred.

        Returns
        -------
        None.

        """
        if not header_directory:
            header_directory = target_directory

        dummy = MUFMeasParser()

        # create legacy data format lists
        cov_data = []
        mc_data = []
        if self.cov is not None:
            cov_data = [self.cov[i, ...] for i in range(1, self.cov.shape[0])]
        if self.mc is not None:
            mc_data = [self.mc[i, ...] for i in range(self.mc.shape[0])]

        # initialize data inside a dummy legacy object
        dummy.init_from_data(
            name=self.name,
            montecarlo_data=mc_data,
            nominal_data=self.nom,
            covariance_data=cov_data,
            umech_id=self.umech_id,
        )

        # save data to csv
        dummy.save_data(target_directory, save_fcn=to_csv, file_ext=data_extension)

        output_file = os.path.join(header_directory, self.name + header_extension)
        dummy.save_meas(output_file)

    def copy(self) -> 'RMEMeas':
        """
        Make a copy of a RMEMeas object.

        Returns
        -------
        RMEMeas
            Copied object.

        """
        if self.mc is None:
            newmc = None
        else:
            newmc = self.mc.copy()

        newname = self.name
        return RMEMeas(
            name=newname,
            cov=self.cov.copy(),
            mc=newmc,
            covdofs=self.covdofs.copy(),
            covcats=self.covcats.copy(),
        )

    def cull_cov(self, tolerance: float = 0):
        """
        Remove trivial linear uncertainty mechanisms set by tolerance.

        Any linear uncertainty mechanisms with a sum of all standard
        uncertainties < tolerance will be removed from the object.

        For example if tolerance is 0.1, the nominal is  [0,0,0] and
        the perturbed data for a mechanisms is [0.0,0.1,-0.2], then the
        function will evaluate the total standard uncertainty for the mechanism
        as 0.1 + 0.0 + 0.2 = 0.3, and not remove the mechanism.

        This function is called by the auto-cull setting in the RME
        propagator.


        Parameters
        ----------
        tolerance : float, optional
            Maximum value a sum of standard uncertainties that is deemed
            trivial, used to determine what mechanisms should be removed.
            The default is 0.

        Returns
        -------
        None.

        """
        unc = np.abs((self.cov[1:, ...] - self.nom).values)
        sums = unc.sum(axis=tuple(np.arange(1, len(unc.shape))))
        ind = sums > tolerance
        keep1 = {'umech_id': self.cov.umech_id[1:].values[ind]}
        ind = np.append([True], ind)
        keep2 = {'umech_id': self.cov.umech_id.values[ind]}
        self.cov = self.cov.sel(**keep2)
        self.covcats = self.covcats.sel(**keep1)
        self.covdofs = self.covdofs.sel(**keep1)
        pass

    def make_umechs_unique(self, same_uid: bool = False):
        """
        Make parameter locations unique by adding a uuid4 string to end of each.

        A uiid4 string is added to the end of each parameter_location string.
        Useful when reading data from different sources that have the same
        names of uncertainty files. (E.G, multiple device definitions come
        from calibration service formats with generic 'ua,ub' naming)

        Parameters
        ----------
        same_uid: bool, optional
            If true, all parameter locations will have the same uid added
            to the end. Default is False.

        Returns
        -------
        None.

        """
        if not same_uid:
            plocs = [p + str(uuid.uuid4()) for p in self.umech_id]
        else:
            plocs = list(np.array(self.umech_id) + str(uuid.uuid4()))

        self.covdofs = self.covdofs.assign_coords(umech_id=plocs)
        self.covcats = self.covcats.assign_coords(umech_id=plocs)

        plocs2 = ['nominal'] + plocs
        self.cov = self.cov.assign_coords(umech_id=plocs2)

    def add_mc_sample(self, sample: xr.DataArray):
        """
        Add a Monte Carlo sample to the distribution.

        Parameters
        ----------
        sample : xr.DataArray
            Single sample of the probability distribution that represents the
            data set. Expected to be the same shape, dimensions, and
            coordinates of the nominal.

        Returns
        -------
        None.

        """
        ind = 0 if self.mc is None else self.mc.sizes['sample_id']
        new_sample = sample.expand_dims({'sample_id': [ind]})
        if self.mc is None:
            self.mc = new_sample
        else:
            self.mc = xr.concat(
                [self.mc, new_sample], dim='sample_id', join='override'
            )

    def add_umech(
        self,
        name: str,
        value: xr.DataArray,
        dof: float = np.inf,
        category: dict = {'Type': 'B'},
        add_uid=None,
    ):
        """
        Add a linear mechanisms to covariance data.

        Note:

        Parameters
        ----------
        name : str
            Name of new mechanism, must be unique.
        value : xr.DataArray
            Array of nominal+1 standard uncertainty of new mechanism.
            If value has the same size as the nominal data, the function will insert a
            new dimension at axis zero before concatenating to the covariance data.
        dof: float,optional
            Degrees of freedom associated with the uncertainty mechanism.
            Default is infinite.
        category: dict,
            Dictionary of key-value string pairs categorizing the uncertainty
            mechanisms. E.g. {'Type':'B','Origin':'Datasheet'}. The default
            is {'Type':'B'}.
        add_uid: bool, optional
            Append a UID to name to guarantee uniqueness.


        Returns
        -------
        None.

        """
        if add_uid:
            name += str(uuid.uuid4())

        if name in self.umech_id:
            raise ValueError('Linear mechanisms name ' + name + ' already exists')

        try:
            value.size
        except AttributeError:
            value = xr.full_like(self.nom, value)

        if value.size == self.nom.size and len(value.shape) == len(self.nom.shape):
            v = value.expand_dims({'umech_id': [name]}).copy()
        else:
            v = value.assign_coords({'umech_id': [name]})

        self.cov = xr.concat(
            [self.cov, v], dim='umech_id', join='override', coords='minimal'
        )
        dofs = xr.DataArray([dof], dims=('umech_id'), coords={'umech_id': [name]})
        self.covdofs = xr.concat(
            [self.covdofs, dofs], dim='umech_id', join='override', coords='minimal'
        )

        # add extra row to the category for the new mechanism
        # if no mechanisms, initialize first row
        if self.covcats.shape[0] == 0:
            dims = ('umech_id', 'categories')
            cats = ['Type']
            coords = {'umech_id': self.umech_id, 'categories': cats}
            values = np.ones((1, 1)).astype('T')
            values[...] = 'B'
            self.covcats = xr.DataArray(values, dims=dims, coords=coords).assign_coords(
                {'umech_id': [name]}
            )
        # otherwise, append a new row
        else:
            new_row = xr.full_like(self.covcats[[0], ...], '').assign_coords(
                {'umech_id': [name]}
            )
            self.covcats = xr.concat(
                [self.covcats, new_row],
                dim='umech_id',
                join='override',
                coords='minimal',
            )

        # assign categories to new row
        cats = list(category.keys())
        designations = list(category.values())
        n = len(cats)
        self.assign_categories([name] * n, cats, designations)

        return name

    @property
    def nom(self):
        """
        Get the nominal values of the RMEMeas object.

        Covariance data provides an explicit nominal value. If only Monte Carlo
        data is available, return the sample mean of all stored stochastic
        trials.

        Raises
        ------
        Exception
            if no data is store.

        Returns
        -------
        xr.DataArray
            Nominal values in an xr.DataArray

        """
        try:
            return self.cov.sel(umech_id='nominal', drop=True)
        except (TypeError, AttributeError):
            pass
        except ValueError as exec:
            raise RMEMeasFormatError('no umech_id in cov') from exec
        try:
            return self.mc.mean(dim='sample_id')
        except (TypeError, AttributeError, KeyError):
            pass
        except ValueError as exec:
            raise RMEMeasFormatError('no sample_id in mc') from exec
        raise RMEMeasFormatError(
            'Both cov and mc attributes are empty. No nominal data available'
        )

    @property
    def umech_id(self):
        """
        Get the names of uncertainty mechanisms, excluding the nominal.

        Returns
        -------
        List
            List of uncertainty mechanism strings.

        """
        try:
            out = list(self.cov.umech_id.values[1:])
        except AttributeError as exc:
            raise RMEMeasFormatError('umech_id dim not present') from exc

        return out

    def confint(self, percent: float, deg: bool = False, rad: bool = False):
        """
        Generate the lower, upper confidence intervals for a fractional percent confidence.

        Confidence intervals are generate using the linear sensitivity analysis
        data (.cov) using a student-t distribution with means of .nominal, and
        a scale of 1 standard uncertainty.

        Parameters
        ----------
        percent : float
            Percent confidence to calculate intervals on, centered about the
            nominal. For example, a value of 0.5 will calculate intervals such
            that 0.25 of samples are between the mean and the lower, and 0.25
            of expected samples are between the mean and the upper.
        deg : bool, optional
            If you are calculated confidence intervals on degrees.
            The default is False.
        rad : bool, optional
            If you are calculating confidence intervals on radians.
            The default is False.

        Returns
        -------
        lower : xr.DataArray
            Lower bounds on the confidence interval.
        upper : xr.DataArray
            Upper bounds on the confidence interval.

        """
        from scipy.stats import t as student_t

        if deg and rad:
            raise ValueError('pick deg or rad')
        stdunc = self.stdunc(deg=deg, rad=rad)
        dof = self.dof(deg=deg, rad=rad)
        lower_p = (1 - percent) / 2
        upper_p = percent + (1 - percent) / 2

        upper = self.nom.copy()
        lower = self.nom.copy()
        upper.values = student_t.ppf(upper_p, dof, scale=stdunc[0], loc=self.nom)
        lower.values = student_t.ppf(lower_p, dof, scale=stdunc[0], loc=self.nom)

        return lower, upper

    @staticmethod
    def _welchsat_dof(nominal, cov, covdofs, deg: bool = False, rad: bool = False):
        """
        Calculate the Welchsat equation.

        https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/welchsat.htm
        """
        # individual uncertainties
        if deg or rad:
            s = _angle_diff(cov[1:, ...], cov[0, ...], deg=deg)
        else:
            s = cov[1:, ...] - cov[0, ...]
        covunc = np.sqrt((s**2).sum(dim='umech_id'))

        num = covunc**4
        den = (s**4 / covdofs).sum(dim='umech_id')
        return num / den

    def dof(self, deg: bool = False, rad: bool = False):
        """
        Get the dof of of the RMEMeas object's covariance data.

        Returns
        -------
        float
            degrees of freedom of standard uncertainty from linear sensitivity.
        float
            degrees of freedom of standard uncertainty from monte-carlo
        """
        # stdunc
        temp = self.group_combine_mechanisms(deg=deg, rad=rad)
        if deg and rad:
            raise ValueError('pick deg or rad')
        dof = self._welchsat_dof(temp.nom, temp.cov, temp.covdofs, deg=deg, rad=rad)

        return dof

    def stdunc(self, k: float = 1, deg: bool = False, rad: bool = False):
        """
        Get the standard uncertainty with expansion factor k.

        Supports uncertainties on angles via the deg/rad key word arguments.

        Parameters
        ----------
        k : float, optional
            Expansion factor. The default is 1.
        deg: bool, optional
            If true, treats the values as angles and finds the minimum
            distance between the perturbed and un-perturbed data sets in
            degrees. The default is False
        rad: bool, optional
            If true, treats the values as angles and finds the minimum
            distance between the perturbed and un-perturbed data sets in
            radians. The default is False

        Returns
        -------
        covunc : xr.DataArray
            xr.DataArray of std unc from covariance data.
        mcunc : xr.DataArray
            standard uncertainty of montecarlo data.

        """
        try:
            if not deg and not rad:
                # Use perturbation magnitude so real and imaginary components
                # contribute nonnegative terms to the root-sum-square.
                covunc = (
                    k
                    * ((np.abs(self.cov - self.cov[0, ...]) ** 2).sum(dim='umech_id'))
                    ** 0.5
                )
            else:
                if deg and rad:
                    raise ValueError('Either degrees or radians. Not both')
                # at this point if degrees is false, rad is true, and vice versa.
                covunc = (
                    k
                    * (
                        ((_angle_diff(self.cov, self.cov[0, ...], deg=deg)) ** 2).sum(
                            dim='umech_id'
                        )
                    )
                    ** 0.5
                )
        except TypeError:
            covunc = None
        try:
            mcunc = k * self.mc.std(dim='sample_id')
        except (AttributeError, TypeError):
            mcunc = None
        return RMEUncTuple(covunc, mcunc)

    def uncbounds(self, k: float = 1, deg: bool = False, rad: bool = False):
        """
        Get uncertainty bounds (nominal + k*stdunc).

        Supports uncertainties on angles via the deg/rad key word arguments.

        Parameters
        ----------
        k : float, optional
            Expansion factor. The default is 1.
        deg: bool, optional
            If true, treats the values as angles and finds the minimum
            distance between the perturbed and un-perturbed data sets in
            degrees. The default is False
        rad: bool, optional
            If true, treats the values as angles and finds the minimum
            distance between the perturbed and un-perturbed data sets in
            radians. The default is False

        Returns
        -------
        xr.DataArray
            Uncertainty bounds from cov data.
        xr.DataArray
            Uncertainty bounds from montecarlo data.

        """
        n = self.nom
        ucov, umc = self.stdunc(k=k, deg=deg, rad=rad)
        if ucov is None:
            covout = None
        else:
            covout = n + ucov
        if umc is None:
            mcout = None
        else:
            mcout = n + umc
        return RMEUncTuple(covout, mcout)

    def assign_categories(
        self, mechanisms: list[str], categories: list[str], designation: list[str]
    ):
        """
        Assign categories to mechanisms.

        Mechanisms, categories, and designation should all be the same shape.

        categories and designation correspond to key,

        Parameters
        ----------
        mechanisms : list[str]
            1-d List of mechanisms to be assigned a category
        categories : list[str]
            1-d list of categories being assigned. Can be new categories that don't
            already exist. Index corresponds to index of mechanisms.
        designation: list[str]
            1-d list of designation to assign each mechanism. Index corresponds
            to index of mechanisms.

        Returns
        -------
        None.

        """
        self.create_empty_categories(categories)
        for m, c, d in zip(mechanisms, categories, designation):
            self.covcats.loc[m, c] = d

    def assign_categories_to_all(self, **categories: dict):
        """
        Assign categories to all the linear uncertainty mechanisms.

        Parameters
        ----------
        **categories : list[str]
            Keyword argument pairs of category names : designation to assign
            to all linear uncertainty mechanisms in a variable.

        Returns
        -------
        None.
        """
        plocs = self.umech_id
        for cat, des in categories.items():
            categories = [cat] * len(plocs)
            designations = [des] * len(plocs)
            self.assign_categories(plocs, categories, designations)

    def create_empty_categories(self, categories: list[str]):
        """
        Add empty categories to the covcats array.

        If an element in categories already exists, it is ignored and not added.

        Parameters
        ----------
        category : list[str]
            List of categories to create. If they already exist, they are not
            added.

        Returns
        -------
        None.

        """
        if isinstance(categories, str):
            categories = [categories]
        categories = list(
            set([c for c in categories if c not in self.covcats.categories])
        )
        if len(categories) > 0:
            # add extra column for any new categories
            def empty_col(c):
                return xr.full_like(self.covcats[:, [0]], '').assign_coords(
                    {'categories': [c]}
                )

            new_cols = [
                empty_col(c) for c in categories if c not in self.covcats.categories
            ]
            if len(new_cols) > 0:
                new_cols = xr.concat(new_cols, dim='categories', join='override')
                self.covcats = xr.concat(
                    [self.covcats, new_cols], dim='categories', join='override'
                )

    def group_combine_mechanisms(
        self, deg: bool = False, rad: bool = False
    ) -> 'RMEMeas':
        """
        Group linear uncertainty mechanism originating from the same combine call.

        Preserves degrees of freedom does NOT preserve categorical information, used
        by the self.dof() function prior to running the welch-stat equation.

        Returns
        -------
        RMEMeas:
            New RMEMeas object with the linear uncertainty mechanisms originating
            from the same combine call (Type A Analysis) grouped together.

        """

        def sum_reexpand(x, k, nominal, deg, rad):
            if deg or rad:
                if deg:
                    const = 180
                else:
                    const = np.pi
                temp = (
                    ((x.values - nominal.values) + const) % (2 * const) - const
                ) ** 2
            else:
                temp = (x.values - nominal.values) ** 2
            temp = np.sum(temp, axis=0) ** 0.5 + nominal.values
            out = nominal.copy()
            out.values = temp
            out = out.assign_coords(umech_id=[k])
            return out

        new = [self.cov.sel(umech_id=['nominal'])]
        nominal = new[0]
        category = 'combine_id'

        # do stuff
        covcats = self.covcats.copy()
        ind = covcats.values == ''
        covcats.values[ind] = 'Uncategorized'

        try:
            groupings = np.unique(covcats.loc[:, category])
        except KeyError:
            groupings = []

        groups = {
            g: list(covcats.umech_id[covcats.loc[:, category] == g].values)
            for g in groupings
        }

        newdofs = xr.DataArray(
            [], dims=('umech_id'), coords={'umech_id': np.array([], dtype='T')}
        )

        try:
            unused_locs = groups.pop('Uncategorized')
            ind = groupings == 'Uncategorized'
            groupings = np.delete(groupings, ind)
        except KeyError:
            if groups:
                unused_locs = []
            else:
                unused_locs = self.umech_id

        # calculate dofs
        for g in groupings:
            new.append(
                sum_reexpand(
                    self.cov.sel(umech_id=groups[g]), g, nominal, deg=deg, rad=rad
                )
            )
            temp = self.covdofs.loc[[groups[g][0]]].assign_coords({'umech_id': [g]})
            newdofs = xr.concat((newdofs, temp), dim='umech_id')

        # else, get the unused mechanisms, and copy their covariance categories
        if len(unused_locs) > 0:
            new.append(self.cov.sel(umech_id=unused_locs))
            newdofs = xr.concat(
                (newdofs, self.covdofs.loc[unused_locs]), dim='umech_id'
            )

        # concatenate things
        new_cov = xr.concat(new, 'umech_id', join='override')
        newdofs = newdofs.sel(umech_id=new_cov.umech_id[1:])
        if self.mc is None:
            new_mc = None
        else:
            new_mc = self.mc.copy()

        out = RMEMeas(name=self.name, cov=new_cov, mc=new_mc, covdofs=newdofs)
        return out

    def get_unique_categories(self):
        """Get list of unique categories."""
        return list(self.covcats.categories.values)

    def print_categories(self):
        """Print out the unique categories."""
        cats = self.get_unique_categories()
        title = self.name + ' cov categories'
        maxl = max([len(c) for c in cats])
        print(title)
        for c in cats:
            print(c.ljust(maxl + 4))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        propagate = propagators.RMEProp._active.propagate
        new = propagate(ufunc)
        return new(*inputs, **kwargs)

    def __add__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __add__(self, o):
            return self + o

        fnc = propagate(__add__)
        return fnc(self, o)

    def __radd__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __radd__(self, o):
            return self + o

        fnc = propagate(__radd__)
        return fnc(self, o)

    def __mul__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __mul__(self, o):
            return self * o

        fnc = propagate(__mul__)
        return fnc(self, o)

    def __rmul__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __rmul__(self, o):
            return self * o

        fnc = propagate(__rmul__)
        return fnc(self, o)

    def __sub__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __sub__(self, o):
            return self - o

        fnc = propagate(__sub__)
        return fnc(self, o)

    def __rsub__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __rsub__(self, o):
            return o - self

        fnc = propagate(__rsub__)
        return fnc(self, o)

    def __pow__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __pow__(self, o):
            return self**o

        fnc = propagate(__pow__)
        return fnc(self, o)

    def __truediv__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __truediv__(self, o):
            return self / o

        fnc = propagate(__truediv__)
        return fnc(self, o)

    def __rtruediv__(self, o):
        propagate = propagators.RMEProp._active.propagate

        def __rtruediv__(self, o):
            return o / self

        fnc = propagate(__rtruediv__)
        return fnc(self, o)

    def interp(
        self,
        coords: dict = None,
        method: str = 'linear',
        assume_sorted: bool = False,
        kwargs: dict = None,
        **coords_kwargs,
    ):
        """
        Interpolate RMEMeas object.

        See xarray interp documentation for more details.

        Parameters
        ----------
        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be a scalar, array-like or DataArray.
            If DataArrays are passed as new coordinates, their dimensions are
            used for the broadcasting. Missing values are skipped.
            The default is None.
        method : str, optional
            {"linear", "nearest", "zero", "slinear", "quadratic", "cubic",
             "quintic", "polynomial", "pchip", "barycentric", "krogh", "akima",
             "makima"}) – Interpolation method. The default is 'linear'.
        assume_sorted : bool, optional
            If False, values of x can be in any order and they are sorted
            first. If True, x has to be an array of monotonically increasing
            values. The default is False.
        kwargs : dict, optional
            Additional keyword arguments passed to scipy’s interpolator.
            Valid options and their behavior depend whether interp1d or
            interpn is used.. The default is None.
        **coords_kwargs : dict like
            The keyword arguments form of coords. One of coords or
            coords_kwargs must be provided..

        Returns
        -------
        None.

        """
        newcov = self.cov.interp(
            coords=coords,
            method=method,
            assume_sorted=assume_sorted,
            kwargs=kwargs,
            **coords_kwargs,
        )

        newmc = None
        if self.mc is not None:
            newmc = self.mc.interp(
                coords=coords,
                method=method,
                assume_sorted=assume_sorted,
                kwargs=kwargs,
                **coords_kwargs,
            )

        out = RMEMeas(
            self.name, newcov, newmc, self.covdofs.copy(), self.covcats.copy()
        )

        return out

    def usel(
        self, umech_id: Iterable[str] = None, sample_id: Iterable[int] = None
    ) -> 'RMEMeas':
        """
        Get a view into specific uncertainty mechanisms or Monte Carlo samples.

        Parameters
        ----------
        umech_id : iter[str], optional
            Linear uncertainty mechanisms to look at. The default is None.
        sample_id : iter[int], optional
            Monte Carlo samples to look at. The default is None.

        Raises
        ------
        ValueError
            If 'nominal' is passed to umech_id. The covariance nominal is always
            included because an uncertainty object must retain a nominal value.

        Returns
        -------
        'RMEMeas'
            View into RMEMeas object with only the selected linear uncertainty
            mechanisms and Monte Carlo samples.

        """
        cov = self.cov
        covcats = self.covcats
        covdofs = self.covdofs
        mc = self.mc
        umechs = umech_id
        # if its just a string, covdofs will not have the right number of
        # dimensions when it gets indexed into.
        if isinstance(umechs, str):
            umechs = [umechs]

        if umechs or isinstance(umechs, list):
            if 'nominal' in umechs:
                raise ValueError('nominal always included, dont pass it.')
            lin1 = np.array(umechs, dtype=UMECHID_DTYPE)
            # keep1 = {'umech_id': lin1 }
            lin2 = np.append(['nominal'], np.array(umechs, dtype=UMECHID_DTYPE))
            # keep2 = {'umech_id': lin2}
            _, _, where_1 = np.intersect1d(lin2, self.cov.umech_id, return_indices=True)
            _, _, where_2 = np.intersect1d(
                lin1, self.covcats.umech_id, return_indices=True
            )
            _, _, where_3 = np.intersect1d(
                lin1, self.covdofs.umech_id, return_indices=True
            )
            # for some reason this would fail if I just
            # used .sel and the strings are <U36 (that was my
            # situation) so I changed it to use the intersect1d
            # function which didn't fail to truthfully evaluate
            # matching strings. IDK why, maybe something to do
            # with forcing it to store strings as U36?
            cov = self.cov[where_1, ...]
            covcats = self.covcats[where_2, ...]
            covdofs = self.covdofs[where_3, ...]
        elif umechs is not None:
            raise ValueError('umech_id not recognized, must be iterable')

        if sample_id is not None:
            if isinstance(sample_id, (str, bytes)) or not isinstance(
                sample_id, Iterable
            ):
                raise ValueError('mcsamples not recognized, must be iterable')
            keep = np.asarray(list(sample_id), dtype=int)
            if keep.size == 0:
                mc = None
            else:
                mc = self.mc.isel(sample_id=keep)
                mc = mc.assign_coords(
                    sample_id=np.arange(mc.sizes['sample_id'])
                )
        # make sure nominal is first
        uid_sorted = np.append(
            cov.umech_id[cov.umech_id == 'nominal'],
            cov.umech_id[cov.umech_id != 'nominal'],
        )
        cov = cov.sel(umech_id=uid_sorted)
        out = RMEMeas(self.name, cov, mc, covdofs, covcats)

        return out

    def __getitem__(self, slices) -> 'RMEMeas':
        """
        Get view into underlying data with integer based indexing.

        Ignores the umech_id dimension, it cannot be indexed into
        with this function.Index into array assuming the umech_id
        dimensions doesn't exist in cov (i.e. as if indexing into only the
        nominal)

        Generates a view on the underlying cov,mc, covcats
        and covdofs array. Use .copy() to create an unconnected version.
        """
        if not isinstance(slices, tuple):
            slices = (slices,)

        cov = self.cov[:, *slices]
        mc = None
        if self.mc is not None:
            mc = self.mc[:, *slices]

        out = RMEMeas(self.name, cov, mc, self.covdofs, self.covcats)
        return out

    @property
    def loc(self):
        """
        Get view into underlying data with label based indexing.

        Ignores the umech_id dimension, it cannot be indexed into
        with this function.Index into array assuming the umech_id
        dimensions doesn't exist in cov (i.e. as if indexing into only the
        nominal)

        Generates a view on the underlying cov,mc, covcats
        and covdofs array. Use .copy() to create an unconnected version.
        """
        return _RMELocIndexer(self)

    def sel(
        self,
        indexers: dict = None,
        method: str = None,
        tolerance: float = None,
        **indexers_kwargs,
    ) -> 'RMEMeas':
        """
        Get view into underlying data with label based selection.

        Ignores the umech_id dimension, it cannot be indexed into
        with this function.Index into array assuming the umech_id
        dimensions doesn't exist in cov (i.e. as if indexing into only the
        nominal)

        Generates a view on the underlying cov,mc, covcats
        and covdofs array. Use .copy() to create an unconnected version.

        See documentation on DataArray.sel in the xarray package for details.

        Parameters
        ----------
        indexers : TYPE, optional
            A dict with keys matching dimensions and values given by scalars,
            slices or arrays of tick labels. For dimensions with multi-index,
            the indexer may also be a dict-like object with keys matching index
            level names.
            umech_id can't be provided.
        method : str, optional
            Method to use for inexact matches:
            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
            The default is None.
        tolerance : float, optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation abs(index[indexer] - target) <= tolerance.
        **indexers_kwargs : dict
            The keyword arguments form of indexers.
            One of indexers or indexers_kwargs must be provided.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        out : RMEMeas
            View into indexed RMEMeas object.

        """
        forbid = ['umech_id', 'sample_id']
        for f in forbid:
            if indexers_kwargs and f in indexers_kwargs:
                raise ValueError(f' Cannot index into {f} here')
            if indexers and f in indexers:
                raise ValueError(f' Cannot index into {f} here')

        kwargs = dict(indexers=indexers, method=method, tolerance=tolerance)
        cov = self.cov.sel(**kwargs, **indexers_kwargs)
        if self.mc is not None:
            mc = mc = self.mc.sel(**kwargs, **indexers_kwargs)
        else:
            mc = None

        out = RMEMeas(self.name, cov, mc, self.covdofs, self.covcats)
        return out

    def isel(
        self,
        indexers: dict = None,
        drop: bool = False,
        missing_dims: str = 'raise',
        **indexers_kwargs,
    ) -> 'RMEMeas':
        """
        Get view into underlying data with integer based selection.

        Ignores the umech_id dimension, it cannot be indexed into
        with this function.Index into array assuming the umech_id
        dimensions doesn't exist in cov (i.e. as if indexing into only the
        nominal)

        Generates a view on the underlying cov,mc, covcats
        and covdofs array. Use .copy() to create an unconnected version.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by integers,
            slice objects or arrays. indexer can be a integer, slice,
            array-like or DataArray. If DataArrays are passed as indexers,
            xarray-style indexing will be carried out.One of indexers or
            indexers_kwargs must be provided.. The default is None.
        drop : bool, optional
            drop coordinates variables indexed by integers instead of making
            them scalar. The default is False.
        missing_dims : str, optional
           What to do if dimensions that should be selected from are not
           present in the DataArray: - “raise”: raise an exception - “warn”:
           raise a warning, and ignore the missing dimensions - “ignore”:
           ignore the missing dimensions. The default is 'raise'.
        **indexers_kwargs : TYPE
           The keyword arguments form of indexers..

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        out : RMEMeas
            View into indexed RMEMeas object.

        """
        forbid = ['umech_id', 'sample_id']
        for f in forbid:
            if indexers_kwargs and f in indexers_kwargs:
                raise ValueError(f' Cannot index into {f} here')
            if indexers and f in indexers:
                raise ValueError(f' Cannot index into {f} here')

        kwargs = dict(indexers=indexers, drop=drop, missing_dims=missing_dims)

        cov = self.cov.isel(**kwargs, **indexers_kwargs)
        if self.mc is not None:
            mc = self.mc.isel(**kwargs, **indexers_kwargs)
        else:
            mc = None
        out = RMEMeas(self.name, cov, mc, self.covdofs, self.covcats)
        return out

    def polyfit(self, dim: str, deg: int, apply_cov: bool = True) -> 'RMEMeas':
        """
        Apply a polynominal fit along dim.

        Parameters
        ----------
        dim : str
            Name of dimension to fit along.
        deg : int
            Degree of fit
        aoo

        Returns
        -------
        RMEMeas
            Coefficients of data.
        """

        def fit(arr):
            fit = arr.polyfit(dim, deg, cov=False)
            coeffs = fit.polyfit_coefficients
            coeffs = coeffs.transpose(..., 'degree')
            return coeffs

        cov = fit(self.cov)
        if self.mc is not None:
            mc = fit(self.mc)
        else:
            mc = None
        return RMEMeas(self.name, cov, mc, self.covdofs.copy(), self.covcats.copy())

    def polyval(self, coord: xr.DataArray, degree_dim: str) -> 'RMEMeas':
        """
        Evaluate a polynomial fit along coord.

        Assumes self is a measurement of fit-parameters, with the
        dimension of polynomial fits along degree_dim.

        Parameters
        ----------
        coord : xr.DataArray
            Coordinate to fit along.
        degree_dim : str
            Polynomial degree dimension.

        Returns
        -------
        RMEMeas
            Coefficients of data.
        """
        covout = xr.polyval(coord, self.cov, degree_dim=degree_dim)
        covout = covout.transpose(..., *coord.dims)
        if self.mc is not None:
            mc = xr.polyval(coord, self.mc, degree_dim=degree_dim)
            mc = mc.transpose(..., *coord.dims)
        else:
            mc = None
        return RMEMeas(self.name, covout, mc, self.covdofs.copy(), self.covcats.copy())

    def curvefit(
        self,
        coords: str | xr.DataArray | Sequence[str] | Sequence[xr.DataArray],
        func: callable,
        reduce_dims: str | Iterable | None = None,
        skipna: bool = True,
        p0: dict = None,
        bounds: dict = None,
        param_names: Sequence | None = None,
        errors: str = 'raise',
        **kwargs,
    ):
        """
        Simple binding to curvefit on xarray.

        The nominal value is solved for first, than used as an initial guess for the
        perturbed datasets.

        See https://docs.xarray.dev/en/stable/generated/xarray.DataArray.curvefit.html for
        more details.

        Parameters
        ----------
        coords : str | xr.DataArray | Sequence[str] | Sequence[xr.DataArray], DataArray
            Independent coordinate(s) over which to perform the curve fitting.
            Must share at least one dimension with the calling object.
            When fitting multi-dimensional functions,
            supply coords as a sequence in the same order as arguments in func.
            To fit along existing dimensions of the calling object, coords can also be specified as a str or sequence of strs.
        func : callable
           User specified function in the form f(x, *params) which returns a numpy array of length len(x).
           Params are the fittable parameters which are optimized by scipy curve_fit.
           x can also be specified as a sequence containing multiple coordinates, e.g. f((x0, x1), *params).
        reduce_dims: str | Iterable | None, optional
            Additional dimension(s) over which to aggregate while fitting. For example, calling ds.curvefit(coords=’time’, reduce_dims=[‘lat’, ‘lon’], …)
            will aggregate all lat and lon points and fit the specified function along the time dimension.
        skipna : bool, optional
            Whether to skip missing values when fitting. Default is True.
        p0 : dict, optional
            Optional dictionary of parameter names to initial guesses passed to the curve_fit p0 arg.
            If the values are DataArrays, they will be appropriately broadcast to the coordinates of the array.
            If none or only some parameters are passed, the rest will be assigned initial values following the default
            scipy behavior
        bounds : dict, optional
            Optional dictionary of parameter names to tuples of bounding values passed to the curve_fit bounds arg.
            If any of the bounds are DataArrays, they will be appropriately broadcast to the coordinates of the array.
            If none or only some parameters are passed, the rest will be unbounded following the default scipy behavior.
        param_names : Sequence | None, optional
            Sequence of names for the fittable parameters of func. If not supplied, this will be automatically determined
            by arguments of func. param_names should be manually supplied when fitting a function that takes a
            variable number of parameters.
        errors : str, optional
            If ‘raise’, any errors from the scipy.optimize_curve_fit optimization will raise an exception.
            If ‘ignore’, the coefficients and covariances for the coordinates where the fitting failed will be NaN.
        **kwargs : optional
            Additional keyword arguments passed to scipy curve_fit.
        """

        # param_names = param_names
        def fit(arr, nominal):
            # Fit the independent nominal first to generate an initial guess.
            coeffs = (
                nominal.curvefit(
                    coords,
                    func,
                    reduce_dims=reduce_dims,
                    skipna=skipna,
                    p0=p0,
                    bounds=bounds,
                    param_names=param_names,
                    errors=errors,
                    kwargs=kwargs,
                )
                .curvefit_coefficients
            )

            # refit to the rest of the data with an accurate initial guess
            fit_param_names = list(coeffs.param.values)
            newp0 = {str(k): coeffs.sel(param=k).values for k in fit_param_names}
            coeffs = arr.curvefit(
                coords,
                func,
                reduce_dims=reduce_dims,
                skipna=skipna,
                p0=newp0,
                bounds=bounds,
                param_names=param_names,
                errors=errors,
                kwargs=kwargs,
            ).curvefit_coefficients
            return coeffs

        nominal = self.nom
        cov = fit(self.cov, nominal)
        mc = None
        if self.mc is not None:
            mc = fit(self.mc, nominal)
        return RMEMeas(self.name, cov, mc, self.covdofs.copy(), self.covcats.copy())

    def curveval(
        self,
        func: callable,
        coords: xr.DataArray,
    ):
        """
        Evaluate the output of curvefit.

        Assumes there is a dimension called 'param' that coresponds.
        to the fit coefficients of func. Assumes function is nonlinear
        and iterates over each function evaluation, which can be slow.

        Parameters
        ----------
        func : callable
            callable function
        coords : xr.DataArray
            coordinates.
        """

        def eval(arr):
            output = None
            use_dim = 'umech_id'
            if use_dim not in arr.coords:
                use_dim = 'sample_id'
            for i, u in enumerate(arr.coords[use_dim]):
                coeffs = [arr.sel({use_dim: u, 'param': p}).values for p in arr.param]
                values = func(coords, *coeffs)
                # preallocate output
                if output is None:
                    output = (
                        xr.zeros_like(values)
                        .expand_dims({use_dim: arr.coords[use_dim]}, axis=0)
                        .copy()
                    )
                output[i, ...] = values
            return output

        cov = eval(self.cov)
        mc = None
        if self.mc is not None:
            mc = eval(self.mc)
        return RMEMeas(self.name, cov, mc, self.covdofs.copy(), self.covcats.copy())


# this took me a minute to figure out so leaving it here for when I need it
class _RMELocIndexer:
    """Index creator class for xarray like indexing on RMEMeas objects."""

    def __init__(self, other):
        self.obj = other

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        cov = self.obj.cov.loc[:, *slices]
        if self.obj.mc is not None:
            mc = self.obj.mc.loc[:, *slices]
        else:
            mc = None

        out = RMEMeas(self.obj.name, cov, mc, self.obj.covdofs, self.obj.covcats)

        return out
