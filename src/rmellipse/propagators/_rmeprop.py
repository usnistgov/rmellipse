"""
Microwave Uncertainty Framework (RME) propagators.Propagator Definition module.

This RME propagators.Propagator is used for propagating RMEMeas objects, and is based
on the xarray package.

Created on Tue Jun  4 14:04:58 2024

@author: dcg2
"""
# These need to be imported this way to delay access
# to the underlying classes to avoid a circular import error
import rmellipse.propagators as propagators
import rmellipse.uobjects as uobjs
import numpy as np
import xarray as xr
import time
import uuid
import warnings
import pandas as pd
from functools import wraps
from typing import Union


class RMEProp(propagators.Propagator):
    """
    Microwave Uncertainty Framework uncertainty propagators.Propagator.

    Stores perturbations to data sets and samples of a Monte Carlo distribution
    along a dimension called 'umech_id' for the cov and mc
    attributes respectively.

    This class is used to represent data sets with uncertainty, and the provided
    propagators.Propagator wrapper and combine function are used to propagate those
    uncertainties through arbitrary functions using first order linear
    sensitivity analysis, or monte carlo simulations. 

    The class structure and algorithms are designed to support vectorized
    operations, and label based indexing with numpy/xarray libraries in the 
    functions being propagated, enabling larger data sets/large numbers of
    uncertainties to be propagated efficiently without parallelization.

    The class also supports the automatic handling of data sets with a common
    grid/dimensions, like frequency points.

    """
    _active = None

    def __init__(self,
                 montecarlo_sims: int = 0,
                 sensitivity: bool = False,
                 handle_common_grid_method: str = None,
                 common_grid: str = 'frequency',
                 common_coords: dict = {},
                 interp_kwargs: dict = {},
                 verbose: bool = False,
                 vectorize: bool = True,
                 set_active: bool = True,
                 ):
        """
        Creates a RMEMeas propagators.Propagator initialized with the defined settings.
        Keyword arguments are initialized into a settings dictionary that 
        can be modified on runtime.

        Parameters
        ----------
        montecarlo_sims : int, optional
            How many Monte Carlo trials to run. 0 turns off. The default is 0.
        sensitivity : bool, optional
            If true, performs a sensitivity analysis, linear first order
            via finite differences. The default is False.
        handle_common_grid_method : str, optional
            How to select common dimensions on RMEMeas inputs, done automatically
            by propagated functions. See
            RMEMeas.handle_common_grid for more info. None turns off.
        common_grid : str, optional
            Name of the common dimension to handle. The default is 'frequency'.
        common_coords : dict, optional
            Coordinates to pair RMEMeas inputs down to. Used for certain
            handle_common_grid_method values.See
            RMEMeas.handle_common_grid The default is {}.
        verbose : bool, optional
            IF true, propagators.Propagator prints information about operations
            as they happen. The default is False.
        vectorize: bool, optional;
            IF true, propagators.Propagator will loop over uncertainty mechanisms and
            repeatedly call the propagating function. The default is False.
        set_active : bool, optional
            Sets this as the active propagators.Propagator, used for some magic
            methods that need to infer what propagators.Propagator to use.
            The default is True.

        """
        self.settings = dict(montecarlo_sims=montecarlo_sims,
                        sensitivity=sensitivity,
                        handle_common_grid_method=handle_common_grid_method,
                        common_grid=common_grid,
                        common_coords=common_coords,
                        vectorize=vectorize,
                        verbose=verbose,
                        interp_kwargs = interp_kwargs
                        )
        """dict: Stores the current settings of the propagator."""

        super().__init__(self.settings)
        if set_active:
            RMEProp._active = self

    @staticmethod
    def _get_unique_umech_id(
            *process_args: tuple[type],
            **process_kwargs: dict[type]) -> list:
        """
        Get a list of all the unique umech_id in process_args,kwargs.

        Parameters
        ----------
        *process_args : tuple[type]
            tuple of any function arguments.
        **process_kwargs : dict[type]
            dictionary of any function key worded arguments.

        Returns
        -------
        list
            List of unique parameters.

        """
        args_param_lists = [m.umech_id if uobjs.RMEMeas._if_quacks(m) else [] for m in process_args]
        kwargs_param_lists = [m.umech_id if uobjs.RMEMeas._if_quacks(m) else [] for m in process_kwargs.values()]
        param_lists = args_param_lists + kwargs_param_lists

        # get a set of all the parameter locations
        param_set = []
        for pl in param_lists:
            param_set += list(pl)
        param_set = list(set(param_set))
        return param_set

    @staticmethod
    def _expand_umech_id_and_fill_nominal(
            m: type,
            new_params: list[str],
            xarray_skip_checks: bool = False,
            include_nominal_index: bool = True,
            nominal_fill_value: type = None
    ):
        """
        Expand the parameter locations in m to match param_set.

        Parameter locations in new_params that don't exist in m are filled with
        nominal values.

        Parameters
        ----------
        m : type
            can be anything. None RMEMeas objects are passed through without
            operation, RMEMeas objects are operated on.
        new_params: list[str]
            new parameter locations to expand to.
        xarray_skip_checks: bool, optional
            If true, assumes input is an xarray with 'umech_id'
            dimensions and immediately runs the function on it.
        include_nominal_index: bool, optional
            If true, includes 'nominal' as the first index  of the output
            xarray.
        nominal_fill_value: type, optional
            if provided, will be used as the fill value when dimensions are
            expanded. If not provided, will use the first index of the array
            as the fill value. 

        Returns
        -------
        data : TYPE
            m with expanded parameter locations, filled with the nominal value

        """

        # xarray based RMEMeas objects get handled
        if uobjs.RMEMeas._if_quacks(m) or xarray_skip_checks:
            param_set = new_params
            # list of parameter locations
            if include_nominal_index:
                new_paramlocs = np.array(['nominal'] + param_set)
            else:
                new_paramlocs = param_set

            if uobjs.RMEMeas._if_quacks(m):
                temp = m.cov
            else:
                temp = m

            if nominal_fill_value is None:
                nominal_fill_value = temp[0, ...]
            # make a copy of nominal equal to length of parameter set

            data = nominal_fill_value.expand_dims({'umech_id': new_paramlocs}).copy()
            # assign umech_id locations to original cov data
            intersections, comm1, comm2 = np.intersect1d(new_paramlocs,
                                                         temp.umech_id,
                                                         assume_unique=True,
                                                         return_indices=True)

            data[comm1, ...] = temp[comm2, ...]

            return data
        else:
            return m

    @staticmethod
    def _sample_distribution(montecarlo_trials: int,
                             m):
        """Sample the Monte Carlo distribution of an object.

        If m is an RMEMeas object, it samples the mc value and returns
        it the new, resampled xarray object

        If it is not an RMEMeas object, it just returns whatever was passed.

        Parameters
        ----------
        montecarlo_trials : int
            Number of samples.
        m : object
            Thing to sample the distribution of.

        Returns
        -------
        object
            xarray if m is an RMEMeas, type of m otherwise.
        """
        # of
        if uobjs.RMEMeas._if_quacks(m) and m.mc is not None:
            # otherwise randomly sample the distribution
            distlength = len(m.mc.coords['umech_id'])
            index = np.random.randint(1, distlength, (montecarlo_trials))
            index = np.append(0, index)
            d = m.mc.isel(umech_id=index)
            # reset sampling index
            d = d.assign_coords({'umech_id': np.arange(0, len(index))})
            return d
        # if its got no MC data, just sample the nominal over and over again
        elif uobjs.RMEMeas._if_quacks(m) and m.mc is None:
            d = m.cov.isel(umech_id=np.zeros(montecarlo_trials+1,dtype = int))
            # reset sampling index
            d = d.assign_coords({'umech_id': np.arange(montecarlo_trials+1)})
            return d
        else:
            return m

    @staticmethod
    def _run_linear(
            process_fcn: callable,
            process_args: tuple[type, 'uobjs.RMEMeas'],
            process_kwargs: dict[type, 'uobjs.RMEMeas'] = None,
            sensitivity_analysis: bool = False,
            vectorize: bool = True,
            verbose: str = False) -> 'uobjs.UObj':
        """
        Perform a linear sensitivity analysis on the process fcn.

        Parameters
        ----------
        process_fcn : callable
            Function being propagated.
        process_args : tuple[type, RMEMeas]
            Positional arguments to process_fcn.
        process_kwargs : dict[type, RMEMeas], optional
            Key word arguments to process_fcn. The default is None.
        sensitivity_analysis : bool, optional
            Whether or not to do a sensitivity analysis. The default is False.
        vectorize: bool, optional
            If true, vectorizes the function by passing all arguments at once.
        verbose : str, optional
            Print propagation information. The default is False.

        Returns
        -------
        xr.DataArray
            New covariance data.
        list
            list of unique uncertainty mechanisms.

        """
        cov_output = None
        param_set = None
        
        if process_kwargs is None:
            process_kwargs = {}
        
        if sensitivity_analysis:
            # get list of lists of all the umech_id
            param_set = RMEProp._get_unique_umech_id(*process_args, **process_kwargs)

            # expand RMEMeas arguments in the positional and keyword arguments
            # to a complete parameter locations grid, fill parameter locations
            # that don't exist in m with the nominal value.
            cov_args = [RMEProp._expand_umech_id_and_fill_nominal(m, param_set) for m in process_args]
            cov_kwargs = {k: RMEProp._expand_umech_id_and_fill_nominal(m, param_set) for k, m in process_kwargs.items()}

            # pass through the vectorized function
            if vectorize:
                cov_output = process_fcn(*cov_args, **cov_kwargs)
            else:
                covargsi = [a.sel(umech_id=['nominal']) if hasattr(a, 'umech_id') else a for a in cov_args]
                covkwi = {k: a.sel(umech_id=['nominal']) if hasattr(a, 'umech_id') else a for a, k in cov_kwargs.items()}
                cov_output = process_fcn(*covargsi, **covkwi)
                # print(cov_output)
                # if the output looks like an RMEMeas cov attr (xaray with umech_id attribute)
                if hasattr(cov_output, 'umech_id'):
                    for i, p in enumerate(param_set):
                        covargsi = [a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a in cov_args]
                        covkwi = {k: a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a, k in cov_kwargs.items()}
                        cov_output = xr.concat([cov_output, process_fcn(*covargsi, **covkwi)], 'umech_id')
                    cov_output = cov_output.assign_coords(umech_id=['nominal'] + param_set)
                # if the output is a tuple, and contains RMEMeas objects
                elif isinstance(cov_output,tuple) and any([hasattr(co, 'umech_id') for co in cov_output]):
                    for i, p in enumerate(param_set):
                        covargsi = [a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a in cov_args]
                        covkwi = {k: a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a, k in cov_kwargs.items()}
                        proc_out = process_fcn(*covargsi, **covkwi)
                        cov_output = [xr.concat([co, po], 'umech_id') if hasattr(po, 'umech_id') else co for co,po in zip(cov_output, proc_out)]
                    cov_output = tuple(cov_output)

            # new_nom = new_cov[0, ...]
        elif vectorize:
            cov_args = [m.cov[[0], ...] if uobjs.RMEMeas._if_quacks(m) else m for m in process_args]
            cov_kwargs = {k: m.cov[[0], ...] if uobjs.RMEMeas._if_quacks(m) else m for k, m in process_kwargs.items()}
            cov_output = process_fcn(*cov_args, **cov_kwargs)
        else:
            cov_args = [m.cov[0, ...] if uobjs.RMEMeas._if_quacks(m) else m for m in process_args]
            cov_kwargs = {k: m.cov[0, ...] if uobjs.RMEMeas._if_quacks(m) else m for k, m in process_kwargs.items()}
            cov_output = process_fcn(*cov_args, **cov_kwargs)
            def exp_nom(c): return c.expand_dims('umech_id', axis=0).assign_coords(umech_id=['nominal'])
            if type(cov_output) is tuple:
                cov_output = tuple([exp_nom(c) if type(c) is xr.DataArray else c
                              for c in cov_output])
            elif type(cov_output) is xr.DataArray:
                cov_output = exp_nom(cov_output)
        return cov_output, param_set

    @staticmethod
    def _run_montecarlo(
            process_fcn: callable,
            process_args: tuple[type, 'uobjs.RMEMeas'],
            process_kwargs: dict[type, ' uobjs.RMEMeas'] = None,
            montecarlo_trials: int = 0,
            vectorize: bool = True,
            verbose: str = False) -> 'uobjs.UObj':
        """
        Perform a montecarlo analysis on the process function.

        Parameters
        ----------
        name : str
            DESCRIPTION.
        process_fcn : callable
            DESCRIPTION.
        process_args : tuple[type, RMEMeas]
            DESCRIPTION.
        process_kwargs : dict[type, RMEMeas], optional
            DESCRIPTION. The default is None.
        montecarlo_trials : int, optional
            DESCRIPTION. The default is 0.
        vectorize: bool, optional
            If true, vectorizes the function by passing all arguments at once.
        verbose : str, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        xr.DataArray
            New montecarlo dataset.

        """
        mc_output = None
        RMEMeas_present = any([uobjs.RMEMeas._if_quacks(o) for o in list(process_args) + list(process_kwargs.items())])
        # if doing montecarlo AND a RMEMeas is present, run the algo
        
        if process_kwargs is None:
            process_kwargs = {}
        
        if montecarlo_trials and RMEMeas_present:

            mc_args = [RMEProp._sample_distribution(montecarlo_trials, m) for m in process_args]
            mc_kwargs = {k: RMEProp._sample_distribution(montecarlo_trials, m) for k, m in process_kwargs.items()}
            # if they aren't, just pass them through

            if vectorize:
                mc_output = process_fcn(*mc_args, **mc_kwargs)
            else:
                mcargsi = [a.sel(umech_id=0) if hasattr(a, 'umech_id') else a for a in mc_args]
                mckwi = {k: a.sel(umech_id=0) if hasattr(a, 'umech_id') else a for a, k in mc_kwargs.items()}
                mc_output = process_fcn(*mcargsi, **mckwi)
                if hasattr(mc_output, 'umech_id'):
                    for i in range(1, montecarlo_trials + 1):
                        mcargsi = [a.sel(umech_id=i) if hasattr(a, 'umech_id') else a for a in mc_args]
                        mckwi = {k: a.sel(umech_id=i) if hasattr(a, 'umech_id') else a for a, k in mc_kwargs.items()}
                        out = process_fcn(*mcargsi, **mckwi)
                        mc_output = xr.concat([mc_output, out], 'umech_id')
                elif isinstance(mc_output,tuple) and any([hasattr(co, 'umech_id') for co in mc_output]):
                    for i in range(1, montecarlo_trials + 1):
                        mcargsi = [a.sel(umech_id=i) if hasattr(a, 'umech_id') else a for a in mc_args]
                        mckwi = {k: a.sel(umech_id=i) if hasattr(a, 'umech_id') else a for a, k in mc_kwargs.items()}
                        proc_out = process_fcn(*mcargsi, **mckwi)
                        mc_output = [xr.concat([co, po], 'umech_id') if hasattr(po, 'umech_id') else co for co,po in zip(mc_output, proc_out)]
                    mc_output = tuple(mc_output)
                    
                #                 # if the output is an RMEMeas object
                # if uobjs.RMEMeas._if_quacks(cov_output):
                #     for i, p in enumerate(param_set):
                #         covargsi = [a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a in cov_args]
                #         covkwi = {k: a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a, k in cov_kwargs.items()}
                #         cov_output = xr.concat([cov_output, process_fcn(*covargsi, **covkwi)], 'umech_id')
                #     cov_output = cov_output.assign_coords(umech_id=['nominal'] + param_set)
                # # if the output is a tuple, and contains RMEMeas objects
                # elif isinstance(cov_output,tuple) and any([uobjs.RMEMeas._if_quacks(co) for co in cov_output]):
                #     for i, p in enumerate(param_set):
                #         covargsi = [a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a in cov_args]
                #         covkwi = {k: a.sel(umech_id=p) if hasattr(a, 'umech_id') else a for a, k in cov_kwargs.items()}
                #         proc_out = process_fcn(*covargsi, **covkwi)
                #         cov_output = [xr.concat([co, po], 'umech_id') if uobjs.RMEMeas._if_quacks(po) else co for co,po in zip(cov_output, proc_out)]
                #     cov_output = tuple(cov_output)

        # otherwise just run the function
        elif montecarlo_trials:
            mc_args = process_args
            mc_kwargs = process_kwargs

            # pass through the vectorized function
            mc_output = process_fcn(*mc_args, **mc_kwargs)

        return mc_output

        #

    @staticmethod
    def _repack_linear_and_mc_outputs(
            name: str,
            param_set: list,
            cov_output: xr.DataArray,
            mc_output: xr.DataArray,
            covcats: dict[set[str]] = None,
            covdofs: xr.DataArray = None,
            montecarlo_trials: int = 0,
            sensitivity: bool = False,
            verbose: str = False
    ):
        """
        Pack the output of run_linear and run_montecarlo back into RMEMeas objs.

        Parameters
        ----------
        name : str
            DESCRIPTION.
        param_set : list
            DESCRIPTION.
        cov_output : xr.DataArray
            DESCRIPTION.
        mc_output : xr.DataArray
            DESCRIPTION.
        covcats: dict[set[str]], optional
            Description. THe default is None.
        covdofs: xr.DataArray, optional
            Description, the default is None
        montecarlo_trials : int, optional
            DESCRIPTION. The default is 0.
        sensitivity : bool, optional
            DESCRIPTION. The default is False.
        verbose : str, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        RMEMeas
            RMEMeas object with propagated covariance and montecarlo data.

        """
        # lambda function so I don't have to write things out twice
        def handle_output(cov, mc):
            if not hasattr(cov, 'umech_id'):
                return cov

            # if it doesn't have a umech_id attributes
            # its not a RMEMeas so bounce it out
            return uobjs.RMEMeas(name=name, cov=cov, mc=mc, covcats=covcats, covdofs=covdofs)

        # handle everything in a tuple, for multi output functions
        # single outputs handle and return
        if type(cov_output) is tuple:
            if not montecarlo_trials:
                mc_output = [None] * len(cov_output)

            return tuple([handle_output(cov, mc) for cov, mc in zip(cov_output, mc_output)])
        else:
            return handle_output(cov_output, mc_output)

    @staticmethod
    def _get_new_categories(
        param_set: str,
        process_args: list,
        process_kwargs: dict,
        sensitivity: bool = False,
        verbose: bool = False
    ) -> xr.DataArray:
        """
        Intersect covcats of process arguments, and combine them accordingly.

        Raises warnings if two uncertainty mechanisms have the same name but different category info.

        Parameters
        ----------
        param_set : str
            list of unique umech_id.
        process_args : list
            positional arguments to process.
        process_kwargs : dict
            keyword arguments to process.
        sensitivity : bool, optional
            If sensitivity is active, by default False
        verbose : bool, optional
            If verbose, by default False

        Returns
        -------
        xr.DataArray
            _description_
        """

        if not sensitivity:
            return None
        # build list of all the covariance categories
        names = [a.name for a in process_args if uobjs.RMEMeas._if_quacks(a) and a.covcats is not None]
        names += [a.name for a in process_kwargs.values() if uobjs.RMEMeas._if_quacks(a) and a.covcats is not None]
        cats = [a.covcats for a in process_args if uobjs.RMEMeas._if_quacks(a) and a.covcats is not None]
        cats += [a.covcats for a in process_kwargs.values() if uobjs.RMEMeas._if_quacks(a) and a.covcats is not None]
        if len(cats) > 0:
            cats = xr.align(*cats, join='outer', fill_value='')
            newcats = cats[0].astype(object)
            # copy over indexes that are empty in the new one from the old ones
            # that are aligned
            msg = '\n'
            warn_collisions = False
            for c,name in zip(cats[1:],names[1:]):
                ind= newcats.values == ''
                ind_c = c.values == ''

                combined_ind = ind | ind_c
                ind_equal = newcats.values == c.values
                mismatch_ind = combined_ind | ind_equal


                if len(mismatch_ind) > 0:
                    if len(mismatch_ind[0]) > 0:
                        if not mismatch_ind.all():
                            # this is bad, we are going to loop over the bad ones and send a warning with info
                            # this warning could give better info
                            warn_collisions = True
                            mechanism_index = np.any(~mismatch_ind, axis=1)
                            newcats_bad = newcats[mechanism_index]
                            cats_bad = c[mechanism_index]
                            
                            for i in range(0, len(newcats_bad)):
                                badind = newcats_bad[i]!=cats_bad[i]
                                msg = "CATEGORY MISMATCH on "+str(newcats_bad[i][badind].umech_id.values)+"\n"
                                msg += 'meas name: '+ names[0] + '\n' + str(newcats_bad[i][badind].drop_vars("umech_id"))+'\n'
                                msg += 'meas name: '+name + '\n' + str(cats_bad[i][badind].drop_vars("umech_id"))+'\n'
                            

                newcats.values[ind] += c.values[ind].astype(object)
            # loop over and assign
            # since mechnisms with the same name are supposed to be identical,
            # don't bother combining the two
            if warn_collisions:
                warnings.warn(msg,stacklevel = 4)
            return newcats
        else:
            return None

    @staticmethod
    def _get_new_covdofs(
        param_set: list,
        process_args: list,
        process_kwargs: dict,
        sensitivity: bool = False,
        verbose: bool = False
    ):
        # make a dof array with dimensions(measurements,paramlocs)
        # if a measurement doesn't have a parameeter location, fill it with -1,

        kwargs = dict(
            xarray_skip_checks=True,
            include_nominal_index=False,
            nominal_fill_value=xr.DataArray(-1.0)
        )
        arr1 = [RMEProp._expand_umech_id_and_fill_nominal(m.covdofs, param_set, **kwargs)
                for m in process_args if uobjs.RMEMeas._if_quacks(m) and len(m.covdofs) > 0]
        arr2 = [RMEProp._expand_umech_id_and_fill_nominal(m.covdofs, param_set, **kwargs)
                for m in process_kwargs.values() if uobjs.RMEMeas._if_quacks(m) and len(m.covdofs) > 0]
        arr = np.array(arr1 + arr2)

        # pool dofs across the measurements dimensins
        if arr.size == 0:
            return None
        # only one measurement, dont need to check for matching DOFS
        elif arr.shape[0] == 1:
            pass
        # multiple measurmeents, check that matching mechanisms have the same DOF
        else:
            pass
            # if not all([all(arr[arr[:,i]>0,i] == arr[arr[:,i]>0,i][0]) for i in range(arr.shape[1])]):
            #     raise Exception('Encountered DOFS of identical uncertaintiy mechanisms that dont match.')

        arr[arr == -1] = np.inf
        covdofs = np.min(arr, axis=0)

        covdofs = xr.DataArray(
            covdofs,
            dims=('umech_id'),
            coords={'umech_id': param_set})

        return covdofs

    @staticmethod
    def _run_propagation_algrothm(
        process: callable,
        process_args: tuple[type],
        process_kwargs: dict[type],
        settings: dict,
    ) -> ('uobjs.RMEMeas', dict):
        """
        Run the propagation algorithm.

        Parameters
        ----------
        process : callable
            DESCRIPTION.
        process_args : tuple[type]
            DESCRIPTION.
        process_kwargs : dict[type]
            DESCRIPTION.
        settings : dict
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        (RMEMeas,dict)
            RMEMeas object and dictionary of metadata about propagation.

        """
        fun = process
        meta = None

        if settings['verbose']:
            t0 = time.time()
            meta = {}

        # run linear sensitivity analysis
        cov_out, param_set = RMEProp._run_linear(
            fun,
            list(process_args),
            process_kwargs,
            sensitivity_analysis=settings['sensitivity'],
            vectorize=settings['vectorize'],
            verbose=settings['verbose']
        )

        if settings['verbose']:
            meta['linear_runtime'] = time.time() - t0
            t0 = time.time()

        # run montecarlo sensitivty analysis
        mc_out = RMEProp._run_montecarlo(
            fun,
            list(process_args),
            process_kwargs,
            montecarlo_trials=settings['montecarlo_sims'],
            vectorize=settings['vectorize'],
            verbose=settings['verbose']
        )
        if settings['verbose']:
            meta['mc_runtime'] = time.time() - t0
            t0 = time.time()

        # re categorize
        covcats = None
        covdofs = None
        if settings['sensitivity']:
            covcats = RMEProp._get_new_categories(
                param_set,
                list(process_args),
                process_kwargs,
                sensitivity=settings['sensitivity'],
                verbose=settings['verbose']
            )
            # if None, then covcats didn't find any RMEObjects
            if covcats is not None:
                covcats = covcats.loc[param_set, ...]

                # put cov dofs onto a new grid
                covdofs = RMEProp._get_new_covdofs(
                    param_set,
                    list(process_args),
                    process_kwargs,
                    sensitivity=settings['sensitivity'],
                    verbose=settings['verbose']
                )

        if settings['verbose']:
            meta['recategorize_runtime'] = time.time() - t0
            t0 = time.time()

        # repack the outputs of linear and montecarlo
        out = RMEProp._repack_linear_and_mc_outputs(
            fun.__name__,
            param_set,
            cov_out,
            mc_out,
            covcats=covcats,
            covdofs=covdofs,
            montecarlo_trials=settings['montecarlo_sims'],
            sensitivity=settings['sensitivity'],
            verbose=settings['verbose']
        )

        if settings['verbose']:
            meta['repack_runtime'] = time.time() - t0
            meta['montecarlo trials'] = settings['montecarlo_sims']
            try:
                meta['n_params'] = len(param_set)
            except TypeError:
                meta['n_params'] = 0

        return out, meta

    def handle_common_grid(
            self,
            process_args: tuple,
            process_kwargs: dict,
            dim: str,
            handle_method: str) -> tuple[tuple, dict]:
        """
        Handle common grids on RMEMeas objects in process_args or process_kwargs.

        This function is called automatically by propagate to align and select
        common grid elements of RMEMeas objects so they are suitable for
        arithemetic and linear algebra.

        Parameters
        ----------
        process_args : tuple
            DESCRIPTION.
        process_kwargs : dict
            DESCRIPTION.
        dim : str
            Name of the dimensions being handled.
        handle_method : str
            Name of the handle method. Valid options are:
            "common","interp_smallest","interp_common"
            "common" will only use values along dim that are shared among ALL
            the inputs.

            "interp_common" will interpolate (1D) to the the provided frequency list
            in the common_coords dictionary of the propagators.Propagators settings. The
            common_coords settings is expected to be a dictionary of key value
            pairs with {dim:array} where dim is the name of the dimension and
            array is the 1d set of indexes.

        Raises
        ------
        Exception
            If a handle common grid method is provided that has not been defined.

        Returns
        -------
        tuple
            Modified positional arguments with common grid handled.
        dict
            Modified key worded arguments with common grid handled.

        """

        if handle_method is None:
            return process_args, process_kwargs

        acceptable = ["common", "interp_smallest", "interp_common"]

        if handle_method not in acceptable:
            raise Exception('handling method "' + handle_method + '" not implemented. Accepable methods are ' + str(acceptable))
        new_args = None
        new_kwargs = None

        if type(process_args) is not list and type(process_args) is not tuple:
            process_args = [process_args]

        if handle_method == "interp_common":
            # just to make it readable,
            # this calls the RMEMeas.interp function, which
            # is a fast method
            def interp(x):
                return x.interp(**self.settings['common_coords'], kwargs = self.settings['interp_kwargs'])

            new_args = [interp(a) if uobjs.RMEMeas._if_quacks(a) and hasattr(a.nom, dim) else a for a in process_args]
            new_kwargs = {k:interp(a) if uobjs.RMEMeas._if_quacks(a) and hasattr(a.nom, dim) else a for k, a in process_kwargs.items()}

        if handle_method == 'common':
            ils = [a.cov[dim] for a in process_args if uobjs.RMEMeas._if_quacks(a) and hasattr(a.nom, dim)]
            ils += [a.cov[dim] for k, a in process_kwargs.items() if uobjs.RMEMeas._if_quacks(a) and hasattr(a.nom, dim)]

            #this only keeps coordinates on dim for each 
            # RME object that are shared between all of them
            new_ind = xr.align(*ils, join = "inner")
            
            # convenience method to make the comprhension easier to read
            def sel(x): 
                return x.sel(**{dim: new_ind[0]})

            new_args = [sel(a) if uobjs.RMEMeas._if_quacks(a) and hasattr(a.nom, dim) else a for a in process_args]
            new_kwargs = {k: sel(a)if uobjs.RMEMeas._if_quacks(a) and hasattr(a.nom, dim) else a for k, a in process_kwargs.items()}

        return new_args, new_kwargs

    def propagate(self, fun):
        """
        Decorate to make function automatically pass itself through propagate.

        Assumes that all the RMEMeas arguments are passed as positional arguments.
        Any positional arguments that are not RMEMeas instances are turned
        into RMEMeas objects without any covariance or nominal data, and named
        'auto_arg'. The  __name__ property of the function is assigned as the name
        of the output RMEMeas object.

        Returns
        -------
        RMEMeas
            RMEMeas object of output.

        """
        @wraps(fun)
        def fun_with_propagation(*args, **kwargs):
            if self.settings['verbose']:
                title = 'Propagating: ' + fun.__name__
                underline = '-' * len(title)
                print('')
                print(title)
                print(underline)
                t0 = time.time()

            # add input info to workflow function
            # add propagators.Propagator function
            # add propagatur state
            # whatever

            # handle a common grid
            args_hg, kwargs_hg = self.handle_common_grid(
                args,
                kwargs,
                self.settings['common_grid'],
                self.settings['handle_common_grid_method']
            )

            if self.settings['verbose']:
                grid_runtime = time.time() - t0

                t0 = time.time()

            out, meta = RMEProp._run_propagation_algrothm(fun, args_hg, kwargs_hg, self.settings)

            if self.settings['verbose']:
                print('grid handling runtime' + ':' + str(grid_runtime) + ' sec')
                print('    linear mechanisms' + ':' + str(meta['n_params']))
                print('       linear runtime' + ':' + str(meta['linear_runtime']) + ' sec')
                print('    montecarlo trials' + ':' + str(meta['montecarlo trials']))
                print('           mc runtime' + ':' + str(meta['mc_runtime']) + ' sec')
                print(' recategorize runtime' + ':' + str(meta['recategorize_runtime']) + ' sec')
                print('       repack runtime' + ':' + str(meta['repack_runtime']) + ' sec')

            return out
        return fun_with_propagation

    @staticmethod
    def _montecarlo_combine(
            measurements: tuple,
            err: np.ndarray,
            montecarlo_trials: int = 0
    ):
        mc_avg = None
        if montecarlo_trials:
            resampled = [RMEProp._sample_distribution(montecarlo_trials, m) for m in measurements]
            stacked = xr.concat(resampled, dim='measurements')
            mc_avg = stacked.mean(dim='measurements')

            #
            weights = np.random.normal(size=(err.shape[0], montecarlo_trials))
            monte_typea = np.array([np.dot(err.T, weights[:, i]) for i in range(montecarlo_trials)]).T
            monte_typea = monte_typea.reshape(mc_avg[1:, ...].shape)
            mc_avg[1:, ...] += monte_typea
        return mc_avg

    @staticmethod
    def _generate_error_vectors(
        measurements: tuple,
        error_of_mean: bool = False,
        n_single_values: Union[float, int] = None,
        sensitivity: bool = False,
        combine_basename: str = 'combine'
    ):

        # average across measurements
        # measdim = 'measdim'

        # # this is just to make sure that the dimension name is unique
        # while measdim in measurements[0].dims:
        #     measdim += '0'
        # avg = xr.concat([m.nom.copy() for m in measurements], dim=measdim)
        # avg = avg.mean(dim=measdim)

        # # pca the different nominal values nominals
        # # doing this with the underlying numpy values

        # construct an array where rows represent measurmeents, column different data points
        nom_shape = measurements[0].nom.shape  # shape not including uncertainty mechanisms
        nominals = np.array([m.nom.values.flatten() for m in measurements])
        mu = np.mean(nominals, axis=0)

        # do pca
        u, sv, vt = np.linalg.svd(nominals - mu, full_matrices=False)

        # create error vectors
        n_or_one = 1
        if error_of_mean:
            n_or_one = nominals.shape[0]
        err = np.abs(np.diag(sv)) / np.sqrt(n_or_one * (nominals.shape[0] - 1)) @ vt

        # decide how many single values to add as uncertainty mechanisms
        if n_single_values is not None and n_single_values != 1:
            if n_single_values < 0:
                raise ValueError('n_single_values must be > 0')
            if n_single_values < 1:
                n_sv = np.where(np.cumsum(sv) / np.sum(sv) > n_single_values)[0][0] + 1
            else:
                n_sv = int(n_single_values)
            err = err[0:n_sv, :]
        type_a = (err + mu)

        # reshape to match original dimensions
        type_a = type_a.reshape([type_a.shape[0]] + list(nom_shape))

        # make new data array out of error mechanisms
        new_params = [combine_basename + '_' + str(i) for i in range(type_a.shape[0])]
        coords = dict(measurements[0].nom.coords)
        coords['umech_id'] = new_params
        type_a = xr.DataArray(
            type_a,
            coords=coords,
            dims=['umech_id'] + list(measurements[0].nom.dims)
        )

        return err, type_a

    @staticmethod
    def _linear_combine(
            measurements: tuple,
            type_a: xr.DataArray,
            sensitivity: bool = False,
    ):
        """
        Run the linear combine algorithm.

        Algorithm is base on PCA.

        Parameters
        ----------
        measurements : tuple
            Tuple of measurments being combined.
        sensitivity : bool, optional
            Whether or not sensitivity is being run. The default is False.
        type_a: xr.DataArray,
            DataArray of Type A uncertainties to be added to  covariance data.

        Raises
        ------
        Exception
            For bad parameters.

        Returns
        -------
        avgcov : RMEMeas
            Covariance data on combined measurment.

        """
        # allign uncertainty mechanisms
        # avgcov = None
        # if sensitivity:
        param_set = RMEProp._get_unique_umech_id(*measurements)
        measurements = [RMEProp._expand_umech_id_and_fill_nominal(m, param_set) for m in measurements]
        # average across measurements
        measdim = 'measdim'

        # this is just to make sure that the dimension name is unique
        while measdim in measurements[0].dims:
            measdim += '0'
        avg = xr.concat([m.copy() for m in measurements], dim=measdim)
        avg = avg.mean(dim=measdim)

        # concat type a data onto the average
        avgcov = xr.concat([avg, type_a], dim='umech_id')
        avgcov.attrs = measurements[0].attrs

        return avgcov

    def combine(self,
                *measurements: 'uobjs.RMEMeas',
                error_of_mean: bool = False,
                n_single_values: Union[float, int] = None,
                combine_basename: str = 'combined',
                add_uuid: bool = True,
                combine_categories: dict[str] = {'Type': 'A'}) -> 'uobjs.RMEMeas':
        """
        Combine repeated measurements with uncertainty into a single measurement.

        Additional uncertainty mechanisms are created with the 'combine_basename'
        as the name of the mechanisms + an iterated integer. Principal component
        analysis is used to create the additional mechanisms.

        Parameters
        ----------
        *measurements : RMEMeas
            DESCRIPTION.
        error_of_mean : bool, optional
            If true, uses the error of the mean when creating the new uncertainty
            mechanisms. The default is False.
        n_single_values : Union[float,int], optional
            Describes how many of the singular values to keep as
            error mechanisms when performing the PCA.If n_single_values<1,
            will provide the min number of values to describe n_single_values
            ratio of the total variance described by the SVD. If
            n_single_values> 1, will utilize the integer n_single_values number
            of singular values. If None, will use all the singular values
            available. Useful for reducing the size of data sets when
            large numbers of repeated measurements are used.The default is None.
        combine_basename : dict[str], optional
            Base name usd when creating new uncertainty mechanisms. 
            Uncertainty mechanisms are named with <basename>+_+<int>, int is
            iterated for each new mechanism. The default is 'combined'.
        add_uid: str, optional
            If true, adds a UID to the combine_basename to make it unique. The
            default is True.


        Returns
        -------
        out : RMEMeas
            Returns a RMEMeas object with combined uncertainties.

        """
        if len(measurements) == 1:
            return measurements[0]
        if len(measurements) < 1:
            raise ValueError('Expected at least 1 RMEMeas object')

        uid = ''
        if add_uuid:
            uid = str(uuid.uuid4())
        combine_basename += uid

        # put on a common grid
        t0 = time.time()
        measurements_hg, empty_kwargs = self.handle_common_grid(
            measurements,
            {},
            self.settings['common_grid'],
            self.settings['handle_common_grid_method']
        )

        grid_runtime = time.time() - t0
        t0 = time.time()

        # make kwarg dictionary
        generate_error_kwargs = {
            'error_of_mean': error_of_mean,
            'n_single_values': n_single_values,
            'sensitivity': self.settings['sensitivity'],
            'combine_basename': combine_basename,
        }

        # get PCA vectors and Type A uncertainties
        err, typeA = RMEProp._generate_error_vectors(
            measurements_hg,
            **generate_error_kwargs
        )

        avgcov = RMEProp._linear_combine(
            measurements_hg,
            typeA,
            sensitivity=self.settings['sensitivity'],
        )

        linear_runtime = time.time() - t0
        t0 = time.time()

        avgmc = RMEProp._montecarlo_combine(
            measurements_hg,
            err,
            montecarlo_trials=self.settings['montecarlo_sims'],
        )

        mc_runtime = time.time() - t0
        t0 = time.time()

        covcats = None
        newcovdofs = None
        if self.settings['sensitivity']:
            param_set = RMEProp._get_unique_umech_id(*measurements_hg)

            # get unique categories from the RMEMeas obect
            covcats = RMEProp._get_new_categories(
                param_set,
                measurements_hg,
                {},
                sensitivity=self.settings['sensitivity'],
                verbose=self.settings['verbose']
            )
            # assign the categories provided, tag with combine id
            # tag all the new type a mechanisms with a uuid
            combine_categories['combine_id'] = str(uuid.uuid4())
            new_mechs = [str(k) for k in typeA.umech_id.values]
            new_categories = {k: combine_categories for k in new_mechs}

            # add extra rows to categories for each new mechanism
            try:
                def empty_row(n): return xr.full_like(covcats[[0], ...], '').assign_coords({'umech_id': [n]})
                new_rows = xr.concat([empty_row(n) for n in new_mechs], dim='umech_id')
                covcats = xr.concat([covcats, new_rows], dim='umech_id')
                # add extra column for any new categories
                def empty_col(c): return xr.full_like(covcats[:, [0]], '').assign_coords({'categories': [c]})
                new_cols = [empty_col(c) for c in combine_categories if c not in covcats.categories]
                if len(new_cols) > 0:
                    new_cols = xr.concat(new_cols, dim='categories')
                    covcats = xr.concat([covcats, new_cols], dim='categories')
                # put on to the same alignment as the paramset
                covcats = covcats.loc[param_set + new_mechs]

                # add the new things
                for k, v in combine_categories.items():
                    covcats.loc[new_mechs, k] = v
            # no zeroth index on covcats means that it was empty to begin with,
            # combining mechanisms that didn't have any uncertainty mechanisms
            # initializing a covcats
            except IndexError:
                dims = ('umech_id', 'categories')
                coords = {'umech_id': new_mechs,
                          'categories': list(combine_categories.keys())}
                vals = np.zeros((len(new_mechs), len(combine_categories)), dtype=object)
                covcats = xr.DataArray(vals, dims=dims, coords=coords)
                for k, v in combine_categories.items():
                    covcats.loc[new_mechs, k] = v

            # add dof to new umech_id
            covdofs = RMEProp._get_new_covdofs(
                param_set,
                measurements_hg,
                {},
                sensitivity=self.settings['sensitivity'],
                verbose=self.settings['verbose']
            )
            combdof = len(measurements_hg) - 1
            nmechs = len(new_mechs)
            newcovdofs = xr.DataArray(
                np.ones(nmechs) * combdof,
                dims=('umech_id'),
                coords={'umech_id': new_mechs}
            )

            if covdofs is not None:
                newcovdofs = xr.concat([covdofs, newcovdofs], dim='umech_id')

        out = uobjs.RMEMeas(
            measurements_hg[0].name,
            avgcov,
            avgmc,
            covcats=covcats,
            covdofs=newcovdofs
        )

        if self.settings['verbose']:
            print('grid handling runtime' + ':' + str(grid_runtime) + ' sec')
            try:
                print('    linear mechanisms' + ':' + str(avgcov.shape[0]))
            except:
                print('    linear mechanisms' + ':' + str(0))
            print('       linear runtime' + ':' + str(linear_runtime) + ' sec')
            print('           mc runtime' + ':' + str(mc_runtime) + ' sec')

        return out