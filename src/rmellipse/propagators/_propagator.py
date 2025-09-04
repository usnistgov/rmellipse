# -*- coding: utf-8 -*-
"""Module defininf the Propagator Class."""


import abc as _abc


class Propagator(_abc.ABC):
    """Defines what a Propagator should look like and how it should behave."""

    active = None

    def __init__(self, settings: dict):

        Propagator.active = self
        self.settings = settings

    def __repr__(self):
        """Make representation."""
        return self.__str__()

    def __str__(self):
        """Make str."""
        msg = type(self).__name__ + ' with settings: \n'
        msg += str(self.settings)
        return msg

    def set_active(self):
        """Make this propagator the active one."""
        Propagator.active = self

    @_abc.abstractmethod
    def propagate(self, *args, **kwargs):
        """
        Wrap a function in an uncertainty propagation engine.

        Parameters
        ----------
        function : Callable
            any function that should be wrapped to be propagateable.
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        Function wrapped in the propagation logic.

        """

    @_abc.abstractmethod
    def combine(self, *measurements):
        """
        Combine realizations of the same measurement into a single measurement.

        Adds uncertainty based on the inputs.

        Parameters
        ----------
        function : Callable
            any function that should be wrapped to be propagateable.
        *measurements: TYPE
            Realizations of the measurement.

        Returns
        -------
        combined measurement.

        """
        # TODO: add a super method for spying
        # Wokflow needs to be global (accessed from here), maybe singleton
