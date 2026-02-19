"""Module containing the UObj Class."""

import abc as _abc


class UObj(_abc.ABC):
    """
    Generic Uncertain Object.

    Defines the interface that all uncertainty objects should inherit from.
    """

    def __init__(self):
        self._is_uobj = True
        pass

    @property
    @_abc.abstractmethod
    def nom(self):
        """
        Get the nominal value of the uncertain object.

        Return type should be the natural typing for the object being
        evauluated.

        Returns
        -------
        Nominal value.

        """
