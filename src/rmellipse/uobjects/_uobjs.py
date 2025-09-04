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

    @classmethod
    @_abc.abstractmethod
    def copy(self):
        """
        Return a copy of a uncertain object.

        Returns
        -------
        None.

        """

    @property
    @_abc.abstractmethod
    def stdunc(self, k: float = 1):
        """
        Get the standard uncertainty with expansion factor k.

        Return type should be the natural typing for the object being
        evauluated.

        Parameters
        ----------
        k : float, optional
            Expansion factor The default is 1.

        Returns
        -------
        Standard uncertainty.

        """

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
