from rmellipse.utils import GroupSaveable, load_object


class RMEMeasCollection(GroupSaveable):
    def __init__(self, cov=None, covcats=None, covdofs=None, mc=None):
        """
        Initialize a RMEMeasCollection object.
        """
        self.add_child(key='cov', data=cov)
        self.add_child(key='covdofs', data=self.covdofs)
        self.add_child(key='covcats', data=self.covcats)
        self.add_child(key='mc', data=mc)
