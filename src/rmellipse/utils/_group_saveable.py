"""
The saveable_types module provides tools for saving/reading python objects.

Every object saved to a group using this module will have (at least) the
following attributes:
    * "__class__.__module__": Name of the module that contains the class.
    * "__class__.__name__": Name of class (for intialization).
    * "save_type": Indicates how the data are stored.

The possible values for the save_type are:
    GROUP_SAVEABLE: Anything that inherits from the group_saveable class.
    DATASET_SAVEABLE: bool, int, float, str, numpy.ndarray, None
    LIST_SAVEABLE: list, set, tuple
    DICT_SAVEABLE: dict
    SLICE_SAVEABLE: slice

GROUP_SAVEABLE ojects also have the following attributes:
    * "name": the name of the object
    * "unique_id": used to prevent saving the same data multiple times

"""

import uuid
import sys
import importlib
import numpy
import xarray
from abc import ABC
from typing import Union, NewType, Callable  # , TypeVar , abstractmethod
import h5py

# define what is part of the exposed API
__all__ = ['load_object', 'save_object', 'GroupSaveable']


class GROUP_SAVEABLE(ABC):
    """Dummy class used for type checking."""

    def __init___(self):
        pass


NONETYPE = type(None)

# usually h5py._hl.dataset.Dataset
# TODO: eventually, we should define an interface for datasets
# I'm using a dummy definition here to avoid an explict dependence on h5py
DATASET = NewType("DATASET", object)

# usually, group are Union[h5py._hl.group.Group, h5py._hl.files.File]
# TODO: eventually, we should define an interface for groups
# I'm using a dummy definition here to avoid an explict dependence on h5py
GROUP = NewType("GROUP", object)

SAVED = Union[DATASET, GROUP]

# list-saveable are things that can be stored as or initialized from lists
# do not use lists when you can use a numpy array.
LIST_SAVEABLE = Union[list, set, tuple]

# dict-saveable are things that can be represented in python by dicts
DICT_SAVEABLE = dict

# dataset-saveable are things that can be represented by a dataset
DATASET_SAVEABLE = Union[bool, int, float, str, numpy.ndarray, NONETYPE] # bool, int, float, str,  don't work so I removed them for now (COLE) I don't think we use them anyways.

# slice_saveable are things that can be represented in python by a slice
SLICE_SAVEABLE = slice

# function_saveable are python functions
FUNCTION_SAVEABLE = Callable

# data array saveables are data arrays
DATAARRAY_SAVEABLE = xarray.DataArray

SAVEABLE = Union[GROUP_SAVEABLE, DATASET_SAVEABLE, LIST_SAVEABLE,
                 DICT_SAVEABLE, SLICE_SAVEABLE, FUNCTION_SAVEABLE, DATAARRAY_SAVEABLE]
# Human-readable keys should be used to index dictionaries wherever possible
# because they will lead to a nicer representation when a dict is saved to as a
# group.
#
# Human-readable keys
# 1) Can be losslessly converted to strings using str().
# 2) Are not ugly when converted to strings.
#
# For now, the only types I can think of that fit that description are int and
# str. But, I am leaving open the possibility to add new types.
HUMAN_READABLE_KEY = Union[int, str]
HUMAN_READABLE_KEY_TYPES = ["int", "str"]


def get_function(module_name: str, function_name: str) -> Callable:
    """
    Look up a function from a module name and funciton name.

    Parameters
    ----------
    module_name : str
        Can be determined by inspecting [function].__module__.

    function_name : str
        Can be determined by inspecting [function].__name__.

    Returns
    -------
    None.

    """
    try:
        module = sys.modules[module_name]

    except KeyError:
        module = importlib.import_module(module_name)

    function = getattr(module, function_name)

    return function


def get_class(module_name: str, class_name: str) -> type:
    """
    Look up a class from module name and class name.

    Used to find constructors for objects.

    Parameters
    ----------
    module_name : str
        Can be determined by inspecting [object].__class__.__module__.

    class_name : str
        Can be determined by inspecting [object].__class__.__name__.

    Returns
    -------
    o_class : TYPE
        The constructor for the object.

    """
    o_class = None
    if class_name == 'NoneType':  # weridly, NoneType is not a class
        o_class = NONETYPE

    elif class_name == "function":
        o_class = Callable

    else:
        try:
            module = sys.modules[module_name]

        except KeyError:
            module = importlib.import_module(module_name)

        o_class = getattr(module, class_name)

    return o_class


def save_object(group: GROUP, name: str, o: any, verbose: bool = False) -> SAVED:
    """
    Save an object to a group.

    Parameters
    ----------
    group : GROUP
        Group where object will be saved.

    name : str
        Name the object will have in the group.

    o : any
        Object to save.

    Returns
    -------
    SAVED
        The newly-created saved object.

    """
    # new_group = None
    if verbose:
        print("save_object", o)
    if isinstance(o, GROUP_SAVEABLE):
        new_group = o.save(group, name, verbose=verbose)

    elif isinstance(o, LIST_SAVEABLE):
        new_group = save_list_saveable(group, name, o, verbose=verbose)

    elif isinstance(o, DICT_SAVEABLE):
        new_group = save_dict_saveable(group, name, o, verbose=verbose)

    elif isinstance(o, DATASET_SAVEABLE):
        new_group = save_dataset_saveable(group, name, o, verbose=verbose)

    elif isinstance(o, SLICE_SAVEABLE):
        new_group = save_slice_saveable(group, name, o, verbose=verbose)

    elif isinstance(o, FUNCTION_SAVEABLE):
        new_group = save_function_saveable(group, name, o, verbose=verbose)

    elif isinstance(o, DATAARRAY_SAVEABLE):
        new_group = save_dataarray_saveable(group, name, o, verbose=verbose)

    return new_group


def save_list_saveable(group: GROUP, name: str, o: LIST_SAVEABLE, verbose: bool = False) -> GROUP:
    """
    Save list-like object to a group.

    Parameters
    ----------
    group : GROUP
        Group where object will be saved.

    name : str
        Name the object will have in the group.

    o : LIST_SAVEABLE
        Object to save.

    verbose: bool,
        If true, prints info. Default is false.

    Returns
    -------
    GROUP
        The newly-created group.
    """
    if verbose:
        print("saving a list", o)
    subgroup = group.require_group(name)
    subgroup.attrs["__class__.__module__"] = o.__class__.__module__
    subgroup.attrs["__class__.__name__"] = o.__class__.__name__
    subgroup.attrs["save_type"] = "LIST_SAVEABLE"
    for i, item in enumerate(o):
        if verbose:
            print(str(i), item)
        save_object(subgroup, str(i), item)

    return subgroup


def save_dict_saveable(group: GROUP, name: str, o: DICT_SAVEABLE, verbose: bool = False) -> GROUP:
    """
    Save dict-like object to a group.

    Parameters
    ----------
    group : GROUP
        Group where object will be saved.

    name : str
        Name the object will have in the group.

    o : DICT_SAVEABLE
        Object to save.

    Returns
    -------
    GROUP
        The newly-created group.

    """
    subgroup = group.require_group(name)
    subgroup.attrs["__class__.__module__"] = o.__class__.__module__
    subgroup.attrs["__class__.__name__"] = o.__class__.__name__
    subgroup.attrs["save_type"] = "DICT_SAVEABLE"

    # check type of keys
    items = o.items()
    key_class = "None"
    key_module = "None"
    for i, item in enumerate(items):
        key, value = item
        if i == 0:
            key_class = key.__class__.__name__
            key_module = key.__class__.__module__

        if key.__class__.__name__ != key_class:
            key_class = "None"
            key_module = "None"

    # print(o, o.keys(), key_class)
    subgroup.attrs["key.__class__.__name__"] = key_class
    subgroup.attrs["key.__class__.__module__"] = key_module

    if key_class in HUMAN_READABLE_KEY_TYPES:
        # print(o)
        # print(o.keys())
        for i, key in enumerate(o.keys()):
            save_object(subgroup, str(key), o[key])

    else:
        for i, item in enumerate(o.items()):
            save_list_saveable(subgroup, "item " + str(i), item)

    return subgroup


def save_dataset_saveable(parent: GROUP, name: str, o: DATASET_SAVEABLE, verbose: bool = False) -> DATASET:
    """
    Save dataset-like object to a group.

    Parameters
    ----------
    group : GROUP
        Group where object will be saved.

    name : str
        Name the object will have in the group.

    o : DATASET_SAVEABLE
        Object to save.

    Returns
    -------
    DATASET
        The newly-created dataset.

    """
    if verbose:
        print(o, type(o))
    if o is not None:
        # try to just assign the dataset
        try:
            parent[name] = o
        except TypeError:
            # if it fails it might be an array of strings, cast
            # strings to a known thing
            dt = h5py.special_dtype(vlen=str)
            try:
                parent.create_dataset(name, o.shape, dtype=o.dtype, data=o)
            except:
                parent.create_dataset(name, o.shape, dtype=dt, data=o.astype('O'))

    else:
        parent[name] = "None"

    dataset = parent[name]
    dataset.attrs["save_type"] = "DATASET_SAVEABLE"
    dataset.attrs["__class__.__module__"] = o.__class__.__module__
    dataset.attrs["__class__.__name__"] = o.__class__.__name__
    return dataset


def save_dataarray_saveable(
        parent: GROUP,
        name: str, o: DATAARRAY_SAVEABLE,
        verbose: bool = False) -> DATASET:
    """
    Save dataarray-like object to a group.

    Parameters
    ----------
    group : GROUP
        Group where object will be saved.

    name : str
        Name the object will have in the group.

    o : DATAARRAY_SAVEABLE
        Object to save.

    Returns
    -------
    DATASET
        The newly-created dataset.

    """
    if verbose:
        print(o, type(o))
    if o is not None:
        # make a new group to store under
        new_parent = parent.require_group(name)
        # save values with dimensions as an attribute
        _save_ndarry(new_parent, 'values', o)
        for i, d in enumerate(o.dims):
            new_parent['values'].attrs['dim' + str(i)] = d
        # save coordinates as datasets
        for k, v in dict(o.coords).items():
            _save_ndarry(new_parent, k, v)
            # data = numpy.array(v)
            # try:
            #     new_parent.create_dataset(k, v.shape, dtype=data.dtype, data=data)
            # except:
            #     dt = h5py.vlen_dtype(data.dtype)
            #     new_parent.create_dataset(k, v.shape, dtype=data.dtype, data=data)

    else:
        parent[name] = "None"

    dataset = parent[name]
    # copy over metadata
    for k,v in o.attrs.items():
        dataset.attrs[k] = v
    dataset.attrs["save_type"] = "DATAARRAY_SAVEABLE"
    dataset.attrs['is_big_object'] = True
    dataset.attrs["__class__.__module__"] = o.__class__.__module__
    dataset.attrs["__class__.__name__"] = o.__class__.__name__

    try:
        dataset.attrs["dataformat"] = o.dataformat
    except:
        AttributeError
    return dataset

def _save_ndarry(parent: GROUP, name: str, arr: numpy.ndarray) -> str:
    # attempt to save as the natural type first
    dtype = arr.dtype
    try:
        parent.create_dataset(name, arr.shape, dtype=arr.dtype, data=arr)
    except TypeError:
        # unicode string type,
        # use the upper bound as a fixed length
        if dtype.kind == 'U':
            dt = h5py.string_dtype(length = int(str(dtype).split('U')[-1]))
            parent.create_dataset(name, arr.shape, dtype=dt, data=arr.astype(dt))
        # otherwise save as a variable length
        else:
            dt = h5py.special_dtype(vlen=str)
            parent.create_dataset(name, arr.shape, dtype=dt, data=arr)
    # save the datatype as a string
    parent[name].attrs['dtype'] = str(dtype)


def save_slice_saveable(group: GROUP, name: str, o: SLICE_SAVEABLE, verbose: bool = False) -> GROUP:
    """
    Save slice-like object to a group.

    Parameters
    ----------
    group : GROUP
        Group where object will be saved.

    name : str
        Name the object will have in the group.

    o : SLICE_SAVEABLE
        Object to save.

    Returns
    -------
    GROUP
        The newly-created group.

    """
    subgroup = group.require_group(name)
    subgroup.attrs["__class__.__module__"] = o.__class__.__module__
    subgroup.attrs["__class__.__name__"] = o.__class__.__name__
    subgroup.attrs["save_type"] = "SLICE_SAVEABLE"
    save_dataset_saveable(subgroup, "start", o.start)
    save_dataset_saveable(subgroup, "stop", o.stop)
    save_dataset_saveable(subgroup, "step", o.step)
    return subgroup


def save_function_saveable(group: GROUP, name: str, o: SLICE_SAVEABLE, verbose: bool = True) -> GROUP:
    """
    Save function to a group.

    Parameters
    ----------
    group : GROUP
        Group where object will be saved.

    name : str
        Name the object will have in the group.

    o : FUNCTION_SAVEABLE
        Object to save.

    Returns
    -------
    GROUP
        The newly-created group.

    """
    subgroup = group.require_group(name)
    subgroup.attrs["__class__.__module__"] = o.__class__.__module__
    subgroup.attrs["__class__.__name__"] = o.__class__.__name__
    subgroup.attrs["__module__"] = o.__module__
    subgroup.attrs["__name__"] = o.__name__
    subgroup.attrs["save_type"] = "FUNCTION_SAVEABLE"
    return subgroup


def load_object(
        saved_object: Union[GROUP, DATASET],
        parent: GROUP_SAVEABLE = None,
        load_big_objects: bool = False,
        vlen_object_encoding: str = str) -> any:
    """
    Construct Python object from group or dataset.

    Parameters
    ----------
    saved_object : Union[GROUP, DATASET]
        Group or datset that contains Python object.

    parent : GROUP_SAVEABLE, optional
        Parent of this object (Python object). The default is None.

    load_big_objects : bool, optional
        If True, fully load all objects into memory. If False,
        only the attributes of big objects will be loaded. The default is False.

    vlen_object_encoding : str, optional
        Variable length byte objects (np.dtype('O')) are cast
        into this type when they are read into numpy arrays. The
        default is str.

    Returns
    -------
    any
        A Python object.


    """
    module_name = saved_object.attrs["__class__.__module__"]
    class_name = saved_object.attrs["__class__.__name__"]
    save_type = saved_object.attrs["save_type"]

    o_class = get_class(module_name, class_name)
    o = None

    if save_type == "GROUP_SAVEABLE":
        o = o_class.load(saved_object, parent=parent, load_big_objects=load_big_objects)

    if save_type == "LIST_SAVEABLE":
        o = load_list_saveable(saved_object, o_class, load_big_objects=load_big_objects)

    elif save_type == "DICT_SAVEABLE":
        o = load_dict_saveable(saved_object, o_class, load_big_objects=load_big_objects)

    elif save_type == "DATASET_SAVEABLE":
        o = load_dataset_saveable(saved_object, o_class, vlen_object_encoding=vlen_object_encoding)

    elif save_type == "SLICE_SAVEABLE":
        o = load_slice_saveable(saved_object, o_class)

    elif save_type == "FUNCTION_SAVEABLE":
        o = load_function_saveable(saved_object, o_class)

    elif save_type == "DATAARRAY_SAVEABLE":
        o = load_dataarray_saveable(saved_object, o_class, vlen_object_encoding=vlen_object_encoding)

    return o


def load_list_saveable(group: GROUP, group_class: type, load_big_objects: bool = False) -> LIST_SAVEABLE:
    """
    Initialize list-like Python object from group.

    Parameters
    ----------
    group : GROUP
        Group to initialize object from.

    group_class : type
        Python class to instansiate.

    load_big_objects : bool
        If true, load big objects into structure.

    Returns
    -------
    new_list: LIST_SAVEABLE
        List-like object initialized from group.

    """
    temp_list = []
    for i, key in enumerate(group.keys()):
        o = load_object(group[str(i)],load_big_objects=load_big_objects)
        temp_list.append(o)

    new_list = group_class(temp_list)
    return new_list


def load_dict_saveable(group: GROUP, group_class: type, load_big_objects: bool = False) -> DICT_SAVEABLE:
    """
    Initialize dict-like Python object from group.

    Parameters
    ----------
    group : GROUP
        Group to initialize object from.

    group_class : type
        Python class to instansiate.

    Returns
    -------
    new_dict: DICT_SAVEABLE
        dict-like object initialized from group.

    """
    return_dict = None
    key_class = None
    key_class_name = group.attrs["key.__class__.__name__"]
    key_module_name = group.attrs["key.__class__.__module__"]

    if key_class_name != "None":
        key_class = get_class(key_module_name, key_class_name)

    if key_class_name in HUMAN_READABLE_KEY_TYPES:
        temp_keys = []
        temp_values = []
        for key in group.keys():
            temp_keys.append(key_class(key))
            temp_values.append(load_object(group[key], load_big_objects = load_big_objects))
        return_dict = group_class(zip(temp_keys, temp_values))

    else:
        return_dict = {}
        for i, item_name in enumerate(group.keys()):
            item_i = load_object(group[item_name])
            key, value = item_i
            return_dict[key] = value

    return return_dict


def load_dataset_saveable(
        dataset: DATASET,
        dataset_class: type,
        vlen_object_encoding: type = str) -> DATASET_SAVEABLE:
    """
    Initialize dataset-like Python object from group.

    Parameters
    ----------
    dataset : DATASET
        Group to initialize object from.

    dataset_class : type
        Python class to instansiate.

    Returns
    -------
    DATASET_SAVEABLE
        datset-like object initialized from group.

    """
    o = None
    # print(dataset, dataset_class)
    if dataset_class is not NONETYPE and dataset_class is not numpy.ndarray:
        o = dataset_class(numpy.array(dataset).astype(dataset_class))

    elif dataset_class is numpy.ndarray:
        o = _load_ndarray(dataset, vlen_object_encoding=str)
    return o


def load_dataarray_saveable(data_set, dataset_class: type, vlen_object_encoding:type=str) -> DATAARRAY_SAVEABLE:
    """
    Initialize dataset-like Python object from group.

    Parameters
    ----------
    dataset : DATASET
        Group to initialize object from.

    dataset_class : type
        Python class to instansiate.

    Returns
    -------
    DATASET_SAVEABLE
        datset-like object initialized from group.

    """
    o = None
    if data_set is not NONETYPE:
        vals = _load_ndarray(data_set['values'], vlen_object_encoding=vlen_object_encoding)
        getting_dims = True
        dims = []
        i = 0
        while getting_dims:
            try:
                dims.append(data_set['values'].attrs['dim' + str(i)])
                i += 1
            except KeyError:
                getting_dims = False
        coords = {}
        for k in dims:
            coords[k] = _load_ndarray(data_set[k], vlen_object_encoding=str)
        o = dataset_class(vals, dims=dims, coords=coords)
        for k in data_set.attrs:
            o.attrs[k] = data_set.attrs[k]
    return o

def _load_ndarray(h5dataset, vlen_object_encoding: numpy.dtype = str) -> numpy.ndarray:
    """
    Load an ndarray with correct type handling.

    vlen_object_encoding is used when encountering
    'O' data types, which are any variable length amount
    of bytes. Usually variable length strings (which is the default) but
    can be supplied if they represent some other object.
    """

    data = numpy.array(h5dataset)
    # if the dtype was defined
    # use that
    try:
        recast_as_dtype = h5dataset.attrs['dtype']
        recast = True
    # otherwise use whatever encoding
    # HDF5 hung onto
    except KeyError:
        dtype = data.dtype
        # 'O' could be any variable length byte string
        # typically strings, but the user needs to specify
        # what this is supposed to be or it will be
        # cast as a 'str' by default
        if dtype == 'O':
            recast_as_dtype = vlen_object_encoding
            recast = True
        else:
            recast = False
    if recast:
        data = data.astype(recast_as_dtype)
    return data

def load_slice_saveable(group: GROUP, group_class: type) -> SLICE_SAVEABLE:
    """
    Initialize slice object from group.

    Parameters
    ----------
    group : GROUP
        Group to initialize object from.

    group_class : type
        Python class to instansiate.

    Returns
    -------
    SLICE_SAVEABLE
        slice object initialized from group.

    """
    start = load_object(group["start"])
    stop = load_object(group["stop"])
    step = load_object(group["step"])
    return slice(start, stop, step)


# TODO: is there a difference between functions and callable classes?
def load_function_saveable(group: GROUP, group_class: type) -> FUNCTION_SAVEABLE:
    """
    Initialize a function Python object from a group.

    Parameters
    ----------
    group : GROUP
        Group to initialize object from.

    group_class : type
        Python class to instansiate.

    Returns
    -------
    FUNCTION_SAVEABLE
        Function-like object initialized from group

    """
    function_name = group.attrs["__name__"]
    module_name = group.attrs["__module__"]
    function = get_function(module_name, function_name)
    return function


class GroupSaveable(GROUP_SAVEABLE):
    """
    Interface for objects that can be saved as HDF5 or Exdir groups or files.

    These objects are organized in a tree-like structure to avoid data
    duplication. Specifically, a group-saveable object is a node in a tree
    graph. It stores references to its children, and also to its parent.

    Each node also has a lookup table that stores the paths to data objects
    below it.

    Save strategy
    -------------
        * Group saveable objects can be saved to groups and initialized from
        groups. There should be a 1-1 mapping of objects to groups.

        * After initialization, the group saveable object is independent
        from the goup that it was initialized from, and the group(s) it was
        saved to. So, changing the group saveable object does not change either
        the group it was initialized from, or the group(s) it was saved to.

    Recomendations for derived classes
    ----------------------------------
        * All attributes should be saveable types (see module description)

        * The names of all attributes match the keywords of contructor
        keyword arguments. Ex. if the object has an attribute called "foo",
        the constructor will take a keyword argument called "foo".

        * In the constructor, you use self.add_child to initialize object
        attributes. Big objects should be marked with is_big_object = True.

        * Any modules that define classes derived from group_saveable are
        in sys.path, so that import <module_name> works.

        * If you plan on saving an object attribute as a group attribute, add
        it to self.attrs.

        * For best performance, do not store any data in a LIST_SAVEABLE type
        (list, set, or tuple) if it can be stored in an array. Arrays are
        stored as datasets (efficient), while list-likes are stored in a custom
        format (inelegant, inefficient).

    Tree structure
    --------------
        * Group saveable objects have a unique id. If you know an object's id,
        you can retrieve the object from a tree.

        * Any group_saveable object can serve as the root of a tree. The only
        thing that makes the root special is that it has no parent.
        Consequently, roots can be assigned parents, and children can be
        detached from their parents.

        * Nodes can't store information about nodes that are not their
        children or their parent. Otherwise, we would have to define a root.

    """

    def __init__(self, name: str = None, parent: GROUP_SAVEABLE = None, attrs: dict = None, **kwargs):
        self.attrs = attrs

        if self.attrs is None:
            self.attrs = {}

        if "name" not in self.attrs.keys():
            self.attrs["name"] = str(name)

        if name is not None:
            self.attrs["name"] = str(name)

        if "__class__.__module__" not in self.attrs.keys():
            self.attrs["__class__.__module__"] = str(self.__class__.__module__)

        if "__class__.__name__" not in self.attrs.keys():
            self.attrs["__class__.__name__"] = str(self.__class__.__name__)

        if "save_type" not in self.attrs.keys():
            self.attrs["save_type"] = "GROUP_SAVEABLE"

        if "unique_id" not in self.attrs.keys():
            self.attrs["unique_id"] = uuid.uuid4().hex

        for key, value in self.attrs.items():
            self.attrs[key] = value

        self.parent = parent
        self.lookup_table = {}
        self.children = {}
        self.is_big_object = {}

    @classmethod
    def load(cls, group: GROUP, parent: GROUP_SAVEABLE = None, load_big_objects: bool = False) -> GROUP_SAVEABLE:
        """
        Initialize GROUP_SAVEABLE object from a group.

        The group_saveable class is designed to be used as an archive, and may
        store many large data sets. So, to save space in memory, some objects
        must be open explicitly using the load_big_objects argument.

        The attribute "is_big_object" determines if the object is
        fully loaded or not. If an object is not loaded, a placeholder with
        the same attributes will be added.

        Parameters
        ----------
        group : GROUP
            An hdf5 (or equivalent) group.

        parent : GROUP_SAVEABLE, optional
            The parent of this object. The default is None.

        load_big_objects : bool, optional
            If False, attributes marked as big objects are not loaded into
            memory. The default is False.

        Returns
        -------
        new_object : GROUP_SAVEABLE
            New data tree object loaded from group.

        """
        attrs = {}
        kwargs = {}
        kwargs["parent"] = parent
        for key in group.attrs.keys():
            attrs[key] = group.attrs[key]

        kwargs["attrs"] = attrs
        for key in group.keys():  # if it is not data, it is a child
            # print("in load", key)
            o = None
            saved_object = group[key]
            is_big_object = bool(saved_object.attrs["is_big_object"])
            if not is_big_object or load_big_objects:
                o = load_object(saved_object, parent, load_big_objects)

            kwargs[key] = o

        new_object = cls(**kwargs)

        return new_object

    def get_root(self) -> GROUP_SAVEABLE:
        """
        Get the root of the data tree.

        Returns
        -------
        GROUP_SAVEABLE
            The root (the tree with no parents).

        """
        root = self
        while root.parent is not None:
            if self.parent is not None:
                return self.parent.get_root()

        else:
            return self

    def look_up_node(self, unique_id: str) -> GROUP_SAVEABLE:
        """
        Find the node that holds an object by unique id.

        Parameters
        ----------
        unique_id : str
            The hex representation of the unique id of an object stored in a
            subtree.

        Returns
        -------
        GROUP_SAVEABLE
            The data tree that holds the object with that unique id.

        """
        return_path = None
        return_value = None

        if unique_id in self.lookup_table.keys():
            return_path = self.lookup_object(unique_id)

        elif self.parent is not None:
            return_path = self.parent.lookup_object(unique_id)

        if return_path is not None:
            return_value = self.recursive_getitem(return_path)

        return return_value

    def update_lookup_table(self, unique_id: str, path: str = "") -> bool:
        """
        Try to update lookup table with path to data stored in a subtree.

        This method will also recursively try to update the parent's lookup
        tables.

        If there is already an object with the same unique id in a
        parent's lookup table, then that is the real one, and this is a copy.
        In that case, do not upate the lookup table.

        Parameters
        ----------
        unique_id : str
            The hex representation of the unique id of an object stored in a
            subtree.

        path : str, optional
            Used for recursion. Do not change. The default is "".

        Returns
        -------
        bool
            True if object is successfully added (does not already exist in
            parents' lookup table)

        """
        name = self.attrs["name"]

        self.lookup_table[unique_id] = path
        if self.parent is not None:
            next_path = name + "/" + path
            self.parent.update_lookup_table(unique_id, next_path)

    def save(self, parent: GROUP, name: str = None, verbose: bool = False):
        """
        Save a group_saveable object as a group.

        Parameters
        ----------
        parent : GROUP
            Parent of group to be created.

        name : str, optional
            If not None, overwrite name attribute of the group. The default is
            None.

        verbose: bool, optional
            if True, prints information about what is being saved

        Returns
        -------
        None.

        """
        save_name = self.attrs["name"]
        if name is not None:
            save_name = name

        group = parent.require_group(save_name)
        group.attrs["name"] = save_name

        for key, value in self.attrs.items():
            # try:
            group.attrs[key] = value
            # except TypeError as e:
            #     print(e)

        #preallocate new object in case there are no children to save
        new_object = None
        for key in self.children.keys():
            if verbose:
                print("about to try to save", key, self)  # , self.children[key])
            if hasattr(self.children[key], "attrs"):
                self.children[key].attrs["is_big_object"] = self.is_big_object[key]
                new_object = save_object(group, key, self.children[key], verbose=verbose)
            else:
                new_object = save_object(group, key, self.children[key], verbose=verbose)
                new_object.attrs["is_big_object"] = self.is_big_object[key]
            if verbose:
                print("*" * 80)
                print(key, "attrs", new_object.attrs.keys())
                print("is_big_object", new_object.attrs["is_big_object"])
                print("*" * 80)
        return new_object

    def __str__(self):
        """Generate short, human-readable description."""
        num_subtrees = len(self.children.keys())
        return_value = f"link_tree with {num_subtrees} subtrees"

        if num_subtrees == 1:
            return_value = "link_tree with 1 subtree"

        return return_value

    def __getitem__(self, key: str):
        """
        Get a child object.

        This method works recursively.That means tree['a/b/c'] is equivalent
        to tree['a']['b']['c']

        Parameters
        ----------
        key : str:
            List of keys separated by "/"

        Returns
        -------
        None.

        """
        if isinstance(key, str):
            keys = key.split(",")

        return_value = self
        for key in keys:
            return_value = return_value.children[key]

        return return_value

    def add_child(self, key: str = None, data: SAVEABLE = None, is_big_object: bool = False):
        """
        Add child to this node.

        If the new data is a GROUP_SAVEABLE object, enforce that
        data.attrs["name"] == key.

        Parameters
        ----------
        key: str, optional
            Name of child. If None, check if the data has a name.
            The defualt is None.

        data : SAVEABLE, optional
            Data to add. If None, initialize an empty group_saveable object.
            The defualt is None.

        is_big_object: bool, optional
            If True, when this object is read from a file, it will
            be ignored if the load_big_objects argument is set to False.
            The default is False.

        Returns
        -------
        None.

        """
        data_key = key
        if data_key is None:
            if isinstance(data, GROUP_SAVEABLE):
                data_key = data.attrs["name"]
            else:
                raise ValueError("Attribute name could not be determined.")

        else:
            if isinstance(data, GROUP_SAVEABLE):
                if data.attrs["name"] == key:
                    data_key = key
                else:
                    raise ValueError("data.attrs['name'] does not match key.")

        self.children[data_key] = data
        self.is_big_object[data_key] = is_big_object
        setattr(self, key, self.children[data_key])

        if isinstance(self.children[data_key], GROUP_SAVEABLE):
            self.children[data_key].update_parents()

    def update_parents(self):
        """
        Recursively update the parents lookup tables all of this object's children.

        Returns
        -------
        None.

        """
        self.update_lookup_table(self.attrs["unique_id"])
        for key in self.children.keys():
            child = self.children[key]
            if isinstance(child, GROUP_SAVEABLE):
                child.update_parents()
