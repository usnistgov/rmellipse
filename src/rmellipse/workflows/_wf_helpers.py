import fnmatch
import h5py
from pathlib import Path
from enum import Enum
from typing import Callable
import rmellipse.workflows._globals as flowbals
from rmellipse.workflows._printtools import cprint
from rmellipse.arrschema._arrschema import SCHEMA_ATTRS_KEY


__all__ = [
    "matches_any",
    "make_globs",
    "walk_project",
    "iter_blobable"
    ]

def enum_matching_dispatch(item: Enum, enum: Enum, dispatch_pattern: dict[Callable]):
    """
    Used an item in enum to match to a dispatch pattern you've defined.

    Critically, requires that all the items in enum have a defined dispatch pattern.

    Parameters
    ----------
    item : Enum
        Item (instance of enum) to match against the dispatch pattern.
    enum : Enum
        Enumeration type to check dispatch pattern against.
    pattern : dict[Callable]
        dict of (fn, *args, **kwargs) with keys that match enum.
        All value of the enum must be present.
    """
    for e in enum:
        if e not in dispatch_pattern:
            raise ValueError("Missing key (e) for enum dispatch pattern")
    pattern = dispatch_pattern[enum(item)]
    try:
        args = pattern["args"]
    except KeyError:
        args = ()
    try:
        kwargs = pattern["kwargs"]
    except KeyError:
        kwargs = {}
    return pattern["fn"](*args, **kwargs)


def matches_any(pattern: str, globs: list[str]):
    """True if pattern matches any of globs"""
    return any([fnmatch.fnmatch(pattern, g) for g in globs])


def make_globs(directory: Path):
    """
    Make include and ignore glob patters from a directory

    Used .gitignore and .rmeinclude files in directory.

    Parameters
    ----------
    directory : Path
        Directory to look inside.

    Returns
    -------
    tuple[list[str]]
        tuple of list of glob patterns, (ignore, include)
    """
    ign_globs = []
    incl_globs = []
    try:
        with open(directory / ".gitignore", "r") as f:
            globs = [line.strip() for line in f]
            globs = [g for g in globs if len(g) > 0 ]
        ign_globs = [g for g in globs if g[0] != "#" and g[0] != "!"]
        incl_globs = [g for g in globs if g[0] == "!"]
    except FileNotFoundError:
        pass

    try:
        with open(directory / ".rmeinclude", "r") as f:
            globs = [line.strip() for line in f]
            globs = [g for g in globs if len(g) > 0 ]
        incl_globs += [g for g in globs if g[0] != "#"]
    except FileNotFoundError:
        pass
    return ign_globs, incl_globs


def map_hdf5(mapping: dict, group: h5py.Group, path: Path, ign_globs=None, incl_globs=None, root: Path = None):
    """
    Recursivley walk an HDF5 File.

    Parameters
    ----------
    group : h5py.Group
        _description_
    ign_globs : _type_, optional
        _description_, by default None
    incl_globs : _type_, optional
        _description_, by default None
    """
    new_ign = ign_globs
    new_incl = incl_globs
    if root is None:
        root = root
    for child_name in group:
        child_path = path / child_name
        child = group[child_name]
        if not matches_any(child_path, new_ign) or matches_any(child_path, new_incl):
            # groups get mapped
            mapping[child_name] = {}
            if SCHEMA_ATTRS_KEY in child.attrs:
                mapping[child_name][flowbals.MAPPING_META_KEYS.ARRSCHEMA.values] = (
                    child.attrs[SCHEMA_ATTRS_KEY]
                )
            if isinstance(child, h5py.Group):
                
                mapping[child_name][flowbals.MAPPING_META_KEYS.PATHSPEC.value] = (
                    child_path.relative_to(root).as_posix()
                )
                mapping[child_name][flowbals.MAPPING_META_KEYS.ATTRS.value] = dict(child.attrs)
                mapping[child_name][flowbals.MAPPING_META_KEYS.ITEM.value] = (
                    flowbals.DIRECTORY_ITEMS.H5_GROUP.value
                )
                map_hdf5(mapping[child_name], child, child_path, new_ign, new_incl, root = root)

            # datasets terminate
            elif isinstance(child, h5py.Dataset):
                mapping[child_name][flowbals.MAPPING_META_KEYS.PATHSPEC.value] = (
                    child_path.relative_to(root).as_posix()
                )
                mapping[child_name][flowbals.MAPPING_META_KEYS.ATTRS.value] = dict(child.attrs)
                mapping[child_name][flowbals.MAPPING_META_KEYS.ITEM.value] = (
                    flowbals.DIRECTORY_ITEMS.H5_DATASET.value
                )

            # if its not one of these, then something bad has happened.
            else:
                raise TypeError(f"type {type(child)} of {child_path} not recognized")




def map_directory(
    mapping: dict, directory: Path, ign_globs=None, incl_globs=None, root: Path = None
) -> Path:
    """
    Recursivley walk a directory and build up a nested mapping of it in JSON form.

    Checks for ignore and include files in each directory and adds
    to the list cumativley when entering a directory (in the
    same manner that gitignore works).

    Parameters
    ----------
    mapping: dict
        Dictionairy to start filling the mapping with. Typically
        begins as empty ({}).
    directory : Path
        Directory to start in.
    ign_globs : _type_, optional
        Starting glob patters, by default None
    incl_globs : _type_, optional
        Starting include patterns, by default None

    Yields
    ------
    Path
        Yields a Path. Does a depth first search.
    """
    if ign_globs is None:
        ign_globs = []
    if incl_globs is None:
        incl_globs = []
    new_ign, new_incl = make_globs(directory)
    new_ign += ign_globs
    new_incl += incl_globs

    if root is None:
        root = directory
    for p in directory.glob("*"):
        if not matches_any(p, new_ign) or matches_any(p, new_incl):
            # is it a directory?
            if p.is_dir():
                mapping[p.name] = {}
                mapping[p.name][flowbals.MAPPING_META_KEYS.PATHSPEC.value] = (
                    p.relative_to(root).as_posix()
                )
                mapping[p.name][flowbals.MAPPING_META_KEYS.ATTRS.value] = {}
                mapping[p.name][flowbals.MAPPING_META_KEYS.ITEM.value] = (
                    flowbals.DIRECTORY_ITEMS.DIRECTORY.value
                )
                map_directory(mapping[p.name], p, new_ign, new_incl, root = root)

            # hdf5 files get special treatement
            elif p.suffix in flowbals.H5_FILE_EXTENSIONS:
                with h5py.File(p, "r") as fopen:
                    mapping[p.name] = {}
                    mapping[p.name][flowbals.MAPPING_META_KEYS.PATHSPEC.value] = (
                        p.relative_to(root).as_posix()
                    )
                    mapping[p.name][flowbals.MAPPING_META_KEYS.ATTRS.value] = dict(
                        fopen.attrs
                    )
                    mapping[p.name][flowbals.MAPPING_META_KEYS.ITEM.value] = (
                        flowbals.DIRECTORY_ITEMS.H5_FILE.value
                    )
                    mapping[p.name][flowbals.MAPPING_META_KEYS.FILE_TYPE_KEY.value] = (
                        flowbals.FILE_TYPES.MISC.value
                    )
                    map_hdf5(mapping[p.name], fopen, p, new_ign, new_incl, root = root)

            # normal files terminate the mapping
            else:
                mapping[p.name] = {}
                mapping[p.name][flowbals.MAPPING_META_KEYS.PATHSPEC.value] = (
                    p.relative_to(root).as_posix()
                )
                mapping[p.name][flowbals.MAPPING_META_KEYS.ATTRS.value] = {}
                mapping[p.name][flowbals.MAPPING_META_KEYS.ITEM.value] = (
                    flowbals.DIRECTORY_ITEMS.FILE.value
                )
                mapping[p.name][flowbals.MAPPING_META_KEYS.FILE_TYPE_KEY.value] = (
                    flowbals.FILE_TYPES.MISC.value
                )

    return mapping

def iter_blobable(
        dmap: dict,
        ):
    """
    Iterate over a mapping and yield globable files.

    Parameters
    ----------
    dmap : dict
        Dictionary that maps a directory or item.

    Yields
    ------
    _type_
        _description_
    """
    for name, child in dmap.items():
        if name[0] != '/':
            child_type = flowbals.DIRECTORY_ITEMS(child[flowbals.MAPPING_META_KEYS.ITEM.value])
            if child_type not in flowbals.BLOBABLE:
                # If the value is a dictionary, recurse
                yield from iter_blobable(child)
            else:
                yield name, child

def show_map(
        map: dict,
        level: int = 0,
        show_only = ['*'],
        show_attrs: bool = True,
        show_blobid: bool = False
        ):
    for name, item in map.items():
        if name[0] != "/":
            path =  item[flowbals.MAPPING_META_KEYS.PATHSPEC.value]
            if matches_any(path, show_only):
                item_type = flowbals.DIRECTORY_ITEMS(item[flowbals.MAPPING_META_KEYS.ITEM.value])
                if level > 0:
                    print(f"{' ' * (level - 1)}|—", sep="", end="")
                cprint(name, color = flowbals.DIRECTORY_ITEMS_COLORS[item_type], end = "")
                if show_attrs:
                    print(item[flowbals.MAPPING_META_KEYS.ATTRS.value], end = "")
                if show_blobid:
                    print(item[flowbals.MAPPING_META_KEYS.BPID.value], end = "")
                print("")
                show_map(item, level=level + 2, show_attrs=show_attrs, show_only=show_only)
