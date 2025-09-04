"""
Store access global variables used by package.

TODO: Should be accesible via environment variables and/or
a local config file.,
"""

from pathlib import Path
from yaml import safe_load
from rmellipse.workflows._printtools import cprint, colors
from enum import Enum
from cdcs import CDCS
import json

__all__ = [
    "ProjectSettings", 
    "SCHEMAS", 
    "FILE_TYPES", 
    "WORKFLOW_SPECIAL_KEYS",
    "DIRECTORY_ITEMS"
    ]


WORKFLOW_EXT = ".rme.yml"

# glob up all the  schemas I want on import
SCHEMAS = {}
SCHEMA_DIR = Path(__file__).parents[0] / "schema"
for path in SCHEMA_DIR.glob("*.json"):
    name = path.relative_to(SCHEMA_DIR).name
    with open(path, "r") as f:
        SCHEMAS[name] = json.load(f)

# basefile names
RMEPROJ_FILENAME = "rmeproject.yml"

H5_FILE_EXTENSIONS = ['.h5','.hdf5']


# identifies type of items that exists within a directory-like structure
class MAPPING_META_KEYS(Enum):
    ATTRS  = '/attrs/'
    PATHSPEC = '/path/'
    ITEM = '/item/'
    FILE_TYPE_KEY = '/file-type/'
    BPID = '/bpid/'
    ARRSCHEMA = '/arrschema/'

class DIRECTORY_ITEMS(Enum):
    DIRECTORY = "Directory"
    FILE = "File"
    H5_FILE = "H5_File"
    H5_GROUP = "H5_Group"
    H5_DATASET = "H5_DATASET"

BLOBABLE = [DIRECTORY_ITEMS.FILE, DIRECTORY_ITEMS.H5_FILE]

# file patterns for file type enumerations
# for annotating directory contents
class FILE_TYPES(Enum):
    WORKFLOW = "workflow"
    CODE = "code"
    DOC = "doc"
    MISC = "misc"
    DATASET = "dataset"

FILE_TYPE_COLORS = {
    FILE_TYPES.WORKFLOW: colors.OKBLUE,
    FILE_TYPES.CODE: colors.OKCYAN,
    FILE_TYPES.MISC: None,
    FILE_TYPES.DATASET: colors.OKGREEN,
    FILE_TYPES.DOC: colors.HEADER,
}

DIRECTORY_ITEMS_COLORS = {
    DIRECTORY_ITEMS.DIRECTORY: colors.UNDERLINE+colors.OKGREEN,
    DIRECTORY_ITEMS.FILE: colors.OKGREEN,
    DIRECTORY_ITEMS.H5_DATASET: colors.OKCYAN,
    DIRECTORY_ITEMS.H5_FILE: colors.UNDERLINE+colors.OKCYAN,
    DIRECTORY_ITEMS.H5_GROUP: colors.UNDERLINE+colors.OKCYAN,
}

for e in FILE_TYPES:
    assert e in FILE_TYPE_COLORS


# Job key is a place holder, could be anything
class WORKFLOW_SPECIAL_KEYS(Enum):
    JOB = 0
    CDCS_RELEASE = "cdcs-release"


class RELEASE_BUILD_FILENAMES(Enum):
    PROJECT_TREE = "rmeproject.json"
    WORKFLOW_TREE = "rmeworkflow.json"

                

class ProjectSettings(dict):
    """Store project settings in a dict-like object."""

    def __init__(self, PROJDIR: Path):
        dict.__init__(self)
        PROJDIR = Path(PROJDIR)
        self["PROJDIR"] = PROJDIR

        if not ProjectSettings.is_rmeproject(PROJDIR):
            raise Exception(f"{RMEPROJ_FILENAME} not found.")

        # read in the settings properties
        with open(self.rmereqfile, "r") as f:
            self.requirements = safe_load(f)

    def cdcssettings(self):
        with open(self.project_dir/'cdcs.yml', 'r') as f:
            d = safe_load(f)
        return d

    @property
    def project_dir(self):
        return self['PROJDIR']
    
    @property
    def project_map(self):
        return self.dotrmedir / 'project-map.json'

    @property
    def processlogdir(self):
        path = self.dotrmedir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dotrmedir(self):
        return self["PROJDIR"] / ".rme"

    @property
    def wft_jsondir(self):
        dir = self.dotrmedir / "workflow-solutions"
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    @property
    def build_dir(self):
        folder = self["PROJDIR"] / "cdcs-release"
        folder.mkdir(parents=True, exist_ok=True)
        ignore_file = folder / ".gitignore"
        with open(ignore_file, "w") as f:
            f.write("#made by rmellipse\n*")
        return folder

    @property
    def this_release_dir(self):
        project_name = self.requirements["name"]
        project_version = self.requirements["version"]
        build_name = f"{project_name}-{project_version}"
        build_dir = self.build_dir / build_name
        build_dir.mkdir(parents=True, exist_ok=True)
        return build_dir

    @property
    def rmereqfile(self):
        return self["PROJDIR"] / RMEPROJ_FILENAME

    @staticmethod
    def is_rmeproject(directory: Path):
        """True if directory is an RME project (i.e. has a rmeproject.yml file.)"""
        # search upwards to se
        has_rmeproj_file = (directory / RMEPROJ_FILENAME).exists()
        if has_rmeproj_file:
            return True
        return False


if __name__ == "__main__":
    print("cdcs-release" in WORKFLOW_SPECIAL_KEYS)
