"""
Store access global variables used by package.

TODO: Should be accesible via environment variables and/or
a local config file.,
"""

from pathlib import Path
from yaml import safe_load, safe_dump
from rmellipse.workflows._printtools import Colors
from enum import Enum
import json
import semver
import re
import os
import dotenv
import copy

__all__ = [
	'ProjectSettings',
	'SCHEMAS',
	'FILE_TYPES',
	'WORKFLOW_SPECIAL_KEYS',
	'DIRECTORY_ITEMS',
	'WorkflowConfig',
	'RMESettings',
]

# folder to store intermediate files
DOTRME_DIRNAME = '.rme'

WORKFLOW_EXT = '.rme.yml'

# where data for external workflows are stored
# and symbolically linked to the cache folder
DATA_FOLDER = 'data'

# config foler
USER_CONFIG_FOLDER = Path.home() / '.rmellipse'

# cache on machine where datasets are stored
CACHE_FOLDER = USER_CONFIG_FOLDER / 'cache'
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

# glob up all the  schemas I want on import
SCHEMAS = {}
SCHEMA_DIR = Path(__file__).parents[0] / 'schema'
for path in SCHEMA_DIR.glob('*.json'):
	name = path.relative_to(SCHEMA_DIR).name
	with open(path, 'r') as f:
		SCHEMAS[name] = json.load(f)

# basefile names
RMEPROJ_FILENAME = 'rmeproject.yml'
RMESETTINGS_FILENAME = 'rme.yml'

H5_FILE_EXTENSIONS = ['.h5', '.hdf5']

PROTECTED_RELEASE_TITLES = ['SELF']


# identifies type of items that exists within a directory-like structure
class MAPPING_META_KEYS(Enum):
	ATTRS = '/attrs/'
	PATHSPEC = '/path/'
	ITEM = '/item/'
	FILE_TYPE_KEY = '/file-type/'
	BPID = '/bpid/'
	BYTES = '/bytes/'
	ARRSCHEMA = '/arrschema/'


class DIRECTORY_ITEMS(Enum):
	DIRECTORY = 'Directory'
	FILE = 'File'
	H5_FILE = 'H5_File'
	H5_GROUP = 'H5_Group'
	H5_DATASET = 'H5_DATASET'


BLOBABLE = [DIRECTORY_ITEMS.FILE, DIRECTORY_ITEMS.H5_FILE]


# file patterns for file type enumerations
# for annotating directory contents
class FILE_TYPES(Enum):
	WORKFLOW = 'workflow'
	CODE = 'code'
	DOC = 'doc'
	MISC = 'misc'
	DATASET = 'dataset'


FILE_TYPE_COLORS = {
	FILE_TYPES.WORKFLOW: Colors.OKBLUE,
	FILE_TYPES.CODE: Colors.OKCYAN,
	FILE_TYPES.MISC: None,
	FILE_TYPES.DATASET: Colors.OKGREEN,
	FILE_TYPES.DOC: Colors.HEADER,
}

DIRECTORY_ITEMS_COLORS = {
	DIRECTORY_ITEMS.DIRECTORY: Colors.UNDERLINE + Colors.OKGREEN,
	DIRECTORY_ITEMS.FILE: Colors.OKGREEN,
	DIRECTORY_ITEMS.H5_DATASET: Colors.OKCYAN,
	DIRECTORY_ITEMS.H5_FILE: Colors.UNDERLINE + Colors.OKCYAN,
	DIRECTORY_ITEMS.H5_GROUP: Colors.UNDERLINE + Colors.OKCYAN,
}

for e in FILE_TYPES:
	assert e in FILE_TYPE_COLORS


# Job key is a place holder, could be anything
class WORKFLOW_SPECIAL_KEYS(Enum):
	JOB = 'jobs'
	RELEASE = 'release'
	DATASETS = 'datasets'


class RELEASE_BUILD_FILENAMES(Enum):
	PROJECT_TREE = 'rmeproject.json'
	WORKFLOW_TREE = 'rmeworkflow.json'


class RMESettings(dict):
	"""
	Settings file for the RME program.


	rme.yml files are used for storing settings that
	can be defined at the user level. For example, what
	archives to use by default, and aliases for the
	archives.

	Settings files are overloaded by higher priority
	settings files.

	Discovery order (lowest to highest priority)
	User config -> project config


	Parameters
	----------
	dict : _type_
		_description_
	"""

	def __init__(self, project_dir: Path):
		dict.__init__(self)
		settings = {}
		# computer config files would go here If i add this

		# overwrite with user config settings
		# if it exists
		try:
			with open(USER_CONFIG_FOLDER / RMESETTINGS_FILENAME, 'r') as f:
				d = safe_load(f)
				for k in d:
					settings[k] = d[k]
		except FileNotFoundError:
			pass

		# overwrite with settings for local project
		try:
			with open(project_dir / RMESETTINGS_FILENAME, 'r') as f:
				d = safe_load(f)
				for k in d:
					settings[k] = d[k]
		except FileNotFoundError:
			pass

		# environment variables get set here if I add those
		# assign to myself
		for k, v in settings.items():
			self[k] = v

	def get_host_settings(self, host_name: str = None):
		"""
		Get the host settings by host name.

		User and password are returned as None
		if they aren't present in the settings.

		Does not attempt to get credentials based on host.

		Parameters
		----------
		host_name : str, optional
			_description_, by default None

		Returns
		-------
		host:
			str
		user:
			str
		password:
			str

		Raises
		------
		LookupError:
			If the host isn't in the settings.
		"""
		use_archive = None
		try:
			use_archive = self.hosts[host_name]
		except KeyError:
			use_archive = None
			for hname, hdict in self.hosts.items():
				if hdict['host'] == host_name:
					use_archive = hdict
					break

		if use_archive is None:
			raise LookupError(f'Couldnt find host {host_name}')
		try:
			user = use_archive['user']
		except KeyError:
			user = None
		try:
			password = use_archive['password']
		except KeyError:
			password = None

		return use_archive['host'], user, password

	@property
	def hosts(self):
		try:
			return self['archives']
		except KeyError:
			return {}

	@property
	def default_host(self):
		count = 0
		for k, v in self.items():
			if 'default' in v and v['default']:
				count += 1
				default_archive = v
		if count > 1 or count == 0:
			raise ValueError('1 default archive must be defined.')
		return default_archive


class WorkflowConfig(dict):
	"""
	Store and access a workflow configuration.

	Parameters
	----------
	dict : _type_
		_description_

	Returns
	-------
	_type_
		_description_

	Raises
	------
	Exception
		_description_
	"""

	def __init__(
		self, workflow_file: Path | str | dict, project_settings: 'ProjectSettings'
	):
		"""

		Parameters
		----------
		workflow_file : Path | str | dict
			Workflow file.
		project_settings : ProjectSettings
			Project settings that contain the
			workflow.

		Raises
		------
		ValueError
			Missing environment variables.
		ValueError
			Empry workflow file.
		"""
		dict.__init__(self)
		self.project_settings = project_settings
		if isinstance(workflow_file, Path) or isinstance(workflow_file, str):
			workflow_file = Path(workflow_file)
			if workflow_file.suffixes == []:
				workflow_file = workflow_file.with_suffix(WORKFLOW_EXT)
			self.path = workflow_file
			with open(workflow_file, 'r') as f:
				d = safe_load(f)

		else:
			d = workflow_file

		# make a shallow copy
		if d is None:
			raise ValueError(f'Loaded in NONE. Is the {str(self.path)} empty?')
		for k in d:
			self[k] = d[k]

		# store the raw string values
		self.raw = copy.copy(dict(self))

		# load in environment variables
		# load in project level env varaibles first
		# then overload with workflow level variables
		project_dotenv = project_settings.project_dir / '.env'
		workflow_dotenv = self.path.parent / '.env'
		evars = dict(dotenv.dotenv_values(project_dotenv))
		evars.update(dict(dotenv.dotenv_values(workflow_dotenv)))
		self.env_variables = evars

		# validate that requirements are satisfied
		for env_var in self.requires_env:
			value = self.env_variables[env_var]
			if value is None:
				raise ValueError(f'Required environment variable: {env_var} not found.')

		# recursively format strings to replace with dataset variables
		new = self.expand_variables({k: v for k, v in self.items()})
		for k in new:
			self[k] = new[k]

	def expand_variables(
		self,
		obj: str | dict | list,
		expand_env: bool = True,
		expand_datasets: bool = True,
	) -> str | dict | list:
		"""
		Expand variables of any object that can be serialized
		to JSON.

		Encodes obj to a JSON string, replaces any variable
		expressions, then de-encodes the stirng. Order of operations
		is encode -> env variables -> dataset names -> decode.

		JSON is a faster serialization format, so using it here instead
		of YAML.

		Parameters
		----------
		obj : str | dict | list
			_description_
		expand_env: bool, optional
			If True, expands environment variables. Default
			is true.
		expand_datasets: bool, optional
			If True, expands dataset variables. Default
			is True

		Returns
		-------
		object
			Same type as input with variables expanded.
		"""
		env_vars = self.requires_env
		datasets = self.datasets
		string = json.dumps(obj)

		# expand environment variables
		if expand_env:
			for evar in env_vars:
				value = self.env_variables[evar]
				if value is None:
					raise ValueError(
						f'Required environment variable: {evar} not found.'
					)
				string = string.replace('${' + evar + '}', value)

		# expand dataset pointers
		if expand_datasets:
			for dname, dpointer in datasets.items():
				string = string.replace('{' + dname + '}', dpointer)

		# reload yml
		expanded = json.loads(string)
		return expanded

	@staticmethod
	def _normalize_package_name(name):
		"""Normalize a package name, validate against ar regex expression."""
		normalized = re.sub(r'[-_.]+', '-', name).lower()
		pattern = r'^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])\Z'
		try:
			int(normalized[0])
			raise Exception(f"{name} can't start with a number.")
		except ValueError:
			pass
		if not re.match(pattern, normalized, re.IGNORECASE):
			raise ValueError(
				f"{name} isn't a valid title. Must contain only letters, numbers, -, or _ and must start with a letter."
			)
		if name in PROTECTED_RELEASE_TITLES:
			raise ValueError(f'{name} is a protected title')
		return normalized

	@property
	def title(self) -> str:
		return self._normalize_package_name(self.release['title'])

	@property
	def release_version(self) -> str:
		return str(semver.Version.parse(self.release['version']))

	@property
	def requires_files(self) -> dict:
		"""
		Get a list of required files

		Includes required files of the project, any
		name-space conflicts are overwritten with the
		workspace requires_files.

		Returns
		-------
		dict[str]
			dictionary of required files or folder
			patterns, keys are the new name of the
			datset within environment.
		"""
		# get the environment variable requirements
		requires_files = self.project_settings.requires_files
		try:
			wf_requires_files = self['requires-files']
		except KeyError:
			wf_requires_files = {}
		requires_files.update(wf_requires_files)
		return self.expand_variables(requires_files)

	@property
	def requires_releases(self) -> None:
		"""
		Releases requirements defined in the workflow file.

		Returns
		-------
		_type_
			_description_
		"""
		# get the environment variable requirements
		proj_requires_releases = self.project_settings.requires_releases
		try:
			wf_requires_releases = self['requires-releases']
		except KeyError:
			wf_requires_releases = []
		return list(set(proj_requires_releases + wf_requires_releases))

	@property
	def requires_env(self) -> None:
		"""
		Environment variable requirements.

		Returns
		-------
		_type_
			_description_
		"""

		# get the environment variable requirements
		proj_requires_env = self.project_settings.requires_env
		try:
			wf_requires_env = self['requires-env']
		except KeyError:
			wf_requires_env = []
		return list(set(proj_requires_env + wf_requires_env))

	@property
	def formatting_keys(self) -> dict:
		"""Dictionary of kwargs for string formating."""
		return self.datasets

	@property
	def datasets(self):
		"""
		List of defined datsets in the workflow config.

		Returns
		-------
		_type_
			_description_
		"""
		return self[WORKFLOW_SPECIAL_KEYS.DATASETS.value]

	@property
	def jobs(self):
		"""
		List of defined datsets in the workflow config.

		Returns
		-------
		_type_
			_description_
		"""
		return self[WORKFLOW_SPECIAL_KEYS.JOB.value]

	@property
	def release(self):
		"""
		Release related settings

		Returns
		-------
		_type_
			_description_
		"""
		return self[WORKFLOW_SPECIAL_KEYS.RELEASE.value]


class ProjectSettings(dict):
	"""
	Store and access project settings.

	Can be serialized to a JSON like file.
	"""

	def __init__(self, project: Path | str | dict):
		dict.__init__(self)
		if isinstance(project, Path) or isinstance(project, str):
			PROJDIR = Path(project)
			self['PROJDIR'] = project

			if not ProjectSettings.is_rmeproject(PROJDIR):
				raise Exception(f'{RMEPROJ_FILENAME} not found.')

			with open(self.rmeproj_file, 'r') as f:
				self['file_settings'] = safe_load(f)
		elif isinstance(project, dict):
			for k in project:
				self[k] = project[k]

	@property
	def requires_files(self) -> dict:
		"""
		Get a list of required files

		Returns
		-------
		dict[str]
			dictionary of required files or folder
			patterns, keys are the new name of the
			datset within environment.
		"""
		try:
			return self['requires-files']
		except (KeyError, TypeError):
			return {}

	# read in the settings properties
	@property
	def requires_releases(self):
		"""
		Releases required by a project.

		Returns
		-------
		list
			list of version expression requirements,
			empty if no requirements.
		"""
		try:
			return self['file_settings']['requires-releases']
		except (KeyError, TypeError):
			return []

	@property
	def requires_env(self):
		"""
		Releases required by a project.

		Returns
		-------
		list
			list of version expression requirements,
			empty if no requirements.
		"""
		try:
			return self['file_settings']['requires-env']
		except (KeyError, TypeError):
			return []

	@property
	def rmesettings(self) -> RMESettings:
		"""
		Get RMESettings for this project.

		Returns
		-------
		dict
			_description_
		"""
		return RMESettings(self.project_dir)

	@property
	def project_dir(self) -> Path:
		"""
		Root directory of the project.

		Returns
		-------
		Path
			Path to the projects root directory.
		"""
		return self['PROJDIR']

	@property
	def active_workflow(self) -> Path:
		"""
		Return information about the active workflow.

		Returns
		-------
		Path:
			Path to the active workflow file.
		"""
		try:
			with open(self.dotrmedir / 'active-workflow.json', 'r') as f:
				out = json.load(f)
			return out
		except FileNotFoundError:
			return None

	@active_workflow.setter
	def active_workflow(self, path: str):
		"""
		Set the active workflow for the data environment.

		Parameters
		----------
		path : str
			Path to the active workflow file.
		"""
		path = Path(path)
		with open(self.dotrmedir / 'active-workflow.json', 'w') as f:
			json.dump(path.as_posix(), f)

	@property
	def project_map(self):
		"""
		Returns a project map.

		Returns
		-------
		Path
			Path to the project map.
		"""
		return self.dotrmedir / 'project-map.json'

	@property
	def data_env_dir(self):
		"""
		Directory containing the dataest environment.
		"""
		out = self.project_dir / 'data_env'
		out.mkdir(parents=True, exist_ok=True)
		ignore_file = out / '.gitignore'
		with open(ignore_file, 'w') as f:
			f.write('#made by rmellipse\n*')
		return out

	@property
	def data_env_packages(self):
		"""
		Directory containing the full linked packages in data_env_dir.
		"""
		out = self.data_env_dir / '.packages'
		out.mkdir(parents=True, exist_ok=True)
		ignore_file = out / '.gitignore'
		with open(ignore_file, 'w') as f:
			f.write('#made by rmellipse\n*')
		return out

	@property
	def processlogdir(self) -> Path:
		"""
		Direcotry that stores process logs.
		"""
		path = self.dotrmedir / 'logs'
		path.mkdir(parents=True, exist_ok=True)
		return path

	@property
	def dotrmedir(self):
		"""
		Directory for storing hidden files within a project.

		Returns
		-------
		Path
		"""
		out = self['PROJDIR'] / DOTRME_DIRNAME
		if not out.exists():
			out.mkdir()
		ignore_file = out / '.gitignore'
		with open(ignore_file, 'w') as f:
			f.write('#made by rmellipse\n*')
		return out

	@property
	def wft_jsondir(self):
		"""
		Directory for storing workfow solution JSON files.

		Returns
		-------
		_type_
			_description_
		"""
		dir = self.dotrmedir / 'workflow-solutions'
		dir.mkdir(parents=True, exist_ok=True)
		return dir

	@property
	def rmeproj_file(self):
		"""
		Path to the project file within a directory.

		"""
		return self['PROJDIR'] / RMEPROJ_FILENAME

	@staticmethod
	def is_rmeproject(directory: Path):
		"""True if directory is an RME project (i.e. has a rmeproject.yml file.)"""
		# search upwards to se
		has_rmeproj_file = (directory / RMEPROJ_FILENAME).exists()
		if has_rmeproj_file:
			return True
		return False


if __name__ == '__main__':
	print('cdcs-release' in WORKFLOW_SPECIAL_KEYS)
