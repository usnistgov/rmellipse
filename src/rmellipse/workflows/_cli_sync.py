"""Syncronize your requirements file with your workflow environment."""

from yaml import safe_dump, safe_load
from pathlib import Path
from packaging.version import Version
from rmellipse.workflows._printtools import cprint, Colors, symbols
from rmellipse.workflows.archive_interface import get_interface, ArchiveInterface
from rmellipse.workflows.local_interfaces import EnvInterface, CacheInterface
import click
import rmellipse.workflows._settings as gv
import semver
import time
import dotenv
import os


@click.command(name='sync')
@click.argument('workflow-file', type=Path)
@click.argument('--default-host', type=Path)
@click.option('--force-download', is_flag=True, default=False)
@click.option('--force-relink', is_flag=True, default=False)
@click.option('--max-threads', type=click.IntRange(1, 100))
@click.option('--update-all', is_flag=True, default=False)
def sync_cli(*args, **kwargs):
	"""Synchronize the data environment with requirements."""
	sync(*args, **kwargs)


def sync(
	workflow_file: Path = None,
	project_dir: Path = Path.cwd(),
	default_host: Path = None,
	force_download: bool = False,
	force_relink: bool = False,
	max_threads: int = 50,
	download_chunk_bytes: int = 4194304,
	update_all: bool = False,
):
	"""
	Sync your workflow environment

	Parameters
	----------
	workflow_file: Path, optional
		If provided, includes requirements for the workflow
		in the environment syncronization.
	project_dir : Path, optional
	    Project direction. Default is the CWD.
	force_relink : bool
	    Force relink the data environment. Default is false.
	force_download : bool
	    Force re-download the data environment. Default is false.
	max_threads: int
		Maximum threads for processes. Default is 50
	download_chunk_bytes: int
		Maximum bytes for downloading chunks (per thread).
		Default is 2^22 (~4.2 Mb). Each file gets assigned a thread.
	update_all: bool
		If True, force updates all requirements. Default is False.
	"""

	project_settings = gv.ProjectSettings(project_dir)
	# figure out what indexes I can use
	rme_settings = project_settings.rmesettings
	if default_host is None:
		if 'archives' not in rme_settings:
			raise Exception(
				'No archives defined. Add a default archive to rme.yml in project or user settings.'
			)
		for k, v in rme_settings['archives'].items():
			if 'default' in v and v['default']:
				default_host = v
	else:
		default_host = {'host': default_host}
	wf_config = gv.WorkflowConfig(Path(project_dir) / workflow_file, project_settings)

	# get the set of release requirements from the
	# project file and the workflow
	requires_releases = wf_config.requires_releases

	# file system requirements
	requires_files = wf_config.requires_files

	# determine which releases should be installed in the environment,
	release_records, release_archives, workspaces = resolve_env(
		requires_releases, default_host, wf_config, project_settings, update_all
	)

	update_cache(
		release_records=release_records,
		release_archives=release_archives,
		workspaces=workspaces,
		force_download=force_download,
		max_threads=max_threads,
		download_chunk_bytes=download_chunk_bytes,
	)

	# build the data environment
	build_data_env(
		release_records=release_records,
		release_archives=release_archives,
		requires_files=requires_files,
		workspaces=workspaces,
		project_settings=project_settings,
		wf_config=wf_config,
		force_relink=force_relink,
	)

	# make a lock file
	lock_file = wf_config.path.with_suffix('.lock')
	lock = {}
	for versioned_release_name, record in release_records.items():
		lock[versioned_release_name] = {
			'name': record['title_versionless'],
			'version': record['version'],
			'host': str(release_archives[versioned_release_name].host),
			'workspace': workspaces[versioned_release_name],
		}

	lock.update(requires_files)

	# save a lock file of requirements
	with open(lock_file, 'w') as fio:
		safe_dump(lock, fio)

	# save the active workflow settings one every sync to .rme
	project_settings.active_workflow = workflow_file


def resolve_env(
	requires_releases: list[str],
	default_host_settings: dict,
	wf_config: gv.WorkflowConfig,
	project_settings: gv.ProjectSettings,
	update_all: bool = False,
):
	"""
	Resolve the data environment.

	Audits the requirment expressions against
	the lock file. Any requirements that aren't satisfied are
	searche for in the archives pointed to by the RME settings.

	And requirement expression satisfied by a dataset in the
	lock file are considered satisified.

	Parameters
	----------
	requires_releases : list[str]
		List of release expressions that satisfy
		data requirements.
	default_host_settings : dict
		keys host, user, password to define authentication
		to the default archive.
	wf_config: gv.WorkflowConfig
		Configuration of the workflow whose data environment
		is being synchronized to.
	project_settings : gv.ProjectSettings
		Project settings.
	update_all: bool, optional
		If true, doesn't check against lock file
		and just searches for most up to date version of each
		requirement condition.


	Returns
	-------
	release_records : dict
		Dictionary of releases (keys are versioned titles) that
		should be installed in the current environment.
	release_archives : dict
		Dictionary of of ArchiveInterfaces (keys are versioned titles)
		for where to pull archival data from.
	audit_passed : bool
		True if the environment already satisfies the requirements.
	new_lock : dict
		Lock file of installed packages that satisfy the requirements

	"""
	archives = {}  # archives by host name
	release_records = {}  # dataset records by datset name
	release_archives = {}  # archives by dataset name (has repeat pointers, but is convenient for lookup
	cache = CacheInterface(gv.CACHE_FOLDER)
	workspaces = {}
	locked_count = 0
	rme_settings = project_settings.rmesettings

	try:
		with open(wf_config.path.with_suffix('.lock'), 'r') as f:
			lock = safe_load(f)
	except FileNotFoundError:
		lock = {}
	new_lock = {}

	# grab the record of every dataset I might need
	# based on version string
	env_resolution_time = time.time()
	if len(requires_releases) > 0:
		cprint(
			'Resolving environment...',
			color=Colors.UNDERLINE + Colors.HEADER,
		)
	for release_expression in requires_releases:
		if '>=' in release_expression:
			char = '>='
			release_name = release_expression.split(char)[0]
			version_expressions = char + release_expression.split(char)[1]
		elif '==' in release_expression:
			char = '=='
			release_name = release_expression.split(char)[0]
			version_expressions = char + release_expression.split(char)[1]
		elif '<' in release_expression:
			char = '<='
			release_name = release_expression.split(char)[0]
			version_expressions = char + release_expression.split(char)[1]
		else:
			release_name = release_expression
			version_expressions = '>=0.0.0'
		version_expressions = version_expressions.split(', ')

		# split up the release name and workspace
		split_release_name = release_name.split('/')
		if len(split_release_name) == 1:
			release_name = split_release_name[0]
			workspace = 'Global Public Workspace'
		elif len(split_release_name) == 2:
			workspace = split_release_name[0]
			release_name = split_release_name[1]
		else:
			raise ValueError(f'Cant resolve workspace/release name of {release_name}')

		# audit the requirment against the lock file
		# if any of the installed releases match the versions
		# expressions then just use that one.

		satisfied_by_lock = None
		if not update_all:
			lock_installed = [n for n in lock if lock[n]['name'] == release_name]
			for installed in lock_installed:
				if all(
					[
						semver.match(lock[installed]['version'], ve)
						for ve in version_expressions
					]
					+ [lock[installed]['workspace'] == workspace]
				):
					satisfied_by_lock = installed
					locked_count += 1
					version_expressions = ['==' + lock[installed]['version']]

		# set which archive it should be coming from
		# use the default if not specified from somewhere else
		use_archive = default_host_settings
		if satisfied_by_lock:
			# match host url against
			# user settings for rme
			host = lock[satisfied_by_lock]['host']
			use_archive = None
			try:
				host, user, password = rme_settings.get_host_settings(host)
				use_archive = {'host': host, 'user': user, 'password': password}
			# not found in settings, so
			# just provided the host and resolve credentials
			# witht he credential manager
			except LookupError:
				use_archive = {'host': host}

		# if the host hasn't already been authenticated
		# and connected to, do that
		# otherwise just use an existing archive
		if use_archive['host'] not in archives:
			try:
				user = use_archive['user']
			except KeyError:
				user = None
			try:
				password = use_archive['password']
			except KeyError:
				password = None

			archives[use_archive['host']] = get_interface(
				host=use_archive['host'],
				user=user,
				password=password,
				resolve_relative_paths_to=project_settings.project_dir,
			)

		# grab a curator add it to a lookup
		# by dataset name
		archive = archives[use_archive['host']]

		# grab a dictionary of valid records sorted by
		# version title.
		# if satisfied by lock, use the source from the lockfile
		if satisfied_by_lock:
			try:
				cached_release_dir = cache.get_cached_release_dir(
					host=host, workspace=workspace, release_title=satisfied_by_lock
				)
				with open(cached_release_dir / '.rme/record.yml') as f:
					record = safe_load(f)
			except FileNotFoundError:
				# record wasn't found locally, so we need to get the record
				# of that version again from the locked source
				available_datasets = archive.get_release_records(
					title_versionless=release_name,
					version_expressions=['==' + lock[satisfied_by_lock]['version']],
					workspace=workspace,
				)
				record = list(available_datasets.values())[-1]

		# otherwise, we are looking for whatever version
		# satisfies the requirements
		else:
			available_datasets = archive.get_release_records(
				title_versionless=release_name,
				version_expressions=version_expressions,
				workspace=workspace,
			)
			record = list(available_datasets.values())[-1]

		# add the record and archive for that record
		# to look up dictionarys, will be used by
		# the build_env function
		versioned_release_title = record['title']
		release_records[versioned_release_title] = record
		release_archives[versioned_release_title] = archive
		workspaces[versioned_release_title] = workspace

	total_requirements = len(requires_releases)

	cprint(f'locked releases : {locked_count}/{total_requirements}')

	env_resolution_time = round(time.time() - env_resolution_time, 2)
	return release_records, release_archives, workspaces


def update_cache(
	release_records: dict,
	release_archives: dict[ArchiveInterface],
	workspaces: dict,
	force_download: bool = False,
	max_threads: int = 50,
	download_chunk_bytes: int = 4194304,
):
	"""
	Download any required releaes to the local cache.

	Parameters
	----------
	release_records : dict
		Dictionary of releases with versioned titles
		as keys, each one is required to be installed into
		the environment.
	release_archives : dict[ArchiveInterface]
		Dictionary of archive interfaces with versioned release
		titles as keys, informs where to download/link packages to and
		from.
	workspaces : dict
		Dictionary of versioned titles as keys, and the values
		are the names of the workspace that each release should
		be pulled from.
	project_settings : gv.ProjectSettings
		_description_
	wf_config : gv.WorkflowConfig
		_description_
	force_download : bool, optional
		Force a re-download to cache of each package, by default False
	force_relink : bool, optional
		Force a re-link to cache of each package, by default False
	max_threads : int, optional
		Maximum number of threads for parallel operations, by default 50
	download_chunk_bytes : int, optional
		Max chunk size per byte, by default 4194304
	"""
	release_versioned_titles = list(release_records.keys())
	# get a list of all the folders in the project
	# they should correspond to the dataset names
	# folders in the cache should be unique by title + version

	cache = CacheInterface(gv.CACHE_FOLDER)

	# remove any packages that were asked to be redownloaded
	for versioned_release_name in release_versioned_titles:
		# should be a method to select packages, but for now
		# force redownload does everything
		release_dict = dict(
			host=release_archives[versioned_release_name].host,
			workspace=workspaces[versioned_release_name],
			release_title=versioned_release_name,
		)
		cache_already_exists = cache.in_cache(**release_dict)
		if cache_already_exists and force_download:
			cache.rm_cached_release(**release_dict)

	# for each one that doesn't already exist, walk the record and
	# download blobs to the local cache
	download_count = 0
	download_time = time.time()
	for versioned_release_name in release_versioned_titles:
		release_dict = dict(
			host=release_archives[versioned_release_name].host,
			workspace=workspaces[versioned_release_name],
			release_title=versioned_release_name,
		)
		cache_already_exists = cache.in_cache(**release_dict)

		if not cache_already_exists:
			if download_count == 0:
				cprint(
					'Downloading packages...', color=Colors.UNDERLINE + Colors.HEADER
				)
			download_count += 1

			# pick the release record and archive
			record = release_records[versioned_release_name]
			archive = release_archives[versioned_release_name]

			# download release to the cache fodler
			archive.download_release(
				record=record,
				workspace=workspaces[versioned_release_name],
				max_threads=max_threads,
				release_cache_folder=cache.get_cached_release_dir(**release_dict),
				download_chunk_bytes=download_chunk_bytes,
				progress_bar=True,
			)

	download_time = round(time.time() - download_time, 2)


def build_data_env(
	release_records: dict,
	release_archives: dict[ArchiveInterface],
	requires_files: dict,
	workspaces: dict,
	project_settings: gv.ProjectSettings,
	wf_config: gv.WorkflowConfig,
	force_relink: bool = False,
):
	"""
	Rebuild a projects data environment based on requirements.

	Parameters
	----------
	release_records : dict
		Dictionary of release records required,
		keys are versioned titles.
	release_archives : dict[ArchiveInterface]
		Dictionary of archive interfaces to use when
		accessing releeases, keys are versioned titles
	requires_files : dict
		Dictionary of file requirements. Keys are the name space
		assigned to the requirement.
	workspaces : dict
		dictionary of workspaces to use when accessing
		releases, keys are versioned titles.
	project_settings : gv.ProjectSettings
		_description_
	wf_config : gv.WorkflowConfig
		Workflow configuration
	force_download : bool, optional
		_description_, by default False
	force_relink : bool, optional
		_description_, by default False
	max_threads : int, optional
		_description_, by default 50
	download_chunk_bytes : int, optional
		_description_, by default 4194304

	Returns
	-------
	_type_
		_description_
	"""
	packages_dir = project_settings.data_env_packages
	env = EnvInterface(project_settings.data_env_dir)
	cache = CacheInterface(gv.CACHE_FOLDER)

	# remove any packages that need to be relinked
	first_install = False
	install_header = 'Installing requirements...'
	install_color = Colors.UNDERLINE + Colors.HEADER
	for versioned_release_name in release_records:
		# should be a method to select packages, but for now
		# force redownload does everything
		if env.in_env(versioned_release_name) and force_relink:
			if not first_install:
				cprint(install_header, color=install_color)
				first_install = True
			env.rm_release(versioned_release_name)
			cprint(f'{symbols.BIGX} {versioned_release_name}', color=Colors.RED)

	# remove anything that shouldn't be installed
	already_installed = set(env.installed_in_packages() + env.installed_in_registry())
	for installed in already_installed:
		if force_relink or (
			(installed not in release_records) and (installed not in requires_files)
		):
			if not first_install:
				cprint(install_header, color=install_color)
				first_install = True
			cprint(f'{symbols.BIGX} {installed}', color=Colors.RED)
			env.rm_release(installed)

	# Install things that aren't already installed
	already_installed = set(env.installed_in_packages() + env.installed_in_registry())

	# install releases first
	for versioned_release_name, record in release_records.items():
		if versioned_release_name not in already_installed:
			if not first_install:
				cprint(install_header, color=install_color)
				first_install = True
			env.install_release(
				release_record=record,
				workspace=workspaces[versioned_release_name],
				host=release_archives[versioned_release_name].host,
				cache=cache,
			)
			cprint(f'{symbols.CHECK} {versioned_release_name}', color=Colors.OKGREEN)

	# install file systems
	for namespace, files in requires_files.items():
		if namespace not in already_installed:
			if not first_install:
				cprint(install_header, color=install_color)
				first_install = True
			env.install_files_namespace(namespace, files, project_settings.project_dir)
			cprint(f'{symbols.CHECK} {namespace}', color=Colors.OKGREEN)

	# Make a versionless link for the "most up to date"
	# of each unique release
	unique_packages = set(
		[record['title_versionless'] for record in release_records.values()]
	)
	package_versions = {uname: [] for uname in unique_packages}
	most_recent_time = time.time()

	def get_version(title):
		return semver.parse(release_records[title]['version'])

	for record in release_records.values():
		package_versions[record['title_versionless']].append(record['title'])

	for uname, titles in package_versions.items():
		titles.sort(key=get_version)
		most_recent = titles[-1]
		cache_release_dir = cache.get_cached_release_dir(
			host=release_archives[most_recent].host,
			workspace=workspaces[most_recent],
			release_title=most_recent,
		)
		env.make_registry(
			release_record=release_records[most_recent],
			cache_release_dir=cache_release_dir,
			use_versioned_name=False,
		)
		cprint(
			f'{symbols.CHECK} {uname} -> {most_recent}',
			color=Colors.OKGREEN,
		)


if __name__ == '__main__':
	import os

	project_dir = Path('tests/first-workflow')
	os.chdir(project_dir)
	sync(
		'second-workflow',
		project_dir=Path.cwd(),
		update_all=True,
		force_download=True,
		force_relink=True,
	)
