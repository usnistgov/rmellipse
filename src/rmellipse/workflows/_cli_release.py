import click
from yaml import safe_load
from itertools import cycle
from pathlib import Path
from rmellipse.workflows._printtools import cprint, Colors, braile_load, symbols
import rmellipse.workflows._cdcs_helpers as cdcsh
import rmellipse.workflows._settings as gv
import rmellipse.workflows._cli_map as maph
import rmellipse.workflows.extras as colh
from rmellipse.workflows.archive_interface import (
	get_interface,
	get_credentials,
	ReleaseRecordNotFoundError,
)
import json
import concurrent.futures
import time
import packaging.version as version

# from rmellipse.workflows.projecttree import ProjectTree
# import rmellipse.workflows._globals as flowbals
# import json
# import jsonschema
# from rmellipse.workflows._collections import walk_project, matches_any
# from rmellipse.workflows.workflowtree import WorkflowTree
# from rmellipse.workflows._printtools import cprint, olors
__all__ = ['release']


@click.command(name='release')
@click.argument('workflow-file', type=Path)
@click.argument('host', type=str)
@click.option('--no-blobs', is_flag=True, default=False)
@click.option('--repeat-release', is_flag=True, default=False)
@click.option('--workspace', type=str, default='Global Public Workspace')
@click.option(
	'--max-threads', type=int, default=50, help='Maximum number of threads to use.'
)
def release_cli(*args, **kwargs):
	"""Release a workflow to an archive."""
	return release(*args, **kwargs)


def release(
	workflow_file: Path,
	host: str | Path,
	project_directory: Path = Path.cwd(),
	workspace: str = 'Global Public Workspace',
	no_blobs: bool = False,
	repeat_release: bool = False,
	max_threads: int = 50,
):
	workflow_file = Path(workflow_file)
	project_directory = Path(project_directory)

	# global config settings
	project_config = gv.ProjectSettings(project_directory)

	# workflow settings
	wf_config = gv.WorkflowConfig(project_directory / workflow_file, project_config)
	rmesettings = project_config.rmesettings

	# requirements
	reqs = list(set(project_config.requires_releases + wf_config.requires_releases))

	# build a project map
	project_mapping = maph.map(
		workflow_file, project_dir=project_directory, no_show=True
	)

	release_title_no_version = wf_config.title
	release_version = wf_config.release_version
	release_title = colh.format_release_verions(
		release_title_no_version, release_version
	)
	cprint('Building release for...', color=Colors.UNDERLINE + Colors.HEADER)
	cprint(f'workflow file : {str(wf_config.path.relative_to(project_directory))}')
	cprint(f'release title : {release_title}')

	# cprint('Connecting to archive...'
	try:
		host, user, password = rmesettings.get_host_settings(host)
	except LookupError:
		host = host
		user = None
		password = None
	archive = get_interface(host, user=user, password=password)
	if repeat_release:
		supports_repeats = False
		try:
			supports_repeats = archive.supports_repeat_releases
		except AttributeError:
			pass
		if not supports_repeats:
			raise ValueError(
				f"Archive {type(archive)} doesn't support repeat releases."
			)
	if not repeat_release:
		# query for any versions matching the version trying to be released
		try:
			releases = archive.get_release_records(
				title_versionless=release_title_no_version,
				version_expressions=['==' + release_version],
				workspace=workspace,
			)
		except ReleaseRecordNotFoundError:
			releases = {}

		if len(releases) > 0:
			msg = f'Release {release_title} already exists at {archive.host} \n'
			msg += ' set --repeat-release to ignore this error.'
			# cprint(msg, color=Colors.FAIL)
			raise Exception(msg)

	cprint('Uploading blobs...', color=Colors.UNDERLINE + Colors.HEADER)
	if no_blobs:
		cprint('NO BLOBS ARE BEING UPLOADED', color=Colors.WARNING)
		input('Continue or Ctrl + C')

	# upload all the plobs and update the project
	# mapping with relecant metadata
	archive.upload_blobs_and_update_mapping(
		release_title=release_title,
		release_title_versionless=release_title_no_version,
		workspace=workspace,
		project_mapping=project_mapping,
		project_directory=project_directory,
		max_threads=max_threads,
		chunk_size=2**20,
		no_blobs=no_blobs,
	)

	# read in the workflow solution
	# generated for the workflow file
	relative_path = (
		(Path(project_config['PROJDIR']) / workflow_file)
		.resolve()
		.relative_to(project_config['PROJDIR'].resolve())
	)
	wft_sol_file = project_config.wft_jsondir / f'{relative_path}.json'
	wft_sol_file.parents[0].mkdir(exist_ok=True, parents=True)
	with open(wft_sol_file, 'r') as f:
		wft = json.load(f)

	# (currently empty)
	cprint('Uploading the release record...', color=Colors.UNDERLINE + Colors.HEADER)
	release_record = {
		'title': release_title,
		'title_versionless': release_title_no_version,
		'version': release_version,
		'workflow_config_path': Path(relative_path).as_posix(),
		'workflow': wft,
		'requirements': reqs,
		'project': project_mapping,
		'datasets': wf_config.datasets,
	}

	# upload the release record
	pid = archive.upload_release_record(release_record, workspace)
	print(f'release uploaded at: {pid}')


if __name__ == '__main__':
	release(
		'first-workflow',
		'http://127.0.0.1',
		project_directory='tests/first-workflow',
	)
