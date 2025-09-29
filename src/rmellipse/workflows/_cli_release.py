import click
from yaml import safe_load
from itertools import cycle
from pathlib import Path
from rmellipse.workflows._printtools import cprint, colors, braile_load, symbols
import rmellipse.workflows._cdcs_helpers as cdcsh
import rmellipse.workflows._globals as gv
import rmellipse.workflows._cli_map as maph
import rmellipse.workflows._wf_helpers as colh
import json
import concurrent.futures
import time

# from rmellipse.workflows.projecttree import ProjectTree
# import rmellipse.workflows._globals as flowbals
# import json
# import jsonschema
# from rmellipse.workflows._collections import walk_project, matches_any
# from rmellipse.workflows.workflowtree import WorkflowTree
# from rmellipse.workflows._printtools import cprint, colors
__all__ = ['release']


@click.command(name='release')
@click.argument('workflow-file', type=Path)
@click.option('--no-blobs', is_flag=True, default=False)
@click.option('--repeat-release', is_flag=True, default=False)
@click.option(
	'--max-threads', type=int, default=50, help='Maximum number of threads to use.'
)
def release_cli(*args, **kwargs):
	"""Release a workflow to CDCS."""
	return release(*args, **kwargs)


def upload_blob_and_mapping(
	cmap: dict,
	curator: cdcsh.CachedCurator,
	project_directory: Path,
	no_blobs: bool,
	thread_finished_map: dict,
):
	# move cursor to beginning of list
	if not no_blobs:
		blob_id = cdcsh.process_file(
			curator=curator,
			posix_rel_path=cmap['/path/'],
			working_dir=project_directory,
			verbose=False,
		)
	else:
		blob_id = 'NONE'
	# assign the blob id to the mapping of the blob
	cmap[gv.MAPPING_META_KEYS.BPID.value] = blob_id
	# update the thread finished portion
	thread_finished_map[cmap[gv.MAPPING_META_KEYS.PATHSPEC.value]] = True


def release(
	workflow_file: Path,
	project_directory: Path = Path.cwd(),
	no_blobs: bool = False,
	repeat_release: bool = False,
	max_threads: int = 50,
):
	workflow_file = Path(workflow_file)
	project_directory = Path(project_directory)

	if workflow_file.suffixes == []:
		workflow_file = workflow_file.with_suffix('.rme.yml')

	# global config settings
	project_config = gv.ProjectSettings(project_directory)

	# requirements
	reqs = project_config.requirements

	# workflow settings
	with open(project_directory / workflow_file, 'r') as f:
		wf_config = safe_load(f)
	cdcssettings = project_config.cdcssettings()

	rel_set = wf_config['cdcs-release']
	# build a project map
	project_mapping = maph.map(project_dir=project_directory, no_show=True)

	release_title_no_version = f'{workflow_file.stem.split(".")[0]}'
	release_version = f'{wf_config["cdcs-release"]["version"]}'
	release_title = f'{release_title_no_version}-v{release_version}'

	cprint('Connecting to CDCS...', color=colors.UNDERLINE + colors.HEADER)
	curator = cdcsh.login(
		hostname=cdcssettings[rel_set['to']]['host'],
		username=cdcssettings[rel_set['to']]['user'],
		password=cdcssettings[rel_set['to']]['password'],
	)

	if not repeat_release:
		cprint('Checking for repeat...', color=colors.UNDERLINE + colors.HEADER)
		releases = curator.query(
			template='Release', mongoquery={'title': release_title}, progress_bar=False
		)
		if len(releases) > 0:
			msg = f'Release {release_title} already exists.'
			msg += 'set --repeat-release to ignore this error.'
			raise SystemExit(msg)

	cprint('Uploading blobs...', color=colors.UNDERLINE + colors.HEADER)
	if no_blobs:
		cprint('NO BLOBS ARE BEING UPLOADED', color=colors.WARNING)
		input('Continue or Ctrl + C')

	# walk the project map,
	# counting number of blobably items and starting
	# a worker to upload them
	total = 0
	max_name = 0
	thread_finished_map = {}
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
		for name, cmap in colh.iter_blobable(project_mapping):
			total += 1
			max_name = max((max_name), len(name))
			thread_finished_map[cmap[gv.MAPPING_META_KEYS.PATHSPEC.value]] = False
			executor.submit(
				upload_blob_and_mapping,
				cmap,
				curator,
				project_directory,
				no_blobs,
				thread_finished_map,
			)

		print('total blobable items: ', total)
		load_sym = cycle(braile_load)
		finished_sym = symbols.CHECK
		finished_count = 0
		cursor_count = 0
		# monitor each uploading thread
		while finished_count < total:
			print('\033[F' * cursor_count, end='')
			msg = ''
			ongoing_sym = next(load_sym)
			finished_count = 0
			cursor_count = 0
			for i, (pathspec, finished) in enumerate(thread_finished_map.items()):
				if finished:
					finished_count += 1
					sym = finished_sym
				else:
					sym = ongoing_sym
				# print ongoing upload if less then 10
				if i < 10:
					cursor_count += 1
					msg += f'{sym} | {pathspec.ljust(max_name)}\n'

			if total > 10:
				msg += 'other processes hidden ...\n'
				cursor_count += 1

			msg = f'completed {finished_count}/{total}\n' + msg
			cursor_count += 1
			cursor_count += 1
			print(msg)

			time.sleep(0.1)

		executor.shutdown(wait=True)

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
	cprint('Uploading the release record...', color=colors.UNDERLINE + colors.HEADER)
	release_record = {
		'title': release_title,
		'tile_versionless': release_title_no_version,
		'version': release_version,
		'workflow': wft,
		'requirements': reqs,
		'project': project_mapping,
	}

	response = cdcsh.upload_record(
		curator=curator,
		title=release_title,
		template_title='Release',
		content=release_record,
	)

	pid = json.loads(response.json().get('content'))['/PID/']
	print(f'release uploaded at: {pid}')


if __name__ == '__main__':
	release(
		'hello',
		project_directory='tests/workflow-hello',
		no_blobs=False,
		repeat_release=True,
	)
