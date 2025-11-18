import click
from pathlib import Path
import rmellipse.workflows._settings as settings
import json
import git
from rmellipse.workflows.extras import map_directory, show_map
from rmellipse.workflows._printtools import cprint, Colors
from yaml import safe_load


@click.command(name='map')
@click.argument('workflow_file', type=Path)
@click.option(
	'--show-attrs', is_flag=True, default=False, help='Show the attrs of mapped items.'
)
def map_cli(*args, **kwargs):
	"""Make a map of a workflow's project structure."""
	return map(*args, **kwargs)


# pseudo code:
# 1. load the workflow tree json object into memory
# 2. validate that the tree was generate at the matching commit number, and when it was fully commited.
# 3. map the project directory into a project_tree (dict of dicts) with pointers to binary blobs using .gitignore
# 4. map the any datasets from the workflow to that project structure
# 5. save the mapping with archival format in the .rme folder
def map(
	workflow_file: str | Path,
	project_dir: str | Path = Path.cwd(),
	no_show: bool = False,
	show_attrs: bool = False,
	show_only: list[str] = ['*'],
):
	"""
	Build a mapping of the project directory.

	These include
	* a project file that maps the directy structure and stores requirements
	* a workflow file that maps the DAG of the processes and data

	optionally, include a folder of the binary blobs, or those can be built
	on demand when publish is called.

	Parameters
	----------
	project_dir : str | Path, optional
	    _description_, by default Path.cwd()

	Returns
	-------
	_type_
	    _description_
	"""
	project_config = settings.ProjectSettings(project_dir)

	# open the workflow file
	workflow_config = settings.WorkflowConfig(
		project_dir / workflow_file, project_config
	)

	if not show_only:
		show_only = '*'

	repo = git.Repo(project_config.project_dir, search_parent_directories=True)

	# add include patterns
	try:
		include_globs = workflow_config['release']['includes']
		include_globs = [str(project_dir / pattern) for pattern in include_globs]
	except KeyError:
		include_globs = []
	# add ignore patterns
	try:
		ign_globs = workflow_config['release']['ignores']
		ign_globs = [str(project_dir / pattern) for pattern in ign_globs]
	except KeyError:
		ign_globs = []

	# map out the project directory
	mapping = map_directory(
		{}, project_config.project_dir, incl_globs=include_globs, ign_globs=ign_globs
	)

	# add any additional annotations

	# write my things to json
	map_file = project_config.project_map
	with open(map_file, 'w') as f:
		json.dump(mapping, f, indent=True)

	if not no_show:
		cprint(project_config.project_dir.name, color=Colors.HEADER + Colors.UNDERLINE)
		if isinstance(show_only, str):
			show_only = [show_only]
		show_map(mapping, level=1, show_attrs=show_attrs, show_only=show_only)
	return mapping


if __name__ == '__main__':
	proj_tree = map(
		'workflow',
		project_dir=Path(r'.\tests\arrschema-workflow\\').resolve(),
		show_attrs='*',
	)
