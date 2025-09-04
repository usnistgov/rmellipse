import click
from pathlib import Path
import rmellipse.workflows._globals as flowbals
import json
from rmellipse.workflows._wf_helpers import map_directory, show_map
from rmellipse.workflows._printtools import cprint, colors

@click.command(name = 'map')
@click.option('--show-attrs', is_flag = True, default = False, help = "Show the attrs of mapped items.")
def map_cli(*args,**kwargs):
    """Map a workflow in the project, building release-ready objects."""
    return map(*args, **kwargs)


# pseudo code:
# 1. load the workflow tree json object into memory
# 2. validate that the tree was generate at the matching commit number, and when it was fully commited.
# 3. map the project directory into a project_tree (dict of dicts) with pointers to binary blobs using .gitignore
# 4. map the any datasets from the workflow to that project structure
# 5. save the mapping with archival format in the .rme folder
def map(
        project_dir: str | Path = Path.cwd(),
        no_show: bool = False,
        show_attrs: bool = False,
        show_only: list[str] = ['*']
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
    project_config = flowbals.ProjectSettings(project_dir)
    if not show_only:
        show_only = '*'
    # map out the project directory
    mapping = map_directory({},project_config.project_dir)



    # write my things to json
    map_file = project_config.project_map
    with open(map_file,'w') as f:
        json.dump(mapping, f, indent = True)

    if not no_show:
        cprint(project_config.project_dir.name, color = colors.HEADER + colors.UNDERLINE)
        if isinstance(show_only, str):
            show_only = [show_only]
        show_map(
            mapping,
            level = 1,
            show_attrs=show_attrs,
            show_only= show_only
            ) 
    return mapping



if __name__ == '__main__':
    path = r'\tests\workflow-hello\.rme\workflow-solutions\workflows\hello.json'
    proj_tree = map(
        project_dir=Path(r'.\tests\workflow-hello\\').resolve(),
        show_attrs='*'
        )