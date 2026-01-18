import click
import json
from pathlib import Path
from rmellipse.workflows.workflowtree import (
    WorkflowTree,
    execute_concurrent_jobs,
    validate_datapointers,
)
from rmellipse.workflows._settings import ProjectSettings, WorkflowConfig
from rmellipse.workflows._cli_sync import sync


@click.command(name='run')
@click.argument('workflow_file', type=Path)
@click.option('--default-host', type=Path)
@click.option('--max-threads', type=int)
def run_cli(*args, **kwargs):
    """Run a workflow in the project."""
    run(*args, **kwargs)


def run(
    workflow_file: Path | str,
    project_dir: Path = Path.cwd(),
    default_host: Path = None,
    max_threads: int = None,
):
    """
    Run a workflow.

    Parameters
    ----------
    workflow_file : Path
        Path to a workflow file, relative to project directory.
    project_dir : Path
            Project path, used to resolve relative paths
    default_host : Path
            Path to the default archival host for synchronizing, if
            not provided it is inferred from user or project settings.
    max_threads : int
            Number of processes to use when running workflow. Default
            is number of available cores - 1.
    """
    # sync the data env first
    sync(workflow_file, project_dir=Path(project_dir), default_host=default_host)

    # don't require .flw.yml suffix to be typed in
    project_settings = ProjectSettings(project_dir)
    wf_config = WorkflowConfig(
        project_settings.project_dir / workflow_file, project_settings
    )
    wf_tree = WorkflowTree.from_workflow_config(project_settings, wf_config)

    # assign datasets

    # do a topological sort
    for i, node_group in enumerate(wf_tree.iter_topological_groups()):
        current_jobs = []
        available_pntrs = []
        # make a list of active jobs and pntrs
        for node_name in node_group:
            if node_name in wf_tree.jobs:
                current_jobs.append(wf_tree.jobs[node_name])
            if node_name in wf_tree.data_pointers:
                available_pntrs.append(wf_tree.data_pointers[node_name])
        # validate data sets that should be available at this group
        # level or execute the list of jobs
        if len(available_pntrs) > 0:
            validate_datapointers(available_pntrs, project_settings, i)
        if len(current_jobs) > 0:
            execute_concurrent_jobs(current_jobs, i, max_threads=max_threads)

    # If we made it here we executed the workflow
    # so lets save the workflow tree into the .rme
    # folder
    relative_path = (
        (Path(project_settings['PROJDIR']) / workflow_file)
        .resolve()
        .relative_to(project_settings['PROJDIR'].resolve())
    )
    output_file = project_settings.wft_jsondir / f'{relative_path}.json'
    output_file.parents[0].mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w') as f:
        json.dump(wf_tree, f, indent=True)


if __name__ == '__main__':
    run(Path(r'first-workflow'), Path(r'tests/first-workflow'))
