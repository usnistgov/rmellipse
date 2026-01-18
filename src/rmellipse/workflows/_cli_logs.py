"""Interact with job logs."""

import click
from pathlib import Path
from rmellipse.workflows._settings import ProjectSettings


@click.command(name='log')
@click.argument('job', type=str)
@click.option('--file', is_flag=True, help='Print the log file path.')
def logs_cli(*args, **kwargs):
    """Interact with log files created by jobs."""
    logs(*args, **kwargs)


def logs(job: str, project_dir: Path = Path.cwd(), file: bool = False):
    """
    Get logs for a job.

    Parameters
    ----------
    jobs: str, optional
        If provided, includes requirements for the workflow
        in the environment syncronization.
    project_dir : Path, optional
        Project direction. Default is the CWD.
    file : bool, optional
        If True, gets the file path.
    """
    project_settings = ProjectSettings(project_dir)
    process_log_dir = project_settings.processlogdir
    file_path = process_log_dir / f'{job}.txt'
    if file:
        print(str(file_path.as_posix()))
        return
    if not file_path.exists():
        raise ValueError(f'Job {job} not found.')
    with open(file_path, 'r') as f:
        for l in f:
            print(l)
