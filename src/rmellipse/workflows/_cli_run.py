import click
from pathlib import Path
from rmellipse.workflows.workflowtree import WorkflowTree, Job, DataPointer
from rmellipse.workflows._printtools import colors, symbols, cstr, cprint
from yaml import safe_load
from typing import Any
from itertools import cycle
from rmellipse.workflows._globals import ProjectSettings, WORKFLOW_SPECIAL_KEYS
from rmellipse.workflows._wf_helpers import enum_matching_dispatch
import json
import time



@click.command(name="run")
@click.argument("workflow_file", type=Path)
def run_cli(*args, **kwargs):
    """Run a workflow in the project."""
    run(*args, **kwargs)

def add_job_to_tree(project_settings, wft, job_name, job_dict):
    # add a job
    k = job_name
    job_dict
    job_dict.update({'name':k})

    try:
        output_dicts = job_dict.pop('outputs')
    except KeyError:
        output_dicts = {}

    try:
        input_dicts = job_dict.pop('inputs')
    except KeyError:
        input_dicts = {}

    j = Job(job_dict, project_settings)
    wft.add_job(j)

    # connect job to inputs
    for name, path in input_dicts.items():
        input = DataPointer({
            'name':name,
            'path':path
        })
        wft.add_data_pointer(input)
        wft.add_edge(input, j)

    # connect job to inputs
    for name, path in output_dicts.items():
        output = DataPointer({
            'name':name,
            'path':path
        })
        wft.add_data_pointer(output)
        wft.add_edge(j, output)

def set_release(wft: WorkflowTree, release_dict: dict):
    wft['release'] = release_dict


def build_workflowtree(workflow_file:Path, project_settings: ProjectSettings) -> WorkflowTree:
    """
    Build a workflow tree from a rme.yml file.

    Parameters
    ----------
    workflow_file : Path
        Path to the file to be built.

    Returns
    -------
    WorkflowTree
        Describes all the jobs and expected datsets of a
        workflow.
    """
    # load in the workflow_config

    with open(project_settings.project_dir / workflow_file, "r") as f:
        workflow_config = safe_load(f)

    wft = WorkflowTree()
    for k, v in workflow_config.items():
        # ignore field starting with a '.'
        # or that belong to my special workflowfile keys
        # treat '.' as "ignore me"
        if k[0] != '.':
            # if not in the special workflow keys
            # treat it like a job
            if k not in WORKFLOW_SPECIAL_KEYS:
                item = WORKFLOW_SPECIAL_KEYS.JOB
            else:
                item = k
            match_pattern = {
                WORKFLOW_SPECIAL_KEYS.JOB:{
                    'fn':add_job_to_tree,
                    'args':(project_settings, wft, k, v)
                },
                WORKFLOW_SPECIAL_KEYS.CDCS_RELEASE:{
                    'fn':set_release,
                    'args': (wft, v)
                }
            }
            enum_matching_dispatch(item, WORKFLOW_SPECIAL_KEYS, match_pattern)

    return wft

def validate_datapointers(
    datapointers: list[DataPointer],
    project_settings: ProjectSettings
):
    # if pnts, then just check that
    # the datasets claimed to exists actually exist
    cprint(f'DataSets', color = colors.HEADER)
    proj_dir = project_settings.project_dir
    for ap in datapointers:
        if (proj_dir / ap.path).is_file() or (proj_dir / ap.path).is_dir():
            cprint(f'  {symbols.CHECK} | {ap.name}',color = colors.OKGREEN)
        else:
            cprint(f'  {symbols.XBOX} | {ap.name}', color = colors.FAIL)
            raise Exception(f'Expected data-set {ap.path} is missing')


def execute_concurrent_jobs(
    jobs:list[Job],
    group_name: Any
    ):
    """
    Execute a set of concurrent jobs in wft.

    Parameters
    ----------
    wft : WorkflowTree
        Tree containing jobs.
    jobs : list[job]
        List of job names to execute concurrently.

    Raises
    ------
    TypeError
        If non-jobs are passed.
    SystemExit
        When a job fails.
    """
    # if on a job level, just execute each job
    cprint(f'Jobs -> level {group_name}', color = colors.HEADER)
    completed = []
    failed = set()
    load_syms = cycle(['⣾','⣽','⣻','⢿','⡿','⣟','⣯','⣷'])
    s = next(load_syms)
    for i, cj in enumerate(jobs):
            try:
                cj.thread.start()
            except TypeError as e:
                raise SystemExit(f'Failed to start {cj.name}: ') from e
            completed.append(False)
            print(f'  {s} | {cj.name}')

    t0 = time.time()
    while not all(completed):
        msg = ''
        if time.time() - t0 > 0.15:
            t0 = time.time()
            s = next(load_syms)
        # move cursor to beginning of list
        msg+="\033[F"*len(jobs)
        # write the status of each job
        for i, cj in enumerate(jobs):
            if not cj.thread.is_alive():
                completed[i] = True
                if cj.result['returncode']==0:
                    msg+= ''.join(cstr(f'  {symbols.CHECK} | {cj.name}\n', color = colors.OKGREEN))
                else:
                    msg+= ''.join(cstr(f'  {symbols.BIGX} | {cj.name}\n', color = colors.FAIL))
                    failed.add(cj.name)
            else:
                msg+= f'  {s} | {cj.name}\n'
            time.sleep(0.001)
        print(msg, end = '')

    # if anything failed, stop the execution
    if failed:
        for j in jobs:
            if j.result['returncode']!=0:
                header = f'{j.name} STDOUT'
                print('')
                cprint(len(header)*'~', color = colors.WARNING)
                cprint(header, color = colors.WARNING)
                cprint(len(header)*'~', color = colors.WARNING)
                with open(j.logfile, "r") as reader:
                    for line in reader:
                        print(line)

                header = f'{j.name} STDERR'
                print('')
                cprint(len(header)*'~', color = colors.FAIL)
                cprint(header, color = colors.FAIL)
                cprint(len(header)*'~', color = colors.FAIL)
                print(str(j.result['stderr']))

        cprint('--------------------------', color = colors.FAIL)
        raise SystemExit(f'Jobs {[f for f in failed]} failed.')

def run(workflow_file: Path | str, project_dir: Path = Path.cwd()):
    """
    Run a workflow.

    Parameters
    ----------
    workflow_file : Path
        Path to a workflow file, relative to project directory.
    """
        # don't require .flw.yml suffix to be typed in
    workflow_file = Path(workflow_file)
    if workflow_file.suffixes == []:
        workflow_file = workflow_file.with_suffix(".rme.yml")
    project_settings = ProjectSettings(project_dir)
    wft = build_workflowtree(workflow_file, project_settings)


    # do a topological sort
    for i, node_group in enumerate(wft.iter_topological_groups()):
        current_jobs = []
        available_pntrs = []
        # make a list of active jobs and pntrs
        for node_name in node_group:
            if node_name in wft.jobs:
                current_jobs.append(wft.jobs[node_name])
            if node_name in wft.data_pointers:
                available_pntrs.append(wft.data_pointers[node_name])
        # validate data sets that should be available at this group
        # level or execute the list of jobs
        if len(available_pntrs)>0:
            validate_datapointers(available_pntrs, project_settings)
        if len(current_jobs)>0:
            execute_concurrent_jobs(current_jobs, i/2)

    # If we made it here we executed the workflow
    # so lets save the workflow tree into the .rme
    # folder
    relative_path = (Path(project_settings['PROJDIR']) / workflow_file).resolve().relative_to(project_settings['PROJDIR'].resolve())
    output_file = project_settings.wft_jsondir / f"{relative_path}.json"
    output_file.parents[0].mkdir(exist_ok=True, parents = True)
    with open(output_file, "w") as f:
        json.dump(wft, f, indent = True)

if __name__ == '__main__':
    run(
        Path(r'tests/workflow-hello/hello'),
        Path(r'tests/workflow-hello')
    )