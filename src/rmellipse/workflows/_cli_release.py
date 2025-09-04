import click
from yaml import safe_load
from itertools import cycle
from pathlib import Path
from rmellipse.workflows._printtools import cprint, colors, braile_load
import rmellipse.workflows._cdcs_helpers as cdcsh
import rmellipse.workflows._globals as gv
import rmellipse.workflows._cli_map as maph
import rmellipse.workflows._wf_helpers as colh
import json

# from rmellipse.workflows.projecttree import ProjectTree
# import rmellipse.workflows._globals as flowbals
# import json
# import jsonschema
# from rmellipse.workflows._collections import walk_project, matches_any
# from rmellipse.workflows.workflowtree import WorkflowTree
# from rmellipse.workflows._printtools import cprint, colors
__all__ = ["release"]


@click.command(name="release")
@click.argument("workflow-file", type=Path)
@click.option("--no-blobs", is_flag=True, default=False)
@click.option("--repeat-release", is_flag=True, default=False)
def release_cli(*args, **kwargs):
    """Release a workflow to CDCS."""
    return release(*args, **kwargs)


def release(
    workflow_file: Path,
    project_directory: Path = Path.cwd(),
    no_blobs: bool = False,
    repeat_release: bool = False,
):
    workflow_file = Path(workflow_file)
    project_directory = Path(project_directory)

    if workflow_file.suffixes == []:
        workflow_file = workflow_file.with_suffix(".rme.yml")

    # global config settings
    project_config = gv.ProjectSettings(project_directory)

    # requirements
    reqs = project_config.requirements

    # workflow settings
    with open(project_directory / workflow_file, "r") as f:
        wf_config = safe_load(f)
    cdcssettings = project_config.cdcssettings()

    rel_set = wf_config["cdcs-release"]
    # build a project map
    project_mapping = maph.map(project_dir=project_directory, no_show=True)

    release_title_no_version = f"{workflow_file.stem.split('.')[0]}"
    release_version = f"{wf_config['cdcs-release']['version']}"
    release_title = f"{release_title_no_version}-v{release_version}"

    cprint("Connecting to CDCS...", color=colors.UNDERLINE + colors.HEADER)
    curator = cdcsh.login(
        hostname=cdcssettings[rel_set["to"]]["host"],
        username=cdcssettings[rel_set["to"]]["user"],
        password=cdcssettings[rel_set["to"]]["password"],
    )

    if not repeat_release:
        cprint("Checking for repeat...", color=colors.UNDERLINE + colors.HEADER)
        releases = curator.query(
            template="Release", mongoquery={"title": release_title}
        )
        if len(releases) > 0:
            msg = f"Release {release_title} already exists."
            msg += "set --repeat-release to ignore this error."
            raise SystemExit(msg)

    cprint("Uploading blobs...", color=colors.UNDERLINE + colors.HEADER)
    if no_blobs:
        cprint("NO BLOBS ARE BEING UPLOADED", color=colors.WARNING)
        input("Continue or Ctrl + C")
    total = 0
    max_name = 0
    for name, cmap in colh.iter_blobable(project_mapping):
        total += 1
        max_name = max((max_name), len(name))

    print("total blobable items: ", total)
    load_sym = cycle(braile_load)
    count = 1
    for name, cmap in colh.iter_blobable(project_mapping):
        # move cursor to beginning of list
        print("\033[F" * 2)
        msg = f"{next(load_sym)} | {name.ljust(max_name)} | {count}/{total}"
        print(msg, end="\n")
        if not no_blobs:
            blob_id = cdcsh.process_file(
                curator=curator, 
                posix_rel_path=cmap["/path/"],
                working_dir=project_directory, 
                verbose=False
            )
        else:
            blob_id = "NONE"
        cmap[gv.MAPPING_META_KEYS.BPID.value] = blob_id
        count += 1

    # read in the workflow solution
    # generated for the workflow file
    relative_path = (
        (Path(project_config["PROJDIR"]) / workflow_file)
        .resolve()
        .relative_to(project_config["PROJDIR"].resolve())
    )
    wft_sol_file = project_config.wft_jsondir / f"{relative_path}.json"
    wft_sol_file.parents[0].mkdir(exist_ok=True, parents=True)
    with open(wft_sol_file, "r") as f:
        wft = json.load(f)

    # (currently empty)
    cprint("Uploading the release record...", color=colors.UNDERLINE + colors.HEADER)
    release_record = {
        "title": release_title,
        "tile_versionless": release_title_no_version,
        "version": release_version,
        "workflow": wft,
        "requirements": reqs,
        "project": project_mapping,
    }

    cdcsh.upload_record(
        curator=curator,
        title=release_title,
        template_title="Release",
        content=release_record,
    )


if __name__ == "__main__":
    release(
        "hello",
        project_directory="tests/workflow-hello",
        no_blobs=False,
        repeat_release=True,
    )
