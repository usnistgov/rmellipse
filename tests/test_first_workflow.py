import rmellipse.workflows._cli_run as run
import rmellipse.workflows._cli_map as map
import rmellipse.workflows._cli_release as release
import rmellipse.workflows._cli_sync as sync
import rmellipse.workflows.archive_interface
import pytest
import shutil
from pathlib import Path

TEST_DIR = Path(__file__).parents[0]
IGNORED_ARCHIVE = TEST_DIR / 'ignored_archive'
TEST_CASE_ARCHIVE = TEST_DIR / 'test_case_archive'


# prefix components:
space = '    '
branch = '│   '
# pointers:
tee = '├── '
last = '└── '


def tree(dir_path: Path, prefix: str = ''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no mor


def clear_ignored_archive():
    for f in IGNORED_ARCHIVE.iterdir():
        if f.is_dir():
            shutil.rmtree(f)


def clean_test_folder(proj):
    folders = [proj / '.rme', proj / 'data_env']
    for f in folders:
        if f.exists():
            shutil.rmtree(f)
    for file in proj.iterdir():
        # remove lock files, can cause a weird
        # state for this kind of thing
        if '.rme.lock' in file.name:
            file.unlink()


def test_first_workflow(host=IGNORED_ARCHIVE):
    print('RUNNING FIRST WORKFLOW')
    print('======================')

    proj_dir = TEST_DIR / 'first-workflow'
    clear_ignored_archive()
    clean_test_folder(proj_dir)

    print(proj_dir)
    try:
        run.run('first-workflow', project_dir=proj_dir, default_host=host)
    except FileNotFoundError as e:
        print('caught error, mapping directory')
        map.map('first-workflow', project_dir=proj_dir)
        raise e from e

    map.map('first-workflow', project_dir=proj_dir)

    # try to push a release to the ignored archive
    # to make sure the release button works
    # the CDCS archive needs repeat releases
    # because I don't have a way to clear it easily for testing purposes
    repeat_release = rmellipse.workflows.archive_interface.is_url(str(host))
    release.release(
        'first-workflow',
        project_directory=proj_dir,
        host=host,
        repeat_release=repeat_release,
    )

    # try to push a release to the ignored archive,
    # it should fail since the tesst case should already be there
    with pytest.raises(Exception):
        release.release('first-workflow', project_directory=proj_dir, host=host)

    # print archive layout for debugging
    if Path(host).exists():
        print('ARCHIVE LAYOUT')
        print('--------------')
        for line in tree(host):
            print(line)

    print('RUNNING SECOND WORKFLOW')
    print('=======================')

    # try to sync the second workflow, which should pull in the first
    sync.sync(
        'second-workflow',
        proj_dir,
        default_host=host,
        force_download=True,
        force_relink=True,
    )

    # try to sync the second workflow, which should pull in the first
    run.run('second-workflow', project_dir=proj_dir, default_host=host)


if __name__ == '__main__':
    from rmellipse.workflows.archive_interface import CDCSArchive

    CDCSArchive('http://127.0.0.1', 'dcg2')
    test_first_workflow(host='http://127.0.0.1')
    test_first_workflow(host=IGNORED_ARCHIVE)
    # "PYTEST_EXTRAS" not in os.environ
