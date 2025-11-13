import rmellipse.workflows._cli_run as run
import rmellipse.workflows._cli_map as map
import rmellipse.workflows._cli_release as release
import rmellipse.workflows._cli_sync as sync
import pytest
import shutil
import checksumdir
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


def clear_rme_folders(proj):
	folders = [proj / '.rme', proj / 'data_env']
	for f in folders:
		if f.exists():
			shutil.rmtree(f)


def test_first_workflow():
	proj_dir = TEST_DIR / 'first-workflow'
	clear_ignored_archive()
	clear_rme_folders(proj_dir)

	print(proj_dir)
	try:
		run.run('first-workflow', project_dir=proj_dir, default_host=IGNORED_ARCHIVE)
	except FileNotFoundError as e:
		print('caught error, mapping directory')
		map.map('first-workflow', project_dir=proj_dir)
		raise e from e

	map.map('first-workflow', project_dir=proj_dir)

	# try to push a release to the ignored archive
	# to make sure the release button works
	release.release('first-workflow', project_directory=proj_dir, host=IGNORED_ARCHIVE)

	# try to push a release to the ignored archive,
	# it should fail since the tesst case should already be there
	with pytest.raises(Exception):
		release.release(
			'first-workflow', project_directory=proj_dir, host=TEST_CASE_ARCHIVE
		)

	# print archive layout for debugging
	print('ARCHIVE LAYOUT')
	print('--------------')
	for line in tree(IGNORED_ARCHIVE):
		print(line)

	# try to sync the second workflow, which should pull in the first
	sync.sync('first-workflow', proj_dir, default_host=IGNORED_ARCHIVE)

	# try to sync the second workflow, which should pull in the first
	sync.sync('second-workflow', proj_dir, default_host=IGNORED_ARCHIVE)


if __name__ == '__main__':
	test_first_workflow()
