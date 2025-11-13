import click
import shutil
from pathlib import Path


@click.command()
@click.argument('src', type=Path)
@click.argument('dest', type=Path)
def main(src, dest):
	print('copying ', src, 'to', dest)
	shutil.copy(src, dest)


if __name__ == '__main__':
	main()
