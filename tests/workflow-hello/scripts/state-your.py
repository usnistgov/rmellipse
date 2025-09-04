import click
from pathlib import Path
import time


@click.command()
@click.argument("value", type=str)
@click.argument("filename", type=Path)
@click.option("--pause", type=float, default = 0)
def main(value, filename, pause = 0):
    time.sleep(pause)   
    with open(filename, "w") as f:
        f.write(value)


if __name__ == "__main__":
    main()
