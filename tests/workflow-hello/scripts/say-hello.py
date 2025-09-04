import click
import time
import h5py
from rmellipse.utils import save_object
from pathlib import Path


@click.command()
@click.argument("name_file", type=Path)
@click.argument("job_file", type=Path)
@click.argument("result_file", type=Path)
@click.option("--pause", type=int, default=0)
def fun_cli(*args, **kwargs):
    return fun(*args, **kwargs)


def fun(name_file, job_file, result_file, pause=0):
    time.sleep(pause)
    with open(name_file, "r") as f:
        name = f.readline().strip()
    with open(job_file, "r") as f:
        job = f.readline().strip()
    result_print = " ".join(["Hello", name, "the", job, "!"])
    obj = {"job": job, "name": name}
    with h5py.File(result_file, "w") as f:
        save_object(f, "my_object", obj, verbose=True)
    print(result_print)


if __name__ == "__main__":
    fun_cli()
    # fun(
    #     name_file="tests/workflow-hello/name-file.txt",
    #     job_file="tests/workflow-hello/job-file.txt",
    #     result_file="tests/workflow-hello/result_file.txt",
    # )
