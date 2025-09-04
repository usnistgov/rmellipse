import rmellipse.workflows._cli_run as run
import rmellipse.workflows._cli_map as map
from pathlib import Path

PROJ_DIR = Path(__file__).parents[0]/'workflow-hello'

def test_workflow_hello():
    run.run(
        Path('hello'),
        project_dir=PROJ_DIR
    )

    map.map(
        project_dir=PROJ_DIR
    )

if __name__ == '__main__':
    test_workflow_hello()