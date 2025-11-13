.. _first-workflow:
Make Your First Workflow
========================

This tutorial will walk through how to set up
a project, make a workflow, and release it to an
archive (without any dependencies).

Create a Project
----------------
Make a folder (this is your project) and in that directory make a file called ``rmeproject.yml``. This
is where project settings will be stored, which is where you can define settings that workflows in
a project should have. For now, leave it empty.

Make a Workflow File
--------------------
Make a workflow file anywhere in your
project with the extension ``.rme.yml``. This can have any filename. For this example
we will do ``first-workflow.rme.yml``.

Describe the Release
---------------------
Add a ``release`` field in ``first-workflow.rme.yml``.

.. code-block:: yaml

    release:
      # name of the release
      title: 'first-workflow'
      # version tag
      version: '0.1.0'

Add Datasets
------------
Datasets are files and/or folders within a project that are intended to be bundled
with a particular release. They are the the inputs and outputs of jobs, which together
comprise the workflow.

Add a ``release`` field in ``first-workflow.rme.yml``:

.. code-block:: yaml

    datasets:
      my_input: 'inputs/input-file.txt'
      my_result: 'results/output-file.txt'


Add Jobs
--------
Jobs are a set of terminal commands that are associated with an input and an output.

Add a ``jobs`` field, and make one called ``copy-file`` that takes in ``my_input`` and
outputs ``my_output``. The command ``cp`` copies the file in the first argument to the
file in the second.

We can use curly braces in a string to replace dataset names defined in the ``datasets`` field
with the actual file path.

.. code-block:: yaml

    jobs:
      copy-file:
        inputs:
          - my_input
        outputs:
          - my_output
        commands:
          - ['cp', '{my_input}', '{my_output}']

Environment Variables
---------------------
Environment variables can be defined for ``jobs``, and expressed
as requirements for a particular workflow. This is done by adding A
``requires-env`` field with a list environment variables. These can then
be used in jobs by adding ``${VARIABLE}`` anywhere in a job configuration.

Environment variables can be defined in a ``.env`` and will be loaded in
and used to expand variables in the workflow file. For rme to discover them
they need to be located:

1. a ``.env`` file located at the project's root directory for project-level environment variables.
2. a ``.env`` file in the same directory as the workflow for workflow level environment variables. This overloads project level environment variables.

Create a file in the project directory called ``.env`` and add the following to it:

.. code-block:: ini

    MY_NAME=READER

Then declare that environment variable as a requirement and add an echo
command to the copy-file job.

.. code-block:: yaml

  requires-env:
  # For something
  - MY_NAME

    jobs:
      copy-file:
        inputs:
          - my_input
        outputs:
          - my_output
        commands:
          - ['echo', 'hello ${MY_NAME}!']
          - ['cp', '{my_input}', '{my_output}']



Run Jobs
--------
You don't need to include the file extension when running
workflows, but you need to include the path to the workflow from
the root directory of your project.

.. code-block:: console

    rme run first-workflow

The runner will print the status of each jobs and their datasets, but not each of their standard
outputs - unless they return a failing status code. Each command is run sequentially within a job,
and checked for a successful status code. The standard output of each job is logged to a file.

You can view the log with the ``rme log`` command.

.. code-block:: console

    rme log copy-file


Mapping the Release
-------------------
When you release a workflow, you release the datasets and the project
structure that it was created in.

``rme`` respects ``.gitignore`` rules when determining what files to
include in the release, except for datasets which are included by
default.

Add this to a folder called ``.gitignore`` at the root of the project.
The .env file may contain sensitivie information, and the .rme file
contains runtime infromation that rme uses. Neither of those should be
commited to version control nor shared with a release, so add them
both to gitignore.

.. code-block:: none

    .rme/
    .env/

And add this to a ``results/.gitignore`` to ignore the contents of the
results folder but keep the folder itself under version control.

.. code-block:: none

    *
    !.gitignore

Check the release mapping with:

.. code-block:: console

    rme map first-workflow

You should see a printout of everything that will be included in the
release.

We can modify what files are included or excluded in the release
by adding the ``release.ignores`` and ``release.includes`` fields.
These fields can utilize curly braces to sub in dataset paths, and
folow the same glob pattern conventions as gitignore relative to the
project's root directory.

In this case, we opt in to include the output file, and elect to ignore the
input file. The default behaviour is to ignore everything not included by git ignore,
so if you want to include a dataset as part of the release then make sure to add it here.

.. code-block:: yaml

    release:
      # name of the release
      title: 'first-workflow'
      # version tag
      version: '0.1.0'
      # make custom include patterns
      includes:
        - '{my_output}'
      ignores:
        - 'inputs/'


At this point your workflow file should look like this:

.. code-block:: yaml

    release:
      # name of the release
      title: 'first-workflow'
      # version tag
      version: '0.1.0'
      # make custom include patterns
      includes:
        - '{my_output}'
      ignores:
        - 'inputs/'

    datasets:
      my_input: 'inputs/input-file.txt'
      my_result: 'results/output-file.txt'

    requires-env:
    # For something
    - MY_NAME

    jobs:
      copy-file:
        inputs:
          - my_input
        outputs:
          - my_output
        commands:
          - ['echo', 'hello ${MY_NAME}!']
          - ['cp', '{my_input}', '{my_output}']

Run the map command again and check that the input file is ignored,
and the output file is included.

.. code-block:: console

    rme map first-workflow

Publish a Release
-----------------
To publish a release, use the release command, replacing
``archive_host`` with the path to the archive you
are publishing to. The default workspace is the ``Global Public Workspace``.

.. code-block:: console

    rme release first-workflow <archive_host>
