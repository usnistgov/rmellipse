.. _manage-requirements:
Manage Requirements
===================
This page will walk through the steps to managing
requirements for your workflow.

It assumes you have followed the
previous steps in the :ref:`getting-started` page.

Make a New Workflow
-------------------
Lets start a new project that depends on the the results of the previous
one. Make a new folder called ``second-workflow``, and copy the contents from
:ref:`first-workflow` into it. Then, change the name of the ``first-workflow.rme.yml``
to ``second-workflow.rme.yml`` and update the release title to ``second-workfow``.

.. code-block:: yaml

    release:
      # name of the release
      title: 'second-workflow'
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

    jobs:
      copy-file:
        inputs:
          - 'my_input'
        outputs:
          - 'my_output'
        commands:
          - ['cp', '{my_input}', '{my_output}']


Add Workflow Requirements
-------------------------
Requirments are declared by adding a ``requires-releases`` field to one of two places:

1. Add a project wide requirment by adding it to ``rmeproject.yml``
2. Add a workflow specific requirement by adding it to ``second-workflow.rme.yml``

A project is a folder, that may contain more then one workflow. A project wide requirement
is a requirement shared by every workflow in a project. A workflow specific dependency is only
shared by the workflow it is included in.

For example,

* some-release==0.1.0
* some-release>=0.1.0
* some-release>=0.1.0, <0.2.0

Make the ``first-workflow`` a requirement of ``second-workflow`` by adding it
to the requires-releases.

.. code-block:: yaml

    requires-releases:
      - 'first-workflow>=0.1.0'

    release:
      # name of the release
      title: 'second-workflow'
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

    jobs:
      copy-file:
        inputs:
          - 'my_input'
        outputs:
          - 'my_output'
        commands:
          - ['cp', '{my_input}', '{my_output}']

Run the ``sync`` command in order to synce your new project.

.. code-block:: console

    rme sync second-workflow

This will create a data environment folder called ``data_env`` inside your project. The data environment
will store shortcuts to all the data you've declared in your requirements file.

Datasets exists within the context of a project folder, that project folder structure is recreated and soft-linked to inside the ``.packages`` folder. If you
go into ``data_env/.packages/first-workflow-v0.1.3`` you will see the project we made previously. Individual files are soft-linked
to the actual files, which are stored in a cache that rme manages.

In the root folder of ``data_env/`` are the registry folders. These folders contain recreations of only the datasets released by a particular
workflow, that have been renamed based on the name for the dataset defined in the workflow file. This is the intended access point for data
within a particular workflow. The extensions of the original datasets (if they have one) are copied onto these registry links to make it easier
to identify what types of files there are.

A requirements file can declare multiple versions of the same workflow, folders in the registry and the .packages folder are versioned. However,
for convenience, the registry will duplicate the most recent version of a particular workflow without a version identifier. This way, you can structure
your code to use the links in the "most recent"" links. If the datasets are updated to newer versions, your code will automatically pull from the most
recent requirement.


Add File System Requirements
----------------------------

It's also possible to declare dependencies on files stored on your local file system
(or on a network drive). You can do that with a ``requires-files`` field. You declare
a folder you want to group the files under, then assign the files to that folder.

This lets you organize files spread out across multiple folders and network drive locations
in a way that is logical and coherent to your project, without having to duplicate the data.

While it's preffered to have external data dependencies be derived from release packages,
so their provenance can be inferred from the package metadata, in cases where that isn't possible
this at least provides a way to track where your data is coming from in a reproducable way.

This example creates a folder in data_env called ``my-files``, and inside that folder a file
called ``my-file.ext`` and a subfolder called ``some-sub-folder`` that links to the files on
a network drive stored in ``${network-drive-path}/path/to/folder``.

.. code-block:: yaml

  requires-files:
    # group under a folder name space
    my-files:
      # link out to files
      my-file: 'path/to/external/file.ext'
      some-sub-folder: '${network-drive-path}/path/to/folder'


Using environment variables allows sensitive information like network-drive paths to be
hidden in the config file.

Run the New Workflow
--------------------

Update the ``datasets`` field of our second workflow to that ``my_input`` points to the datasets
``my_result`` of our first workflow.

.. code-block:: yaml

    datasets:
      my_input: 'data_env/first-workflow/my_result.txt'
      my_result: 'results/output-file.txt'


If we run our second workflow,

.. code-block:: console

    rme run second-workflow

you will notice that the environment is synchronized. The core idea is that rme will
always attempt to synchronize your data environment with the definition of your data requirements.

In addition, you will notice that the after synchronizing your environment for the fist time, there will
be a file called ``second-workflow.rme.lock``. The lock file stores the solution to your data requirements
for a given workflow.

For example, our requirement  ``first-workflow>=0.1.0`` is statisfied by any workflow ``first-workflow`` with
a version greater then or equal to 0.1.0. To facilitate recreating analysis, the lock file stores the exact solution
to that requirement found at runtime. If you wish to update a dataset, then you will need to update the lockfile.