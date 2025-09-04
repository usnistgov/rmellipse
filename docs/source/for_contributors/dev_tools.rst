Using the Developer Tools
=========================

Clone the repo and run, from the root directory:

.. code-block:: console

    uv sync

This will generate the virtual environment for the package.

Local Tests
------------

In git bash run:

.. code-block:: console

    tools/test.sh
    tools/test.sh open

The open command will launch a view of a webpage with
missing code coverage highlighted.

Local Documentation
-------------------

Clean the local documentation build (this needs to be run sometimes
if you are modifying the documentation and it gets into a broken state). It will reset the build directories and the next call to
build it will be completely from scratch.

.. code-block:: console

    tools/docs.sh clean


To build the documentation.

.. code-block:: console

    tools/docs.sh html



Serve the build files (this will freeze the console).
Files are served on localhost on port 8000.

.. code-block:: console

    tools/docs.sh serve

Open the webpage (in a seperate console).

.. code-block:: console

    tools/docs.sh open


Code Profiling
--------------
You can profile test scripts. This will launch a localhost webpage
on port 8080. Test scripts should go in the tests directory.

.. code-block:: console

    tools/profile.sh tests/<test_script_name.py>
