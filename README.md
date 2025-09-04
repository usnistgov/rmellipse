# Rocky Mountain Ellipse
Rocky Mountain Ellipse (RME) is a software package designed to provide an explicit digital record of the metrological traceability of a measurement result. In other words, RME will allows users to build a record that describes how a measurement result can be related to a reference through a documented unbroken chain of calibrations, each contributing to the [measurement uncertainty](https://jcgm.bipm.org/vim/en/2.41.html). To show the traceability of measurements, RME provides a flexible, explicit system to organize and annotate scientific data and data analysis workflows.Because measurement uncertainty is closely related to traceability, RME allows users to track how the uncertainty of a measurement results derives from other measurements by providing tools to facilitate measurement uncertainty propagation (linear finite-difference and Monte-Carlo) through arbitrary Python functions. The system is compatible with the BIPM Guide to the expression of uncertainty in measurement (GUM). RME is intended to be part of a [FAIR](https://www.go-fair.org/fair-principles/) software ecosystem that will facilitate re-use of code and data. This vision includes an online archive that could eventually store records of NIST’s entire traceability chain, and beyond.


## Installation
Install the most recent stable build with [uv](https://docs.astral.sh/uv/):

```
uv add git+https://github.com/usnistgov/rmellipse --branch stable
```

Or with pip:

```
pip install git+https://github.com/usnistgov/rmellipse@stable
```

## Developer Tools
It is assumed you have the following programs installed on your computer.

* [uv](https://docs.astral.sh/uv/) for package management
* [git bash](https://git-scm.com/downloads) or similar terminal emulator to run shell scripts if you are on windows.

Clone the repo and run, from the root directory:
```
uv sync
```
This will generate the virtual environment for the package.

### Running Local Tests
In a bash terminal, run:

```
tools/test.sh
tools/test.sh open
```
This should execute all the defined tests with
[pytest](https://docs.pytest.org/en/stable/). In addition, the
open command will open a webpage with detailed reports about
code coverage.

### Building Local Documentation
Clean the local documentation build (this needs to be run sometimes
if you are modifying the documentation and it gets into a broken state). It will reset the build directories and the next call to
build it will be completely from scratch.
```
tools/docs.sh clean
```

To build a copy of the current state
of the documentation with your changes run:
```
tools/docs.sh html
```

To build the full documentation with tagged
versions and the most recent stable and development
changes (this takes a while and is usually only run
as part of the release jobs) run:

```
tools/docs.sh html-multiversioned
```

To open the documentation (you may have to call open
in a newe console) run

```
tools/docs.sh serve
tools/docs.sh open
```

This will serve a local copy of the documentation on your local host,
and the open command will launch your default web browser directly to that page.
Currently, this web page is on port 8000.

### Code Profiling
If you are writing a test script, you can run it in a code
profile from the cloned directory. This will open a webpage
to navigate the statistics of your tests script once it
complete.

```
tools/profile.sh <path/to/script.py>
```

## Authors

Contributors names and contact info

Daniel C. Gray, Zenn C. Roberts, Aaron M. Hagerstrom

