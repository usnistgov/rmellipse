# Rocky Mountain Ellipse
This package is a tool for tracable uncertainty analysis. It is backwards compatable with the
[Microwave Uncertainty Framework](https://www.nist.gov/services-resources/software/wafer-calibration-software)
and built using the [xarray](https://docs.xarray.dev/en/stable/) package.


### Recommendations
It is recommended to download [HDFView](https://www.hdfgroup.org/downloads/hdfview/) if you haven't already.

It is also recommended to briefly familiarize yourself with the [xarray](https://docs.xarray.dev/en/stable/) package.

### Installing with pip
```
pip install rmellipse
```

### Installing from source with pip
Clone this repo and switch to the cloned directory. In your virtual env,
install in developer mode.
```
pip install -e .
```

Add additional dependencies for documentation and unit tests.

```
pip install -r dev_deps.txt
```

## Building documentation locally
In the command prompt, from the cloned directory.

```
make_docs clean
make_docs html
``` 

## Running tests locally

With your virtual-enviornment activated run:
```
test
```
This will generate a coverage report in the terminal, as well searchable html coverage report in
[htmlcov/index.html](htmlcov/index.html) that highlights the
lines of code in each file that weren't covered by the tests.

## Authors

Contributors names and contact info

Daniel C. Gray, Zenn C. Roberts, Aaron M. Hagerstrom

## Version History

* 0.1
    * Initial development as an internal tool at NIST's RF Technology Division.

