Why Array Schema?
=================
Tools like xarray_ have shown to be very useful for creating self describing data of homogenous
data sets, which carry with them coordinate arrays, labels, and potentiolly complex metadata.
While it's convenient to make rich, descriptive data, writing scripts or functions is difficult, as
it becomes difficult to describe the specification of the structure of data that a function or script is
expecting. Managing a project with multiple different data formats, each with arbitrarily complex specifications
of what structure, and metadata, is required or even allowed, becomes hard to manage. It also becomes
difficult, when looking at the data, to parse what specification that dataset is trying to adhere to.

Arrschema (Array schema) is a tool for annotating the structure of homogenous array-like data
and its metadata, along with the specification that data is intended to meet.

They are designed to be JSON documents so they can be uploaded to
and discoverable by databases like mongodb. In general, array schema are intended
to be an annotation tool, but provides some utility for managing encoding, decoding, converting
between schemas, and validating datasets against schemas for the Python and xarray package.
The model of an array schema takes heavy inspiration of the DataArray model made by xarray_
and the model of data with metadata used in HDF5_. It is intended to work closely with both packages,
but they are not required for annotation.

For example, imagine a schema to describe some current measurements that were made
by a supplied voltage.

.. code-block:: json

    {
        "name":"current-measurements"
        "shape": ["..."]
    }

Shape can be specified further by adding integers to indicate fixed length
dimensions, and letter to indicate variable length dimensions. In this
example the array has arbitrary leading dimensions, and the second to last dimensions
is length N, and the final dimension is length 2.

.. code-block:: json

    {
        "name": "current-measurements",
        "shape": [
            "...",
            "N",
            2
        ]
    }



Dimension names can be added to describe their meaning. Any specified dimension (i.e. not an ellipses)
must have a defined dimension name. Now we are describing a multidimensional array with arbitrary leading
dimensions, where the second to last dimension describes the various times made, and the last dimension
describes the voltage source level.

.. code-block:: json

    {
        "name": "current-measurements",
        "shape": [
            "...",
            "N",
            2
        ],
        "dim": [
            "...",
            "time",
            "voltage_source"
        ]
    }


We can specify the expected type of the data following numpy dtype_ string conventions.
In this case we want our data to be 64 bit floating point numbers.

.. code-block:: json

    {
        "name": "current-measurements",
        "shape": [
            "...",
            "N",
            2
        ],
        "dim": [
            "...",
            "time",
            "voltage_source"
        ],
        "dtype": "f8"
    }

We can add descriptions about the coordinate sets required
as part of the schema's specification - the ``time`` and ``voltage_source``
dimensions. In this case, we specify that that the time dimension is in units
of ``s``, and the voltage source dimension is in units of ``V``. If you are using
xarray, then those coordinates are bundled with the data. Otherwise, when sharing
data with schemas and specified coordinates the coordinate data should be provided
as well.

.. code-block:: json

    {
        "name": "current-measurements",
        "shape": [
            "...",
            "N",
            2
        ],
        "dim": [
            "...",
            "time",
            "voltage_source"
        ],
        "dtype": "f8",
        "coords": {
            "time": {
                "units": "s",
                "dtype": "f8"
            },
            "voltage_source": {
                "units": "V",
                "dtype": "f8"
            }
        }
    }


Finally, often times there are data formats with required metadata. Metadata
is modelled after the HDF5_ model of data with metadata, and is specified with
the ``attrs_schema`` key. Arrschema allow you to specify metadata format using
`JSON Schema`_.

.. code-block:: json

    {
        "name": "current-measurements",
        "shape": [
            "...",
            "N",
            2
        ],
        "dim": [
            "...",
            "time",
            "voltage_source"
        ],
        "dtype": "f8",
        "coords": {
            "time": {
                "units": "s",
                "dtype": "f8"
            },
            "voltage_source": {
                "units": "V",
                "dtype": "f8"
            }
        },
        "attrs_schema": {
            "type": "object",
            "properties": {
                "date_created": {
                    "type": "string"
                },
                "required": [
                    "date_created"
                ]
            }
        }
    }
    }



.. _xarray: https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataarray
.. _dtype: https://numpy.org/doc/stable/reference/arrays.dtypes.html
.. _JSON schema: https://json-schema.org/
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/