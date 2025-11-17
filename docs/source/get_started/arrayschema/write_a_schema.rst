Add an Array Schema to a Project
================================

Using the project from the , create a file called
``current-measurements.json``. This schema describes a format
for array data that can have arbitrary leading dimensions, with
two dimensions called "time" and "voltage" as the trailing dimensions
with shape N and D. The data in the array should be ``f8`` (64 bit float)
, and the coordinates time and voltage shouuld also be floats with units
``s`` and ``V`` respectivley. In addition, we require that each array with
this specification include string metadata in a field called ``date_created``

.. code-block:: json

    {
        "name": "current-measurements",
        "shape": [
            "...",
            "N",
            "D"
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






