# -*- coding: utf-8 -*-
"""
Using Annotated Arrays
======================

The primary use cases of AnnotatedArrays are to documenting datamodels for
users and developers, annotate functions which are expecting specific kinds
of data, validate data expected to conform to a paritcular model, and to
facilitate instantiation datamodels with complex structures.

"""

import rmellipse as rme
import numpy as np
from pydantic import BaseModel, ConfigDict

# %%
# Instantiation
# -------------
#
# Having a defined schema makes it easier to initialized empty arrays for
# a given datatype by using the AnnotatedArray.zero() method. Coordinate
# dimensions can be supplied, along with metadata you may need. Information
# that is expected. For example, TimeDomainVoltage2x2 is a rather complex
# data structure with a lot of requirements, including specific metadata. Using
# the AnnotatedArray format, you can generate an empty data set fairly
# concisely.


class MeasurementMetadata(BaseModel):
    # Enable extra fields
    model_config = ConfigDict(extra='allow')
    operator: str
    temperature_celcius: float


class TimeAndFrequencySweep(rme.AnnotatedArray):
    schema = rme.ArraySchema(
        shape=('N', 'M'),
        dims=('time', 'frequency'),
        dtype=float,
        coords={
            'time': rme.CoordinateSchema(dtype=float, units='s'),
            'frequency': rme.CoordinateSchema(dtype=float, units='GHz'),
        },
        attrs=MeasurementMetadata,
    )


voltages = TimeAndFrequencySweep.zeros(
    time=np.linspace(0, 100, 100),
    frequency=np.linspace(1, 10, 10),
    attrs={
        'operator': 'reader',
        'temperature_celcius': 23.5,
    },
)

current = TimeAndFrequencySweep.zeros(
    time=np.linspace(0, 100, 100),
    frequency=np.linspace(1, 10, 10),
    attrs={
        'operator': 'reader',
        'temperature_celcius': 22.5,
    },
)

# fill with random values
voltages[...] = np.random.rand(*voltages.shape)
current[...] = np.random.rand(*current.shape)

# %%
# Validation
# ----------
#
# If you already have an array that you think should perform to a particular
# schema, you can try validating it.

my_sample = np.zeros((2, 2))

try:
    TimeAndFrequencySweep(my_sample).validate()
except rme.ValidationError as e:
    print(f'array fails validation for : {e}')


# %%
# Annotating Functions
# --------------------
#
# Since the type is available, you can use it as part of Python's type
# annotation system to help document your code. For example, annotating
# functions. Suppose we had a dataset of multiple time sweeps of voltage
# and current measurements at multiple different frequencies. We wanted to
# determine what frequency the time average power was the highest, and
# what the approximate temperature was. We could write a function to do that,
# and then annotate the inputs and outputs of that function using the
# AnnotatedArray types we defined.


def max_time_average_power_frequency(
    voltages: TimeAndFrequencySweep, current: TimeAndFrequencySweep
) -> tuple[float, float]:
    """
    Get the frequency corresponding to the maximum time average power.

    Returns the frequency (in GHz) and the approximate temperature based
    on measurement metadata.

    Parameters
    ----------
    voltages : TimeAndFrequencySweep
        Voltage measurements.
    current : TimeAndFrequencySweep
        Current measurements.

    Returns
    -------
    frequency : float
        Frequency where the most time average power was measured
    temperature : float
        Approximate temperatue from metadata.
    """

    mean_temp = np.mean(
        [voltages.attrs['temperature_celcius'], current.attrs['temperature_celcius']]
    )
    power = voltages * current
    time_average = power.mean(dim='time')
    i_max = np.argmax(time_average.data)
    return float(time_average.frequency[i_max]), float(mean_temp)


freq, temp = max_time_average_power_frequency(voltages, current)

print(freq, temp)
