.. Rocky Mountain Ellipse documentation master file, created by
   sphinx-quickstart on Wed Jul 24 09:45:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Rocky Mountain Ellipse's documentation!
==================================================

Rocky Mountain Ellipse (RME) is a software package designed to provide an explicit digital record of the metrological
traceability of a measurement result. In other words, RME will allows users to build a record that describes how
a measurement result can be related to a reference through a documented unbroken chain of calibrations,
each contributing to the measurement uncertainty. To show the traceability of measurements,
RME provides a flexible, explicit system to organize and annotate scientific data and data analysis workflows.Because measurement
uncertainty is closely related to traceability, RME allows users to track how the uncertainty of a measurement results
derives from other measurements by providing tools to facilitate measurement uncertainty propagation (linear finite-difference and Monte-Carlo)
through arbitrary Python functions. The system is compatible with the BIPM Guide to the expression of uncertainty in measurement (GUM_).
RME is intended to be part of a FAIR_ software ecosystem that will facilitate re-use of code and data.
This vision includes an online archive that could eventually store records of NIST’s entire traceability chain, and beyond. As part of that vision,
this package focuses on the development of three core utilities:


Table of Contents
=================

.. toctree::
   :maxdepth: 3

   auto_examples/index.rst
   cli/index.rst
   for_contributors/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GUM: https://jcgm.bipm.org/vim/en/2.41.html
.. _FAIR: https://www.go-fair.org/fair-principles/