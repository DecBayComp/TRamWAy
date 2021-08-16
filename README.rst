TRamWAy
=======

The *TRamWAy* library features various tools for the analysis of particle dynamics in Single Molecule Localization Microscopy (SMLM) data.
It can resolve the diffusivity, drift, force and potential energy in space and time.

The `original documentation <https://tramway.readthedocs.io>`_ is now on `readthedocs <https://tramway.readthedocs.io>`_.

An attempt to rewrite the project documentation is available as a `separate project and web resource <https://tramway-tour.readthedocs.io>`_.

Installation
------------

You will need Python >= 3.6.

From PyPI
^^^^^^^^^

::

	pip install tramway

`pip install` will install some Python dependencies if missing, but you may still need to install the `HDF5 reference library <https://support.hdfgroup.org/downloads/index.html>`_.

Several installation targets are available, including `full`, that install optional dependencies::

        pip install tramway[full]

Most of the functionalities and code examples described in the documentation will run without optional dependencies.
It is safe to first install *TRamWAy* with minimal requirements and then `pip install` the missing dependencies as you hit `ImportError` while using *TRamWAy*.

Using Conda
^^^^^^^^^^^

::

        conda install tramway -c conda-forge

Compared with the bare package `pip` installs (with no installation targets), the `conda` package specifies some optional dependencies as required, so that the above command installs them.
However, many other optional dependencies are omitted.

