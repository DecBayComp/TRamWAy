.. _installation:

Installation
============

You will need Python >= 3.8.

Windows users
-------------

Please favor :ref:`Conda <conda>`, as Conda will seamlessly install the HDF5 standard library which is a required dependency.

From PyPI
---------

::

	pip install tramway

`pip install` will install some Python dependencies if missing, but you may still need to install the `HDF5 reference library <https://support.hdfgroup.org/downloads/index.html>`_.

Note that the HDF5 library can usually be installed using any OS' package manager.
Only Windows users may have to :ref:`manually download and install the HDF5 library <hdf5_install>`, if they do not wish to use Conda instead of pip.

Several installation targets are available; they install optional dependencies. For example (recommended)::

        pip install tramway[roi,animate]

Most of the functionalities and code examples described in this documentation will run without the optional dependencies this target installs in addition to the required dependencies.
It is safe to first install |tramway| with minimal requirements and then ``pip install`` the missing dependencies as you hit ``ImportError`` while using |tramway|.

.. _conda:
Using Conda
-----------

::

        conda install tramway -c conda-forge

Compared with the bare package ``pip`` installs (with no installation targets), the ``conda`` package specifies some optional dependencies as required, so that the above command installs them.
However, many other optional dependencies are omitted.

Git version
-----------

Initial install::

	git clone https://github.com/DecBayComp/TRamWAy
	cd TRamWAy
	pip install -e .[roi,animate]

Can be updated with ``git pull`` run in the local repository.

Documentation
-------------

To compile the documentation and get a local copy, after installing |tramway| do::

	cd doc
	make html

The generated documentation will be available at ``_build/html/index.html`` from the ``doc`` repository.

Building the documentation requires Sphinx.

