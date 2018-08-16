.. _installation:

Installation
============

You will need Python >= 2.7 or >= 3.5.

::

	pip install --user tramway

or::

	git clone https://github.com/DecBayComp/TRamWAy
	cd TRamWAy
	pip install --user -r requirements.txt


``pip install`` will install some Python dependencies if missing, but you may still need to install the `HDF5 reference library <https://support.hdfgroup.org/downloads/index.html>`_.

To compile the documentation and get a local copy, after installing |tramway| do::

	cd doc
	make html

The generated documentation will be available at ``_build/html/index.html`` from the ``doc`` repository.


Notes for Installation on Windows
---------------------------------

If after following the above steps, launching |tramway| on Windows generates the following error:

::

	ImportError: HDFStore requires PyTables, "DLL load failed: The specified procedure could not be found." problem importing

please remove the ``PyTables`` version installed by ``pip`` and manually download and install the appropriate version from the following `web page <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables>`_.

