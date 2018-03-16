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


