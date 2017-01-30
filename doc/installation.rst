.. _installation:

Installation
============

You will need Python >= 2.7 or >= 3.5.

::

	git clone https://github.com/influencecell/inferencemap
	cd inferencemap
	pip install -U -e .

The ``-e`` option is necessary if you intend to update or modify the code and have the modifications reflected in your installed *InferenceMAP*.

``pip install`` will install some Python dependencies if missing, but you may still need to install the [HDF5 reference library](https://support.hdfgroup.org/downloads/index.html "support.hdfgroup.org/downloads").

To compile the documentation and get a local copy, after installing *InferenceMAP* do::

	cd doc
	make html

The generated documentation will be available at ``_build/html/index.html`` from the ``doc`` repository.

