.. _installation:

Installation
============

You will need Python >= 3.6.

From PyPI
---------

::

	pip install tramway

`pip install` will install some Python dependencies if missing, but you may still need to install the `HDF5 reference library <https://support.hdfgroup.org/downloads/index.html>`_.

Several installation targets are available, including ``full``, that install optional dependencies::

        pip install tramway[full]

Most of the functionalities and code examples described in this documentation will run without the optional dependencies this target installs in addition to the required dependencies.
It is safe to first install |tramway| with minimal requirements and then ``pip install`` the missing dependencies as you hit ``ImportError`` while using |tramway|.

Using Conda
-----------

::

        conda install tramway

Compared with the bare package ``pip`` installs (with no installation targets), the ``conda`` package specifies some optional dependencies as required, so that the above command installs them.
However, many other optional dependencies are omitted.

Git version
-----------

Initial install::

	git clone https://github.com/DecBayComp/TRamWAy
	cd TRamWAy
	pip install -e .

Can be updated with ``git pull`` run in the local repository.

Documentation
-------------

To compile the documentation and get a local copy, after installing |tramway| do::

	cd doc
	make html

The generated documentation will be available at ``_build/html/index.html`` from the ``doc`` repository.

Building the documentation requires Sphinx, preferably for Python3.


OS and version specific notes
-----------------------------

Some modules require *Python>=3.7*.
These are *bayes_factor* for force “detection” based on Bayesian statistics, *snr* that extracts signal-to-noise ratios required by the *bayes_factor* module, and *d.conj_prior* which estimates the diffusion similarly to *ddrift* with no regularization but with additional confidence intervals.


(Old) note for installation on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If after following the above steps, launching |tramway| on Windows generates the following error:

::

	ImportError: HDFStore requires PyTables, "DLL load failed: The specified procedure could not be found." problem importing

please remove the ``PyTables`` version installed by ``pip`` and manually download and install the appropriate version from the following `web page <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables>`_. This may require an existing installation of `Visual C++ Build Tools 2015 <https://go.microsoft.com/fwlink/?LinkId=691126>`_.

