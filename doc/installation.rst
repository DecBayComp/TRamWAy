.. _installation:

Installation
============

You will need Python >= 3.6.

::

	pip install --user tramway

or::

	git clone https://github.com/DecBayComp/TRamWAy
	cd TRamWAy
	pip install --user .


``pip install`` will install some Python dependencies if missing, but you may still need to install the `HDF5 reference library <https://support.hdfgroup.org/downloads/index.html>`_.

Note also that Python3's *pip* command may be available as *pip3* and installing |tramway| for Python2 or Python3 does not make it available for Python3 or Python2 respectively.
You may want to install |tramway| twice, once for each Python version.

To compile the documentation and get a local copy, after installing |tramway| do::

	cd doc
	make html

The generated documentation will be available at ``_build/html/index.html`` from the ``doc`` repository.

Building the documentation requires Sphinx, preferably for Python3.


Notes on Python versions
------------------------

Some modules require Python>=3.7.
These are *bayes_factor* for force “detection” based on Bayesian statistics, *snr* that extracts signal-to-noise ratios required by the *bayes_factor* module, and *d.conj_prior* which estimates the diffusion similarly to *ddrift* with no regularization but with additional confidence intervals.


(Old) note for installation on Windows
--------------------------------------

If after following the above steps, launching |tramway| on Windows generates the following error:

::

	ImportError: HDFStore requires PyTables, "DLL load failed: The specified procedure could not be found." problem importing

please remove the ``PyTables`` version installed by ``pip`` and manually download and install the appropriate version from the following `web page <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables>`_. This may require an existing installation of `Visual C++ Build Tools 2015 <https://go.microsoft.com/fwlink/?LinkId=691126>`_.

