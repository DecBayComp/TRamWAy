# TRamWAy

*TRamWAy* helps analyzing single molecule dynamics. It infers the diffusivity, drift, force and potential across space and time. It will eventually localize and track molecules in stacks of images.

**Disclaimer:**
This implementation is under heavy development and is not yet suitable even for testing purposes!
To get similar software, please follow [this link](https://research.pasteur.fr/en/software/inferencemap/ "research.pasteur.fr/en/software/inferencemap").

The [official documentation](http://TRamWAy.readthedocs.io/en/latest/) is now on [readthedocs](http://TRamWAy.readthedocs.io/en/latest/).

## Installation

You will need Python >= 2.7 or >= 3.5.

	git clone https://github.com/DecBayComp/TRamWAy
	cd TRamWAy
	pip install --user -e .

`pip install` will install some Python dependencies if missing, but you may still need to install the [HDF5 reference library](https://support.hdfgroup.org/downloads/index.html "support.hdfgroup.org/downloads").

Get the documentation locally with:

	cd doc
	make html

The html documentation will be available at `_build/html/index.html` from the `doc` repository.

