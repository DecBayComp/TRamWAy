# InferenceMAP

This re-implementation of InferenceMAP for now focuses on meshing the space of the moving particles, as a preliminary step for infering spatial maps of diffusivity and potential.

**Disclaimer:**
This implementation is under heavy development and is not yet suitable even for testing purposes!
To get the official implementation, please follow [this link](https://research.pasteur.fr/en/software/inferencemap/ "research.pasteur.fr/en/software/inferencemap").

The [official documentation](http://inferencemap.readthedocs.io/en/latest/) is now on [readthedocs](http://inferencemap.readthedocs.io/en/latest/).

## Installation

You will need Python >= 2.7 or >= 3.5.

	git clone https://github.com/influencecell/inferencemap
	cd inferencemap
	pip install -U -e .

`pip install` will install some Python dependencies if missing, but you may still need to install the [HDF5 reference library](https://support.hdfgroup.org/downloads/index.html "support.hdfgroup.org/downloads").

Get the documentation locally with:

	cd doc
	make html

The html documentation will be available at `_build/html/index.html` from the `doc` repository.
