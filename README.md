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
	pip install .

`pip install` will install some Python dependencies if missing, but you may still need to install the [HDF5 reference library](https://support.hdfgroup.org/downloads/index.html "support.hdfgroup.org/downloads").

## Command line

### General usage

	python -m inferencemap [options] command [command-options]

Where `command` for now can be `tesselate` or `show-cells`.

Especially, the command line help is available typing either:

> python -m inferencemap -h

or:

> python -m inferencemap command -h

### Meshing

Considering an example trajectory file at location `data/glycine_receptor.trxyt`:

> python -m inferencemap -v -i glycine_receptor.trxyt tesselate -m gwr

A `glycine_receptor.imt.h5` file will be generated. This file contains both the tesselation and the partition.

#### Available methods and options
The `-m` option controls the tesselation method. The following methods are available:

* Regular mesh: `-m grid`
* k-d-tree (or quadtree in the 2D case); `-m kdtree`
* k-means-based mesh: `-m kmeans`
* Growing When Required-based mesh: `-m gwr`

A key parameter is `-knn`. It combines with any of the above methods and allows to impose an upper bound on the number of points (or nearest neighbors) associated with each cell of the mesh, independently of the way the mesh has been grown.

For more options:

	python -m inferencemap tesselate -h

### Plotting

	python -m inferencemap -i glycine_receptor.imt.h5 show-cells
	python -m inferencemap -i glycine_receptor.imt.h5 show-cells -H cdp
	python -m inferencemap -i glycine_receptor.imt.h5 show-cells --print png
(to do)


## Library

A comprehensive Sphinx-generated package documentation will soon be available.

The main functionalities are exposed in the :mod:`inferencemap.helper` package and, for now, the onlye module :mod:`inferencemap.helper.tesselation`.


## Data file structure

### .imt.h5 tesselation file

* `cells`:
	* `points`: point coordinates (original data; size `n` x `d+`)
	* `cell_index`: index (from `0` to `c - 1`) of the containing cell (size `n`)
	* `cell_count`: number of points per cell (size `c`)
	* etc (to do)
	* `tesselation`:
		* `cell_centers`: centroid coordinates (size `c` x `d`)
		* `cell_adjacency`: CSR sparse matrix representing the adjacency relationship between cells (size `c` x `c`)
		* etc (to do)

