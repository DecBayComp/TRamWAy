
Command-line
============

General usage
-------------
::

	python -m inferencemap [options] command [command-options]

Where `command` for now can be `tesselate` or `show-cells`.

Especially, the command line help is available typing either::

	> python -m inferencemap -h

or::

	> python -m inferencemap command -h

Meshing
-------

Considering an example trajectory file at location `data/glycine_receptor.trxyt`::

	> python -m inferencemap -v -i glycine_receptor.trxyt tesselate -m gwr

A `glycine_receptor.imt.h5` file will be generated. This file contains both the tesselation and the partition.

=============================
Available methods and options
=============================

The `-m` option controls the tesselation method. The following methods are available:

	* Regular mesh: `-m grid`
	* k-d-tree (or quadtree in the 2D case); `-m kdtree`
	* k-means-based mesh: `-m kmeans`
	* Growing When Required-based mesh: `-m gwr`

A key parameter is `-knn`. It combines with any of the above methods and allows to impose an upper bound on the number of points (or nearest neighbors) associated with each cell of the mesh, independently of the way the mesh has been grown.

For more options::

	python -m inferencemap tesselate -h

Plotting
--------
::

	python -m inferencemap -i glycine_receptor.imt.h5 show-cells
	python -m inferencemap -i glycine_receptor.imt.h5 show-cells -H cdp
	python -m inferencemap -i glycine_receptor.imt.h5 show-cells --print png

(to do)

