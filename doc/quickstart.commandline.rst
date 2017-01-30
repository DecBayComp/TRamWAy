.. _quickstart.commandline:

Command-line starter
====================

Input data
----------

The input data are tracking data of single molecules, stored in a text file.

The extension of input data files will usually be ``.trxyt`` but this is not required.

Partitioning the data points
----------------------------

The first step consists of tesselating the space and partitioning the data points into the cells of the tesselation::

	> python -m inferencemap tesselate -h
	> python -m inferencemap -i example.trxyt tesselate -m gwr

``-h`` shows the help message for the ``tesselate`` command. ``-i`` allows to specify the tracking data file. ``-m`` allows to specify the tesselation method. 

The available methods are:

* ``grid``: regular grid with equally sized square or cubic areas.
* ``kdtree``: kd-tree tesselation with midpoint splits.
* ``kmeans``: tesselation based on the k-means clustering algorithm.
* ``gwr``: tesselation based on the Growing-When-Required self-organizing gas.

``'kmeans'`` and ``'gwr'`` methods run optimization on the data and consequently are vulnerable to numerical scaling. It is recommended to scale your data adding the option ``-w``.


Visualizing the partition
-------------------------

To visualize on the screen::

	> python -m inferencemap -i example.imt.h5 show-cells

To print the figure in an image file::

	> python -m inferencemap -i example.imt.h5 show-cells -p png

To overlay the Delaunay graph instead of the Voronoi graph::

	> python -m inferencemap -i example.imt.h5 show-cells -D

