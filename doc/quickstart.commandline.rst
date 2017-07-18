.. _quickstart.commandline:

Command-line starter
====================

Input data
----------

The input data are tracking data of single molecules, stored in a text file.

The extension of input data files will usually be |xyt| or |trxyt| but this is not required.

Partitioning the data points
----------------------------

The first step consists of tesselating the space and partitioning the data points into the cells of the tesselation::

	> tramway tesselate -h
	> tramway -i example.trxyt tesselate -m kmeans

``-h`` shows the help message for the ``tesselate`` command. ``-i`` allows to specify the tracking data file. ``-m`` allows to specify the tesselation method. 

The available methods are:

* ``grid``: regular grid with equally sized square or cubic areas.
* ``kdtree``: kd-tree tesselation with midpoint splits.
* ``kmeans``: tesselation based on the k-means clustering algorithm.
* ``gwr``: tesselation based on the Growing-When-Required self-organizing gas.

``'kmeans'`` and ``'gwr'`` methods run optimization on the data and consequently are vulnerable to numerical scaling. It is recommended to scale your data adding the option ``-w``.

A key parameter is ``--knn`` (or shorter ``-n``). It combines with any of the above methods and allows to impose a lower bound on the number of points (or nearest neighbors) associated with each cell of the mesh, independently of the way the mesh has been grown::

	> tramway -i example.trxyt tesselate -m gwr -w -n 50


Visualizing the partition
-------------------------

To visualize on the screen::

	> tramway -i example.rwa show-cells

To print the figure in an image file::

	> tramway -i example.rwa show-cells -p png

To overlay the Delaunay graph instead of the Voronoi graph::

	> tramway -i example.rwa show-cells -D


.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*

