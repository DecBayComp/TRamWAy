.. _quickstart.helpers:

High-level library helpers
==========================

Input data
----------

The input data usually are tracking data of single molecules.

They can be provided as text files with four or five numerical columns, respectively the trajectory index, 2 or 3 spatial coordinates, and time in milliseconds.

Partitioning the data points
----------------------------

The first step consists of tesselating the space and partitioning the data points into the cells of the tesselation.

:mod:`inferencemap.helper.tesselation` exposes the :func:`~inferencemap.helper.tesselation.tesselate` method that be called the following way::

	path_to_tracking_data = 'example.trxyt'
	cells = tesselate(path_to_tracking_data, 'grid', avg_cell_count=40)

The input data space will be divided by a regular grid such that on average a cell will accomodate 40 points as specified by the ``avg_cell_count`` argument. This argument is optional with default value 80.

An ``example.imt.h5`` file will be generated.

The ``'grid'`` argument can be replaced by the following alternative method names:

* ``'kdtree'``: for a kd-tree tesselation.
* ``'kmeans'``: for a tesselation based on the k-means clustering algorithm.
* ``'gwr'``: for a tesselation based on the Growing-When-Required self-organizing gas.

.. note:: the ``avg_cell_count`` argument is equally useful with ``'kmeans'`` but not with the other methods.

``'kmeans'`` and ``'gwr'`` methods run optimization on the data and consequently are vulnerable to numerical scaling. It is recommended to scale your data::

	tesselate(path_to_tracking_data, method='kmeans', scaling=True)

The tesselation step takes several arguments supposed to control the number of points per cell (``min_cell_count``, ``avg_cell_count``, ``max_cell_count``). This count information is usually taken as hints, and the resulting partition may exhibit cells with a smaller number of points than indicated.

To control the cell size at the partition step, a useful option ``knn`` (read "`k` nearest neighbors") ensures an actual minimum number of points per cell and enables cell overlap if necessary::

	tesselate(path_to_tracking_data, method='gwr', scaling=True, knn=40)

``'grid'`` and ``'kdtree'`` methods usually have several totally empty cells and it may not make sense to extend such cells until they contain the required number of points. An (undocumented, as of version *0.1*) option applies a threshold that determines whether as cell is included or not depending on the number of points it has::

	tesselate(path_to_tracking_data, method='grid', knn=40, inclusive_min_cell_size=1)

In addition, ``'kdtree'`` requires an extra undocumented option so that the cells are not viewed as square or cubic areas which is not compatible with the concept of nearest neighbor::

	tesselate(path_to_tracking_data, method='kdtree', knn=40, inclusive_min_cell_size=1, metric='euclidean')

A maximum number of nearest neighbors can also be defined by providing a pair of integers (or ``None``) instead of a single integer::

	tesselate(path_to_tracking_data, method='gwr', knn=(None, 40))

The above expression takes only a maximum number of points per cell. The minimum is set to ``None`` and could instead be a number equal to or lower than the second value in the tuple.


Visualizing the partition
-------------------------

A simple command from the :mod:`inferencemap.helper.tesselation` module::

	cell_plot(cells)

The Delaunay graph can be overlain instead of the Voronoi graph::

	cell_plot(cells, xy_layer='delaunay')

Here ``cells`` can equally be a path to a ``.h5`` tesselation file or the object returned by :func:`~inferencemap.helper.tesselation.tesselate`.

The generated figure can be saved into a file instead of being shown on the screen::

	cell_plot(path_to_imt_file, fig_format='png')

or::

	cell_plot(cells_object, output_file='example.png')


Infering physical parameters
----------------------------

This step is not implemented yet.

