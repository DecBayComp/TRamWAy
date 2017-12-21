.. _quickstart.helpers:

High-level library helpers
==========================

.. warning:: This section is outdated and needs to be reworked.

Partitioning the tracking data
------------------------------

The first step consists of tesselating the space so that the point data can be partitioned in cells with an adequate number of points and spatial extent.

:mod:`tramway.helper.tesselation` exposes the :func:`~tramway.helper.tesselation.tesselate` method that be called the following way::

	path_to_tracking_data = 'example.trxyt'
	cells = tesselate(path_to_tracking_data, 'grid', avg_location_count=40)

The input data space will be divided by a regular grid such that on average a cell will accomodate 40 points as specified by the ``avg_cell_count`` argument. This argument is optional with default value 80 (more exactly ``4 * min_cell_count``).

An ``example.imt.h5`` file will be generated.

The ``'grid'`` argument can be replaced by the following alternative method names:

* ``'kdtree'``: for a kd-tree tesselation.
* ``'kmeans'``: for a tesselation based on the k-means clustering algorithm.
* ``'gwr'``: for a tesselation based on the Growing-When-Required self-organizing gas.

.. note:: the ``avg_cell_count`` argument is equally useful with ``'kmeans'`` but not with the other methods.

``'kmeans'`` and ``'gwr'`` methods run optimization algorithms on the data and consequently are vulnerable to numerical errors. It is recommended to scale your data::

	tesselate(path_to_tracking_data, method='kmeans', scaling=True)

The tesselation step takes several arguments supposed to control the number of points per cell (``min_cell_count``, ``avg_cell_count``, ``max_cell_count``). This count information is usually taken as hints, and the resulting partition may exhibit cells with a smaller number of points than indicated.

To control the cell size at the partition step, a useful option ``knn`` (read "`k` nearest neighbors") ensures an actual minimum number of points per cell and enables cell overlap if necessary::

	tesselate(path_to_tracking_data, method='gwr', scaling=True, knn=40)

``'grid'`` and ``'kdtree'`` methods usually have several totally empty cells and it may not make sense to extend such cells until they contain the required number of points. The ``inclusive_min_cell_size`` argument determines how many points a cell should contain to be then extended if necessary. If there are too few points, the cell is ignored and not any point will be associated to this cell::

	tesselate(path_to_tracking_data, method='grid', knn=40, inclusive_min_cell_size=1)

In addition, ``'kdtree'`` requires an extra argument so that the cells are not viewed as square or cubic areas. Indeed, square cells have such a variable size that a point in a cell may be closer to the center of another cell. This peculiarity is not compatible with the concept of nearest neighbor::

	tesselate(path_to_tracking_data, method='kdtree', knn=40, inclusive_min_cell_size=1, metric='euclidean')

A maximum number of nearest neighbors can also be defined by providing a pair of integers (or ``None``) instead of a single integer::

	tesselate(path_to_tracking_data, method='gwr', knn=(None, 40))

The above expression takes only a maximum number of points per cell. The minimum is set to ``None`` and could instead be a number equal to or lower than the second value in the tuple.


Visualizing the partition
-------------------------

A simple command from the :mod:`tramway.helper.tesselation` module::

	cell_plot(cells)

The Delaunay graph can be overlain instead of the Voronoi graph::

	cell_plot(cells, xy_layer='delaunay')

Here ``cells`` can equally be a path to a |h5| tesselation file or the object returned by :func:`~tramway.helper.tesselation.tesselate`.

The generated figure can be saved into a file instead of being shown on the screen::

	cell_plot(path_to_imt_file, fig_format='png')

or::

	cell_plot(cells_object, output_file='example.png')


Infering physical parameters
----------------------------

The data should first be prepared::

	from tramway.inference import *

	prepared_map = Distributed(cells)

For now only the diffusivity can be estimated::

	diffusivity_map = inferD(prepared_map, localization_error=0.2)

See also :mod:`tramway.inference`.

Visualizing maps
----------------

::

	from tramway.plot.map import *
	import matplotlib.pyplot as plt

	plot_scalar_2d(diffusivity_map)
	plt.show()

See also :mod:`tramway.plot.map`.

.. |h5| replace:: *.h5*

