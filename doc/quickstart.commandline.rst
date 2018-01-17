.. _quickstart.commandline:

Command-line starter
====================

Input data
----------

The input data are tracking data of single molecules, stored in a text file.

The extension of input data files will usually be |xyt| or |trxyt| but this is not required.

Partitioning the data points
----------------------------

The first step consists of tessellating the space and partitioning the data points into the cells of the tessellation::

	> tramway tessellate -h
	> tramway -i example.trxyt tessellate kmeans -l my-mesh*

``-h`` shows the help message for the ``tessellate`` command. 
``-i`` permits to specify the tracking data file, ``kmeans`` is the tessellation method and ``-l`` specifies a label to attach to the analysis for further reference.
The ``*`` symbol will be replaced by a natural integer starting from 0.

The available methods are:

* ``grid``: regular grid with equally sized square or cubic areas.
* ``kdtree``: kd-tree tessellation with midpoint splits.
* ``kmeans``: tessellation based on the k-means clustering algorithm.
* ``gwr`` (or ``gas``): tessellation based on the Growing-When-Required self-organizing gas.

``kmeans`` and ``gwr`` methods run optimization on the data and consequently are vulnerable to numerical scaling. 
It is recommended to scale your data adding the option ``-w``.

A key parameter is ``--knn`` (or shorter ``-n``). 
It combines with any of the above methods and allows to impose a lower bound on the number of points (or nearest neighbors) associated with each cell of the mesh, independently of the way the mesh has been grown::

	> tramway -i example.rwa tessellate gwr -w -n 50 -l my-mesh*

Note that in the above example the *example.rwa* file already exists and we add the analysis.
The ``*`` again prevents from overwriting an analysis with the same label, if any.

You can check the content of the *example.rwa* file::

	> tramway -i example.rwa dump

	in example.rwa:
		<class 'pandas.core.frame.DataFrame'>
			'my-mesh0' <class 'tramway.tessellation.base.CellStats'>
			'my-mesh1' <class 'tramway.tessellation.base.CellStats'>


Visualizing the partition
-------------------------

To visualize on the screen::

	> tramway -i example.rwa draw cells -L my-mesh0

To print the figure in an image file::

	> tramway -i example.rwa draw cells -L my-mesh0 -p png

To overlay the Delaunay graph instead of the Voronoi graph::

	> tramway -i example.rwa draw cells -L my-mesh0 -D

Inferring diffusivity and force
-------------------------------

DF mode::

	> tramway -i example.rwa infer df -L my-mesh0 -l my-map*

Other inference modes are ``d``, ``dd`` and ``dv``.

Visualizing maps
----------------

::

	> tramway -i example.rwa draw map -L my-mesh0,my-map0


.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*

