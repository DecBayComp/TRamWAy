.. _commandline:

Command-line starter
====================

Input data
----------

The input data are tracking data of single molecules, stored in a text file.

The extension of input data files will usually be |txt|, |xyt| or |trxyt| but this is not required.

We will first import an example |trxyt| file and sample the molecule locations in a step referred to as "tessellation", before we perform the inference and feature extraction.

Note that there is no standalone import.
Molecule locations and trajectories should always be sampled in a way or another.

All the analyses derivating from a same dataset are stored in a single |rwa| file.

.. seealso::

	:ref:`datamodel`


.. _commandline_tessellation:

Partitioning the data points
----------------------------

The first step consists of tessellating the space and partitioning the data points into the cells of the tessellation::

	> tramway tessellate -h
	> tramway tessellate kmeans -i example.trxyt -l kmeans

``-h`` shows the help message for the ``tessellate`` command. 

``-i`` takes the input data file, the ``kmeans`` subcommand is the tessellation method and ``-l`` specifies a label to attach to the analysis for further reference.

The available methods are:

* ``grid``: regular grid with equally sized square or cubic areas.
* ``kdtree``: kd-tree tessellation with midpoint splits.
* ``kmeans``: tessellation based on the k-means clustering algorithm.
* ``gwr`` (or ``gas``): tessellation based on the Growing-When-Required self-organizing gas.

``kmeans`` and ``gwr`` methods run optimization on the data and consequently are vulnerable to numerical scaling. 
It is recommended to scale your data adding the option ``-w``.

A key parameter is ``--knn`` (or shorter ``-n``). 
It combines with any of the above methods and allows to impose a lower bound on the number of points (or nearest neighbors) associated with each cell of the mesh, independently of the way the mesh has been grown::

	> tramway tessellate gwr -i example.rwa -w -n 50 -l gwr*

Note that in the above example the *example.rwa* file already exists and we add the analysis.

The ``*`` symbol will be replaced by the lowest available natural integer starting from 0.
This prevents from overwriting an analysis with the same label, if any.

Other key parameters are ``--distance`` (shorter ``-d``) and ``--min-location-count`` (shorter ``-s``).

The former drives how the cells scale, especially in dense areas. Per default it is set to the average translocation distance.
A lower value will yield smaller cells.

The latter parameter allows to discard the cells that would contain less locations than thereby specified.
This filter applies before ``knn``::

	> tramway tessellate gwr -i example.rwa -d 0.1 -s 10 -n 30 -l gwr*

You can check the content of the *example.rwa* file::

	> tramway dump -i example.rwa

	in example.rwa:
		<class 'pandas.core.frame.DataFrame'>
			'kmeans' <class 'tramway.tessellation.base.CellStats'>
			'gwr0' <class 'tramway.tessellation.base.CellStats'>
			'gwr1' <class 'tramway.tessellation.base.CellStats'>

.. seealso::

	:ref:`tessellation`


Visualizing the partition
-------------------------

To visualize spatial 2D tessellations::

	> tramway draw cells -i example.rwa -L kmeans

To print the figure in an image file::

	> tramway draw cells -i example.rwa -L gwr0 -p png

This will generate an *example.png* file.

To overlay the Delaunay graph instead of the Voronoi graph::

	> tramway draw cells -i example.rwa -L gwr1 -D


.. _commandline_inference:

Inferring diffusivity and other parameters
------------------------------------------

Inferring diffusivity and force with the *DF* mode::

	> tramway infer df -i example.rwa -L kmeans -l df-map*

Other inference modes are *D* (``d``), *DD* (``dd``) and *DV* (``dv``).

*DV* is notably more time-consuming than the other inference modes and generates diffusivity and potential energy maps::

	> tramway infer dv -i example.rwa -L gwr1 -l dv-map*


.. seealso::

	:ref:`inference`


Visualizing maps
----------------

2D maps can be plotted with::

	> tramway draw map -i example.rwa -L gwr1,dv-map0

One can overlay the locations as white dots with high transparency over maps colored with one of the *matplotlib* supported colormaps (see also https://matplotlib.org/users/colormaps.html)::

	> tramway draw map -i example.rwa -L kmeans,df-map0 -cm jet -P size=1,color='w',alpha=.05


Extracting features
-------------------

The only feature available for now is curl for 2D force maps::

	> tramway extract curl -i example.rwa -L kmeans,df-map0 --radius 2 -l curl_2

For each cell, if a contour of successively adjacent cells can be found the curl is calculated along this contour and a map of local curl values can thus be extracted.

The optional ``radius`` argument drives the radius of the contour in number of cells.
At radius ``1`` the contour is formed by cells that are immediately adjacent to the center cell.
At radius ``2`` the contour is formed by cells that are adjacent to the radius-1 cells.
And so on.

Note that at higher radii the contours may partly consist of segments of lower-radii contours.

The extracted map can be plotted just like any map::

	> tramway draw map -i example.rwa -L kmeans,df-map0,curl_2


Final analysis tree
-------------------

To sum up this primer, the content of the *example.rwa* file that results from all the above steps is dumped below::

	> tramway dump -i example.rwa

	in example.rwa:
		<class 'pandas.core.frame.DataFrame'>
			'kmeans' <class 'tramway.tessellation.base.CellStats'>
				'df-map0' <class 'tramway.inference.base.Maps'>
					'curl_2' <class 'tramway.inference.base.Maps'>
			'gwr0' <class 'tramway.tessellation.base.CellStats'>
			'gwr1' <class 'tramway.tessellation.base.CellStats'>
				'dv-map0' <class 'tramway.inference.base.Maps'>



.. |txt| replace:: *.txt*
.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*
.. |rwa| replace:: *.rwa*

