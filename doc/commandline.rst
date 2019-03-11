.. _commandline:

Command-line starter
====================

Input data
----------

The input data are tracking data of single molecules, stored in a text file.

The extension of input data files will usually be |txt|, |xyt| or |trxyt| but this is not required.

The trajectories from an example |trxyt| file can be visualized with the following command::

    > tramway -v animate trajectories -i example.trxyt

The *tramway* command is followed by the *animate* subcommand, itself followed by the *trajectories* subsubcommand.

``-i`` stands for *input file* and can be omitted as long as the file path comes last in the command.

``-v`` stands for *verbose* and is a flag that does not take any argument after.

Equivalent commands are::

    > tramway -i example.trxyt animate trajectories -v
    > tramway animate trajectories -v example.trxyt

Note that it is recommended to specify the time step with the ``--time-step`` (or ``-s``) option.
In the absence of this option as in the above example, *tramway animate trajectories* will try to determine the inter-frame duration, but some floating-point precision error may occur.

The documentation for the *animate* and *animate trajectories* subcommands can be printed using the ``-h`` flag::

    > tramway animate -h
    > tramway animate trajectories -h

We will first import the example |trxyt| file and bin the molecule locations in a step referred to as *tessellation*, before we perform the inference.

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

Some of the available spatial methods are:

* ``random``: Voronoi tessellation based on random cell centers.
* ``grid``: regular grid with equally sized square or cubic areas.
* ``kdtree``: kd-tree tessellation with midpoint splits.
* ``hexagon``: hexagonal grid with equally sized hexagons (2D only).
* ``kmeans``: tessellation based on the k-means clustering algorithm.
* ``gwr`` (or ``gas``): tessellation based on the Growing-When-Required self-organizing gas.

``kmeans`` and ``gwr`` methods run optimization on the data and consequently are vulnerable to numerical scaling. 
It is recommended to scale your data adding the option ``-w``.

Each method exposes specific parameters.
Some parameters however apply after a mesh has been grown and therefore apply to all the methods.
Some of these parameters are ``--knn`` (or ``-n`` or ``--min-nn``), ``--radius`` (or shorter ``-r``) and ``--min-n``.

``--knn`` allows to impose a lower bound on the number of points (nearest neighbours) associated with each cell of the mesh, independently of the way the mesh has been grown.

Note that ``-n`` controls the minimum number of nearest neighbours and expands the cells if necessary.
To make all the cells contain a uniform number of points, the ``-N`` argument controls the maximum number
of nearest neighbours and can be set to the same value as ``-n``::

	> tramway tessellate gwr -i example.rwa -w -n 50 -N 50 -l gwr*

``--radius`` forces all the cells to be hyperspheres of uniform radius.
Values are specified in the units of the single molecule data, typically |um|.

With the above two arguments, some cells may overlap.

``--min-n`` discards the cells that contain less locations than thereby specified.
Note that this filter applies before ``--knn``.
Cells with too few locations will be discarded anyway.

Other parameters directly affect the tessellation methods but can be found in all these methods.
The main such parameter is ``--distance`` (shorter ``-d``).
It drives how the cells scale and offers some degree of control over the distance between neighbour
cell centers, especially in dense areas.
Per default it is set to the average translocation distance.
A lower value may yield smaller cells.

The following example combines specified inter-cell distance (``-d``), sparse cell removal (``--min-n``) and cell expansion to a minimum location number (``--min-nn``)::

	> tramway tessellate gwr -i example.rwa -d 0.1 --min-n 10 --min-nn 30 -l gwr*

Note that, in the above two examples, the *example.rwa* file already exists and we add the meshes to the existing analysis tree.

The ``*`` symbol is replaced by the lowest available natural integer starting from 0.
This prevents from overwriting an analysis with the same label, if any.


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

The parameters used to build a tessellation can be listed with the *dump* subcommand::

    > tramway dump -i example.rwa -L gwr0


.. _commandline_inference:

Inferring diffusivity and other parameters
------------------------------------------

Inferring diffusivity and force with the *DF* mode::

	> tramway infer standard.df -i example.rwa -L kmeans -l df-map*

Other inference modes are *D* (``standard.d``), *DD* (``standard.dd``) and *DV* (``dv``).

*D*, *DD* and *DF* have *degraded* variants, respectively: ``degraded.d``, ``degraded.dd`` and ``degraded.df``.

A common parameter is the localization error, which default value is :math:`\sigma = 0.03 \textrm{µm}`.
See the :ref:`Common parameters section <inference_parameters>` to learn more about it.

*DV* is notably more time-consuming than the other inference modes and generates diffusivity and potential energy maps::

	> tramway infer dv -i example.rwa -L gwr1 -l dv-map*


.. seealso::

	:ref:`inference`


Visualizing maps
----------------

2D maps can be plotted with::

	> tramway draw map -i example.rwa -L gwr1,dv-map0 --feature force

If the mapped feature to be drawn is not specified, *tramway draw map* will make a figure for each of the mapped features.

One can overlay the locations as white dots with high transparency over maps colored with one of the *matplotlib* supported colormaps (see also https://matplotlib.org/users/colormaps.html)::

	> tramway draw map -i example.rwa -L kmeans,df-map0 -cm jet -P size=1,color='w',alpha=.05

The parameters used to infer a set of maps can be listed with the *dump* subcommand::

    > tramway dump -i example.rwa -L kmeans,df-map0


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


Inspecting an *rwa* file
------------------------

The content of the *example.rwa* file that results from all the above steps can be superficially inspected as below::

	> tramway dump -i example.rwa

	in example.rwa:
		<class 'pandas.core.frame.DataFrame'>
			'kmeans' <class 'tramway.tessellation.base.CellStats'>
				'df-map0' <class 'tramway.inference.base.Maps'>
					'curl_2' <class 'tramway.inference.base.Maps'>
			'gwr0' <class 'tramway.tessellation.base.CellStats'>
			'gwr1' <class 'tramway.tessellation.base.CellStats'>
				'dv-map0' <class 'tramway.inference.base.Maps'>

As mentioned before, some analysis artefacts can be inspected specifying the corresponding label.

The *dump* subcommand can also export some analysis artefacts for use in **InferenceMAP** using the ``--cluster`` (for spatial meshes) and ``--vmesh`` (for maps) options.
Learn more from the *tramway dump* help::

    > tramway dump -h


.. _commandline_time:

Segmenting time
---------------

The *tramway tessellate* command features temporal windowing as an addition to spatial binning.
Let us consider the following example::

    > tramway -i example.trxyt -o example2.rwa tessellate gwr --knn 10 --time-window-duration 2 --time-window-shift 0.2

Note first that we are making a new *rwa* file with the ``-o`` flag.
We could have kept on working on the existing *rwa* file with ``-i example.rwa`` instead of ``-i example.trxyt -o example2.rwa``.

Note second that we do not specify any label for the resulting sampling of the locations.
Of course we could have done so.

In the example above, we bin the locations using the *gwr* spatial tessellation method.
At the spatial binning step, all the locations considered independently of their onset time.

Temporal windowing comes next and requires the ``--time-window-duration`` argument followed by the duration of the window in seconds.

Optionally, the time shift between successive segments can be specified with the ``--time-window-shift`` argument.
In the above example every pair of successive segments will share a 90% overlap (1800 ms).
The default is a shift equal to the duration, so that there is no overlap.

At the inference step, the temporal sampling is transparent::

    > tramway -i example2.rwa infer ddrift

Note that drawing the spatial mesh or the inferred map now requires the index of a time segment to be specified::

    > tramway -i example2.rwa draw cells --segment 0
    > tramway -i example2.rwa draw map --feature drift --segment 0

A movie can also be generated out of the inferred maps::

    > tramway -v -i example2.rwa animate map --feature drift

Note that *tramway animate map* requires a mapped feature to be specified unless a single feature is found.

This actually generates a temporary *mp4* file.
To keep the generated file, an output file name has to be specified with the ``-o`` option.

*tramway animate map* can also subsample in time with the ``--time-step`` (or ``-s``) option.
Overlapping segments will be averaged wrt the distance from the segment centers.

