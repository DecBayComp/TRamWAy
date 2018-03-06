.. _datamodel:

Data model
==========

Analyses
--------

The |tramway| pipeline generates several artefacts along the way from single molecule locations to quantifications of the dynamics of these molecules. 
All analysis artefacts that derive from the same input data are stored in a single file.

The term *analysis* here refers to a process that takes a single data input and generates a single output.

Artefacts are thus organized in a tree structure such that artefacts are nodes and analyses are edges.

A typical *.rwa* file or :class:`~tramway.core.analyses.base.Analyses` object will contain an array of molecule locations or trajectories as topmost data element.
A first level of analyses will consist of spatial tessellations (or data partitions) with resulting :class:`~tramway.tessellation.base.CellStats` partition objects (one per analysis).
A second level of analyses will consist of inferences with resulting :class:`~tramway.inference.base.Maps` map objects (again, one per analysis).

Each analysis can be addressed with a label and augmented with comments.

 
Location and trajectory files
-----------------------------

|tramway| accepts text files as input data files with |txt|, |xyt|, |trxyt| or any other extension. These text files contain numerical values organized in tab-delimited columns such that each row specifies the euclidean coordinates of the position of a molecule along with time and optionally trajectory number.

Trajectory number, if available, is the first column. Time in |seconds| is always the last columns. All the intermediate columns - usually two (`x`, `y`) or three (`x`, `y`, `z`) - are the coordinates in |um| of the position.

A single dataset can be split in several files.


Analyses *.rwa* files
---------------------

In Python, an |rwa| file can be loaded as follows:

.. code-block:: python

	from tramway.core import *

	analyses = load_rwa(path_to_rwa_file)


The :class:`~tramway.core.analyses.base.Analyses` object features a dict-like interface.

In the REPL, the *analyses* object can be quickly inspected as follows:

.. code-block:: python

	>>> print(analyses)
	<class 'pandas.core.frame.DataFrame'>
		'kmeans' <class 'tramway.tessellation.base.CellStats'>
			'df-map0' <class 'tramway.inference.base.Maps'>
				'curl_2' <class 'tramway.inference.base.Maps'>
		'gwr0' <class 'tramway.tessellation.base.CellStats'>
		'gwr1' <class 'tramway.tessellation.base.CellStats'>
			'dv-map0' <class 'tramway.inference.base.Maps'>
	>>> analyses['kmeans']['df-map0']
	<tramway.core.analyses.lazy.Analyses object at 0x7fc41e5b5f08>
	>>> analyses['kmeans']['df-map0'].data
	<tramway.inference.base.Maps object at 0x7fc468359e10>


The above example shows that every analysis artefact is encapsulated in an :class:`~tramway.core.analyses.lazy.Analyses` object and can be accessed with the `data` (or `artefact`) attribute.

To extract analysis artefacts of a particular type from an analysis tree with a single pathway:

.. code-block:: python

	>>> print(analyses)
	<class 'pandas.core.frame.DataFrame'>
		'kmeans' <class 'tramway.tessellation.base.CellStats'>
			'df-map0' <class 'tramway.inference.base.Maps'>
				'curl_2' <class 'tramway.inference.base.Maps'>

	>>> from tramway.tessellation import CellStats
	>>> from tramway.inference import Maps

	>>> cells, maps = find_artefacts(analyses, (CellStats, Maps))

Here `maps` will correspond to the *curl_2* label.
To select *df-map0* instead:

.. code-block:: python

	>>> cells, maps = find_artefacts(analyses, (CellStats, Maps), quantifiers=('last', 'first'))


Quantifier '*last*' is the default one.

See also :func:`~tramway.core.analyses.lazy.find_artefacts` for more options.


.. |txt| replace:: *.txt*
.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*
.. |rwa| replace:: *.rwa*
.. |seconds| replace:: **seconds**
.. |um| replace:: **µm**
.. |tramway| replace:: **TRamWAy**

