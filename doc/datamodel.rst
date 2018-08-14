.. _datamodel:

Data model
==========

Analyses
--------

The |tramway| pipeline generates several artefacts along the way from single molecule locations to quantifications of the dynamics of these molecules. 
All analysis artefacts that derive from the same input data are stored in a single file.

The term *analysis* here refers to a process that takes a single data input and generates a single output.

Artefacts are thus organized in a tree structure such that artefacts are nodes and analyses are edges.

A typical *.rwa* file or :class:`~tramway.core.analyses.lazy.Analyses` object will contain an array of molecule locations or trajectories as topmost data element.
The first levels of analyses will typically consist of spatial tessellations (or data partitions) with resulting :class:`~tramway.tessellation.base.CellStats` partition objects (one per analysis).
The next levels of analyses will usually consist of :class:`~tramway.inference.base.Maps` maps (again, one per analysis) that result from inference or feature extraction.

Each analysis can be identified with a label and documented with comments.


Location and trajectory files
-----------------------------

|tramway| accepts text files as input data files. 
Usual extensions are |txt|, |xyt| or |trxyt| but they are not taken into account. 
These text files contain numerical values organized in tab-delimited columns such that each row specifies the euclidean coordinates of the position of a molecule along with time and optionally trajectory number.

Trajectory number, if available, is the first column. 
Time (in |seconds|) is always the last columns. 
All the intermediate columns - usually two (`x`, `y`) or three (`x`, `y`, `z`) - are spatial coordinates, typically in |um|.
These coordinates should be locations, NOT translocations, even if they represent trajectories.


Analyses *.rwa* files
---------------------

In Python, an |rwa| file can be loaded as follows:

.. code-block:: python

	from tramway.core import *

	analyses = load_rwa(path_to_rwa_file)


The :class:`~tramway.core.analyses.lazy.Analyses` object features a dict-like interface.

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

A path of analyses can be extracted from such a tree:

.. code-block:: python

	analyses = extract_analysis(analyses, ('kmeans', 'df-map0', 'curl_2'))

To extract analysis artefacts of particular types from an analysis tree with a single pathway:

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


