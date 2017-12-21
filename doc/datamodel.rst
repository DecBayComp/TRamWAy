.. _datamodel:

Data model
==========

Analyses
--------

The |tramway| pipeline generates several artefacts along the way from single molecule locations to quantifications of the dynamics of these molecules. 
All analysis artefacts that derive from the same input data are stored in a single file.

The term *analysis* here refers to a process that takes a single data input and generates a single output.

Artefacts are thus organized in a tree structure such that artefacts are nodes and analyses are edges.

A typical *.rwa* file or :class:`~tramway.core.analyses.Analyses` object will contain an array of molecule locations or trajectories as topmost data element.
A first level of analyses will consist of spatial tesselations (or data partitions) with resulting :class:`~tramway.tesselation.base.CellStats` partition objects (one per analysis).
A second level of analyses will consist of inferences with resulting :class:`~tramway.inference.base.Maps` map objects (again, one per analysis).

Each analysis can be addressed with a label and augmented with comments.

 
Location and trajectory files
-----------------------------

|tramway| accepts text files as input data files with |txt|, |xyt|, |trxyt| or any other extension. These text files contain numerical values organized in tab-delimited columns such that each row specifies the euclidean coordinates of the position of a molecule along with time and optionally trajectory number.

Trajectory number, if available, is the first column. Time in |seconds| is always the last columns. All the intermediate columns - usually two (`x`, `y`) or three (`x`, `y`, `z`) - are the coordinates in |um| of the position.

A single dataset can be split in several files.


Analyses *.rwa* files
---------------------

In Python, an |rwa| file can be loaded as follows::

	from tramway.io import HDF5Store

	hdf = HDF5Store(path_to_rwa_file)
	analyses = hdf.peek('analyses')
	hdf.close()

or in a slightly shorter way::

	from tramway.helper.analysis import *

	analyses = find_analysis(path_to_rwa_file)

and if one needs a particular analysis chain, providing analysis labels::

	analyses = find_analysis(path_to_rwa_file, labels=('my-mesh', 'my-df-maps'))

A convenient way to browse the labels, comments and artefact types in a file is::

	> tramway dump -i path_to_rwa_file

or in Python::

	print(format_analyses(analyses))



.. |txt| replace:: *.txt*
.. |xyt| replace:: *.xyt*
.. |trxyt| replace:: *.trxyt*
.. |rwa| replace:: *.rwa*
.. |seconds| replace:: **seconds**
.. |um| replace:: **µm**
.. |tramway| replace:: **TRamWAy**

